# -*- coding: utf-8 -*-
"""Trajectory examples for checking learned reachability values.

The figure selects three beta-r initial states from the learned value contour
and rolls out the closed-loop stochastic dynamics from each point.  It is meant
to show that high-value states tend to reach the target and low-value states
tend to fail under the same learned policy.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from plot_drift_value_contours import (
    DEFAULT_PIRL_CHECKPOINT,
    add_target_patch,
    backend_supports_show,
    evaluate_value_grid,
    import_from_string,
    resolve_mu,
    set_paper_style,
)


@dataclass
class ExamplePoint:
    name: str
    target_value: float
    state: np.ndarray
    beta: float
    r: float
    value: float


def states_from_beta_r(env, T: float, mu: float, beta: np.ndarray, r: np.ndarray) -> np.ndarray:
    target = env.get_drift_target(mu)
    beta = np.asarray(beta, dtype=np.float32)
    r = np.asarray(r, dtype=np.float32)
    vx = np.full_like(beta, target["vx"], dtype=np.float32)
    states = np.stack(
        [
            np.full_like(beta, T, dtype=np.float32),
            np.full_like(beta, target["ey"], dtype=np.float32),
            np.full_like(beta, target["epsi"], dtype=np.float32),
            vx,
            vx * np.tan(beta),
            r,
            np.full_like(beta, target["delta"], dtype=np.float32),
            np.full_like(beta, mu, dtype=np.float32),
        ],
        axis=-1,
    )
    return states.astype(np.float32)


def select_from_mc_npz(env, T: float, mu: float, selection_npz: str):
    data = np.load(selection_npz)
    V = np.clip(data["V"], 0.0, 1.0)
    P = np.asarray(data["reach_prob"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    B, R = np.meshgrid(x, y, indexing="ij")
    states = states_from_beta_r(env, T, mu, B.reshape(-1), R.reshape(-1))
    safe = ~env.is_unsafe(states)
    target = env.is_target(states)

    # Avoid selecting the exact domain boundary; those examples are visually
    # less useful than interior low-value states.
    interior = (np.abs(B.reshape(-1)) < 0.90 * env.beta_max) & (np.abs(R.reshape(-1)) < 0.92 * env.r_max)
    candidate_mask = safe & ~target & interior

    specs = [
        ("High V", 0.9, 0.95, 1.0),
        ("Moderate V", 0.7, 0.9, 1.0),
        ("Low V", 0.02, 0.0, 1.5),
    ]
    examples = []
    selected = np.zeros(states.shape[0], dtype=bool)
    values = V.reshape(-1)
    probs = P.reshape(-1)
    beta = B.reshape(-1)
    r = R.reshape(-1)
    for name, target_value, target_prob, prob_weight in specs:
        score = np.abs(values - target_value) + prob_weight * np.abs(probs - target_prob)
        score[~candidate_mask] = np.inf
        for prev in np.flatnonzero(selected):
            dist = np.hypot((beta - beta[prev]) / env.beta_max, (r - r[prev]) / env.r_max)
            score[dist < 0.30] = np.inf
        idx = int(np.argmin(score))
        selected[idx] = True
        examples.append(
            ExamplePoint(
                name=name,
                target_value=target_value,
                state=states[idx].copy(),
                beta=float(beta[idx]),
                r=float(r[idx]),
                value=float(values[idx]),
            )
        )
    return examples


def select_beta_r_examples(agent, env, T: float, mu: float, num_grid: int, batch_size: int,
                           selection_npz: str | None):
    V, meta, states = evaluate_value_grid(
        agent,
        env,
        plane="beta_r",
        T=T,
        mu=mu,
        num_grid=num_grid,
        batch_size=batch_size,
    )
    if selection_npz and os.path.exists(selection_npz):
        return select_from_mc_npz(env, T, mu, selection_npz), V, meta

    values = np.clip(V.reshape(-1), 0.0, 1.0)
    beta = np.repeat(meta["x"], num_grid)
    r = np.tile(meta["y"], num_grid)

    safe = ~env.is_unsafe(states)
    target = env.is_target(states)
    candidate_mask = safe & ~target

    examples = []
    selected = np.zeros(len(values), dtype=bool)
    for name, target_value in [("High V", 0.9), ("Mid V", 0.5), ("Low V", 0.1)]:
        score = np.abs(values - target_value)
        score[~candidate_mask] = np.inf

        # Keep selected points visually separated on the beta-r plane.
        for prev in np.flatnonzero(selected):
            dist = np.hypot((beta - beta[prev]) / env.beta_max, (r - r[prev]) / env.r_max)
            score[dist < 0.35] = np.inf

        idx = int(np.argmin(score))
        selected[idx] = True
        examples.append(
            ExamplePoint(
                name=name,
                target_value=target_value,
                state=states[idx].copy(),
                beta=float(beta[idx]),
                r=float(r[idx]),
                value=float(values[idx]),
            )
        )

    return examples, V, meta


def rollout_examples(agent, env, examples, num_rollouts: int, action_batch_size: int,
                     deterministic: bool, seed: int):
    rng = np.random.default_rng(seed)
    results = []
    max_steps = int(np.ceil(max(float(ex.state[0]) for ex in examples) / env.dt)) + 2

    for ex in examples:
        states = np.repeat(ex.state[None, :].astype(np.float32), num_rollouts, axis=0)
        paths = np.full((num_rollouts, max_steps + 1, env.state_dim), np.nan, dtype=np.float32)
        paths[:, 0, :] = states
        status = np.full(num_rollouts, "timeout", dtype=object)
        active = np.ones(num_rollouts, dtype=bool)

        for step in range(max_steps):
            idx = np.flatnonzero(active)
            if len(idx) == 0:
                break

            xa = states[idx]
            is_target = env.is_target(xa)
            is_unsafe = env.is_unsafe(xa)
            is_time_over = xa[:, 0] < env.dt
            done = is_target | is_unsafe | is_time_over
            if np.any(done):
                done_idx = idx[done]
                status[done_idx[is_target[done]]] = "reached"
                status[done_idx[is_unsafe[done]]] = "unsafe"
                active[done_idx] = False

            idx = np.flatnonzero(active)
            if len(idx) == 0:
                break

            actions_scaled = []
            scaled = env.scale_state(states[idx])
            for start in range(0, len(idx), action_batch_size):
                actions_scaled.append(agent.get_action(scaled[start:start + action_batch_size]))
            actions_scaled = np.concatenate(actions_scaled, axis=0).astype(np.float32)
            actions_phys = env._clip_action(env.unscale_action(actions_scaled))

            drift, sigma = env.drift_and_diffusion(states[idx], actions_phys)
            if deterministic:
                noise_term = 0.0
            else:
                noise = rng.standard_normal(size=states[idx].shape).astype(np.float32)
                noise_term = np.sqrt(env.dt) * sigma * noise

            next_states = states[idx] + env.dt * drift + noise_term
            next_states[:, 0] = np.maximum(next_states[:, 0], 0.0)
            next_states[:, 7] = np.clip(next_states[:, 7], env.mu_min, env.mu_max)
            states[idx] = next_states.astype(np.float32)
            paths[idx, step + 1, :] = states[idx]

        for i in np.flatnonzero(active):
            if env.is_target(states[i]):
                status[i] = "reached"
            elif env.is_unsafe(states[i]):
                status[i] = "unsafe"
            else:
                status[i] = "timeout"

        results.append({"example": ex, "paths": paths, "status": status})

    return results


def add_value_background(ax, V: np.ndarray, meta: dict, alpha: float = 0.26):
    x = meta["x"]
    y = meta["y"]
    return ax.imshow(
        np.clip(V, 0.0, 1.0).T,
        origin="lower",
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="Greys",
        alpha=alpha,
        interpolation="bilinear",
    )


def path_xy_from_relative_states(env, path: np.ndarray):
    valid = ~np.isnan(path[:, 0])
    path = path[valid]
    if len(path) == 0:
        return np.array([]), np.array([])

    ey = path[:, 1]
    epsi = path[:, 2]
    vx = path[:, 3]
    vy = path[:, 4]
    denom = np.maximum(1.0 - env.kappa_ref * ey, 0.2)
    s_dot = (vx * np.cos(epsi) - vy * np.sin(epsi)) / denom

    s = np.zeros(len(path), dtype=np.float64)
    if len(path) > 1:
        s[1:] = np.cumsum(env.dt * s_dot[:-1])

    theta = env.kappa_ref * s
    if abs(env.kappa_ref) < 1.0e-9:
        x_ref = s
        y_ref = np.zeros_like(s)
    else:
        x_ref = np.sin(theta) / env.kappa_ref
        y_ref = (1.0 - np.cos(theta)) / env.kappa_ref

    normal_x = -np.sin(theta)
    normal_y = np.cos(theta)
    return x_ref + ey * normal_x, y_ref + ey * normal_y


def plot_trajectory_cloud(
    ax,
    env,
    results,
    plane: str,
    colors,
    value_grid=None,
    value_meta=None,
    show_title: bool = False,
):
    if value_grid is not None and value_meta is not None:
        add_value_background(ax, value_grid, value_meta)
    add_target_patch(ax, env, plane, label="target set")
    for result, color in zip(results, colors):
        ex = result["example"]
        paths = result["paths"]
        status = result["status"]
        for path, st in zip(paths, status):
            valid = ~np.isnan(path[:, 0])
            path = path[valid]
            if plane == "beta_r":
                x = np.arctan2(path[:, 4], np.maximum(path[:, 3], 1.0e-6))
                y = path[:, 5]
            elif plane == "ey_epsi":
                x = path[:, 1]
                y = path[:, 2]
            else:
                raise ValueError(plane)
            alpha = 0.42 if st == "reached" else 0.22
            linestyle = "-" if st == "reached" else "--"
            ax.plot(x, y, color=color, alpha=alpha, linewidth=1.0, linestyle=linestyle)

        if plane == "beta_r":
            ax.scatter(ex.beta, ex.r, s=70, color=color, edgecolor="black", zorder=5, label=ex.name)
        else:
            ax.scatter(ex.state[1], ex.state[2], s=70, color=color, edgecolor="black", zorder=5, label=ex.name)

    if plane == "beta_r":
        ax.set_xlim(-env.beta_max, env.beta_max)
        ax.set_ylim(-env.r_max, env.r_max)
        ax.set_xlabel(r"$\beta$ [rad]")
        ax.set_ylabel(r"$r$ [rad/s]")
        if show_title:
            ax.set_title(r"Rollouts projected to $\beta$-$r$")
    else:
        ax.set_xlim(-env.ey_max, env.ey_max)
        ax.set_ylim(env.reset_epsi_min, env.reset_epsi_max)
        ax.set_xlabel(r"$e_y$ [m]")
        ax.set_ylabel(r"$e_\psi$ [rad]")
        if show_title:
            ax.set_title(r"Rollouts projected to $e_y$-$e_\psi$")
    ax.legend(frameon=True, loc="best", fontsize=9)


def plot_xy_trajectories(ax, env, results, colors, equal_aspect: bool = False, show_title: bool = False):
    all_x = []
    all_y = []
    for result, color in zip(results, colors):
        ex = result["example"]
        paths = result["paths"]
        status = result["status"]
        for path, st in zip(paths, status):
            x, y = path_xy_from_relative_states(env, path)
            if len(x) == 0:
                continue
            all_x.append(x)
            all_y.append(y)
            alpha = 0.42 if st == "reached" else 0.22
            linestyle = "-" if st == "reached" else "--"
            ax.plot(x, y, color=color, alpha=alpha, linewidth=1.0, linestyle=linestyle)

        x0, y0 = path_xy_from_relative_states(env, ex.state[None, :])
        if len(x0):
            ax.scatter(x0[0], y0[0], s=70, color=color, edgecolor="black", zorder=5, label=ex.name)

    if all_x:
        x_min = min(float(np.nanmin(x)) for x in all_x)
        x_max = max(float(np.nanmax(x)) for x in all_x)
        y_min = min(float(np.nanmin(y)) for y in all_y)
        y_max = max(float(np.nanmax(y)) for y in all_y)
        x_plot_max = max(20.0, x_max + 0.5)
        y_ref_max = min(max(y_max + 0.5, 1.0), 1.95 / max(abs(env.kappa_ref), 1.0e-9))
        if abs(env.kappa_ref) < 1.0e-9:
            s_max = max(x_plot_max + 1.0, 1.0)
        else:
            theta_y = np.arccos(np.clip(1.0 - env.kappa_ref * y_ref_max, -1.0, 1.0))
            theta_x = np.arcsin(np.clip(env.kappa_ref * x_plot_max, -1.0, 1.0))
            theta_max = max(theta_y, theta_x)
            s_max = theta_max / env.kappa_ref
        s_grid = np.linspace(0.0, max(float(s_max), 1.0), 500)
        theta = env.kappa_ref * s_grid
        if abs(env.kappa_ref) < 1.0e-9:
            x_ref = s_grid
            y_ref = np.zeros_like(s_grid)
        else:
            x_ref = np.sin(theta) / env.kappa_ref
            y_ref = (1.0 - np.cos(theta)) / env.kappa_ref
        normal_x = -np.sin(theta)
        normal_y = np.cos(theta)
        ax.plot(x_ref, y_ref, color="black", linewidth=1.1, alpha=0.65, label="reference path")
        for offset in [-env.ey_max, env.ey_max]:
            ax.plot(
                x_ref + offset * normal_x,
                y_ref + offset * normal_y,
                color="black",
                linewidth=0.7,
                alpha=0.35,
                linestyle=":",
            )
        ax.set_xlim(min(-0.5, x_min - 0.5), x_plot_max)
        ax.set_ylim(min(-2.5, y_min - 0.5), max(y_max + 0.5, 1.0))
        ax.set_xticks(np.arange(0.0, x_plot_max + 1.0e-6, 5.0))

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    if show_title:
        ax.set_title(r"Trajectories in physical $x$-$y$ space")
    ax.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=9)


def plot_outcome_bars(ax, results, colors, show_title: bool = False):
    labels = [r["example"].name for r in results]
    reached = np.array([np.mean(r["status"] == "reached") for r in results])
    unsafe = np.array([np.mean(r["status"] == "unsafe") for r in results])
    timeout = 1.0 - reached - unsafe
    x = np.arange(len(results))

    ax.bar(x, reached, color="#2ca25f", label="reached")
    ax.bar(x, unsafe, bottom=reached, color="#de2d26", label="unsafe")
    ax.bar(x, timeout, bottom=reached + unsafe, color="#9e9e9e", label="timeout")
    for i, (result, color) in enumerate(zip(results, colors)):
        ex = result["example"]
        ax.text(i, 1.04, f"V={ex.value:.2f}", ha="center", va="bottom", color=color, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.14)
    ax.set_ylabel("rollout fraction")
    if show_title:
        ax.set_title("Closed-loop outcomes")
    ax.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=9)


def save_figure(fig, out_dir: str, stem: str):
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_PIRL_CHECKPOINT)
    parser.add_argument("--out_dir", default="plot/drift_value_trajectory_examples")
    parser.add_argument("--env_cls", default="examples.env_drifting_control.Env")
    parser.add_argument("--agent_cls", default="agent.TD3_PIRL_ray.PIRLAgent")
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--mu", default="target")
    parser.add_argument("--num_grid", type=int, default=101)
    parser.add_argument("--num_rollouts", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--action_batch_size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--levels", type=int, default=41)
    parser.add_argument(
        "--xy_auto_aspect",
        action="store_true",
        help="Fill the x-y panel without preserving physical scale. Default uses equal physical x-y aspect.",
    )
    parser.add_argument(
        "--selection_npz",
        default=(
            "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_001/"
            "mix85_expand003_R2/eval/mc_reachability_beta_r.npz"
        ),
        help="Optional MC-eval NPZ used only to choose representative High/Mid/Low states.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_paper_style()

    Env = import_from_string(args.env_cls)
    Agent = import_from_string(args.agent_cls)
    env = Env()
    mu = resolve_mu(env, args.mu)
    agent = Agent.from_checkpoint(args.checkpoint, device=args.device, learner=False)

    examples, V, meta = select_beta_r_examples(
        agent,
        env,
        T=args.T,
        mu=mu,
        num_grid=args.num_grid,
        batch_size=args.batch_size,
        selection_npz=args.selection_npz,
    )
    V_ey_epsi, meta_ey_epsi, _ = evaluate_value_grid(
        agent,
        env,
        plane="ey_epsi",
        T=args.T,
        mu=mu,
        num_grid=args.num_grid,
        batch_size=args.batch_size,
    )
    results = rollout_examples(
        agent,
        env,
        examples,
        num_rollouts=args.num_rollouts,
        action_batch_size=args.action_batch_size,
        deterministic=args.deterministic,
        seed=args.seed,
    )

    colors = ["#1b9e77", "#7570b3", "#d95f02"]

    figures = []

    fig_beta_r, ax_beta_r = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    plot_trajectory_cloud(ax_beta_r, env, results, "beta_r", colors, value_grid=V, value_meta=meta)
    save_figure(fig_beta_r, args.out_dir, "panel_rollouts_beta_r")
    figures.append(fig_beta_r)

    fig_outcomes, ax_outcomes = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    plot_outcome_bars(ax_outcomes, results, colors)
    save_figure(fig_outcomes, args.out_dir, "panel_closed_loop_outcomes")
    figures.append(fig_outcomes)

    fig_ey_epsi, ax_ey_epsi = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    plot_trajectory_cloud(
        ax_ey_epsi,
        env,
        results,
        "ey_epsi",
        colors,
        value_grid=V_ey_epsi,
        value_meta=meta_ey_epsi,
    )
    save_figure(fig_ey_epsi, args.out_dir, "panel_rollouts_ey_epsi")
    figures.append(fig_ey_epsi)

    fig_xy, ax_xy = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    plot_xy_trajectories(ax_xy, env, results, colors, equal_aspect=not args.xy_auto_aspect)
    save_figure(fig_xy, args.out_dir, "panel_rollouts_xy")
    figures.append(fig_xy)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
    plot_trajectory_cloud(axes[0, 0], env, results, "beta_r", colors, value_grid=V, value_meta=meta)
    plot_trajectory_cloud(
        axes[0, 1],
        env,
        results,
        "ey_epsi",
        colors,
        value_grid=V_ey_epsi,
        value_meta=meta_ey_epsi,
    )
    plot_outcome_bars(axes[1, 0], results, colors)
    plot_xy_trajectories(axes[1, 1], env, results, colors, equal_aspect=not args.xy_auto_aspect)
    save_figure(fig, args.out_dir, "value_trajectory_examples")
    figures.append(fig)

    np.savez(
        os.path.join(args.out_dir, "value_trajectory_examples_data.npz"),
        example_names=np.array([ex.name for ex in examples]),
        example_values=np.array([ex.value for ex in examples]),
        example_beta=np.array([ex.beta for ex in examples]),
        example_r=np.array([ex.r for ex in examples]),
        statuses=np.array([r["status"] for r in results], dtype=object),
    )

    print("--------------------------------------------")
    print(f"checkpoint: {args.checkpoint}")
    print(f"out_dir:    {args.out_dir}")
    for result in results:
        ex = result["example"]
        status = result["status"]
        print(
            f"{ex.name}: V={ex.value:.3f}, beta={ex.beta:.3f}, r={ex.r:.3f}, "
            f"reached={np.mean(status == 'reached'):.2f}, "
            f"unsafe={np.mean(status == 'unsafe'):.2f}, "
            f"timeout={np.mean(status == 'timeout'):.2f}"
        )
    print("--------------------------------------------")

    if backend_supports_show():
        plt.show()
    else:
        for fig_i in figures:
            plt.close(fig_i)


if __name__ == "__main__":
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    main()
