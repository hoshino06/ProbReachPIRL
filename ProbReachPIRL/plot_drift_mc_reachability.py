# -*- coding: utf-8 -*-
"""
Monte Carlo reachability maps for the drifting-control task.

This script uses the same 2-D slices as plot_drift_value_contours.py and
compares the learned value V(x) with the empirical probability of reaching the
target under the learned actor.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from plot_drift_value_contours import (
    set_paper_style,
    import_from_string,
    resolve_mu,
    evaluate_value_grid,
    evaluate_vector_field,
    add_target_patch,
)


def rollout_reach_probability(
    agent,
    env,
    states_phys: np.ndarray,
    num_grid: int,
    num_rollouts: int,
    action_batch_size: int,
    deterministic: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate reach probability from each physical initial state."""
    rng = np.random.default_rng(seed)
    n_states = states_phys.shape[0]
    total = n_states * num_rollouts

    states = np.repeat(states_phys.astype(np.float32), num_rollouts, axis=0)
    active = np.ones(total, dtype=bool)
    reached = np.zeros(total, dtype=bool)
    unsafe = np.zeros(total, dtype=bool)
    timed_out = np.zeros(total, dtype=bool)

    max_steps = int(np.ceil(float(np.max(states[:, 0])) / env.dt)) + 2

    for _ in range(max_steps):
        if not np.any(active):
            break

        idx = np.flatnonzero(active)
        xa = states[idx]

        is_target = env.is_target(xa)
        is_unsafe = env.is_unsafe(xa)
        is_time_over = xa[:, 0] < env.dt
        done = is_target | is_unsafe | is_time_over

        if np.any(done):
            done_idx = idx[done]
            reached[done_idx] = is_target[done]
            unsafe[done_idx] = is_unsafe[done]
            timed_out[done_idx] = is_time_over[done] & ~is_target[done] & ~is_unsafe[done]
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

    # Treat any remaining active samples as time-outs at the horizon.
    timed_out[active] = True

    prob = reached.reshape(n_states, num_rollouts).mean(axis=1).reshape(num_grid, num_grid)
    unsafe_prob = unsafe.reshape(n_states, num_rollouts).mean(axis=1).reshape(num_grid, num_grid)
    timeout_prob = timed_out.reshape(n_states, num_rollouts).mean(axis=1).reshape(num_grid, num_grid)
    return prob, unsafe_prob, timeout_prob


def plot_scalar_panel(
    ax,
    env,
    data: np.ndarray,
    meta: dict,
    plane: str,
    title: str,
    levels: int,
    vmin: float,
    vmax: float,
    cmap: str,
    vector_field=None,
    vector_stride: int = 8,
    vector_scale=None,
):
    x = meta["x"]
    y = meta["y"]
    cf = ax.contourf(
        x,
        y,
        data.T,
        levels=np.linspace(vmin, vmax, levels),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extend="both",
    )
    cs = ax.contour(x, y, data.T, levels=np.linspace(vmin, vmax, 7), linewidths=0.55, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

    if vector_field is not None:
        x_dot, y_dot = vector_field
        stride = max(1, int(vector_stride))
        X, Y = np.meshgrid(x, y, indexing="xy")
        ax.quiver(
            X[::stride, ::stride],
            Y[::stride, ::stride],
            x_dot.T[::stride, ::stride],
            y_dot.T[::stride, ::stride],
            color="white",
            edgecolor="black",
            linewidth=0.2,
            alpha=0.75,
            pivot="mid",
            angles="xy",
            scale_units="xy",
            scale=vector_scale,
            width=0.0028,
        )

    add_target_patch(ax, env, plane)
    ax.set_xlabel(meta["xlabel"].replace("beta", r"$\beta$").replace("epsi", r"$e_\psi$").replace("ey", r"$e_y$"))
    ax.set_ylabel(meta["ylabel"].replace("r", r"$r$").replace("epsi", r"$e_\psi$").replace("ey", r"$e_y$"))
    ax.set_title(title)
    return cf


def save_plane_npz(path, value, reach_prob, unsafe_prob, timeout_prob, meta, vector_field=None):
    data = {
        "V": value,
        "reach_prob": reach_prob,
        "unsafe_prob": unsafe_prob,
        "timeout_prob": timeout_prob,
        "value_error": reach_prob - value,
        "abs_value_error": np.abs(reach_prob - value),
        "x": meta["x"],
        "y": meta["y"],
        "T": meta["T"],
        "plane": meta["plane"],
        "xlabel": meta["xlabel"],
        "ylabel": meta["ylabel"],
    }
    if vector_field is not None:
        data["x_dot"] = vector_field[0]
        data["y_dot"] = vector_field[1]
    np.savez(path, **data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs/drift/td3/0603_1707_seed_2/ckpt-1000000")
    parser.add_argument("--out_dir", default="plot/drift_mc_reachability")
    parser.add_argument("--env_cls", default="examples.env_drifting_control.Env")
    parser.add_argument("--agent_cls", default="agent.TD3_PIRL_ray.PIRLAgent")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--mu", default="target")
    parser.add_argument("--num_grid", type=int, default=41)
    parser.add_argument("--num_rollouts", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--action_batch_size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ey_max_override", type=float, default=None,
                        help="Override env.ey_max for MC unsafe checks and plot ranges only.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", type=int, default=41)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_vector_field", action="store_true")
    parser.add_argument("--vector_stride", type=int, default=6)
    parser.add_argument("--vector_scale", type=float, default=None)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_paper_style()

    Env = import_from_string(args.env_cls)
    PIRLAgent = import_from_string(args.agent_cls)
    env = Env()
    if args.ey_max_override is not None:
        env.ey_max = float(args.ey_max_override)
    mu = resolve_mu(env, args.mu)
    agent = PIRLAgent.from_checkpoint(args.checkpoint, device=args.device, learner=False)

    results = {}
    for i, plane in enumerate(["beta_r", "ey_epsi"]):
        V, meta, states = evaluate_value_grid(
            agent, env, plane=plane, T=args.T, mu=mu,
            num_grid=args.num_grid, batch_size=args.batch_size,
        )
        reach_prob, unsafe_prob, timeout_prob = rollout_reach_probability(
            agent,
            env,
            states,
            num_grid=args.num_grid,
            num_rollouts=args.num_rollouts,
            action_batch_size=args.action_batch_size,
            deterministic=args.deterministic,
            seed=args.seed + 1000 * i,
        )

        vector = None
        if not args.no_vector_field:
            xdot, ydot, _ = evaluate_vector_field(
                agent, env, plane=plane, states_phys=states,
                num_grid=args.num_grid, batch_size=args.batch_size,
            )
            vector = (xdot, ydot)

        results[plane] = {
            "V": V,
            "meta": meta,
            "reach_prob": reach_prob,
            "unsafe_prob": unsafe_prob,
            "timeout_prob": timeout_prob,
            "vector": vector,
        }
        save_plane_npz(
            os.path.join(args.out_dir, f"mc_reachability_{plane}.npz"),
            V,
            reach_prob,
            unsafe_prob,
            timeout_prob,
            meta,
            vector_field=vector,
        )

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.1), constrained_layout=True)
    for row, plane in enumerate(["beta_r", "ey_epsi"]):
        res = results[plane]
        V = np.clip(res["V"], 0.0, 1.0)
        P = res["reach_prob"]
        E = P - V
        meta = res["meta"]
        vector = res["vector"]

        cf0 = plot_scalar_panel(
            axes[row, 0], env, V, meta, plane,
            title=f"{plane}: learned value",
            levels=args.levels, vmin=0.0, vmax=1.0, cmap="viridis",
            vector_field=vector, vector_stride=args.vector_stride,
            vector_scale=args.vector_scale,
        )
        cf1 = plot_scalar_panel(
            axes[row, 1], env, P, meta, plane,
            title=f"{plane}: Monte Carlo reach prob.",
            levels=args.levels, vmin=0.0, vmax=1.0, cmap="viridis",
            vector_field=vector, vector_stride=args.vector_stride,
            vector_scale=args.vector_scale,
        )
        cf2 = plot_scalar_panel(
            axes[row, 2], env, E, meta, plane,
            title=f"{plane}: MC - value",
            levels=args.levels, vmin=-1.0, vmax=1.0, cmap="coolwarm",
            vector_field=None,
        )
        fig.colorbar(cf0, ax=axes[row, 0], shrink=0.88)
        fig.colorbar(cf1, ax=axes[row, 1], shrink=0.88)
        fig.colorbar(cf2, ax=axes[row, 2], shrink=0.88)

    fig.suptitle(
        f"Closed-loop reachability check, T={args.T:g}s, mu={mu:.2f}, "
        f"rollouts/grid={args.num_rollouts}, ey_max={env.ey_max:g}"
    )
    for ext in ["png", "pdf"]:
        path = os.path.join(args.out_dir, f"mc_vs_value_contours.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    print("--------------------------------------------")
    print(f"checkpoint: {args.checkpoint}")
    print(f"out_dir:    {args.out_dir}")
    print(f"T:          {args.T:g}")
    print(f"mu:         {mu:.4f}")
    print(f"grid:       {args.num_grid} x {args.num_grid}")
    print(f"rollouts:   {args.num_rollouts}")
    print(f"mode:       {'deterministic' if args.deterministic else 'stochastic'}")
    print(f"ey_max:     {env.ey_max:g}")
    for plane, res in results.items():
        err = res["reach_prob"] - np.clip(res["V"], 0.0, 1.0)
        print(
            f"{plane}: mean|MC-V|={np.nanmean(np.abs(err)):.4f}, "
            f"max|MC-V|={np.nanmax(np.abs(err)):.4f}, "
            f"mean MC={np.nanmean(res['reach_prob']):.4f}, "
            f"mean V={np.nanmean(np.clip(res['V'], 0.0, 1.0)):.4f}"
        )
    print("--------------------------------------------")


if __name__ == "__main__":
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    main()
