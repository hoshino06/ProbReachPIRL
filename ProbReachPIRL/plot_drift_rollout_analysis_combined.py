# -*- coding: utf-8 -*-
"""
Closed-loop rollout analysis for the drifting-control reach-avoid task.

Outputs for each seed:
  - time histories
  - beta-r phase plot
  - ey vs path coordinate
  - approximate global path
  - normalized target-error plot
  - trajectory CSV

This script does not modify env_drifting_control.py.
"""

from __future__ import annotations

import os
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
except Exception:
    torch = None

from agent.TD3_PIRL_ray import PIRLAgent
from examples.env_drifting_control import Env


#DEFAULT_CHECKPOINT = "logs/drift/scheduling/xxxx_seed_1/ckpt-200000"
DEFAULT_CHECKPOINT = "logs/drift/0530_0446_seed_1/ckpt-67000"


@dataclass
class RolloutResult:
    states: np.ndarray
    actions_scaled: np.ndarray
    actions_phys: np.ndarray
    rewards: np.ndarray
    done: bool
    terminal_info: Dict[str, bool]
    unsafe_reason: str
    seed: int


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def get_action_scaled(agent: PIRLAgent, state_scaled: np.ndarray) -> np.ndarray:
    """Return deterministic scaled action."""
    state_scaled = np.asarray(state_scaled, dtype=np.float32).reshape(1, -1)

    for name in ["select_action", "get_action", "act"]:
        if hasattr(agent, name):
            fn = getattr(agent, name)
            try:
                action = fn(state_scaled)
            except Exception:
                action = fn(state_scaled.reshape(-1))
            return np.asarray(action, dtype=np.float32).reshape(-1)

    if not hasattr(agent, "actor"):
        raise AttributeError("No action API found. Expected select_action/get_action/act or actor.")

    if torch is None:
        raise ImportError("PyTorch is required to call agent.actor.")

    device = getattr(agent, "device", "cpu")
    with torch.no_grad():
        x = torch.as_tensor(state_scaled, dtype=torch.float32, device=device)
        action = agent.actor(x).detach().cpu().numpy().reshape(-1)
    return np.asarray(action, dtype=np.float32)


def unsafe_reason(env: Env, X: np.ndarray) -> str:
    """Classify which avoid condition is violated."""
    X = np.asarray(X, dtype=np.float32).reshape(-1)
    beta = float(env.beta(X))

    checks = [
        ("road_departure", abs(X[1]) > env.ey_max),
        ("heading_error", abs(X[2]) > env.epsi_max),
        ("low_forward_speed", X[3] < env.vx_min),
        ("high_forward_speed", X[3] > env.vx_max),
        ("large_lateral_speed", abs(X[4]) > env.vy_max),
        ("large_yaw_rate", abs(X[5]) > env.r_max),
        ("large_steering_angle", abs(X[6]) > env.delta_max),
        ("excessive_sideslip", abs(beta) > env.beta_max),
        ("low_friction", X[7] < env.mu_min),
        ("high_friction", X[7] > env.mu_max),
    ]

    for name, flag in checks:
        if flag:
            return name
    return "none"


def target_error(env: Env, X: np.ndarray) -> Dict[str, float]:
    """Normalized distance from the drift target. <=1 means inside tolerance."""
    X = np.asarray(X, dtype=np.float32).reshape(-1)
    beta = float(env.beta(X))
    return {
        "ey": abs(X[1] - env.ey_d) / env.ey_tol,
        "epsi": abs(X[2] - env.epsi_d) / env.epsi_tol,
        "vx": abs(X[3] - env.vx_d) / env.vx_tol,
        "beta": abs(beta - env.beta_d) / env.beta_tol,
        "r": abs(X[5] - env.r_d) / env.r_tol,
        "delta": abs(X[6] - env.delta_d) / env.delta_tol,
        "mu": abs(X[7] - env.mu_d) / 0.20,
    }


def set_initial_state(env: Env, initial: str) -> np.ndarray:
    """Set a reproducible initial state and return scaled state."""
    if initial == "env_reset":
        return env.reset()

    T = env.Tmax
    mu = env.mu_d

    presets = {
        "mild":          (0.8, 0.25, 1.0, 0.20, -0.50, 0.15),
        "wide":          (1.3, 0.35, 2.0, 0.35, -0.90, 0.25),
        "unsafe_prone":  (1.6, 0.45, 2.5, 0.45, -1.00, 0.30),
    }
    if initial not in presets:
        raise ValueError("initial must be env_reset, mild, wide, or unsafe_prone")

    ey, epsi_offset, vx_offset, beta_offset, r_offset, delta_offset = presets[initial]
    epsi = env.epsi_d + epsi_offset
    vx = env.vx_d + vx_offset
    beta = env.beta_d + beta_offset
    vy = vx * np.tan(beta)
    r = env.r_d + r_offset
    delta = env.delta_d + delta_offset

    env.state = np.array([T, ey, epsi, vx, vy, r, delta, mu], dtype=np.float32)
    return env.scale_state(env.state)


def configure_noise(env: Env, deterministic: bool) -> None:
    if not deterministic:
        return
    env.sigma_T = 0.0
    env.sigma_ey = 0.0
    env.sigma_epsi = 0.0
    env.sigma_vx = 0.0
    env.sigma_vy = 0.0
    env.sigma_r = 0.0
    env.sigma_delta = 0.0
    env.sigma_mu = 0.0


def rollout(agent: PIRLAgent, seed: int, initial: str, deterministic: bool,
            max_steps: int | None = None) -> RolloutResult:
    np.random.seed(seed)
    env = Env()
    configure_noise(env, deterministic=deterministic)

    state_scaled = set_initial_state(env, initial=initial)

    if max_steps is None:
        max_steps = int(np.ceil(env.Tmax / env.dt)) + 2

    states = [env.state.copy()]
    actions_scaled = []
    actions_phys = []
    rewards = []
    terminal_info = {"is_time_over": False, "is_target": False, "is_unsafe": False}

    done = False
    for _ in range(max_steps):
        action_scaled = np.clip(get_action_scaled(agent, state_scaled), -1.0, 1.0)
        action_phys = env._clip_action(env.unscale_action(action_scaled))

        next_state_scaled, reward, done, info = env.step(action_scaled)

        states.append(env.state.copy())
        actions_scaled.append(action_scaled.copy())
        actions_phys.append(action_phys.copy())
        rewards.append(float(reward))
        terminal_info = dict(info)
        state_scaled = next_state_scaled

        if done:
            break

    states = np.asarray(states, dtype=np.float32)
    actions_scaled = np.asarray(actions_scaled, dtype=np.float32)
    actions_phys = np.asarray(actions_phys, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)

    reason = unsafe_reason(env, states[-1]) if terminal_info.get("is_unsafe", False) else "none"

    return RolloutResult(
        states=states,
        actions_scaled=actions_scaled,
        actions_phys=actions_phys,
        rewards=rewards,
        done=done,
        terminal_info=terminal_info,
        unsafe_reason=reason,
        seed=seed,
    )


def reconstruct_path_coordinates(env: Env, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate path coordinate s and global XY for visualization."""
    s = np.zeros(states.shape[0], dtype=np.float64)

    for k in range(states.shape[0] - 1):
        ey = states[k, 1]
        epsi = states[k, 2]
        vx = states[k, 3]
        vy = states[k, 4]
        denom = max(1.0 - env.kappa_ref * ey, 0.2)
        s_dot = (vx * np.cos(epsi) - vy * np.sin(epsi)) / denom
        s[k + 1] = s[k] + env.dt * s_dot

    kappa = env.kappa_ref
    if abs(kappa) < 1.0e-8:
        x_ref = s
        y_ref = np.zeros_like(s)
        psi_ref = np.zeros_like(s)
    else:
        R = 1.0 / kappa
        psi_ref = kappa * s
        x_ref = R * np.sin(psi_ref)
        y_ref = R * (1.0 - np.cos(psi_ref))

    ey = states[:, 1].astype(np.float64)
    x_global = x_ref - ey * np.sin(psi_ref)
    y_global = y_ref + ey * np.cos(psi_ref)
    return s, x_global, y_global


def save_rollout_csv(result: RolloutResult, out_dir: str, prefix: str) -> None:
    env = Env()
    states = result.states
    beta = env.beta(states)
    t = env.Tmax - states[:, 0]

    df = pd.DataFrame({
        "t": t,
        "T": states[:, 0],
        "ey": states[:, 1],
        "epsi": states[:, 2],
        "vx": states[:, 3],
        "vy": states[:, 4],
        "r": states[:, 5],
        "delta": states[:, 6],
        "mu": states[:, 7],
        "beta": beta,
    })

    for j, name in enumerate(["delta_dot", "Fx"]):
        arr = np.full(len(df), np.nan)
        if len(result.actions_phys) > 0:
            arr[:-1] = result.actions_phys[:, j]
        df[name] = arr

    path = os.path.join(out_dir, f"{prefix}_trajectory.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def plot_time_histories(result: RolloutResult, out_dir: str, prefix: str) -> None:
    env = Env()
    states = result.states
    t = env.Tmax - states[:, 0]
    beta = env.beta(states)

    fig, axes = plt.subplots(4, 2, figsize=(8.0, 8.0), sharex=True)
    axes = axes.ravel()

    series = [
        (states[:, 1], r"$e_y$ [m]", (-env.ey_max, env.ey_max), env.ey_d, env.ey_tol),
        (states[:, 2], r"$e_\psi$ [rad]", (-env.epsi_max, env.epsi_max), env.epsi_d, env.epsi_tol),
        (states[:, 3], r"$v_x$ [m/s]", (env.vx_min, env.vx_max), env.vx_d, env.vx_tol),
        (beta, r"$\beta$ [rad]", (-env.beta_max, env.beta_max), env.beta_d, env.beta_tol),
        (states[:, 5], r"$r$ [rad/s]", (-env.r_max, env.r_max), env.r_d, env.r_tol),
        (states[:, 6], r"$\delta$ [rad]", (-env.delta_max, env.delta_max), env.delta_d, env.delta_tol),
    ]

    for ax, (y, ylabel, bounds, ref, tol) in zip(axes[:6], series):
        ax.plot(t, y)
        ax.axhline(bounds[0], linestyle="--", linewidth=1)
        ax.axhline(bounds[1], linestyle="--", linewidth=1)
        ax.axhline(ref, linestyle="-.", linewidth=1)
        ax.axhspan(ref - tol, ref + tol, alpha=0.15)
        ax.set_ylabel(ylabel)

    if len(result.actions_phys) > 0:
        ta = t[:-1]
        axes[6].plot(ta, result.actions_phys[:, 0])
        axes[6].axhline(-env.delta_dot_max, linestyle="--", linewidth=1)
        axes[6].axhline(env.delta_dot_max, linestyle="--", linewidth=1)
        axes[6].set_ylabel(r"$\dot{\delta}$ [rad/s]")

        axes[7].plot(ta, result.actions_phys[:, 1])
        axes[7].axhline(env.Fx_min, linestyle="--", linewidth=1)
        axes[7].axhline(env.Fx_max, linestyle="--", linewidth=1)
        axes[7].set_ylabel(r"$F_x$ [N]")

    axes[-2].set_xlabel("Time [s]")
    axes[-1].set_xlabel("Time [s]")

    title = (
        f"seed={result.seed}, "
        f"target={result.terminal_info.get('is_target', False)}, "
        f"unsafe={result.terminal_info.get('is_unsafe', False)}, "
        f"reason={result.unsafe_reason}"
    )
    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    path = os.path.join(out_dir, f"{prefix}_time_histories.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_phase_plane(result: RolloutResult, out_dir: str, prefix: str) -> None:
    env = Env()
    states = result.states
    beta = env.beta(states)
    r = states[:, 5]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(beta, r, marker="o", markersize=2)
    ax.scatter(beta[0], r[0], marker="s", label="initial")
    ax.scatter(beta[-1], r[-1], marker="*", s=120, label="terminal")

    ax.axvline(-env.beta_max, linestyle="--", linewidth=1)
    ax.axvline(env.beta_max, linestyle="--", linewidth=1)
    ax.axhline(-env.r_max, linestyle="--", linewidth=1)
    ax.axhline(env.r_max, linestyle="--", linewidth=1)

    beta_lo = env.beta_d - env.beta_tol
    beta_hi = env.beta_d + env.beta_tol
    r_lo = env.r_d - env.r_tol
    r_hi = env.r_d + env.r_tol
    ax.fill([beta_lo, beta_hi, beta_hi, beta_lo],
            [r_lo, r_lo, r_hi, r_hi], alpha=0.2, label="target window")

    ax.set_xlabel(r"$\beta$ [rad]")
    ax.set_ylabel(r"$r$ [rad/s]")
    #ax.legend(frameon=True)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{prefix}_beta_r_phase.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_path_view(result: RolloutResult, out_dir: str, prefix: str) -> None:
    env = Env()
    states = result.states
    s, x_global, y_global = reconstruct_path_coordinates(env, states)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(s, states[:, 1], marker="o", markersize=2)
    ax.axhline(-env.ey_max, linestyle="--", linewidth=1)
    ax.axhline(env.ey_max, linestyle="--", linewidth=1)
    ax.axhline(env.ey_d, linestyle="-.", linewidth=1)
    ax.axhspan(env.ey_d - env.ey_tol, env.ey_d + env.ey_tol, alpha=0.15)
    ax.set_xlabel(r"path coordinate $s$ [m]")
    ax.set_ylabel(r"$e_y$ [m]")
    fig.tight_layout()

    path = os.path.join(out_dir, f"{prefix}_ey_vs_s.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot(x_global, y_global, marker="o", markersize=2)
    ax.scatter(x_global[0], y_global[0], marker="s", label="initial")
    ax.scatter(x_global[-1], y_global[-1], marker="*", s=120, label="terminal")
    ax.set_xlabel(r"$X$ [m]")
    ax.set_ylabel(r"$Y$ [m]")
    ax.axis("equal")
    #ax.legend(frameon=True)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{prefix}_global_path.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_target_error(result: RolloutResult, out_dir: str, prefix: str) -> None:
    env = Env()
    states = result.states
    t = env.Tmax - states[:, 0]

    keys = ["ey", "epsi", "vx", "beta", "r", "delta", "mu"]
    E = {k: [] for k in keys}

    for X in states:
        e = target_error(env, X)
        for k in keys:
            E[k].append(e[k])

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for k in keys:
        ax.plot(t, E[k], label=k)

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("normalized target error")
    ax.set_yscale("log")
    #ax.legend(frameon=True, ncol=4)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{prefix}_target_error.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_combined_phase_plane(results: List[RolloutResult], out_dir: str) -> None:
    env = Env()

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    for res in results:
        beta = env.beta(res.states)
        r = res.states[:, 5]
        label = f"seed {res.seed}"
        ax.plot(beta, r, marker="o", markersize=2, linewidth=1.2, alpha=0.8, label=label)
        ax.scatter(beta[0], r[0], marker="s", s=30, alpha=0.8)
        ax.scatter(beta[-1], r[-1], marker="*", s=80, alpha=0.9)

    ax.axvline(-env.beta_max, linestyle="--", linewidth=1)
    ax.axvline(env.beta_max, linestyle="--", linewidth=1)
    ax.axhline(-env.r_max, linestyle="--", linewidth=1)
    ax.axhline(env.r_max, linestyle="--", linewidth=1)

    beta_lo = env.beta_d - env.beta_tol
    beta_hi = env.beta_d + env.beta_tol
    r_lo = env.r_d - env.r_tol
    r_hi = env.r_d + env.r_tol
    ax.fill(
        [beta_lo, beta_hi, beta_hi, beta_lo],
        [r_lo, r_lo, r_hi, r_hi],
        alpha=0.18,
        label="target window",
    )

    ax.scatter(env.beta_d, env.r_d, marker="x", s=80, linewidths=2, label="drift equilibrium")

    ax.set_xlabel(r"$\beta$ [rad]")
    ax.set_ylabel(r"$r$ [rad/s]")
    #ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    path = os.path.join(out_dir, "combined_beta_r_phase.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    plt.close(fig)


def plot_combined_path_view(results: List[RolloutResult], out_dir: str) -> None:
    env = Env()

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    for res in results:
        s, _, _ = reconstruct_path_coordinates(env, res.states)
        ax.plot(s, res.states[:, 1], marker="o", markersize=2, linewidth=1.2, alpha=0.8, label=f"seed {res.seed}")
        ax.scatter(s[0], res.states[0, 1], marker="s", s=30, alpha=0.8)
        ax.scatter(s[-1], res.states[-1, 1], marker="*", s=80, alpha=0.9)

    ax.axhline(-env.ey_max, linestyle="--", linewidth=1)
    ax.axhline(env.ey_max, linestyle="--", linewidth=1)
    ax.axhline(env.ey_d, linestyle="-.", linewidth=1)
    ax.axhspan(env.ey_d - env.ey_tol, env.ey_d + env.ey_tol, alpha=0.15)

    ax.set_xlabel(r"path coordinate $s$ [m]")
    ax.set_ylabel(r"$e_y$ [m]")
    #ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    path = os.path.join(out_dir, "combined_ey_vs_s.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    for res in results:
        _, x_global, y_global = reconstruct_path_coordinates(env, res.states)
        ax.plot(x_global, y_global, marker="o", markersize=2, linewidth=1.2, alpha=0.8, label=f"seed {res.seed}")
        ax.scatter(x_global[0], y_global[0], marker="s", s=30, alpha=0.8)
        ax.scatter(x_global[-1], y_global[-1], marker="*", s=80, alpha=0.9)

    ax.set_xlabel(r"$X$ [m]")
    ax.set_ylabel(r"$Y$ [m]")
    ax.axis("equal")
    #ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    path = os.path.join(out_dir, "combined_global_path.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    plt.close(fig)


def plot_combined_target_error(results: List[RolloutResult], out_dir: str) -> None:
    env = Env()

    keys = ["ey", "epsi", "vx", "beta", "r", "delta"]
    labels = {
        "ey": r"$e_y$",
        "epsi": r"$e_\psi$",
        "vx": r"$v_x$",
        "beta": r"$\beta$",
        "r": r"$r$",
        "delta": r"$\delta$",
    }

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0), sharex=True)
    axes = axes.ravel()

    for ax, key in zip(axes, keys):
        for res in results:
            t = env.Tmax - res.states[:, 0]
            vals = []
            for X in res.states:
                vals.append(target_error(env, X)[key])
            vals = np.asarray(vals, dtype=np.float64)
            ax.plot(t, vals, linewidth=1.1, alpha=0.75, label=f"seed {res.seed}")

        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylabel(labels[key])

    axes[-2].set_xlabel("Time [s]")
    axes[-1].set_xlabel("Time [s]")
    #axes[0].legend(frameon=True, fontsize=8, ncol=2)

    fig.tight_layout()

    path = os.path.join(out_dir, "combined_target_error_panel.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    plt.close(fig)


def plot_failure_reason_histogram(results: List[RolloutResult], out_dir: str) -> None:
    reasons = []
    for res in results:
        if res.terminal_info.get("is_target", False):
            reasons.append("target")
        elif res.terminal_info.get("is_unsafe", False):
            reasons.append(res.unsafe_reason)
        elif res.terminal_info.get("is_time_over", False):
            reasons.append("time_over")
        else:
            reasons.append("not_done")

    unique = sorted(set(reasons))
    counts = [reasons.count(u) for u in unique]

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.bar(np.arange(len(unique)), counts)
    ax.set_xticks(np.arange(len(unique)))
    ax.set_xticklabels(unique, rotation=25, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Terminal condition summary")
    fig.tight_layout()

    path = os.path.join(out_dir, "combined_terminal_reason_histogram.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    plt.close(fig)


def plot_combined_summary(results: List[RolloutResult], out_dir: str) -> None:
    plot_combined_phase_plane(results, out_dir)
    plot_combined_path_view(results, out_dir)
    plot_combined_target_error(results, out_dir)
    plot_failure_reason_histogram(results, out_dir)


def summarize_results(results: List[RolloutResult], out_dir: str) -> None:
    rows = []

    for res in results:
        env = Env()
        final = res.states[-1]
        beta_final = float(env.beta(final))
        terr = target_error(env, final)

        row = {
            "seed": res.seed,
            "steps": len(res.states) - 1,
            "done": res.done,
            "is_time_over": res.terminal_info.get("is_time_over", False),
            "is_target": res.terminal_info.get("is_target", False),
            "is_unsafe": res.terminal_info.get("is_unsafe", False),
            "unsafe_reason": res.unsafe_reason,
            "final_T": final[0],
            "final_ey": final[1],
            "final_epsi": final[2],
            "final_vx": final[3],
            "final_vy": final[4],
            "final_beta": beta_final,
            "final_r": final[5],
            "final_delta": final[6],
            "final_mu": final[7],
        }
        for k, v in terr.items():
            row[f"target_error_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "rollout_summary.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

    print("--------------------------------------------")
    print("Rollout summary")
    print(df[["seed", "steps", "is_target", "is_unsafe", "unsafe_reason"]])
    print("--------------------------------------------")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", 
                        default = "logs/drift/0530_0446_seed_1/ckpt-100000")
    parser.add_argument("--out_dir", default="plot/drift_rollout_analysis")
    parser.add_argument("--seeds", type=int, nargs="+", default=[i for i in range(20)])
    parser.add_argument("--initial", default="env_reset",
                        choices=["env_reset", "mild", "wide", "unsafe_prone"])
    parser.add_argument("--stochastic", action="store_true",
                        help="Use environment process noise. Default is deterministic.")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_paper_style()

    agent = PIRLAgent.from_checkpoint(args.checkpoint, learner=False)

    results = []
    for seed in args.seeds:
        res = rollout(
            agent=agent,
            seed=seed,
            initial=args.initial,
            deterministic=not args.stochastic,
            max_steps=args.max_steps,
        )
        results.append(res)

        prefix = f"seed_{seed:03d}_{args.initial}"
        save_rollout_csv(res, args.out_dir, prefix)
        plot_time_histories(res, args.out_dir, prefix)
        plot_phase_plane(res, args.out_dir, prefix)
        plot_path_view(res, args.out_dir, prefix)
        plot_target_error(res, args.out_dir, prefix)

    summarize_results(results, args.out_dir)
    plot_combined_summary(results, args.out_dir)


if __name__ == "__main__":
    main()
