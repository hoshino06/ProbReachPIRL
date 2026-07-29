# -*- coding: utf-8 -*-
"""
Value-function contour plots for the drifting-control reach-avoid task.

This script evaluates a trained PIRL/TD3 agent on two 2-D slices:
  1) beta-r plane
  2) ey-epsi plane

It saves:
  - value_contours_combined.png / .pdf
  - value_contour_beta_r.png / .pdf
  - value_contour_ey_epsi.png / .pdf
  - value_grid_beta_r.npz
  - value_grid_ey_epsi.npz

Typical usage from the project root:
    python plot_drift_value_contours.py \
        --checkpoint logs/drift/0530_0446_seed_1/ckpt-67000 \
        --out_dir plot/drift_value_contours \
        --num_grid 151 \
        --T 5.0 \
        --mu target

Assumptions:
  - The agent was trained with scaled states.
  - Env.make_eval_states returns physical states, so this script scales them
    before passing them to agent.get_value.
"""

from __future__ import annotations

import os
import sys
import argparse
import importlib
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "mathtext.fontset": "cm",
        "figure.dpi": 120,
    })


def backend_supports_show() -> bool:
    """Return True for interactive backends such as Spyder's plots pane."""
    backend = plt.get_backend().lower()
    noninteractive_tokens = ("agg", "pdf", "svg", "ps", "cairo", "template")
    return not any(token in backend for token in noninteractive_tokens)


def import_from_string(path: str):
    """Import 'module:object' or 'module.object'."""
    if ":" in path:
        module_name, obj_name = path.split(":", 1)
    else:
        module_name, obj_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def resolve_mu(env, mu_arg: str) -> float:
    if mu_arg.lower() in ["target", "drift", "mu_d"]:
        return float(env.mu_d)
    return float(mu_arg)


def evaluate_value_grid(agent, env, plane: str, T: float, mu: float,
                        num_grid: int, batch_size: int) -> Tuple[np.ndarray, dict, np.ndarray]:
    """Return V grid with shape (num_grid, num_grid), metadata, and physical states."""
    states_phys, meta = env.make_eval_states(T=T, num_grid=num_grid, plane=plane, mu=mu)
    states_scaled = env.scale_state(states_phys)

    values = []
    for i in range(0, len(states_scaled), batch_size):
        vb = agent.get_value(states_scaled[i:i + batch_size])
        values.append(np.asarray(vb, dtype=np.float64).reshape(-1))

    V = np.concatenate(values, axis=0).reshape(num_grid, num_grid)
    return V, meta, states_phys


def evaluate_vector_field(agent, env, plane: str, states_phys: np.ndarray,
                          num_grid: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closed-loop vector field projected onto the selected 2-D plane."""
    states_scaled = env.scale_state(states_phys)

    actions_scaled = []
    for i in range(0, len(states_scaled), batch_size):
        if hasattr(agent, "get_action"):
            ab = agent.get_action(states_scaled[i:i + batch_size])
        elif hasattr(agent, "select_action"):
            ab = agent.select_action(states_scaled[i:i + batch_size])
        elif hasattr(agent, "act"):
            ab = agent.act(states_scaled[i:i + batch_size])
        else:
            raise AttributeError("No action API found. Expected get_action/select_action/act.")
        actions_scaled.append(np.asarray(ab, dtype=np.float32).reshape(-1, env.action_dim))

    actions_scaled = np.concatenate(actions_scaled, axis=0)
    actions_phys = env._clip_action(env.unscale_action(actions_scaled))
    drift, _ = env.drift_and_diffusion(states_phys, actions_phys)
    drift = np.asarray(drift, dtype=np.float64)

    if plane == "beta_r":
        vx = np.asarray(states_phys[:, 3], dtype=np.float64)
        vy = np.asarray(states_phys[:, 4], dtype=np.float64)
        vx_dot = drift[:, 3]
        vy_dot = drift[:, 4]
        denom = np.maximum(vx * vx + vy * vy, 1.0e-9)
        x_dot = (vx * vy_dot - vy * vx_dot) / denom
        y_dot = drift[:, 5]
    elif plane == "ey_epsi":
        x_dot = drift[:, 1]
        y_dot = drift[:, 2]
    else:
        raise ValueError(plane)

    return (
        x_dot.reshape(num_grid, num_grid),
        y_dot.reshape(num_grid, num_grid),
        actions_phys.reshape(num_grid, num_grid, env.action_dim),
    )


def target_rectangle(env, plane: str):
    if plane == "beta_r":
        x0 = env.beta_d - env.beta_tol
        x1 = env.beta_d + env.beta_tol
        y0 = env.r_d - env.r_tol
        y1 = env.r_d + env.r_tol
    elif plane == "ey_epsi":
        x0 = env.ey_d - env.ey_tol
        x1 = env.ey_d + env.ey_tol
        y0 = env.epsi_d - env.epsi_tol
        y1 = env.epsi_d + env.epsi_tol
    else:
        raise ValueError(plane)
    return x0, x1, y0, y1


def plane_axis_labels(plane: str) -> tuple[str, str]:
    if plane == "beta_r":
        return r"$\beta$ [rad]", r"$r$ [rad/s]"
    if plane == "ey_epsi":
        return r"$e_y$ [m]", r"$e_\psi$ [rad]"
    raise ValueError(plane)


def add_target_patch(ax, env, plane: str, label: str = "target set") -> None:
    x0, x1, y0, y1 = target_rectangle(env, plane)
    ax.fill([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            facecolor="none", edgecolor="black", linewidth=1.8,
            linestyle="-", label=label)

    if plane == "beta_r":
        ax.scatter(env.beta_d, env.r_d, marker="x", s=70, linewidths=2.0,
                   color="black", label="equilibrium")
    elif plane == "ey_epsi":
        ax.scatter(env.ey_d, env.epsi_d, marker="x", s=70, linewidths=2.0,
                   color="black", label="equilibrium")


def plot_value_contour(ax, env, V: np.ndarray, meta: dict, plane: str,
                       levels: int, vmin: Optional[float], vmax: Optional[float],
                       title: str, vector_field: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                       vector_stride: int = 8, vector_scale: Optional[float] = None):
    x = meta["x"]
    y = meta["y"]

    if vmin is None:
        vmin = float(np.nanmin(V))
    if vmax is None:
        vmax = float(np.nanmax(V))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0e-6

    cf = ax.contourf(x, y, V.T, levels=np.linspace(vmin, vmax, levels),
                     vmin=vmin, vmax=vmax, extend="both")
    cs = ax.contour(x, y, V.T, levels=np.linspace(vmin, vmax, 8),
                    linewidths=0.6, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    if vector_field is not None:
        x_dot, y_dot = vector_field
        stride = max(1, int(vector_stride))
        X, Y = np.meshgrid(x, y, indexing="xy")
        q = ax.quiver(
            X[::stride, ::stride],
            Y[::stride, ::stride],
            x_dot.T[::stride, ::stride],
            y_dot.T[::stride, ::stride],
            color="white",
            edgecolor="black",
            linewidth=0.25,
            alpha=0.85,
            pivot="mid",
            angles="xy",
            scale_units="xy",
            scale=vector_scale,
            width=0.0032,
            label="closed-loop drift",
        )
        ax.quiverkey(q, X=0.86, Y=1.04, U=1.0, label="1 unit/s", labelpos="E")

    add_target_patch(ax, env, plane)

    xlabel, ylabel = plane_axis_labels(plane)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return cf


def save_npz(path: str, V: np.ndarray, meta: dict,
             vector_field: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             actions_phys: Optional[np.ndarray] = None) -> None:
    data = {
        "V": V,
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
    if actions_phys is not None:
        data["actions_phys"] = actions_phys
    np.savez(path, **data)


DEFAULT_PIRL_CHECKPOINT = (
    "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_004/"
    "mix85_back003_R5/train/0708_2216_mix85_back003_R5_seed_1/ckpt-15000000"
)


def plot_value_panel(ax, env, V: np.ndarray, meta: dict, plane: str, title: str,
                     levels: int, vector_field: Optional[Tuple[np.ndarray, np.ndarray]],
                     vector_stride: int, vector_scale: Optional[float]):
    V = np.clip(V, 0.0, 1.0)
    x = meta["x"]
    y = meta["y"]
    cf = ax.contourf(
        x,
        y,
        V.T,
        levels=np.linspace(0.0, 1.0, levels),
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        extend="both",
    )
    cs = ax.contour(x, y, V.T, levels=np.linspace(0.0, 1.0, 6), linewidths=0.5, alpha=0.55)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

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
            alpha=0.72,
            pivot="mid",
            angles="xy",
            scale_units="xy",
            scale=vector_scale,
            width=0.0025,
        )

    add_target_patch(ax, env, plane)
    xlabel, ylabel = plane_axis_labels(plane)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return cf


def evaluate_checkpoint(agent_cls, env, checkpoint: str, label: str, args, mu: float):
    agent = agent_cls.from_checkpoint(checkpoint, device=args.device, learner=False)
    results = {}
    for plane in ["beta_r", "ey_epsi"]:
        V, meta, states = evaluate_value_grid(
            agent,
            env,
            plane=plane,
            T=args.T,
            mu=mu,
            num_grid=args.num_grid,
            batch_size=args.batch_size,
        )
        vector = None
        actions = None
        if not args.no_vector_field:
            xdot, ydot, actions = evaluate_vector_field(
                agent,
                env,
                plane=plane,
                states_phys=states,
                num_grid=args.num_grid,
                batch_size=args.batch_size,
            )
            vector = (xdot, ydot)

        save_npz(
            os.path.join(args.out_dir, f"value_grid_{label}_{plane}.npz"),
            V,
            meta,
            vector_field=vector,
            actions_phys=actions,
        )
        results[plane] = {"V": V, "meta": meta, "vector": vector}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_PIRL_CHECKPOINT,
                        help="Checkpoint to plot. Defaults to the selected mixed-HJB PIRL 15M checkpoint.")
    parser.add_argument("--label", default="pirl_15M_mix85",
                        help="Short label used in output NPZ filenames.")
    parser.add_argument("--out_dir", default="plot/drift_value_contours")
    parser.add_argument("--env_cls", default="examples.env_drifting_control.Env")
    parser.add_argument("--agent_cls", default="agent.TD3_PIRL_ray.PIRLAgent")
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--mu", default="target")
    parser.add_argument("--num_grid", type=int, default=31)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--levels", type=int, default=41)
    parser.add_argument("--no_vector_field", action="store_true")
    parser.add_argument("--vector_stride", type=int, default=1)
    parser.add_argument("--vector_scale", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_paper_style()
    show_figures = backend_supports_show()

    Env = import_from_string(args.env_cls)
    Agent = import_from_string(args.agent_cls)
    env = Env()
    mu = resolve_mu(env, args.mu)

    result = evaluate_checkpoint(Agent, env, args.checkpoint, args.label, args, mu)

    plane_specs = [
        ("beta_r", "", "fig_section_vb3_value_contour_beta_r"),
        ("ey_epsi", "", "fig_section_vb3_value_contour_ey_epsi"),
    ]
    for plane, title, stem in plane_specs:
        fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
        cf = plot_value_panel(
            ax,
            env,
            result[plane]["V"],
            result[plane]["meta"],
            plane,
            title,
            args.levels,
            result[plane]["vector"],
            args.vector_stride,
            args.vector_scale,
        )
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r"learned reachability value $V(x)$", fontsize=17)
        cbar.ax.tick_params(labelsize=14)
        legend_loc = "lower right" if plane == "beta_r" else "lower left"
        ax.legend(frameon=True, loc=legend_loc, borderpad=0.35, handlelength=1.5, labelspacing=0.35)
        for ext in ["png", "pdf"]:
            path = os.path.join(args.out_dir, f"{stem}.{ext}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved: {path}")
        if not show_figures:
            plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    combined_cf = None
    for ax, (plane, title, _) in zip(axes, plane_specs):
        combined_cf = plot_value_panel(
            ax,
            env,
            result[plane]["V"],
            result[plane]["meta"],
            plane,
            title,
            args.levels,
            result[plane]["vector"],
            args.vector_stride,
            args.vector_scale,
        )
        legend_loc = "lower right" if plane == "beta_r" else "best"
        ax.legend(frameon=True, loc=legend_loc, borderpad=0.35, handlelength=1.5, labelspacing=0.35)
    if combined_cf is not None:
        cbar = fig.colorbar(combined_cf, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label(r"learned reachability value $V(x)$", fontsize=17)
        cbar.ax.tick_params(labelsize=14)
    for ext in ["png", "pdf"]:
        path = os.path.join(args.out_dir, f"fig_section_vb3_value_contours.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    if not show_figures:
        plt.close(fig)
    else:
        plt.show()

    print("--------------------------------------------")
    print(f"checkpoint:      {args.checkpoint}")
    print(f"out_dir:         {args.out_dir}")
    print(f"T:               {args.T:g}")
    print(f"mu:              {mu:.4f}")
    print(f"grid:            {args.num_grid} x {args.num_grid}")
    print("--------------------------------------------")


if __name__ == "__main__":
    # Make execution from a copied script slightly more forgiving.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    main()
