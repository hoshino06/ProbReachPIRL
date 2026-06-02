# -*- coding: utf-8 -*-
"""
Plot learned value, ground truth, and absolute error for the 1D reachability example.

Changes from the initial 3D script:
  1. The 3D surface uses T as the horizontal axis and x as the depth axis.
  2. The viewing angle is adjusted so that the transition surface is easier to see.
  3. Paper-friendly heatmaps are also saved, because the error distribution is
     often clearer in 2D than in 3D.

Example:
    python plot_1d_surface_scheduling_revised.py \
        --checkpoint logs/1D/scheduling/0528_2258_seed_1/ckpt-200000 \
        --out_dir plot/1d_surface_revised
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from agent.TD3 import PIRLAgent


def true_reach_probability(T, x, mu=1.0, sigma=1.0, x_target=2.0):
    """Finite-horizon reach probability for dX = mu dt + sigma dW."""
    T = np.asarray(T, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    T, x = np.broadcast_arrays(T, x)

    prob = np.ones_like(x, dtype=np.float64)

    mask_t0 = T <= 0.0
    prob[mask_t0] = (x[mask_t0] >= x_target).astype(np.float64)

    mask = (~mask_t0) & (x < x_target)
    sqrt_T = np.sqrt(T[mask])
    z1 = (x[mask] - x_target + mu * T[mask]) / (sigma * sqrt_T)
    z2 = (x[mask] - x_target - mu * T[mask]) / (sigma * sqrt_T)

    prob[mask] = (
        norm.cdf(z1)
        + np.exp(2.0 * mu * (x_target - x[mask]) / (sigma**2)) * norm.cdf(z2)
    )
    return np.clip(prob, 0.0, 1.0)


def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "mathtext.fontset": "dejavuserif",
    })


def plot_surface(T_grid, X_grid, Z, title, zlabel, out_path,
                 zlim=None, cmap="viridis", elev=28, azim=-60):
    """
    3D surface with T on the x-axis and x on the y-axis.
    This is often easier to interpret for V(T,x), because the transition line
    x ~= x_target - mu T appears diagonally on the base plane.
    """
    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        T_grid, X_grid, Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$x$")
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    ax.set_xlim(np.nanmin(T_grid), np.nanmax(T_grid))
    ax.set_ylim(np.nanmin(X_grid), np.nanmax(X_grid))
    if zlim is not None:
        ax.set_zlim(*zlim)

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.25, 1.25, 0.75))

    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_heatmap(T, x, Z, title, cbar_label, out_path,
                 vmin=None, vmax=None, cmap="viridis"):
    """2D heatmap with T on the horizontal axis and x on the vertical axis."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[T.min(), T.max(), x.min(), x.max()],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$x$")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_panel(T, x, V_learned, V_true, V_error, out_path):
    """One paper-friendly panel: learned / ground truth / absolute error."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)

    data = [V_learned, V_true, V_error]
    titles = ["Learned value", "Ground truth", "Absolute error"]
    cmaps = ["viridis", "viridis", "magma"]
    vmins = [0.0, 0.0, 0.0]
    vmaxs = [1.0, 1.0, max(float(np.nanmax(V_error)), 1.0e-12)]

    for ax, Z, title, cmap, vmin, vmax in zip(axes, data, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[T.min(), T.max(), x.min(), x.max()],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel(r"$T$")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if title == "Absolute error":
            cbar.set_label(r"$|V_{\mathrm{learned}}-V_{\mathrm{true}}|$")
        else:
            cbar.set_label("Reach probability")

    axes[0].set_ylabel(r"$x$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs/1D/scheduling/0530_0026_seed_1/ckpt-200000")
    #parser.add_argument("--checkpoint", default="logs/1D/0509_0555_seed_1/ckpt-20000")
    parser.add_argument("--out_dir", default="plot/1d_surface")
    parser.add_argument("--num_x", type=int, default=121)
    parser.add_argument("--num_T", type=int, default=81)
    parser.add_argument("--x_min", type=float, default=-2.0)
    parser.add_argument("--x_max", type=float, default=2.0)
    parser.add_argument("--Tmax", type=float, default=2.0)
    parser.add_argument("--x_target", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--elev", type=float, default=28.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_paper_style()

    agent = PIRLAgent.from_checkpoint(args.checkpoint)

    x = np.linspace(args.x_min, args.x_max, args.num_x)
    T = np.linspace(0.0, args.Tmax, args.num_T)

    # Shape: (num_x, num_T), so heatmaps naturally use x as vertical and T as horizontal.
    T_grid, X_grid = np.meshgrid(T, x)
    states = np.column_stack([T_grid.ravel(), X_grid.ravel()]).astype(np.float32)

    V_learned = np.asarray(agent.get_value(states)).reshape(X_grid.shape)
    V_learned = np.clip(V_learned, 0.0, 1.0)
    V_true = true_reach_probability(
        T_grid, X_grid, mu=args.mu, sigma=args.sigma, x_target=args.x_target
    )
    V_error = np.abs(V_learned - V_true)

    np.savez(
        os.path.join(args.out_dir, "fig_1d_surface_data.npz"),
        x=x,
        T=T,
        X_grid=X_grid,
        T_grid=T_grid,
        V_learned=V_learned,
        V_true=V_true,
        V_error=V_error,
    )

    # 3D surfaces: T-axis and x-axis are swapped compared with the first version.
    plot_surface(
        T_grid, X_grid, V_learned,
        title="Learned value (Scheduling)",
        zlabel="Reach probability",
        out_path=os.path.join(args.out_dir, "fig_1d_surface_learned.png"),
        zlim=(0.0, 1.0),
        elev=args.elev,
        azim=args.azim,
    )
    plot_surface(
        T_grid, X_grid, V_true,
        title="Ground truth",
        zlabel="Reach probability",
        out_path=os.path.join(args.out_dir, "fig_1d_surface_ground_truth.png"),
        zlim=(0.0, 1.0),
        elev=args.elev,
        azim=args.azim,
    )
    plot_surface(
        T_grid, X_grid, V_error,
        title="Absolute error",
        zlabel="Absolute error",
        out_path=os.path.join(args.out_dir, "fig_1d_surface_error.png"),
        zlim=(0.0, max(float(np.nanmax(V_error)), 1.0e-12)),
        cmap="magma",
        elev=args.elev,
        azim=args.azim,
    )

    # 2D heatmaps: usually easier for paper figures and error discussion.
    plot_heatmap(
        T, x, V_learned,
        title="Learned value",
        cbar_label="Reach probability",
        out_path=os.path.join(args.out_dir, "fig_1d_heatmap_learned.png"),
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        T, x, V_true,
        title="Ground truth",
        cbar_label="Reach probability",
        out_path=os.path.join(args.out_dir, "fig_1d_heatmap_ground_truth.png"),
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        T, x, V_error,
        title="Absolute error",
        cbar_label=r"$|V_{\mathrm{learned}}-V_{\mathrm{true}}|$",
        out_path=os.path.join(args.out_dir, "fig_1d_heatmap_error.png"),
        vmin=0.0,
        vmax=max(float(np.nanmax(V_error)), 1.0e-12),
        cmap="magma",
    )
    plot_heatmap_panel(
        T, x, V_learned, V_true, V_error,
        out_path=os.path.join(args.out_dir, "fig_1d_heatmap_panel.png"),
    )

    print(f"Saved figures and data to: {args.out_dir}")


if __name__ == "__main__":
    main()
