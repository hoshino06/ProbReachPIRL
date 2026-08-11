# -*- coding: utf-8 -*-
"""Paper figure for value accuracy versus distance from TD3 occupancy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_drift_value_contours import set_paper_style


METHODS = [
    ("TD3 fixed tau 10M", r"TD3 fixed $\tau$ 10M", "#222222", "s", "--"),
    ("TD3 random tau 10M", r"TD3 random $\tau$ 10M", "#777777", "o", "-"),
    ("PIRL 10M", "PIRL 10M", "#4c78a8", "^", "-"),
    ("PIRL mixed 15M", "PIRL 15M", "#d95f02", "D", "-"),
]


def bootstrap_mse_ci(error: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    if len(error) < 2:
        return np.nan, np.nan
    indices = rng.integers(0, len(error), size=(draws, len(error)))
    samples = np.mean(error[indices] ** 2, axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def add_region_guides(ax, centers: np.ndarray) -> None:
    split_1 = 0.5 * (centers[1] + centers[2])
    split_2 = 0.5 * (centers[3] + centers[4])
    left = centers[0] - 0.55 * (centers[1] - centers[0])
    right = centers[-1] + 0.35 * (centers[-1] - centers[-2])
    ax.axvspan(left, split_1, color="#e8f1f8", alpha=0.65, zorder=0)
    ax.axvspan(split_1, split_2, color="#fff1d6", alpha=0.62, zorder=0)
    ax.axvspan(split_2, right, color="#eeeeee", alpha=0.70, zorder=0)
    ax.axvline(split_1, color="#8aa9bd", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axvline(split_2, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.8)
    ymax = ax.get_ylim()[1]
    ax.text(np.mean([left, split_1]), 0.96 * ymax, "near TD3\nsupport", ha="center", va="top", fontsize=11)
    ax.text(np.mean([split_1, split_2]), 0.96 * ymax, "extrapolation\nregion", ha="center", va="top", fontsize=11)
    ax.text(np.mean([split_2, right]), 0.96 * ymax, "sparse TD3\nsupport", ha="center", va="top", fontsize=10.5)


def panel(
    ax,
    data,
    methods,
    mask: np.ndarray,
    title: str | None,
    draws: int,
    seed: int,
) -> None:
    distance = data["occupancy_distance"]
    bins = data["distance_bin"]
    num_bins = int(bins.max()) + 1
    centers = np.array([distance[bins == b].mean() for b in range(num_bins)])
    positions = np.arange(num_bins, dtype=float)
    counts = np.array([np.sum(mask & (bins == b)) for b in range(num_bins)])
    rng = np.random.default_rng(seed)

    ymax = 0.0
    plotted = []
    for key, label, color, marker, linestyle in methods:
        error = data[f"{key}__error"]
        means, lows, highs = [], [], []
        for b in range(num_bins):
            selected = error[mask & (bins == b)]
            means.append(float(np.mean(selected ** 2)) if len(selected) else np.nan)
            lo, hi = bootstrap_mse_ci(selected, rng, draws)
            lows.append(lo)
            highs.append(hi)
        means = np.asarray(means)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        ymax = max(ymax, float(np.nanmax(highs)))
        plotted.append((label, color, marker, linestyle, means, lows, highs))

    ax.set_ylim(0.0, max(0.12, 1.22 * ymax))
    add_region_guides(ax, positions)
    for label, color, marker, linestyle, means, lows, highs in plotted:
        yerr = np.vstack([means - lows, highs - means])
        ax.errorbar(
            positions,
            means,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=6.2,
            linewidth=2.0,
            linestyle=linestyle,
            elinewidth=1.0,
            capsize=2.5,
            label=label,
            zorder=3,
        )

    semantic_ticks = ["closest", "near", "moderate", "far", "farthest"]
    tick_labels = [f"{name}\n$d={d:.2f}$\n$n={n}$" for name, d, n in zip(semantic_ticks, centers, counts)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=11.5)
    ax.set_xlim(-0.45, num_bins - 0.55)
    ax.set_ylabel(r"value-function MSE  $\mathbb{E}[(V-\hat p_{MC})^2]$", fontsize=13)
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=9)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="plot/drift_error_validation/ood_value_accuracy_raw.npz",
    )
    parser.add_argument("--out_dir", default="plot/drift_error_validation")
    parser.add_argument("--bootstrap_draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    data = np.load(args.input)
    methods = [method for method in METHODS if f"{method[0]}__error" in data.files]
    n = len(data["occupancy_distance"])
    all_states = np.ones(n, dtype=bool)
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.7, 5.35), constrained_layout=True)
    panel(
        ax, data, methods, all_states, None,
        args.bootstrap_draws, args.seed,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2 if len(methods) > 3 else 3,
        frameon=True,
        fontsize=10.5,
        columnspacing=1.0,
        handlelength=2.0,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "panel_b_value_accuracy.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
