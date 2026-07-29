# -*- coding: utf-8 -*-
"""Paper-style summary of conservative reachability values in drift.

The figure separates three claims:

1. Calibration: PIRL values are conservative relative to Monte Carlo reachability.
2. Asymmetric error: PIRL reduces dangerous safety overestimation, even if it
   increases conservative underestimation.
3. Loss alignment: the checkpoint with low HJB/BDR residuals also has the
   smallest safety-overestimation metric.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_drift_value_contours import set_paper_style


CASES = [
    ("TD3 fixed tau 10M", r"TD3 fixed $\tau$ 10M", "td3_T01_10M.npz"),
    ("TD3 random tau 10M", r"TD3 random $\tau$ 10M", "td3_randT_10M.npz"),
    ("PIRL 10M", "PIRL 10M", "pirl_hold0015_R3_10M.npz"),
    ("PIRL mixed 15M", "PIRL 15M", "pirl_mixed_15M.npz"),
]

LOSS_NAME = {
    "TD3 fixed tau 10M": "TD3 fixed tau 10M",
    "TD3 random tau 10M": "TD3 random tau 10M",
    "PIRL 10M": "PIRL replay 10M",
    "PIRL mixed 15M": "PIRL mixed 15M",
}


def load_mc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    value = np.clip(np.asarray(data["V"], dtype=np.float64).reshape(-1), 0.0, 1.0)
    prob = np.asarray(data["P"], dtype=np.float64).reshape(-1)
    return value, prob


def binned_curve(value: np.ndarray, prob: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (lo <= value) & (value < hi)
        if hi == edges[-1]:
            mask = (lo <= value) & (value <= hi)
        if not np.any(mask):
            continue
        x.append(float(value[mask].mean()))
        y.append(float(prob[mask].mean()))
        n.append(int(mask.sum()))
    return np.asarray(x), np.asarray(y), np.asarray(n)


def read_loss_summary(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["method"]] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"method", "plot_label", "checkpoint"}
            }
    return rows


def read_boundary_summary(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["method"]] = {
                key: float(value)
                for key, value in row.items()
                if key != "method"
            }
    return rows


def read_cached_summary(path: Path) -> list[dict[str, float | str]]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | str] = {"method": row["method"]}
            for key, value in row.items():
                if key == "method":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def read_cached_bins(path: Path) -> dict[str, dict[str, np.ndarray | str]]:
    grouped: dict[str, dict[str, list[float] | str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            if method not in grouped:
                grouped[method] = {"label": row["plot_label"], "V": [], "P": []}
            grouped[method]["V"].append(float(row["mean_v"]))  # type: ignore[union-attr]
            grouped[method]["P"].append(float(row["mean_reachability"]))  # type: ignore[union-attr]
    return {
        method: {
            "label": values["label"],
            "V": np.asarray(values["V"], dtype=np.float64),
            "P": np.asarray(values["P"], dtype=np.float64),
        }
        for method, values in grouped.items()
    }


def write_cached_bins(path: Path, series: dict[str, dict[str, np.ndarray | str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "plot_label", "bin_index", "mean_v", "mean_reachability"],
        )
        writer.writeheader()
        for method, values in series.items():
            value = np.asarray(values["V"], dtype=np.float64)
            prob = np.asarray(values["P"], dtype=np.float64)
            for i, (v, p) in enumerate(zip(value, prob)):
                writer.writerow(
                    {
                        "method": method,
                        "plot_label": values["label"],
                        "bin_index": i,
                        "mean_v": v,
                        "mean_reachability": p,
                    }
                )


def short_label(label: str) -> str:
    return (
        label.replace("TD3 fixed tau 10M", "TD3 fixed\n$\\tau$ 10M")
        .replace("TD3 random tau 10M", "TD3 random\n$\\tau$ 10M")
        .replace("PIRL mixed 15M", "PIRL\n15M")
        .replace("PIRL 10M", "PIRL\n10M")
    )


def apply_panel_font_sizes(ax, label_size: int = 22, tick_size: int = 14, legend_size: int = 12) -> None:
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="plot/drift_mc_reachability/full_state_compare_20260729_mc256")
    parser.add_argument("--loss_csv", default="plot/drift_hjb_safety_link_mc256/hjb_safety_link_summary.csv")
    parser.add_argument("--boundary_csv", default="plot/drift_pirl_effect_audit_mc256/boundary_summary.csv")
    parser.add_argument("--out_dir", default="plot/drift_error_analysis")
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument(
        "--panels",
        choices=["two", "three", "boundary"],
        default="boundary",
        help="Use two paper panels or keep the original three-panel diagnostic.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "pirl_safety_story_summary.csv"
    bins_csv_path = out_dir / "value_vs_reachability_bins.csv"
    data_dir = Path(args.data_dir)
    raw_paths = [data_dir / filename for _, _, filename in CASES]
    raw_inputs_available = (
        all(path.exists() for path in raw_paths)
        and Path(args.loss_csv).exists()
        and Path(args.boundary_csv).exists()
    )

    edges = np.linspace(0.0, 1.0, args.num_bins + 1)
    if raw_inputs_available:
        source_desc = "raw MC/loss inputs"
        loss_rows = read_loss_summary(Path(args.loss_csv))
        boundary_rows = read_boundary_summary(Path(args.boundary_csv))
        rows = []
        series = {}
        for key, plot_label, filename in CASES:
            value, prob = load_mc(data_dir / filename)
            diff = prob - value
            over = np.maximum(value - prob, 0.0)
            under = np.maximum(prob - value, 0.0)
            xb, yb, _ = binned_curve(value, prob, edges)
            loss = loss_rows[LOSS_NAME[key]]
            row = {
                "method": key,
                "n": len(value),
                "mean_mc": float(prob.mean()),
                "mean_v": float(value.mean()),
                "mean_abs_error": float(np.abs(diff).mean()),
                "conservative_underestimate": float(under.mean()),
                "safety_overestimate": float(over.mean()),
                "bias_mc_minus_v": float(diff.mean()),
                "hjb_residual_mean": loss["hjb_residual_mean"],
                "boundary_value_mae": loss["boundary_value_mae"],
                "target_value_mae": boundary_rows[key]["target_value_mae"],
                "avoid_value_mae": boundary_rows[key]["avoid_value_mae"],
            }
            rows.append(row)
            series[key] = {"label": plot_label, "V": xb, "P": yb}

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        write_cached_bins(bins_csv_path, series)
    else:
        source_desc = "cached CSVs"
        if not csv_path.exists() or not bins_csv_path.exists():
            raise FileNotFoundError(
                "Raw inputs are unavailable, and cached summary/bin CSVs are missing in "
                f"{out_dir}. Re-run once before deleting raw analysis artifacts."
            )
        rows = read_cached_summary(csv_path)
        series = read_cached_bins(bins_csv_path)

    set_paper_style()
    colors = {
        "TD3 fixed tau 10M": "#5f6368",
        "TD3 random tau 10M": "#8a8d91",
        "PIRL 10M": "#1f77b4",
        "PIRL mixed 15M": "#0b5cad",
    }
    markers = {
        "TD3 fixed tau 10M": "o",
        "TD3 random tau 10M": "s",
        "PIRL 10M": "^",
        "PIRL mixed 15M": "D",
    }
    def draw_value_vs_reachability(ax, label_size: int = 22, tick_size: int = 14, legend_size: int = 12):
        ax.fill_between(
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            color="#f6d7c8",
            alpha=0.7,
            zorder=0,
            label="risk underestimation",
        )
        ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.0, linestyle="--")
        for key, plot_label, _ in CASES:
            ax.plot(
                series[key]["V"],
                series[key]["P"],
                marker=markers[key],
                color=colors[key],
                linewidth=1.6,
                markersize=4.8,
                label=plot_label,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"learned value $V$")
        ax.set_ylabel("empirical reachability")
        ax.legend(frameon=True, fontsize=8, loc="lower right")
        apply_panel_font_sizes(ax, label_size=label_size, tick_size=tick_size, legend_size=legend_size)

    def draw_boundary_safety_errors(ax, label_size: int = 22, tick_size: int = 14, legend_size: int = 12):
        loss_width = 0.30
        ax.bar(
            x - loss_width / 2,
            [row["boundary_value_mae"] for row in rows],
            width=loss_width,
            color="#4c78a8",
            label="Boundary MAE",
        )
        ax.bar(
            x + loss_width / 2,
            [row["safety_overestimate"] for row in rows],
            width=loss_width,
            color="#d95f02",
            label="Risk underest.",
        )
        ax.set_ylabel("mean error")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(row["method"]) for row in rows], fontsize=9)
        ax.legend(frameon=True, fontsize=8, loc="upper right")
        apply_panel_font_sizes(ax, label_size=label_size, tick_size=tick_size, legend_size=legend_size)

    if args.panels in {"two", "boundary"}:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
        loss_ax = axes[1]
        output_stem = "pirl_safety_story_boundary_panel" if args.panels == "boundary" else "pirl_safety_story_two_panel"
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9), constrained_layout=True)
        loss_ax = axes[2]
        output_stem = "pirl_safety_story_summary"

    combined_label_size = 18 if args.panels in {"two", "boundary"} else 22
    combined_tick_size = 12 if args.panels in {"two", "boundary"} else 14
    combined_legend_size = 10 if args.panels in {"two", "boundary"} else 12
    draw_value_vs_reachability(
        axes[0],
        label_size=combined_label_size,
        tick_size=combined_tick_size,
        legend_size=combined_legend_size,
    )

    x = np.arange(len(rows))
    if args.panels == "three":
        width = 0.36
        ax = axes[1]
        ax.bar(
            x - width / 2,
            [row["conservative_underestimate"] for row in rows],
            width=width,
            color="#4c78a8",
            label=r"conservative: $\max(\mathrm{reachability}-V,0)$",
        )
        ax.bar(
            x + width / 2,
            [row["safety_overestimate"] for row in rows],
            width=width,
            color="#d95f02",
            label=r"risk underest.: $\max(V-\mathrm{reachability},0)$",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(row["method"]) for row in rows], fontsize=9)
        ax.set_ylabel("mean asymmetric error")
        ax.legend(frameon=True, fontsize=8, loc="upper left")
        ax.grid(True, axis="y", alpha=0.25)

    ax = loss_ax
    loss_width = 0.24
    if args.panels == "boundary":
        draw_boundary_safety_errors(
            ax,
            label_size=combined_label_size,
            tick_size=combined_tick_size,
            legend_size=combined_legend_size,
        )
    else:
        ax.bar(
            x - loss_width,
            [row["hjb_residual_mean"] for row in rows],
            width=loss_width,
            color="#4c78a8",
            label="HJB residual",
        )
        ax.bar(
            x,
            [row["boundary_value_mae"] for row in rows],
            width=loss_width,
            color="#f28e2b",
            label="Boundary MAE",
        )
        ax.bar(
            x + loss_width,
            [row["safety_overestimate"] for row in rows],
            width=loss_width,
            color="#d95f02",
            label="Risk underest.",
        )
        ax.set_yscale("log")
        ax.set_ylabel("metric value")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(row["method"]) for row in rows], fontsize=9)
        ax.legend(frameon=True, fontsize=8, loc="upper right")

    for ext in ["png", "pdf"]:
        path = out_dir / f"{output_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        print(f"Saved: {path}")

    if args.panels == "boundary":
        panel_specs = [
            ("value_vs_reachability", draw_value_vs_reachability),
            ("boundary_safety_errors", draw_boundary_safety_errors),
        ]
        for stem, draw in panel_specs:
            panel_fig, panel_ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
            draw(panel_ax)
            for ext in ["png", "pdf"]:
                path = out_dir / f"{stem}.{ext}"
                panel_fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
                print(f"Saved: {path}")
            plt.close(panel_fig)

    print("--------------------------------------------")
    print(f"source:   {source_desc}")
    if raw_inputs_available:
        print(f"data_dir: {data_dir}")
        print(f"loss_csv: {args.loss_csv}")
        print(f"boundary_csv: {args.boundary_csv}")
    print(f"csv:      {csv_path}")
    print(f"bins_csv: {bins_csv_path}")
    for row in rows:
        print(
            f"{row['method']}: under={row['conservative_underestimate']:.4f}, "
            f"over={row['safety_overestimate']:.4f}, "
            f"HJB={row['hjb_residual_mean']:.3e}, BDR={row['boundary_value_mae']:.4f}, "
            f"target={row['target_value_mae']:.4f}, avoid={row['avoid_value_mae']:.4f}"
        )
    print("--------------------------------------------")

    if "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
