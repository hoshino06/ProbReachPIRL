# -*- coding: utf-8 -*-
"""Validation HJB loss and wall-clock cost summary for the drift example."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_drift_value_contours import backend_supports_show, set_paper_style


@dataclass(frozen=True)
class EventRun:
    label: str
    path: str


TD3_FIXED_10M = [
    EventRun("0-1M", "logs/drift/td3_T01/up01M_scale04_mix442/events.out.tfevents.1781570534.poincare.1866042.0"),
    EventRun("1-3M", "logs/drift/td3_T01/up03M_scale10_mix334/events.out.tfevents.1781589764.poincare.1898936.0"),
    EventRun("3-5M", "logs/drift/td3_T01/up05M_scale10_mix334/events.out.tfevents.1781622292.poincare.1943215.0"),
    EventRun("5-10M", "logs/drift/td3_T01/up10M_scale10_mix334/events.out.tfevents.1781906014.maxwell.1709051.0"),
]

TD3_RANDOM_10M = [
    EventRun("0-2M", "logs/drift/td3_T01_randT/up02M_scale08_mix334_randT/events.out.tfevents.1781755445.maxwell.3586949.0"),
    EventRun("2-3M", "logs/drift/td3_T01_randT/up03M_scale10_mix334_randT/events.out.tfevents.1781857436.maxwell.2.2"),
    EventRun("3-10M", "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/events.out.tfevents.1781861480.maxwell.3672424.0"),
]

PIRL_5M_PRETRAIN = TD3_FIXED_10M[:3]

PIRL_5M_TO_10M = [
    EventRun(
        "5-6M",
        "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/"
        "round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/"
        "events.out.tfevents.1782461410.poincare.3072105.0",
    ),
    EventRun(
        "6-7M",
        "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/"
        "round_001/cont0001to0002/train/0627_0211_cont0001to0002_seed_1/"
        "events.out.tfevents.1782493891.poincare.3124546.0",
    ),
    EventRun(
        "7-8M",
        "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/"
        "round_000/hold0002_R1/train/0630_0945_hold0002_R1_seed_1/"
        "events.out.tfevents.1782780320.poincare.3589938.0",
    ),
    EventRun(
        "8-9M",
        "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/"
        "round_001/back002to0015_R2/train/0630_1958_back002to0015_R2_seed_1/"
        "events.out.tfevents.1782817110.poincare.3628251.0",
    ),
    EventRun(
        "9-10M",
        "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/"
        "round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/"
        "events.out.tfevents.1782853569.poincare.3665275.0",
    ),
]

PIRL_10M_TO_15M = [
    EventRun(
        "10-11M",
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/"
        "round_000/mix90_expand003_R1/train/0706_2022_mix90_expand003_R1_seed_1/"
        "events.out.tfevents.1783369376.ip-172-31-37-159.238308.0",
    ),
    EventRun(
        "11-12M",
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/"
        "round_001/mix85_expand003_R2/train/0707_0901_mix85_expand003_R2_seed_1/"
        "events.out.tfevents.1783414914.ip-172-31-37-159.248701.0",
    ),
    EventRun(
        "12-13M",
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/"
        "round_002/mix80_expand003_R3/train/0707_2126_mix80_expand003_R3_seed_1/"
        "events.out.tfevents.1783459614.ip-172-31-37-159.255619.0",
    ),
    EventRun(
        "13-14M",
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/"
        "round_003/mix80_expand0035_R4/train/0708_0956_mix80_expand0035_R4_seed_1/"
        "events.out.tfevents.1783504611.ip-172-31-37-159.270766.0",
    ),
    EventRun(
        "14-15M",
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/"
        "round_004/mix85_back003_R5/train/0708_2216_mix85_back003_R5_seed_1/"
        "events.out.tfevents.1783548969.ip-172-31-37-159.275260.0",
    ),
]

METHODS = [
    ("td3_fixed_10M", r"TD3 fixed $\tau$ 10M", r"TD3 fixed $\tau$" + "\n10M", TD3_FIXED_10M),
    ("td3_random_10M", r"TD3 random $\tau$ 10M", r"TD3 random $\tau$" + "\n10M", TD3_RANDOM_10M),
    ("pirl_10M", "PIRL 10M", "PIRL\n10M", PIRL_5M_PRETRAIN + PIRL_5M_TO_10M),
    ("pirl_15M", "PIRL 15M", "PIRL\n15M", PIRL_5M_PRETRAIN + PIRL_5M_TO_10M + PIRL_10M_TO_15M),
]

ERROR_ANALYSIS_METHOD = {
    "td3_fixed_10M": "TD3 fixed tau 10M",
    "td3_random_10M": "TD3 random tau 10M",
    "pirl_10M": "PIRL 10M",
    "pirl_15M": "PIRL mixed 15M",
}

COLORS = {
    "td3_fixed_10M": "#5f6368",
    "td3_random_10M": "#8a8d91",
    "pirl_10M": "#1f77b4",
    "pirl_15M": "#0b5cad",
}


def event_elapsed_hours(path: str) -> float:
    from tensorboard.backend.event_processing import event_accumulator

    acc = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    wall_times = []
    for tag in acc.Tags().get("scalars", []):
        wall_times.extend(event.wall_time for event in acc.Scalars(tag))
    if not wall_times:
        return 0.0
    return (max(wall_times) - min(wall_times)) / 3600.0


def read_validation_hjb_loss(path: Path) -> dict[str, float]:
    with open(path, newline="") as f:
        rows = {row["method"]: float(row["hjb_residual_mean"]) for row in csv.DictReader(f)}
    return rows


def build_summary(error_csv: Path) -> list[dict[str, float | str]]:
    validation_loss = read_validation_hjb_loss(error_csv)
    rows = []
    for key, label, tick_label, event_runs in METHODS:
        elapsed = sum(event_elapsed_hours(run.path) for run in event_runs)
        rows.append(
            {
                "method": key,
                "plot_label": label,
                "tick_label": tick_label,
                "validation_hjb_loss": validation_loss[ERROR_ANALYSIS_METHOD[key]],
                "wall_clock_hours": elapsed,
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, float | str]], out_dir: Path, save_pdf: bool) -> None:
    x = np.arange(len(rows))
    labels = [str(row["tick_label"]) for row in rows]
    colors = [COLORS[str(row["method"])] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)

    ax = axes[0]
    hjb = [float(row["validation_hjb_loss"]) for row in rows]
    ax.bar(x, hjb, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("validation HJB loss")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", which="both", alpha=0.25)

    ax = axes[1]
    hours = [float(row["wall_clock_hours"]) for row in rows]
    bars = ax.bar(x, hours, color=colors, width=0.62)
    ax.set_ylabel("wall-clock time [h]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0.0, max(hours) * 1.16)
    for bar, value in zip(bars, hours):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for ext in ["png"] + (["pdf"] if save_pdf else []):
        fig.savefig(out_dir / f"validation_hjb_vs_runtime.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06)
    if not backend_supports_show():
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="plot/drift_hjb_runtime_summary")
    parser.add_argument("--error_csv", default="plot/drift_error_analysis/pirl_safety_story_summary.csv")
    parser.add_argument("--save_pdf", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    rows = build_summary(Path(args.error_csv))
    write_summary(out_dir / "validation_hjb_runtime_summary.csv", rows)
    plot_summary(rows, out_dir, args.save_pdf)

    print("--------------------------------------------")
    print(f"out_dir: {out_dir}")
    for row in rows:
        print(
            f"{row['method']}: validation HJB={row['validation_hjb_loss']:.3e}, "
            f"wall-clock={row['wall_clock_hours']:.2f} h"
        )
    print("--------------------------------------------")

    if backend_supports_show():
        plt.show()


if __name__ == "__main__":
    main()
