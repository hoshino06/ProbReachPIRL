# -*- coding: utf-8 -*-
"""Validation HJB loss and wall-clock cost summary for the drift example."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from plot_drift_value_contours import backend_supports_show, set_paper_style


@dataclass(frozen=True)
class EventRun:
    label: str
    path: str


@dataclass(frozen=True)
class SpeedGroup:
    key: str
    label: str
    tick_label: str
    event_runs: tuple[EventRun, ...]


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
    ("td3_fixed_10M", r"TD3 fixed $\tau$ 10M", "TD3\n" + r"fixed $\tau$" + "\n10M", TD3_FIXED_10M),
    ("td3_random_10M", r"TD3 random $\tau$ 10M", "TD3\n" + r"random $\tau$" + "\n10M", TD3_RANDOM_10M),
    ("pirl_10M", "PIRL 10M", "PIRL\n10M", PIRL_5M_PRETRAIN + PIRL_5M_TO_10M),
    ("pirl_15M", "PIRL 15M", "PIRL\n15M", PIRL_5M_PRETRAIN + PIRL_5M_TO_10M + PIRL_10M_TO_15M),
]

SPEED_GROUPS = [
    SpeedGroup(
        "td3_fixed",
        r"TD3 fixed $\tau$",
        r"TD3 fixed $\tau$",
        tuple(TD3_FIXED_10M),
    ),
    SpeedGroup(
        "td3_random",
        r"TD3 random $\tau$",
        r"TD3 random $\tau$",
        tuple(TD3_RANDOM_10M),
    ),
    SpeedGroup(
        "pirl_transition",
        "PIRL transition",
        "PIRL\ntransition",
        tuple(PIRL_5M_TO_10M),
    ),
    SpeedGroup(
        "pirl_refinement",
        "PIRL refinement",
        "PIRL\nrefinement",
        tuple(PIRL_10M_TO_15M),
    ),
]

ERROR_ANALYSIS_METHOD = {
    "td3_fixed_10M": "TD3 fixed tau 10M",
    "td3_random_10M": "TD3 random tau 10M",
    "pirl_10M": "PIRL 10M",
    "pirl_15M": "PIRL mixed 15M",
}

BENCHMARK_METHOD = {
    "td3_fixed_10M": "td3_fixed",
    "td3_random_10M": "td3_random",
    "pirl_10M": "pirl_transition",
    "pirl_15M": "pirl_refinement",
}

COLORS = {
    "td3_fixed_10M": "#5f6368",
    "td3_random_10M": "#8a8d91",
    "pirl_10M": "#1f77b4",
    "pirl_15M": "#0b5cad",
    "td3_fixed": "#5f6368",
    "td3_random": "#8a8d91",
    "pirl_transition": "#2f7ebc",
    "pirl_refinement": "#0b5cad",
}


@lru_cache(maxsize=None)
def event_training_metrics(path: str) -> tuple[float, int, int, int]:
    from tensorboard.backend.event_processing import event_accumulator

    acc = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    wall_times = []
    steps = []
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        wall_times.extend(event.wall_time for event in events)
        steps.extend(event.step for event in events)
    if not wall_times or not steps:
        return 0.0, 0, 0, 0
    elapsed_hours = (max(wall_times) - min(wall_times)) / 3600.0
    min_step = min(steps)
    max_step = max(steps)
    return elapsed_hours, max_step - min_step, min_step, max_step


def summarize_event_runs(event_runs: list[EventRun] | tuple[EventRun, ...]) -> dict[str, float | int]:
    elapsed = 0.0
    logged_updates = 0
    min_steps = []
    max_steps = []
    for run in event_runs:
        run_elapsed, run_updates, min_step, max_step = event_training_metrics(run.path)
        elapsed += run_elapsed
        logged_updates += run_updates
        if run_updates > 0:
            min_steps.append(min_step)
            max_steps.append(max_step)

    update_million = logged_updates / 1_000_000.0
    updates_per_hour = update_million / elapsed if elapsed > 0.0 else 0.0
    hours_per_million = elapsed / update_million if update_million > 0.0 else 0.0
    return {
        "wall_clock_hours": elapsed,
        "logged_updates": logged_updates,
        "logged_updates_million": update_million,
        "updates_per_hour_million": updates_per_hour,
        "hours_per_million_updates": hours_per_million,
        "sec_per_1k_updates": hours_per_million * 3.6,
        "min_step": min(min_steps) if min_steps else 0,
        "max_step": max(max_steps) if max_steps else 0,
    }


def read_validation_losses(path: Path) -> dict[str, dict[str, float]]:
    with open(path, newline="") as f:
        rows = {
            row["method"]: {
                "hjb_loss": float(row["hjb_residual_mean"]),
                "bdr_loss": float(row["boundary_value_mae"]),
            }
            for row in csv.DictReader(f)
        }
    return rows


def read_compute_benchmark(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["method"]] = {
                "updates_per_hour_million": float(row["median_million_updates_per_hour"]),
                "hours_per_million_updates": float(row["median_hours_per_million_updates"]),
                "sec_per_1k_updates": float(row["median_hours_per_million_updates"]) * 3.6,
            }
    return rows


def build_summary(error_csv: Path, benchmark_csv: Path) -> list[dict[str, float | str]]:
    validation_losses = read_validation_losses(error_csv)
    benchmark_rows = read_compute_benchmark(benchmark_csv)
    rows = []
    for key, label, tick_label, event_runs in METHODS:
        event_summary = summarize_event_runs(event_runs)
        benchmark = benchmark_rows.get(BENCHMARK_METHOD[key])
        if benchmark is not None:
            event_summary = {**event_summary, **benchmark}
        losses = validation_losses[ERROR_ANALYSIS_METHOD[key]]
        rows.append(
            {
                "method": key,
                "plot_label": label,
                "tick_label": tick_label,
                "hjb_loss": losses["hjb_loss"],
                "bdr_loss": losses["bdr_loss"],
                **event_summary,
            }
        )
    return rows


def build_speed_summary(benchmark_csv: Path) -> list[dict[str, float | str]]:
    benchmark_rows = read_compute_benchmark(benchmark_csv)
    rows = []
    for group in SPEED_GROUPS:
        event_summary = summarize_event_runs(group.event_runs)
        benchmark = benchmark_rows.get(group.key)
        if benchmark is not None:
            event_summary = {**event_summary, **benchmark}
        rows.append(
            {
                "method": group.key,
                "plot_label": group.label,
                "tick_label": group.tick_label,
                **event_summary,
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_loss_panel(ax, rows: list[dict[str, float | str]]) -> None:
    x = np.arange(len(rows))
    labels = [str(row["tick_label"]) for row in rows]
    width = 0.34

    hjb = [float(row["hjb_loss"]) for row in rows]
    bdr = [float(row["bdr_loss"]) for row in rows]
    ax.bar(x - width / 2, hjb, color="#4c78a8", width=width, label="HJB")
    ax.bar(x + width / 2, bdr, color="#f28e2b", width=width, label="BDR")
    ax.set_yscale("log")
    ax.set_ylabel("loss", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(frameon=True, fontsize=10, loc="upper right")


def draw_compute_cost_panel(ax, rows: list[dict[str, float | str]]) -> None:
    x = np.arange(len(rows))
    labels = [str(row["tick_label"]) for row in rows]
    colors = [COLORS[str(row["method"])] for row in rows]
    hours_per_million = [float(row["hours_per_million_updates"]) for row in rows]
    bars = ax.bar(x, hours_per_million, color=colors, width=0.62)
    ax.set_ylabel("cost [h / M updates]", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0.0, max(hours_per_million) * 1.16)
    for bar, value in zip(bars, hours_per_million):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_summary(rows: list[dict[str, float | str]], out_dir: Path, save_pdf: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), constrained_layout=True)
    draw_loss_panel(axes[0], rows)
    draw_compute_cost_panel(axes[1], rows)

    for ext in ["png"] + (["pdf"] if save_pdf else []):
        fig.savefig(out_dir / f"validation_hjb_vs_compute_cost.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06)
    if not backend_supports_show():
        plt.close(fig)

    panel_specs = [
        ("loss_panel", draw_loss_panel),
        ("compute_cost_panel", draw_compute_cost_panel),
    ]
    for stem, draw in panel_specs:
        panel_fig, panel_ax = plt.subplots(figsize=(4.3, 4.3), constrained_layout=True)
        draw(panel_ax, rows)
        for ext in ["png"] + (["pdf"] if save_pdf else []):
            panel_fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06)
        if not backend_supports_show():
            plt.close(panel_fig)


def plot_speed_summary(rows: list[dict[str, float | str]], out_dir: Path, save_pdf: bool) -> None:
    x = np.arange(len(rows))
    labels = [str(row["tick_label"]) for row in rows]
    colors = [COLORS[str(row["method"])] for row in rows]
    speeds = [float(row["updates_per_hour_million"]) for row in rows]
    costs = [float(row["hours_per_million_updates"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(x, speeds, color=colors, width=0.62)
    ax.set_ylabel("speed [M updates / h]", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0.0, max(speeds) * 1.18)
    for bar, value in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    bars = ax.bar(x, costs, color=colors, width=0.62)
    ax.set_ylabel("cost [h / M updates]", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0.0, max(costs) * 1.18)
    for bar, value in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    for ext in ["png"] + (["pdf"] if save_pdf else []):
        fig.savefig(out_dir / f"compute_speed_summary.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06)
    if not backend_supports_show():
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="plot/drift_hjb_runtime_summary")
    parser.add_argument("--error_csv", default="plot/drift_error_analysis/pirl_safety_story_summary.csv")
    parser.add_argument(
        "--benchmark_csv",
        default="plot/drift_hjb_runtime_summary/compute_speed_benchmark_5k_r2.csv",
        help="Independent compute-speed benchmark CSV. Falls back to TensorBoard wall time if missing.",
    )
    parser.add_argument("--plot_speed_summary", action="store_true")
    parser.add_argument("--save_pdf", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    rows = build_summary(Path(args.error_csv), Path(args.benchmark_csv))
    speed_rows = build_speed_summary(Path(args.benchmark_csv))
    write_summary(out_dir / "validation_hjb_runtime_summary.csv", rows)
    write_summary(out_dir / "compute_speed_summary.csv", speed_rows)
    plot_summary(rows, out_dir, args.save_pdf)
    if args.plot_speed_summary:
        plot_speed_summary(speed_rows, out_dir, args.save_pdf)

    print("--------------------------------------------")
    print(f"out_dir: {out_dir}")
    for row in rows:
        print(
            f"{row['method']}: HJB={row['hjb_loss']:.3e}, "
            f"BDR={row['bdr_loss']:.3e}, "
            f"wall-clock={row['wall_clock_hours']:.2f} h, "
            f"compute cost={row['hours_per_million_updates']:.2f} h/M updates"
        )
    print("compute speed groups:")
    for row in speed_rows:
        print(
            f"{row['method']}: {row['updates_per_hour_million']:.3f} M updates/h, "
            f"{row['hours_per_million_updates']:.2f} h/M updates, "
            f"wall-clock={row['wall_clock_hours']:.2f} h"
        )
    print("--------------------------------------------")

    if backend_supports_show():
        plt.show()


if __name__ == "__main__":
    main()
