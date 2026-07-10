# -*- coding: utf-8 -*-
"""Training-process plots for the drift PIRL study.

The PIRL curve follows the exact lineage that leads to the selected mixedHJB
15M checkpoint:

    fixedT TD3 5M
      -> replayHJB ramp0to0001 6M
      -> replayHJB cont0001to0002 7M
      -> replayHJB restart2 hold0002_R1 8M
      -> replayHJB restart2 back002to0015_R2 9M
      -> replayHJB restart2 hold0015_R3 10M
      -> mixedHJB mix90_expand003_R1 11M
      -> mixedHJB mix85_expand003_R2 12M
      -> mixedHJB mix80_expand003_R3 13M
      -> mixedHJB mix80_expand0035_R4 14M
      -> mixedHJB mix85_back003_R5 15M

TD3 is shown from the curated random-horizon baseline for comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUT_DIR = "plot/drift_training_process"


@dataclass(frozen=True)
class EventRun:
    label: str
    path: str


PIRL_LINEAGE = [
    EventRun(
        "replayHJB ramp 0->1e-3",
        "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/"
        "round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/"
        "events.out.tfevents.1782461410.poincare.3072105.0",
    ),
    EventRun(
        "replayHJB 1e-3->2e-3",
        "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/"
        "round_001/cont0001to0002/train/0627_0211_cont0001to0002_seed_1/"
        "events.out.tfevents.1782493891.poincare.3124546.0",
    ),
    EventRun(
        "replayHJB hold 2e-3",
        "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/"
        "round_000/hold0002_R1/train/0630_0945_hold0002_R1_seed_1/"
        "events.out.tfevents.1782780320.poincare.3589938.0",
    ),
    EventRun(
        "replayHJB 2e-3->1.5e-3",
        "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/"
        "round_001/back002to0015_R2/train/0630_1958_back002to0015_R2_seed_1/"
        "events.out.tfevents.1782817110.poincare.3628251.0",
    ),
    EventRun(
        "replayHJB hold 1.5e-3",
        "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/"
        "round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/"
        "events.out.tfevents.1782853569.poincare.3665275.0",
    ),
    EventRun(
        "mixedHJB 90% replay",
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/"
        "round_000/mix90_expand003_R1/train/0706_2022_mix90_expand003_R1_seed_1/"
        "events.out.tfevents.1783369376.ip-172-31-37-159.238308.0",
    ),
    EventRun(
        "mixedHJB 85% replay",
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/"
        "round_001/mix85_expand003_R2/train/0707_0901_mix85_expand003_R2_seed_1/"
        "events.out.tfevents.1783414914.ip-172-31-37-159.248701.0",
    ),
    EventRun(
        "mixedHJB 80% replay",
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/"
        "round_002/mix80_expand003_R3/train/0707_2126_mix80_expand003_R3_seed_1/"
        "events.out.tfevents.1783459614.ip-172-31-37-159.255619.0",
    ),
    EventRun(
        "mixedHJB 80% replay, expanded",
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/"
        "round_003/mix80_expand0035_R4/train/0708_0956_mix80_expand0035_R4_seed_1/"
        "events.out.tfevents.1783504611.ip-172-31-37-159.270766.0",
    ),
    EventRun(
        "mixedHJB 85% replay, backoff",
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/"
        "round_004/mix85_back003_R5/train/0708_2216_mix85_back003_R5_seed_1/"
        "events.out.tfevents.1783548969.ip-172-31-37-159.275260.0",
    ),
]


TD3_BASELINE = [
    EventRun(
        "TD3 randT early 2M",
        "logs/drift/td3_T01_randT/up02M_scale08_mix334_randT/"
        "events.out.tfevents.1781755445.maxwell.3586949.0",
    ),
    EventRun(
        "TD3 randT early 3M",
        "logs/drift/td3_T01_randT/up03M_scale10_mix334_randT/"
        "events.out.tfevents.1781857436.maxwell.2.2",
    ),
    EventRun(
        "TD3 randT baseline",
        "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/"
        "events.out.tfevents.1781861480.maxwell.3672424.0",
    ),
]

TD3_FIXED_BASELINE = [
    EventRun(
        "TD3 fixedT early 1M",
        "logs/drift/td3_T01/up01M_scale04_mix442/"
        "events.out.tfevents.1781570534.poincare.1866042.0",
    ),
    EventRun(
        "TD3 fixedT early 3M",
        "logs/drift/td3_T01/up03M_scale10_mix334/"
        "events.out.tfevents.1781589764.poincare.1898936.0",
    ),
    EventRun(
        "TD3 fixedT pretraining",
        "logs/drift/td3_T01/up05M_scale10_mix334/"
        "events.out.tfevents.1781622292.poincare.1943215.0",
    ),
    EventRun(
        "TD3 fixedT continuation",
        "logs/drift/td3_T01/up10M_scale10_mix334/"
        "events.out.tfevents.1781906014.maxwell.1709051.0",
    ),
]

PIRL_PRETRAIN = TD3_FIXED_BASELINE[:3]


PANEL_SPECS = [
    ("Loss/HJB", "Uniform HJB loss", True),
    ("Loss/BDR", "Boundary loss", True),
    ("Loss/RL", "TD3 loss", True),
    ("RL/Average Reward", "Average reward", False),
]

LOSS_PANEL_SPECS = PANEL_SPECS[:3]
REWARD_PANEL_SPEC = PANEL_SPECS[3]


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "mathtext.fontset": "cm",
        "figure.dpi": 140,
    })


def load_scalar(path: str, tag: str) -> np.ndarray:
    from tensorboard.backend.event_processing import event_accumulator

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    acc = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return np.empty((0, 2), dtype=np.float64)
    rows = [(event.step / 1.0e6, event.value) for event in acc.Scalars(tag)]
    return np.asarray(rows, dtype=np.float64)


def merge_series(runs: Iterable[EventRun], tag: str, xmin: float, xmax: float) -> np.ndarray:
    pieces = []
    for run in runs:
        data = load_scalar(run.path, tag)
        if len(data) == 0:
            continue
        mask = (data[:, 0] >= xmin) & (data[:, 0] <= xmax)
        data = data[mask]
        if len(data):
            pieces.append(data)
    if not pieces:
        return np.empty((0, 2), dtype=np.float64)

    data = np.concatenate(pieces, axis=0)
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]

    # Keep the last value if multiple event files log the same update.
    _, last_indices = np.unique(data[:, 0], return_index=True)
    if len(last_indices) != len(data):
        _, rev_indices = np.unique(data[::-1, 0], return_index=True)
        keep = len(data) - 1 - rev_indices
        data = data[np.sort(keep)]
    return data


def smooth_series(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(data) < window:
        return data
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    y = np.pad(data[:, 1], (pad, pad), mode="edge")
    smoothed = np.convolve(y, kernel, mode="valid")
    return np.column_stack([data[:, 0], smoothed])


def smooth_series_with_breaks(data: np.ndarray, window: int, breaks: Iterable[float]) -> np.ndarray:
    """Smooth without leaking values across checkpoint/phase boundaries."""
    if len(data) == 0:
        return data

    pieces = []
    start = float(data[0, 0])
    bounds = [b for b in breaks if data[0, 0] < b < data[-1, 0]]
    for stop in bounds + [float(data[-1, 0]) + 1.0e-9]:
        mask = (data[:, 0] >= start) & (data[:, 0] < stop)
        piece = data[mask]
        if len(piece):
            pieces.append(smooth_series(piece, window))
        start = stop
    if not pieces:
        return data
    return np.concatenate(pieces, axis=0)


def resample_series(data: np.ndarray, step: float, xmin: float, xmax: float) -> np.ndarray:
    if step <= 0 or len(data) == 0:
        return data

    start = np.ceil(xmin / step) * step
    edges = np.arange(start, xmax + step, step, dtype=np.float64)
    if len(edges) < 2:
        return data

    x = data[:, 0]
    y = data[:, 1]
    sampled = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == edges[-1]:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        if np.any(mask):
            center = (left + right) * 0.5
            local_idx = np.flatnonzero(mask)
            idx = local_idx[np.argmin(np.abs(x[local_idx] - center))]
            sampled.append((center, float(y[idx])))
    if not sampled:
        return data
    return np.asarray(sampled, dtype=np.float64)


def prepare_curve(data: np.ndarray, args, breaks: Iterable[float] = ()) -> np.ndarray:
    if len(data) == 0:
        return data

    pieces = []
    bounds = [args.xmin] + [b for b in breaks if args.xmin < b < args.xmax] + [args.xmax]
    for left, right in zip(bounds[:-1], bounds[1:]):
        mask = (data[:, 0] >= left) & (data[:, 0] <= right)
        piece = data[mask]
        if len(piece) == 0:
            continue
        piece = resample_series(piece, args.plot_step_million, left, right)
        piece = smooth_series(piece, args.smooth_window)
        pieces.append(piece)

    if not pieces:
        return data
    return downsample(np.concatenate(pieces, axis=0), args.max_points)


def downsample(data: np.ndarray, max_points: int) -> np.ndarray:
    if len(data) <= max_points:
        return data
    idx = np.linspace(0, len(data) - 1, max_points).astype(int)
    return data[idx]


def save_csv(path: str, curves: dict[tuple[str, str], np.ndarray]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["curve", "tag", "updates_million", "value"])
        for (curve, tag), data in curves.items():
            for step_m, value in data:
                writer.writerow([curve, tag, f"{step_m:.7f}", f"{value:.9g}"])


def load_csv(path: str) -> dict[tuple[str, str], np.ndarray]:
    curves: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["curve"], row["tag"])
            curves.setdefault(key, []).append((float(row["updates_million"]), float(row["value"])))
    return {key: np.asarray(rows, dtype=np.float64) for key, rows in curves.items()}


def add_phase_marks(ax) -> None:
    spans = [
        (0.0, 5.0, "TD3 pretraining", "#f2f2f2"),
        (5.0, 10.0, "PIRL transition", "#eaf3ff"),
        (10.0, 15.0, "PIRL refinement", "#fff2dd"),
    ]
    for x0, x1, _, color in spans:
        ax.axvspan(x0, x1, color=color, alpha=0.35, linewidth=0, zorder=-2)
    for x, label in [(5.0, "PIRL starts"), (10.0, "mixed starts")]:
        ax.axvline(x, color="0.50", linewidth=1.0, linestyle="-", zorder=0)
    for x, label in [
        (6.0, "6M"),
        (7.0, "7M"),
        (8.0, "8M"),
        (9.0, "9M"),
        (11.0, "11M"),
        (12.0, "12M"),
        (13.0, "13M"),
        (14.0, "14M"),
    ]:
        ax.axvline(x, color="0.72", linewidth=0.8, linestyle=":", zorder=0)


def add_phase_labels(ax) -> None:
    labels = [
        (2.5, "TD3 pretraining"),
        (7.5, "PIRL transition"),
        (12.5, "PIRL refinement"),
    ]
    for x, label in labels:
        ax.text(x, 1.04, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color="0.20", fontsize=10, clip_on=False)


def build_curves(args) -> dict[tuple[str, str], np.ndarray]:
    curves: dict[tuple[str, str], np.ndarray] = {}
    for tag, _, _ in PANEL_SPECS:
        pretrain_raw = merge_series(PIRL_PRETRAIN, tag, args.xmin, min(args.xmax, 5.0))
        pirl_raw = merge_series(PIRL_LINEAGE, tag, args.xmin, args.xmax)
        pirl_raw = np.concatenate([pretrain_raw, pirl_raw], axis=0) if len(pretrain_raw) else pirl_raw
        pirl_raw = pirl_raw[np.argsort(pirl_raw[:, 0], kind="stable")] if len(pirl_raw) else pirl_raw
        fixed_td3_raw = merge_series(TD3_FIXED_BASELINE, tag, args.xmin, min(args.xmax, 10.0))
        td3_raw = merge_series(TD3_BASELINE, tag, args.xmin, min(args.xmax, 10.0))

        pirl = prepare_curve(pirl_raw, args, breaks=[5.0, 10.0])
        fixed_td3 = prepare_curve(fixed_td3_raw, args)
        td3 = prepare_curve(td3_raw, args)
        curves[("PIRL lineage to mixedHJB 15M", tag)] = pirl
        curves[("TD3 fixedT baseline", tag)] = fixed_td3
        curves[("TD3 randT baseline", tag)] = td3
    return curves


def plot_panel(ax, curves: dict[tuple[str, str], np.ndarray], tag: str, ylabel: str,
               logy: bool, args, show_legend: bool = False,
               show_phase_labels: bool = False) -> None:
    pirl = curves.get(("PIRL lineage to mixedHJB 15M", tag), np.empty((0, 2)))
    fixed_td3 = curves.get(("TD3 fixedT baseline", tag), np.empty((0, 2)))
    td3 = curves.get(("TD3 randT baseline", tag), np.empty((0, 2)))

    if len(fixed_td3):
        ax.plot(fixed_td3[:, 0], fixed_td3[:, 1], color="0.40", linewidth=1.5,
                linestyle=":", label=r"TD3 (fixed $\tau$)")
    if len(td3):
        ax.plot(td3[:, 0], td3[:, 1], color="0.35", linewidth=1.7,
                linestyle="--", label=r"TD3 (random $\tau$)")
    if len(pirl):
        ax.plot(pirl[:, 0], pirl[:, 1], color="#1f77b4", linewidth=2.1,
                label="PIRL")

    add_phase_marks(ax)
    if show_phase_labels:
        add_phase_labels(ax)
    ax.set_ylabel(ylabel)
    if logy:
        ymin = min([float(np.nanmin(d[:, 1])) for d in [pirl, fixed_td3, td3] if len(d)] or [1.0])
        if ymin > 0:
            ax.set_yscale("log")
    ax.set_xlim(args.xmin, args.xmax)
    if show_legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.18), ncol=3,
                  frameon=True, framealpha=0.95, columnspacing=1.2,
                  handlelength=2.0)


def plot_training_process(args) -> dict[tuple[str, str], np.ndarray]:
    set_style()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "fig_drift_training_process_losses.csv")
    if os.path.exists(csv_path) and not args.rebuild_csv:
        curves = load_csv(csv_path)
    else:
        curves = build_curves(args)
        save_csv(csv_path, curves)

    fig_loss, axes = plt.subplots(3, 1, figsize=(5.9, 7.0), sharex=True)
    for i, (ax, (tag, ylabel, logy)) in enumerate(zip(axes, LOSS_PANEL_SPECS)):
        plot_panel(
            ax,
            curves,
            tag,
            ylabel,
            logy,
            args,
            show_legend=(i == 0),
            show_phase_labels=(i == 0),
        )
    axes[-1].set_xlabel("training updates [million]")
    fig_loss.tight_layout()
    loss_png_path = os.path.join(args.out_dir, "fig_drift_training_process_losses.png")
    loss_pdf_path = os.path.join(args.out_dir, "fig_drift_training_process_losses.pdf")
    fig_loss.savefig(loss_png_path, bbox_inches="tight")
    if not args.no_pdf:
        fig_loss.savefig(loss_pdf_path, bbox_inches="tight")

    fig_reward, ax_reward = plt.subplots(1, 1, figsize=(5.9, 2.7))
    tag, ylabel, logy = REWARD_PANEL_SPEC
    plot_panel(
        ax_reward,
        curves,
        tag,
        ylabel,
        logy,
        args,
        show_legend=True,
        show_phase_labels=True,
    )
    ax_reward.set_xlabel("training updates [million]")
    fig_reward.tight_layout()
    reward_png_path = os.path.join(args.out_dir, "fig_drift_training_process_reward.png")
    reward_pdf_path = os.path.join(args.out_dir, "fig_drift_training_process_reward.pdf")
    fig_reward.savefig(reward_png_path, bbox_inches="tight")
    if not args.no_pdf:
        fig_reward.savefig(reward_pdf_path, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(fig_loss)
        plt.close(fig_reward)
    return curves


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--xmin", type=float, default=0.0)
    parser.add_argument("--xmax", type=float, default=15.0)
    parser.add_argument("--plot_step_million", type=float, default=0.02,
                        help="Resample curves onto this update spacing before smoothing.")
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--max_points", type=int, default=1200)
    parser.add_argument("--rebuild_csv", action="store_true",
                        help="Reload TensorBoard events instead of reusing the cached CSV.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no_pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves = plot_training_process(args)
    for (curve, tag), data in curves.items():
        if len(data):
            print(f"{curve:28s} {tag:18s}: {data[0, 0]:.3f}M -> {data[-1, 0]:.3f}M, "
                  f"{data[0, 1]:.4g} -> {data[-1, 1]:.4g}")


if __name__ == "__main__":
    main()
