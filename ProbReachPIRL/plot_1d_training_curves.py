# -*- coding: utf-8 -*-
"""
Plot 1D training curves with flexible run groups.

This version lets you explicitly choose method, date/run-name substring, and
seed list for each plotted curve. This is useful for separating, for example,
PINN-success and PINN-failure runs, or comparing two scheduling settings.

Directory assumption, e.g.:
    logs/1D/pinn/0508_0739_seed_1/
    logs/1D/scheduling/0528_2258_seed_1/

Examples:
    python plot_1d_training_curves_groups.py \
      --root logs/1D \
      --group "TD3|td3|0508_1853|1,2,3,4,5,6,7,8" \
      --group "PINN success|pinn|0508_0739|1,2,3,4" \
      --group "PINN failure|pinn|0508_0739|5,6,7,8" \
      --group "Scheduling A|scheduling|0528_2258|1,2,3,4,5,6,7,8" \
      --group "Scheduling B|scheduling|0529_0100|1,2,3,4,5,6,7,8"

Group format:
    "label|method|date_or_run_substring|seed_list"

The date_or_run_substring can be a comma-separated list. For example:
    "Scheduling mixed|scheduling|0528_2258,0529_0100|1,2,3,4"

The script first reads TensorBoard event files. If no event files are found, it
falls back to CSV files.
"""

from __future__ import annotations

import os
import glob
import argparse
import warnings
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import ScaledTranslation


@dataclass
class RunGroup:
    label: str
    method: str
    run_keys: List[str]
    seeds: List[int]


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })



def get_group_style(label: str):
    label_low = label.lower()

    if "td3" in label_low:
        return {"color": "#1f77b4", "linestyle": "-"}

    if "pinn" in label_low and "fail" in label_low:
        return {"color": "#ffbb78", "linestyle": "--"}

    if "pinn" in label_low:
        return {"color": "#ff7f0e", "linestyle": "-"}

    if "ours" in label_low or "pirl" in label_low or "scheduling" in label_low:
        return {"color": "#2ca02c", "linestyle": "-"}

    return {"color": None, "linestyle": "-"}


def format_update_count(x, pos):
    """Format update counts as 10k, 100k, etc."""
    if abs(x) >= 1000:
        value = x / 1000.0
        if abs(value - round(value)) < 1e-8:
            return f"{int(round(value))}k"
        return f"{value:g}k"
    if abs(x - round(x)) < 1e-8:
        return f"{int(round(x))}"
    return f"{x:g}"


def parse_group_spec(spec: str) -> RunGroup:
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) != 4:
        raise ValueError(
            "Invalid --group format. Use: "
            '"label|method|date_or_run_substring|1,2,3"'
        )
    label, method, run_key_text, seed_text = parts
    run_keys = [s.strip() for s in run_key_text.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in seed_text.split(",") if s.strip()]
    if not label or not method or not run_keys or not seeds:
        raise ValueError(f"Invalid --group: {spec}")
    return RunGroup(
        label=label,
        method=method,
        run_keys=run_keys,
        seeds=seeds,
    )


def find_run_dirs(root: str, method: str, run_key: str, seed: int) -> List[str]:
    """
    Find run directories matching method, run-name substring, and seed.

    This version matches the seed exactly, so seed_1 does not match seed_10.
    Expected examples:
        logs/1D/td3/0529_0602_seed_1
        logs/1D/pinn/0529_0602_seed_10
        logs/1D/scheduling/0529_0602_seed_1
    """
    method_roots = [
        os.path.join(root, method),
        *glob.glob(os.path.join(root, f"*{method}*")),
    ]

    # Match seed_1 or seed1, but not seed_10 or seed10.
    seed_re = re.compile(
        rf"(?:^|[_\\-])seed_?{seed}(?:$|[^0-9])",
        re.IGNORECASE,
    )

    dirs = []
    for method_root in method_roots:
        if not os.path.isdir(method_root):
            continue

        for p in glob.glob(os.path.join(method_root, "*")):
            if not os.path.isdir(p):
                continue

            name = os.path.basename(p)
            name_lower = name.lower()

            if run_key.lower() not in name_lower:
                continue
            if seed_re.search(name) is None:
                continue

            dirs.append(p)

    return sorted(set(dirs))


def read_tensorboard_scalars(run_dir: str) -> Dict[str, pd.DataFrame]:
    event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        warnings.warn(f"TensorBoard is not available: {exc}")
        return {}

    curves: Dict[str, pd.DataFrame] = {}
    for event_file in event_files:
        try:
            ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                df = pd.DataFrame({
                    "step": [e.step for e in events],
                    "value": [e.value for e in events],
                })
                curves[tag] = pd.concat([curves.get(tag, pd.DataFrame()), df], ignore_index=True)
        except Exception as exc:
            warnings.warn(f"Failed to read {event_file}: {exc}")

    for tag, df in list(curves.items()):
        if df.empty:
            curves.pop(tag, None)
            continue
        df = df.sort_values("step").drop_duplicates("step", keep="last")
        curves[tag] = df.reset_index(drop=True)
    return curves


def read_csv_scalars(run_dir: str) -> Dict[str, pd.DataFrame]:
    curves: Dict[str, pd.DataFrame] = {}
    csv_files = glob.glob(os.path.join(run_dir, "**", "*.csv"), recursive=True)
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        step_col = next((c for c in ["step", "iteration", "update", "update_count", "episode", "ep"] if c in df.columns), None)
        if step_col is None:
            continue
        for col in df.columns:
            if col == step_col or not np.issubdtype(df[col].dtype, np.number):
                continue
            key = col.replace("_", "/")
            d = df[[step_col, col]].rename(columns={step_col: "step", col: "value"}).dropna()
            curves[key] = pd.concat([curves.get(key, pd.DataFrame()), d], ignore_index=True)

    for tag, df in list(curves.items()):
        if df.empty:
            curves.pop(tag, None)
            continue
        df = df.sort_values("step").drop_duplicates("step", keep="last")
        curves[tag] = df.reset_index(drop=True)
    return curves


def read_run_curves(run_dir: str) -> Dict[str, pd.DataFrame]:
    curves = read_tensorboard_scalars(run_dir)
    return curves if curves else read_csv_scalars(run_dir)


def pick_tag(curves: Dict[str, pd.DataFrame], candidates: Sequence[str]) -> str | None:
    available = list(curves.keys())
    for cand in candidates:
        if cand in curves:
            return cand

    # case-insensitive partial match
    for cand in candidates:
        c = cand.lower().replace("_", "/")
        for tag in available:
            t = tag.lower().replace("_", "/")
            if c == t or c in t or t in c:
                return tag
    return None


def interpolate_curves(seed_dfs: Sequence[pd.DataFrame], num_points: int) -> Tuple[np.ndarray | None, np.ndarray | None]:
    valid = [df.sort_values("step") for df in seed_dfs if len(df) >= 2]
    if not valid:
        return None, None

    start = max(float(df["step"].min()) for df in valid)
    end = min(float(df["step"].max()) for df in valid)
    if end <= start:
        return None, None

    grid = np.linspace(start, end, num_points)
    values = []
    for df in valid:
        values.append(np.interp(grid, df["step"].to_numpy(), df["value"].to_numpy()))
    return grid, np.vstack(values)


def collect_group_curves(root: str, group: RunGroup, tag_groups: Dict[str, List[str]]) -> Tuple[Dict[str, List[pd.DataFrame]], Dict[str, str]]:
    collected = {metric: [] for metric in tag_groups}
    selected_tags: Dict[str, str] = {}

    for seed in group.seeds:
        matched = []
        for run_key in group.run_keys:
            matched.extend(find_run_dirs(root, group.method, run_key, seed))
        matched = sorted(set(matched))

        if not matched:
            warnings.warn(f"No run directory: label={group.label}, method={group.method}, run_keys={group.run_keys}, seed={seed}")
            continue

        # If multiple directories match the same seed, choose the newest modified one.
        run_dir = max(matched, key=os.path.getmtime)
        curves = read_run_curves(run_dir)
        if not curves:
            warnings.warn(f"No scalar curves found in {run_dir}")
            continue

        for metric, candidates in tag_groups.items():
            tag = pick_tag(curves, candidates)
            if tag is None:
                warnings.warn(f"Tag not found for metric={metric} in {run_dir}. Available examples: {list(curves)[:12]}")
                continue
            selected_tags.setdefault(metric, tag)
            collected[metric].append(curves[tag])

    return collected, selected_tags


def plot_metric(
    all_curves: Dict[str, Dict[str, List[pd.DataFrame]]],
    groups: Sequence[RunGroup],
    metric: str,
    ylabel: str,
    out_path: str,
    num_points: int,
    yscale: str = "linear",
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_bbox: Tuple[float, float] | None = None,
    legend_offset_mm: Tuple[float, float] = (0.0, 0.0),
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    plotted = False

    for group in groups:
        dfs = all_curves[group.label].get(metric, [])
        grid, values = interpolate_curves(dfs, num_points=num_points)
        if grid is None or values is None:
            continue

        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)

        style = get_group_style(group.label)

        ax.plot(
            grid, mean,
            label=f"{group.label} (n={values.shape[0]})",
            color=style["color"],
            linestyle=style["linestyle"],
        )
        ax.fill_between(
            grid, mean - std, mean + std,
            color=style["color"],
            alpha=0.18,
        )
        plotted = True

    ax.set_xlabel("Update count")
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.xaxis.set_major_formatter(FuncFormatter(format_update_count))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_legend:
        # Legend position control:
        #   legend_loc       : standard matplotlib location, e.g., "lower right"
        #   legend_bbox      : anchor point in axes coordinates, e.g., (0.98, 0.05)
        #   legend_offset_mm : fine offset from legend_bbox in millimeters, e.g., (-2.0, 1.0)
        #
        # If legend_bbox is None, matplotlib uses legend_loc only.
        # If legend_bbox is given, legend_offset_mm shifts it by an exact physical length.
        if legend_bbox is None:
            ax.legend(frameon=True, loc=legend_loc)
        else:
            dx_mm, dy_mm = legend_offset_mm
            offset = ScaledTranslation(
                dx_mm / 25.4,
                dy_mm / 25.4,
                fig.dpi_scale_trans,
            )
            ax.legend(
                frameon=True,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox,
                bbox_transform=ax.transAxes + offset,
            )
    fig.tight_layout()

    if plotted:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        print(f"Skipped: {out_path} (no data)")
    
    plt.show()
    plt.close(fig)


def plot_loss_panel(
    all_curves: Dict[str, Dict[str, List[pd.DataFrame]]],
    groups: Sequence[RunGroup],
    out_path: str,
    num_points: int,
    yscale: str = "log",
    xlim: Tuple[float, float] | None = None,
) -> None:
    metrics = [
        ("rl_loss", "RL loss"),
        ("hjb_loss", "HJB loss"),
        ("bc_loss", "Boundary loss"),
    ]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(7.0, 4.8),
        sharex=True,
    )

    any_plotted = False

    for ax, (metric, ylabel) in zip(axes, metrics):
        for group in groups:
            dfs = all_curves[group.label].get(metric, [])
            grid, values = interpolate_curves(dfs, num_points=num_points)
            if grid is None or values is None:
                continue

            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)

            style = get_group_style(group.label)

            ax.plot(
                grid, mean,
                label=f"{group.label} (n={values.shape[0]})",
                color=style["color"],
                linestyle=style["linestyle"],
            )
            ax.fill_between(
                grid, mean - std, mean + std,
                color=style["color"],
                alpha=0.18,
            )
            any_plotted = True

        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)

        if xlim is not None:
            ax.set_xlim(xlim)

    #axes[0].legend(frameon=True)
    axes[-1].set_xlabel("Update count")
    axes[-1].xaxis.set_major_formatter(FuncFormatter(format_update_count))

    fig.tight_layout()

    if any_plotted:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        print(f"Skipped: {out_path} (no data)")

    plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs/1D")
    parser.add_argument("--group", action="append",
                        help='Run group: "label|method|date_or_run_substring|1,2,3"',
                        default=["TD3|td3|0530_0427|1,2,3,4,5,6,7,8,9,10",
                                 "PINN (success)|pinn|0530_0631|1,2,3,7",
                                 "PINN (fail)|pinn|0530_0631|4,5,6,8,9,10",
                                 "PIRL (ours)|scheduling|0530_0026|1,2,3,4,5,6,7,8,9,10"])
    parser.add_argument("--out_dir", default="plot/1d_training_curves_groups")
    parser.add_argument("--num_points", type=int, default=400)
    parser.add_argument("--loss_yscale", default="log", choices=["linear", "log"])
    args = parser.parse_args()

    groups = [parse_group_spec(spec) for spec in args.group]

    os.makedirs(args.out_dir, exist_ok=True)
    set_paper_style()

    tag_groups = {
        "reward": ["RL/Episode Reward", "Episode Reward", "Reward", "episode_reward", "reward"],
        "rl_loss": ["Loss/RL", "Loss/TD", "TD Loss", "td_loss", "loss_td", "td"],
        "hjb_loss": ["Loss/HJB", "HJB Loss", "hjb_loss", "loss_hjb", "hjb"],
        "bc_loss": ["Loss/BDR", "Loss/Boundary", "Boundary Loss", "bc_loss", "boundary_loss", "loss_bc", "bc"],
        #"total_loss": ["Loss/Total", "Total Loss", "loss_total", "total_loss"],
    }

    all_curves: Dict[str, Dict[str, List[pd.DataFrame]]] = {}
    print("Selected runs and tags:")
    for group in groups:
        curves, selected_tags = collect_group_curves(args.root, group, tag_groups)
        all_curves[group.label] = curves
        print(f"  {group.label}: method={group.method}, run_keys={group.run_keys}, seeds={group.seeds}")
        print(f"    tags: {selected_tags}")

    xlim = [0,100_000]

    plot_metric(all_curves, groups, "reward", "Episode reward",
                os.path.join(args.out_dir, "fig_1d_reward_mean_std.png"), args.num_points,
                yscale="linear", xlim=xlim, ylim= [0,0.6], show_legend=True, 
                legend_bbox = (0.98, 0.42))
    plot_metric(all_curves, groups, "rl_loss", "RL loss",
                os.path.join(args.out_dir, "fig_1d_rl_loss_mean_std.png"), args.num_points,
                yscale=args.loss_yscale, xlim=xlim, ylim=None, show_legend=False)
    plot_metric(all_curves, groups, "hjb_loss", "HJB loss",
                os.path.join(args.out_dir, "fig_1d_hjb_loss_mean_std.png"), args.num_points,
                yscale=args.loss_yscale, xlim=xlim, ylim=None, show_legend=False)
    plot_metric(all_curves, groups, "bc_loss", "Boundary loss",
                os.path.join(args.out_dir, "fig_1d_boundary_loss_mean_std.png"), args.num_points,
                yscale=args.loss_yscale, xlim=xlim, ylim=None, show_legend=False)

    plot_loss_panel(
        all_curves,
        groups,
        os.path.join(args.out_dir, "fig_1d_loss_panel.png"),
        num_points=args.num_points,
        yscale=args.loss_yscale,
        xlim=xlim,
    )
    # plot_metric(all_curves, groups, "total_loss", "Total loss",
    #             os.path.join(args.out_dir, "fig_1d_total_loss_mean_std.png"), args.num_points, yscale=args.loss_yscale)


if __name__ == "__main__":
    main()
