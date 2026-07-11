# -*- coding: utf-8 -*-
"""
Plot 1D MSE/RMSE comparison against the analytical ground truth.

Standalone script:
- reads trained checkpoints for each method/seed,
- evaluates the learned value function on a (T, x) grid,
- compares it with the analytical reach probability,
- plots mean +/- std over seeds.

Directory examples:
    logs/1D/td3/0529_0602_seed_1/ckpt-100000
    logs/1D/pinn/0529_1343_seed_3/ckpt-100000
    logs/1D/scheduling/0529_1809_seed_1/ckpt-100000

Group format:
    "label|method|date_or_run_substring|seed_list"
"""

from __future__ import annotations

import os
import glob
import argparse
import warnings
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from agent.TD3_PIRL_ray import PIRLAgent


METHOD_COLORS = {
    "td3": "#1f77b4",
    "pinn": "#ff7f0e",
    "pinn_fail": "#ffbb78",
    "pirl": "#2ca02c",
}

@dataclass
class RunGroup:
    label: str
    method: str
    run_keys: List[str]
    seeds: List[int]


@dataclass
class HighlightPoint:
    # x-axis label of the bar where the marker is plotted
    label: str
    seed: int
    text: str | None = None

    # If method/run_keys are given, this point is evaluated independently
    # and is NOT included in the bar statistics.
    method: str | None = None
    run_keys: List[str] | None = None


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def get_group_color(label: str) -> str:
    label_low = label.lower()
    if "td3" in label_low:
        return METHOD_COLORS["td3"]
    if "pinn" in label_low and "fail" in label_low:
        return METHOD_COLORS["pinn_fail"]
    if "pinn" in label_low:
        return METHOD_COLORS["pinn"]
    if "ours" in label_low or "pirl" in label_low or "scheduling" in label_low:
        return METHOD_COLORS["pirl"]
    return "#7f7f7f"


def split_method_label(label: str) -> tuple[str, str]:
    label_low = label.lower()
    if "pinn" in label_low and "success" in label_low:
        return "PINN-PI", "success"
    if "pinn" in label_low and "fail" in label_low:
        return "PINN-PI", "fail"
    if "td3" in label_low:
        return "TD3", ""
    if "ours" in label_low or "pirl" in label_low or "scheduling" in label_low:
        return "PIRL", ""
    return label, ""


def set_grouped_bar_xticklabels(fig, ax, x: np.ndarray, labels: list[str], ns: list[int]) -> None:
    tick_labels = []
    centers: dict[str, list[float]] = {}
    subtype_positions: list[tuple[float, str]] = []

    for xi, label, n in zip(x, labels, ns):
        method, subtype = split_method_label(label)
        centers.setdefault(method, []).append(float(xi))
        tick_labels.append(f"(n={n})")
        if subtype:
            subtype_positions.append((float(xi), subtype))

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", pad=4)

    for xi, subtype in subtype_positions:
        ax.text(
            xi,
            -0.085,
            subtype,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    for method, xs in centers.items():
        center = float(np.mean(xs))
        ax.text(
            center,
            -0.15,
            method,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    fig.subplots_adjust(bottom=0.21)


def parse_group_spec(spec: str) -> RunGroup:
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) != 4:
        raise ValueError(
            "Invalid --group format. Use: "
            "\"label|method|date_or_run_substring|1,2,3\""
        )

    label, method, run_key_text, seed_text = parts
    run_keys = [s.strip() for s in run_key_text.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in seed_text.split(",") if s.strip()]

    if not label or not method or not run_keys or not seeds:
        raise ValueError(f"Invalid --group: {spec}")

    return RunGroup(label=label, method=method, run_keys=run_keys, seeds=seeds)


def parse_highlight_spec(spec: str) -> HighlightPoint:
    """
    Parse highlight point specification.

    Two modes are supported.

    1) Highlight a seed already included in a bar group:
        "label|seed"
        "label|seed|legend_text"

    2) Evaluate a seed independently and plot it on an existing bar:
       This point is NOT included in the bar mean/std/n.
        "label|method|date_or_run_substring|seed"
        "label|method|date_or_run_substring|seed|legend_text"

    Examples:
        "PINN-PI (success)|1"
        "PINN-PI (success)|1|seed 1"
        "PINN-PI (success)|pinn|0530_0631|1"
        "PINN-PI (success)|pinn|0530_0631|1|seed 1"
    """
    parts = [p.strip() for p in spec.split("|")]

    # Old format: label|seed or label|seed|legend_text
    if len(parts) in (2, 3):
        try:
            seed = int(parts[1])
        except ValueError:
            raise ValueError(
                "Invalid --highlight format. Use either "
                '"label|seed", "label|seed|legend_text", '
                '"label|method|date_or_run_substring|seed", or '
                '"label|method|date_or_run_substring|seed|legend_text"'
            )

        label = parts[0]
        text = parts[2] if len(parts) == 3 and parts[2] else None
        if not label:
            raise ValueError(f"Invalid --highlight: {spec}")
        return HighlightPoint(label=label, seed=seed, text=text)

    # Independent-evaluation format:
    # label|method|run_key_text|seed or label|method|run_key_text|seed|legend_text
    if len(parts) in (4, 5):
        label = parts[0]
        method = parts[1]
        run_keys = [s.strip() for s in parts[2].split(",") if s.strip()]
        seed = int(parts[3])
        text = parts[4] if len(parts) == 5 and parts[4] else None

        if not label or not method or not run_keys:
            raise ValueError(f"Invalid --highlight: {spec}")

        return HighlightPoint(
            label=label,
            method=method,
            run_keys=run_keys,
            seed=seed,
            text=text,
        )

    raise ValueError(
        "Invalid --highlight format. Use either "
        '"label|seed", "label|seed|legend_text", '
        '"label|method|date_or_run_substring|seed", or '
        '"label|method|date_or_run_substring|seed|legend_text"'
    )

def true_reach_probability(
    T: np.ndarray,
    x: np.ndarray,
    mu: float = 1.0,
    sigma: float = 1.0,
    x_target: float = 2.0,
) -> np.ndarray:
    """Vectorized finite-horizon reach probability for dX = mu dt + sigma dW."""
    T = np.asarray(T, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    prob = np.ones_like(x, dtype=np.float64)

    mask_T0 = T <= 0.0
    prob[mask_T0] = (x[mask_T0] >= x_target).astype(np.float64)

    mask = (T > 0.0) & (x < x_target)
    if np.any(mask):
        sqrtT = np.sqrt(T[mask])
        z1 = (x[mask] - x_target + mu * T[mask]) / (sigma * sqrtT)
        z2 = (x[mask] - x_target - mu * T[mask]) / (sigma * sqrtT)
        prob[mask] = (
            norm.cdf(z1)
            + np.exp(2.0 * mu * (x_target - x[mask]) / (sigma**2)) * norm.cdf(z2)
        )

    return np.clip(prob, 0.0, 1.0)


def make_eval_grid(
    num_T: int,
    num_x: int,
    T_min: float,
    T_max: float,
    x_min: float,
    x_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = np.linspace(T_min, T_max, num_T)
    x = np.linspace(x_min, x_max, num_x)
    TT, XX = np.meshgrid(T, x, indexing="xy")
    states = np.stack([TT.ravel(), XX.ravel()], axis=1).astype(np.float32)
    return TT, XX, states


def find_run_dirs(root: str, method: str, run_key: str, seed: int) -> List[str]:
    """Find run directories with exact seed matching.

    This avoids matching seed_1 against seed_10.
    """
    method_dirs = [
        os.path.join(root, method),
        *glob.glob(os.path.join(root, f"*{method}*")),
    ]
    seed_re = re.compile(
        rf"(?:^|[_\-])seed_?{seed}(?:$|[^0-9])",
        re.IGNORECASE,
    )

    dirs = []
    for base in method_dirs:
        if not os.path.isdir(base):
            continue
        for p in glob.glob(os.path.join(base, "*")):
            if not os.path.isdir(p):
                continue
            name = os.path.basename(p)
            if run_key.lower() not in name.lower():
                continue
            if seed_re.search(name) is None:
                continue
            dirs.append(p)

    return sorted(set(dirs))


def find_checkpoint(run_dir: str, checkpoint: str | None = None) -> str | None:
    """
    Find checkpoint file directly under run_dir.

    The checkpoint is assumed to be a torch-loadable file, e.g.,
        logs/1D/td3/0529_0602_seed_1/ckpt-100000

    If checkpoint is given:
        use run_dir/checkpoint if it exists as a file.
    If checkpoint is None:
        choose the ckpt-* file with the largest numeric suffix.
    """
    if checkpoint is not None:
        candidate = os.path.join(run_dir, checkpoint)
        if os.path.isfile(candidate):
            return candidate

        warnings.warn(f"Checkpoint file not found: {candidate}")
        return None

    ckpts = glob.glob(os.path.join(run_dir, "ckpt-*"))
    ckpts = [p for p in ckpts if os.path.isfile(p)]

    if not ckpts:
        warnings.warn(f"No checkpoint file found in {run_dir}")
        return None

    def ckpt_key(path: str) -> int:
        name = os.path.basename(path)
        try:
            return int(name.split("-")[-1])
        except Exception:
            return -1

    return max(ckpts, key=ckpt_key)


def evaluate_checkpoint(
    checkpoint_path: str,
    states: np.ndarray,
    V_true: np.ndarray,
    batch_size: int = 20000,
):
    agent = PIRLAgent.from_checkpoint(checkpoint_path, learner=False)

    values = []
    for i in range(0, len(states), batch_size):
        batch = states[i:i + batch_size]
        pred = agent.get_value(batch)
        values.append(np.asarray(pred).reshape(-1))

    V_pred = np.concatenate(values, axis=0).astype(np.float64)
    V_pred = np.clip(V_pred, 0.0, 1.0)

    err = V_pred - V_true.reshape(-1)
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    return mse, rmse, mae


def collect_group_errors(
    root: str,
    group: RunGroup,
    states: np.ndarray,
    V_true: np.ndarray,
    checkpoint: str | None,
    batch_size: int,
) -> dict:
    rows = []

    for seed in group.seeds:
        matched = []
        for run_key in group.run_keys:
            matched.extend(find_run_dirs(root, group.method, run_key, seed))
        matched = sorted(set(matched))

        if not matched:
            warnings.warn(
                f"No run directory: label={group.label}, method={group.method}, "
                f"run_keys={group.run_keys}, seed={seed}"
            )
            continue

        run_dir = max(matched, key=os.path.getmtime)
        ckpt = find_checkpoint(run_dir, checkpoint=checkpoint)
        if ckpt is None:
            continue

        try:
            mse, rmse, mae = evaluate_checkpoint(
                ckpt,
                states=states,
                V_true=V_true,
                batch_size=batch_size,
            )
        except Exception as exc:
            warnings.warn(f"Failed to evaluate {ckpt}: {exc}")
            continue

        rows.append({
            "label": group.label,
            "method": group.method,
            "seed": seed,
            "run_dir": run_dir,
            "checkpoint": ckpt,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        })

        print(
            f"{group.label:16s} seed={seed:2d} "
            f"MSE={mse:.6e}, RMSE={rmse:.6e}, MAE={mae:.6e}"
        )

    return {"group": group, "rows": rows}


def collect_highlight_errors(
    root: str,
    highlight_points: List[HighlightPoint],
    states: np.ndarray,
    V_true: np.ndarray,
    checkpoint: str | None,
    batch_size: int,
) -> List[dict]:
    """Evaluate highlight points that are not included in bar statistics."""
    rows = []

    for hp in highlight_points:
        # Old format: the point is already included in the bar group.
        if hp.method is None or hp.run_keys is None:
            continue

        matched = []
        for run_key in hp.run_keys:
            matched.extend(find_run_dirs(root, hp.method, run_key, hp.seed))
        matched = sorted(set(matched))

        if not matched:
            warnings.warn(
                f"No highlight run directory: label={hp.label}, method={hp.method}, "
                f"run_keys={hp.run_keys}, seed={hp.seed}"
            )
            continue

        run_dir = max(matched, key=os.path.getmtime)
        ckpt = find_checkpoint(run_dir, checkpoint=checkpoint)
        if ckpt is None:
            continue

        try:
            mse, rmse, mae = evaluate_checkpoint(
                ckpt,
                states=states,
                V_true=V_true,
                batch_size=batch_size,
            )
        except Exception as exc:
            warnings.warn(f"Failed to evaluate highlight {ckpt}: {exc}")
            continue

        row = {
            "label": hp.label,
            "method": hp.method,
            "seed": hp.seed,
            "run_dir": run_dir,
            "checkpoint": ckpt,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "legend_text": hp.text,
            "independent_highlight": True,
        }
        rows.append(row)

        print(
            f"Highlight {hp.label:16s} seed={hp.seed:2d} "
            f"MSE={mse:.6e}, RMSE={rmse:.6e}, MAE={mae:.6e}"
        )

    return rows


def plot_error_bar(
    results: List[dict],
    metric: str,
    out_path: str,
    ylabel: str,
    highlight_points: List[HighlightPoint] | None = None,
    independent_highlight_rows: List[dict] | None = None,
) -> None:
    labels = []
    means = []
    stds = []
    ns = []
    rows_by_label = {}

    for res in results:
        group = res["group"]
        values = np.array([row[metric] for row in res["rows"]], dtype=np.float64)
        if values.size == 0:
            continue

        labels.append(group.label)
        means.append(values.mean())
        stds.append(values.std(ddof=1) if values.size > 1 else 0.0)
        ns.append(values.size)
        rows_by_label[group.label] = res["rows"]

    if not labels:
        print(f"Skipped: {out_path} (no data)")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(4.2, 4.8))

    ax.bar(x, means, yerr=stds, capsize=4, color=[get_group_color(lab) for lab in labels])

    used_legend_labels = set()

    # 1) Highlights from seeds already included in the bar statistics.
    if highlight_points:
        for hp in highlight_points:
            if hp.method is not None:
                # This is handled below as an independent highlight.
                continue

            if hp.label not in rows_by_label:
                warnings.warn(f"Highlight label not found: {hp.label}")
                continue

            matched_rows = [row for row in rows_by_label[hp.label] if row["seed"] == hp.seed]
            if not matched_rows:
                warnings.warn(f"Highlight seed not found: label={hp.label}, seed={hp.seed}")
                continue

            xi = labels.index(hp.label)
            yi = matched_rows[0][metric]
            legend_label = hp.text if hp.text is not None else f"{hp.label} seed {hp.seed}"

            ax.scatter(
                xi,
                yi,
                s=70,
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                zorder=10,
                label=legend_label if legend_label not in used_legend_labels else None,
            )
            used_legend_labels.add(legend_label)

    # 2) Independent highlights evaluated outside the bar statistics.
    if independent_highlight_rows:
        for row in independent_highlight_rows:
            label = row["label"]
            if label not in labels:
                warnings.warn(f"Independent highlight target label not found: {label}")
                continue

            xi = labels.index(label)
            yi = row[metric]
            legend_label = row.get("legend_text") or f"{label} seed {row['seed']}"

            ax.scatter(
                xi,
                yi,
                s=50,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
                label='Outlier', #legend_label if legend_label not in used_legend_labels else None,
            )
            used_legend_labels.add(legend_label)

    if used_legend_labels:
        ax.legend(frameon=True, loc="upper left")

    set_grouped_bar_xticklabels(fig, ax, x, labels, ns)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(labels) - 0.4)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.21)
    finish_figure(fig, out_path, save_pdf=True)

def finish_figure(fig, out_path: str, save_pdf: bool = True) -> None:
    save_figure(fig, out_path, save_pdf=save_pdf)
    plt.show()
    plt.close(fig)


def save_figure(fig, out_path: str, save_pdf: bool = True) -> None:
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if save_pdf:
        root, _ = os.path.splitext(out_path)
        pdf_path = root + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")


def read_tensorboard_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        warnings.warn(f"TensorBoard is not available: {exc}")
        return {}

    curves: dict[str, list[tuple[int, float]]] = {}
    for event_file in event_files:
        try:
            ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                curves.setdefault(tag, [])
                curves[tag].extend((e.step, float(e.value)) for e in ea.Scalars(tag))
        except Exception as exc:
            warnings.warn(f"Failed to read {event_file}: {exc}")

    for tag in list(curves):
        by_step = {}
        for step, value in curves[tag]:
            by_step[int(step)] = float(value)
        curves[tag] = sorted(by_step.items())
        if not curves[tag]:
            curves.pop(tag, None)
    return curves


def read_csv_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    curves: dict[str, list[tuple[int, float]]] = {}
    csv_files = glob.glob(os.path.join(run_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        return curves

    try:
        import pandas as pd
    except Exception as exc:
        warnings.warn(f"pandas is not available for CSV scalar fallback: {exc}")
        return curves

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
            values = df[[step_col, col]].dropna().to_numpy()
            curves.setdefault(key, [])
            curves[key].extend((int(step), float(value)) for step, value in values)

    for tag in list(curves):
        by_step = {}
        for step, value in curves[tag]:
            by_step[int(step)] = float(value)
        curves[tag] = sorted(by_step.items())
        if not curves[tag]:
            curves.pop(tag, None)
    return curves


def read_run_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    curves = read_tensorboard_scalars(run_dir)
    return curves if curves else read_csv_scalars(run_dir)


def pick_scalar_tag(curves: dict[str, list[tuple[int, float]]], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in curves:
            return cand

    available = list(curves)
    for cand in candidates:
        c = cand.lower().replace("_", "/")
        for tag in available:
            t = tag.lower().replace("_", "/")
            if c == t or c in t or t in c:
                return tag
    return None


def collect_group_performance(
    root: str,
    group: RunGroup,
    reward_tags: list[str],
    final_window: int,
) -> dict:
    rows = []

    for seed in group.seeds:
        matched = []
        for run_key in group.run_keys:
            matched.extend(find_run_dirs(root, group.method, run_key, seed))
        matched = sorted(set(matched))

        if not matched:
            warnings.warn(
                f"No run directory for reward: label={group.label}, method={group.method}, "
                f"run_keys={group.run_keys}, seed={seed}"
            )
            continue

        run_dir = max(matched, key=os.path.getmtime)
        curves = read_run_scalars(run_dir)
        if not curves:
            warnings.warn(f"No scalar curves found in {run_dir}")
            continue

        tag = pick_scalar_tag(curves, reward_tags)
        if tag is None:
            warnings.warn(f"Reward tag not found in {run_dir}. Available examples: {list(curves)[:12]}")
            continue

        values = np.array([value for _, value in curves[tag]], dtype=np.float64)
        if values.size == 0:
            continue

        if final_window > 1:
            value = float(np.mean(values[-final_window:]))
        else:
            value = float(values[-1])

        rows.append({
            "label": group.label,
            "method": group.method,
            "seed": seed,
            "run_dir": run_dir,
            "tag": tag,
            "reward": value,
        })
        print(f"{group.label:16s} seed={seed:2d} reward={value:.6e} tag={tag}")

    return {"group": group, "rows": rows}


def plot_performance_bar(
    results: list[dict],
    metric: str,
    out_path: str,
    ylabel: str,
    save_pdf: bool = True,
) -> None:
    labels = []
    means = []
    stds = []
    ns = []

    for res in results:
        group = res["group"]
        values = np.array([row[metric] for row in res["rows"]], dtype=np.float64)
        if values.size == 0:
            continue
        labels.append(group.label)
        means.append(values.mean())
        stds.append(values.std(ddof=1) if values.size > 1 else 0.0)
        ns.append(values.size)

    if not labels:
        print(f"Skipped: {out_path} (no data)")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(4.2, 4.8))
    ax.bar(x, means, yerr=stds, capsize=4, color=[get_group_color(lab) for lab in labels])
    set_grouped_bar_xticklabels(fig, ax, x, labels, ns)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.21)
    finish_figure(fig, out_path, save_pdf=save_pdf)


def flatten_rows(results: list[dict], metric: str) -> dict[tuple[str, int], dict]:
    rows = {}
    for res in results:
        for row in res["rows"]:
            rows[(row["label"], row["seed"])] = row
    return rows


def plot_tradeoff_scatter(
    error_results: list[dict],
    performance_results: list[dict],
    error_metric: str,
    performance_metric: str,
    out_path: str,
    xlabel: str,
    ylabel: str,
    save_pdf: bool = True,
) -> None:
    error_rows = flatten_rows(error_results, error_metric)
    perf_rows = flatten_rows(performance_results, performance_metric)

    labels = [res["group"].label for res in error_results]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    plotted = False

    for label in labels:
        points = []
        for key, err_row in error_rows.items():
            row_label, _seed = key
            if row_label != label or key not in perf_rows:
                continue
            points.append((perf_rows[key][performance_metric], err_row[error_metric]))

        if not points:
            continue

        arr = np.asarray(points, dtype=np.float64)
        color = get_group_color(label)
        ax.scatter(arr[:, 0], arr[:, 1], s=34, color=color, alpha=0.55)
        ax.scatter(
            arr[:, 0].mean(),
            arr[:, 1].mean(),
            s=120,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            label=label,
            zorder=10,
        )
        plotted = True

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    guide_color = "0.35"
    ax.annotate(
        "",
        xy=(0.36, 0.10),
        xytext=(0.24, 0.10),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": guide_color},
    )
    ax.text(
        0.30,
        0.15,
        "higher reward",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color=guide_color,
    )
    ax.annotate(
        "",
        xy=(0.14, 0.35),
        xytext=(0.14, 0.47),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": guide_color},
    )
    ax.text(
        0.105,
        0.41,
        "lower error",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
        color=guide_color,
    )
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()

    if plotted:
        save_figure(fig, out_path, save_pdf=save_pdf)
        plt.show()
    else:
        print(f"Skipped: {out_path} (no paired data)")
    plt.close(fig)


def save_performance_csv(results: List[dict], out_path: str) -> None:
    import pandas as pd

    rows = []
    for res in results:
        rows.extend(res["rows"])

    if not rows:
        print(f"Skipped: {out_path} (no data)")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

def save_csv(results: List[dict], out_path: str) -> None:
    import pandas as pd

    rows = []
    for res in results:
        rows.extend(res["rows"])

    if not rows:
        print(f"Skipped: {out_path} (no data)")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs/1D")
    parser.add_argument(
        "--group",
        action="append",
        help='Run group: "label|method|date_or_run_substring|1,2,3"',
        default=["TD3|td3|0530_0427|1,2,3,4,5,6,7,8,9,10",
                 "PINN-PI (success)|pinn|0530_0631|1,2,3,7",
                 "PINN-PI (fail)|pinn|0530_0631|4,5,6,8,9,10",
                 "PIRL (ours)|scheduling|0530_0026|1,2,3,4,5,6,7,8,9,10"]
    )
    parser.add_argument(
        "--highlight",
        action="append",
        default=[],
        help='Highlight point. Existing-bar format: "label|seed" or "label|seed|legend_text". '
             'Independent format: "label|method|date_or_run_substring|seed" or '
             '"label|method|date_or_run_substring|seed|legend_text". '
             'Example: "PINN-PI (success)|pinn|0530_0631|1|Outlier"',
    )
    
    parser.add_argument("--out_dir", default="plot/1d_mse_rmse_comparison")

    parser.add_argument("--num_T", type=int, default=101)
    parser.add_argument("--num_x", type=int, default=201)
    parser.add_argument("--T_min", type=float, default=0.0)
    parser.add_argument("--T_max", type=float, default=2.0)
    parser.add_argument("--x_min", type=float, default=-2.0)
    parser.add_argument("--x_max", type=float, default=2.0)

    parser.add_argument(
        "--checkpoint",
        default="ckpt-200000",
        help='Checkpoint name such as "ckpt-100000". If omitted, the latest ckpt-* is used.',
    )
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--plot_metric", default="mse", choices=["mse", "rmse", "mae"])
    parser.add_argument(
        "--reward_tag",
        action="append",
        default=["RL/Episode Reward", "Episode Reward", "Reward", "episode_reward", "reward"],
        help="Candidate scalar tag for the task-performance bar and scatter.",
    )
    parser.add_argument(
        "--reward_final_window",
        type=int,
        default=10,
        help="Average this many final scalar samples for the reward/success-rate summary.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_paper_style()

    groups = [parse_group_spec(spec) for spec in args.group]
    highlight_points = [parse_highlight_spec(spec) for spec in args.highlight]

    TT, XX, states = make_eval_grid(
        num_T=args.num_T,
        num_x=args.num_x,
        T_min=args.T_min,
        T_max=args.T_max,
        x_min=args.x_min,
        x_max=args.x_max,
    )
    V_true = true_reach_probability(TT, XX)

    print("--------------------------------------------")
    print("Evaluation grid")
    print(f"  T: [{args.T_min}, {args.T_max}], num_T={args.num_T}")
    print(f"  x: [{args.x_min}, {args.x_max}], num_x={args.num_x}")
    print(f"  total points: {len(states)}")
    print("--------------------------------------------")

    results = []
    for group in groups:
        print(f"Evaluating group: {group.label}")
        res = collect_group_errors(
            root=args.root,
            group=group,
            states=states,
            V_true=V_true,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
        )
        results.append(res)

    independent_highlight_rows = collect_highlight_errors(
        root=args.root,
        highlight_points=highlight_points,
        states=states,
        V_true=V_true,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
    )

    csv_path = os.path.join(args.out_dir, "fig_1d_error_metrics.csv")
    save_csv(results, csv_path)

    performance_results = []
    print("--------------------------------------------")
    print("Collecting final task performance")
    for group in groups:
        performance_results.append(collect_group_performance(
            root=args.root,
            group=group,
            reward_tags=args.reward_tag,
            final_window=args.reward_final_window,
        ))

    performance_csv_path = os.path.join(args.out_dir, "fig_1d_reward_metrics.csv")
    save_performance_csv(performance_results, performance_csv_path)
    ylabel_map = {
        "mse": "Mean squared error",
        "rmse": "Root mean squared error",
        "mae": "Mean absolute error",
    }

    fig_path = os.path.join(args.out_dir, f"fig_1d_{args.plot_metric}_comparison.png")
    plot_error_bar(
        results,
        metric=args.plot_metric,
        out_path=fig_path,
        ylabel=ylabel_map[args.plot_metric],
        highlight_points=highlight_points,
        independent_highlight_rows=independent_highlight_rows,
    )

    reward_fig_path = os.path.join(args.out_dir, "fig_1d_reward_comparison.png")
    plot_performance_bar(
        performance_results,
        metric="reward",
        out_path=reward_fig_path,
        ylabel="Average reward",
    )

    scatter_fig_path = os.path.join(args.out_dir, f"fig_1d_reward_vs_{args.plot_metric}.png")
    plot_tradeoff_scatter(
        results,
        performance_results,
        error_metric=args.plot_metric,
        performance_metric="reward",
        out_path=scatter_fig_path,
        xlabel="Average reward",
        ylabel=ylabel_map[args.plot_metric],
    )
    for metric in ["mse"]: #["mse", "rmse"]:
        extra_path = os.path.join(args.out_dir, f"fig_1d_{metric}_comparison.png")
        if extra_path != fig_path:
            plot_error_bar(
                results,
                metric=metric,
                out_path=extra_path,
                ylabel=ylabel_map[metric],
                highlight_points=highlight_points,
                independent_highlight_rows=independent_highlight_rows,
            )


if __name__ == "__main__":
    main()
