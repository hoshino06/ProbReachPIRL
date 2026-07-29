# -*- coding: utf-8 -*-
"""Short same-workstation speed benchmark for drift TD3/PIRL training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/home/user/anaconda3/envs/pirl/bin/python")


@dataclass(frozen=True)
class BenchmarkCase:
    key: str
    label: str
    method: str
    checkpoint: str
    env: dict[str, str]


COMMON_ENV = {
    "CASE": "drift",
    "SEEDS": "1",
    "NUM_WORKERS": "2",
    "DRIFT_DT": "0.1",
    "DRIFT_RESET_SCALE": "1.0",
    "DRIFT_RESET_MODE": "mixture",
    "DRIFT_RESET_MIXTURE_PROBS": "0.3,0.3,0.4",
    "INITIAL_EXPLORATION_POLICY": "policy",
    "LEARNER_NUM_GPUS": "0.16",
    "HJB_LAPLACIAN_MODE": "loop",
    "SCHEDULE_TIME_BASE": "local",
    "PINN_EXPAND_TIME_BASE": "local",
    "CHECKPOINT_FREQ": "1000000000",
}


CASES = [
    BenchmarkCase(
        key="td3_fixed",
        label="TD3 fixed tau",
        method="td3",
        checkpoint="logs/drift/td3_T01/up10M_scale10_mix334/ckpt-10000000",
        env={
            "DRIFT_RESET_T_MODE": "fixed",
        },
    ),
    BenchmarkCase(
        key="td3_random",
        label="TD3 random tau",
        method="td3",
        checkpoint="logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-10000000",
        env={
            "DRIFT_RESET_T_MODE": "random",
            "DRIFT_RESET_T_MIN": "0.0",
            "DRIFT_RESET_T_MAX": "5.0",
        },
    ),
    BenchmarkCase(
        key="pirl_transition",
        label="PIRL transition",
        method="scheduling",
        checkpoint=(
            "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/"
            "round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000"
        ),
        env={
            "DRIFT_RESET_T_MODE": "random",
            "DRIFT_RESET_T_MIN": "0.0",
            "DRIFT_RESET_T_MAX": "5.0",
            "PINN_SAMPLE_MODE": "replay",
            "PINN_REPLAY_FRACTION": "1.0",
            "PINN_REPLAY_JITTER": "0.0",
            "SCHEDULE_INITIAL": "1 0.001 0.001",
            "SCHEDULE_FINAL": "1 0.001 0.001",
            "SCHEDULE_CENTER": "500000",
            "SCHEDULE_SHARPNESS": "1e-05",
        },
    ),
    BenchmarkCase(
        key="pirl_refinement",
        label="PIRL refinement",
        method="scheduling",
        checkpoint=(
            "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/"
            "round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/ckpt-10000000"
        ),
        env={
            "DRIFT_RESET_T_MODE": "random",
            "DRIFT_RESET_T_MIN": "0.0",
            "DRIFT_RESET_T_MAX": "5.0",
            "PINN_SAMPLE_MODE": "replay_expand",
            "PINN_REPLAY_FRACTION": "0.9",
            "PINN_REPLAY_JITTER": "0.0",
            "PINN_EXPAND_JITTER_INITIAL": "0.02",
            "PINN_EXPAND_JITTER_FINAL": "0.03",
            "PINN_EXPAND_CENTER": "500000",
            "PINN_EXPAND_SHARPNESS": "1e-05",
            "SCHEDULE_INITIAL": "1 0.0015 0.0015",
            "SCHEDULE_FINAL": "1 0.0015 0.0015",
            "SCHEDULE_CENTER": "500000",
            "SCHEDULE_SHARPNESS": "1e-05",
        },
    ),
]


def load_last_scalar(event_dir: Path, tag: str) -> tuple[int, float]:
    from tensorboard.backend.event_processing import event_accumulator

    event_files = sorted(event_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        raise RuntimeError(f"No TensorBoard event file found under {event_dir}")

    best_step = -1
    best_value = float("nan")
    for event_file in event_files:
        acc = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            continue
        values = acc.Scalars(tag)
        if values and values[-1].step > best_step:
            best_step = values[-1].step
            best_value = float(values[-1].value)
    if best_step < 0:
        raise RuntimeError(f"Scalar tag {tag!r} not found under {event_dir}")
    return best_step, best_value


def run_case(
    case: BenchmarkCase,
    repeat_idx: int,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, str | int | float]:
    log_tag = f"{case.key}_r{repeat_idx:02d}"
    log_dir = out_dir / log_tag
    run_log = out_dir / f"{log_tag}.stdout.log"
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env.update(case.env)
    env.update(
        {
            "NUM_UPDATES": str(args.updates),
            "LOG_FREQ": str(args.updates),
            "LOG_TAG": log_tag,
            "LOG_DIR_OVERRIDE": str(log_dir),
        }
    )

    python = Path(args.python)
    command = [
        str(python),
        "main_training_pirl.py",
        "--case",
        env["CASE"],
        "--method",
        case.method,
        "--seed",
        "1",
        "--num_workers",
        env["NUM_WORKERS"],
        "--num_updates",
        str(args.updates),
        "--checkpoint",
        case.checkpoint,
        "--pinn_sample_mode",
        env.get("PINN_SAMPLE_MODE", "uniform"),
        "--pinn_replay_fraction",
        env.get("PINN_REPLAY_FRACTION", "1.0"),
        "--pinn_replay_jitter",
        env.get("PINN_REPLAY_JITTER", "0.0"),
        "--pinn_expand_jitter_initial",
        env.get("PINN_EXPAND_JITTER_INITIAL", "0.0"),
        "--pinn_expand_jitter_final",
        env.get("PINN_EXPAND_JITTER_FINAL", "0.15"),
        "--pinn_expand_center",
        env.get("PINN_EXPAND_CENTER", "500000"),
        "--pinn_expand_sharpness",
        env.get("PINN_EXPAND_SHARPNESS", "1e-05"),
        "--pinn_expand_time_base",
        env["PINN_EXPAND_TIME_BASE"],
        "--learner_num_gpus",
        env["LEARNER_NUM_GPUS"],
        "--hjb_laplacian_mode",
        env["HJB_LAPLACIAN_MODE"],
        "--drift_reset_scale",
        env["DRIFT_RESET_SCALE"],
        "--drift_reset_mode",
        env["DRIFT_RESET_MODE"],
        "--drift_reset_mixture_probs",
        env["DRIFT_RESET_MIXTURE_PROBS"],
        "--drift_reset_t_mode",
        env["DRIFT_RESET_T_MODE"],
        "--drift_reset_t_min",
        env.get("DRIFT_RESET_T_MIN", "0.0"),
        "--initial_exploration_policy",
        env["INITIAL_EXPLORATION_POLICY"],
        "--schedule_time_base",
        env["SCHEDULE_TIME_BASE"],
        "--checkpoint_freq",
        env["CHECKPOINT_FREQ"],
        "--log_freq",
        str(args.updates),
        "--log_tag",
        log_tag,
        "--log_dir_override",
        str(log_dir),
        "--drift_dt",
        env["DRIFT_DT"],
        "--verbose",
        "0",
        "--device",
        "auto",
    ]
    if "DRIFT_RESET_T_MAX" in env:
        command.extend(["--drift_reset_t_max", env["DRIFT_RESET_T_MAX"]])
    if case.method == "scheduling":
        command.extend(["--schedule_center", env["SCHEDULE_CENTER"]])
        command.extend(["--schedule_sharpness", env["SCHEDULE_SHARPNESS"]])
        command.extend(["--schedule_initial", *env["SCHEDULE_INITIAL"].split()])
        command.extend(["--schedule_final", *env["SCHEDULE_FINAL"].split()])

    started = time.perf_counter()
    with open(run_log, "w") as f:
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    process_seconds = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"{case.key} repeat {repeat_idx} failed with code {proc.returncode}; see {run_log}")

    step, train_ups = load_last_scalar(log_dir, "Train/Updates Per Second")
    _, elapsed_seconds = load_last_scalar(log_dir, "Train/Elapsed Seconds")
    return {
        "method": case.key,
        "label": case.label,
        "repeat": repeat_idx,
        "updates": args.updates,
        "event_step": step,
        "train_updates_per_second": train_ups,
        "train_million_updates_per_hour": train_ups * 3600.0 / 1_000_000.0,
        "train_hours_per_million_updates": 1_000_000.0 / (train_ups * 3600.0),
        "train_elapsed_seconds": elapsed_seconds,
        "process_seconds": process_seconds,
        "log_dir": str(log_dir),
        "stdout_log": str(run_log),
        "checkpoint": case.checkpoint,
    }


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path) -> list[dict[str, str | int | float]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in [
            "repeat",
            "updates",
            "event_step",
        ]:
            row[key] = int(row[key])
        for key in [
            "train_updates_per_second",
            "train_million_updates_per_hour",
            "train_hours_per_million_updates",
            "train_elapsed_seconds",
            "process_seconds",
        ]:
            row[key] = float(row[key])
    return rows


def write_summary(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    by_method: dict[str, list[dict[str, str | int | float]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    summary = []
    for case in CASES:
        if case.key not in by_method:
            continue
        values = by_method[case.key]
        speeds = sorted(float(row["train_million_updates_per_hour"]) for row in values)
        costs = sorted(float(row["train_hours_per_million_updates"]) for row in values)
        mid = len(values) // 2
        if len(values) % 2 == 0:
            median_speed = 0.5 * (speeds[mid - 1] + speeds[mid])
            median_cost = 0.5 * (costs[mid - 1] + costs[mid])
        else:
            median_speed = speeds[mid]
            median_cost = costs[mid]
        summary.append(
            {
                "method": case.key,
                "label": case.label,
                "repeats": len(values),
                "updates_per_repeat": int(values[0]["updates"]),
                "median_million_updates_per_hour": median_speed,
                "median_hours_per_million_updates": median_cost,
                "min_million_updates_per_hour": min(speeds),
                "max_million_updates_per_hour": max(speeds),
                "min_hours_per_million_updates": min(costs),
                "max_hours_per_million_updates": max(costs),
            }
        )
    write_csv(path, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--cases", nargs="+", default=None, choices=[case.key for case in CASES])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "logs/drift/compute_speed_benchmark" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at": stamp,
        "updates": args.updates,
        "repeats": args.repeats,
        "python": args.python,
        "cases": [case.__dict__ for case in CASES],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    raw_csv = out_dir / "raw_results.csv"
    rows = read_existing_rows(raw_csv)
    requested_cases = set(args.cases) if args.cases is not None else {case.key for case in CASES}
    if args.force:
        rows = [
            row for row in rows
            if not (str(row["method"]) in requested_cases and 1 <= int(row["repeat"]) <= args.repeats)
        ]
    completed = {(str(row["method"]), int(row["repeat"])) for row in rows}
    if rows:
        write_summary(out_dir / "summary.csv", rows)

    for repeat in range(1, args.repeats + 1):
        for case in CASES:
            if case.key not in requested_cases:
                continue
            if (case.key, repeat) in completed:
                print(f"[{repeat}/{args.repeats}] {case.key} skip existing", flush=True)
                continue
            print(f"[{repeat}/{args.repeats}] {case.key} start", flush=True)
            row = run_case(case, repeat, args, out_dir)
            rows.append(row)
            completed.add((case.key, repeat))
            write_csv(raw_csv, rows)
            write_summary(out_dir / "summary.csv", rows)
            print(
                f"[{repeat}/{args.repeats}] {case.key}: "
                f"{row['train_million_updates_per_hour']:.4f} M updates/h, "
                f"{row['train_hours_per_million_updates']:.2f} h/M",
                flush=True,
            )

    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
