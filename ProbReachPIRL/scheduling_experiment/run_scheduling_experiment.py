#!/usr/bin/env python3
"""PIRL scheduling experiment loop.

This script launches scheduling runs from TD3/PIRL checkpoints, evaluates each
resulting checkpoint with MC reachability, summarizes TensorBoard scalars, and
optionally asks an external advisor such as Codex CLI to write the next round's
candidate plan.

The training implementation remains in main_script.sh/main_training_pirl.py.
This file is intentionally a thin orchestration layer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - for Python < 3.11 environments.
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MC_LINE_RE = re.compile(
    r"^(?P<plane>[^:]+): mean\|MC-V\|=(?P<mean_abs>[0-9.eE+-]+), "
    r"max\|MC-V\|=(?P<max_abs>[0-9.eE+-]+), "
    r"mean MC=(?P<mean_mc>[0-9.eE+-]+), mean V=(?P<mean_v>[0-9.eE+-]+)"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError(
            "TOML config requires Python 3.11+ tomllib or the tomli package. "
            "Use the pirl conda environment if it provides one of them."
        )
    with path.open("rb") as f:
        return tomllib.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"$ {printable}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed with return code {proc.returncode}: {printable}")
    return proc


def checkpoint_itr(path: Path) -> int:
    # Avoid importing torch in the orchestrator until necessary.
    code = "import sys, torch; print(int(torch.load(sys.argv[1], map_location='cpu').get('itr', 0)))"
    proc = run([sys.executable, "-c", code, str(path)], cwd=PROJECT_ROOT)
    return int(proc.stdout.strip().splitlines()[-1])


def latest_checkpoint(run_parent: Path) -> Path:
    ckpts = list(run_parent.rglob("ckpt-*"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {run_parent}")

    def key(p: Path) -> tuple[int, float]:
        m = re.search(r"ckpt-(\d+)$", p.name)
        itr = int(m.group(1)) if m else -1
        return itr, p.stat().st_mtime

    return max(ckpts, key=key)


def parse_mc_stdout(stdout: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for line in stdout.splitlines():
        m = MC_LINE_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        metrics[d["plane"]] = {
            "mean_abs_mc_v": float(d["mean_abs"]),
            "max_abs_mc_v": float(d["max_abs"]),
            "mean_mc": float(d["mean_mc"]),
            "mean_v": float(d["mean_v"]),
        }
    return metrics


def read_tb_last_scalars(run_dir: Path) -> dict[str, float | None]:
    tags = [
        "RL/Average Reward",
        "Loss/RL",
        "Loss/HJB",
        "Loss/HJB_replay",
        "Loss/BDR",
        "Weights/RL",
        "Weights/HJB",
        "Weights/BDR",
    ]
    out: dict[str, float | None] = {t: None for t in tags}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return out

    event_files = sorted(run_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        return out
    # EventAccumulator accepts a directory and merges event files in it. Use leaf dirs.
    dirs = sorted({p.parent for p in event_files}, key=lambda p: p.stat().st_mtime)
    for d in dirs:
        try:
            ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
            ea.Reload()
            available = set(ea.Tags().get("scalars", []))
            for tag in tags:
                if tag in available:
                    vals = ea.Scalars(tag)
                    if vals:
                        out[tag] = float(vals[-1].value)
        except Exception:
            continue
    return out


def make_codex_prompt(
    round_dir: Path,
    results: list[dict[str, Any]],
    next_plan_path: Path,
    advisor_context: dict[str, Any],
    baseline_checkpoint: Path,
    target_total_updates: int,
    max_parallel_candidates: int,
) -> str:
    manual_notes = advisor_context.get("manual_notes", "").strip()
    reference_paths = advisor_context.get("reference_paths", [])
    context_block = ""
    if manual_notes or reference_paths:
        context_block = "\nAdvisor context from TOML:\n"
        if manual_notes:
            context_block += f"\nManual notes:\n{manual_notes}\n"
        if reference_paths:
            context_block += "\nReference paths:\n"
            for path in reference_paths:
                context_block += f"- {path}\n"

    return f"""You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat {target_total_updates} total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from {baseline_checkpoint}.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: {next_plan_path}
- Return exactly {max_parallel_candidates} candidate(s).

Schema:
{{
  "round_note": "brief rationale",
  "candidates": [
    {{
      "name": "short_unique_name",
      "start_checkpoint": "path/to/ckpt-N",
      "schedule_initial": [1.0, hjb0, bdr0],
      "schedule_final": [1.0, hjb1, bdr1],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-5
    }}
  ]
}}

Selection rules:
- Continue from the best safe checkpoint when reward and MC are stable.
- If reward or meanMC degraded, reduce weights or slow the schedule before trying larger weights.
- Increase HJB/BDR gradually.
- Use at most one TD3-restart control per round, unless all scheduling checkpoints collapsed.
- Do not repeat an existing start_checkpoint + schedule_initial + schedule_final combination unless round_note explains why.
{context_block}

Completed results JSON:
{json.dumps(results, indent=2, ensure_ascii=False)}
"""


def call_advisor(advisor_command: str, prompt: str, prompt_path: Path, next_plan_path: Path) -> None:
    prompt_path.write_text(prompt, encoding="utf-8")
    cmd = shlex.split(advisor_command) + [prompt]
    proc = run(cmd, cwd=PROJECT_ROOT, log_path=prompt_path.with_suffix(".advisor.log"))
    # Some agents print JSON instead of writing the requested file. Accept that too.
    if not next_plan_path.exists():
        text = proc.stdout.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(text[start : end + 1])
            dump_json(next_plan_path, data)
    if not next_plan_path.exists():
        raise RuntimeError(f"Advisor did not create {next_plan_path}")


def validate_advisor_command(advisor_command: str | None) -> None:
    if not advisor_command:
        return
    parts = shlex.split(advisor_command)
    if not parts:
        raise ValueError("advisor_command is set but empty after shell parsing.")
    executable = parts[0]
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"advisor_command executable not found before training starts: {executable!r}. "
            "Set [experiment].advisor_command = \"\" for manual mode, or activate/install the advisor CLI."
        )


def prepare_plan(
    path: Path | None,
    baseline_checkpoint: Path,
    max_candidates: int,
    inline_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if inline_plan is not None:
        plan = inline_plan
    elif path is not None and path.exists():
        plan = load_json(path)
    else:
        raise FileNotFoundError("No candidate plan was provided by TOML config or next-round JSON.")
    candidates = list(plan.get("candidates", []))
    if len(candidates) > max_candidates:
        plan = dict(plan)
        plan["dropped_candidates"] = candidates[max_candidates:]
        plan["candidates"] = candidates[:max_candidates]
        print(
            f"Candidate plan has {len(candidates)} candidates; "
            f"running only the first {max_candidates} to match max_parallel_candidates.",
            flush=True,
        )
    for c in plan.get("candidates", []):
        if not c.get("start_checkpoint"):
            c["start_checkpoint"] = str(baseline_checkpoint)
    return plan


def run_candidate(candidate: dict[str, Any], round_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    name = candidate["name"]
    trial_dir = round_dir / name
    trial_log_dir = trial_dir / "train"
    trial_dir.mkdir(parents=True, exist_ok=True)
    dump_json(trial_dir / "candidate.json", candidate)

    start_ckpt = Path(candidate["start_checkpoint"])
    start_itr = checkpoint_itr(start_ckpt)
    target_updates = start_itr + args.updates_per_round

    env = os.environ.copy()
    for key, value in getattr(args, "training_env", {}).items():
        if value is not None:
            env[str(key)] = str(value)
    env.update({
        "TARGET_UPDATES": str(target_updates),
        "LOG_DIR_OVERRIDE": str(trial_log_dir.resolve()),
        "LOG_TAG": name,
        "CHECKPOINT_FREQ": str(args.checkpoint_freq),
        "LOG_FREQ": str(args.log_freq),
        "SCHEDULE_INITIAL": " ".join(str(x) for x in candidate["schedule_initial"]),
        "SCHEDULE_FINAL": " ".join(str(x) for x in candidate["schedule_final"]),
        "SCHEDULE_CENTER": str(candidate["schedule_center"]),
        "SCHEDULE_SHARPNESS": str(candidate["schedule_sharpness"]),
    })
    if args.learner_num_gpus is not None:
        env["LEARNER_NUM_GPUS"] = str(args.learner_num_gpus)

    run(["bash", "./main_script.sh", "scheduling", str(start_ckpt)], cwd=PROJECT_ROOT, env=env, log_path=trial_dir / "train.log")
    ckpt = latest_checkpoint(trial_log_dir)

    eval_dir = trial_dir / "eval"
    proc = run([
        sys.executable,
        "plot_drift_mc_reachability.py",
        "--checkpoint", str(ckpt),
        "--out_dir", str(eval_dir),
        "--T", str(args.eval_T),
        "--num_grid", str(args.eval_num_grid),
        "--num_rollouts", str(args.eval_num_rollouts),
        "--device", args.eval_device,
        "--no_vector_field",
        "--no_show",
    ], cwd=PROJECT_ROOT, log_path=trial_dir / "eval.log")

    result = {
        "name": name,
        "start_checkpoint": str(start_ckpt),
        "start_itr": start_itr,
        "checkpoint": str(ckpt),
        "target_updates": target_updates,
        "candidate": candidate,
        "mc_metrics": parse_mc_stdout(proc.stdout),
        "tensorboard_last": read_tb_last_scalars(trial_log_dir),
    }
    dump_json(trial_dir / "result.json", result)
    return result


def run_candidates(candidates: list[dict[str, Any]], round_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    max_workers = max(1, int(args.max_parallel_candidates))
    if max_workers == 1:
        return [run_candidate(candidate, round_dir, args) for candidate in candidates]

    results_by_name: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_candidate = {
            executor.submit(run_candidate, candidate, round_dir, args): candidate
            for candidate in candidates
        }
        for future in as_completed(future_to_candidate):
            candidate = future_to_candidate[future]
            results_by_name[candidate["name"]] = future.result()

    return [results_by_name[candidate["name"]] for candidate in candidates]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path,
                   help="TOML config containing experiment, training_env, evaluation, and initial_plan settings.")
    args = p.parse_args()

    inline_plan = None
    args.training_env = {}

    cfg = load_toml(args.config)
    for section in ("experiment", "training_env", "evaluation", "initial_plan"):
        if section not in cfg:
            p.error(f"[{section}] section is required in the TOML config")

    experiment = cfg["experiment"]
    evaluation = cfg["evaluation"]
    args.training_env = cfg["training_env"]
    args.advisor_context = cfg.get("advisor_context", {})

    required_experiment = (
        "baseline_checkpoint",
        "rounds",
        "updates_per_round",
        "target_total_updates",
        "max_parallel_candidates",
        "learner_num_gpus",
        "advisor_command",
        "checkpoint_freq",
        "log_freq",
    )
    for key in required_experiment:
        if key not in experiment:
            p.error(f"[experiment].{key} is required in the TOML config")

    required_training_env = (
        "CASE",
        "METHOD",
        "SEEDS",
        "NUM_WORKERS",
        "DRIFT_DT",
        "DRIFT_RESET_SCALE",
        "DRIFT_RESET_MODE",
        "DRIFT_RESET_MIXTURE_PROBS",
        "DRIFT_RESET_T_MODE",
        "DRIFT_RESET_T_MIN",
        "DRIFT_RESET_T_MAX",
        "INITIAL_EXPLORATION_POLICY",
        "PINN_SAMPLE_MODE",
        "PINN_REPLAY_FRACTION",
        "PINN_REPLAY_JITTER",
        "SCHEDULE_TIME_BASE",
        "HJB_LAPLACIAN_MODE",
    )
    for key in required_training_env:
        if key not in args.training_env:
            p.error(f"[training_env].{key} is required in the TOML config")

    required_evaluation = ("T", "num_grid", "num_rollouts", "device")
    for key in required_evaluation:
        if key not in evaluation:
            p.error(f"[evaluation].{key} is required in the TOML config")

    if "candidates" not in cfg["initial_plan"]:
        p.error("[initial_plan].candidates are required in the TOML config")

    args.baseline_checkpoint = Path(experiment["baseline_checkpoint"])
    args.config = args.config.resolve()
    args.result_root = args.config.with_suffix("")
    args.rounds = int(experiment["rounds"])
    args.updates_per_round = int(experiment["updates_per_round"])
    args.target_total_updates = int(experiment["target_total_updates"])
    args.max_parallel_candidates = int(experiment["max_parallel_candidates"])
    args.advisor_command = str(experiment["advisor_command"]) if experiment["advisor_command"] else None
    validate_advisor_command(args.advisor_command)
    args.learner_num_gpus = float(experiment["learner_num_gpus"])
    args.checkpoint_freq = int(experiment["checkpoint_freq"])
    args.log_freq = int(experiment["log_freq"])

    args.eval_T = float(evaluation["T"])
    args.eval_num_grid = int(evaluation["num_grid"])
    args.eval_num_rollouts = int(evaluation["num_rollouts"])
    args.eval_device = str(evaluation["device"])

    inline_plan = {
        "round_note": cfg["initial_plan"].get("round_note", ""),
        "candidates": cfg["initial_plan"]["candidates"],
    }

    result_root = args.result_root
    result_root.mkdir(parents=True, exist_ok=True)
    plan_path = None

    all_results: list[dict[str, Any]] = []
    for r in range(args.rounds):
        round_dir = result_root / f"round_{r:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        plan = prepare_plan(
            plan_path,
            args.baseline_checkpoint,
            args.max_parallel_candidates,
            inline_plan if r == 0 else None,
        )
        dump_json(round_dir / "plan.json", plan)

        results = run_candidates(plan.get("candidates", []), round_dir, args)
        all_results.extend(results)
        dump_json(round_dir / "results.json", results)
        dump_json(result_root / "all_results.json", all_results)

        next_plan_path = result_root / f"round_{r+1:03d}_plan.json"
        prompt = make_codex_prompt(
            round_dir,
            all_results,
            next_plan_path,
            args.advisor_context,
            args.baseline_checkpoint,
            args.target_total_updates,
            args.max_parallel_candidates,
        )
        prompt_path = round_dir / "codex_next_plan_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")

        if r == args.rounds - 1:
            break
        if args.advisor_command:
            call_advisor(args.advisor_command, prompt, prompt_path, next_plan_path)
        else:
            print(f"Manual mode: edit {next_plan_path} using prompt {prompt_path}")
            break
        plan_path = next_plan_path


if __name__ == "__main__":
    main()
