# PIRL Scheduling Experiments

This workflow treats PIRL weight scheduling as an outer-loop experiment.
Each round runs several 1M-update scheduling trials, evaluates the resulting
checkpoints by Monte Carlo reachability, and asks an external advisor such as
Codex CLI to choose the next round's starting checkpoint and weights.

## Basic Run

Recommended entry point:

```bash
conda activate pirl
python scheduling_experiment/run_scheduling_experiment.py --config scheduling_experiment/drift_scheduling.toml
```

Edit `scheduling_experiment/drift_scheduling.toml` to keep the baseline
checkpoint, candidate plan, training reset distribution, and evaluation settings
visible in one place. Results are written next to the TOML file in a directory
with the same stem, e.g. `scheduling_experiment/drift_scheduling/`.

`[experiment].max_parallel_candidates` controls how many candidates run at the
same time. Set it to `1` for sequential runs, or `2`/`3` if GPU/CPU memory can
handle multiple scheduling trials concurrently.
`[experiment].learner_num_gpus` controls the fractional GPU allocation for each
Ray learner. For example, `0.16` allows several learner actors to share one GPU,
subject to actual memory use.
`[experiment].checkpoint_freq` controls checkpoint frequency in learner updates.
For 1M-update continuation trials, `1000000` saves only the final checkpoint.
`[experiment].log_freq` controls TensorBoard scalar frequency; larger values
produce smaller event files.

After a round finishes, inspect:

- `scheduling_experiment/drift_scheduling/round_000/results.json`
- `scheduling_experiment/drift_scheduling/round_000/codex_next_plan_prompt.md`

If `advisor_command = ""`, the loop stops after the first round and asks you to
write `scheduling_experiment/drift_scheduling/round_001_plan.json` manually. If
`advisor_command` is set, the advisor writes that file and the loop continues
until `[experiment].rounds` is reached.

## Codex-advised run

Set `advisor_command` in the TOML config to the non-interactive Codex command
available in your environment. For example, if your CLI supports `codex exec`,
use:

```toml
[experiment]
rounds = 4
learner_num_gpus = 0.16
advisor_command = "codex exec"
```

The advisor is constrained to write the next plan JSON. Training and evaluation
are still performed by this deterministic orchestration script.

Use `[advisor_context]` to pass manual experiment knowledge into the advisor
prompt. The loop does not automatically mine `logs/`; instead, put trusted
notes and relevant paths in the TOML so the advisor can account for prior
manual runs.

## Initial Candidate Schema

```toml
[[initial_plan.candidates]]
name = "hjb001_bdr001_ramp500k"
start_checkpoint = ""
schedule_initial = [1.0, 0.0, 0.0]
schedule_final = [1.0, 0.01, 0.01]
schedule_center = 500000
schedule_sharpness = 0.00001
```

Use `start_checkpoint = ""` to start from `[experiment].baseline_checkpoint`.

For continuation runs, `SCHEDULE_TIME_BASE=local` is used so the schedule is
interpreted over the current 1M-update block rather than the absolute global
checkpoint iteration.
