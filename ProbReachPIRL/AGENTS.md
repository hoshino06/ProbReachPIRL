# ProbReachPIRL Experiment Workflow

This repository is used for PIRL / TD3 experiments on probabilistic reachability.
When a new Codex session starts, use this file as the project-specific operating
guide before changing scripts or launching long training jobs.

## Overall Workflow

1. Inspect `main_script.sh`, `main_training_pirl.py`, and `agent/TD3_PIRL_ray.py`
   before changing training behavior.
2. For PIRL-style experiments, first establish or reuse a TD3 baseline.
3. Check learning progress with TensorBoard event files.
4. Evaluate candidate checkpoints with MC reachability, not reward alone.
5. Only after TD3 behavior is understood, run `scheduling` experiments that add
   HJB/BDR weights.

For drift experiments, curated TD3 baselines are under:

- `logs/drift/td3_T01/`: fixed `T=5.0` TD3.
- `logs/drift/td3_T01_randT/`: random `T in [0, 5]` TD3.
- `logs/drift/td3_T002/`: older short-horizon TD3 baseline.

See `logs/drift/README.md` for the current curated checkpoint map.

## Training Entry Point

Use `main_script.sh` as the stable training entry point. It forwards environment
variables into `main_training_pirl.py`.

Common settings for the drift case:

- `CASE=drift`
- `NUM_WORKERS=2`
- `DRIFT_DT=0.1`
- `DRIFT_RESET_SCALE=1.0`
- `DRIFT_RESET_MODE=mixture`
- `DRIFT_RESET_MIXTURE_PROBS=0.3,0.3,0.4` for `mix334`
- `DRIFT_RESET_T_MODE=fixed` for fixed `T=5.0`
- `DRIFT_RESET_T_MODE=random` for random `T in [0, 5]`
- `INITIAL_EXPLORATION_POLICY=policy` when continuing from a trained policy
- `LEARNER_NUM_GPUS=0.16` when running several Ray jobs on one RTX 6000 Ada

Use `TARGET_UPDATES` when continuing from a checkpoint. `main_script.sh` reads
the checkpoint `itr` and runs only `TARGET_UPDATES - itr` updates.

Example continuation pattern:

```bash
env \
  CASE=drift \
  METHOD=td3 \
  SEEDS="1" \
  NUM_WORKERS=2 \
  TARGET_UPDATES=15000000 \
  DRIFT_DT=0.1 \
  DRIFT_RESET_SCALE=1.0 \
  DRIFT_RESET_MODE=mixture \
  DRIFT_RESET_MIXTURE_PROBS=0.3,0.3,0.4 \
  DRIFT_RESET_T_MODE=random \
  INITIAL_EXPLORATION_POLICY=policy \
  LEARNER_NUM_GPUS=0.16 \
  LOG_TAG=td3_dt01_randT_scale10_mix334_from10m_to15m \
  LOG_DIR_OVERRIDE=logs/drift/td3_T01_randT/15M_scale10_mix334_randT \
  ./main_script.sh td3 logs/drift/td3_T01_randT/10M_scale10_mix334_randT/ckpt-10000000
```

Prefer the scripts in `scripts/` for repeated launches. Recent useful scripts:

- `run_td3_15m_from_10m_scale10_mix334_randT.sh`
- `run_scheduling_hjb001_bdr001_5m_from_0619_1649.sh`
- `retag_tensorboard_hjb_loss.py`

## TD3 Before Scheduling

For a new drift PIRL study:

1. Reuse a TD3 checkpoint from `logs/drift/td3_T01/` or
   `logs/drift/td3_T01_randT/` when available.
2. If needed, extend TD3 first to verify that reward and MC reachability are
   still improving.
3. Use the TD3 checkpoint as the starting point for `scheduling`.

Scheduling runs should normally use:

- `METHOD=scheduling`
- `PINN_SAMPLE_MODE=replay`
- `PINN_REPLAY_FRACTION=1.0`
- `PINN_REPLAY_JITTER=0.0`
- `INITIAL_EXPLORATION_POLICY=policy`
- `SCHEDULE_TIME_BASE=local` for continuation runs

For small HJB/BDR tests, use weights like:

```bash
SCHEDULE_INITIAL="1 0.01 0.01"
SCHEDULE_FINAL="1 0.01 0.01"
```

For ramp-up tests, set `SCHEDULE_INITIAL="1 0 0"` and
`SCHEDULE_FINAL` to the target weights.

## Monitoring

Use TensorBoard event files for reward, losses, and weights. Important tags:

- `RL/Average Reward`
- `RL/Episode Q0`
- `Loss/RL`
- `Loss/HJB`
- `Loss/HJB_replay`
- `Loss/BDR`
- `Weights/RL`
- `Weights/HJB`
- `Weights/BDR`

After the HJB logger change:

- `Loss/HJB` means uniform-sampled HJB evaluation.
- `Loss/HJB_replay` means replay-sampled HJB loss used for replay-based PIRL.

If old event files contain `Loss/HJB_uniform`, retag them with:

```bash
python scripts/retag_tensorboard_hjb_loss.py <run_dir>
```

The script writes a sibling `<run_dir>_retagged`; move the new event file back
only after checking the tags.

## MC Reachability Evaluation

Reward alone is not enough. Use `plot_drift_mc_reachability.py` to compare
Monte Carlo closed-loop reachability against the learned value on both planes:

- `beta-r`
- `ey-epsi`

Standard drift evaluation used in this project:

```bash
python plot_drift_mc_reachability.py \
  --checkpoint <path/to/ckpt> \
  --out_dir plot/<descriptive_eval_name> \
  --T 5.0 \
  --num_grid 31 \
  --num_rollouts 16 \
  --device cuda \
  --no_vector_field \
  --no_show
```

Record these metrics from the output:

- `mean MC`
- `mean V`
- `mean|MC-V|`

Interpretation:

- Higher `mean MC` usually means a better closed-loop policy for that plane.
- Lower `mean|MC-V|` means better value calibration.
- A low error with both `mean MC` and `mean V` low may indicate collapse rather
  than policy improvement.

## Resource Checks

Before launching long runs, check:

```bash
nvidia-smi
pgrep -af 'main_training_pirl.py|ray::Learner'
ps -eo user:24,pcpu,pmem --no-headers
```

This workstation has a 48GB RTX 6000 Ada. GPU memory is usually not the limiting
factor; CPU contention from Ray workers and other users often is.

## File Safety

- Do not delete checkpoints unless explicitly requested.
- When replacing TensorBoard event files, first move old events to `/tmp`.
- Running Python processes do not pick up code changes after `git pull`; restart
  jobs when logger or training semantics change.
- Keep curated summaries in `logs/drift/README.md` after moving or renaming logs.
