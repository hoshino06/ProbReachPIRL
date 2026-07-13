# ProbReachPIRL

ProbReachPIRL is a research codebase for learning probabilistic reachability with
reinforcement learning and physics-informed losses. It combines TD3 with
Hamilton–Jacobi–Bellman (HJB) and boundary-condition residuals, and supports
fixed or scheduled loss weights.

The repository contains three experiment cases:

- `1D`: one-dimensional stochastic reachability
- `drift`: stochastic drifting-control / path-tracking problem


## Repository layout

```text
.
├── conda_pirl_env.yml            # Reproducible Conda environment
├── LICENSE
└── ProbReachPIRL/
    ├── main_script.sh            # Recommended training entry point
    ├── main_training_pirl.py     # Training configuration and CLI
    ├── agent/
    │   ├── TD3.py
    │   └── TD3_PIRL_ray.py       # Distributed TD3/PIRL implementation
    ├── examples/                 # Dynamics and reach/avoid environments
    ├── scripts/                  # Reusable experiment/maintenance scripts
    ├── scheduling_experiment/    # Automated weight-schedule experiments
    ├── logs/                     # Checkpoints and TensorBoard events
    └── plot*.py                  # Evaluation and plotting utilities
```

## Installation

The checked-in environment targets Linux with CUDA. Install
[Miniconda or Anaconda](https://docs.conda.io/) and create the environment from
the repository root:

```bash
conda env create -f conda_pirl_env.yml
conda activate pirl
cd ProbReachPIRL
```

The environment file is an exact export and therefore includes platform- and
CUDA-specific packages. On a different operating system or GPU stack, use it as
a dependency reference and adapt the PyTorch/CUDA packages to the machine.

## Quick start

`main_script.sh` is the stable training entry point. Its first positional
argument is the method; an optional second argument is a checkpoint path.

Run a short TD3 training job for the default drifting-control case:

```bash
CASE=drift \
METHOD=td3 \
SEEDS="1" \
NUM_WORKERS=2 \
NUM_UPDATES=100000 \
./main_script.sh td3
```

Other cases can be selected with `CASE=1D`:

```bash
CASE=1D SEEDS="1" NUM_WORKERS=1 NUM_UPDATES=100000 ./main_script.sh td3
```

Available methods are:

| Method | Loss weights |
| --- | --- |
| `td3` | TD3 only |
| `pinn` | HJB and boundary residuals only |
| `fixed` | Fixed TD3 + HJB + boundary losses |
| `scheduling` | Loss weights vary during training |

Training runs in the background for every value in `SEEDS`; the wrapper waits
for all seeds to finish. Logs and checkpoints are written below
`logs/<case>/<method>/` unless `LOG_DIR_OVERRIDE` is set.

## Physics-informed scheduling

For PIRL fine-tuning, start from a trained TD3 checkpoint and explicitly set the
initial and final weights in `(TD3, HJB, boundary)` order:

```bash
CASE=drift \
SEEDS="1" \
NUM_WORKERS=2 \
NUM_UPDATES=1000000 \
PINN_SAMPLE_MODE=replay \
PINN_REPLAY_FRACTION=1.0 \
INITIAL_EXPLORATION_POLICY=policy \
SCHEDULE_TIME_BASE=local \
SCHEDULE_INITIAL="1 0 0" \
SCHEDULE_FINAL="1 0.01 0.01" \
./main_script.sh scheduling logs/drift/td3/<run>/ckpt-5000000
```

Common configuration variables include:

- `NUM_WORKERS`, `NUM_UPDATES`, `SEEDS`
- `NUM_COLLOCATIONS` (three space-separated counts)
- `PINN_SAMPLE_MODE` (`uniform`, `replay`, or `replay_expand`)
- `SCHEDULE_INITIAL`, `SCHEDULE_FINAL`, `SCHEDULE_CENTER`,
  `SCHEDULE_SHARPNESS`
- `DRIFT_DT`, `DRIFT_RESET_MODE`, `DRIFT_RESET_T_MODE`
- `LEARNER_NUM_GPUS`, `LOG_FREQ`, `CHECKPOINT_FREQ`
- `LOG_TAG`, `LOG_DIR_OVERRIDE`

For the complete set of options, run:

```bash
python main_training_pirl.py --help
```

### Continue to an absolute update count

Set `TARGET_UPDATES` when resuming. The wrapper reads the checkpoint iteration
and performs only the remaining updates:

```bash
CASE=drift \
SEEDS="1" \
TARGET_UPDATES=15000000 \
INITIAL_EXPLORATION_POLICY=policy \
./main_script.sh td3 logs/drift/td3/<run>/ckpt-10000000
```

## Monitoring

Launch TensorBoard from `ProbReachPIRL/`:

```bash
tensorboard --logdir logs
```

Useful scalar tags include:

- `RL/Average Reward`, `RL/Episode Q0`
- `Loss/RL`, `Loss/HJB`, `Loss/HJB_replay`, `Loss/BDR`
- `Weights/RL`, `Weights/HJB`, `Weights/BDR`

Reward alone is not sufficient to assess reachability. Evaluate candidate
checkpoints with the Monte Carlo tools as well.

## Drift-case evaluation

The standard closed-loop Monte Carlo evaluation compares the learned value with
empirical reachability on the `beta-r` and `ey-epsi` planes:

```bash
python plot_drift_mc_reachability.py \
  --checkpoint logs/drift/<method>/<run>/ckpt-<iteration> \
  --out_dir plot/<evaluation-name> \
  --T 5.0 \
  --num_grid 31 \
  --num_rollouts 16 \
  --device cuda \
  --no_vector_field \
  --no_show
```

Inspect `mean MC` for closed-loop performance and `mean|MC-V|` for value
calibration. A low calibration error with both mean MC and mean value near zero
can indicate collapse rather than improvement.

Additional scripts such as `plot_1d_surface.py`,
`plot_1d_mse_comparison.py`, and `plot_1d_training_curves.py` generate the 1D
analyses and figures. Use each script's `--help` output for its current options.

## Automated scheduling experiments

The `scheduling_experiment/` directory contains a TOML-driven runner for
systematic schedule searches. See
[`ProbReachPIRL/scheduling_experiment/README.md`](ProbReachPIRL/scheduling_experiment/README.md)
for its workflow and configuration format.

## Reproducibility notes

- Record the seed, checkpoint, environment variables, and commit used for each
  experiment.
- Do not compare reward alone; retain Monte Carlo reachability outputs alongside
  TensorBoard logs.
- Checkpoints can be large. Avoid deleting or overwriting curated runs.
- A running Python/Ray process does not pick up later code changes; restart it
  after modifying training or logging behavior.

## License

See [LICENSE](LICENSE) for the license terms.
