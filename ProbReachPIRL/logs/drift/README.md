# Drift training logs

This directory contains curated training logs for the drifting-control task.

## TD3 logs

The main curated TD3 series are:

- `td3_T002/`: fixed short horizon experiments.  The reset horizon is effectively `T=0.2`.
- `td3_T01/`: fixed horizon experiments with `T=5.0`.
- `td3_T01_randT/`: random horizon experiments.  Initial states use `T` sampled in `[0, 5]`.

Directory names use the following convention:

- `upXXM`: training has reached approximately `XX` million TD3 updates.
- `scaleYY`: reset region scale, for example `scale10` means `DRIFT_RESET_SCALE=1.0`.
- `mix334`: reset mixture probabilities `(0.3, 0.3, 0.4)`.
- `mix442`: reset mixture probabilities `(0.4, 0.4, 0.2)`.
- `mix45451`: reset mixture probabilities `(0.45, 0.45, 0.10)`.
- `randT`: reset horizon was randomized instead of fixed.

### `td3_T002`

Older fixed-short-horizon TD3 curriculum.

Main sequence:

- `up01M_scale02_mix001/ckpt-1000000`
- `up03M_scale10_mix45451/ckpt-3000000`
- `up05M_scale10_mix45451/ckpt-5000000`
- `up07M_scale10_mix45451/ckpt-7000000`
- `up09M_scale10_mix45451/ckpt-9000000`
- `up11M_scale10_mix442/ckpt-11000000`
- `up13M_scale10_mix442/ckpt-13000000`

This series is mostly a historical baseline for short-horizon training.

### `td3_T01`

Fixed `T=5.0` TD3 training.

Main sequence:

- `up01M_scale04_mix442/ckpt-1000000`
- `up03M_scale10_mix334/ckpt-3000000`
- `up05M_scale10_mix334/ckpt-5000000`
- `up10M_scale10_mix334/ckpt-10000000`

At the `T=5`, `mu=0.55` MC reachability evaluation slice, the 10M fixed-`T`
runs gave high closed-loop reachability.  Among the evaluated seeds, seed3 was
the best overall.

Figures and `.npz` files:

- `../../plot/drift_mc_reachability_0620_0653_td3_T5_10M/`

### `td3_T01_randT`

Random-horizon TD3 training.

Main sequence:

- `up02M_scale08_mix334_randT/ckpt-2000000`
- `up03M_scale10_mix334_randT/ckpt-3000000`
- `up10M_scale10_mix334_randT/ckpt-10000000`
- `15M_scale10_mix334_randT/` is a running continuation from 10M to 15M.

The random-`T` 10M runs calibrate better than the 3M random-`T` runs, but the
learned policy is still weaker than the fixed-`T` 10M policy on the `T=5`
beta-r slice.  Among the evaluated random-`T` seeds, seed3 had the best overall
MC/value calibration.

Figures and `.npz` files:

- `../../plot/drift_mc_reachability_0619_1831_td3_10M/`

## Scheduling logs

Scheduling/PIRL results should be added here after the current scheduling runs
are evaluated and curated.

## Notes

- TensorBoard scalar logs were retagged after the HJB logger change: use
  `Loss/HJB` for uniform HJB evaluation and `Loss/HJB_replay` for replay-sampled
  HJB loss.
- Checkpoints named directly under a curated directory are the representative
  checkpoints after log cleanup.  Archived per-seed raw runs may exist outside
  these curated directories.
