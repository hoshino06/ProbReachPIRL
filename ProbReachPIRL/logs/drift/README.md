# Drift training logs

This directory contains curated training logs for the drifting-control task.

## TD3 logs

The main curated TD3 series are:

- `td3_T002/`: fixed reset horizon experiments with `T=5.0`.  The time step is  `dt=0.02`.
- `td3_T01/`: fixed reset horizon experiments with `T=5.0`. The time step is  `dt=0.1`.
- `td3_T01_randT/`: random horizon experiments.  Initial states use `T` sampled in `[0, 5]`.

Directory names use the following convention:

- `upXXM`: training has reached approximately `XX` million TD3 updates.
- `scaleYY`: reset region scale, for example `scale10` means `DRIFT_RESET_SCALE=1.0`.
- `mix334`: reset mixture probabilities `(0.3, 0.3, 0.4)`.
- `mix442`: reset mixture probabilities `(0.4, 0.4, 0.2)`.
- `mix45451`: reset mixture probabilities `(0.45, 0.45, 0.10)`.
- `randT`: reset horizon was randomized instead of fixed.

### `td3_T002`

Older fixed-`T=5.0` TD3 curriculum with a smaller time step, `dt=0.02`.

Main sequence:

- `up01M_scale02_mix001/ckpt-1000000`
- `up03M_scale10_mix45451/ckpt-3000000`
- `up05M_scale10_mix45451/ckpt-5000000`
- `up07M_scale10_mix45451/ckpt-7000000`
- `up09M_scale10_mix45451/ckpt-9000000`
- `up11M_scale10_mix442/ckpt-11000000`
- `up13M_scale10_mix442/ckpt-13000000`

This series is mostly a historical baseline for the smaller-`dt` training.

### `td3_T01`

Fixed `T=5.0` TD3 training.

Main sequence:

- `up01M_scale04_mix442/ckpt-1000000`
- `up03M_scale10_mix334/ckpt-3000000`
- `up05M_scale10_mix334/ckpt-5000000`
- `up10M_scale10_mix334/ckpt-10000000`

At the `T=5`, `mu=0.55` MC reachability evaluation slice, the 10M fixed-`T`
runs gave high closed-loop reachability.


### `td3_T01_randT`

Random-horizon TD3 training.

Main sequence:

- `up02M_scale08_mix334_randT/ckpt-2000000`
- `up03M_scale10_mix334_randT/ckpt-3000000`
- `up10M_scale10_mix334_randT/ckpt-10000000`
- `up15M_scale10_mix334_randT/ckpt-15000000`

The random-`T` 15M representative is seed 3 from the `0622_1051` continuation.
It had the best MC reachability meanMC among the 15M seeds on both evaluated
planes at `T=5.0`, `mu=0.55`.


## Scheduling logs

Scheduling/PIRL runs were started after checking the TD3 baselines above.  The
main hand-run workflow was:

1. train or select a TD3 checkpoint,
2. continue from that checkpoint with HJB/BDR scheduling,
3. monitor TensorBoard reward and losses, and
4. evaluate checkpoints with MC reachability on the beta-r and ey-epsi planes at
   `T=5.0`, `mu=0.55`.

Curated scheduling logs are grouped by the conditions that mattered most during
the manual comparison:

- `scheduling_randT_replayHJB/`: random reset horizon, replay-memory HJB
  sampling. This is the main scheduling continuation from the random-`T` TD3
  baseline.
- `scheduling_fixedT_replayHJB/`: fixed `T=5.0`, replay-memory HJB sampling.
- `scheduling_fixedT_uniformHJB/`: fixed `T=5.0`, uniform HJB sampling.
- `scheduling_fixed2randT_uniformHJB/`: started from a fixed-`T` TD3 baseline,
  then continued with randomized `T` and uniform HJB sampling.

Within each scheduling group, directory names keep the date/time prefix for
traceability and use short condition tags:

- `ramp001`: HJB and BDR weights were ramped up to `0.01`.
- `ramp005`: HJB and BDR weights were ramped up to `0.05`.
- `const001`: HJB and BDR weights were held at `0.01`.
- `fail` or `fail_hjb01`: kept failure reference where larger HJB/BDR weights
  degraded reward.
- `XMtoYM`: continuation range in TD3/PIRL update count.

Representative seeds were selected by MC reachability meanMC on the evaluated
planes.  For active continuations, more than one seed may be kept until the next
selection step is made.  Checkpoints are kept at 1M-update intervals whenever
possible; stopped failure references keep the final useful checkpoint.

The earlier long names such as `scheduling_T01_randT_scale10_mix334...` were
archived after curation because `scale10_mix334` mainly describes the inherited
TD3 checkpoint rather than the scheduling condition being compared.

## Notes

- TensorBoard scalar logs were retagged after the HJB logger change: use
  `Loss/HJB` for uniform HJB evaluation and `Loss/HJB_replay` for replay-sampled
  HJB loss.
- Checkpoints named directly under a curated directory are the representative
  checkpoints after log cleanup.  Archived per-seed raw runs may exist outside
  these curated directories.
