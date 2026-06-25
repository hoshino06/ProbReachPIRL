# Random-T replay-HJB scheduling logs

This directory contains curated scheduling/PIRL runs for the drifting-control
task.

Naming convention:

- `randT`: the reset horizon is randomized.
- `replayHJB`: the HJB residual is sampled from replay memory.
- `ramp001`: HJB and BDR weights were ramped up to `0.01`.
- `const001`: HJB and BDR weights were kept at `0.01`.
- `fail_hjb01`: failure reference where HJB and BDR weights were ramped up to
  `0.1`.

Representative seeds are selected by the MC reachability mean-MC score across
the evaluated beta-r and ey-epsi planes.  For active continuation studies, two
seeds may be kept until the next selection step is made. Extra seeds and pruned
checkpoints are moved under `../../drift_archive/`.

## Runs

- `0619_1649_ramp001_to3M_seed1/`
  - representative seed from the 3M scheduling run.
  - kept checkpoint: `ckpt-3000000`.
  - MC reachability at `T=5.0`, `mu=0.55`:
    - beta-r: meanMC `0.1898`, meanV `0.1149`, meanAbsDiff `0.0772`.
    - ey-epsi: meanMC `0.5746`, meanV `0.4100`, meanAbsDiff `0.1773`.

- `0622_0904_const001_3Mto5M_seed1/`
  - representative seed from the continuation of the 3M scheduling run.
  - kept checkpoints: `ckpt-4000000`, `ckpt-5000000`.
  - MC reachability at `T=5.0`, `mu=0.55`:
    - beta-r: meanMC `0.2438`, meanV `0.1526`, meanAbsDiff `0.0927`.
    - ey-epsi: meanMC `0.5903`, meanV `0.4872`, meanAbsDiff `0.1136`.

The 5M continuation improved meanMC relative to the 3M scheduling checkpoint,
but its beta-r meanMC remained below the TD3 10M baselines.

- `0623_1415_const001_5Mto7M_seed_1/`
  - continuation of the `const001` random-`T`, replay-HJB run.
  - kept checkpoints: `ckpt-6000000`, `ckpt-7000000`.
  - event file was downsampled to 5000-update scalar intervals before upload.
  - MC reachability at `T=5.0`, `mu=0.55`, `ckpt-7000000`:
    - beta-r: meanMC `0.2477`, meanV `0.1657`, meanAbsDiff `0.0851`.
    - ey-epsi: meanMC `0.5831`, meanV `0.4930`, meanAbsDiff `0.0988`.

- `0623_1415_const001_5Mto7M_seed_2/`
  - second seed of the same 5M-to-7M continuation.
  - kept checkpoints: `ckpt-6000000`, `ckpt-7000000`.
  - event file was downsampled to 5000-update scalar intervals before upload.
  - MC reachability at `T=5.0`, `mu=0.55`, `ckpt-7000000`:
    - beta-r: meanMC `0.2578`, meanV `0.1618`, meanAbsDiff `0.0973`.
    - ey-epsi: meanMC `0.5933`, meanV `0.4870`, meanAbsDiff `0.1128`.

The 7M continuation increased meanMC relative to the 5M checkpoint, especially
for seed 2, while seed 1 remained slightly better calibrated by meanAbsDiff.

## Failure reference

- `0619_0938_fail_hjb01_to2p4M_seed1/`
  - representative failed run where HJB and BDR weights were increased to `0.1`.
  - training was stopped around 2.4M updates after reward degradation.
  - kept checkpoint: `ckpt-2400000`.
  - this run is kept to document that `0.1` scheduling weights were too large
    for this continuation setting.
