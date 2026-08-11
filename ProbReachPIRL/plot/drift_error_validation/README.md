# Drift error validation

This directory contains the validation figures and the cached Monte Carlo data
used to reproduce the value-function accuracy panel.

Run the commands below from `ProbReachPIRL/`.

```bash
# Recompute the Monte Carlo validation data (computationally expensive).
python analyze_drift_ood_value_accuracy.py

# Recreate panel (b) from the cached NPZ data.
python plot_drift_ood_value_accuracy.py

# Recreate the HJB/boundary-loss panel, then give it the validation-panel name.
python plot_drift_hjb_runtime_summary.py --out_dir plot/drift_error_validation
mv plot/drift_error_validation/loss_panel.png \
   plot/drift_error_validation/panel_a_hjb_bdr.png

# Recreate the side-by-side presentation figure.
python plot_drift_error_validation_combined.py
```

The committed validation run used 1,024 TD3 occupancy episodes, 512 proposed
evaluation states (503 remained after terminal-state filtering), and 256 Monte
Carlo rollouts per state and method. PDF outputs are intentionally not tracked.
