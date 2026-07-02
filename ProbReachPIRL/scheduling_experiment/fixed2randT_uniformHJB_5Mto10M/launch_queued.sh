#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/ProbReachPIRL/ProbReachPIRL

while pgrep -f "main_training_pirl.py" >/dev/null; do
  date
  echo "waiting for existing main_training_pirl.py jobs to finish"
  sleep 300
done

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pirl
python scheduling_experiment/run_scheduling_experiment.py \
  --config scheduling_experiment/fixed2randT_uniformHJB_5Mto10M.toml
