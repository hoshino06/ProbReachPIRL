#!/bin/bash
set -euo pipefail

METHOD=${1:-td3}
CHECKPOINT=${2:-${CHECKPOINT:-}}

#CASE="1D"
#NUM_WORKERS=1
#SEEDS=(1 2 3 4 5 6 7 8 9 10)

CASE=${CASE:-drift}
NUM_WORKERS=${NUM_WORKERS:-2}
NUM_UPDATES=${NUM_UPDATES:-1000000}
TARGET_UPDATES=${TARGET_UPDATES:-}
NUM_COLLOCATIONS=${NUM_COLLOCATIONS:-}
LEARNER_NUM_GPUS=${LEARNER_NUM_GPUS:-}
HJB_LAPLACIAN_MODE=${HJB_LAPLACIAN_MODE:-loop}
SCHEDULE_CENTER=${SCHEDULE_CENTER:-}
SCHEDULE_SHARPNESS=${SCHEDULE_SHARPNESS:-}
SCHEDULE_INITIAL=${SCHEDULE_INITIAL:-}
SCHEDULE_FINAL=${SCHEDULE_FINAL:-}
SCHEDULE_TIME_BASE=${SCHEDULE_TIME_BASE:-global}
LOG_TAG=${LOG_TAG:-}
LOG_DIR_OVERRIDE=${LOG_DIR_OVERRIDE:-}
DRIFT_RESET_SCALE=${DRIFT_RESET_SCALE:-1.0}
DRIFT_RESET_MODE=${DRIFT_RESET_MODE:-full}
DRIFT_RESET_MIXTURE_PROBS=${DRIFT_RESET_MIXTURE_PROBS:-0.45,0.45,0.10}
DRIFT_RESET_T_MODE=${DRIFT_RESET_T_MODE:-fixed}
DRIFT_RESET_T_MIN=${DRIFT_RESET_T_MIN:-0.2}
DRIFT_RESET_T_MAX=${DRIFT_RESET_T_MAX:-}
DRIFT_DT=${DRIFT_DT:-}
INITIAL_EXPLORATION_POLICY=${INITIAL_EXPLORATION_POLICY:-random}
REPLAY_MEMORY_SIZE=${REPLAY_MEMORY_SIZE:-}
EXPLORATION_NOISE=${EXPLORATION_NOISE:-}
POLICY_UPDATE_FREQ=${POLICY_UPDATE_FREQ:-}
INITIAL_EXPLORATION_NUM=${INITIAL_EXPLORATION_NUM:-}
LEARNING_RATE=${LEARNING_RATE:-}
CRITIC_LR=${CRITIC_LR:-}
ACTOR_LR=${ACTOR_LR:-}
SEEDS=(${SEEDS:-1 2})


for SEED in "${SEEDS[@]}"; do
  RUN_NUM_UPDATES="${NUM_UPDATES}"
  if [[ -n "${TARGET_UPDATES}" ]]; then
    START_UPDATES=0
    if [[ -n "${CHECKPOINT}" ]]; then
      START_UPDATES=$(python - "${CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
print(int(checkpoint.get("itr", 0)))
PY
)
    fi
    RUN_NUM_UPDATES=$((TARGET_UPDATES - START_UPDATES))
    if (( RUN_NUM_UPDATES < 0 )); then
      echo "TARGET_UPDATES (${TARGET_UPDATES}) is smaller than checkpoint itr (${START_UPDATES})." >&2
      exit 1
    fi
  fi

  LOG_DIR="logs/${CASE}/${METHOD}"
  mkdir -p "${LOG_DIR}"

  CMD=(
    python main_training_pirl.py
    --case ${CASE} \
    --method ${METHOD} \
    --seed ${SEED} \
    --num_workers ${NUM_WORKERS} \
    --num_updates ${RUN_NUM_UPDATES} \
    --hjb_laplacian_mode ${HJB_LAPLACIAN_MODE} \
    --drift_reset_scale ${DRIFT_RESET_SCALE} \
    --drift_reset_mode ${DRIFT_RESET_MODE} \
    --drift_reset_mixture_probs ${DRIFT_RESET_MIXTURE_PROBS} \
    --drift_reset_t_mode ${DRIFT_RESET_T_MODE} \
    --drift_reset_t_min ${DRIFT_RESET_T_MIN} \
    --initial_exploration_policy ${INITIAL_EXPLORATION_POLICY} \
    --schedule_time_base ${SCHEDULE_TIME_BASE} \
    --verbose 0 \
    --device auto
  )

  if [[ -n "${CHECKPOINT}" ]]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
  fi

  if [[ -n "${DRIFT_DT}" ]]; then
    CMD+=(--drift_dt "${DRIFT_DT}")
  fi

  if [[ -n "${DRIFT_RESET_T_MAX}" ]]; then
    CMD+=(--drift_reset_t_max "${DRIFT_RESET_T_MAX}")
  fi

  if [[ -n "${REPLAY_MEMORY_SIZE}" ]]; then
    CMD+=(--replay_memory_size "${REPLAY_MEMORY_SIZE}")
  fi

  if [[ -n "${EXPLORATION_NOISE}" ]]; then
    CMD+=(--exploration_noise "${EXPLORATION_NOISE}")
  fi

  if [[ -n "${POLICY_UPDATE_FREQ}" ]]; then
    CMD+=(--policy_update_freq "${POLICY_UPDATE_FREQ}")
  fi

  if [[ -n "${INITIAL_EXPLORATION_NUM}" ]]; then
    CMD+=(--initial_exploration_num "${INITIAL_EXPLORATION_NUM}")
  fi

  if [[ -n "${LEARNING_RATE}" ]]; then
    CMD+=(--learning_rate "${LEARNING_RATE}")
  fi

  if [[ -n "${CRITIC_LR}" ]]; then
    CMD+=(--critic_lr "${CRITIC_LR}")
  fi

  if [[ -n "${ACTOR_LR}" ]]; then
    CMD+=(--actor_lr "${ACTOR_LR}")
  fi

  if [[ -n "${LOG_DIR_OVERRIDE}" ]]; then
    CMD+=(--log_dir_override "${LOG_DIR_OVERRIDE}")
    LOG_DIR="${LOG_DIR_OVERRIDE}"
    mkdir -p "${LOG_DIR}"
  fi

  if [[ -n "${NUM_COLLOCATIONS}" ]]; then
    read -r -a COLLOCATIONS <<< "${NUM_COLLOCATIONS}"
    CMD+=(--num_collocations "${COLLOCATIONS[@]}")
  fi

  if [[ -n "${LEARNER_NUM_GPUS}" ]]; then
    CMD+=(--learner_num_gpus "${LEARNER_NUM_GPUS}")
  fi

  if [[ -n "${SCHEDULE_CENTER}" ]]; then
    CMD+=(--schedule_center "${SCHEDULE_CENTER}")
  fi

  if [[ -n "${SCHEDULE_SHARPNESS}" ]]; then
    CMD+=(--schedule_sharpness "${SCHEDULE_SHARPNESS}")
  fi

  if [[ -n "${SCHEDULE_INITIAL}" ]]; then
    read -r -a INITIAL_WEIGHTS <<< "${SCHEDULE_INITIAL}"
    CMD+=(--schedule_initial "${INITIAL_WEIGHTS[@]}")
  fi

  if [[ -n "${SCHEDULE_FINAL}" ]]; then
    read -r -a FINAL_WEIGHTS <<< "${SCHEDULE_FINAL}"
    CMD+=(--schedule_final "${FINAL_WEIGHTS[@]}")
  fi

  if [[ -n "${LOG_TAG}" ]]; then
    CMD+=(--log_tag "${LOG_TAG}")
  fi

  RUN_LOG="${METHOD}_seed${SEED}.log"
  if [[ -n "${LOG_TAG}" ]]; then
    RUN_LOG="${METHOD}_${LOG_TAG}_seed${SEED}.log"
  fi

  "${CMD[@]}" > "${LOG_DIR}/${RUN_LOG}" 2>&1 &
done

wait
