#!/bin/bash

METHOD=${1:-scheduling}

#CASE="1D"
#NUM_WORKERS=1
#SEEDS=(1 2 3 4 5 6 7 8 9 10)

CASE="drift"
NUM_WORKERS=4
SEEDS=(1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  python main01_training_pirl.py \
    --case ${CASE} \
    --method ${METHOD} \
    --seed ${SEED} \
    --num_workers ${NUM_WORKERS} \
    --verbose 0 \
    --device auto \
    > logs/${CASE}/${METHOD}/${METHOD}_seed${SEED}.log 2>&1 &
done

wait
