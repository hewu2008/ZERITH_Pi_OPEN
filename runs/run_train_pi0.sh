#!/bin/bash

LOG_DIR="logs"

export LEROBOT_HOME=/home/jszn/hewu/dataset/

mkdir -p "$LOG_DIR"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_zerith \
  --exp_name pick_and_place_v2_2.5e-5_30k_bs8 \
  --overwrite 2>&1 | tee "$LOG_DIR"/"train_log.txt"