#!/bin/bash

LOG_DIR="logs"

export LEROBOT_HOME=/home/jszn/hewu/dataset/
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_zerith \
  --exp_name clear_bin_box_20260720_1e-4_30k_bs16_ah30_pi05 \
  --overwrite 2>&1 | tee "$LOG_DIR"/"train_log_pi05.txt"
