#!/bin/bash

LOG_DIR="logs"

export LEROBOT_HOME=/home/jszn/hewu/dataset/
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_zerith_sanity \
  --exp_name clear_bin_box_sanity_5e-5_30k_bs8_h10_pi05 \
  --resume 2>&1 | tee "$LOG_DIR"/"train_log_pi05.txt"
