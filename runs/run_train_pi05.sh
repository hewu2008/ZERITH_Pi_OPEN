#!/bin/bash

LOG_DIR="logs"

export LEROBOT_HOME=/home/jszn/hewu/dataset/
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_zerith \
  --exp_name clear_bin_box_20260720_1e-4_30k_bs16_ah30_pi05 \
  --eval_data_path /home/jszn/hewu/dataset/hewu2008/clear_bin_box_20260721 \
  --eval_traj_ids 0 1 2 3 \
  --eval_max_infer_time 40 \
  --eval_default_prompt 'Pick and place the two white parts in the box' \
  --resume 2>&1 | tee "$LOG_DIR"/"train_log_pi05.txt"
