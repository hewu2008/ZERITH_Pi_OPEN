#!/bin/bash

LOG_DIR="logs"

export LEROBOT_HOME=/data/4T-1/hewu/dataset
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=0,1

mkdir -p "$LOG_DIR"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_zerith \
  --exp_name clear_bin_box_20260720_v1_5e-5_30k_bs16_ah30_pi05_full \
  --eval_data_path /data/4T-1/hewu/dataset/hewu2008/clear_the_bin_box_20260721_v1 \
  --eval_traj_ids 0 1 2 3 \
  --eval_max_infer_time 40 \
  --resume 2>&1 | tee "$LOG_DIR"/"train_log_pi05_full.txt"
