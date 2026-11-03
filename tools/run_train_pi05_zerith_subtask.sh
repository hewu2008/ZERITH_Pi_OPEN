#!/bin/bash

export PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_VISIBLE_DEVICES=0,1
export LEROBOT_HOME=/data/4T-2/dataset/

python scripts/train.py pi05_zerith_subtask \
  --project_name openpi_zerith \
  --exp_name part_sorting_20261027_subtask_60k_lr5e-5_bs16_ah50_pi05_lora \
  --num_workers 8 \
  --no-wandb_enabled \
  --overwrite 2>&1 | tee logs/train_log_pi05_subtask_lora.log
