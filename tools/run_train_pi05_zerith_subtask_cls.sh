#!/bin/bash
# Warm-start the subtask classification head from the finished `pi05_zerith_subtask` run (59999).
# No `--overwrite` on purpose: the experiment directory is new, and the flag would delete it if a
# previous attempt already wrote checkpoints there.

export PYTHON_CLIENT_MEM_FRACTION=0.45
export CUDA_VISIBLE_DEVICES=1
export LEROBOT_HOME=/data/4T-2/dataset/

python scripts/train.py pi05_zerith_subtask_cls \
  --project_name openpi_zerith \
  --exp_name part_sorting_20261027_subtask_cls_8k_lr1e-3_bs8 \
  --num_workers 8 \
  --overwrite \
  --no-wandb_enabled 2>&1 | tee logs/train_log_pi05_subtask_cls.log
