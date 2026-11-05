#!/bin/bash
# Train the subtask classifier head from scratch (warm-started from the 60k LoRA run, which has no head) with
# grouped learning rates: head 1e-3, backbone (LoRA + action expert + SigLIP) 1e-5.
# No `--overwrite` on purpose: the experiment directory is new, and the flag would delete it if a
# previous attempt already wrote checkpoints there.

export PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_VISIBLE_DEVICES=1
export LEROBOT_HOME=/data/4T-2/dataset/

python scripts/train.py pi05_zerith_subtask_cls \
  --project_name openpi_zerith \
  --exp_name part_sorting_20261027_subtask_cls_headlr1e-3_16k_lr1e-5_bs8 \
  --num_workers 8 \
  --no-wandb_enabled 2>&1 | tee logs/train_log_pi05_subtask_cls.log
