#!/bin/bash
# Continue the `pi05_zerith_subtask_cls` run (7999) with ALL parameters unfrozen and a much lower lr.
# The classifier head is already trained, so this stage only fine-tunes the whole policy at 1e-5.
# No `--overwrite` on purpose: the experiment directory is new, and the flag would delete it if a
# previous attempt already wrote checkpoints there.

export PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_VISIBLE_DEVICES=1
export LEROBOT_HOME=/data/4T-2/dataset/

python scripts/train.py pi05_zerith_subtask_cls \
  --project_name openpi_zerith \
  --exp_name part_sorting_20261027_subtask_cls_allparams_8k_lr1e-5_bs8 \
  --num_workers 8 \
  --no-wandb_enabled 2>&1 | tee logs/train_log_pi05_subtask_cls_allparams.log
