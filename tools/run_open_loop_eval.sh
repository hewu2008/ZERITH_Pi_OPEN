#!/bin/bash
# Open-loop evaluation of a trained pi0_zerith policy on the training set.
# Evaluates hard-coded episodes 0-3 of the clear_the_bin_box training dataset.
# Runs on GPU 1: the subtask inference server holds GPU 0.

export CUDA_VISIBLE_DEVICES=1
export LEROBOT_HOME=/data/4T-2/dataset/
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs

python scripts/open_loop_eval.py \
  --config_name pi05_zerith_subtask \
  --checkpoint_dir openpi_checkpoints/pi05_zerith_subtask/part_sorting_20261027_subtask_60k_lr2.5e-5_bs8_ah50_pi05_lora/59999 \
  --data_path /data/4T-2/dataset/hewu2008/part_sorting_20261027_subtask \
  --traj_ids 0 1 2 3 \
  --max_infer_time 240 \
  --action_horizon 30 \
  --default_prompt "Sort all objects into the correct bins using both arms" \
  --save_plot_path ./open_loop_test/subtask_59999/ \
  2>&1 | tee -a logs/open_loop_eval_subtask.log
