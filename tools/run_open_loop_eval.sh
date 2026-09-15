#!/bin/bash
# Open-loop evaluation of a trained pi0_zerith policy on the training set.
# Evaluates hard-coded episodes 0-3 of the clear_the_bin_box training dataset.

export CUDA_VISIBLE_DEVICES=0
export LEROBOT_HOME=/data/4T-2/dataset/
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs

python scripts/open_loop_eval.py \
  --config_name pi0_zerith \
  --checkpoint_dir openpi_checkpoints/pi0_zerith/clear_the_bin_box_20260910_v1_60k_lr1e-5_bs8_ah50_pi0_full/59999 \
  --data_path /data/4T-2/dataset/EmbodiedLab/clear_the_bin_box_20260910_v1 \
  --traj_ids 0 1 2 3 \
  --max_infer_time 240 \
  --action_horizon 30 \
  --save_plot_path ./open_loop_test/ \
  2>&1 | tee -a logs/open_loop_eval.log