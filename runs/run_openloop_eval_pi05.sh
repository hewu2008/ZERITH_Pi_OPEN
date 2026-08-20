#!/bin/bash

python scripts/open_loop_eval.py \
  --config_name pi05_full_subtask_zerith \
  --checkpoint_dir ./checkpoints/pi05_full_subtask_zerith/clear_the_bin_box_20260720_v2_1e-4_30k_bs16_ah30_pi05_full/29999 \
  --data_path /data/4T-1/hewu/dataset/hewu2008/clear_the_bin_box_20260720_v2 \
  --traj_ids 0 1 2 3 \
  --max_infer_time 40 \
  --default_prompt "Put the two parts into the bins"
