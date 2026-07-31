#!/bin/bash

python scripts/open_loop_eval.py \
  --config_name pi05_zerith \
  --checkpoint_dir ./checkpoints/pi05_zerith_sanity/clear_bin_box_sanity_5e-5_30k_bs8_h10_pi05/27000 \
  --data_path /home/jszn/hewu/dataset/hewu2008/clear_bin_box_20260721 \
  --traj_ids 0 1 2 3 \
  --max_infer_time 120 \
  --default_prompt "Pick and place the two white parts in the box"
