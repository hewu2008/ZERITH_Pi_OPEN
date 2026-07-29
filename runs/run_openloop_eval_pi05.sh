#!/bin/bash

python scripts/open_loop_eval.py \
  --config_name pi05_zerith \
  --checkpoint_dir checkpoints/pi05_zerith/clear_bin_box_20260720_5e-5_30k_bs8_pi05/14000 \
  --data_path /home/jszn/hewu/dataset/hewu2008/clear_bin_box_20260721 \
  --traj_ids 0 1 2 3 \
  --max_infer_time 120 \
  --default_prompt "Pick and place the two white parts in the box"
