#!/bin/bash

python scripts/open_loop_eval.py \
  --vla_base_model_path /home/jszn/hewu/model_zoo/lingbot-vla-v2-6b \
  --model_path /home/jszn/hewu/alg-product/ZERITH_Lingbot_VLA_V2/output/zerith_lora/checkpoints/global_step_32000/hf_ckpt \
  --robo_name zerith \
  --data_path /home/jszn/hewu/dataset/hewu2008/1_clear_the_bin_box_20260721 \
  --traj_ids 0 1 2 3 \
  --max_infer_time 40 \
  --use_length 50