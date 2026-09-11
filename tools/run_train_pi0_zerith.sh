#!/bin/bash

export PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_VISIBLE_DEVICES=1
export LEROBOT_HOME=/data/4T-1/hewu/dataset

python scripts/train.py pi0_zerith \
  --project_name openpi_zerith \
  --exp_name clear_the_bin_box_20260720_v2_30k_lr1e-5_bs8_ah50_pi0_full \
  --overwrite