#!/bin/bash

set -euo pipefail

export PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_VISIBLE_DEVICES=1
export LEROBOT_HOME=/data/4T-2/dataset/
# Avoid remote repo checks (dataset is local)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs

python scripts/compute_norm_stats.py \
  --config_name pi0_zerith \
  2>&1 | tee -a logs/compute_norm_stats.log
