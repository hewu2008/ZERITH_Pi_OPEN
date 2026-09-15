#!/bin/bash
# Start the remote policy inference websocket server.
# Serves a trained pi0_zerith policy over websockets on the given port.

export CUDA_VISIBLE_DEVICES=1
export LEROBOT_HOME=/data/4T-2/dataset/
# Avoid remote repo checks (assets/dataset are local).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs

python scripts/remote_infer_server.py \
  --policy.config pi0_zerith \
  --policy.dir openpi_checkpoints/pi0_zerith/clear_the_bin_box_20260910_v1_60k_lr1e-5_bs8_ah50_pi0_full/59999 \
  --default-prompt "Pick and place the door handle into the left bin and the L-shaped bent pipe into the right bin" \
  --port 55555 \
  2>&1 | tee -a logs/remote_infer_server.log