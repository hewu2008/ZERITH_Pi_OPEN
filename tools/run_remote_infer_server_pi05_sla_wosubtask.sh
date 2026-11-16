#!/bin/bash
# Start the remote policy inference websocket server.
# Serves a trained pi0_zerith policy over websockets on the given port.
#
# RTC (Real-Time Chunking) guidance is enabled by default in identity mode. To
# switch modes or tune it, append e.g.:
#   --rtc.mode full --rtc.beta 10.0 --rtc.s_min 15
# (mode: "identity" = default, "full" = exact VJP per the paper,
#  "off" = advertise but never guide; disable the feature entirely with
#  --rtc.no-enabled. The server pre-compiles both the plain and guided samplers
#  at startup so the first request cannot hit a JIT storm.)

export CUDA_VISIBLE_DEVICES=0
export LEROBOT_HOME=/data/4T-2/dataset/

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs

python scripts/remote_infer_server.py \
  --policy.config pi05_zerith \
  --policy.dir /data/4T-1/zhangrui/ckpt/train_part_sorting/experiments_n/20500 \
  --default-prompt "Sort all objects into the correct bins using both arms" \
  --subtask.mode none \
  --port 55555 \
  2>&1 | tee -a logs/remote_infer_server_pi05.log