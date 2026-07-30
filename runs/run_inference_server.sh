#!/bin/bash

python scripts/remote_infer_server.py \
  --policy.config=pi05_zerith \
  --policy.dir=./checkpoints/pi05_zerith/clear_bin_box_20260720_5e-5_30k_bs8_pi05/29999 \
  --port=55555
