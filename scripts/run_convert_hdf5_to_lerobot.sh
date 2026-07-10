#!/bin/bash

export LEROBOT_HOME=/mnt/d/wsl/dataset/

python scripts/convert_new.py \
  --raw_dir /mnt/d/wsl/dataset/1_put_the_rectangular_block_on_the_box \
  --repo_id hewu2008/pick_and_place_v1
