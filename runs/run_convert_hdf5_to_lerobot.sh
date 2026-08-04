#!/bin/bash

export LEROBOT_HOME=/data/4T-1/hewu/dataset

python scripts/convert_new.py \
  --raw_dir /data/4T-1/hewu/dataset/hdf5/clear_the_bin_box_20260720 \
  --repo_id hewu2008/clear_the_bin_box_20260720
