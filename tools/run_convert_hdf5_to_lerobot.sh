#!/bin/bash

export LEROBOT_HOME=/data/4T-2/dataset

python scripts/convert_new.py \
  --raw_dir /data/4T-2/mocap_data/hdf5/1_clear_the_bin_box_20261026 \
  --repo_id hewu2008/clear_the_bin_box_20261026
