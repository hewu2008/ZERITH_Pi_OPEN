#!/bin/bash

export LEROBOT_HOME=/data/4T-2/dataset

python scripts/convert_new.py \
  --raw_dir /data/4T-2/mocap_data/hdf5/part-sorting-20261027 \
  --repo_id hewu2008/part_sorting_20261027
