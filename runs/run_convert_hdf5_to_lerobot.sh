#!/bin/bash

export LEROBOT_HOME=/home/jszn/hewu/dataset/

python scripts/convert_new.py \
  --raw_dir /home/jszn/hewu/dataset/hdf5/clear_the_bin_box_20260721 \
  --repo_id hewu2008/clear_the_bin_box_20260721
