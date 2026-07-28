#!/bin/bash

export LEROBOT_HOME=/home/jszn/hewu/dataset/

python scripts/convert_new.py \
  --raw_dir /home/jszn/hewu/dataset/hdf5/1_clear_the_bin_box_20260720 \
  --repo_id hewu2008/clear_bin_box_pi0
