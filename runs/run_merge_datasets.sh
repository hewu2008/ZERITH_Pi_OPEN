#!/bin/bash

export LEROBOT_HOME=/data/4T-1/hewu/dataset

python scripts/merge_datasets.py \
    --target-repo-id hewu2008/pick_and_place_v4 \
    --source-repo-ids hewu2008/clear_the_bin_box_20260720_v1 hewu2008/pick_and_place_v3
