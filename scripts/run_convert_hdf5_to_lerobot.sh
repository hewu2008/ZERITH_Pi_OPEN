#!/bin/bash

export LEROBOT_HOME=/home/jszn/hewu/dataset/

python scripts/convert_new.py \
  --raw_dir /home/jszn/hewu/dataset/1_put_the_part_in_the_box/ \
  --repo_id hewu2008/pick_and_place_v2
