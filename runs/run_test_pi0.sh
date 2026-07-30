#!/bin/bash

python robot_infer/scripts/test_pi0.py \
  --host 172.31.200.250 \
  --port 55555 \
  --prompt "Pick and place the two white parts in the box" \
  --init_hdf5 /home/jszn/hewu/dataset/hdf5/1_clear_the_bin_box_20260720/014cedb5f6d040188f953e97f832dd23/episode.hdf5 \