#!/bin/bash

python robot_infer/scripts/test_pi05.py \
  --host 172.31.200.250 \
  --port 55555 \
  --prompt "Pick and place the two white parts in the box" \
  --init_hdf5 ./assets/init_episode/episode.hdf5 \