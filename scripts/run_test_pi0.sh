#!/bin/bash

python robot_infer/scripts/test_pi0.py \
  --host 172.31.200.250 \
  --port 55555 \
  --prompt "Use right hand to move the white rectangular block onto the cardboard box" \
  --init_hdf5 /home/robot/hewu/dataset/1_put_the_rectangular_block_on_the_box/01e334154d7e4c8c90ddb3ffa2d8ea64/episodes.hdf5 \