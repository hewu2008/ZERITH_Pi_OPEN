#!/bin/bash

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py test \
  --exp_name put_rectangular_block_on_the_box_1e-4_100k \
  --overwrite