#!/bin/bash

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py test \
  --exp_name pick_and_place_v1_2.5e-5_30k_bs8 \
  --overwrite