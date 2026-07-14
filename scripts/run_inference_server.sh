#!/bin/bash

python scripts/remote_infer_server.py \
  --policy.config=test \
  --policy.dir=/home/jszn/hewu/alg-product/ZERITH_Pi_OPEN/openpi_checkpoints/test/pick_and_place_v1_2.5e-5_30k_bs8/22000 \
  --port=55555