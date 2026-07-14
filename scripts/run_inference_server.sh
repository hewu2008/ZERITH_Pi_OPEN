#!/bin/bash

python scripts/remote_infer_server.py \
  --policy.config=test \
  --policy.dir=/media/jszn/Data/hewu/openpi_checkpoints/test/pick_and_place_v2_2.5e-5_30k_bs8/2000 \
  --port=55555