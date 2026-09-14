#!/bin/bash

python scripts/visualize_norm_stats.py \
  /data/4T-1/hewu/dataset/hewu2008/libero/norm_stats.json \
  --output logs/norm_stats_viz.png \
  2>&1 | tee -a logs/visualize_norm_stats.log