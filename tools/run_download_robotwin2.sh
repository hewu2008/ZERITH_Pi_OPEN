#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

hf download \
    --repo-type dataset TianxingChen/RoboTwin2.0 \
    --include "lerobot_dataset/*" \
    --local-dir /data/4T-1/hewu/dataset/hewu2008/RoboTwin2.0