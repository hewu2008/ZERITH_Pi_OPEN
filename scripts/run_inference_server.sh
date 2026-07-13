#!/bin/bash

python scripts/remote_infer_server.py policy:checkpoint \
  --policy.config=test \
  --policy.dir=openpi_checkpoints/test/my_first_run/30000 \
  --port=55555