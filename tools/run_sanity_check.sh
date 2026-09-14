#!/bin/bash

set -euo pipefail

mkdir -p logs

python scripts/inspect_sanity_dump.py \
  ./sanity_dump \
  2>&1 | tee -a logs/inspect_sanity_dump.log