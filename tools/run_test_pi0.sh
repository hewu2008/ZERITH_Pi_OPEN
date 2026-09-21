#!/bin/bash
#
# RTC guidance is enabled by default (identity mode on the server). Passing
# --rtc is kept for explicitness/compatibility. To tune it, append e.g.:
#   --s_min 15
# The client refuses to start against a server that does not advertise RTC
# support (the server enables RTC by default; --rtc.no-enabled turns it off).

sudo /home/robot/miniconda3/envs/lingbot-va/bin/python robot_infer/scripts/test_pi0.py \
  --host 10.42.0.1 \
  --port 55555 \
  --no_pin_head \
  --prompt "Pick and place the door handle into the left bin and the L-shaped bent pipe into the right bin" \
  --init_hdf5 /data/zerith_data/1_clear_the_bin_box/002e81207a19441bae7f4653ad759d7c/episode.hdf5 \
