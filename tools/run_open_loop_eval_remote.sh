#!/bin/bash
# Open-loop evaluation against the already-running policy server (no local checkpoint load).
# Same measurement as open_loop_eval.sh, but every chunk goes through the websocket service started by
# run_remote_infer_server_pi05_subtask.sh, including its 224x224 + JPEG wire format.
# No CUDA pinning here: the server owns GPU 0 and this script only runs the dataset loop.

export LEROBOT_HOME=/data/4T-2/dataset/
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs

python scripts/open_loop_eval_remote.py \
  --host 127.0.0.1 \
  --port 55555 \
  --data_path /data/4T-2/dataset/hewu2008/part_sorting_20261027_subtask \
  --traj_ids 0 1 2 3 \
  --max_infer_time 240 \
  --action_horizon 30 \
  --default_prompt "Sort all objects into the correct bins using both arms" \
  --save_plot_path ./open_loop_test/subtask_59999_remote/ \
  2>&1 | tee -a logs/open_loop_eval_remote.log
