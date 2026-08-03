# 实验记录

## exp01: pi05_zerith open-loop eval

- **日期**: 2026-08-03
- **脚本**: `runs/run_openloop_eval_pi05.sh`
- **命令**:
  ```bash
  python scripts/open_loop_eval.py \
    --config_name pi05_zerith \
    --checkpoint_dir ./checkpoints/pi05_zerith/clear_bin_box_20260720_1e-4_30k_bs16_ah30_pi05/29999 \
    --data_path /home/jszn/hewu/dataset/hewu2008/clear_bin_box_20260721 \
    --traj_ids 0 1 2 3 \
    --max_infer_time 40 \
    --default_prompt "Pick and place the two white parts in the box"
  ```

### 实验配置

| 参数 | 值 |
|---|---|
| config_name | pi05_zerith |
| checkpoint | clear_bin_box_20260720_1e-4_30k_bs16_ah30_pi05/29999 |
| dataset | hewu2008/clear_bin_box_20260721 |
| traj_ids | 0, 1, 2, 3 |
| max_infer_time | 40 |
| prompt | Pick and place the two white parts in the box |
| action_horizon | 30 |
| dataset FPS | 30 |
| dataset length | 4239 |

### 结果

| Trajectory | 长度 | MSE | MAE |
|---|---|---|---|
| 0 | 1020 | 0.001311 | 0.019054 |
| 1 | 1110 | 0.001922 | 0.018959 |
| 2 | 1110 | 0.002033 | 0.021507 |
| 3 | 1080 | 0.002247 | 0.021166 |
| **平均** | — | **0.001878** | **0.020172** |

### 备注

- 4 条轨迹的 MSE 在 0.0013~0.0022 范围内,平均 0.001878
- 4 条轨迹的 MAE 在 0.019~0.022 范围内,平均 0.020172
- 轨迹 0 表现最好(MSE 0.001311),轨迹 3 表现最差(MSE 0.002247)
- 整体误差较小,模型在该任务上有较好的 open-loop 动作预测能力
