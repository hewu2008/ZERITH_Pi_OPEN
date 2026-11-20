# 08 脚本与工具

本仓库的入口集中在 `scripts/`、`robot_infer/scripts/` 与 `tools/`。`scripts/` 是真正的 Python 入口；`tools/` 只是作者本机使用的 `.sh` 命令模板，**含大量绝对路径 / GPU 编号 / conda 路径，复用前必须修改**。

---

## 1. `scripts/` 全量说明

| 脚本 | 用途 | 关键参数 |
| --- | --- | --- |
| [convert_new.py](../../scripts/convert_new.py) | HDF5 → LeRobot | `--raw_dir`、`--repo_id`、`--task`、`--episodes`、`--mode` |
| [compute_norm_stats.py](../../scripts/compute_norm_stats.py) | 计算 `state/actions` 统计 | `--config_name`、`--max_frames` |
| [train.py](../../scripts/train.py) | 训练 pi0/pi0.5 | 配置名、`--exp_name`、`--overwrite/--resume` |
| [remote_infer_server.py](../../scripts/remote_infer_server.py) | WebSocket 策略服务 | `--policy.*`、`--rtc.*`、`--subtask.*` |
| [open_loop_eval.py](../../scripts/open_loop_eval.py) | 本地 Checkpoint 开环评估 | `--config_name`、`--checkpoint_dir`、`--data_path` |
| [open_loop_eval_remote.py](../../scripts/open_loop_eval_remote.py) | 通过 WebSocket 开环评估 | `--host`、`--port`、`--data_path` |
| [merge_bin_box_datasets.py](../../scripts/merge_bin_box_datasets.py) | 合并两个 LeRobot 数据集 | 脚本内硬编码路径 |
| [inspect_sanity_dump.py](../../scripts/inspect_sanity_dump.py) | 检查 sanity dump | 位置参数 `<dump_dir>` |
| [visualize_norm_stats.py](../../scripts/visualize_norm_stats.py) | 可视化 norm stats | `<json_path>`、`--output` |

### 1.1 `convert_new.py`

```bash
uv run scripts/convert_new.py \
  --raw_dir /data/mocap_data/hdf5/part-sorting-20261027 \
  --repo_id hewu2008/part_sorting_20261027
```

- 递归扫描 `--raw_dir` 下的 `.hdf5`；
- 目标目录已存在会**先删除**；
- 图像默认写入 `image`/`video` 模式（`--mode`）；
- 需要 `h5py`、`opencv-python`、`lerobot`、`datasets`。

### 1.2 `compute_norm_stats.py`

```bash
uv run scripts/compute_norm_stats.py --config_name pi05_zerith
```

- 统计 `state` 与 `actions`；
- 默认随机抽取约 25% 的 batch 计算（代码固定 `num_frames = num_frames/4`，且 `shuffle=True`）；
- 输出 `<assets_base_dir>/<asset_id>/norm_stats.json`。

### 1.3 `train.py`

```bash
uv run scripts/train.py pi05_zerith --exp_name my_run --overwrite
```

- 配置名来自 `_CONFIGS`；
- `--exp_name` 决定 `openpi_checkpoints/<config>/<exp_name>/`；
- `--no-wandb_enabled` 关闭上报；
- `--sanity_mode` 固定 batch 调试。

### 1.4 `remote_infer_server.py`

```bash
uv run scripts/remote_infer_server.py \
  --policy.config pi05_zerith_subtask_cls \
  --policy.dir openpi_checkpoints/pi05_zerith_subtask_cls/run/15999 \
  --default-prompt "Sort all objects into the correct bins using both arms" \
  --port 55555 \
  --subtask.mode classifier \
  --subtask.jsonl assets/subtasks.jsonl \
  --rtc.s_min 18
```

- RTC 默认启用（`identity`）；
- 启动时预热 plain + guided 两条 JIT 路径；
- 连接后先发 metadata，再逐请求推理。

### 1.5 `open_loop_eval.py`

```bash
uv run scripts/open_loop_eval.py \
  --config_name pi05_zerith_subtask \
  --checkpoint_dir openpi_checkpoints/pi05_zerith_subtask/run/59999 \
  --data_path /data/dataset/hewu2008/part_sorting_20261027_subtask \
  --traj_ids 0 1 2 3 \
  --max_infer_time 240 \
  --action_horizon 30 \
  --default_prompt "Sort all objects into the correct bins using both arms" \
  --save_plot_path ./open_loop_test/subtask_59999/
```

评估逻辑：

1. 用 `LeRobotDataset` 读取 episode，按 `action_horizon` 跨步采样；
2. `build_observation` 从 dataset 取出三路图像（CHW→HWC、uint8）与 `observation.state`；
3. 调 `policy.infer`，得到反归一化后的动作块；
4. 与 GT 动作逐帧对齐（`min(len(gt), len(pred))`）；
5. 计算 **MSE / MAE**；
6. 若响应含 `subtask`，与 `subtask_index` 对应文本做**归一化精确匹配**（无标注帧跳过）；
7. 为每个关节绘制 `state / gt / pred` 轨迹图。

### 1.6 `open_loop_eval_remote.py`

与本地评估相同的指标，但：

- 不加载 Checkpoint，直接连 `--host/--port`；
- 图像按机器人端格式编码：RGB→BGR → `resize_with_pad(224)` → JPEG（默认 quality 100）；
- 额外统计服务端延迟 `mean / p95 / max`；
- `--no_rgb_to_bgr` 可关闭通道交换（用于对照训练时的通道顺序）。

### 1.7 `merge_bin_box_datasets.py`

- 将两个 LeRobot 数据集合并为一个（默认 `clear_the_bin_box_20260910_v1` 128 集 + `clear_the_bin_box_20261010_v1` 37 集 → v2 165 集）；
- 图像字节原样保留，只重写 `index`、`episode_index`、`task_index`；
- 所有 episode 统一使用第二个数据集的单一任务文本；
- 路径与数据集名**硬编码在脚本顶部**，使用前修改 `V1/V2/OUT`。

### 1.8 `inspect_sanity_dump.py`

```bash
python scripts/inspect_sanity_dump.py ./sanity_dump
```

输出：

- `state.npy`、`actions.npy`、`raw_state.npy`、`raw_actions.npy` 的 shape/dtype、mean/std；
- NaN/Inf 计数；
- **零方差维度告警**（归一化会放大约 1e6 倍）；
- normalized 与 raw 的形状配对检查。

### 1.9 `visualize_norm_stats.py`

```bash
python scripts/visualize_norm_stats.py assets/xxx/norm_stats.json --output logs/norm.png
```

- 每个 key（`state/actions`）一张子图，绘制 `mean/std/q01/q99`；
- 对 `std == 0` 的维度打印告警；
- 默认输出 `<json_stem>_viz.png`。

---

## 2. `tools/` 命令模板

| 脚本 | 作用 | 关键设置 |
| --- | --- | --- |
| [run_convert_hdf5_to_lerobot.sh](../../tools/run_convert_hdf5_to_lerobot.sh) | 转换 | `LEROBOT_HOME=/data/4T-2/dataset` |
| [run_compute_norm_stats.sh](../../tools/run_compute_norm_stats.sh) | 统计 | `CUDA_VISIBLE_DEVICES=1`、离线 HF |
| [run_train_pi0_zerith.sh](../../tools/run_train_pi0_zerith.sh) | 训练 pi0 | 单卡、`--overwrite` |
| [run_train_pi05_zerith.sh](../../tools/run_train_pi05_zerith.sh) | 训练 pi0.5 | 单卡 |
| [run_train_pi05_zerith_subtask.sh](../../tools/run_train_pi05_zerith_subtask.sh) | 训练 pi0.5 子任务 | 双卡、`--no-wandb_enabled` |
| [run_train_pi05_zerith_subtask_cls.sh](../../tools/run_train_pi05_zerith_subtask_cls.sh) | 训练分类头 | 新实验目录，**不加** `--overwrite` |
| [run_train_pi0_zerith_sanity.sh](../../tools/run_train_pi0_zerith_sanity.sh) | sanity 训练 | `--sanity_mode` |
| [run_sanity_check.sh](../../tools/run_sanity_check.sh) | 检查 dump | `inspect_sanity_dump.py` |
| [run_visualize_norm.sh](../../tools/run_visualize_norm.sh) | 可视化统计 | 硬编码 JSON 路径 |
| [run_remote_infer_server.sh](../../tools/run_remote_infer_server.sh) | pi0 服务 | GPU 0、RTC 默认 identity |
| [run_remote_infer_server_pi05.sh](../../tools/run_remote_infer_server_pi05.sh) | pi0.5 服务 | 60k clean 权重 |
| [run_remote_infer_server_pi05_subtask.sh](../../tools/run_remote_infer_server_pi05_subtask.sh) | 子任务分类服务 | `--subtask.mode classifier`、`--s_min 18` |
| [run_remote_infer_server_pi05_sla_wosubtask.sh](../../tools/run_remote_infer_server_pi05_sla_wosubtask.sh) | 无子任务 SLA 对照 | `--subtask.mode none` |
| [run_test_pi0.sh](../../tools/run_test_pi0.sh) | 机器人端启动 | 机器人 Python 绝对路径、`--s_min 18` |
| [run_open_loop_eval.sh](../../tools/run_open_loop_eval.sh) | 本地开环评估 | GPU 1（服务占 GPU 0） |
| [run_open_loop_eval_remote.sh](../../tools/run_open_loop_eval_remote.sh) | 远程开环评估 | 连本机 55555 |
| [run_download_robotwin2.sh](../../tools/run_download_robotwin2.sh) | 下载 RoboTwin2 数据 | 使用 `hf-mirror.com` |

### 2.1 典型工具脚本内容

```bash
# run_train_pi05_zerith.sh
export PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_VISIBLE_DEVICES=1
export LEROBOT_HOME=/data/4T-2/dataset/
python scripts/train.py pi05_zerith \
  --project_name openpi_zerith \
  --exp_name part_sorting_20261027_clean_60k_lr2.5e-5_bs8_ah50_pi05_lora \
  --num_workers 4 --overwrite
```

```bash
# run_remote_infer_server_pi05_subtask.sh（节选）
export CUDA_VISIBLE_DEVICES=0
export LEROBOT_HOME=/data/4T-2/dataset/
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python scripts/remote_infer_server.py \
  --policy.config pi05_zerith_subtask_cls \
  --policy.dir openpi_checkpoints/pi05_zerith_subtask_cls/.../15999 \
  --subtask.mode classifier \
  --subtask.jsonl assets/subtasks.jsonl \
  --port 55555 --rtc.s_min 18
```

```bash
# run_test_pi0.sh（节选）
sudo /home/robot/miniconda3/envs/lingbot-va/bin/python robot_infer/scripts/test_pi0.py \
  --host 172.31.200.200 --port 55555 \
  --no_pin_head --num_steps 864000 \
  --prompt "Sort all objects into the correct bins using both arms" \
  --init_hdf5 /data/zerith_data/.../episode.hdf5 \
  --subtask.mode classifier --subtask.jsonl assets/subtasks.jsonl \
  --home_consecutive 3 --s_min 18
```

> `tools/` 中的 `--s_min 18` 是因为子任务阶段把推理延迟推到约 300ms（≈9 个控制步），需要高于论文下界；普通 pi0 可用 15。

---

## 3. 一键脚本 `train.sh`

仓库根目录的 [train.sh](../../train.sh) 串联三步：

```text
convert_new.py --raw_dir $RAW_DIR --repo_id $REPO_ID
compute_norm_stats.py --config_name $CONFIG_NAME
train.py $CONFIG_NAME --exp_name $EXP_NAME --overwrite
```

环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAW_DIR` | `/path/to/hdf5_files` | HDF5 根目录 |
| `REPO_ID` | `zerith/test` | 必须与 `config.repo_id` 一致 |
| `CONFIG_NAME` | `test` | 训练配置名 |
| `EXP_NAME` | `zerithtest` | 实验名 |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 使用的 GPU |
| `XLA_FLAGS` | `--xla_gpu_force_compilation_parallelism=8` | JAX 编译并行度 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.9` | 显存占比 |

日志：`logs/{convert,compute_norm_stats,train}_<timestamp>.log`、`logs/summary_<timestamp>.log`。

```bash
RAW_DIR=/data/hdf5 REPO_ID=zerith/part_sorting CONFIG_NAME=pi05_zerith EXP_NAME=run1 ./train.sh
```

---

## 4. 评估脚本对比

| 维度 | `open_loop_eval.py` | `open_loop_eval_remote.py` |
| --- | --- | --- |
| 策略来源 | 本地加载 Checkpoint | 远程 WebSocket 服务 |
| 是否覆盖服务端逻辑 | 否 | 是（图像编码、RTC 关闭、子任务） |
| 图像编码 | 直接使用数据集图像 | BGR + 224×224 + JPEG |
| 延迟统计 | 仅本地 `infer` 时间 | mean/p95/max |
| GPU 占用 | 需要 | 不需要 |
| 适用场景 | 训练收敛检查 | 部署一致性检查 |

---

下一步：[09 依赖关系](./09-dependencies.md)
