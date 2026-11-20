# 10 运行与排障

本页给出从零到机器人闭环的完整操作顺序，并列出常见问题。所有命令默认在仓库根目录执行。

---

## 1. 环境准备

### 1.1 训练/服务机

- Ubuntu 22.04
- Python 3.11
- NVIDIA GPU + CUDA 12（LoRA 微调建议 ≥24GB 显存）
- `git`、`git-lfs`、`uv`

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖（跳过 LeRobot 子模块的 LFS 大文件）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

可选缓存目录：

```bash
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_datasets
export LEROBOT_HOME=/path/to/lerobot_home
export TMPDIR=/path/to/tmp
```

> `LEROBOT_HOME` 很关键：转换后的数据集写入 `<LEROBOT_HOME>/<repo_id>`。

### 1.2 机器人机

- Zerith H1 Pro 标准机器人
- Zerith H1 SDK + 相机 SDK（仓库绑定 **1.3.4**）
- 三路 RGB 相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist`
- 能访问策略服务器端口

部署时**只需**：

```text
robot_infer/                 # 拷贝到机器人
packages/openpi-client/      # pip install -e .
```

机器人环境不需要安装 JAX/CUDA/torch/lerobot。

```bash
cd /path/to/openpi-client
pip install -e .
```

> SDK 1.3.6+ 的函数名有细微变化；若导入失败，按 [zerith_public_sdk](https://github.com/inFpZero/zerith_public_sdk) 对应版本替换/重编 `robot_infer/lib/`。

---

## 2. 数据准备

### 2.1 HDF5 目录

```text
/path/to/hdf5_root/
├── episode_0000.hdf5
├── episode_0001.hdf5
└── ...
```

每个文件需包含：

```text
/observation/state/{arm,effector,waist,head,base}/...
/observation/images/{cam_high,cam_left_wrist,cam_right_wrist}/color
```

### 2.2 转换为 LeRobot

```bash
uv run scripts/convert_new.py \
  --raw_dir /path/to/hdf5_root \
  --repo_id zerith/part_sorting_v1 \
  --mode image
```

- `repo_id` 必须与后续 `config.repo_id` 一致；
- 若目录已存在会被删除重建；
- 检查输出：`$LEROBOT_HOME/zerith/part_sorting_v1/meta/info.json`。

### 2.3 配置数据集

编辑 [config.py](../../src/openpi/training/config.py) 中对应 `TrainConfig` 的 `data`：

```python
data=LeRobotZerithJointDataConfig(
    repo_id="zerith/part_sorting_v1",
    asset_id="zerith/part_sorting_v1",
    ...
)
```

四个 Zerith 配置默认指向作者的数据集；训练自有数据时必须改 `repo_id`/`asset_id`。

### 2.4 计算归一化统计

```bash
uv run scripts/compute_norm_stats.py --config_name pi05_zerith
```

产物：`assets/<asset_id>/norm_stats.json`。

```bash
# 可选：可视化检查
python scripts/visualize_norm_stats.py assets/zerith/part_sorting_v1/norm_stats.json
```

---

## 3. 训练

### 3.1 单步训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_zerith \
  --project_name openpi_zerith \
  --exp_name my_first_run \
  --overwrite
```

- `--resume`：从 `openpi_checkpoints/pi05_zerith/my_first_run/` 恢复；
- `--no-wandb_enabled`：关闭 wandb；
- 双卡：`--fsdp_devices 2` 或设置 `CUDA_VISIBLE_DEVICES=0,1`（`fsdp_devices` 需与设备数匹配）。

### 3.2 一键脚本

```bash
RAW_DIR=/path/to/hdf5_root \
REPO_ID=zerith/part_sorting_v1 \
CONFIG_NAME=pi05_zerith \
EXP_NAME=my_first_run \
bash train.sh
```

`train.sh` 会依次执行转换 → 统计 → 训练，日志在 `logs/`。

### 3.3 训练前自检

```bash
uv run scripts/train.py pi05_zerith --exp_name sanity --sanity_mode --overwrite --no-wandb_enabled
python scripts/inspect_sanity_dump.py ./sanity_dump
```

关注：

- `state/actions` 是否有 NaN/Inf；
- 是否有 `std=0` 的维度（归一化会放大 1e6 倍）；
- 图像是否正常。

---

## 4. 开环评估

### 4.1 本地 Checkpoint

```bash
uv run scripts/open_loop_eval.py \
  --config_name pi05_zerith_subtask \
  --checkpoint_dir openpi_checkpoints/pi05_zerith_subtask/my_run/59999 \
  --data_path /path/to/dataset/hewu2008/part_sorting_20261027_subtask \
  --traj_ids 0 1 2 3 \
  --max_infer_time 240 \
  --action_horizon 30 \
  --save_plot_path ./open_loop_test/my_run/
```

### 4.2 已启动的服务

```bash
uv run scripts/open_loop_eval_remote.py \
  --host 127.0.0.1 --port 55555 \
  --data_path /path/to/dataset \
  --traj_ids 0 1 2 3 \
  --action_horizon 30 \
  --save_plot_path ./open_loop_test/remote/
```

输出：每关节 `state/gt/pred` 曲线、MSE/MAE、子任务准确率、服务端延迟。

---

## 5. 启动策略服务

```bash
uv run scripts/remote_infer_server.py \
  --policy.config pi05_zerith_subtask_cls \
  --policy.dir openpi_checkpoints/pi05_zerith_subtask_cls/my_run/15999 \
  --default-prompt "Sort all objects into the correct bins using both arms" \
  --port 55555 \
  --subtask.mode classifier \
  --subtask.jsonl assets/subtasks.jsonl \
  --rtc.s_min 18
```

要点：

- 服务监听 `0.0.0.0:55555`；
- 启动时预热 plain + guided JIT；
- RTC 默认启用（`identity`）；`--rtc.no-enabled` 关闭；
- 子任务模式：`--subtask.mode classifier|generate|off`；
- Checkpoint 目录必须包含 `params/` 与 `assets/<asset_id>/norm_stats.json`。

> README 中写的 `scripts/serve_policy.py` 在当前仓库不存在；README 的 `remote_infer_server.py policy:checkpoint --policy.config=...` 是上游旧写法，实际使用上面的 tyro 参数（`--policy.config` 等），且当前 `_CONFIGS` 中没有名为 `test` 的配置。

---

## 6. 机器人端运行

```bash
python robot_infer/scripts/test_pi0.py \
  --host <server-ip> --port 55555 \
  --prompt "Sort all objects into the correct bins using both arms" \
  --init_hdf5 /path/to/demo.hdf5 \
  --subtask.mode classifier \
  --subtask.jsonl assets/subtasks.jsonl \
  --rtc --s_min 18 \
  --control_freq 30 \
  --num_steps 9000
```

前 6 个步骤：

1. 连接机器人并 `robot_init`；
2. `move_to_init_pose()`；
3. 连接策略服务器并校验 RTC metadata；
4. 预热推理 10 次；
5. 可选移动到 `--init_hdf5` 首帧姿态；
6. 停在 `pdb`，人工确认场地后继续。

运行中：

- 每 30Hz 取一个动作；
- RTC 模式按延迟调度，非 RTC 模式用 `ActionSmooth`；
- 夹爪回滞、头部锁定；
- Home 子任务连续 3 块 → 自动回位并重置队列。

---

## 7. 完整流程清单

```text
[ ] 1. uv sync / uv pip install -e .
[ ] 2. export LEROBOT_HOME / HF_HOME
[ ] 3. convert_new.py 生成数据集
[ ] 4. 修改 config.py 的 repo_id / asset_id
[ ] 5. compute_norm_stats.py
[ ] 6. visualize_norm_stats.py 检查统计
[ ] 7. train.py（先 sanity 再正式）
[ ] 8. open_loop_eval.py 检查 MSE/MAE
[ ] 9. remote_infer_server.py 启动服务
[ ] 10. open_loop_eval_remote.py 验证服务一致性
[ ] 11. 机器人端 test_pi0.py
[ ] 12. 现场低速验证 → 全速运行
```

---

## 8. 常见问题

### 8.1 `Norm stats not found`

**原因**：`config.data.asset_id` 指向的 `assets/<asset_id>/norm_stats.json` 不存在。

**处理**：

```bash
uv run scripts/compute_norm_stats.py --config_name <your_config>
ls assets/<asset_id>/norm_stats.json
```

注意：推理时 `create_trained_policy` 优先读取 **Checkpoint 内部**的 `assets/`；若手动替换 Checkpoint，需一并拷贝 `assets/`。

### 8.2 `Dataset not found`

- 检查 `LEROBOT_HOME` 是否正确；
- 检查 `<LEROBOT_HOME>/<repo_id>` 是否存在 `meta/info.json`；
- 当前 `data_loader.py` 固定离线加载（`local_files_only=True`），不会自动下载。

### 8.3 客户端连不上服务

- 确认服务进程与端口：`ss -lntp | grep 55555`；
- 服务器防火墙/安全组放行；
- 机器人能访问服务器 IP；
- 若服务启动后立即退出，查看 traceback（Checkpoint 路径、norm stats、CUDA OOM）。

### 8.4 CUDA OOM

- 降低 `batch_size`；
- 降低 `XLA_PYTHON_CLIENT_MEM_FRACTION`（注意不是越低越好，过低会碎片化）；
- 减少可见 GPU；
- pi0.5 + 子任务生成模式在推理时显存/延迟更高，可改用 `classifier`。

### 8.5 `Action dimension mismatch`

- 数据集必须是 23 维；
- 模型 `action_dim` 为 32，由 `PadStatesAndActions` 补齐；
- 若修改了 HDF5 布局，必须同步修改：
  - `convert_new.py` 的拼接顺序；
  - `zerith_joint_policy.py` 的 `MobilearxInputs/Outputs`；
  - `config.py` 的 Delta mask / `ZeroDims`。

### 8.6 子任务分类总错位

- 训练数据 `meta/subtasks.jsonl` 的 `subtask_index` 是否从 0 连续；
- 服务端/机器人端 `--subtask.jsonl` 是否与训练顺序完全一致；
- `num_subtask_classes` 是否等于类别数（13）。

### 8.7 RTC 启动失败

- 服务端未启用 RTC（`--rtc.no-enabled`）；
- metadata 缺少 `rtc.version`；
- 客户端与服务端协议版本不一致；
- `--s_min` 违反 `d <= s <= H - d`，会导致权重退化/报错。

### 8.8 机器人动作抖动/跳变

- RTC 与 `ActionSmooth` 不要同时启用（代码已互斥：RTC 模式不创建 ActionSmooth）；
- Home 回位后必须 `worker.drain()` + `reset()`（代码已做）；
- 检查 norm stats 与训练是否一致；
- 检查图像通道顺序（训练 RGB，机器人 JPEG 前转 BGR）；
- 夹爪回滞阈值可能需要按硬件调整。

### 8.9 首块超时（30s）

- 服务端首次请求触发 JIT 编译；确认 `--rtc` 时已执行 `warmup`；
- GPU 被占用 / 显存不足；
- 网络带宽不足（三路 224×224 JPEG 通常很小）；
- `generate` 子任务模式会显著增加首块延迟，可先用 `classifier` 或 `off`。

### 8.10 转换脚本报 `h5py` 缺失

```bash
uv pip install h5py
```

### 8.11 图像键找不到

支持的形式：

```text
cam_high
rs/cam_high
rs.cam_high
rs_cam_high
images["rs"]["cam_high"]
```

若相机名不同，使用 `--camera_names` 或修改 `real_env_sdk.camera_aliases`。

### 8.12 wandb 初始化失败

```bash
uv run scripts/train.py ... --no-wandb_enabled
```

或先 `wandb login`。

### 8.13 训练无法 resume

- `--resume` 需要 `openpi_checkpoints/<config>/<exp_name>/` 下已有 Checkpoint；
- 若无 Checkpoint，`initialize_checkpoint_dir` 会自动降级为从头训练；
- `--overwrite` 与 `--resume` 互斥。

---

## 9. 安全须知

- 首次运行务必**低速、空场地、手按急停**；
- `--init_hdf5` 使用与任务同分布的演示首帧，确保初始姿态合理；
- 不要跳过第一块动作的同步等待；
- 机器人端看门狗只告警，不会自动停机；建议外接硬件急停；
- 修改 `pin_head`、`GripperHysteresis`、RTC 参数后先做空跑验证。

---

下一步：[11 代码速查](./11-code-reference.md)
