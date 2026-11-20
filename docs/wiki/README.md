# Zerith H1 Pro OpenPI Adapter — Code Wiki

本 Wiki 面向需要**理解、二次开发或部署**本仓库的工程师，系统梳理了项目的整体架构、模块职责、关键类与函数、依赖关系以及完整运行方式。

> 仓库定位：在 Physical Intelligence 的开源 **OpenPI（pi0 / pi0.5）** 训练栈基础上，增加 **Zerith H1 Pro** 机器人的数据转换、策略微调、WebSocket 服务与机器人侧闭环推理代码。

---

## 文档地图

| 文档 | 内容 | 适合谁 |
| --- | --- | --- |
| [01 项目概览](./01-overview.md) | 项目定位、核心能力、术语表、仓库结构 | 所有人 |
| [02 系统架构](./02-architecture.md) | 训练/服务/机器人三段式架构、数据流、部署拓扑、模块分层 | 架构与集成 |
| [03 数据管线](./03-data-pipeline.md) | HDF5 → LeRobot 转换、23 维状态/动作布局、归一化统计、Transform 流水线 | 数据/训练 |
| [04 模型体系](./04-models.md) | Pi0 / Pi0.5 / Pi0-FAST、PaliGemma、SigLIP、LoRA、Tokenizer、RTC 模型层 | 算法/模型 |
| [05 训练系统](./05-training.md) | TrainConfig、四个内置配置、train.py 流程、优化器、分片、Checkpoint、权重加载 | 训练 |
| [06 策略服务与客户端](./06-serving-and-client.md) | Policy / RTCPolicy、remote_infer_server、WebSocket 协议、openpi-client | 服务端/联调 |
| [07 机器人运行时](./07-robot-runtime.md) | test_pi0 控制循环、Real_Env SDK、异步推理、RTC 调度、Home 守卫 | 机器人部署 |
| [08 脚本与工具](./08-scripts-and-tools.md) | `scripts/` 与 `tools/` 全量说明及命令模板 | 日常使用 |
| [09 依赖关系](./09-dependencies.md) | 模块依赖图、调用链、第三方依赖、路径约定 | 维护/排障 |
| [10 运行与排障](./10-running.md) | 安装、转换、归一化、训练、评估、服务、机器人运行、常见问题 | 上手/运维 |
| [11 代码速查](./11-code-reference.md) | 关键类/函数/配置项/CLI 参数索引 | 开发查阅 |

---

## 一分钟了解项目

```text
Zerith HDF5 演示数据
        │  scripts/convert_new.py
        ▼
LeRobot 数据集（3 路相机 + 23 维 state/action）
        │  scripts/compute_norm_stats.py
        ▼
norm_stats.json（assets/<asset_id>/）
        │  scripts/train.py <config_name>
        ▼
LoRA / pi0.5 / 子任务分类器 Checkpoint
        │  scripts/remote_infer_server.py（GPU 服务器）
        ▼
WebSocket（msgpack-numpy）
        │  robot_infer/scripts/test_pi0.py（机器人端 30Hz）
        ▼
Zerith H1 Pro 关节控制（RTC / 动作平滑 / Home 守卫）
```

## 快速启动

```bash
# 1) 安装（仓库根目录）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 2) 数据转换
uv run scripts/convert_new.py --raw_dir /path/to/hdf5 --repo_id zerith/my_dataset

# 3) 计算归一化统计（需先在 config.py 中把 repo_id/asset_id 指向数据集）
uv run scripts/compute_norm_stats.py --config_name pi05_zerith

# 4) 训练
uv run scripts/train.py pi05_zerith --exp_name my_run --overwrite

# 5) 启动策略服务
uv run scripts/remote_infer_server.py \
  --policy.config pi05_zerith \
  --policy.dir openpi_checkpoints/pi05_zerith/my_run/59999 \
  --port 55555

# 6) 机器人端（在机器人 zerith 环境，仅需 robot_infer/）
python robot_infer/scripts/test_pi0.py \
  --host <server-ip> --port 55555 \
  --prompt "Sort all objects into the correct bins using both arms" \
  --init_hdf5 /path/to/demo.hdf5 \
  --subtask.mode classifier --subtask.jsonl assets/subtasks.jsonl
```

## 阅读约定

- 文中的路径均相对于仓库根目录 `/workspace`。
- 配置名（`pi0_zerith`、`pi05_zerith`、`pi05_zerith_subtask`、`pi05_zerith_subtask_cls`）对应 [config.py](../../src/openpi/training/config.py) 中的 `_CONFIGS`。
- Checkpoint 路径统一为：`<checkpoint_base_dir>/<config.name>/<exp_name>/<step>/`，四个 Zerith 配置的 `checkpoint_base_dir` 均为 `./openpi_checkpoints`。
- 资源路径统一为：`<assets_base_dir>/<asset_id>/norm_stats.json`，默认 `./assets/<asset_id>/norm_stats.json`。
- 数据集路径统一为：`<$LEROBOT_HOME>/<repo_id>`。

## 重要提示

- README 中提到的 `scripts/serve_policy.py` 在当前仓库中**不存在**；实际部署使用的是 [scripts/remote_infer_server.py](../../scripts/remote_infer_server.py)，通用 OpenPI 服务端实现位于 [src/openpi/serving/websocket_policy_server.py](../../src/openpi/serving/websocket_policy_server.py)。
- 仓库中的 `tools/*.sh` 带有作者本机的绝对路径（如 `/data/4T-2/...`、GPU 编号、conda 路径），复用前必须按你的环境修改。
- `robot_infer/lib/` 内含 Zerith SDK 1.3.4 的 `.so`/`.pyi` 绑定，机器人端需保证 SDK 版本匹配。
