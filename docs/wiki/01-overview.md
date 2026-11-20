# 01 项目概览

## 1. 项目定位

本仓库是 **OpenPI（pi0 / pi0.5）在 Zerith H1 Pro 标准机器人上的适配层**。它保留了上游 OpenPI 的 JAX/Flax 训练栈、模型实现、Transform 体系和 WebSocket 推理框架，并新增了 Zerith 专有的数据格式、训练配置、子任务预测、实时分块（RTC）服务端与机器人端控制逻辑。

一句话概括：

> **把 Zerith H1 Pro 的三路相机 + 23 维关节数据，转换成 OpenPI 可训练的 LeRobot 数据集；微调 pi0 / pi0.5 策略；通过 WebSocket 在 GPU 工作站上提供动作块推理；在机器人端以 30Hz 闭环执行。**

## 2. 核心能力

| 能力 | 入口 | 说明 |
| --- | --- | --- |
| HDF5 → LeRobot 数据转换 | [scripts/convert_new.py](../../scripts/convert_new.py) | 递归扫描 `.hdf5`，组装 23 维 `state/action`，统一三路相机到 `640×480` |
| 归一化统计 | [scripts/compute_norm_stats.py](../../scripts/compute_norm_stats.py) | 统计 `state/actions` 的 mean/std/q01/q99，写入 `assets/<asset_id>/norm_stats.json` |
| pi0 / pi0.5 LoRA 微调 | [scripts/train.py](../../scripts/train.py) + [config.py](../../src/openpi/training/config.py) | 冻结基座、训练 LoRA 适配器与 action expert |
| 高层子任务预测 | [pi0.py](../../src/openpi/models/pi0.py)、[tokenizer.py](../../src/openpi/models/tokenizer.py) | 自回归文本生成（generate）或分类头（classifier）两种推理路径 |
| 实时分块 RTC | [rtc_policy.py](../../src/openpi/policies/rtc_policy.py)、[rtc_client.py](../../robot_infer/utils/rtc_client.py) | 服务端 PiGDM 引导采样 + 客户端调度，降低动作切换抖动 |
| WebSocket 策略服务 | [remote_infer_server.py](../../scripts/remote_infer_server.py) | 接收 JPEG 图像与 23 维状态，返回 `action_horizon × 23` 动作块 |
| 机器人闭环推理 | [test_pi0.py](../../robot_infer/scripts/test_pi0.py) | 相机采集、动作平滑、夹爪回滞、头部锁定、Home 守卫、30Hz 控制 |
| 开环评估 | [open_loop_eval.py](../../scripts/open_loop_eval.py)、[open_loop_eval_remote.py](../../scripts/open_loop_eval_remote.py) | 在训练集上计算动作 MSE/MAE 与子任务精确匹配率，并绘制轨迹 |
| 数据体检 | [inspect_sanity_dump.py](../../scripts/inspect_sanity_dump.py)、[visualize_norm_stats.py](../../scripts/visualize_norm_stats.py) | 检查 NaN/Inf、零方差维度与归一化统计异常 |

## 3. 仓库结构

```text
/workspace
├── train.sh                     # 转换 → 归一化 → 训练 一键脚本
├── pyproject.toml               # Python 3.11；uv workspace（packages/*）
├── README.md                    # 面向使用者的说明
├── scripts/                     # 可执行入口（转换/统计/训练/服务/评估）
│   ├── convert_new.py
│   ├── compute_norm_stats.py
│   ├── train.py
│   ├── remote_infer_server.py
│   ├── open_loop_eval.py
│   ├── open_loop_eval_remote.py
│   ├── merge_bin_box_datasets.py
│   ├── inspect_sanity_dump.py
│   └── visualize_norm_stats.py
├── src/openpi/                  # OpenPI 核心库（JAX/Flax）
│   ├── models/                  # Pi0/Pi0.5/FAST、Gemma、SigLIP、ViT、LoRA、Tokenizer、RTC
│   ├── models_pytorch/          # PyTorch 参考实现（非部署主路径）
│   ├── policies/                # Policy 封装、PolicyConfig、RTCPolicy、Zerith Transform
│   ├── training/                # TrainConfig、DataLoader、Optimizer、Checkpoint、Sharding
│   ├── serving/                 # 通用 WebSocket 策略服务器
│   ├── shared/                  # 归一化、下载、类型、图像工具、NNX 工具
│   └── transforms.py            # 数据处理 Transform 原语
├── robot_infer/                 # 仅需拷贝到机器人端的运行时代码
│   ├── scripts/test_pi0.py      # 机器人主控制循环
│   ├── utils/real_env_sdk.py    # Zerith H1 Pro 环境封装
│   ├── utils/async_infer.py     # 后台推理线程
│   ├── utils/rtc_client.py      # RTC 客户端调度器
│   ├── utils/subtask_home_guard.py
│   └── lib/                     # Zerith SDK 1.3.4 绑定（.so/.pyi/proto）
├── packages/
│   ├── openpi-client/           # 轻量 WebSocket 客户端与 runtime
│   └── lerobot/                 # 随仓库锁定的 LeRobot 依赖
├── assets/
│   └── subtasks.jsonl           # 子任务类别表（13 类）
├── tools/                       # 作者本机用的 .sh 命令模板
└── docs/wiki/                   # 本 Code Wiki
```

## 4. 四个内置训练配置

| 配置名 | 模型 | 数据 | 特点 |
| --- | --- | --- | --- |
| `pi0_zerith` | pi0（`pi05=False`） | `clear_the_bin_box_20261010_v1` | LoRA 2B + 300M action expert，30k steps，z-score 归一化 |
| `pi05_zerith` | pi0.5（`pi05=True`） | `part_sorting_20261027_clean` | 状态离散化进语言 token，action expert 使用 adaRMS，60k steps，quantile 归一化 |
| `pi05_zerith_subtask` | pi0.5 + 子任务预测 | `part_sorting_20261027_subtask` | 增加子任务文本交叉熵（α=0.1），支持 generate 推理 |
| `pi05_zerith_subtask_cls` | pi0.5 + 子任务分类头 | `part_sorting_20261027_subtask` | 新增 13 类分类头（权重 0.2），分类头学习率 ×100，从 60k 权重热启动 |

> 所有配置均从公开/本地 pi0、pi0.5 基座权重热启动，冻结基座 LLM，仅训练 LoRA、action expert（以及可选分类头）。

## 5. 术语表

| 术语 | 含义 |
| --- | --- |
| **pi0 / pi0.5** | Physical Intelligence 的视觉-语言-动作（VLA）策略；pi0.5 将状态离散为语言 token，并用 adaRMS 注入流匹配时间步 |
| **Action Expert** | pi0 的第二个 Gemma 专家（300M），专门处理动作 token 与去噪 |
| **PaliGemma** | 前缀 VLM：SigLIP 图像编码器 + Gemma 语言模型 |
| **Flow Matching** | 训练目标：学习从噪声到动作的常微分方程速度场，推理时用 Euler 积分 |
| **Action Horizon** | 一次推理预测的动作步数，Zerith 配置为 50 |
| **Prefix / Suffix** | 前缀 = 图像 + 语言（+ 状态 token）；后缀 = 动作 token + 时间步 |
| **Subtask** | 高层子任务文本（如 "place the door handle in the left bin"），pi0.5 可先预测再生成动作 |
| **RTC** | Real-Time Chunking，利用上一块剩余动作引导新动作块，减少块切换突变 |
| **LoRA** | 低秩适配器，仅训练少量参数即可微调 2B 基座 |
| **Norm Stats** | `state/actions` 的 mean/std/q01/q99，训练与推理必须一致 |
| **LeRobot** | HuggingFace 机器人数据集格式与工具库，本仓库将其作为数据容器 |
| **Asset ID** | 归一化统计在 `assets/` 下的子目录名，通常等于 `repo_id` |
| **ActionSmooth** | 机器人端的时间集成：把多个动作块按指数权重融合后执行 |

## 6. 硬件与软件要求

| 角色 | 要求 |
| --- | --- |
| 训练机 | Ubuntu 22.04、Python 3.11、NVIDIA GPU（CUDA 12），LoRA 微调建议 ≥24GB 显存 |
| 策略服务器 | NVIDIA GPU（README 实测 RTX 3090/4080/4090/5080 等），与机器人网络互通 |
| 机器人 | Zerith H1 Pro 标准机器人、Zerith H1 SDK + 相机 SDK（仓库绑定 1.3.4）、三路 RGB 相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist` |
| 机器人环境 | 内置 `zerith` conda 环境；仅需 `robot_infer/` 与 `openpi-client` |

## 7. 与上游 OpenPI 的差异

新增：

1. `scripts/convert_new.py`：Zerith HDF5 → LeRobot 转换器。
2. `src/openpi/policies/zerith_joint_policy.py`：Zerith 观测/动作映射（`MobilearxInputs/Outputs`）。
3. `src/openpi/training/config.py`：四个 Zerith 配置、多相机 Repack、Delta/ZeroDims 动作处理。
4. `scripts/remote_infer_server.py`：带 RTC、子任务、相机别名解码的部署服务端。
5. `robot_infer/`：机器人端完整闭环推理与 SDK 封装。
6. 子任务预测/分类：`Pi0Config`、`Pi0`、`PaligemmaTokenizer`、`Policy` 中的扩展。
7. 评估与数据体检脚本。

沿用：模型主体（Gemma/SigLIP/Pi0）、Flow Matching、LoRA、NNX/JAX 训练循环、Orbax Checkpoint、msgpack-numpy WebSocket 协议。

---

下一步：[02 系统架构](./02-architecture.md)
