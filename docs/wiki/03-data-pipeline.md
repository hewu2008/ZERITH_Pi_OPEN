# 03 数据管线

数据管线负责把 Zerith H1 Pro 的 HDF5 演示数据变成 OpenPI 可训练的 LeRobot 数据集，并生成训练/推理必须一致的归一化统计。

涉及文件：

- [scripts/convert_new.py](../../scripts/convert_new.py) — HDF5 → LeRobot
- [scripts/compute_norm_stats.py](../../scripts/compute_norm_stats.py) — 归一化统计
- [src/openpi/policies/zerith_joint_policy.py](../../src/openpi/policies/zerith_joint_policy.py) — Zerith 输入/输出映射
- [src/openpi/transforms.py](../../src/openpi/transforms.py) — Transform 原语
- [src/openpi/training/config.py](../../src/openpi/training/config.py) — 数据配置与 Transform 组合

---

## 1. HDF5 原始数据格式

转换器假设每个 episode 是一个 `.hdf5` 文件（递归扫描 `--raw_dir`）。

### 1.1 观测组

```text
/observation/
├── images/                    # 也可为 /observation/images/rs
│   ├── cam_high/color         # 头部/全局相机
│   ├── cam_left_wrist/color   # 左腕相机
│   ├── cam_right_wrist/color  # 右腕相机
│   └── depth                  # 存在时忽略
├── state/
│   ├── arm/position           # 14 维：左臂 7 + 右臂 7
│   ├── effector/position      # 2 维：左/右夹爪
│   ├── waist/position         # 3 维
│   ├── head/position          # 2 维：yaw / pitch
│   └── base/velocity          # 2 维：底盘速度（不执行）
└── ...
```

- 相机根路径优先 `/observation/images`，其次 `/observation/images/rs`。
- 支持的路径为 `/observation/images/<camera>/color` 或 `/observation/images/rs/<camera>/color`。
- 图像可能是普通数据集，也可能是压缩字节数据集；两种都由 `_load_images_for_camera` 处理。
- 图像统一 resize 到 **640×480**（`_resize_if_needed`），忽略 `depth`。
- episode 的任务描述从 `ep.attrs["task_name"]` 读取，作为 LeRobot 的 `task`。

### 1.2 23 维状态/动作布局

`convert_new.py` 的 `_build_state_and_action` 按以下顺序拼接：

| 维度 | 组成 | 来源 | 是否 delta |
| --- | --- | --- | --- |
| 0–6 | 左臂 7 关节 | `state/arm/position[:, :7]` | 是 |
| 7 | 左夹爪 | `state/effector/position[:, :-1]` | 否 |
| 8–14 | 右臂 7 关节 | `state/arm/position[:, 7:]` | 是 |
| 15 | 右夹爪 | `state/effector/position[:, -1:]` | 否 |
| 16–18 | 腰部 3 关节 | `state/waist/position` | 是 |
| 19–20 | 头部 yaw/pitch | `state/head/position` | 是 |
| 21–22 | 底盘速度 2 维 | `state/base/velocity` | 否（推理时置零） |

动作构造：

```text
action[t] = state[t+1]，最后一帧重复 action[T-1] = state[T-1]
```

因此 `action` 与 `state` 同为 23 维，且 shape 一致，便于 Flow Matching 直接使用。

> 注意：`action` 是**绝对状态**；训练时 `DeltaActions` 才会按 mask 转成部分维度增量。

### 1.3 LeRobot 输出结构

```text
$LEROBOT_HOME/<repo_id>/
├── data/                      # parquet 分片
├── videos/                    # 视频模式下的相机流（或 images/ 图像模式）
└── meta/
    ├── info.json              # fps=30, robot_type=mobile_arx, features
    ├── episodes.jsonl
    ├── tasks.jsonl
    └── subtasks.jsonl         # 若数据带子任务
```

特征定义（创建空数据集时）：

| Feature | dtype | shape | 说明 |
| --- | --- | --- | --- |
| `observation.state` | `float32` | `(23,)` | 关节状态 |
| `action` | `float32` | `(23,)` | 下一帧状态 |
| `observation.images.cam_high` | `video` 或 `image` | `(3, 480, 640)` | 头部相机 |
| `observation.images.cam_left_wrist` | 同上 | `(3, 480, 640)` | 左腕 |
| `observation.images.cam_right_wrist` | 同上 | `(3, 480, 640)` | 右腕 |

- `fps = 30`，`robot_type = "mobile_arx"`。
- `--mode` 控制 `image`（parquet 图像，**默认**）或 `video`；推荐 `image` 以便逐帧训练。
- 图像写入使用 `AsyncImageWriter`，`populate_dataset` 固定 `num_processes=64, num_threads=8`。

### 1.4 CLI

```bash
uv run scripts/convert_new.py \
  --raw_dir /path/to/hdf5_root \
  --repo_id zerith/part_sorting_20261027_subtask \
  [--raw_repo_id ...] \
  [--task "Sort all objects into the correct bins using both arms"] \
  [--episodes 0 1 2] \
  [--mode image|video] \
  [--push_to_hub] \
  [--is_mobile] \
  [--error_log_path ./convert_errors.log]
```

行为要点：

1. 目标 `repo_id` 已存在时会**递归删除**后重建（不支持断点续转）。
2. 转换结束会调用 `dataset.consolidate(run_compute_stats=False)`；异常 episode 被跳过并写入错误日志。
3. 实际 prompt 来自每个 episode 的 `task_name` 属性；`--task` 参数会传给 `populate_dataset`，但当前实现不用它覆盖 episode 文本。

---

## 2. 归一化统计

### 2.1 统计对象

`compute_norm_stats.py` 只统计 **`state`** 与 **`actions`**（不含图像、不含子任务文本）。流程：

1. 通过 `config_name` 找到 `TrainConfig` 与 `DataConfig`。
2. 用 `create_dataset` 构建数据集，应用 `repack_transforms.inputs` 与 `data_transforms.inputs`（即 Repack、MobilearxInputs、DeltaActions、ZeroDims），保证统计口径与训练一致。
3. 采样（默认随机 25% 帧，`shuffle=True`，batch=8）用 `RunningStats` 累积。
4. 写入 `assets/<asset_id>/norm_stats.json`。

### 2.2 `norm_stats.json` 结构

```json
{
  "state": {
    "mean": [23 floats],
    "std": [23 floats],
    "q01": [23 floats],
    "q99": [23 floats]
  },
  "actions": {
    "mean": [23 floats],
    "std": [23 floats],
    "q01": [23 floats],
    "q99": [23 floats]
  }
}
```

- **z-score**：`(x - mean) / (std + 1e-6)`。
- **quantile**（pi0.5 默认）：`(x - q01) / (q99 - q01 + 1e-6) * 2 - 1`。
- 具体使用哪种由 `DataConfig` 决定：`use_quantile_norm = model_type != PI0`，因此 `pi0_zerith` 使用 z-score，三个 pi0.5 配置使用 quantile。

### 2.3 CLI

```bash
uv run scripts/compute_norm_stats.py --config_name pi05_zerith [--max_frames 200000]
```

> 统计结果会落在 `config.assets_base_dir/asset_id/norm_stats.json`。如果训练时提示找不到 norm stats，先检查 `DataConfig.asset_id` 与实际目录是否一致，或执行 `scripts/visualize_norm_stats.py` 检查内容。

---

## 3. Zerith 输入/输出 Transform

文件：[zerith_joint_policy.py](../../src/openpi/policies/zerith_joint_policy.py)

### 3.1 `MobilearxInputs`

| 步骤 | 行为 |
| --- | --- |
| 图像收集 | 读取 `cam_high`、`cam_left_wrist`、`cam_right_wrist` |
| 格式转换 | CHW → HWC、`float32` → `uint8`（若输入是 0–255 浮点） |
| 键重命名 | `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb` |
| 掩码 | 为每个图像键生成 `image_mask`（布尔） |
| 状态 | `pad_to_dim(state, model.action_dim)` → 32 维 |
| 透传 | `prompt`、`subtask`、`subtask_class` |

### 3.2 `MobilearxOutputs`

- 从模型输出 `actions`（`[B, H, 32]`）截取前 **23** 维；
- 返回 `{"actions": ...}`，供机器人执行 `actions[:-2]`（21 个电机）。

---

## 4. 通用 Transform 原语

[transforms.py](../../src/openpi/transforms.py) 采用可组合的 `DataTransform` 设计，常用的有：

| Transform | 作用 |
| --- | --- |
| `RepackTransform` / `RepackTransformWithFallback` | 按键名重映射；后者支持缺少首选键时回退到备选键（兼容不同 LeRobot 元数据） |
| `Group` / `Composite` | 将多个 Transform 顺序或分组组合 |
| `DeltaActions` | 按布尔 mask 把绝对动作转为相对动作 |
| `AbsoluteActions` | 逆操作，把相对动作累加回绝对动作 |
| `ZeroDims` | 将指定维度置零（Zerith 的底盘 21/22） |
| `Normalize` / `Unnormalize` | 支持 z-score 与 quantile 两种模式，读取 `norm_stats` |
| `ResizeImages` | 用 `resize_with_pad` 把图像统一到 `224×224` |
| `InjectDefaultPrompt` | 没有 prompt 时注入默认任务描述 |
| `TokenizePrompt` | pi0/pi0.5 的 PaliGemma tokenization（可含状态、子任务） |
| `TokenizeFASTInputs` / `ExtractFASTActions` | FAST 离散动作的 tokenization/反解 |
| `PadStatesAndActions` | 将 state/action 补齐到 `action_dim`（32） |
| `PromptFromLeRobotTask` | 从 LeRobot 的 `task` 生成 `prompt` |
| `SubtaskFromIndex` | 用 `subtask_index` 从 `meta/subtasks.jsonl` 取子任务文本 |

Zerith 配置还定义了：

- **多相机 Repack**：把三路 `observation.images.*` 重命名成模型键。
- **Delta mask**：`make_bool_mask(7, -1, 7, -1, 5, -2)`，即 `0-6, 8-14, 16-20` 为 delta，`7, 15, 21-22` 为绝对。
- **`TokenizePi05SubtaskInputs`**：pi0.5 子任务训练时的 token 布局（`Task: … State: …; Subtask: …; Action: …`）。

### 4.1 训练时的组合顺序

```text
repack.inputs
  → data.inputs（MobilearxInputs → DeltaActions → ZeroDims）
  → Normalize
  → model.inputs（InjectDefaultPrompt → ResizeImages → Tokenize* → PadStatesAndActions）
```

### 4.2 推理时的组合顺序

```text
repack.inputs
  → data.inputs（MobilearxInputs → DeltaActions(no-op) → ZeroDims(no-op)）
  → Normalize(state)
  → model.inputs（InjectDefaultPrompt → ResizeImages → Tokenize* → PadStatesAndActions）
  → model.sample_actions
  → Unnormalize(actions)
  → data.outputs（ZeroDims → AbsoluteActions → MobilearxOutputs）
```

> `data.outputs` 的顺序是 `ZeroDims → AbsoluteActions → MobilearxOutputs`；`MobilearxOutputs` 最后截取 23 维。

---

## 5. 子任务数据

### 5.1 标注格式

- 子任务定义位于数据集 `meta/subtasks.jsonl`，每行一个 JSON，含 `subtask_index` 与文本。
- 数据集需要额外列 `subtask_index`（**从 0 连续编号**）。
- `-1` 表示该帧无子任务标注；`SubtaskFromIndex` 会跳过或填默认值（取决于配置）。

### 5.2 类别表

仓库提供 [assets/subtasks.jsonl](../../assets/subtasks.jsonl)，共 **13 类**，例如：

```text
0  move both arms back to the home position
1  place the L-shaped bent pipe in the right bin
2  place the door handle in the left bin
3  place the silver pipe in the front bin
...
12 use the right arm to pick up the silver pipe in the right area of the blue bin
```

> **一致性要求**：训练时的 `model.subtask_classes`（来自数据集 `meta/subtasks.jsonl` 的顺序）必须与机器人/服务端使用的 `assets/subtasks.jsonl` 完全一致，否则分类头输出的索引会错位。

### 5.3 数据配置中的开关

| 配置 | `subtask_enabled` | `subtask_loss_weight` | `subtask_classifier_enabled` | 说明 |
| --- | --- | --- | --- | --- |
| `pi0_zerith` | 否 | — | 否 | 纯动作 |
| `pi05_zerith` | 否 | — | 否 | 纯动作 |
| `pi05_zerith_subtask` | 是 | 0.1 | 否 | 文本生成子任务 |
| `pi05_zerith_subtask_cls` | 是 | 0.1 | 是 | 分类头 + 文本标签 |

---

## 6. 数据质检工具

| 工具 | 用途 |
| --- | --- |
| [inspect_sanity_dump.py](../../scripts/inspect_sanity_dump.py) | 打印 `state/actions/raw_state/raw_actions` 的 shape/dtype，检测 NaN/Inf、零方差维度 |
| [visualize_norm_stats.py](../../scripts/visualize_norm_stats.py) | 绘制 23 维的 mean/std/q01/q99，发现异常量纲或统计偏置 |
| [merge_bin_box_datasets.py](../../scripts/merge_bin_box_datasets.py) | 合并多个 bin/box 数据集，统一动作空间与任务标签 |

### 常见数据问题

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| 某维度归一化后爆炸 | 该维方差接近 0（如底盘） | 用 `ZeroDims` 排除，或检查 `visualize_norm_stats` |
| 训练 loss 正常但机器人抖动 | norm stats 与训练不一致 / 图像通道顺序不一致 | 重新计算统计；检查 `cv2` BGR 与训练 RGB |
| 子任务分类总是同一类 | `subtasks.jsonl` 顺序与训练不一致 | 对齐两份 JSONL |
| 动作维度不匹配 | 数据集不是 23 维 / 模型 `action_dim` 不是 32 | 检查转换脚本与配置 |

---

下一步：[04 模型体系](./04-models.md)
