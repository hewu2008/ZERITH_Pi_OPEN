# 11 代码速查

本页是开发时的快速索引：符号、文件、职责、关键签名。

---

## 1. 入口与脚本

| 入口 | 文件 | 说明 |
| --- | --- | --- |
| 数据转换 | [convert_new.py](../../scripts/convert_new.py) | `port_aloha`（tyro）、`_build_state_and_action`、`_load_images_for_camera` |
| 归一化 | [compute_norm_stats.py](../../scripts/compute_norm_stats.py) | `main(config_name, max_frames)` → `assets/<asset_id>/norm_stats.json` |
| 训练 | [train.py](../../scripts/train.py) | `main(config_name, ...)`、`init_train_state`、`train_step` |
| 服务 | [remote_infer_server.py](../../scripts/remote_infer_server.py) | `Args`、`create_policy`、`_build_observation`、`websocket_handler` |
| 本地评估 | [open_loop_eval.py](../../scripts/open_loop_eval.py) | `evaluate_single_trajectory`、`build_observation`、`plot_trajectory_results` |
| 远程评估 | [open_loop_eval_remote.py](../../scripts/open_loop_eval_remote.py) | `encode_image`、`build_remote_observation` |
| 合并数据集 | [merge_bin_box_datasets.py](../../scripts/merge_bin_box_datasets.py) | 重写 `index/episode_index/task_index` |
| 数据体检 | [inspect_sanity_dump.py](../../scripts/inspect_sanity_dump.py) | `_summary`、`_check_pair` |
| 统计可视化 | [visualize_norm_stats.py](../../scripts/visualize_norm_stats.py) | `_plot_key`、`main` |
| 机器人主循环 | [test_pi0.py](../../robot_infer/scripts/test_pi0.py) | `main`、`control_step`、`ActionSmooth`、`GripperHysteresis` |

---

## 2. 训练配置

文件：[config.py](../../src/openpi/training/config.py)

| 配置 | 模型关键项 | 数据 | 优化 |
| --- | --- | --- | --- |
| `pi0_zerith` | `pi05=False`, `gemma_2b_lora`, `gemma_300m`, H=50 | `clear_the_bin_box_20261010_v1` | lr 1e-5, 30k, bs 8 |
| `pi05_zerith` | `pi05=True`, `discrete_state_input=True`, H=50 | `part_sorting_20261027_clean` | lr 2.5e-5, 60k, bs 8 |
| `pi05_zerith_subtask` | + `train/sample_subtask_prediction`, α=0.1 | `part_sorting_20261027_subtask` | lr 5e-5, 60k, bs 16 |
| `pi05_zerith_subtask_cls` | + `subtask_classification`, 13 类, β=0.2 | 同上 | head lr ×100, 16k, bs 8 |

### 配置相关类

| 符号 | 作用 |
| --- | --- |
| `DataConfig` | 数据集、assets、Transform、quantile 开关 |
| `LeRobotZerithJointDataConfig` | Zerith 数据工厂：Repack + `MobilearxInputs` + delta/zero + tokenize |
| `TrainConfig` | 训练超参、路径、权重加载、EMA、FSDP |
| `_CONFIGS` | 名称 → `TrainConfig` 映射 |
| `get_config(name)` | 按名取配置（不存在会断言） |

---

## 3. Transform

文件：[transforms.py](../../src/openpi/transforms.py)、[zerith_joint_policy.py](../../src/openpi/policies/zerith_joint_policy.py)

| 符号 | 输入 | 输出/行为 |
| --- | --- | --- |
| `RepackTransform` | 任意 dict | 键重命名 |
| `RepackTransformWithFallback` | 任意 dict | 首选键缺失时回退 |
| `Group` / `Composite` | — | 组合 Transform |
| `MobilearxInputs` | `state`, `images`, `prompt` | `state` 补 32、三路图像 → `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb` |
| `MobilearxOutputs` | `actions` | 截取 `[..., :23]` |
| `DeltaActions` | `state`, `actions` | 按 mask 转相对动作 |
| `AbsoluteActions` | `state`, `actions` | 累加回绝对动作 |
| `ZeroDims` | `actions` | 指定维度置零 |
| `Normalize` / `Unnormalize` | `state`, `actions` | z-score / quantile |
| `ResizeImages` | `images` | `resize_with_pad(224,224)` |
| `InjectDefaultPrompt` | `prompt` | 缺失时注入默认值 |
| `TokenizePrompt` | `prompt`, `state` | `tokenized_prompt(_mask)` |
| `TokenizePi05SubtaskInputs` | `prompt`, `subtask` | pi0.5 子任务 token 布局 |
| `TokenizeFASTInputs` / `ExtractFASTActions` | — | FAST 离散动作 |
| `PadStatesAndActions` | `state`, `actions` | 补到 `action_dim=32` |
| `PromptFromLeRobotTask` | `task` | `prompt` |
| `SubtaskFromIndex` | `subtask_index` | `subtask` 文本 |

Delta mask：`make_bool_mask(7, -1, 7, -1, 5, -2)`。

---

## 4. 模型

### 4.1 契约

文件：[model.py](../../src/openpi/models/model.py)

| 符号 | 说明 |
| --- | --- |
| `ModelType` | `PI0` / `PI0_FAST` / `PI05` |
| `IMAGE_KEYS` | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` |
| `IMAGE_RESOLUTION` | `(224, 224)` |
| `Observation` | `images`, `image_masks`, `state`, `tokenized_prompt(_mask)`, `token_ar_mask`, `token_loss_mask`, `subtask_class` |
| `Actions` | `float[*b, ah, ad]` |
| `BaseModelConfig` | `create`, `load`, `inputs_spec`, `fake_obs`, `fake_act`, `get_freeze_filter` |

### 4.2 Pi0

文件：[pi0.py](../../src/openpi/models/pi0.py)、[pi0_config.py](../../src/openpi/models/pi0_config.py)

| 符号 | 说明 |
| --- | --- |
| `Pi0Config` | `pi05`, `discrete_state_input`, `action_dim=32`, `action_horizon=50`, `max_token_len`, 子任务字段 |
| `Pi0.create` | 构建 PaliGemma + SigLIP + action expert + 投影层 |
| `Pi0.embed_prefix` | 图像 token + 语言 token |
| `Pi0.embed_suffix` | state token（pi0）/ adaRMS 时间（pi0.5）+ 动作 token |
| `Pi0.compute_loss` | 兼容旧接口 |
| `Pi0.compute_loss_terms` | 返回 `(action_loss, subtask_ce, class_ce)` |
| `Pi0.sample_actions` | 10 步 Euler，prefix KV cache |
| `Pi0.sample_subtask` | 自回归生成子任务文本 |
| `Pi0.predict_subtask_class` | 分类头 argmax |
| `Pi0.guided_sample_actions` | RTC PiGDM 引导采样 |
| `Pi0.suffix_velocity` | 单步速度场（供 VJP） |
| `SubtaskClassifier` | `LayerNorm + Linear(512) + GELU + Linear(classes)`，fp32 |

### 4.3 其他模型

| 符号 | 文件 | 说明 |
| --- | --- | --- |
| `Pi0FAST` / `Pi0FASTConfig` | [pi0_fast.py](../../src/openpi/models/pi0_fast.py) | FAST 离散动作，`max_token_len=250`, H=32 |
| `Config` / `Module` | [gemma.py](../../src/openpi/models/gemma.py) | Gemma 变体与多专家 |
| `Config` / `Module` | [gemma_fast.py](../../src/openpi/models/gemma_fast.py) | FAST 版 Gemma |
| `Module` | [siglip.py](../../src/openpi/models/siglip.py) | SigLIP So400m/14，`pool_type=none` |
| `Module` | [vit.py](../../src/openpi/models/vit.py) | ViT（供实验） |
| `LoRAConfig` / `apply_lora` | [lora.py](../../src/openpi/models/lora.py) | LoRA 注入 |
| `PaligemmaTokenizer` | [tokenizer.py](../../src/openpi/models/tokenizer.py) | pi0/pi0.5 prompt、state、subtask |
| `FASTTokenizer` | 同上 | FAST 动作 token |
| `rtc_weights` / `guidance_scale` | [rtc.py](../../src/openpi/models/rtc.py) | RTC 权重与引导 |
| `PI0Pytorch` | [models_pytorch/pi0_pytorch.py](../../src/openpi/models_pytorch/pi0_pytorch.py) | PyTorch 参考实现 |

---

## 5. 策略与服务

### 5.1 Policy

文件：[policy.py](../../src/openpi/policies/policy.py)

| 符号 | 说明 |
| --- | --- |
| `Policy` | `infer(obs) -> {"actions", "subtask", "subtask_source"}` |
| `Policy._run_subtask_stage` | `None` / `classifier` / `generate` |
| `Policy._finalize_outputs` | 反归一化 + 输出 Transform |
| `Policy.metadata` | 握手元数据 |
| `PolicyRecorder` | `--record` 时保存输入输出 |

### 5.2 组装

文件：[policy_config.py](../../src/openpi/policies/policy_config.py)

| 符号 | 说明 |
| --- | --- |
| `create_trained_policy(train_config, checkpoint_dir, ...)` | 加载 bf16 参数、Checkpoint 内 norm stats、Transform、RTC |
| `_get_norm_stats` | 优先显式传入 → Checkpoint assets → 报错 |
| `_subtask_mode` | 由模型配置推导默认模式 |

### 5.3 RTC 服务端

文件：[rtc_policy.py](../../src/openpi/policies/rtc_policy.py)

| 符号 | 说明 |
| --- | --- |
| `RTCServerConfig` | `mode`（off/identity/full）、`beta=10`、`guidance_steps`、`num_steps=10`、`s_min=15` |
| `RTCPolicy` | `infer`、`warmup`、`metadata`、`rtc_config` |
| `_rtc` | `openpi.models.rtc` 别名 |
| `RTC_PROTOCOL_VERSION` | `1` |

### 5.4 服务端与客户端

| 符号 | 文件 | 说明 |
| --- | --- | --- |
| `Args` | [remote_infer_server.py](../../scripts/remote_infer_server.py) | `policy`, `rtc`, `subtask`, `port`, `record` |
| `WebsocketPolicyServer` | [serving/websocket_policy_server.py](../../src/openpi/serving/websocket_policy_server.py) | 通用服务端 |
| `WebsocketClientPolicy` | [openpi_client/websocket_client_policy.py](../../packages/openpi-client/src/openpi_client/websocket_client_policy.py) | 通用客户端 |
| `msgpack_numpy.Packer/Unpacker` | [msgpack_numpy.py](../../packages/openpi-client/src/openpi_client/msgpack_numpy.py) | ndarray 编解码 |
| `ActionChunkBroker` | [action_chunk_broker.py](../../packages/openpi-client/src/openpi_client/action_chunk_broker.py) | 动作块逐条返回 |
| `Runtime` / `Agent` / `Environment` / `Subscriber` | [runtime/](../../packages/openpi-client/src/openpi_client/runtime/) | 通用运行框架 |

---

## 6. 机器人运行时

| 符号 | 文件 | 说明 |
| --- | --- | --- |
| `WebsocketClientPolicy` | [test_pi0.py](../../robot_infer/scripts/test_pi0.py) | 机器人端客户端，含 JPEG 压缩 |
| `ActionSmooth` | 同上 | 时间集成，`k=0.01`，不平滑 19–22 |
| `GripperHysteresis` | 同上 | 夹爪阈值 + 去抖 |
| `pin_head_action` | 同上 | 锁定 19/20 维 |
| `prepare_observation` | 同上 | state/prompt/图像编码 |
| `load_hdf5` | 同上 | 23 维 state/action |
| `Real_Env` | [real_env_sdk.py](../../robot_infer/utils/real_env_sdk.py) | `get_observation`、`_set_joint_action`、`move_to_*` |
| `ImageRecorder` | 同上 | 三路相机采集线程 |
| `make_real_env` | 同上 | 工厂 |
| `InferenceWorker` | [async_infer.py](../../robot_infer/utils/async_infer.py) | latest-wins 后台推理、`drain`、`latest_subtask` |
| `run_rtc_loop` | 同上 | 绝对截止时间 30Hz 循环 |
| `RTCScheduler` | [rtc_client.py](../../robot_infer/utils/rtc_client.py) | RTC 调度 |
| `NumpyActionQueue` | 同上 | 双轨动作队列 + `merge` |
| `LatencyTracker` | 同上 | 滑动窗口延迟最大值 |
| `rtc_weights_np` / `execution_horizon` | 同上 | 客户端权重公式 |
| `validate_server_metadata` | 同上 | RTC 握手校验 |
| `HomeSubtaskGuard` / `normalize_subtask` | [subtask_home_guard.py](../../robot_infer/utils/subtask_home_guard.py) | Home 检测与归一化 |

---

## 7. 训练基础设施

| 符号 | 文件 | 说明 |
| --- | --- | --- |
| `create_dataset` / `transform_dataset` | [data_loader.py](../../src/openpi/training/data_loader.py) | LeRobot + `delta_timestamps` + Transform |
| `TorchDataLoader` / `DataLoaderImpl` | 同上 | 多 worker、sharding |
| `CosineDecaySchedule` / `RsqrtDecaySchedule` | [optimizer.py](../../src/openpi/training/optimizer.py) | 学习率 |
| `AdamW` / `AdamWGroupedLR` | 同上 | 优化器与分组 lr |
| `initialize_checkpoint_dir` | [checkpoints.py](../../src/openpi/training/checkpoints.py) | 目录/覆盖/恢复 |
| `save_state` / `restore_state` | 同上 | Orbax 保存/恢复 |
| `load_norm_stats` | 同上 | 从 Checkpoint assets 读取 |
| `CheckpointWeightLoader` / `PaliGemmaWeightLoader` | [weight_loaders.py](../../src/openpi/training/weight_loaders.py) | 部分权重加载 |
| `make_mesh` / `fsdp_sharding` / `set_mesh` | [sharding.py](../../src/openpi/training/sharding.py) | 分片 |
| `TrainState` | [utils.py](../../src/openpi/training/utils.py) | 训练状态容器 |

---

## 8. 共享工具

| 符号 | 文件 | 说明 |
| --- | --- | --- |
| `RunningStats` / `NormStats` | [shared/normalize.py](../../src/openpi/shared/normalize.py) | 统计与归一化 |
| `nnx_utils.module_jit` | [shared/nnx_utils.py](../../src/openpi/shared/nnx_utils.py) | NNX jit 包装 |
| `download.maybe_download` | [shared/download.py](../../src/openpi/shared/download.py) | 本地/远端路径 |
| `resize_with_pad` | [shared/image_tools.py](../../src/openpi/shared/image_tools.py) | 等比缩放 + 填充 |
| `Array` 类型 | [shared/array_typing.py](../../src/openpi/shared/array_typing.py) | jaxtyping 别名 |

---

## 9. 关键数据契约

### 9.1 观测

```python
# 机器人 → 服务端
{
  "images": {"cam_high": bytes, "cam_left_wrist": bytes, "cam_right_wrist": bytes},
  "state": np.float32[23],
  "prompt": str,
  "rtc": {  # 可选
    "version": 1,
    "prev_chunk_left_over": np.float32[T_prev, 23],
    "inference_delay": int,
    "execution_horizon": int,
  },
}
```

### 9.2 响应

```python
{
  "actions": np.float32[50, 23],
  "subtask": str | absent,
  "subtask_source": "classifier" | "generate" | absent,
}
```

### 9.3 模型内部

| 张量 | 形状 |
| --- | --- |
| `images[key]` | `[b, 224, 224, 3]` |
| `state` | `[b, 32]` |
| `tokenized_prompt` | `[b, L]`（pi0: 48，pi0.5: 200） |
| `actions` | `[b, 50, 32]` |
| 模型输出 `v_t` | `[b, 50, 32]` |

### 9.4 23 维布局

```text
0-6   左臂
7     左夹爪
8-14  右臂
15    右夹爪
16-18 腰
19-20 头
21-22 底盘（不执行）
```

---

## 10. 常量与路径

| 常量 | 值 | 位置 |
| --- | --- | --- |
| `IMAGE_RESOLUTION` | `(224, 224)` | `models/model.py` |
| 数据集图像尺寸 | `640×480` | `convert_new.py` |
| `action_dim` | `32` | `Pi0Config` |
| `action_horizon` | `50` | Zerith 配置 |
| `max_token_len` | `48`（pi0）/ `200`（pi0.5） | `Pi0Config` |
| `RTC_PROTOCOL_VERSION` | `1` | `models/rtc.py`、`rtc_client.py` |
| 默认端口 | `55555` | 服务端/客户端 |
| 控制频率 | `30 Hz` | `test_pi0.py` |
| `FIRST_CHUNK_TIMEOUT` | `30.0 s` | `test_pi0.py` |
| 默认 `s_min` | `15`（工具脚本用 18） | RTC |
| `HOME_SUBTASK` | `"move both arms back to the home position"` | `subtask_home_guard.py` |

---

## 11. CLI 速查

```bash
# 转换
uv run scripts/convert_new.py --raw_dir DIR --repo_id ID [--mode image]

# 统计
uv run scripts/compute_norm_stats.py --config_name NAME [--max_frames N]

# 训练
uv run scripts/train.py NAME --exp_name EXP [--overwrite|--resume] [--no-wandb_enabled]

# 服务
uv run scripts/remote_infer_server.py --policy.config NAME --policy.dir CKPT \
  [--port 55555] [--subtask.mode classifier|generate|off] [--rtc.mode identity|full|off]

# 本地评估
uv run scripts/open_loop_eval.py --config_name NAME --checkpoint_dir CKPT --data_path DS

# 远程评估
uv run scripts/open_loop_eval_remote.py --host IP --port 55555 --data_path DS

# 机器人
python robot_infer/scripts/test_pi0.py --host IP --port 55555 \
  --prompt TEXT --init_hdf5 FILE [--no-rtc] [--s_min 15]
```

---

## 12. 按任务查找代码

| 我想… | 看这里 |
| --- | --- |
| 改 23 维映射 | `convert_new.py::_build_state_and_action`、`zerith_joint_policy.py::MobilearxInputs/Outputs` |
| 改 delta 维度 | `config.py` 的 `make_bool_mask(...)` |
| 改相机键 | `zerith_joint_policy.py`、`remote_infer_server.py::_build_observation`、`real_env_sdk.py` |
| 改归一化方式 | `DataConfig.use_quantile_norm`、`transforms.Normalize` |
| 改动作块长度 | `Pi0Config.action_horizon`、`ActionSmooth.action_horizon`、RTC `action_horizon` |
| 改子任务类别 | `assets/subtasks.jsonl`、`Pi0Config.num_subtask_classes`、服务端 `--subtask.jsonl` |
| 改 RTC 行为 | `RTCServerConfig`、`rtc.py::rtc_weights`、`rtc_client.py::RTCScheduler` |
| 改控制频率/平滑 | `test_pi0.py::ActionSmooth`、`--control_freq`、`--query_frequency` |
| 改夹爪阈值 | `test_pi0.py::GripperHysteresis.channel_config` |
| 改 Home 触发 | `HomeSubtaskGuard`、`--home_consecutive`、`assets/subtasks.jsonl[0]` |
| 换基座权重 | `TrainConfig.weight_loader` |
| 改优化器/学习率 | `optimizer.py`、`TrainConfig.lr_schedule` |

---

返回 [Wiki 首页](./README.md)
