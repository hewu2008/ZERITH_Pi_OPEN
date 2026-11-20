# 06 策略服务与客户端

服务层把训练好的 Checkpoint 包装成 `infer(obs) -> dict` 的在线策略，通过 WebSocket 向机器人提供动作块。Zerith 的部署入口是 [scripts/remote_infer_server.py](../../scripts/remote_infer_server.py)，它比通用服务端多了 **RTC、子任务阶段、相机别名解码** 与 **预热**。

---

## 1. Policy 抽象

### 1.1 `BasePolicy`

文件：[packages/openpi-client/src/openpi_client/base_policy.py](../../packages/openpi-client/src/openpi_client/base_policy.py)

```python
class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> dict: ...
    @abc.abstractmethod
    def reset(self) -> None: ...
```

### 1.2 `Policy`

文件：[policy.py](../../src/openpi/policies/policy.py)

`Policy` 组合：

| 成员 | 作用 |
| --- | --- |
| `_sample_actions` | jit 后的 `model.sample_actions` |
| `_sample_subtask` | jit 后的 `model.sample_subtask`（仅开启生成时） |
| `_classify_subtask` | jit 后的 `model.predict_subtask_class`（仅开启分类时） |
| `_input_transform` | 输入 Transform 组合 |
| `_output_transform` | 输出 Transform 组合 |
| `_subtask_mode` | `None` / `"classifier"` / `"generate"` |
| `_metadata` | 启动握手元数据 |

`infer(obs)` 流程：

```text
_run_subtask_stage(obs)
  ├─ 输入 Transform + Normalize
  ├─ 无子任务：直接采样动作
  ├─ classifier：一次前缀前向 + argmax → 文本 → 重建输入
  └─ generate：自回归生成子任务 → 重建输入
sample_actions(rng, observation)
_finalize_outputs：Unnormalize → 输出 Transform
附加 subtask / subtask_source（若存在）
```

返回：

```python
{
  "actions": np.ndarray,        # (50, 23) float32
  "subtask": str,               # 可选，分类/生成模式
  "subtask_source": str,        # "classifier" | "generate"
}
```

> `actions` 已在输出 Transform 中反归一化、还原 delta、并截取 23 维。

### 1.3 `PolicyRecorder`

`--record` 时包装 Policy，把每一步 `inputs/outputs` 扁平化保存到 `policy_records/step_<n>.npy`，用于复现问题。

---

## 2. 从 Checkpoint 构建 Policy

文件：[policy_config.py](../../src/openpi/policies/policy_config.py)

```python
create_trained_policy(
    train_config,
    checkpoint_dir,
    *,
    repack_transforms=None,
    sample_kwargs=None,
    default_prompt=None,
    norm_stats=None,
    rtc_config=None,
    subtask_classes=None,
    subtask_mode=None,
) -> Policy
```

关键行为：

1. `download.maybe_download(checkpoint_dir)`：支持本地路径或远端 URI。
2. `train_config.model.load(restore_params(checkpoint_dir/"params", dtype=jnp.bfloat16))`：**bf16 加载**。
3. 若未显式传 `norm_stats`，从 **Checkpoint 内部** `assets/<asset_id>/norm_stats.json` 读取，保证与训练一致。
4. 组装输入/输出 Transform：

```text
输入：repack.inputs → InjectDefaultPrompt → data.inputs → Normalize → model.inputs
输出：model.outputs → Unnormalize → data.outputs → repack.outputs
```

5. `rtc_config` 非空时返回 `RTCPolicy`，否则返回普通 `Policy`。
6. 子任务开启时构造 `PaligemmaTokenizer` 用于 detokenize。

> 部署时不需要 `$LEROBOT_HOME`（除非使用 `--subtask.jsonl` 的默认路径），但 Checkpoint 必须包含 `params/` 与 `assets/`。

---

## 3. RTC 服务端策略

文件：[rtc_policy.py](../../src/openpi/policies/rtc_policy.py)

### 3.1 `RTCServerConfig`

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `mode` | `"identity"` | `"off"` / `"identity"`（零反向开销）/ `"full"`（精确 VJP） |
| `beta` | `10.0` | 最大引导权重 |
| `guidance_steps` | `None` | 前 N 步使用精确 VJP；`None` = 全部 |
| `num_steps` | `10` | 去噪步数，须与普通采样一致 |
| `s_min` | `15` | 握手时告知客户端的最小执行步数 |

`jit_guidance_steps`：

- `identity` → 0（全部走 identity）
- `full` → `guidance_steps` 或 `num_steps`
- `off` → 不使用

### 3.2 元数据握手

`RTCPolicy.__init__` 会向外暴露：

```json
{
  "rtc": {
    "version": 1,
    "action_horizon": 50,
    "mode": "identity",
    "beta": 10.0,
    "s_min": 15
  }
}
```

机器人端 `validate_server_metadata()` 会校验 `version`，缺失或不匹配直接拒绝启动（避免静默降级）。

### 3.3 `infer`

```text
rtc_info = obs.pop("rtc", None)
if 无 payload 或 mode == "off":  走普通 Policy.infer
else:
  解析 prev_chunk_left_over / inference_delay / execution_horizon
  把 leftover 写入 obs["actions"]，复用输入 Transform 链重新编码为
     「归一化 + delta + pad」空间（与训练目标完全一致）
  计算 rtc_weights(d, s, H, prev_len)
  （可选）先跑子任务阶段
  guided_sample_actions(rng, observation, prev_padded, weights, beta, num_steps, guidance_steps)
  输出后处理（反归一化 / 绝对动作 / 23 维）
```

RTC 请求字段（协议 v1）：

```python
obs["rtc"] = {
    "version": 1,
    "prev_chunk_left_over": np.float32[T_prev, 23],  # 原始（未后处理）动作
    "inference_delay": int,      # 实测延迟 d（控制步）
    "execution_horizon": int,    # s = max(d, s_min, H - T_prev)
}
```

### 3.4 预热

`RTCPolicy.warmup(fake_obs)` 会：

1. 跑一次普通路径（编译 plain sampler）；
2. 构造零 leftover 的 RTC payload，再跑一次 guided 路径（编译 VJP + 子任务阶段）。

`remote_infer_server.py` 在 RTC 启用时于监听前执行预热，避免首个真实请求触发 JIT 编译导致首块超时。

---

## 4. 远程推理服务端

文件：[remote_infer_server.py](../../scripts/remote_infer_server.py)

### 4.1 CLI

```bash
uv run scripts/remote_infer_server.py \
  --policy.config pi05_zerith_subtask_cls \
  --policy.dir openpi_checkpoints/pi05_zerith_subtask_cls/run/15999 \
  [--default-prompt "..."] \
  [--port 55555] \
  [--record] \
  [--rtc.enabled/--rtc.no-enabled] \
  [--rtc.mode identity|full|off] \
  [--rtc.beta 10.0] \
  [--rtc.guidance-steps N] \
  [--rtc.s_min 15] \
  [--subtask.mode classifier|generate|off] \
  [--subtask.jsonl assets/subtasks.jsonl]
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--policy.config` | 必填 | `TrainConfig` 名称 |
| `--policy.dir` | 必填 | Checkpoint 目录（含 `params/assets`） |
| `--default-prompt` | `None` | 请求未带 prompt 时使用 |
| `--port` | `55555` | 监听端口（`0.0.0.0`） |
| `--record` | `False` | 记录推理输入输出 |
| `--rtc.enabled` | `True` | 是否启用 RTC |
| `--rtc.mode` | `identity` | 引导模式 |
| `--rtc.beta` | `10.0` | 引导权重 |
| `--rtc.s_min` | `15` | 最小执行步数 |
| `--subtask.mode` | `classifier` | 子任务模式；`off/none` 关闭 |
| `--subtask.jsonl` | 数据集 meta | 类别表路径 |

### 4.2 启动流程

1. `get_config(args.policy.config)`；
2. 若启用 RTC，构造 `RTCServerConfig`；
3. 若 `subtask.mode == classifier`，加载并校验类别表（`subtask_index` 必须从 0 连续）；
4. `create_trained_policy(...)`；
5. 若为 `RTCPolicy`，执行 `policy.warmup(_fake_observation())`；
6. 可选 `PolicyRecorder`；
7. 启动 asyncio WebSocket 服务。

### 4.3 请求处理

```text
连接建立 → 发送 metadata 帧
循环：
  接收 msgpack obs
  _build_observation(obs)
  policy.infer(observation)
  发送 msgpack 响应
异常：
  发送 traceback 字符串帧 → close(1011)
```

`_build_observation`：

- 相机名：`cam_high`、`cam_left_wrist`、`cam_right_wrist`；
- 兼容别名：`rs/<cam>`、`rs.<cam>`、`rs_<cam>`、嵌套 `images["rs"][cam]`；
- 图像为 `bytes` 或 `uint8` ndarray 时用 `cv2.imdecode(..., IMREAD_COLOR)` 解码；
- 透传 `state`、`prompt`、`rtc`。

### 4.4 日志

每个请求打印：

```text
receive time: 0.0123s
infer time: 0.1456s | subtask: place the ... in the left bin
```

机器人端可据此观察端到端延迟与子任务阶段。

---

## 5. WebSocket 线协议

### 5.1 传输层

- 客户端 → 服务端：一条 msgpack 二进制消息，内容为观测字典；
- 服务端 → 客户端：
  - 连接后的**第一帧**：metadata（dict）；
  - 之后每次请求：结果 dict；
  - 出错：**字符串** traceback，随后关闭码 `1011`。
- `compression=None`、`max_size=None`（图像较大）。

### 5.2 `msgpack_numpy`

文件：[msgpack_numpy.py](../../packages/openpi-client/src/openpi_client/msgpack_numpy.py)

- `np.ndarray` → `{b"__ndarray__": True, b"data": bytes, b"dtype": dtype.str, b"shape": shape}`；
- `np.generic` → `{b"__npgeneric__": True, b"data": item, b"dtype": dtype.str}`；
- 不支持 `object` / 结构化 / 复数 dtype；
- 使用 `object_hook` 在解码时还原。

### 5.3 观测字典

机器人端 `prepare_observation` 发送：

```python
{
  "images": {
    "cam_high": np.uint8[N],  # JPEG 字节（也支持 Python bytes）
    "cam_left_wrist": np.uint8[N],
    "cam_right_wrist": np.uint8[N],
  },
  "state": np.float32[23],   # qpos 补 2 维 0
  "prompt": str,
  "qpos": ..., "timestamp": ...,   # 其余字段服务端忽略
  # 可选
  "rtc": {...},
}
```

图像压缩：`cv2.cvtColor(RGB→BGR)` → `resize_with_pad(224, 224)` → JPEG quality 100。

---

## 6. 客户端

### 6.1 仓库内通用客户端

文件：[websocket_client_policy.py](../../packages/openpi-client/src/openpi_client/websocket_client_policy.py)

- `WebsocketClientPolicy(host, port=None, api_key=None)`；
- 连接成功后立即读取第一帧 metadata；
- `get_server_metadata()` 返回该 metadata；
- `infer(obs)`：发送 msgpack，接收 bytes；若收到 `str` 则抛 `RuntimeError`；
- `reset()` 为空操作。

### 6.2 `ActionChunkBroker`

文件：[action_chunk_broker.py](../../packages/openpi-client/src/openpi_client/action_chunk_broker.py)

- 包装任意 Policy，把 `(H, ...)` 动作块逐条返回；
- 只有当前块耗尽（`_cur_step >= action_horizon`）才重新推理；
- `reset()` 清空缓存并重置内层策略。

> Zerith 机器人端没有使用 `ActionChunkBroker`，而是使用自研的 `ActionSmooth` / `RTCScheduler`（见 [07 机器人运行时](./07-robot-runtime.md)）。

### 6.3 `runtime` 抽象

目录：[packages/openpi-client/src/openpi_client/runtime/](../../packages/openpi-client/src/openpi_client/runtime/)

| 组件 | 职责 |
| --- | --- |
| `Runtime` | 按 `max_hz` 循环：`get_observation → agent.get_action → apply_action`，并通知 subscribers；支持 episode 数、最大步数、线程运行 |
| `Agent` | `get_action(observation)` / `reset()` |
| `Environment` | `reset()` / `get_observation()` / `apply_action()` / `is_episode_complete()` |
| `Subscriber` | `on_episode_start` / `on_step` / `on_episode_end`，用于录制、可视化 |

这是一个通用、与硬件无关的运行框架；Zerith 的 `test_pi0.py` 没有直接继承它，而是实现了更贴近实时控制的循环。

---

## 7. 并发与性能

| 关注点 | 现状 |
| --- | --- |
| 并发连接 | asyncio 可接受多个连接，但 GPU 推理实际串行；每个连接的处理协程按请求顺序执行 |
| 首块延迟 | RTC 启用时启动预热；客户端另设 `FIRST_CHUNK_TIMEOUT = 30s` |
| 图像开销 | JPEG quality 100，三路 224×224；网络与编解码通常是主要非模型开销 |
| 采样步数 | 默认 10 步 Euler；RTC `full` 模式每步多一次 VJP，延迟更高 |
| 子任务延迟 | classifier ≈ 一次前缀前向；generate ≈ 每 token 一次 2B 前向（明显更慢） |

---

## 8. 最小联调示例

```python
import numpy as np
from openpi_client import websocket_client_policy as _wcp

client = _wcp.WebsocketClientPolicy(host="127.0.0.1", port=55555)
print(client.get_server_metadata())

obs = {
    "images": {
        "cam_high": np.zeros((224, 224, 3), dtype=np.uint8),
        "cam_left_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        "cam_right_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
    },
    "state": np.zeros(23, dtype=np.float32),
    "prompt": "Sort all objects into the correct bins using both arms",
}
result = client.infer(obs)
print(result["actions"].shape)   # (50, 23)
```

> 纯色图像只用于连通性测试；真实策略需要有效相机画面与正确 state。

---

下一步：[07 机器人运行时](./07-robot-runtime.md)
