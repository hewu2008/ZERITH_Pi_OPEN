import asyncio
import dataclasses
import json
import logging
import os
import pathlib
import socket
import time
import traceback

import cv2
import numpy as np
import tyro
import websockets.asyncio.server
import websockets.frames

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import rtc_policy as _rtc_policy
from openpi.training import config as _config
from openpi_client import msgpack_numpy


CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def camera_aliases(camera_name: str) -> list[str]:
    return [
        camera_name,
        f"rs/{camera_name}",
        f"rs.{camera_name}",
        f"rs_{camera_name}",
    ]


def get_camera_image(images: dict, camera_name: str):
    for alias in camera_aliases(camera_name):
        if alias in images:
            return images[alias]

    rs_images = images.get("rs")
    if isinstance(rs_images, dict) and camera_name in rs_images:
        return rs_images[camera_name]

    raise KeyError(f"Camera '{camera_name}' not found. Available keys: {list(images.keys())}")


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "test").
    config: str
    # Checkpoint directory (e.g., "openpi_checkpoints/test/exp/10000").
    dir: str


@dataclasses.dataclass
class RTCArgs:
    """Real-Time Chunking (RTC) guidance settings (server side).

    Enabled by default: the metadata handshake advertises RTC support and
    requests carrying an `obs["rtc"]` payload (previous chunk leftover +
    delay/horizon counters) are sampled with PiGDM guidance; see
    `openpi.policies.rtc_policy` for the wire protocol. Disable with
    `--rtc.no-enabled`. Requests without the `rtc` field always follow the
    plain path regardless of these settings.
    """

    # Master switch: advertise RTC and enable guided sampling.
    enabled: bool = True
    # Guidance mode: "identity" (legacy identity-Jacobian approximation,
    # default), "full" (exact VJP correction, paper Eq. pigdm1), or "off"
    # (advertise but never guide).
    mode: str = "identity"
    # Maximum guidance weight (paper beta).
    beta: float = 10.0
    # Number of first denoising steps using the exact VJP (None = all steps;
    # a sparsification lever if the full-VJP latency exceeds budget).
    guidance_steps: int | None = None
    # Minimum execution horizon advertised to clients in the handshake.
    s_min: int = 15


@dataclasses.dataclass
class SubtaskArgs:
    """Settings for the pi0.5 high-level subtask stage."""

    # "classifier" (default): one prefix forward + a fixed-class head, then the subtask text is looked up;
    # "generate": autoregressive subtask text, the original slower path kept for A/B comparisons.
    mode: str = "classifier"
    # Class table path; defaults to `$LEROBOT_HOME/<repo_id>/meta/subtasks.jsonl` of the training config.
    jsonl: str | None = None


@dataclasses.dataclass
class Args:
    """Arguments for the websocket policy server."""

    # Policy loading from a trained checkpoint.
    policy: Checkpoint

    # Fallback prompt used by the policy if the request does not include one.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 55555
    # Record the policy's behavior for debugging.
    record: bool = False

    # Real-Time Chunking guidance (enabled by default, identity mode).
    rtc: RTCArgs = dataclasses.field(default_factory=RTCArgs)

    # pi0.5 staged inference: subtask classifier (fast) or autoregressive generation (fallback).
    subtask: SubtaskArgs = dataclasses.field(default_factory=SubtaskArgs)


def _resolve_subtask_jsonl(path: str | None, train_config: _config.TrainConfig) -> pathlib.Path:
    if path is not None:
        return pathlib.Path(path)
    home = os.environ.get("LEROBOT_HOME")
    if home is None:
        raise ValueError("`--subtask.jsonl` was not given and `LEROBOT_HOME` is unset.")
    return pathlib.Path(home) / str(train_config.data.repo_id) / "meta" / "subtasks.jsonl"


def _load_subtask_classes(path: pathlib.Path) -> list[str]:
    """Loads the class table ordered by `subtask_index`; the classifier output index must match the row order."""
    if not path.exists():
        raise FileNotFoundError(f"Subtask class table not found: {path}. Pass `--subtask.jsonl` explicitly.")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records.sort(key=lambda record: int(record["subtask_index"]))
    indices = [int(record["subtask_index"]) for record in records]
    if indices != list(range(len(records))):
        raise ValueError(f"`subtask_index` must be contiguous from 0 in {path}, got {indices}.")
    return [str(record["subtask"]) for record in records]


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    train_config = _config.get_config(args.policy.config)
    rtc_config = None
    if args.rtc.enabled:
        rtc_config = _rtc_policy.RTCServerConfig(
            mode=args.rtc.mode,
            beta=args.rtc.beta,
            guidance_steps=args.rtc.guidance_steps,
            s_min=args.rtc.s_min,
        )
    subtask_classes = None
    if args.subtask.mode == "classifier":
        subtask_path = _resolve_subtask_jsonl(args.subtask.jsonl, train_config)
        subtask_classes = _load_subtask_classes(subtask_path)
        logging.info("Subtask classifier enabled with %d classes from %s", len(subtask_classes), subtask_path)
    else:
        logging.info("Subtask mode: %s (autoregressive generation)", args.subtask.mode)
    return _policy_config.create_trained_policy(
        train_config,
        args.policy.dir,
        default_prompt=args.default_prompt,
        rtc_config=rtc_config,
        subtask_classes=subtask_classes,
        subtask_mode=args.subtask.mode,
    )


def _fake_observation() -> dict:
    """Synthetic observation shaped like `_build_observation` output, for warmup."""
    return {
        "images": {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in CAMERA_NAMES},
        "state": np.zeros(23, dtype=np.float32),
        "prompt": "warmup",
    }


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 55555,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                receive_start = time.time()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                logging.info("receive time: %.4fs", time.time() - receive_start)

                infer_start = time.time()
                observation = self._build_observation(obs)
                action = self._policy.infer(observation)
                infer_time = time.time() - infer_start
                # pi0.5 staged inference attaches the decoded high-level subtask; log it so the robot-side
                # client (which only consumes `actions`) can still be inspected for phase errors.
                subtask = action.get("subtask")
                if subtask is not None:
                    logging.info("infer time: %.4fs | subtask: %s", infer_time, subtask)
                else:
                    logging.info("infer time: %.4fs", infer_time)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _build_observation(self, obs: dict) -> dict:
        observation = {
            "images": {
                camera_name: self.uncompress_image(get_camera_image(obs["images"], camera_name))
                for camera_name in CAMERA_NAMES
            },
            "state": obs["state"],
        }
        if "prompt" in obs:
            observation["prompt"] = obs["prompt"]
        # RTC payload passthrough: `RTCPolicy.infer` pops this key; plain
        # policies never see it (unknown keys are ignored downstream).
        if "rtc" in obs:
            observation["rtc"] = obs["rtc"]
        return observation

    def uncompress_image(self, image):
        if isinstance(image, bytes):
            image = np.frombuffer(image, dtype=np.uint8)
        if isinstance(image, np.ndarray) and image.ndim == 1:
            return cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if isinstance(policy, _rtc_policy.RTCPolicy):
        # Compile the plain and guided samplers before accepting connections,
        # so the first real request cannot hit a JIT compile storm that would
        # blow the client's first-chunk timeout.
        logging.info("RTC enabled: warming up samplers (jit compile, may take a while)...")
        warmup_start = time.time()
        policy.warmup(_fake_observation())
        logging.info("RTC warmup complete in %.1fs", time.time() - warmup_start)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))