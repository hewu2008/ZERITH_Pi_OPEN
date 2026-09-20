"""Real-Time Chunking (RTC) policy wrapper.

Extends the standard serving `Policy` with RTC-guided sampling: when a client
includes an `obs["rtc"]` payload with the previous chunk's leftover actions,
the next chunk is generated with PiGDM guidance towards that leftover (paper
Algorithm 1). Requests without the field follow the plain path unchanged, so
the protocol extension is fully backward compatible.

Client -> server protocol (version 1), carried inside the observation dict:

    obs["rtc"] = {
        "version": 1,
        "prev_chunk_left_over": float32 (T_prev <= H, action_dim_of_robot),
            # Already-executed-space actions from the *original* (unprocessed)
            # server outputs, i.e. pre-hysteresis/pre-pinning.
        "inference_delay": int,   # predicted delay d in control steps
        "execution_horizon": int, # s = max(d, s_min, H - T_prev)
    }

The leftover is re-encoded into the model's normalized delta-action space by
riding the *input* transform chain (`obs["actions"] = prev` re-anchors the
deltas against the current observation state, exactly like training targets),
so no manual normalization code is duplicated.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any
from collections.abc import Sequence

import jax
import numpy as np

from typing_extensions import override

from openpi.models import model as _model
from openpi.models import rtc as _rtc
from openpi.policies import policy as _policy
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi import transforms as _transforms

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class RTCServerConfig:
    """Server-side RTC settings. Fixed at startup (static for jit)."""

    # Guidance mode:
    # - "off":      plain sampling for every request (feature disabled).
    # - "identity": identity-Jacobian correction (`correction = err`), the
    #               legacy torch behavior; zero backward cost. A/B baseline.
    # - "full":     exact VJP correction per paper Eq. (pigdm1). Default.
    mode: str = "full"
    # Maximum guidance weight (paper beta).
    beta: float = 10.0
    # Number of *first* denoising steps that use the exact VJP; the remaining
    # steps fall back to the identity approximation. None means all steps.
    guidance_steps: int | None = None
    # Number of denoising steps (must match the plain policy's num_steps).
    num_steps: int = 10
    # s_min advertised to clients in the metadata handshake.
    s_min: int = 15

    def __post_init__(self):
        if self.mode not in ("off", "identity", "full"):
            raise ValueError(f"Invalid RTC mode: {self.mode!r}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

    @property
    def jit_guidance_steps(self) -> int:
        """Static guidance_steps value for the jitted guided sampler."""
        if self.mode == "identity":
            return 0
        if self.mode == "full":
            return self.guidance_steps if self.guidance_steps is not None else self.num_steps
        return self.num_steps  # unused in "off" mode


class RTCPolicy(_policy.Policy):
    """Serving policy with Real-Time Chunking guidance.

    See the module docstring for the wire protocol. The guided sampler is
    jitted once with static `num_steps`/`guidance_steps`; `d`/`s` only enter
    through the numpy-precomputed weight schedule, so per-request variation
    never retraces.
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rtc_config: RTCServerConfig,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not hasattr(model, "guided_sample_actions"):
            raise TypeError(f"Model {type(model).__name__} does not support RTC (no guided_sample_actions).")
        super().__init__(
            model,
            transforms=transforms,
            output_transforms=output_transforms,
            sample_kwargs=sample_kwargs,
            metadata=metadata,
        )
        self._model = model
        self._rtc_config = rtc_config
        # static_argnames: num_steps/guidance_steps unroll the Python loop at
        # trace time and must not become traced scalars.
        self._guided_sample_actions = nnx_utils.module_jit(
            model.guided_sample_actions, static_argnames=("num_steps", "guidance_steps")
        )
        self._metadata = {
            **(metadata or {}),
            "rtc": {
                "version": _rtc.RTC_PROTOCOL_VERSION,
                "action_horizon": model.action_horizon,
                "mode": rtc_config.mode,
                "beta": rtc_config.beta,
                "s_min": rtc_config.s_min,
            },
        }
        if rtc_config.mode == "off":
            logger.info("RTC policy initialized in 'off' mode (guidance disabled)")
        else:
            logger.info(
                "RTC policy initialized: mode=%s beta=%.1f guidance_steps=%s/%d",
                rtc_config.mode,
                rtc_config.beta,
                rtc_config.jit_guidance_steps,
                rtc_config.num_steps,
            )

    @property
    def rtc_config(self) -> RTCServerConfig:
        return self._rtc_config

    def warmup(self, fake_obs: dict | None = None) -> None:
        """Compile the plain and guided samplers ahead of the first real request.

        `fake_obs` must be shaped like the server's `_build_observation` output
        (raw camera images, proprioceptive state and a prompt); it never touches
        real robot state. The guided warmup uses a zero leftover with an
        all-zero weight schedule (a mathematical no-op) so the compiled graph
        is exactly the one used at serving time. Both calls are timed and
        synchronized so the reported numbers reflect real compile cost and the
        first client request cannot hit a JIT compile storm.
        """
        if self._rtc_config.mode == "off":
            return
        if fake_obs is None:
            return

        # Plain path: every episode's first chunk has no leftover and goes
        # through `super().infer` unchanged.
        start = time.perf_counter()
        super().infer(dict(fake_obs))
        logger.info("RTC warmup: plain sampler compiled in %.1fs", time.perf_counter() - start)

        horizon, dim = self._model.action_horizon, self._model.action_dim
        prev = np.zeros((1, horizon, dim), dtype=np.float32)
        weights = np.zeros(horizon, dtype=np.float32)
        obs = self._prepare_inputs(fake_obs)
        self._rng, sample_rng = jax.random.split(self._rng)
        start = time.perf_counter()
        actions = self._guided_sample_actions(
            sample_rng,
            _model.Observation.from_dict(obs),
            prev,
            weights,
            beta=self._rtc_config.beta,
            num_steps=self._rtc_config.num_steps,
            guidance_steps=self._rtc_config.jit_guidance_steps,
        )
        jax.block_until_ready(actions)
        logger.info("RTC warmup: guided sampler compiled in %.1fs", time.perf_counter() - start)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        rtc_info = obs.pop("rtc", None)
        if rtc_info is None or self._rtc_config.mode == "off":
            return super().infer(obs)

        horizon = self._model.action_horizon
        dim = self._model.action_dim

        prev = rtc_info.get("prev_chunk_left_over")
        if prev is None or len(prev) == 0:
            # No leftover (e.g. first chunk after a queue clear): plain path.
            return super().infer(obs)

        d = int(rtc_info.get("inference_delay", 0))
        s = int(rtc_info.get("execution_horizon", self._rtc_config.s_min))
        # Guard the paper's constraint d <= s <= H - d: `rtc_weights` degrades
        # to a pure frozen mask when d >= H - s, so we only warn here.
        if d >= horizon - s:
            logger.warning(
                "RTC degenerate schedule: d=%d >= H-s=%d (H=%d). Guidance reduces to a frozen mask.",
                d,
                horizon - s,
                horizon,
            )

        # Ride the input transform chain to re-encode the leftover into the
        # model's normalized delta-action space (pad -> delta vs current state
        # -> zero chassis dims -> normalize), identical to training targets.
        obs["actions"] = np.asarray(prev, dtype=np.float32)
        inputs = self._prepare_inputs(obs)
        if "actions" not in inputs:
            raise KeyError(
                "RTC: the input transform chain dropped the 'actions' key; "
                "the leftover re-encoding relies on it surviving (no repack whitelist)."
            )
        prev_model = np.asarray(inputs["actions"])[0]  # (T_prev, action_dim), normalized
        prev_len = min(prev_model.shape[0], horizon)

        # Fixed shapes across requests (no retrace): pad to (1, H, A).
        prev_padded = np.zeros((1, horizon, dim), dtype=np.float32)
        prev_padded[0, :prev_len, : prev_model.shape[1]] = prev_model[:prev_len]

        weights = _rtc.rtc_weights(d, s, horizon, prev_len=prev_len)

        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self._guided_sample_actions(
            sample_rng,
            _model.Observation.from_dict(inputs),
            prev_padded,
            weights,
            beta=self._rtc_config.beta,
            num_steps=self._rtc_config.num_steps,
            guidance_steps=self._rtc_config.jit_guidance_steps,
        )

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        return self._finalize_outputs(outputs)
