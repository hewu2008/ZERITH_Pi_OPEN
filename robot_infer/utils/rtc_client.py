"""Real-Time Chunking (RTC) client-side components (numpy-only).

Robot-host counterpart of the server-side implementation
(`src/openpi/models/rtc.py` + `src/openpi/policies/rtc_policy.py`): the action
bookkeeping, latency accounting and request scheduling of paper Algorithm 1,
with no torch/openpi dependency so it runs inside the robot's own environment.

Wire protocol (version 1), attached to the observation dict when requesting a
guided chunk (see `openpi.policies.rtc_policy` for the authoritative contract):

    obs["rtc"] = {
        "version": 1,
        "prev_chunk_left_over": float32 (T_prev <= H, action_dim),
            # Already-executed-space actions from the *original* (unprocessed)
            # server outputs, i.e. pre-hysteresis/pre-pinning.
        "inference_delay": int,   # d, predicted delay in control steps
        "execution_horizon": int, # s = max(d, s_min, H - T_prev)
    }

The first chunk of an episode (empty leftover) omits the field entirely and is
served by the plain (unguided) path.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

RTC_PROTOCOL_VERSION = 1


def execution_horizon(d: int, s_min: int, horizon: int, prev_len: int | None = None) -> int:
    """Paper's adaptive horizon: s = max(d, s_min, H - T_prev).

    A larger measured delay grows the execution horizon instead of truncating
    the frozen region (the reverse truncation was the second deviation of the
    torch reference implementation).
    """
    if prev_len is None:
        prev_len = horizon
    return max(int(d), int(s_min), int(horizon) - int(prev_len))


def rtc_weights_np(d: int, s_min: int, horizon: int, prev_len: int | None = None) -> np.ndarray:
    """Client copy of the server's `openpi.models.rtc.rtc_weights` schedule.

    Same closed form (paper Eq. (weights), three segments):

        W_i = 1                                        for i < d
        W_i = c_i * expm1(c_i) / (e - 1)              for d <= i < H - s
        W_i = 0                                        for i >= H - s
        c_i = (H - s - i) / (H - s - d + 1)

    with s = max(d, s_min, H - prev_len). Positions >= prev_len are additionally
    zeroed (they are zero padding of a short leftover). The dual-end tests in
    `robot_infer/tests/test_rtc_client.py` cross-check this against the server
    formula so the two implementations cannot drift apart.

    Returns a float32 array of shape `(horizon,)`.
    """
    s = execution_horizon(d, s_min, horizon, prev_len)
    weights = np.zeros(horizon, dtype=np.float32)
    d = min(int(d), horizon)
    weights[:d] = 1.0
    lo, hi = d, horizon - int(s)
    if hi > lo:
        span = hi - lo
        c = (span - np.arange(span)) / (span + 1)
        weights[lo:hi] = (c * np.expm1(c) / (math.e - 1)).astype(np.float32)
    if prev_len is not None and prev_len < horizon:
        weights[prev_len:] = 0.0
    return weights


def validate_server_metadata(metadata: dict) -> dict:
    """Return the RTC handshake block from the server metadata.

    Raises RuntimeError when the server does not advertise RTC support (e.g. an
    old server would otherwise silently downgrade RTC clients to plain mode).
    """
    rtc_info = (metadata or {}).get("rtc")
    if not isinstance(rtc_info, dict):
        raise RuntimeError(
            "Server metadata has no 'rtc' handshake entry; refusing to enable RTC mode. "
            "RTC is enabled by default on the server, so check that it was not started with "
            "--rtc.no-enabled and that it runs a model with guided_sample_actions."
        )
    version = rtc_info.get("version")
    if version != RTC_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Server RTC protocol version {version!r} is incompatible with client version {RTC_PROTOCOL_VERSION}."
        )
    return rtc_info


class LatencyTracker:
    """Sliding-window latency tracker, in control steps (paper alg_main lines 30/42).

    Fixed relative to the torch reference: `max()` is the *window* maximum, not
    a monotonic running maximum, so one transient spike stops dominating the
    predicted delay d once it leaves the window.
    """

    def __init__(self, maxlen: int = 20) -> None:
        self._values: deque[int] = deque(maxlen=maxlen)

    def reset(self) -> None:
        """Clear all recorded latencies."""
        self._values.clear()

    def add(self, latency: int) -> None:
        """Add a latency sample (control steps). Negative values are ignored."""
        val = int(latency)
        if val < 0:
            return
        self._values.append(val)

    def __len__(self) -> int:
        return len(self._values)

    def max(self) -> int | None:
        """Return the window maximum, or None when no sample has been recorded."""
        return max(self._values) if self._values else None

    def mean(self) -> float | None:
        """Return the window mean, or None when no sample has been recorded."""
        return sum(self._values) / len(self._values) if self._values else None


class NumpyActionQueue:
    """Numpy port of the torch `ActionQueue` (robot_infer/utils/rtc.py L621-832).

    Two tracks are kept per chunk:
    - `queue`: the processed actions handed to the robot one per control step;
    - `original_queue`: the unprocessed server outputs, whose unconsumed suffix
      (`get_left_over()`) becomes the RTC `prev_chunk_left_over` (paper A_prev).

    RTC mode replaces the queue on every chunk and skips the `real_delay`
    actions that elapsed while the chunk was being computed; the disabled mode
    appends (legacy continuous-rollout behavior). Unlike the `ActionSmooth`
    bookkeeping it replaces, memory is O(H) per chunk instead of the O(T^2)
    `all_time_actions` buffer.
    """

    def __init__(self) -> None:
        self.queue: np.ndarray | None = None  # processed actions (T, action_dim)
        self.original_queue: np.ndarray | None = None  # original actions (T, action_dim)
        self.lock = threading.Lock()
        self.last_index = 0

    def get(self) -> np.ndarray | None:
        """Return the next processed action `(action_dim,)` copy, or None when empty."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return np.array(action)

    def clear(self) -> None:
        """Clear queued actions and reset the consumption index."""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        """Number of remaining (unconsumed) actions."""
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """True when no actions remain."""
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        """Index of the next action to be consumed."""
        with self.lock:
            return self.last_index

    def get_left_over(self) -> np.ndarray | None:
        """Unconsumed suffix of the *original* actions, for RTC guidance.

        Returns `(remaining_steps, action_dim)` or None when no original queue
        exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return np.array(self.original_queue[self.last_index :])

    def merge(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        action_index_before_inference: int | None = None,
        *,
        rtc_enabled: bool = True,
    ) -> None:
        """Merge a new chunk into the queue.

        RTC mode replaces the queue and skips the first `real_delay` actions
        (they elapsed while the chunk was being computed); non-RTC mode appends
        and maintains continuity.

        Args:
            original_actions: Unprocessed actions from the policy (T, action_dim).
            processed_actions: Post-processed actions for the robot (T, action_dim).
            real_delay: Measured delay in control steps between submission and now.
            action_index_before_inference: Queue index at submission time, for
                consistency validation.
        """
        with self.lock:
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)
            if rtc_enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay)
                return
            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: np.ndarray, processed_actions: np.ndarray, real_delay: int) -> None:
        """Replace the queue with the new chunk, skipping the stale prefix."""
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = np.array(original_actions[clamped_delay:], dtype=np.float32)
        self.queue = np.array(processed_actions[clamped_delay:], dtype=np.float32)
        logger.debug(
            "RTC queue replace: real_delay=%d clamped=%d remaining=%d",
            real_delay,
            clamped_delay,
            len(self.queue),
        )
        self.last_index = 0

    def _append_actions_queue(self, original_actions: np.ndarray, processed_actions: np.ndarray) -> None:
        """Append the new chunk, dropping already-consumed actions (non-RTC mode)."""
        if self.queue is None:
            self.original_queue = np.array(original_actions, dtype=np.float32)
            self.queue = np.array(processed_actions, dtype=np.float32)
            return

        self.original_queue = np.concatenate([self.original_queue, np.array(original_actions, dtype=np.float32)])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = np.concatenate([self.queue, np.array(processed_actions, dtype=np.float32)])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_and_resolve_delays(
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        """Validate the measured delay against the actions consumed meanwhile."""
        effective_delay = max(0, int(real_delay))

        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != effective_delay:
                logger.warning(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    effective_delay,
                )
                return effective_delay

        return effective_delay


class RTCScheduler:
    """Paper Algorithm 1 client-side scheduler driving an `InferenceWorker`.

    Per control step (`get_action`):
    1. Poll worker results; each returned chunk is merged with its *measured*
       delay `real_delay = current_step - query_t` (control steps, not
       wall-clock estimates), which also feeds the `LatencyTracker` and resets
       `steps_since_swap` to the delay (paper `t = t - s` accounting).
    2. Submit a new request when `steps_since_swap >= s_min` while the worker
       is idle, or immediately when the queue is empty (episode bootstrap and
       starvation bailout). The request carries `obs["rtc"]` built from the
       leftover snapshot; an empty leftover omits the field (plain first chunk).
    3. Return the next processed action (a copy), or None when the queue ran
       dry so the caller can hold the current pose.
    """

    def __init__(
        self,
        worker=None,
        *,
        s_min: int = 15,
        action_horizon: int = 50,
        latency_window: int = 20,
    ) -> None:
        self.worker = worker
        self.s_min = int(s_min)
        self.action_horizon = int(action_horizon)
        self.queue = NumpyActionQueue()
        self.tracker = LatencyTracker(maxlen=latency_window)
        # Paper step accounting.
        self.t = 0  # control step counter
        self.steps_since_swap = 0
        # Stale-result guards (latest-wins slot + failure recovery).
        self._last_merged_query_t = -1
        self._pending_submit: tuple[int, int] | None = None  # (query_t, action_index)

    def reset(self) -> None:
        """Reset all per-episode state (queue, tracker, counters)."""
        self.queue.clear()
        self.tracker.reset()
        self.t = 0
        self.steps_since_swap = 0
        self._last_merged_query_t = -1
        self._pending_submit = None

    def observe_chunk(self, query_t: int, actions: np.ndarray) -> int | None:
        """Merge a chunk returned for submission step `query_t`.

        Returns the measured delay in control steps, or None when the result is
        stale (out-of-order or older than a full horizon) and was dropped.
        """
        query_t = int(query_t)
        real_delay = self.t - query_t
        if query_t <= self._last_merged_query_t:
            logger.warning("RTC: dropping out-of-order chunk (query_t=%d <= %d)", query_t, self._last_merged_query_t)
            return None
        if real_delay < 0 or real_delay > self.action_horizon:
            logger.warning(
                "RTC: dropping stale chunk (query_t=%d, delay=%d > horizon=%d)",
                query_t,
                real_delay,
                self.action_horizon,
            )
            return None

        original = np.asarray(actions, dtype=np.float32)
        # Post-processing (gripper hysteresis, head pinning) happens on copies
        # at consumption time, so both tracks share the server outputs here.
        action_index_before_inference = None
        if self._pending_submit is not None and self._pending_submit[0] == query_t:
            action_index_before_inference = self._pending_submit[1]
            self._pending_submit = None

        self.queue.merge(original, original, real_delay, action_index_before_inference)
        self._last_merged_query_t = query_t
        self.tracker.add(real_delay)
        # Paper `t = t - s`: the swap is accounted from the submission step.
        self.steps_since_swap = real_delay
        return real_delay

    def _build_rtc_payload(self) -> tuple[dict | None, int, int, int]:
        """Snapshot the leftover and build the `obs["rtc"]` payload.

        Returns `(payload, d, s, prev_len)`; payload is None when the leftover
        is empty (vanilla request without the rtc field).
        """
        leftover = self.queue.get_left_over()
        prev_len = 0 if leftover is None else len(leftover)
        if leftover is None or prev_len == 0:
            return None, 0, self.s_min, 0

        d = self.tracker.max()
        if d is None:
            d = 0
        s = execution_horizon(d, self.s_min, self.action_horizon, prev_len)
        payload = {
            "version": RTC_PROTOCOL_VERSION,
            "prev_chunk_left_over": np.ascontiguousarray(leftover, dtype=np.float32),
            "inference_delay": int(d),
            "execution_horizon": int(s),
        }
        return payload, int(d), int(s), prev_len

    def _trigger(self) -> bool:
        """Paper-trigger approximation: elapsed steps reached s_min, or starvation."""
        return self.queue.empty() or self.steps_since_swap >= self.s_min

    def maybe_submit(self, observation: dict, *, force: bool = False) -> bool:
        """Submit a request when the trigger condition holds.

        `force=True` bypasses both the trigger and the in-flight suppression
        (used for the synchronous first chunk of an episode).

        Returns True when a request was submitted.
        """
        payload, d, s, prev_len = self._build_rtc_payload()

        if not force:
            if self.worker is None or self.worker.in_flight:
                return False
            if not self._trigger():
                return False

        obs = dict(observation)
        if payload is not None:
            obs["rtc"] = payload
        self._pending_submit = (self.t, self.queue.get_action_index())
        self.worker.submit(obs, query_t=self.t)
        self.steps_since_swap = 0
        logger.info(
            "RTC request submitted at step %d (d=%d s=%d leftover=%d)",
            self.t,
            d,
            s,
            prev_len,
        )
        return True

    def submit_query(self, observation: dict) -> bool:
        """Force a submission now (first-chunk bootstrap)."""
        return self.maybe_submit(observation, force=True)

    def _consume_results(self) -> None:
        if self.worker is None:
            return
        while True:
            result = self.worker.poll()
            if result is None:
                break
            query_t, actions = result
            self.observe_chunk(query_t, actions)

    def get_action(self, observation: dict) -> np.ndarray | None:
        """Per-control-step driver: consume, maybe submit, advance one action."""
        self._consume_results()
        self.maybe_submit(observation)
        action = self.queue.get()
        self.t += 1
        self.steps_since_swap += 1
        return action
