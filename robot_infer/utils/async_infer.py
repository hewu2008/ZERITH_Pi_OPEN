import logging
import threading
import time

_IDLE_POLL_INTERVAL = 0.001


class InferenceWorker(threading.Thread):
    """Runs policy inference in a background thread.

    Observations are exchanged through a latest-wins slot: a new submission
    overwrites any observation not picked up yet, which naturally applies
    back-pressure when inference is slower than the query rate.
    """

    def __init__(self, client) -> None:
        super().__init__(daemon=True, name="inference-worker")
        self._client = client
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pending_obs = None  # (query_t, obs) awaiting inference
        self._result = None  # latest unconsumed (query_t, actions)
        self._last_result_time = None  # monotonic time of the last successful infer
        self._consecutive_failures = 0
        self._infer_times = []
        self._max_infer_time = 0.0

    def submit(self, observation, query_t: int) -> None:
        with self._lock:
            self._pending_obs = (query_t, observation)

    def poll(self):
        """Return the latest unconsumed (query_t, actions), or None."""
        with self._lock:
            result = self._result
            self._result = None
            return result

    def last_result_age(self):
        """Seconds since the last successful inference, or None if none yet."""
        with self._lock:
            if self._last_result_time is None:
                return None
            return time.monotonic() - self._last_result_time

    def wait_for_first_result(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not self._stop_event.is_set():
            with self._lock:
                ready = self._last_result_time is not None
            if ready:
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return False

    def stats(self):
        with self._lock:
            count = len(self._infer_times)
            mean_time = sum(self._infer_times) / count if count else 0.0
            max_time = self._max_infer_time
        if count == 0:
            return None
        return count, mean_time, max_time

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self.join(timeout=timeout)

    def run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                pending = self._pending_obs
                self._pending_obs = None
            if pending is None:
                time.sleep(_IDLE_POLL_INTERVAL)
                continue

            query_t, observation = pending
            time0 = time.monotonic()
            try:
                response = self._client.infer(observation)
                actions = response["actions"]
            except Exception as exc:
                self._consecutive_failures += 1
                logging.error(
                    "Inference failed (%d consecutive failures): %s",
                    self._consecutive_failures,
                    exc,
                )
                continue

            elapsed = time.monotonic() - time0
            self._consecutive_failures = 0
            with self._lock:
                self._result = (query_t, actions)
                self._last_result_time = time.monotonic()
                self._infer_times.append(elapsed)
                self._max_infer_time = max(self._max_infer_time, elapsed)


def run_rtc_loop(step_fn, freq: float, num_steps: int) -> None:
    """Drive `step_fn(step)` at a fixed frequency on an absolute deadline schedule."""
    period = 1.0 / freq
    next_deadline = time.monotonic() + period
    for step in range(num_steps):
        step_fn(step)
        delay = next_deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            logging.warning("Control loop overrun: %.1f ms late", -delay * 1000)
        next_deadline += period
