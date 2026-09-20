"""Real-Time Chunking (RTC) math, implemented in JAX per the paper.

This is the authoritative implementation of the guidance algorithms from
"Real-Time Chunking" (Physical Intelligence; see `rtc_paper/` in this repo).
It fixes two deviations found in the torch reference implementation
(`robot_infer/utils/rtc.py`, extracted from lerobot 0.6.1):

1. VJP degeneration: the torch version computes `v_t` *before* marking `x_t`
   as requiring grad, so the autograd graph never records the `x_t -> v_t`
   edge and the "vector-Jacobian product" silently degrades to an identity
   map (`correction == err`). The paper (Eq. pigdm1) requires the full
   Jacobian `d(x - t*v)/dx = I - t*dv/dx`. Here we apply `jax.vjp` to a
   function that computes both `x1_t` and `v_t` from `x_t` inside the traced
   graph, which yields the exact Jacobian transpose.

2. Weight schedule / execution horizon: the torch version uses a fixed
   `execution_horizon` and truncates the frozen region when `d > s`
   (`start = min(d, s)`). The paper (Eq. weights, Sec. "Real-Time Chunking")
   requires `s = max(d, s_min)` per chunk with the soft-decay region covering
   `[d, H - s)` and the frozen region `[0, d)` never truncated. `rtc_weights`
   below implements the paper's three-piece schedule exactly.

Time conventions: the paper integrates flow matching time `tau` from 0
(noise) to 1 (data); openpi uses `t = 1 - tau` (t=1 is noise). All formulas
below are written in openpi's convention and are algebraically equivalent to
the paper's under `tau = 1 - t`.

References:
- Paper: https://www.physicalintelligence.company/download/real_time_chunking.pdf
"""

from __future__ import annotations

import logging
import math

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("openpi")

# Version of the client/server RTC protocol (advertised in policy metadata).
RTC_PROTOCOL_VERSION = 1


def rtc_weights(d: int, s: int, horizon: int, prev_len: int | None = None) -> np.ndarray:
    """Prefix attention weights `W`, Eq. (weights) in the RTC paper.

    W_i = 1                              for i < d           (frozen: guaranteed to execute)
        = c_i * expm1(c_i) / (e - 1)     for d <= i < H - s  (soft decay)
        = 0                              for i >= H - s      (fresh actions)
    with c_i = (H - s - i) / (H - s - d + 1).

    Computed in numpy on the host: `d`/`s` are Python ints that change per
    request, so keeping them out of the jit graph avoids retracing.

    Args:
        d: Predicted inference delay in control steps.
        s: Execution horizon, `s = max(d, s_min, H - len(leftover))`.
        horizon: Action horizon H.
        prev_len: Length of the (unpadded) previous chunk leftover. Positions
            `>= prev_len` are zero padding in the padded prefix and get
            weight 0. `None` means the leftover covers the full horizon.

    Returns:
        float32 array of shape `(horizon,)`.
    """
    d = int(max(0, min(d, horizon)))
    s = int(max(1, min(s, horizon)))

    weights = np.zeros(horizon, dtype=np.float32)
    weights[:d] = 1.0

    # Soft-decay region [d, H - s). Empty when d >= H - s (degenerate: only
    # the frozen mask remains); callers should log a warning in that case.
    lo, hi = d, horizon - s
    if hi > lo:
        span = hi - lo
        c = (span - np.arange(span)) / (span + 1)
        weights[lo:hi] = (c * np.expm1(c) / (math.e - 1)).astype(np.float32)

    if prev_len is not None and prev_len < horizon:
        weights[prev_len:] = 0.0

    return weights


def guidance_scale(time, beta: float):
    """Guidance weight `min(beta, (t^2 + (1-t)^2) / (t*(1-t)))`.

    This is the paper's `min(beta, (1 - tau) / (tau * r_tau^2))`
    (Eqs. pigdm1/pigdm3) rewritten under `tau = 1 - t`. As `t -> 1` (pure
    noise, first denoising step) the raw value diverges to infinity and is
    clipped to `beta`, which is the paper's stabilization mechanism.
    """
    t = jnp.asarray(time, dtype=jnp.float32)
    raw = (t**2 + (1.0 - t) ** 2) / (t * (1.0 - t))
    return jnp.minimum(jnp.asarray(beta, dtype=jnp.float32), raw)
