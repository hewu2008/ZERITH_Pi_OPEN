"""Inspect a sanity dump directory produced by the training sanity_mode.

Loads the dumped tensors (state.npy / actions.npy / raw_state.npy / raw_actions.npy)
and prints a summary flagging suspicious values (NaN / Inf / zero-variance dimensions)
that would indicate a broken data pipeline.

Usage:
    python scripts/inspect_sanity_dump.py <sanity_dump_dir>
"""

import argparse
from pathlib import Path

import numpy as np


def _summary(name: str, arr: np.ndarray) -> None:
    print(f"[{name}] shape={arr.shape} dtype={arr.dtype}")
    # Guard against NaN/Inf which numpy would otherwise swallow in min/max.
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count or inf_count:
        print(f"  ** WARN: NaN={nan_count}, Inf={inf_count} **")
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr.reshape(-1, 1)
    print(f"  mean={np.nanmean(flat, axis=0)}")
    print(f"  std ={np.nanstd(flat, axis=0)}")
    # Flag constant (zero-variance) dimensions -> normalization would amplify by ~1e6x.
    std = np.nanstd(flat, axis=0)
    for i in range(std.shape[0]):
        if std[i] == 0:
            print(f"  ** WARN: dim={i} std=0 (constant), normalization amplifies ~1e6x **")


def _check_pair(norm: Path, raw: Path) -> None:
    n = np.load(norm)
    r = np.load(raw)
    if n.shape != r.shape:
        print(f"  ** WARN: shape mismatch {norm.name}={n.shape} vs {raw.name}={r.shape} **")


def main(dump_dir: str) -> None:
    dump_dir = Path(dump_dir)
    if not dump_dir.exists():
        raise FileNotFoundError(f"Sanity dump dir does not exist: {dump_dir}")

    print(f"Inspecting sanity dump: {dump_dir}\n")

    for fname in ("state.npy", "actions.npy", "raw_state.npy", "raw_actions.npy"):
        path = dump_dir / fname
        if not path.exists():
            print(f"[{fname}] MISSING")
            continue
        print(f"===== {fname} =====")
        _summary(fname, np.load(path))

    print("\n===== normalized vs raw shape pair-check =====")
    _check_pair(dump_dir / "state.npy", dump_dir / "raw_state.npy")
    _check_pair(dump_dir / "actions.npy", dump_dir / "raw_actions.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", help="path to the sanity dump directory")
    args = parser.parse_args()
    main(args.dump_dir)