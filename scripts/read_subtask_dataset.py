"""Read and inspect a LeRobot dataset that contains subtask annotations.

This script loads a dataset with ``subtask_index`` column in the parquet files
and ``meta/subtasks.jsonl`` mapping, then prints:
  - The subtask index → string mapping
  - Per-episode subtask transition timeline
  - Global subtask frame distribution

It also exposes ``load_subtasks()`` and ``SubtaskFromLeRobotSubtask`` which can
be plugged into the training data pipeline to inject the ``subtask`` string
field (mirroring how ``PromptFromLeRobotTask`` injects ``prompt`` from
``task_index``).

Example:
    python scripts/read_subtask_dataset.py \
        --data_path /data/4T-1/hewu/dataset/hewu2008/clear_the_bin_box_20260721_v2

    # Or using repo_id with LEROBOT_HOME:
    LEROBOT_HOME=/data/4T-1/hewu/dataset \
    python scripts/read_subtask_dataset.py \
        --repo_id hewu2008/clear_the_bin_box_20260721_v2
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid remote repo checks when loading local-only datasets.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import load_jsonlines

import openpi.transforms as transforms

SUBTASKS_PATH = "meta/subtasks.jsonl"


def load_subtasks(dataset_root: Path) -> dict[int, str]:
    """Load ``meta/subtasks.jsonl`` and return a ``{subtask_index: subtask_str}`` dict.

    This mirrors ``lerobot.common.datasets.utils.load_tasks`` but for subtasks.
    """
    fpath = dataset_root / SUBTASKS_PATH
    if not fpath.exists():
        raise FileNotFoundError(
            f"subtasks.jsonl not found at {fpath}. "
            "Ensure the dataset was created with subtask annotations."
        )
    items = load_jsonlines(fpath)
    return {item["subtask_index"]: item["subtask"] for item in sorted(items, key=lambda x: x["subtask_index"])}


@dataclasses.dataclass(frozen=True)
class SubtaskFromLeRobotSubtask(transforms.DataTransformFn):
    """Inject the ``subtask`` string field from ``subtask_index``.

    This is the subtask counterpart of ``PromptFromLeRobotTask``.  Place it in
    the repack transforms (after ``PromptFromLeRobotTask``) so downstream
    transforms such as ``TokenizePi05SubtaskInputs`` receive the ``subtask``
    string they expect.
    """

    subtasks: dict[int, str]

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if "subtask_index" not in data:
            raise ValueError('Cannot extract subtask without "subtask_index"')

        subtask_index = int(data["subtask_index"])
        if (subtask := self.subtasks.get(subtask_index)) is None:
            raise ValueError(f"{subtask_index=} not found in subtask mapping: {self.subtasks}")

        return {**data, "subtask": subtask}


def _resolve_dataset(data_path: str | None, repo_id: str | None) -> tuple[str, Path | None]:
    """Return ``(repo_id, root)`` for LeRobotDataset construction."""
    if data_path is not None:
        p = Path(data_path)
        if p.is_absolute() and p.exists():
            return p.name, p
        return str(p), None
    if repo_id is None:
        raise ValueError("Either --data_path or --repo_id must be provided.")
    return repo_id, None


def print_subtask_mapping(subtasks: dict[int, str]) -> None:
    print("=" * 80)
    print("Subtask mapping (subtask_index → subtask)")
    print("-" * 80)
    for idx, subtask in sorted(subtasks.items()):
        print(f"  [{idx}] {subtask}")
    print()


def print_episode_summary(
    dataset: LeRobotDataset,
    subtasks: dict[int, str],
    *,
    max_episodes: int | None = None,
) -> None:
    """Print per-episode subtask transitions using column-level access (no image decoding)."""
    # Access subtask_index and episode_index columns directly to avoid decoding images.
    # See project memory: accessing hf_dataset columns via .data.column() bypasses image decoding.
    subtask_col = dataset.hf_dataset.data.column("subtask_index").to_pylist()
    episode_col = dataset.hf_dataset.data.column("episode_index").to_pylist()
    frame_index_col = dataset.hf_dataset.data.column("frame_index").to_pylist()

    num_episodes = dataset.num_episodes
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)

    print("=" * 80)
    print(f"Per-episode subtask summary (showing {num_episodes}/{dataset.num_episodes} episodes)")
    print("-" * 80)

    global_counts: dict[int, int] = {}

    for ep_idx in range(num_episodes):
        start = int(dataset.episode_data_index["from"][ep_idx])
        end = int(dataset.episode_data_index["to"][ep_idx])

        ep_subtasks = subtask_col[start:end]
        ep_frames = frame_index_col[start:end]

        # Find transitions
        transitions: list[tuple[int, int]] = []  # (frame_index, subtask_index)
        prev_st = None
        for f_idx, st in zip(ep_frames, ep_subtasks):
            if st != prev_st:
                transitions.append((f_idx, st))
                prev_st = st

        ep_length = end - start
        print(f"  Episode {ep_idx} (length={ep_length}):")
        for f_idx, st in transitions:
            label = subtasks.get(st, f"<unknown:{st}>")
            print(f"    frame {f_idx:>6d} → subtask [{st}] {label}")

        # Count frames per subtask in this episode
        ep_counts: dict[int, int] = {}
        for st in ep_subtasks:
            ep_counts[st] = ep_counts.get(st, 0) + 1
        parts = [f"[{st}]={cnt}" for st, cnt in sorted(ep_counts.items())]
        print(f"    frame counts: {', '.join(parts)}")
        print()

        for st, cnt in ep_counts.items():
            global_counts[st] = global_counts.get(st, 0) + cnt

    print("=" * 80)
    print("Global subtask frame distribution")
    print("-" * 80)
    total = sum(global_counts.values())
    for st, cnt in sorted(global_counts.items()):
        label = subtasks.get(st, f"<unknown:{st}>")
        pct = 100.0 * cnt / total if total else 0.0
        print(f"  [{st}] {label:<70s}  {cnt:>6d} frames ({pct:5.1f}%)")
    print(f"  {'Total':<70s}  {total:>6d} frames")
    print()


def print_sample_frame(dataset: LeRobotDataset, subtasks: dict[int, str]) -> None:
    """Print one sample frame to show available fields including subtask_index."""
    print("=" * 80)
    print("Sample frame (index=0)")
    print("-" * 80)
    sample = dataset[0]
    for key in sorted(sample.keys()):
        val = sample[key]
        if hasattr(val, "shape"):
            print(f"  {key:40s}  shape={val.shape}  dtype={val.dtype}")
        else:
            print(f"  {key:40s}  value={val}")

    st_idx = int(sample["subtask_index"])
    print()
    print(f"  subtask_index = {st_idx}")
    print(f"  subtask       = {subtasks.get(st_idx, '<unknown>')}")
    print()


def main(
    data_path: str | None = None,
    repo_id: str | None = None,
    max_episodes: int | None = None,
    show_sample: bool = True,
) -> None:
    """Read and inspect a LeRobot dataset with subtask annotations.

    Args:
        data_path: Absolute path to the dataset directory on disk.
        repo_id: HuggingFace-style repo_id (e.g. ``hewu2008/clear_the_bin_box_20260721_v2``).
            Requires ``LEROBOT_HOME`` to point to the parent directory.
        max_episodes: Limit the number of episodes to summarize.
        show_sample: If True, print one sample frame's fields.
    """
    repo_id, root = _resolve_dataset(data_path, repo_id)

    meta = LeRobotDatasetMetadata(repo_id, root=root, local_files_only=True)
    dataset_root = meta.root

    subtasks = load_subtasks(dataset_root)
    print_subtask_mapping(subtasks)

    dataset = LeRobotDataset(repo_id, root=root, local_files_only=True)
    print(f"Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes\n")

    print_episode_summary(dataset, subtasks, max_episodes=max_episodes)

    if show_sample:
        print_sample_frame(dataset, subtasks)


if __name__ == "__main__":
    tyro.cli(main)
