"""Merge multiple LeRobotDataset datasets into a single dataset.

This script copies episodes from multiple source datasets into a new merged
dataset by directly copying parquet tables (with updated index/episode/task
columns) and video files. Image data stays as binary in the parquet — no
decode/encode cycle — so merging is fast.

Example:
    uv run scripts/merge_datasets.py \
        --target-repo-id myorg/merged_dataset \
        --source-repo-ids myorg/task_a myorg/task_b \
        --task-prefixes task_a task_b

    # Merge without prefixing tasks (assumes task names are already unique):
    uv run scripts/merge_datasets.py \
        --target-repo-id myorg/merged_dataset \
        --source-repo-ids myorg/task_a myorg/task_b

    # Merge from custom roots (each source read from its own directory):
    uv run scripts/merge_datasets.py \
        --target-repo-id myorg/merged_dataset \
        --source-repo-ids task_a task_b \
        --source-roots /data/ds_a /data/ds_b \
        --target-root /data/merged
"""

import dataclasses
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import (
    LEROBOT_HOME,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(filename)s:%(lineno)s %(message)-80s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def _validate_meta_compatibility(metas: list[LeRobotDatasetMetadata]) -> None:
    """Ensure all source datasets share the same feature schema and fps."""
    if len(metas) <= 1:
        return

    ref = metas[0]
    ref_features = ref.features
    ref_fps = ref.fps

    for i, m in enumerate(metas[1:], start=1):
        if m.fps != ref_fps:
            raise ValueError(
                f"fps mismatch: source[0] has fps={ref_fps}, "
                f"source[{i}] ('{m.repo_id}') has fps={m.fps}. "
                "All sources must share the same fps."
            )
        if set(m.features.keys()) != set(ref_features.keys()):
            missing = set(ref_features.keys()) - set(m.features.keys())
            extra = set(m.features.keys()) - set(ref_features.keys())
            raise ValueError(
                f"Feature key mismatch in source[{i}] ('{m.repo_id}'): "
                f"missing={missing}, extra={extra}."
            )
        for key, ref_ft in ref_features.items():
            src_ft = m.features[key]
            if src_ft["dtype"] != ref_ft["dtype"]:
                raise ValueError(
                    f"dtype mismatch for feature '{key}' in source[{i}] ('{m.repo_id}'): "
                    f"expected '{ref_ft['dtype']}', got '{src_ft['dtype']}'."
                )
            if tuple(src_ft["shape"]) != tuple(ref_ft["shape"]):
                raise ValueError(
                    f"shape mismatch for feature '{key}' in source[{i}] ('{m.repo_id}'): "
                    f"expected {ref_ft['shape']}, got {src_ft['shape']}."
                )


def _build_target_features(meta: LeRobotDatasetMetadata) -> dict:
    """Copy the feature schema from a source dataset, skipping auto-managed defaults."""
    features = {}
    for key, ft in meta.features.items():
        if key in ("index", "frame_index", "episode_index", "timestamp", "task_index"):
            continue
        features[key] = {
            "dtype": ft["dtype"],
            "shape": tuple(ft["shape"]),
            "names": ft.get("names"),
        }
    return features


def _resolve_source_root(roots: list[Path] | None, idx: int) -> Path | None:
    """Pick the root directory for source[idx], or None to use LEROBOT_HOME."""
    if roots is None:
        return None
    if idx < len(roots):
        return roots[idx]
    return None


@dataclasses.dataclass(frozen=True)
class MergeConfig:
    # Tolerance in seconds for timestamp sync checks on the merged dataset.
    tolerance_s: float = 1e-4
    # Whether to compute dataset statistics at the end of consolidation.
    run_compute_stats: bool = True


def _copy_episode_fast(
    src_meta: LeRobotDatasetMetadata,
    src_ep_idx: int,
    target: LeRobotDataset,
    new_ep_idx: int,
    new_task_idx: int,
    frame_offset: int,
    video_keys: list[str],
) -> int:
    """Copy one episode from source to target via direct parquet copy.

    Reads the source parquet table with pyarrow, updates the index/episode_index/
    task_index columns, writes it to the target path. For video mode, also copies
    the mp4 files. Returns the episode length (number of frames).
    """
    # ---- Read source parquet ----
    src_pq_path = src_meta.root / src_meta.get_data_file_path(src_ep_idx)
    table = pq.read_table(src_pq_path)
    ep_length = table.num_rows

    # ---- Update meta columns ----
    col_names = table.column_names

    if "index" in col_names:
        col_idx = table.schema.get_field_index("index")
        table = table.set_column(
            col_idx,
            table.schema.field(col_idx),
            pa.array(
                np.arange(frame_offset, frame_offset + ep_length, dtype=np.int64),
                type=table.column("index").type,
            ),
        )

    if "episode_index" in col_names:
        col_idx = table.schema.get_field_index("episode_index")
        table = table.set_column(
            col_idx,
            table.schema.field(col_idx),
            pa.array(
                np.full(ep_length, new_ep_idx, dtype=np.int64),
                type=table.column("episode_index").type,
            ),
        )

    if "task_index" in col_names:
        col_idx = table.schema.get_field_index("task_index")
        table = table.set_column(
            col_idx,
            table.schema.field(col_idx),
            pa.array(
                np.full(ep_length, new_task_idx, dtype=np.int64),
                type=table.column("task_index").type,
            ),
        )

    # ---- Write target parquet ----
    tgt_pq_path = target.root / target.meta.get_data_file_path(new_ep_idx)
    tgt_pq_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tgt_pq_path)

    # ---- Copy video files (video mode only) ----
    for vid_key in video_keys:
        src_video = src_meta.root / src_meta.get_video_file_path(src_ep_idx, vid_key)
        tgt_video = target.root / target.meta.get_video_file_path(new_ep_idx, vid_key)
        tgt_video.parent.mkdir(parents=True, exist_ok=True)
        if src_video.exists():
            shutil.copy2(src_video, tgt_video)

    return ep_length


def merge_datasets(
    target_repo_id: str,
    source_repo_ids: list[str],
    *,
    source_roots: list[Path] | None = None,
    task_prefixes: list[str] | None = None,
    target_root: str | Path | None = None,
    use_videos: bool | None = None,
    config: MergeConfig = MergeConfig(),
    episodes_per_source: list[list[int]] | None = None,
    error_log_path: Path | None = None,
) -> LeRobotDataset:
    """Merge multiple LeRobotDataset sources into a single new dataset.

    Uses direct parquet copy (pyarrow) instead of frame-by-frame decoding,
    so image data is never decoded/re-encoded — making the merge orders of
    magnitude faster for image-mode datasets.

    Args:
        target_repo_id: repo_id for the output dataset.
        source_repo_ids: list of source repo_ids to merge.
        source_roots: optional per-source root directories. If None, uses LEROBOT_HOME.
        task_prefixes: optional per-source prefix prepended to each task string.
        target_root: root directory for the output dataset. If None, uses LEROBOT_HOME.
        use_videos: whether the target stores images as videos. If None, inherits from source.
        config: merge configuration.
        episodes_per_source: optional per-source list of episode indices to include.
        error_log_path: optional path to write a log of skipped episodes.

    Returns:
        The consolidated merged LeRobotDataset.
    """
    if len(source_repo_ids) == 0:
        raise ValueError("source_repo_ids must contain at least one dataset.")

    if task_prefixes is not None and len(task_prefixes) != len(source_repo_ids):
        raise ValueError(
            f"task_prefixes (len={len(task_prefixes)}) must match "
            f"source_repo_ids (len={len(source_repo_ids)})."
        )

    if episodes_per_source is not None and len(episodes_per_source) != len(source_repo_ids):
        raise ValueError(
            f"episodes_per_source (len={len(episodes_per_source)}) must match "
            f"source_repo_ids (len={len(source_repo_ids)})."
        )

    init_logging()
    logging.info(f"Merging {len(source_repo_ids)} datasets into '{target_repo_id}'")

    # ---- Load source metadata (fast — no hf_dataset loading) ----
    source_metas: list[LeRobotDatasetMetadata] = []
    for i, repo_id in enumerate(source_repo_ids):
        root = _resolve_source_root(source_roots, i)
        logging.info(f"[{i+1}/{len(source_repo_ids)}] Loading source metadata '{repo_id}' (root={root})")
        meta = LeRobotDatasetMetadata(repo_id, root=root, local_files_only=True)
        logging.info(
            f"  -> {meta.total_episodes} episodes, {meta.total_frames} frames, "
            f"fps={meta.fps}, features={list(meta.features.keys())}"
        )
        source_metas.append(meta)

    # ---- Validate schema compatibility ----
    _validate_meta_compatibility(source_metas)
    logging.info("Schema compatibility check passed.")

    # ---- Determine target storage mode ----
    ref_meta = source_metas[0]
    video_keys = list(ref_meta.video_keys)
    if use_videos is None:
        use_videos = len(video_keys) > 0
        logging.info(f"Inherited use_videos={use_videos} from source '{ref_meta.repo_id}'.")

    # ---- Create the target dataset ----
    target_features = _build_target_features(ref_meta)
    if target_root is None:
        target_path = LEROBOT_HOME / target_repo_id
    else:
        target_path = Path(target_root) / target_repo_id

    if target_path.exists():
        logging.warning(f"Target path '{target_path}' already exists; removing it.")
        shutil.rmtree(target_path)

    logging.info(f"Creating target dataset at '{target_path}' (use_videos={use_videos})")
    target = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=ref_meta.fps,
        root=str(target_path) if target_root is not None else None,
        features=target_features,
        use_videos=use_videos,
        tolerance_s=config.tolerance_s,
    )

    skipped_episodes: list[tuple[int, str, int, str]] = []

    # ---- Copy episodes from each source ----
    new_ep_idx = 0
    frame_offset = 0

    for src_idx, src_meta in enumerate(source_metas):
        prefix = task_prefixes[src_idx] if task_prefixes is not None else None
        src_repo = src_meta.repo_id

        if episodes_per_source is not None and episodes_per_source[src_idx] is not None:
            ep_indices = list(episodes_per_source[src_idx])
        else:
            ep_indices = list(range(src_meta.total_episodes))

        logging.info(
            f"[{src_idx+1}/{len(source_metas)}] Copying {len(ep_indices)} episodes "
            f"from '{src_repo}' (task_prefix={prefix!r})"
        )

        for ep_idx in tqdm.tqdm(
            ep_indices, desc=f"source[{src_idx}] '{src_repo}'", dynamic_ncols=True
        ):
            try:
                if ep_idx < 0 or ep_idx >= src_meta.total_episodes:
                    raise IndexError(f"episode_index {ep_idx} out of range [0, {src_meta.total_episodes})")

                # Resolve the task string for this episode.
                ep_info = src_meta.episodes[ep_idx]
                task_list = ep_info.get("tasks", [])
                task_str = task_list[0] if task_list else ""
                if prefix:
                    task_str = f"{prefix}: {task_str}" if task_str else prefix

                # Get or create task index in the target.
                new_task_idx = target.meta.get_task_index(task_str)

                # Fast copy: read parquet, update columns, write parquet, copy videos.
                ep_length = _copy_episode_fast(
                    src_meta=src_meta,
                    src_ep_idx=ep_idx,
                    target=target,
                    new_ep_idx=new_ep_idx,
                    new_task_idx=new_task_idx,
                    frame_offset=frame_offset,
                    video_keys=video_keys,
                )

                # Update target metadata (episodes.jsonl, tasks.jsonl, info.json).
                target.meta.save_episode(new_ep_idx, ep_length, task_str, new_task_idx)

                new_ep_idx += 1
                frame_offset += ep_length

            except (OSError, KeyError, ValueError, RuntimeError, IndexError) as exc:
                skipped_episodes.append((src_idx, src_repo, ep_idx, f"{type(exc).__name__}: {exc}"))
                logging.warning(f"  [SKIP] source[{src_idx}] '{src_repo}' episode {ep_idx}: {exc}")
                continue

    # ---- Finalize ----
    logging.info(
        f"All sources processed. Copied {new_ep_idx} episodes, {frame_offset} frames. "
        f"Skipped {len(skipped_episodes)} episodes. Consolidating target dataset..."
    )
    target.consolidate(run_compute_stats=config.run_compute_stats)

    # ---- Write error log if any episodes were skipped ----
    if skipped_episodes:
        if error_log_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_log_path = Path("logs") / f"merge_skipped_{timestamp}.log"
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with error_log_path.open("w", encoding="utf-8") as f:
            f.write("Skipped episodes during merge\n")
            f.write("source_idx\trepo_id\tepisode_index\treason\n")
            for src_idx, repo_id, ep_idx, reason in skipped_episodes:
                f.write(f"{src_idx}\t{repo_id}\t{ep_idx}\t{reason}\n")
        logging.warning(f"Skipped-episodes log written to: {error_log_path}")

    logging.info(
        f"Merge complete: target='{target.repo_id}', "
        f"episodes={target.meta.total_episodes}, frames={target.meta.total_frames}, "
        f"tasks={target.meta.total_tasks}"
    )
    return target


@dataclasses.dataclass
class MergeCLIArgs:
    """CLI arguments for merging multiple LeRobotDataset datasets into one.

    Example:
        uv run scripts/merge_datasets.py \
            --target-repo-id myorg/merged \
            --source-repo-ids myorg/a myorg/b
    """

    target_repo_id: str
    """repo_id for the merged output dataset."""

    source_repo_ids: list[str]
    """One or more source repo_ids to merge."""

    source_roots: list[str] | None = None
    """Optional per-source root directories. If omitted, each source uses LEROBOT_HOME.
    Must match the length of --source-repo-ids when provided."""

    task_prefixes: list[str] | None = None
    """Optional per-source prefix prepended to each task string to disambiguate
    sources. Must match the length of --source-repo-ids when provided."""

    target_root: str | None = None
    """Optional root directory for the output dataset. Defaults to LEROBOT_HOME."""

    use_videos: bool | None = None
    """Whether the target stores images as videos. If None, inherits from the first source."""

    tolerance_s: float = 1e-4
    """Tolerance (seconds) for timestamp sync checks."""

    no_compute_stats: bool = False
    """If set, skip computing dataset statistics during consolidation."""


def main(args: MergeCLIArgs) -> None:
    source_roots = [Path(p) for p in args.source_roots] if args.source_roots else None
    config = MergeConfig(
        tolerance_s=args.tolerance_s,
        run_compute_stats=not args.no_compute_stats,
    )
    merge_datasets(
        target_repo_id=args.target_repo_id,
        source_repo_ids=args.source_repo_ids,
        source_roots=source_roots,
        task_prefixes=args.task_prefixes,
        target_root=args.target_root,
        use_videos=args.use_videos,
        config=config,
    )


if __name__ == "__main__":
    main(tyro.cli(MergeCLIArgs))
