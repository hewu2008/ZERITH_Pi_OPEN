"""Merge multiple LeRobotDataset datasets into a single dataset.

This script reads several source LeRobotDataset datasets and copies all their
episodes/frames into a new merged dataset. It preserves task labels (by their
string description) and optionally prefixes each task with a source-specific tag
to avoid name collisions across datasets.

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
from datetime import datetime
from pathlib import Path

import einops
import numpy as np
import torch
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset


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
    logger.handlers[0].setFormatter(formatter)


def _frame_image_to_hwc_uint8(image) -> np.ndarray:
    """Convert an image returned by LeRobotDataset.__getitem__ to HWC uint8.

    __getitem__ returns images as torch tensors in CHW layout (possibly float in
    [0,1]). add_frame expects HWC numpy arrays. This helper normalizes the input.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255.0 * image).clip(0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        # CHW -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _validate_schema_compatibility(sources: list[LeRobotDataset]) -> None:
    """Ensure all source datasets share the same feature schema and fps."""
    if len(sources) <= 1:
        return

    ref = sources[0]
    ref_features = ref.features
    ref_fps = ref.fps

    for i, src in enumerate(sources[1:], start=1):
        if src.fps != ref_fps:
            raise ValueError(
                f"fps mismatch: source[0] has fps={ref_fps}, "
                f"source[{i}] ('{src.repo_id}') has fps={src.fps}. "
                "All sources must share the same fps."
            )
        if set(src.features.keys()) != set(ref_features.keys()):
            missing = set(ref_features.keys()) - set(src.features.keys())
            extra = set(src.features.keys()) - set(ref_features.keys())
            raise ValueError(
                f"Feature key mismatch in source[{i}] ('{src.repo_id}'): "
                f"missing={missing}, extra={extra}."
            )
        for key, ref_ft in ref_features.items():
            src_ft = src.features[key]
            if src_ft["dtype"] != ref_ft["dtype"]:
                raise ValueError(
                    f"dtype mismatch for feature '{key}' in source[{i}] ('{src.repo_id}'): "
                    f"expected '{ref_ft['dtype']}', got '{src_ft['dtype']}'."
                )
            if tuple(src_ft["shape"]) != tuple(ref_ft["shape"]):
                raise ValueError(
                    f"shape mismatch for feature '{key}' in source[{i}] ('{src.repo_id}'): "
                    f"expected {ref_ft['shape']}, got {src_ft['shape']}."
                )


def _build_target_features(source: LeRobotDataset) -> dict:
    """Copy the feature schema from a source dataset, preserving image/video dtype."""
    features = {}
    for key, ft in source.features.items():
        # Skip the auto-managed default features (index, frame_index, episode_index,
        # timestamp, task_index); LeRobotDataset.create adds them automatically via
        # DEFAULT_FEATURES.
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
    # Number of async image-writer processes (0 = use threads only).
    image_writer_processes: int = 10
    # Number of async image-writer threads.
    image_writer_threads: int = 5
    # Tolerance in seconds for timestamp sync checks on the merged dataset.
    tolerance_s: float = 1e-4
    # Whether to compute dataset statistics at the end of consolidation.
    run_compute_stats: bool = True
    # If True, keep the intermediate image files after video encoding.
    keep_image_files: bool = False


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

    Args:
        target_repo_id: repo_id for the output dataset.
        source_repo_ids: list of source repo_ids to merge.
        source_roots: optional per-source root directories. If None, uses LEROBOT_HOME.
        task_prefixes: optional per-source prefix prepended to each task string
            (e.g. "task_a: do something"). Must have the same length as source_repo_ids
            if provided. If None, task strings are copied verbatim.
        target_root: root directory for the output dataset. If None, uses LEROBOT_HOME.
        use_videos: whether the target dataset stores images as videos. If None,
            inherits from the first source dataset.
        config: merge configuration (image writer, tolerance, stats).
        episodes_per_source: optional per-source list of episode indices to include.
            If None, all episodes from each source are included.
        error_log_path: optional path to write a log of skipped episodes.

    Returns:
        The consolidated merged LeRobotDataset.
    """
    if len(source_repo_ids) == 0:
        raise ValueError("source_repo_ids must contain at least one dataset.")

    if task_prefixes is not None:
        if len(task_prefixes) != len(source_repo_ids):
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

    # ---- Load all source datasets ----
    sources: list[LeRobotDataset] = []
    for i, repo_id in enumerate(source_repo_ids):
        root = _resolve_source_root(source_roots, i)
        logging.info(f"[{i+1}/{len(source_repo_ids)}] Loading source '{repo_id}' (root={root})")
        src = LeRobotDataset(repo_id, root=root, local_files_only=True)
        logging.info(
            f"  -> {src.num_episodes} episodes, {src.num_frames} frames, "
            f"fps={src.fps}, features={list(src.features.keys())}"
        )
        sources.append(src)

    # ---- Validate schema compatibility ----
    _validate_schema_compatibility(sources)
    logging.info("Schema compatibility check passed.")

    # ---- Determine target storage mode ----
    ref = sources[0]
    if use_videos is None:
        # Inherit from the reference source: if it has any video-key feature, use videos.
        use_videos = len(ref.meta.video_keys) > 0
        logging.info(f"Inherited use_videos={use_videos} from source '{ref.repo_id}'.")

    # ---- Create the target dataset ----
    target_features = _build_target_features(ref)
    if target_root is None:
        target_path = LEROBOT_HOME / target_repo_id
    else:
        target_path = Path(target_root) / target_repo_id

    if target_path.exists():
        import shutil

        logging.warning(f"Target path '{target_path}' already exists; removing it.")
        shutil.rmtree(target_path)

    logging.info(f"Creating target dataset at '{target_path}' (use_videos={use_videos})")
    target = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=ref.fps,
        root=str(target_path) if target_root is not None else None,
        features=target_features,
        use_videos=use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
    )
    # create() may have started the image writer already; ensure it's running.
    if target.image_writer is None:
        target.start_image_writer(
            num_processes=config.image_writer_processes,
            num_threads=config.image_writer_threads,
        )

    # Keys that should be copied verbatim from each source frame (non-image, non-meta).
    meta_keys = {"index", "frame_index", "episode_index", "timestamp", "task_index"}
    copyable_keys = [
        key for key in ref.features.keys() if key not in meta_keys and ref.features[key]["dtype"] not in ("image", "video")
    ]
    camera_keys = list(ref.meta.camera_keys)

    skipped_episodes: list[tuple[int, str, int, str]] = []  # (source_idx, repo_id, ep_idx, reason)

    # ---- Copy episodes from each source ----
    for src_idx, src in enumerate(sources):
        prefix = task_prefixes[src_idx] if task_prefixes is not None else None
        src_repo = src.repo_id

        # Determine which episodes to copy from this source.
        if episodes_per_source is not None and episodes_per_source[src_idx] is not None:
            ep_indices = list(episodes_per_source[src_idx])
        else:
            ep_indices = list(range(src.num_episodes))

        logging.info(
            f"[{src_idx+1}/{len(sources)}] Copying {len(ep_indices)} episodes "
            f"from '{src_repo}' (task_prefix={prefix!r})"
        )

        for ep_idx in tqdm.tqdm(
            ep_indices, desc=f"source[{src_idx}] '{src_repo}'", dynamic_ncols=True
        ):
            try:
                if ep_idx < 0 or ep_idx >= src.num_episodes:
                    raise IndexError(f"episode_index {ep_idx} out of range [0, {src.num_episodes})")

                ep_from = int(src.episode_data_index["from"][ep_idx])
                ep_to = int(src.episode_data_index["to"][ep_idx])
                ep_length = ep_to - ep_from
                if ep_length <= 0:
                    raise ValueError(f"episode {ep_idx} has non-positive length {ep_length}")

                # Resolve the task string for this episode.
                ep_meta = src.meta.episodes[ep_idx]
                src_task_index = ep_meta["task_index"]
                task_str = src.meta.tasks.get(src_task_index, "")
                if prefix:
                    task_str = f"{prefix}: {task_str}" if task_str else prefix

                # Copy each frame.
                for frame_idx in range(ep_from, ep_to):
                    frame = src[frame_idx]

                    new_frame = {}
                    for key in copyable_keys:
                        if key in frame:
                            val = frame[key]
                            new_frame[key] = val.numpy() if isinstance(val, torch.Tensor) else np.asarray(val)

                    for cam_key in camera_keys:
                        if cam_key in frame:
                            new_frame[cam_key] = _frame_image_to_hwc_uint8(frame[cam_key])

                    target.add_frame(new_frame)

                target.save_episode(task=task_str, encode_videos=False)

            except (OSError, KeyError, ValueError, RuntimeError, IndexError) as exc:
                skipped_episodes.append((src_idx, src_repo, ep_idx, f"{type(exc).__name__}: {exc}"))
                logging.warning(f"  [SKIP] source[{src_idx}] '{src_repo}' episode {ep_idx}: {exc}")
                # Reset the target episode buffer so the next episode starts clean.
                target.clear_episode_buffer()
                continue

    # ---- Finalize ----
    target.stop_image_writer()
    logging.info(
        f"All sources processed. Skipped {len(skipped_episodes)} episodes. "
        "Consolidating target dataset..."
    )
    target.consolidate(
        run_compute_stats=config.run_compute_stats,
        keep_image_files=config.keep_image_files,
    )

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

    image_writer_processes: int = 10
    """Number of async image-writer processes."""

    image_writer_threads: int = 5
    """Number of async image-writer threads."""

    tolerance_s: float = 1e-4
    """Tolerance (seconds) for timestamp sync checks."""

    no_compute_stats: bool = False
    """If set, skip computing dataset statistics during consolidation."""

    keep_image_files: bool = False
    """If set, keep intermediate image files after video encoding."""


def main(args: MergeCLIArgs) -> None:
    source_roots = [Path(p) for p in args.source_roots] if args.source_roots else None
    config = MergeConfig(
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        tolerance_s=args.tolerance_s,
        run_compute_stats=not args.no_compute_stats,
        keep_image_files=args.keep_image_files,
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
