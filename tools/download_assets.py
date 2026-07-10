#!/usr/bin/env python3
"""
Pre-download assets for offline use.

This tool downloads pre-trained model weights and other assets to the local cache,
so they can be used in offline environments without network connectivity.

Examples:
    # Download pi0_base checkpoint
    python tools/download_assets.py pi0_base

    # Download all pre-defined assets
    python tools/download_assets.py --all

    # Download a custom URL
    python tools/download_assets.py --url s3://openpi-assets/checkpoints/my_model/params

    # Download to a specific directory
    python tools/download_assets.py pi0_base --output /path/to/cache
"""

import argparse
import concurrent.futures
import datetime
import getpass
import os
import re
import shutil
import stat
import sys
import time
import urllib.parse
from pathlib import Path

import boto3
import boto3.s3.transfer as s3_transfer
import botocore
import filelock
import fsspec
import s3transfer.futures as s3_transfer_futures
import tqdm_loggable.auto as tqdm
from types_boto3_s3.service_resource import ObjectSummary


def get_cache_dir() -> Path:
    """Get the cache directory for storing downloaded assets."""
    _OPENPI_DATA_HOME = "OPENPI_DATA_HOME"
    default_dir = "~/.cache/openpi"
    if os.path.exists("/mnt/weka"):
        default_dir = f"/mnt/weka/{getpass.getuser()}/.cache/openpi"

    cache_dir = Path(os.getenv(_OPENPI_DATA_HOME, default_dir)).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def _set_permission(path: Path, target_permission: int):
    if path.stat().st_mode & target_permission == target_permission:
        return
    path.chmod(target_permission)


def _set_folder_permission(folder_path: Path) -> None:
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: Path) -> None:
    def _setup_folder_permission_between_cache_dir_and_path(path: Path) -> None:
        cache_dir = get_cache_dir()
        relative_path = path.relative_to(cache_dir)
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)
            moving_path = moving_path / part

    def _set_file_permission(file_path: Path) -> None:
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:
            _set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(file_path, file_rw)

    _setup_folder_permission_between_cache_dir_and_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            _set_file_permission(file_path)

        for dir in dirs:
            dir_path = root_path / dir
            _set_folder_permission(dir_path)


def _is_openpi_url(url: str) -> bool:
    return url.startswith("s3://openpi-assets/")


def _get_mtime(year: int, month: int, day: int) -> float:
    date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)
    return time.mktime(date.timetuple())


_INVALIDATE_CACHE_DIRS: dict[re.Pattern, float] = {
    re.compile("openpi-assets/checkpoints/pi0_aloha_pen_uncap"): _get_mtime(2025, 2, 17),
    re.compile("openpi-assets/checkpoints/pi0_libero"): _get_mtime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): _get_mtime(2025, 2, 3),
}


def _should_invalidate_cache(cache_dir: Path, local_path: Path) -> bool:
    assert local_path.exists(), f"File not found at {local_path}"

    relative_path = str(local_path.relative_to(cache_dir))
    for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():
        if pattern.match(relative_path):
            return local_path.stat().st_mtime <= expire_time

    return False


def _download_fsspec(url: str, local_path: Path, **kwargs) -> None:
    fs, _ = fsspec.core.url_to_fs(url, **kwargs)
    info = fs.info(url)
    if is_dir := (info["type"] == "directory"):
        total_size = fs.du(url)
    else:
        total_size = info["size"]
    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fs.get, url, local_path, recursive=is_dir)
        while not future.done():
            current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
            pbar.update(current_size - pbar.n)
            time.sleep(1)
        pbar.update(total_size - pbar.n)


def _get_s3_transfer_manager(
    session: boto3.Session, workers: int, botocore_config: botocore.config.Config | None = None
) -> s3_transfer.TransferManager:
    config = botocore.config.Config(max_pool_connections=workers + 2)
    if botocore_config is not None:
        config = config.merge(botocore_config)
    s3client = session.client("s3", config=config)
    transfer_config = s3_transfer.TransferConfig(
        use_threads=True,
        max_concurrency=workers,
    )
    return s3_transfer.create_transfer_manager(s3client, transfer_config)


def _download_boto3(
    url: str,
    local_path: Path,
    *,
    boto_session: boto3.Session | None = None,
    botocore_config: botocore.config.Config | None = None,
    workers: int = 16,
) -> None:
    def validate_and_parse_url(maybe_s3_url: str) -> tuple[str, str]:
        parsed = urllib.parse.urlparse(maybe_s3_url)
        if parsed.scheme != "s3":
            raise ValueError(f"URL must be an S3 URL (s3://), got: {maybe_s3_url}")
        bucket_name = parsed.netloc
        prefix = parsed.path.strip("/")
        return bucket_name, prefix

    bucket_name, prefix = validate_and_parse_url(url)
    session = boto_session or boto3.Session()

    s3api = session.resource("s3", config=botocore_config)
    bucket = s3api.Bucket(bucket_name)

    try:
        bucket.Object(prefix).load()
    except botocore.exceptions.ClientError:
        if not prefix.endswith("/"):
            prefix = prefix + "/"

    objects = [x for x in bucket.objects.filter(Prefix=prefix) if not x.key.endswith("/")]
    if not objects:
        raise FileNotFoundError(f"No objects found at {url}")

    total_size = sum(obj.size for obj in objects)

    s3t = _get_s3_transfer_manager(session, workers, botocore_config=botocore_config)

    def transfer(
        s3obj: ObjectSummary, dest_path: Path, progress_func
    ) -> s3_transfer_futures.TransferFuture | None:
        if dest_path.exists():
            dest_stat = dest_path.stat()
            if s3obj.size == dest_stat.st_size:
                progress_func(s3obj.size)
                return None
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        return s3t.download(
            bucket_name,
            s3obj.key,
            str(dest_path),
            subscribers=[
                s3_transfer.ProgressCallbackInvoker(progress_func),
            ],
        )

    try:
        with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
            def update_progress(size: int) -> None:
                pbar.update(size)

            futures = []
            for obj in objects:
                relative_path = Path(obj.key).relative_to(prefix)
                dest_path = local_path / relative_path
                if future := transfer(obj, dest_path, update_progress):
                    futures.append(future)
            for future in futures:
                future.result()
    finally:
        s3t.shutdown()


def maybe_download(url: str, *, force_download: bool = False, **kwargs) -> Path:
    """Download a file or directory from a remote filesystem to the local cache, and return the local path."""
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme == "":
        path = Path(url)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()

    cache_dir = get_cache_dir()

    local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
    local_path = local_path.resolve()

    invalidate_cache = False
    if local_path.exists():
        if force_download or _should_invalidate_cache(cache_dir, local_path):
            invalidate_cache = True
        else:
            return local_path

    try:
        lock_path = local_path.with_suffix(".lock")
        with filelock.FileLock(lock_path):
            _ensure_permissions(lock_path)
            if invalidate_cache:
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()

            scratch_path = local_path.with_suffix(".partial")

            if _is_openpi_url(url):
                _download_boto3(
                    url,
                    scratch_path,
                    boto_session=boto3.Session(
                        region_name="us-west-1",
                    ),
                    botocore_config=botocore.config.Config(signature_version=botocore.UNSIGNED),
                )
            elif url.startswith("s3://"):
                _download_boto3(url, scratch_path)
            else:
                _download_fsspec(url, scratch_path, **kwargs)

            shutil.move(scratch_path, local_path)
            _ensure_permissions(local_path)

    except PermissionError as e:
        msg = (
            f"Local file permission error was encountered while downloading {url}. "
            f"Please try again after removing the cached data using: `rm -rf {local_path}*`"
        )
        raise PermissionError(msg) from e

    return local_path


def download_asset(url: str, output_dir: str | None = None):
    """Download an asset from the given URL."""
    if output_dir:
        os.environ["OPENPI_DATA_HOME"] = output_dir

    local_path = maybe_download(url, force_download=False)
    print(f"Downloaded: {url}")
    print(f"Local path: {local_path}")
    return local_path


PREDEFINED_ASSETS = {
    "pi0_base": "s3://openpi-assets/checkpoints/pi0_base/params",
    "pi0_aloha_pen_uncap": "s3://openpi-assets/checkpoints/pi0_aloha_pen_uncap",
    "pi0_libero": "s3://openpi-assets/checkpoints/pi0_libero",
    "paligemma": "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz",
}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download assets for offline use.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "asset",
        nargs="?",
        choices=list(PREDEFINED_ASSETS.keys()),
        help="Predefined asset to download",
    )

    parser.add_argument(
        "--url",
        type=str,
        help="Custom URL to download (overrides asset argument)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all predefined assets",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for cached files (default: ~/.cache/openpi)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )

    args = parser.parse_args()

    urls_to_download = []

    if args.url:
        urls_to_download.append(args.url)
    elif args.all:
        urls_to_download.extend(PREDEFINED_ASSETS.values())
    elif args.asset:
        urls_to_download.append(PREDEFINED_ASSETS[args.asset])
    else:
        parser.print_help()
        print("\nPredefined assets:", file=sys.stderr)
        for name, url in PREDEFINED_ASSETS.items():
            print(f"  {name}: {url}", file=sys.stderr)
        sys.exit(1)

    if args.force:
        cache_dir = Path(args.output) if args.output else get_cache_dir()
        print(f"Clearing existing cache in: {cache_dir}")

        for url in urls_to_download:
            parsed = urllib.parse.urlparse(url)
            local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
            if local_path.exists():
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()
                print(f"Removed: {local_path}")

    print(f"\nDownloading {len(urls_to_download)} asset(s)...\n")

    for i, url in enumerate(urls_to_download, 1):
        print(f"[{i}/{len(urls_to_download)}] {url}")
        try:
            download_asset(url, args.output)
        except Exception as e:
            print(f"Failed to download {url}: {e}", file=sys.stderr)

    print("\nDone! Assets are ready for offline use.")


if __name__ == "__main__":
    main()
