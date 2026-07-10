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
import sys
from pathlib import Path


def download_asset(url: str, output_dir: str | None = None):
    """Download an asset from the given URL."""
    try:
        from openpi.shared.download import maybe_download, get_cache_dir
    except ImportError:
        print(
            "Error: openpi.shared.download module not found. "
            "Please install the project dependencies first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if output_dir:
        import os

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
        import os

        from openpi.shared.download import get_cache_dir

        cache_dir = Path(args.output) if args.output else get_cache_dir()
        print(f"Clearing existing cache in: {cache_dir}")
        import shutil

        for url in urls_to_download:
            import urllib.parse

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
