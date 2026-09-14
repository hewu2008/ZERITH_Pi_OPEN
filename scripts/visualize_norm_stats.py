"""Visualize normalization statistics from a norm_stats.json file.

Reads norm_stats.json and, for each key (e.g. state / actions), plots per-dimension
mean / std / q01 / q99, and prints a warning for any std=0 dimension (which can
amplify values during normalization).

Usage:
    python scripts/visualize_norm_stats.py <path> [--output out.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render without a display
import matplotlib.pyplot as plt


def _plot_key(ax, dims, key: str, stats: dict) -> None:
    mean = stats["mean"]
    std = stats["std"]
    q01 = stats["q01"]
    q99 = stats["q99"]

    x = list(range(dims))
    ax.bar(x, mean, color="C0", label="mean")
    ax.bar(x, [s * 5 for s in std], color="C1", alpha=0.5, label="std*x5")
    ax.plot(x, q01, "r--", marker="o", label="q01")
    ax.plot(x, q99, "g--", marker="o", label="q99")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("dimension")
    ax.set_title(f"{key}")
    ax.legend()


def main(json_path: str, output: str | None = None) -> None:
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    norm_stats = data["norm_stats"] if isinstance(data, dict) and "norm_stats" in data else data

    n_keys = len(norm_stats)
    fig, axes = plt.subplots(1, n_keys, figsize=(6 * n_keys + 2, 4), squeeze=False)
    axes = axes[0]

    for ax, (key, stats) in zip(axes, norm_stats.items()):
        dims = len(stats["mean"])
        _plot_key(ax, dims, key, stats)

        # Warn: std=0 amplifies normalized values by ~1e6x.
        for i, s in enumerate(stats["std"]):
            if s == 0:
                print(f"[WARN] {key} dim={i} std=0, consider replacing with 1.0")

    fig.suptitle(f"norm_stats: {json_path}")
    fig.tight_layout()

    out_path = Path(output) if output else json_path.with_name(f"{json_path.stem}_viz.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to: {out_path}")

    # Print per-dimension stats.
    for key, stats in norm_stats.items():
        mean = stats["mean"]
        std = stats["std"]
        q01 = stats["q01"]
        q99 = stats["q99"]
        print(f"\n[{key}]")
        for i in range(len(mean)):
            strd = f"{std[i]:.4f}" if std[i] != 0 else "0.0000 (WARN)"
            print(
                f"  dim={i:3d}  mean={mean[i]:9.4f}  std={strd}  "
                f"q01={q01[i]:9.4f}  q99={q99[i]:9.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="path to norm_stats.json")
    parser.add_argument("--output", default=None, help="output image path (default: next to json)")
    args = parser.parse_args()
    main(args.json_path, args.output)