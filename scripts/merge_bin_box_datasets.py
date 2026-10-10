#!/usr/bin/env python
"""Merge two LeRobot datasets into a new one, relabeling all tasks of the first
dataset to the task(s) of the second dataset.

Sources:
  clear_the_bin_box_20260910_v1  (128 episodes) -> new episode_index 0..127
  clear_the_bin_box_20261010_v1  (37  episodes) -> new episode_index 128..164

Output:
  clear_the_bin_box_20261010_v2 (165 episodes), single task = v2 task text.

Image bytes are preserved as-is (stored as struct<bytes, path> in the parquet).
Only the scalar columns `index` (global frame counter), `episode_index` and
`task_index` are rewritten. All episodes from both sources are labeled with the
merged dataset's single task (the one from the 20261010_v1 tasks file).

Run with the `pi0` conda env:  /home/jszn/miniconda3/envs/pi0/bin/python ...
"""

import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

V1 = pathlib.Path("/data/4T-2/dataset/EmbodiedLab/clear_the_bin_box_20260910_v1")
V2 = pathlib.Path("/data/4T-2/dataset/EmbodiedLab/clear_the_bin_box_20261010_v1")
OUT = pathlib.Path("/data/4T-2/dataset/EmbodiedLab/clear_the_bin_box_20261010_v2")

CHUNKS_SIZE = 1000
NEW_TASK_INDEX = 0


def load_episode_lengths(meta_dir: pathlib.Path) -> dict[int, int]:
    out: dict[int, int] = {}
    for line in (meta_dir / "episodes.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[int(d["episode_index"])] = int(d["length"])
    return out


def load_tasks(tasks_file: pathlib.Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for line in tasks_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        tasks[int(d["task_index"])] = d["task"]
    return tasks


def main() -> None:
    v2_tasks = load_tasks(V2 / "meta" / "tasks.jsonl")

    # Merged dataset's only task comes from the 20261010_v1 tasks file.
    if NEW_TASK_INDEX not in v2_tasks:
        raise ValueError(f"task_index {NEW_TASK_INDEX} not found in {V2}/meta/tasks.jsonl")
    merged_task = v2_tasks[NEW_TASK_INDEX]
    print(f"merged task: {merged_task!r}")

    out_chunk = OUT / f"data/chunk-{0:03d}"
    out_chunk.mkdir(parents=True, exist_ok=True)
    (OUT / "meta").mkdir(parents=True, exist_ok=True)

    new_episodes = []  # records for episodes.jsonl
    global_offset = 0
    new_ep = 0

    src_order = [V1, V2]

    for src in src_order:
        lengths = load_episode_lengths(src / "meta")
        for ep in sorted(lengths.keys()):
            chunk = ep // CHUNKS_SIZE
            src_file = src / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
            table = pq.read_table(str(src_file))
            n = table.num_rows
            if n != lengths[ep]:
                print(f"  WARN {src.name} ep{ep}: parquet rows={n} != episodes.jsonl length={lengths[ep]}")

            names = table.schema.names
            table = table.set_column(
                names.index("index"),
                "index",
                pa.array(range(global_offset, global_offset + n), type=pa.int64()),
            )
            table = table.set_column(
                names.index("episode_index"),
                "episode_index",
                pa.array([new_ep] * n, type=pa.int64()),
            )
            table = table.set_column(
                names.index("task_index"),
                "task_index",
                pa.array([NEW_TASK_INDEX] * n, type=pa.int64()),
            )

            out_file = out_chunk / f"episode_{new_ep:06d}.parquet"
            pq.write_table(table, str(out_file))
            new_episodes.append({"episode_index": new_ep, "tasks": [merged_task], "length": n})
            global_offset += n
            print(f"  {src.name} ep{ep} -> {OUT.name} ep{new_ep} (rows={n}, offset now {global_offset})")
            new_ep += 1

    # ---- meta/info.json ----
    info = json.loads((V1 / "meta" / "info.json").read_text())
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = global_offset
    info["total_tasks"] = 1
    info["total_videos"] = 0
    info["total_chunks"] = 1
    info["chunks_size"] = CHUNKS_SIZE
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    (OUT / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    # ---- meta/tasks.jsonl ----
    with (OUT / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": NEW_TASK_INDEX, "task": merged_task}) + "\n")

    # ---- meta/episodes.jsonl ----
    with (OUT / "meta" / "episodes.jsonl").open("w") as f:
        for e in new_episodes:
            f.write(json.dumps(e) + "\n")

    print(f"done: {len(new_episodes)} episodes, {global_offset} frames -> {OUT}")


if __name__ == "__main__":
    main()