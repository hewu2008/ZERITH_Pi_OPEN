"""Open-loop evaluation against a running policy websocket server.

Same measurements as `scripts/open_loop_eval.py` (per-episode action MSE/MAE plus subtask exact-match
accuracy and trajectory plots), but instead of loading a checkpoint in-process it queries the server
started by `scripts/remote_infer_server.py` over websockets - the exact path the robot uses.

Images are resized/padded to 224x224 and JPEG-compressed exactly like `robot_infer/scripts/test_pi0.py`,
so the per-request latency reported here matches the deployed pipeline. Requests carry no `rtc` payload,
so the server takes its plain path (subtask staged decoding still runs there) and no chunk bookkeeping
is involved.

Usage:
    python scripts/open_loop_eval_remote.py \
      --host 127.0.0.1 --port 55555 \
      --data_path /data/4T-2/dataset/hewu2008/part_sorting_20261027_subtask \
      --traj_ids 0 1 2 3 \
      --max_infer_time 240 \
      --action_horizon 30 \
      --default_prompt "Sort all objects into the correct bins using both arms"
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Reuse the dataset-side helpers (observation building, subtask labels, plotting) of the local evaluator.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from open_loop_eval import build_observation
from open_loop_eval import load_subtasks
from open_loop_eval import normalize_subtask
from open_loop_eval import plot_trajectory_results
from open_loop_eval import _to_numpy

JPEG_QUALITY = 100
IMAGE_SIZE = 224


def encode_image(image: np.ndarray, rgb_to_bgr: bool = True) -> np.ndarray:
    """RGB HWC uint8 -> JPEG bytes array, matching the robot client's wire format.

    `rgb_to_bgr` reproduces `robot_infer/scripts/test_pi0.py` exactly: it converts the RGB image to BGR
    before encoding, so the server receives channel-swapped arrays (JPEG encode/decode otherwise preserves
    channel order). Disable it to feed the server the same channel order the model was trained on.
    """
    if rgb_to_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image_tools.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer


def build_remote_observation(sample: dict, default_prompt: str | None, rgb_to_bgr: bool = True) -> dict:
    """Build the request payload the server expects: encoded images, raw state, prompt."""
    obs = build_observation(sample, default_prompt=default_prompt)
    obs["images"] = {name: encode_image(image, rgb_to_bgr) for name, image in obs["images"].items()}
    return obs


def evaluate_single_trajectory(
    client,
    dataset,
    traj_id: int,
    action_horizon: int,
    save_plot_path: str | None = None,
    max_infer_time: int = 10,
    default_prompt: str | None = None,
    subtasks: dict[int, str] | None = None,
    rgb_to_bgr: bool = True,
):
    start_id = int(dataset.episode_data_index["from"][traj_id])
    end_id = int(dataset.episode_data_index["to"][traj_id])

    gt_action_chunks = []
    state_joints_chunks = []
    pred_action_chunks = []
    subtask_pairs = []  # (ground-truth label, predicted label), one pair per chunk
    infer_times = []

    count = 0
    for data_id in range(start_id, end_id, action_horizon):
        sample = dataset[data_id]
        obs = build_remote_observation(sample, default_prompt=default_prompt, rgb_to_bgr=rgb_to_bgr)

        gt_action_chunk = _to_numpy(sample["action"])
        if gt_action_chunk.ndim == 1:
            gt_action_chunk = gt_action_chunk[np.newaxis, :]

        infer_start = time.time()
        result = client.infer(obs)
        infer_time = time.time() - infer_start
        infer_times.append(infer_time)

        pred_action_chunk = np.asarray(result["actions"])
        if pred_action_chunk.ndim == 1:
            pred_action_chunk = pred_action_chunk[np.newaxis, :]

        if "subtask" in result:
            gt_index = sample.get("subtask_index")
            gt_subtask = "" if gt_index is None else (subtasks or {}).get(int(_to_numpy(gt_index).item()), "")
            pred_subtask = str(result["subtask"])
            subtask_pairs.append((gt_subtask, pred_subtask))
            logging.info(
                "Chunk %d (frames %d-%d): infer %.4fs | subtask gt=%r pred=%r",
                count,
                data_id,
                data_id + action_horizon,
                infer_time,
                gt_subtask,
                pred_subtask,
            )

        # The model predicts `config.action_horizon` frames while the dataset window has `action_horizon`
        # frames; align per chunk so they can be compared frame by frame.
        n_frames = min(len(gt_action_chunk), len(pred_action_chunk))
        gt_action_chunks.append(gt_action_chunk[:n_frames])
        pred_action_chunks.append(pred_action_chunk[:n_frames])
        state_joints_chunks.append(obs["state"][:n_frames])
        count += 1

        if count >= max_infer_time:
            break

    gt_action_across_time = np.concatenate(gt_action_chunks, axis=0)
    state_joints_across_time = np.concatenate(state_joints_chunks, axis=0)
    pred_action_across_time = np.concatenate(pred_action_chunks, axis=0)

    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")
    if infer_times:
        logging.info(
            "Server latency across single traj: mean %.4fs, p95 %.4fs, max %.4fs",
            float(np.mean(infer_times)),
            float(np.percentile(infer_times, 95)),
            float(np.max(infer_times)),
        )

    # Exact match after normalization; chunks without an annotation are excluded (same mask as training).
    subtask_matches = 0
    subtask_total = 0
    if subtask_pairs:
        compared = [(normalize_subtask(gt), normalize_subtask(pred)) for gt, pred in subtask_pairs if gt]
        subtask_total = len(compared)
        subtask_matches = sum(gt == pred for gt, pred in compared)
        if subtask_total:
            logging.info(
                "Subtask accuracy across single traj: %d/%d = %.4f",
                subtask_matches,
                subtask_total,
                subtask_matches / subtask_total,
            )
            for gt, pred in [pair for pair in compared if pair[0] != pair[1]][:10]:
                logging.info("  subtask mismatch gt=%r pred=%r", gt, pred)

    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval_remote/traj_{traj_id}.jpeg",
    )

    return mse, mae, subtask_matches, subtask_total, infer_times


def main(
    host: str,
    port: int,
    data_path: str,
    traj_ids: list[int],
    save_plot_path: str,
    max_infer_time: int,
    action_horizon: int,
    default_prompt: str | None = None,
    rgb_to_bgr: bool = True,
):
    data_path = Path(data_path)
    if data_path.is_absolute() and data_path.exists():
        repo_id = data_path.name
        root = data_path
    else:
        repo_id = data_path
        root = None

    dataset_meta = LeRobotDatasetMetadata(repo_id, root=root, local_files_only=True)
    fps = dataset_meta.fps
    logging.info("Dataset FPS: %d", fps)

    subtasks = load_subtasks(root)
    logging.info("Loaded %d subtask labels", len(subtasks))

    dataset = LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={"action": [t / fps for t in range(action_horizon)]},
        local_files_only=True,
    )
    logging.info("Dataset length: %d", len(dataset))

    client = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    logging.info("Server metadata: %s", client.get_server_metadata())

    all_mse = []
    all_mae = []
    all_latencies = []
    subtask_matches_total = 0
    subtask_total = 0

    num_episodes = dataset.num_episodes
    for traj_id in traj_ids:
        if traj_id < 0 or traj_id >= num_episodes:
            logging.warning("Trajectory ID %d is out of range [0, %d). Skipping.", traj_id, num_episodes)
            continue

        logging.info("Running trajectory: %d", traj_id)
        mse, mae, traj_subtask_matches, traj_subtask_total, infer_times = evaluate_single_trajectory(
            client,
            dataset,
            traj_id,
            action_horizon=action_horizon,
            save_plot_path=os.path.join(save_plot_path, f"{traj_id}.png"),
            max_infer_time=max_infer_time,
            default_prompt=default_prompt,
            subtasks=subtasks,
            rgb_to_bgr=rgb_to_bgr,
        )
        logging.info("MSE for trajectory %d: %.6f, MAE: %.6f", traj_id, mse, mae)
        all_mse.append(mse)
        all_mae.append(mae)
        all_latencies.extend(infer_times)
        subtask_matches_total += traj_subtask_matches
        subtask_total += traj_subtask_total

    if all_mse:
        logging.info("Average MSE across all trajs: %.6f", np.mean(np.array(all_mse)))
        logging.info("Average MAE across all trajs: %.6f", np.mean(np.array(all_mae)))
    else:
        logging.info("No valid trajectories were evaluated.")
    if all_latencies:
        logging.info(
            "Server latency across all trajs: %d requests, mean %.4fs, p95 %.4fs, max %.4fs",
            len(all_latencies),
            float(np.mean(all_latencies)),
            float(np.percentile(all_latencies, 95)),
            float(np.max(all_latencies)),
        )
    if subtask_total:
        logging.info(
            "Subtask accuracy across all trajs: %d/%d = %.4f",
            subtask_matches_total,
            subtask_total,
            subtask_matches_total / subtask_total,
        )
    logging.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-loop evaluation against a policy websocket server")

    parser.add_argument("--host", type=str, default="127.0.0.1", help="policy server host")
    parser.add_argument("--port", type=int, default=55555, help="policy server port")
    parser.add_argument("--data_path", type=str, required=True, help="Path to a local LeRobot dataset")
    parser.add_argument("--traj_ids", type=int, nargs="+", default=[0], help="Episode IDs to evaluate")
    parser.add_argument(
        "--max_infer_time",
        type=int,
        default=10,
        help="Max number of action chunks to request per trajectory",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=50,
        help="Chunk cadence in frames (must match the training data sampling cadence)",
    )
    parser.add_argument("--save_plot_path", type=str, default="./open_loop_test/", help="Plot output directory")
    parser.add_argument(
        "--default_prompt",
        type=str,
        default=None,
        help="Prompt sent with every request (the dataset carries no prompt column)",
    )
    parser.add_argument(
        "--no_rgb_to_bgr",
        dest="rgb_to_bgr",
        action="store_false",
        default=True,
        help="Reproduce the robot client's RGB->BGR conversion before JPEG encoding (default: on, as deployed). "
        "Pass this flag to send the model's training channel order instead.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    os.makedirs(args.save_plot_path, exist_ok=True)

    main(
        host=args.host,
        port=args.port,
        data_path=args.data_path,
        traj_ids=args.traj_ids,
        save_plot_path=args.save_plot_path,
        max_infer_time=args.max_infer_time,
        action_horizon=args.action_horizon,
        default_prompt=args.default_prompt,
        rgb_to_bgr=args.rgb_to_bgr,
    )
