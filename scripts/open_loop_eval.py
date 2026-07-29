import argparse
import logging
import os
import sys
import time
from pathlib import Path

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_image(image) -> np.ndarray:
    image = _to_numpy(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def build_observation(sample: dict, default_prompt: str | None = None) -> dict:
    obs = {
        "images": {
            "cam_high": _parse_image(sample["observation.images.cam_high"]),
            "cam_left_wrist": _parse_image(sample["observation.images.cam_left_wrist"]),
            "cam_right_wrist": _parse_image(sample["observation.images.cam_right_wrist"]),
        },
        "state": _to_numpy(sample["observation.state"]),
    }
    if "prompt" in sample:
        obs["prompt"] = sample["prompt"]
    elif default_prompt is not None:
        obs["prompt"] = default_prompt
    return obs


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    action_horizon: int,
    save_plot_path: str,
) -> None:
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))
    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]

    fig.suptitle(
        f"Trajectory {traj_id}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro", label="inference point")
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)
    plt.close()


def evaluate_single_trajectory(
    policy,
    dataset,
    traj_id: int,
    action_horizon: int,
    save_plot_path: str | None = None,
    max_infer_time: int = 10,
    default_prompt: str | None = None,
):
    start_id = int(dataset.episode_data_index["from"][traj_id])
    end_id = int(dataset.episode_data_index["to"][traj_id])

    gt_action_chunks = []
    state_joints_chunks = []
    pred_action_chunks = []

    count = 0
    for data_id in range(start_id, end_id, action_horizon):
        sample = dataset[data_id]
        obs = build_observation(sample, default_prompt=default_prompt)

        gt_action_chunk = _to_numpy(sample["action"])
        if gt_action_chunk.ndim == 1:
            gt_action_chunk = gt_action_chunk[np.newaxis, :]

        gt_action_chunks.append(gt_action_chunk)
        state_joints_chunks.append(obs["state"])

        infer_start = time.time()
        result = policy.infer(obs)
        infer_time = time.time() - infer_start
        logging.info("Infer time: %.4fs", infer_time)

        pred_action_chunk = result["actions"]
        if pred_action_chunk.ndim == 1:
            pred_action_chunk = pred_action_chunk[np.newaxis, :]

        pred_action_chunks.append(pred_action_chunk)
        count += 1

        if count >= max_infer_time:
            break

    gt_action_across_time = np.concatenate(gt_action_chunks, axis=0)
    state_joints_across_time = np.concatenate(state_joints_chunks, axis=0)
    pred_action_across_time = np.concatenate(pred_action_chunks, axis=0)

    pred_action_across_time = np.array(pred_action_across_time)
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    return mse, mae


def main(
    config_name: str,
    checkpoint_dir: str,
    data_path: str,
    traj_ids: list[int],
    save_plot_path: str,
    max_infer_time: int,
    default_prompt: str | None = None,
):
    config = _config.get_config(config_name)
    logging.info("Loaded config: %s", config_name)

    policy = _policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        default_prompt=default_prompt,
    )
    logging.info("Policy created successfully")

    action_horizon = config.model.action_horizon
    logging.info("Action horizon: %d", action_horizon)

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

    delta_timestamps = {"action": [t / fps for t in range(action_horizon)]}
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps, local_files_only=True)
    logging.info("Dataset length: %d", len(dataset))
    logging.info("Running evaluation on trajectories: %s", traj_ids)

    all_mse = []
    all_mae = []

    num_episodes = dataset.num_episodes
    for traj_id in traj_ids:
        if traj_id < 0 or traj_id >= num_episodes:
            logging.warning("Trajectory ID %d is out of range [0, %d). Skipping.", traj_id, num_episodes)
            continue

        logging.info("Running trajectory: %d", traj_id)
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            action_horizon=action_horizon,
            save_plot_path=os.path.join(save_plot_path, f"{traj_id}.png"),
            max_infer_time=max_infer_time,
            default_prompt=default_prompt,
        )
        logging.info("MSE for trajectory %d: %.6f, MAE: %.6f", traj_id, mse, mae)
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info("Average MSE across all trajs: %.6f", avg_mse)
        logging.info("Average MAE across all trajs: %.6f", avg_mae)
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pi model open loop evaluation")

    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Training config name (e.g., 'pi0_zerith', 'pi05_zerith')",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., openpi_checkpoints/pi0_zerith/exp/10000)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to LeRobot dataset (local path or repo_id)",
    )
    parser.add_argument(
        "--traj_ids",
        type=int,
        nargs="+",
        default=[0],
        help="Trajectory IDs to evaluate",
    )
    parser.add_argument(
        "--max_infer_time",
        type=int,
        default=10,
        help="Max number of action chunks to infer per trajectory",
    )
    parser.add_argument(
        "--save_plot_path",
        type=str,
        default="./open_loop_test/",
        help="Directory to save evaluation plots",
    )
    parser.add_argument(
        "--default_prompt",
        type=str,
        default=None,
        help="Default prompt for the policy (if dataset doesn't have one)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    os.makedirs(args.save_plot_path, exist_ok=True)

    main(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        traj_ids=args.traj_ids,
        save_plot_path=args.save_plot_path,
        max_infer_time=args.max_infer_time,
        default_prompt=args.default_prompt,
    )