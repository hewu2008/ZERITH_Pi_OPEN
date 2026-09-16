import argparse
import logging
import os
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import image_tools
from openpi_client import msgpack_numpy
from typing_extensions import override
import websockets.sync.client

subprocess.run(
    ["sudo", "rm", "-rf", "/dev/shm/zcm"],
    capture_output=True,
    text=True,
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.async_infer import InferenceWorker
from utils.async_infer import run_rtc_loop
from utils.real_env_sdk import make_real_env


DEFAULT_PROMPT = "grab all the ducks and put them into the basket"
FIRST_CHUNK_TIMEOUT = 30.0


def camera_aliases(camera_name: str) -> list[str]:
    return [
        camera_name,
        f"rs/{camera_name}",
        f"rs.{camera_name}",
        f"rs_{camera_name}",
    ]


def get_camera_image(images: dict, camera_name: str):
    for alias in camera_aliases(camera_name):
        if alias in images:
            return images[alias]

    rs_images = images.get("rs")
    if isinstance(rs_images, dict) and camera_name in rs_images:
        return rs_images[camera_name]

    raise KeyError(f"Camera '{camera_name}' not found in observation images.")


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket."""

    def __init__(self, host: str = "127.0.0.1", port: int = 55555) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self):
        return self._server_metadata

    def _wait_for_server(self):
        logging.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs):  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass

    def compress_image(self, image, depth=False):
        if depth:
            return image
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return buffer


class ActionSmooth:
    # Channels excluded from exponential smoothing, taken directly from the latest prediction:
    # grippers (7, 15), head (19, 20) and base velocity (21, 22).
    unsmoothed_channels = (7, 15, 19, 20, 21, 22)

    def __init__(self, worker: InferenceWorker, max_timesteps: int, query_frequency: int = 15) -> None:
        self.action_horizon = 50
        self.base_delay = 0
        self.query_frequency = query_frequency
        self.all_time_actions = np.zeros(
            [max_timesteps, max_timesteps + self.action_horizon - self.base_delay, 23],
            dtype=np.float32,
        )
        self.t = 0
        self.worker = worker
        self._submitted_ts = set()

    def submit_query(self, observation) -> None:
        """Submit an inference request for the current step, skipping duplicates."""
        if self.t in self._submitted_ts:
            return
        self.worker.submit(observation, query_t=self.t)
        self._submitted_ts.add(self.t)

    def _consume_results(self) -> None:
        while True:
            result = self.worker.poll()
            if result is None:
                break
            query_t, actions = result
            if query_t > self.t:
                continue
            action_keep = actions[self.base_delay : self.action_horizon, ...]
            self.all_time_actions[query_t, query_t : query_t + self.action_horizon - self.base_delay] = action_keep

    def has_actions(self) -> bool:
        actions_for_curr_step = self.all_time_actions[:, self.t]
        return bool(np.any(actions_for_curr_step != 0, axis=1).any())

    def get_action(self, observation):
        if self.t % self.query_frequency == 0:
            self.submit_query(observation)

        self._consume_results()

        actions_for_curr_step = self.all_time_actions[:, self.t]
        actions_populated = np.any(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        if actions_for_curr_step.shape[0] == 0:
            self.t += 1
            return None

        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / np.sum(exp_weights)
        exp_weights = exp_weights[:, np.newaxis]
        action = np.sum(actions_for_curr_step * exp_weights, axis=0)

        unsmoothed = list(self.unsmoothed_channels)
        action[unsmoothed] = actions_for_curr_step[-1, unsmoothed]

        self.t += 1
        return action


def prepare_observation(observation, client: WebsocketClientPolicy, camera_names: list[str], prompt: str):
    observation["state"] = observation["qpos"]
    observation["prompt"] = prompt

    for camera_name in camera_names:
        image = get_camera_image(observation["images"], camera_name)
        observation["images"][camera_name] = client.compress_image(
            image_tools.resize_with_pad(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 224, 224)
        )

    return observation


def warm_up(client: WebsocketClientPolicy, observation, args):
    logging.info("Warm up")
    observation = prepare_observation(observation, client, args.camera_names, "")

    for _ in range(args.warmup_steps):
        client.infer(observation)


def load_hdf5(ep_path):
    with h5py.File(ep_path, "r") as ep:
        state_arm = ep["/observation/state/arm/position"][:]
        state_effector = ep["/observation/state/effector/position"][:]
        state_waist = ep["/observation/state/waist/position"][:]
        state_head = ep["/observation/state/head/position"][:]
        state_base = ep["/observation/state/base/velocity"][:]

        state = np.concatenate(
            [
                state_arm[:, :7],
                state_effector[:, :-1],
                state_arm[:, 7:],
                state_effector[:, -1:],
                state_waist[:],
                state_head[:],
                state_base[:],
            ],
            axis=1,
        )
        action = state
        prompt = ep.attrs["task_name"]
        logging.info("Loaded task prompt: %s", prompt)
    return action


class GripperHysteresis:
    """Hysteresis for gripper close commands: enter closed above the close threshold,
    only reopen when the prediction drops below the open threshold."""

    closed_value = 1.3
    # Per-channel thresholds: left gripper (channel 7), right gripper (channel 15).
    channel_config = {
        7: {"close": 0.45, "open": 0.45},
        15: {"close": 0.45, "open": 0.45},
    }

    def __init__(self) -> None:
        self._closed = {idx: False for idx in self.channel_config}

    def apply(self, action):
        for idx, config in self.channel_config.items():
            if self._closed[idx]:
                if action[idx] < config["open"]:
                    self._closed[idx] = False
                else:
                    action[idx] = self.closed_value
            elif action[idx] > config["close"]:
                self._closed[idx] = True
                action[idx] = self.closed_value


def pin_head_action(action, data_action):
    """Pin the head joints to the reference pose from the initialization HDF5 frame."""
    if data_action is not None:
        action[-4:-2] = data_action[-4:-2]


def main(args):
    env = make_real_env(camera_names=args.camera_names)

    time.sleep(2)
    env.move_to_init_pose()

    openpi_client = WebsocketClientPolicy(host=args.host, port=args.port)

    observation = env.reset().observation
    observation["state"] = observation["qpos"]
    warm_up(openpi_client, observation, args)

    worker = InferenceWorker(openpi_client)
    action_smooth = ActionSmooth(
        worker,
        max_timesteps=args.num_steps,
        query_frequency=args.query_frequency,
    )
    worker.start()

    data_action = None
    if args.init_hdf5:
        data = load_hdf5(args.init_hdf5)
        data_action = data[args.init_frame_idx]
        env.move_to_target_joint(data_action[:-2])
        time.sleep(4)
        logging.info("Loaded initialization action: %s", data_action)

    logging.info("Paused before inference. Adjust the robot, then continue from pdb to start policy control.")
    import pdb

    pdb.set_trace()

    observation = env.get_observation().observation
    observation["state"] = observation["qpos"]
    observation = prepare_observation(observation, openpi_client, args.camera_names, args.prompt)

    # Synchronously wait for the first chunk so the motors are never driven by empty actions.
    action_smooth.submit_query(observation)
    if not worker.wait_for_first_result(FIRST_CHUNK_TIMEOUT):
        worker.stop()
        raise RuntimeError(f"Timed out waiting for the first action chunk after {FIRST_CHUNK_TIMEOUT:.0f}s")
    logging.info("First action chunk received")

    gripper = GripperHysteresis()

    def control_step(step):
        observation = env.get_observation().observation
        observation["state"] = observation["qpos"]
        observation = prepare_observation(observation, openpi_client, args.camera_names, args.prompt)
        action = action_smooth.get_action(observation)

        if action is None:
            logging.warning("step %d: no action available, holding current pose", step)
            env._set_joint_action(observation["qpos"][:-2])
            return

        action = np.copy(action)
        gripper.apply(action)
        if args.pin_head:
            pin_head_action(action, data_action)

        logging.info("action: %s", action)
        env._set_joint_action(action[:-2])

        result_age = worker.last_result_age()
        if result_age is not None and result_age > args.watchdog_timeout:
            logging.error("Inference stalled for %.1fs", result_age)

    run_rtc_loop(control_step, args.control_freq, args.num_steps)

    worker.stop()
    stats = worker.stats()
    if stats is not None:
        count, mean_time, max_time = stats
        logging.info(
            "Inference stats: %d calls, mean %.0f ms, max %.0f ms",
            count,
            mean_time * 1000,
            max_time * 1000,
        )

    logging.info("Inference completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="policy server host")
    parser.add_argument("--port", type=int, default=55555, help="policy server port")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="language instruction")
    parser.add_argument("--num_steps", type=int, default=20000, help="number of control steps")
    parser.add_argument("--warmup_steps", type=int, default=10, help="number of warmup inference calls")
    parser.add_argument("--control_freq", type=float, default=30.0, help="real-time control loop frequency in Hz")
    parser.add_argument(
        "--query_frequency",
        type=int,
        default=15,
        help="query the inference server every N control steps",
    )
    parser.add_argument(
        "--watchdog_timeout",
        type=float,
        default=2.0,
        help="seconds without inference results before logging a stall error",
    )
    parser.add_argument("--init_hdf5", type=str, required=True, help="optional HDF5 file used for initialization")
    parser.add_argument("--init_frame_idx", type=int, default=0, help="frame index used from the initialization HDF5")
    parser.add_argument(
        "--no_pin_head",
        dest="pin_head",
        action="store_false",
        default=True,
        help="disable pinning the head joints to the HDF5 reference pose",
    )
    parser.add_argument(
        "--camera_names",
        nargs="+",
        type=str,
        choices=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        help="camera names",
    )

    logging.basicConfig(level=logging.INFO, force=True)
    args = parser.parse_args()
    main(args)
