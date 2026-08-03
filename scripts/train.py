import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

import einops
from pathlib import Path as _Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata as _LeRobotDatasetMetadata, LeRobotDataset as _LeRobotDataset
import openpi.policies.policy_config as _policy_config


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


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def eval_step(
    action_scale: jnp.ndarray,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> at.Float[at.Array, ""]:
    """Run full denoising on the current batch and compute unnormalized MAE."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    observation, actions = batch
    eval_rng = jax.random.fold_in(rng, state.step)
    pred_actions = model.sample_actions(eval_rng, observation, num_steps=10)

    # Both pred and gt are in normalized space; multiply by action_scale to denormalize.
    return jnp.mean(jnp.abs(pred_actions - actions) * action_scale)


def _parse_image(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def run_open_loop_eval(
    config: _config.TrainConfig,
    checkpoint_dir: str,
    step: int,
) -> dict[str, float]:
    """Run open-loop evaluation on a test set and return metrics."""
    ckpt_path = _Path(checkpoint_dir) / str(step)
    policy = _policy_config.create_trained_policy(
        config,
        ckpt_path,
        default_prompt=config.eval_default_prompt,
    )
    logging.info("Eval policy created from checkpoint step %d", step)

    action_horizon = config.model.action_horizon

    data_path = _Path(config.eval_data_path)
    if data_path.is_absolute() and data_path.exists():
        repo_id = data_path.name
        root = data_path
    else:
        repo_id = config.eval_data_path
        root = None

    dataset_meta = _LeRobotDatasetMetadata(repo_id, root=root, local_files_only=True)
    fps = dataset_meta.fps
    delta_timestamps = {"action": [t / fps for t in range(action_horizon)]}
    dataset = _LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps, local_files_only=True)

    num_episodes = dataset.num_episodes
    all_mae = []
    all_mse = []

    for traj_id in config.eval_traj_ids:
        if traj_id < 0 or traj_id >= num_episodes:
            continue

        start_id = int(dataset.episode_data_index["from"][traj_id])
        end_id = int(dataset.episode_data_index["to"][traj_id])

        gt_chunks = []
        pred_chunks = []
        count = 0

        for data_id in range(start_id, end_id, action_horizon):
            sample = dataset[data_id]
            obs = {
                "images": {
                    "cam_high": _parse_image(sample["observation.images.cam_high"]),
                    "cam_left_wrist": _parse_image(sample["observation.images.cam_left_wrist"]),
                    "cam_right_wrist": _parse_image(sample["observation.images.cam_right_wrist"]),
                },
                "state": np.asarray(sample["observation.state"]),
            }
            if "prompt" in sample:
                obs["prompt"] = sample["prompt"]
            elif config.eval_default_prompt is not None:
                obs["prompt"] = config.eval_default_prompt

            gt_chunk = np.asarray(sample["action"])
            if gt_chunk.ndim == 1:
                gt_chunk = gt_chunk[np.newaxis, :]
            gt_chunks.append(gt_chunk)

            result = policy.infer(obs)
            pred_chunk = result["actions"]
            if pred_chunk.ndim == 1:
                pred_chunk = pred_chunk[np.newaxis, :]
            pred_chunks.append(pred_chunk)

            count += 1
            if count >= config.eval_max_infer_time:
                break

        gt = np.concatenate(gt_chunks, axis=0)
        pred = np.concatenate(pred_chunks, axis=0)
        # Truncate to same length.
        min_len = min(len(gt), len(pred))
        gt, pred = gt[:min_len], pred[:min_len]

        mse = float(np.mean((gt - pred) ** 2))
        mae = float(np.mean(np.abs(gt - pred)))
        all_mse.append(mse)
        all_mae.append(mae)
        logging.info("Eval traj %d: MSE=%.6f, MAE=%.6f", traj_id, mse, mae)

    if all_mae:
        avg_mae = float(np.mean(all_mae))
        avg_mse = float(np.mean(all_mse))
        logging.info("Eval avg: MSE=%.6f, MAE=%.6f", avg_mse, avg_mae)
        return {"eval_mse": avg_mse, "eval_mae": avg_mae}
    return {}


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # Compute action scale for unnormalized eval MAE.
    data_config = data_loader.data_config()
    action_dim = config.model.action_dim
    norm_stats = data_config.norm_stats
    if norm_stats is not None and "actions" in norm_stats:
        stats = norm_stats["actions"]
        if data_config.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            scale = (stats.q99 - stats.q01 + 1e-6) / 2.0
        else:
            scale = stats.std + 1e-6
        scale = np.pad(np.asarray(scale), (0, max(0, action_dim - len(scale))), constant_values=1.0)
        action_scale = jnp.asarray(scale[:action_dim])
    else:
        action_scale = jnp.ones((action_dim,))

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        functools.partial(eval_step, action_scale),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    lr_schedule_fn = config.lr_schedule.create()
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info["lr"] = float(lr_schedule_fn(step))
            # Run full denoising eval on current batch.
            with sharding.set_mesh(mesh):
                eval_mae = jax.device_get(peval_step(train_rng, train_state, batch))
            reduced_info["eval_mae"] = float(eval_mae)
            info_str = ", ".join(f"{k}={v:.6f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

            # Run open-loop evaluation if eval_data_path is configured.
            if config.eval_data_path is not None:
                logging.info("Waiting for checkpoint save to finish before eval...")
                checkpoint_manager.wait_until_finished()
                eval_metrics = run_open_loop_eval(config, str(config.checkpoint_dir), step)
                if eval_metrics:
                    wandb.log(eval_metrics, step=step)
                    pbar.write(f"Step {step} eval: {', '.join(f'{k}={v:.6f}' for k, v in eval_metrics.items())}")

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
