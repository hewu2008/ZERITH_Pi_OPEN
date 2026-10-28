"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.zerith_joint_policy as zerith_joint_policy

import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("action",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


def _build_multi_camera_repack_transform() -> _transforms.Group:
    return _transforms.Group(
        inputs=[
            _transforms.RepackTransformWithFallback(
                {
                    "images": {
                        "cam_high": [
                            "observation.images.cam_high",
                            "observation.images.rs.cam_high",
                        ],
                        "cam_left_wrist": [
                            "observation.images.cam_left_wrist",
                            "observation.images.rs.cam_left_wrist",
                        ],
                        "cam_right_wrist": [
                            "observation.images.cam_right_wrist",
                            "observation.images.rs.cam_right_wrist",
                        ],
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                }
            )
        ]
    )


@dataclasses.dataclass(frozen=True)
class LeRobotZerithJointDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _build_multi_camera_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[zerith_joint_policy.MobilearxInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[zerith_joint_policy.MobilearxOutputs()],
        )

        # Action layout (23 dims): left arm 0-6, left gripper 7, right arm 8-14, right gripper 15,
        # waist 16-18, head yaw/pitch 19-20, base velocity 21-22.
        # Dataset check (clear_the_bin_box_20261010_v1): action[t] == state[t+1], and for the
        # arm/waist dims the delta targets are only 0.18-0.60x as spread out as the absolute ones,
        # so delta is the better-conditioned representation there. The grippers are bimodal
        # (q01/q99 ~ 0.01/0.90, with single-frame jumps up to ~0.6), hence they stay absolute.
        delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1, 5, -2)

        # Dims the policy does not need to command. They end up with different semantics because
        # ZeroDims is applied in a different space for each group (see the ordering note below):
        # - 21, 22: chassis/base velocity. In the dataset they are white noise (std ~0.02, lag-1
        #   autocorrelation ~0, 14-18% of frames above 0.01) and the base is not actuated at
        #   deployment, so zeroing them means "no motion": zero in the targets and absolute 0 in
        #   the outputs.
        # Note: dim 20 (head pitch) was zeroed for clear_the_bin_box_20261010 (constant ~0.51 rad
        # in 36/37 episodes), but in clear_the_bin_box_20261026 the pitch moves within 54/55
        # episodes (median per-episode range ~0.4 rad) and the pi05 deployment path
        # (real_env_sdk._set_head_joint) actually commands the head motors, so it stays under
        # policy control via delta actions.
        zeroed_dims = (21, 22)

        # Group.push() appends inputs but *prepends* outputs, so the actual execution order is:
        #   inputs:  MobilearxInputs -> DeltaActions -> ZeroDims
        #   outputs: ZeroDims -> AbsoluteActions -> MobilearxOutputs
        # ZeroDims must stay before AbsoluteActions (i.e. in delta space, where zeroed dims are
        # the delta targets). Dims 21/22 are not in delta_action_mask, so for them the two
        # placements are equivalent, but keep the ordering consistent for any delta dim that
        # gets zeroed in the future.
        data_transforms = data_transforms.push(
            inputs=[
                _transforms.DeltaActions(delta_action_mask),
                _transforms.ZeroDims(zeroed_dims),
            ],
            outputs=[
                _transforms.ZeroDims(zeroed_dims),
                _transforms.AbsoluteActions(delta_action_mask),
            ],
        )

        model_transforms = ModelTransformFactory(default_prompt='')(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi_eval"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 64
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 10
    # How often (in steps) to save checkpoints.
    save_interval: int = 200
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 1000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # Sanity-check mode: reuse a single fixed batch for every step (never advance the
    # data loader) and dump the raw (denormalized) input batch to the sanity_dump_dir.
    # Intended for verifying that the training forward/backward runs reproducibly on a
    # controlled input. Turn on only for debugging; not for real training.
    sanity_mode: bool = False
    # Directory (created on demand) where sanity raw inputs/outputs are saved.
    sanity_dump_dir: str = "./sanity_dump"

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir)).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.

_CONFIGS = [
    TrainConfig(
        # This name is used by compute_norm_stats.py, train.py, and serve_policy.py.
        name="pi0_zerith",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_horizon=50, action_expert_variant="gemma_300m"),
        data=LeRobotZerithJointDataConfig(
            # LeRobot stores the converted dataset under <LEROBOT_HOME>/<repo_id>.
            # Keep this value aligned with train.sh REPO_ID and the dataset path used during conversion.
            repo_id="hewu2008/clear_the_bin_box_20261010_v1",
            assets=AssetsConfig(
                # Normalization stats are loaded from <assets_dir>/<asset_id>/norm_stats.json.
                # The asset_id can match repo_id, or point to another robot/task's stats when reusing assets.
                assets_dir="./assets",
                asset_id="hewu2008/clear_the_bin_box_20261010_v1",
            ),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=30000,
            decay_lr=1e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/data/4T-1/hewu/model_zoo/pi0_base_params/params"),
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_horizon=50, action_expert_variant="gemma_300m"
        ).get_freeze_filter(),
        ema_decay=None,
        # Checkpoints are saved to <checkpoint_base_dir>/<config name>/<exp_name>/.
        checkpoint_base_dir="./openpi_checkpoints",
        batch_size=8,
        log_interval=1,
        save_interval=1000,
        num_train_steps=30000,
    ),
    TrainConfig(
        # Pi0.5 fine-tuning on the Zerith dataset. Differences from pi0_zerith:
        # - pi05=True: the state is discretized into language tokens (discrete_state_input=True,
        #   max_token_len becomes 200) instead of being fed as a continuous suffix input, and the
        #   action expert injects the flow matching timestep via adaRMSNorm (see Pi0Config).
        # - Quantile normalization (q01/q99) is enabled automatically for non-PI0 model types
        #   (see DataConfigFactory.create_base_config); the existing norm_stats.json files
        #   already contain q01/q99.
        # - LoRA fine-tuning of PaliGemma: the freeze filter freezes the base LLM weights
        #   (.*llm.*, excluding the action expert .*llm.*_1.*) and trains the LoRA adapters;
        #   the action expert is fully trained. The pi05 base checkpoint has no LoRA weights;
        #   CheckpointWeightLoader backfills them from the initialized model
        #   (missing_regex=".*lora.*").
        # - EMA is enabled (ema_decay=0.999).
        name="pi05_zerith",
        # discrete_state_input=True is redundant with pi05=True (Pi0Config.__post_init__ derives
        # it automatically), but is kept explicit for clarity.
        model=pi0_config.Pi0Config(pi05=True, discrete_state_input=True, paligemma_variant="gemma_2b_lora", action_horizon=50, action_expert_variant="gemma_300m"),
        data=LeRobotZerithJointDataConfig(
            # LeRobot stores the converted dataset under <LEROBOT_HOME>/<repo_id>.
            # Keep this value aligned with the dataset path used during conversion.
            repo_id="hewu2008/clear_the_bin_box_20261026_clean",
            assets=AssetsConfig(
                # Normalization stats are loaded from <assets_dir>/<asset_id>/norm_stats.json.
                assets_dir="./assets",
                asset_id="hewu2008/clear_the_bin_box_20261026_clean",
            ),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=60000,
            decay_lr=1e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/data/4T-1/hewu/model_zoo/pi05_base_params/params"),
        # Freeze the PaliGemma base LLM weights; train LoRA adapters + the action expert.
        # get_freeze_filter only depends on the two variants, so pi05 is irrelevant here.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_horizon=50, action_expert_variant="gemma_300m"
        ).get_freeze_filter(),
        ema_decay=0.999,
        # Checkpoints are saved to <checkpoint_base_dir>/<config name>/<exp_name>/.
        checkpoint_base_dir="./openpi_checkpoints",
        batch_size=8,
        log_interval=1,
        save_interval=1000,
        num_train_steps=60000,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
