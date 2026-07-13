import dataclasses
import logging
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class HuggingFaceWeightLoader(WeightLoader):
    """Loads weights from a local file or directory.

    This loader loads weights from local files and converts them to the OpenPI parameter format.
    It supports both PyTorch (.bin) and SafeTensors (.safetensors) formats, as well as NumPy (.npz) format.

    Args:
        local_path: Local path to the weight file (.bin, .safetensors, .npz) or directory containing weights
        filename: Optional specific filename to load if local_path is a directory (if None, will auto-detect)
    """

    local_path: str
    filename: str | None = None

    def load(self, params: at.Params) -> at.Params:
        try:
            import torch
            import safetensors
        except ImportError as e:
            raise ImportError(
                "To use HuggingFaceWeightLoader, please install the required dependencies: "
                "pip install torch safetensors"
            ) from e

        local_path = Path(self.local_path).expanduser().resolve()

        if not local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")

        if local_path.is_file():
            weight_path = local_path
        else:
            if self.filename:
                weight_path = local_path / self.filename
            else:
                weight_path = None
                for ext in ["safetensors", "bin", "npz"]:
                    candidates = list(local_path.glob(f"*.{ext}"))
                    if candidates:
                        weight_path = candidates[0]
                        break

        if weight_path is None or not weight_path.exists():
            raise FileNotFoundError(f"No weight files found in {local_path}")

        logger.info(f"Loading weights from: {weight_path}")

        if weight_path.suffix == ".npz":
            with weight_path.open("rb") as f:
                flat_params = dict(np.load(f, allow_pickle=False))
            loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        else:
            if weight_path.suffix == ".safetensors":
                import safetensors.torch
                pt_params = safetensors.torch.load_file(str(weight_path), device="cpu")
            else:
                pt_params = torch.load(str(weight_path), map_location="cpu", weights_only=True)

            loaded_params = {}
            for key, value in pt_params.items():
                if isinstance(value, torch.Tensor):
                    np_value = value.numpy()
                    loaded_params[key] = np_value

            loaded_params = flax.traverse_util.unflatten_dict(
                {k.replace(".", "/"): v for k, v in loaded_params.items()}, sep="/"
            )

        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
