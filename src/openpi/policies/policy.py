from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        subtask_tokenizer: _tokenizer.PaligemmaTokenizer | None = None,
        subtask_classes: Sequence[str] | None = None,
        subtask_mode: str | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        # pi0.5 two-stage inference: sample the subtask first, then condition the actions on it.
        self._sample_subtask = (
            nnx_utils.module_jit(model.sample_subtask, static_argnames=("temperature",))
            if getattr(model, "sample_subtask_prediction", False)
            else None
        )
        # Fast alternative to the autoregressive stage: one prefix forward + a fixed-class readout.
        self._classify_subtask = (
            nnx_utils.module_jit(model.predict_subtask_class)
            if getattr(model, "subtask_classification", False)
            else None
        )
        self._subtask_tokenizer = None
        self._subtask_temperature = None
        self._subtask_classes = None
        # Default to the fastest stage the model supports; `generate` remains available for A/B comparisons.
        if subtask_mode is None:
            subtask_mode = "classifier" if self._classify_subtask is not None else (
                "generate" if self._sample_subtask is not None else None
            )
        if subtask_mode == "classifier":
            if self._classify_subtask is None:
                raise ValueError("`subtask_mode='classifier'` requires a model trained with a subtask classifier.")
            if subtask_classes is None:
                raise ValueError("`subtask_classes` is required when the subtask classifier is used.")
            self._subtask_classes = list(subtask_classes)
        elif subtask_mode == "generate":
            if self._sample_subtask is None:
                raise ValueError("`subtask_mode='generate'` requires a model with `sample_subtask_prediction`.")
            if subtask_tokenizer is None:
                raise ValueError("`subtask_tokenizer` is required when the model samples subtasks.")
            self._subtask_tokenizer = subtask_tokenizer
            self._subtask_temperature = model.subtask_temperature
        elif subtask_mode is not None:
            raise ValueError(f"Invalid subtask mode: {subtask_mode!r}")
        self._subtask_mode = subtask_mode
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    def _prepare_inputs(self, obs: dict) -> dict:
        """Run the input transforms and add a batch dimension."""
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

    def _finalize_outputs(self, outputs: dict) -> dict:
        """Remove the batch dimension and run the output transforms."""
        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        inputs, observation, sample_rng, subtask_text, subtask_source = self._run_subtask_stage(obs)

        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, observation, **self._sample_kwargs),
        }

        result = self._finalize_outputs(outputs)
        if subtask_text is not None:
            # Attached after the output transforms, which forward `actions` only.
            result["subtask"] = subtask_text
            result["subtask_source"] = subtask_source
        return result

    def _run_subtask_stage(
        self,
        obs: dict,
        prepared: tuple[dict, _model.Observation] | None = None,
    ) -> tuple[dict, _model.Observation, at.Array, str | None, str | None]:
        """Runs the high-level subtask stage and returns (inputs, observation, sample_rng, text, source).

        `prepared` lets callers that already ran the input transforms (RTC re-encodes the leftover) skip a
        redundant pass. In classifier mode the model inputs are rebuilt from the predicted subtask text so the
        action expert sees exactly the training prompt; in generate mode the prompt is already written in place.
        """
        if prepared is None:
            inputs = self._prepare_inputs(obs)
            observation = _model.Observation.from_dict(inputs)
        else:
            inputs, observation = prepared

        if self._subtask_mode is None:
            self._rng, sample_rng = jax.random.split(self._rng)
            return inputs, observation, sample_rng, None, None

        if self._subtask_mode == "classifier":
            # One prefix forward, no autoregressive decoding: the dominant latency saving of this path.
            self._rng, sample_rng = jax.random.split(self._rng)
            class_ids = self._classify_subtask(observation)
            subtask_text = self._subtask_classes[int(class_ids[0])]
            inputs = self._prepare_inputs({**obs, "subtask": subtask_text})
            observation = _model.Observation.from_dict(inputs)
            return inputs, observation, sample_rng, subtask_text, "classifier"

        self._rng, subtask_rng, sample_rng = jax.random.split(self._rng, 3)
        observation, subtask_tokens, subtask_mask = self._sample_subtask(
            subtask_rng, observation, temperature=self._subtask_temperature
        )
        return inputs, observation, sample_rng, self._decode_subtask(subtask_tokens, subtask_mask), "generate"

    def _decode_subtask(self, tokens: at.Array, mask: at.Array) -> str:
        return self._subtask_tokenizer.detokenize(np.asarray(tokens[0]), np.asarray(mask[0]))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
