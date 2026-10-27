import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi05_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def _subtask_observation(
    config: pi0_config.Pi0Config,
    batch_size: int,
    *,
    prefix_len: int = 4,
    subtask_len: int = 0,
    gap: int = 0,
    cue_len: int = 2,
) -> _model.Observation:
    """Builds the pi0.5 subtask prompt layout: prefix | subtask (or `gap` reserved slots) | action cue."""
    length = config.max_token_len
    cue_start = prefix_len + gap + subtask_len
    tokens = np.zeros((batch_size, length), dtype=np.int32)
    token_mask = np.zeros((batch_size, length), dtype=bool)
    ar_mask = np.zeros((batch_size, length), dtype=np.int32)
    loss_mask = np.zeros((batch_size, length), dtype=bool)

    tokens[:, :prefix_len] = 10
    token_mask[:, :prefix_len] = True
    if subtask_len:
        tokens[:, prefix_len : prefix_len + subtask_len] = 11
        token_mask[:, prefix_len : prefix_len + subtask_len] = True
        loss_mask[:, prefix_len : prefix_len + subtask_len] = True
    tokens[:, cue_start : cue_start + cue_len] = 12
    token_mask[:, cue_start : cue_start + cue_len] = True
    ar_mask[:, prefix_len:] = 1

    return dataclasses.replace(
        config.fake_obs(batch_size),
        tokenized_prompt=jnp.asarray(tokens),
        tokenized_prompt_mask=jnp.asarray(token_mask),
        token_ar_mask=jnp.asarray(ar_mask),
        token_loss_mask=jnp.asarray(loss_mask),
    )


def test_pi05_subtask_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        train_subtask_prediction=True,
        sample_subtask_prediction=True,
        max_subtask_len=3,
    )
    model = config.create(key)
    batch_size = 2
    act = config.fake_act(batch_size)

    # Training: supervised subtask tokens (no reserved gap) add a cross-entropy term to the action loss.
    train_obs = _subtask_observation(config, batch_size, subtask_len=3)
    loss = nnx_utils.module_jit(model.compute_loss)(key, train_obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert jnp.isfinite(loss).all()

    # Inference: the subtask is generated into the reserved slots, then actions are decoded from the prompt.
    infer_obs = _subtask_observation(config, batch_size, gap=config.max_subtask_len)
    generate = nnx_utils.module_jit(model.sample_subtask, static_argnames=("temperature",))
    obs, subtask_tokens, subtask_mask = generate(key, infer_obs, temperature=0.0)
    assert subtask_tokens.shape == (batch_size, config.max_subtask_len)
    assert subtask_mask.shape == (batch_size, config.max_subtask_len)
    assert int(obs.tokenized_prompt_mask.sum()) >= batch_size * 4

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
