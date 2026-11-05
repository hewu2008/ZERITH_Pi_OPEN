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


def test_subtask_cue_index_is_shared_by_train_and_inference_layouts():
    """The classification readout must sit on the same prefix token in both token layouts."""
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        train_subtask_prediction=True,
        sample_subtask_prediction=True,
        max_subtask_len=3,
    )
    model = config.create(jax.random.key(0))
    batch_size = 2
    train_obs = _subtask_observation(config, batch_size, subtask_len=3)
    infer_obs = _subtask_observation(config, batch_size, gap=config.max_subtask_len)

    # One distinct value per prefix position so the selected position is identifiable.
    text_len = config.max_token_len
    num_image = 5
    prefix_out = jnp.broadcast_to(
        jnp.arange(num_image + text_len, dtype=jnp.float32)[None, :, None], (batch_size, num_image + text_len, 1)
    )

    train_hidden = model._subtask_cue_hidden(prefix_out, train_obs)
    infer_hidden = model._subtask_cue_hidden(prefix_out, infer_obs)
    # Both layouts put the first AR token right after the 4-token prefix, so the cue is the 4th text token.
    assert jnp.array_equal(train_hidden, infer_hidden)
    assert jnp.all(train_hidden == float(num_image + 4 - 1))
    # Training's `argmax(token_loss_mask) - 1` (the subtask CE readout) selects the same token.
    assert jnp.all(jnp.argmax(train_obs.token_loss_mask, axis=-1) - 1 == 4 - 1)


def test_pi05_subtask_classification_head():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        train_subtask_prediction=True,
        sample_subtask_prediction=True,
        max_subtask_len=3,
        subtask_classification=True,
        num_subtask_classes=4,
    )
    model = config.create(key)
    batch_size = 2
    act = config.fake_act(batch_size)

    train_obs = dataclasses.replace(
        _subtask_observation(config, batch_size, subtask_len=3),
        subtask_class=jnp.asarray([0, -1], dtype=jnp.int32),
    )
    _, _, class_loss = nnx_utils.module_jit(model.compute_loss_terms)(key, train_obs, act)
    assert class_loss.shape == (batch_size,)
    assert jnp.isfinite(class_loss).all()

    # Frames without an annotation are ignored: an all-unannotated batch contributes exactly zero.
    unannotated = dataclasses.replace(train_obs, subtask_class=jnp.full((batch_size,), -1, dtype=jnp.int32))
    _, _, unannotated_loss = nnx_utils.module_jit(model.compute_loss_terms)(key, unannotated, act)
    assert jnp.all(unannotated_loss == 0)

    # Inference path: one prefix forward classifies into the configured class range, no AR decoding.
    infer_obs = _subtask_observation(config, batch_size, gap=config.max_subtask_len)
    class_ids = nnx_utils.module_jit(model.predict_subtask_class)(infer_obs)
    assert class_ids.shape == (batch_size,)
    assert ((class_ids >= 0) & (class_ids < config.num_subtask_classes)).all()


def test_subtask_classification_config_validation():
    # The classifier reads out the pi0.5 subtask prompt layout, so both prerequisites must hold.
    with pytest.raises(ValueError):
        pi0_config.Pi0Config(pi05=True, train_subtask_prediction=True, subtask_classification=True)
    with pytest.raises(ValueError):
        pi0_config.Pi0Config(
            pi05=True, train_subtask_prediction=False, subtask_classification=True, num_subtask_classes=13
        )
    # Valid: classification needs pi0.5, the subtask prompt layout and a non-empty class set.
    config = pi0_config.Pi0Config(
        pi05=True, train_subtask_prediction=True, subtask_classification=True, num_subtask_classes=13
    )
    assert config.num_subtask_classes == 13


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
