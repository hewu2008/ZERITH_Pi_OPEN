import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import optax
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import rtc as _rtc
from openpi.models import tokenizer as _tokenizer
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.train_subtask_prediction = config.train_subtask_prediction
        self.sample_subtask_prediction = config.sample_subtask_prediction
        self.subtask_loss_weight = config.subtask_loss_weight
        self.max_subtask_len = config.max_subtask_len
        self.subtask_temperature = config.subtask_temperature
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask.append(jnp.zeros(image_tokens.shape[:2], dtype=jnp.bool_))

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            if obs.token_ar_mask is None:
                # full attention between image and language inputs
                token_ar_mask = jnp.zeros_like(obs.tokenized_prompt_mask)
            else:
                # pi0-fast / pi0.5 subtask prediction: the answer tokens attend causally
                token_ar_mask = obs.token_ar_mask.astype(jnp.bool_)
            ar_mask.append(token_ar_mask)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        batch_size = obs.state.shape[0]
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask.append(
            jnp.broadcast_to(
                jnp.asarray([True] + ([False] * (self.action_horizon - 1))), action_expert_tokens.shape[:2]
            )
        )
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if not self.train_subtask_prediction:
            return action_loss
        # Paper Eq. 1: the flow-matching term plus a weighted cross-entropy on the subtask tokens.
        return action_loss + self.subtask_loss_weight * self.subtask_lm_loss(prefix_out, observation)[:, None]

    @at.typecheck
    def subtask_lm_loss(
        self, prefix_out: at.Float[at.Array, "b s emb"], observation: _model.Observation
    ) -> at.Float[at.Array, " b"]:
        """Cross-entropy of the subtask tokens, read out of the VLM prefix hidden states (paper Eq. 1, text term)."""
        assert observation.tokenized_prompt is not None, "Tokenized prompt is required for subtask prediction"
        assert observation.token_loss_mask is not None, "Token loss mask is required for subtask prediction"

        text_len = observation.tokenized_prompt.shape[1]
        # `prefix_out` is the image tokens followed by the prompt tokens; shifting by one aligns every hidden state
        # with its next token, so the trailing `Subtask: ` token predicts the first subtask token.
        logits = self.PaliGemma.llm(prefix_out[:, -text_len:-1], method="deembed")
        targets = observation.tokenized_prompt[:, 1:]
        weights = observation.token_loss_mask[:, 1:].astype(jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), targets)
        # Frames without an annotation carry an all-False mask, hence a zero (and gradient-free) loss.
        return jnp.sum(ce * weights, axis=-1) / jnp.clip(jnp.sum(weights, axis=-1), 1.0)

    @at.typecheck
    def suffix_velocity(
        self,
        obs: _model.Observation,
        x_t: _model.Actions,
        time: at.Float[at.Array, ""],
        kv_cache,
        prefix_mask: at.Bool[at.Array, "b s"],
    ) -> _model.Actions:
        """One suffix-only forward pass: the flow velocity `v_t` at `(x_t, time)`.

        Given the prefix KV cache computed once per sample, this is a pure
        function of `x_t` (and the scalar `time`), which is what allows
        Real-Time Chunking guidance to differentiate through it via `jax.vjp`
        (see `guided_sample_actions`).
        """
        batch_size = obs.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            obs, x_t, jnp.broadcast_to(time, batch_size)
        )
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        # other
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        # prefix tokens
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_mask.shape[1] + suffix_tokens.shape[1],
        )
        # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            v_t = self.suffix_velocity(observation, x_t, time, kv_cache, prefix_mask)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @at.typecheck
    def sample_subtask(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, temperature: float = 0.0
    ) -> tuple[_model.Observation, at.Int[at.Array, "b l"], at.Bool[at.Array, "b l"]]:
        """High-level stage of pi0.5 inference: generate the subtask autoregressively.

        `transforms.TokenizePi05SubtaskInputs` reserves `max_subtask_len` masked slots right after the
        `Subtask: ` prefix, so each sampled token is written in place and the `\\nAction: ` cue already sits behind
        the reserved block. The returned observation carries the generated subtask inside its prompt and can be fed
        straight to `sample_actions` (the low-level stage); the last two outputs are the generated ids and their
        validity mask, ready for detokenization.
        """
        if not self.pi05:
            raise ValueError("Subtask prediction is only supported for pi0.5 models.")

        observation = _model.preprocess_observation(None, observation, train=False)
        assert observation.tokenized_prompt is not None, "Tokenized prompt is required for subtask prediction"
        assert observation.tokenized_prompt_mask is not None, "Tokenized prompt mask is required for subtask prediction"
        tokens = observation.tokenized_prompt
        token_mask = observation.tokenized_prompt_mask
        batch_size = tokens.shape[0]
        rows = jnp.arange(batch_size)
        # First reserved slot, i.e. where the generated subtask starts (the prompt is prefix + slots + cue).
        start = jnp.argmax(~token_mask, axis=-1)

        def cond(carry):
            index, _, _, done = carry
            return (index < self.max_subtask_len) & ~jnp.all(done)

        def step(carry):
            index, tokens, token_mask, done = carry
            step_observation = dataclasses.replace(
                observation, tokenized_prompt=tokens, tokenized_prompt_mask=token_mask
            )
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(step_observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), _ = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
            )
            # The previous slot holds the hidden state that predicts this subtask token (first step: the last
            # prefix token, which is the trailing space of `Subtask: `).
            hidden = prefix_out[rows, jnp.clip(start + index - 1, 0, tokens.shape[1] - 1)][:, None, :]
            logits = self.PaliGemma.llm(hidden, method="deembed")[:, 0]
            if temperature > 0.0:
                next_token = jax.random.categorical(rng, logits / temperature, axis=-1)
            else:
                next_token = jnp.argmax(logits, axis=-1)
            next_token = jnp.where(done, 0, next_token).astype(jnp.int32)
            slot = start + index
            tokens = tokens.at[rows, slot].set(jnp.where(done, tokens[rows, slot], next_token))
            token_mask = token_mask.at[rows, slot].set(~done)
            return index + 1, tokens, token_mask, done | (next_token == _tokenizer.PALIGEMMA_EOS_TOKEN)

        _, tokens, token_mask, _ = jax.lax.while_loop(
            cond, step, (jnp.asarray(0), tokens, token_mask, jnp.zeros((batch_size,), dtype=jnp.bool_))
        )
        # Compact the reserved slots back into a (b, max_subtask_len) block; unwritten slots stay masked.
        slots = jnp.clip(start[:, None] + jnp.arange(self.max_subtask_len)[None, :], 0, tokens.shape[1] - 1)
        subtask_tokens = jnp.take_along_axis(tokens, slots, axis=1)
        subtask_mask = jnp.take_along_axis(token_mask, slots, axis=1)
        observation = dataclasses.replace(
            observation,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=token_mask,
            token_loss_mask=None,
        )
        return observation, subtask_tokens, subtask_mask

    @at.typecheck
    def guided_sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prev_actions: at.Float[at.Array, "b ah ad"],
        weights: at.Float[at.Array, " ah"],
        *,
        beta: float = 10.0,
        num_steps: int = 10,
        guidance_steps: int | None = None,
    ) -> _model.Actions:
        """Real-Time Chunking guided sampling (paper Algorithm 1, `GuidedInference`).

        Unlike `sample_actions`, the denoising loop is unrolled as a Python
        loop so each step can apply PiGDM guidance towards the previous
        chunk's leftover (`prev_actions`, already padded to the action horizon
        and aligned with the new chunk). `weights` is the prefix attention
        schedule from `_rtc.rtc_weights` and must cover the full horizon.

        `guidance_steps` of the *first* denoising steps use the exact VJP
        correction (Eq. pigdm1: `(I - t*dv/dx)^T @ err`); the remaining steps
        fall back to the identity-Jacobian approximation (`correction = err`,
        zero backward cost). This trades guidance accuracy for latency; the
        default (`None`) applies the exact VJP at every step. `num_steps` and
        `guidance_steps` must stay static (they unroll the loop at trace
        time). `beta` must stay a Python `float`: `@at.typecheck` rejects
        traced scalars, so callers that jit this method must list `beta` in
        `static_argnames` (see `RTCPolicy`).
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if guidance_steps is None:
            guidance_steps = num_steps
        action_weights = jnp.broadcast_to(weights, (self.action_horizon,))[None, :, None]  # (1, H, 1)
        prev_actions = jnp.asarray(prev_actions, dtype=jnp.float32)

        x_t = noise
        time = jnp.asarray(1.0, dtype=jnp.float32)
        for step_idx in range(num_steps):
            if step_idx < guidance_steps:
                # Exact VJP (Eq. pigdm1): differentiate `x1 = x - t*v(x)` wrt
                # `x` with the previous chunk as target. The cotangent for
                # the `v` output is zero, so this yields exactly
                # `(I - t*dv/dx)^T @ err` -- unlike the torch reference whose
                # correction silently degenerated to `err`.
                def f(x):
                    v = self.suffix_velocity(observation, x, time, kv_cache, prefix_mask)
                    return (x - time * v, v)

                (x1_t, v_t), vjp_fn = jax.vjp(f, x_t)
                err = (prev_actions - x1_t) * action_weights
                (correction,) = vjp_fn((err, jnp.zeros_like(v_t)))
            else:
                # Identity-Jacobian approximation (legacy behavior, no backward pass).
                v_t = self.suffix_velocity(observation, x_t, time, kv_cache, prefix_mask)
                x1_t = x_t - time * v_t
                err = (prev_actions - x1_t) * action_weights
                correction = err
            v_guided = v_t - _rtc.guidance_scale(time, beta) * correction
            x_t = x_t + dt * v_guided
            time = time + dt
        return x_t
