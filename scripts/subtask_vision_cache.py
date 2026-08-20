"""Cache Siglip vision encoder output for subtask autoregressive sampling.

The original `Pi0._sample_subtask_tokens` calls `embed_prefix` at every
autoregressive step (64 steps), which re-encodes all 3 camera images through
Siglip vision encoder each time. Since images don't change during subtask
generation, this wastes ~90% of the per-step compute.

This module monkey-patches `_sample_subtask_tokens` to pre-compute image tokens
once and reuse them in every step. The prompt (which includes generated subtask
tokens) is still re-embedded each step since it changes.

Expected speedup: ~10x for the subtask sampling phase (3-6s -> 0.3-0.6s).

Usage:
    import scripts.subtask_vision_cache
    scripts.subtask_vision_cache.patch()

    # Then create policy as usual.
    policy = _policy_config.create_trained_policy(config, checkpoint_dir, ...)
"""

from __future__ import annotations

import logging

import einops
import jax
import jax.numpy as jnp

from openpi.models.pi0 import _last_valid_indices, _scatter_sequence, make_attn_mask

logger = logging.getLogger("openpi")


def _sample_subtask_tokens_vision_cached(self, rng, observation):
    """Drop-in replacement for Pi0._sample_subtask_tokens that caches vision encoder output."""
    assert observation.tokenized_prompt is not None, "Tokenized prompt is required for subtask prediction"
    assert observation.tokenized_prompt_mask is not None, "Tokenized prompt mask is required for subtask prediction"

    batch_size = observation.state.shape[0]
    output_tokens = jnp.zeros((batch_size, self.max_subtask_len), dtype=jnp.int32)
    output_mask = jnp.zeros((batch_size, self.max_subtask_len), dtype=jnp.bool_)
    done = jnp.zeros((batch_size,), dtype=jnp.bool_)

    # =====================================================================
    # Pre-compute image tokens ONCE (the expensive Siglip vision encoder).
    # =====================================================================
    cached_image_tokens = []
    cached_image_input_masks = []
    cached_image_ar_masks = []

    for name in observation.images:
        image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
        cached_image_tokens.append(image_tokens)
        cached_image_input_masks.append(
            einops.repeat(
                observation.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            )
        )
        cached_image_ar_masks.append(jnp.zeros(image_tokens.shape[:2], dtype=jnp.bool_))

    # =====================================================================
    # fori_loop: same structure as original, but embed_prefix is replaced by
    # a version that reuses cached image tokens.
    # =====================================================================
    def step(index, carry):
        step_rng, tokens, token_mask, is_done = carry
        step_rng, sample_rng = jax.random.split(step_rng)

        prompt_observation = self._with_generated_subtask_prompt(
            observation, tokens, token_mask, include_action_suffix=False
        )

        # Rebuild prefix using cached image tokens (skip vision encoder!)
        tokens_list = list(cached_image_tokens)
        input_mask_list = list(cached_image_input_masks)
        ar_mask_list = list(cached_image_ar_masks)

        if prompt_observation.tokenized_prompt is not None:
            assert prompt_observation.tokenized_prompt_mask is not None
            tokenized_inputs = self.PaliGemma.llm(prompt_observation.tokenized_prompt, method="embed")
            tokens_list.append(tokenized_inputs)
            input_mask_list.append(prompt_observation.tokenized_prompt_mask)
            if prompt_observation.token_ar_mask is None:
                token_ar_mask = jnp.zeros_like(prompt_observation.tokenized_prompt_mask)
            else:
                token_ar_mask = prompt_observation.token_ar_mask.astype(jnp.bool_)
            ar_mask_list.append(token_ar_mask)

        prefix_tokens = jnp.concatenate(tokens_list, axis=1)
        prefix_mask = jnp.concatenate(input_mask_list, axis=1)
        prefix_ar_mask = jnp.concatenate(ar_mask_list, axis=1)

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        last_indices = _last_valid_indices(prefix_mask)
        last_hidden = prefix_out[jnp.arange(prefix_out.shape[0]), last_indices][:, None, :]
        logits = self.PaliGemma.llm(last_hidden, method="deembed")[:, 0]

        if self.subtask_temperature > 0.0:
            next_token = jax.random.categorical(sample_rng, logits / self.subtask_temperature, axis=-1)
        else:
            next_token = jnp.argmax(logits, axis=-1)

        next_token = jnp.where(is_done, 0, next_token).astype(jnp.int32)
        next_is_valid = ~is_done
        positions = jnp.broadcast_to(index, (batch_size, 1))
        tokens = _scatter_sequence(tokens, positions, next_token[:, None], ~is_done[:, None])
        token_mask = _scatter_sequence(token_mask, positions, next_is_valid[:, None], ~is_done[:, None])
        is_done = is_done | (next_token == self.subtask_eos_token)
        return step_rng, tokens, token_mask, is_done

    _, output_tokens, output_mask, _ = jax.lax.fori_loop(
        0, self.max_subtask_len, step, (rng, output_tokens, output_mask, done)
    )
    return output_tokens, output_mask


_original = None
_patched = False


def patch() -> None:
    global _original, _patched

    import openpi.models.pi0 as _pi0

    if not _patched:
        _original = _pi0.Pi0._sample_subtask_tokens
        _patched = True

    _pi0.Pi0._sample_subtask_tokens = _sample_subtask_tokens_vision_cached
    logger.info("Patched Pi0._sample_subtask_tokens with vision-cache optimized version")


def unpatch() -> None:
    global _original, _patched

    import openpi.models.pi0 as _pi0

    if _original is not None:
        _pi0.Pi0._sample_subtask_tokens = _original
        logger.info("Restored original Pi0._sample_subtask_tokens")
