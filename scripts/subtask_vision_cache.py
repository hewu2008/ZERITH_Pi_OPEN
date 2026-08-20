"""Optimize subtask autoregressive sampling with vision cache + pre-allocated KV buffer.

Three optimizations:
1. Vision cache: pre-compute Siglip image tokens once (saves 64x vision encoder calls)
2. Pre-allocated KV buffer: fixed-size buffer + dynamic_update_slice (no growing concatenate)
3. fori_loop: single compiled graph for all autoregressive steps (no Python unroll dispatch)

Usage:
    import scripts.subtask_vision_cache
    scripts.subtask_vision_cache.patch()
"""

from __future__ import annotations

import logging

import einops
import jax
import jax.numpy as jnp

from openpi.models.pi0 import _last_valid_indices, _scatter_sequence, make_attn_mask

logger = logging.getLogger("openpi")


def _sample_subtask_tokens_cached(self, rng, observation):
    """Drop-in replacement using vision cache + pre-allocated KV buffer + fori_loop."""
    assert observation.tokenized_prompt is not None, "Tokenized prompt is required"
    assert observation.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"

    batch_size = observation.state.shape[0]
    max_subtask_len = self.max_subtask_len

    # 1. Pre-compute image tokens ONCE (vision cache)
    cached_image_tokens = []
    cached_image_input_masks = []
    cached_image_ar_masks = []

    for name in observation.images:
        image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
        cached_image_tokens.append(image_tokens)
        cached_image_input_masks.append(
            einops.repeat(observation.image_masks[name], "b -> b s", s=image_tokens.shape[1])
        )
        cached_image_ar_masks.append(jnp.zeros(image_tokens.shape[:2], dtype=jnp.bool_))

    # 2. Build prefix (images + prompt, NO subtask tokens)
    tokens_list = list(cached_image_tokens)
    input_mask_list = list(cached_image_input_masks)
    ar_mask_list = list(cached_image_ar_masks)

    tokenized_inputs = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
    tokens_list.append(tokenized_inputs)
    input_mask_list.append(observation.tokenized_prompt_mask)
    if observation.token_ar_mask is None:
        token_ar_mask = jnp.zeros_like(observation.tokenized_prompt_mask)
    else:
        token_ar_mask = observation.token_ar_mask.astype(jnp.bool_)
    ar_mask_list.append(token_ar_mask)

    prefix_tokens = jnp.concatenate(tokens_list, axis=1)
    prefix_mask = jnp.concatenate(input_mask_list, axis=1)
    prefix_ar_mask = jnp.concatenate(ar_mask_list, axis=1)

    prefix_buffer_len = prefix_tokens.shape[1]
    prefix_valid_len = jnp.sum(prefix_mask, axis=1)  # (b,)

    # 3. Forward prefix -> get prefix_out and prefix KV cache
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), prefix_kv = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
    )

    # Predict first subtask token from last valid hidden state
    last_indices = _last_valid_indices(prefix_mask)
    last_hidden = prefix_out[jnp.arange(batch_size), last_indices][:, None, :]
    logits = self.PaliGemma.llm(last_hidden, method="deembed")[:, 0]

    step_rng = rng
    if self.subtask_temperature > 0.0:
        step_rng, sample_rng = jax.random.split(step_rng)
        token_0 = jax.random.categorical(sample_rng, logits / self.subtask_temperature, axis=-1)
    else:
        token_0 = jnp.argmax(logits, axis=-1)
    token_0 = token_0.astype(jnp.int32)

    # Initialize output arrays
    output_tokens = jnp.zeros((batch_size, max_subtask_len), dtype=jnp.int32)
    output_mask = jnp.zeros((batch_size, max_subtask_len), dtype=jnp.bool_)
    done = jnp.zeros((batch_size,), dtype=jnp.bool_)

    # Write token_0 at position 0
    pos_zero = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    output_tokens = _scatter_sequence(output_tokens, pos_zero, token_0[:, None], ~done[:, None])
    output_mask = _scatter_sequence(output_mask, pos_zero, (~done)[:, None], ~done[:, None])
    done = done | (token_0 == self.subtask_eos_token)

    # 4. Pre-allocate KV cache buffer and initialize with prefix KV
    prefix_kv_k, prefix_kv_v = prefix_kv
    num_layers = prefix_kv_k.shape[0]
    num_kv_heads = prefix_kv_k.shape[3]
    head_dim = prefix_kv_k.shape[4]
    max_total_len = prefix_buffer_len + max_subtask_len

    buffer_k = jnp.zeros(
        (num_layers, batch_size, max_total_len, num_kv_heads, head_dim), dtype=prefix_kv_k.dtype
    )
    buffer_v = jnp.zeros(
        (num_layers, batch_size, max_total_len, num_kv_heads, head_dim), dtype=prefix_kv_v.dtype
    )
    buffer_k = jax.lax.dynamic_update_slice(buffer_k, prefix_kv_k, start_indices=(0, 0, 0, 0, 0))
    buffer_v = jax.lax.dynamic_update_slice(buffer_v, prefix_kv_v, start_indices=(0, 0, 0, 0, 0))

    # 5. while_loop: forward prev_token, predict next_token
    #    Early-exits when all batch elements are done (short subtasks skip wasted iterations)
    #    Token 0 predicted above; tokens 1..max_subtask_len-1 predicted in loop
    def cond_fn(carry):
        step_rng, prev_token, output_tokens, output_mask, done, buffer_k, buffer_v, step_idx = carry
        return jnp.logical_and(step_idx < max_subtask_len - 1, jnp.any(~done))

    def body_fn(carry):
        step_rng, prev_token, output_tokens, output_mask, done, buffer_k, buffer_v, step_idx = carry

        step_rng, sample_rng = jax.random.split(step_rng)

        # Embed previous token (1 token, PaliGemma expert only)
        prev_embedded = self.PaliGemma.llm(prev_token[:, None], method="embed")  # (b, 1, d)

        # Construct attention mask: (b, 1, max_total_len)
        subtask_valid = jnp.arange(max_subtask_len) <= step_idx
        subtask_valid = jnp.broadcast_to(subtask_valid, (batch_size, max_subtask_len))
        full_mask = jnp.concatenate([prefix_mask, subtask_valid], axis=1)
        new_attn_mask = full_mask[:, None, :]

        # Positions: logical position = prefix_valid_len + step_idx (per batch)
        new_positions = (prefix_valid_len + step_idx)[:, None]

        # Physical write position in the buffer
        write_pos = prefix_buffer_len + step_idx

        # Forward 1 token with pre-allocated KV buffer
        (token_out, _), updated_kv = self.PaliGemma.llm(
            [prev_embedded, None],
            mask=new_attn_mask,
            positions=new_positions,
            kv_cache=(buffer_k, buffer_v),
            kv_cache_write_pos=write_pos,
        )
        buffer_k, buffer_v = updated_kv

        # Predict next token from the new token's hidden state
        token_logits = self.PaliGemma.llm(token_out, method="deembed")[:, 0]
        if self.subtask_temperature > 0.0:
            next_token = jax.random.categorical(
                sample_rng, token_logits / self.subtask_temperature, axis=-1
            )
        else:
            next_token = jnp.argmax(token_logits, axis=-1)
        next_token = jnp.where(done, 0, next_token).astype(jnp.int32)

        # Write next_token at position step_idx+1
        next_is_valid = ~done
        pos_next = jnp.broadcast_to(step_idx + 1, (batch_size, 1))
        output_tokens = _scatter_sequence(
            output_tokens, pos_next, next_token[:, None], ~done[:, None]
        )
        output_mask = _scatter_sequence(
            output_mask, pos_next, next_is_valid[:, None], ~done[:, None]
        )
        done = done | (next_token == self.subtask_eos_token)

        prev_token = next_token
        step_idx = step_idx + 1

        return (step_rng, prev_token, output_tokens, output_mask, done, buffer_k, buffer_v, step_idx)

    init_carry = (step_rng, token_0, output_tokens, output_mask, done, buffer_k, buffer_v, jnp.int32(0))
    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    _, _, output_tokens, output_mask, _, _, _, _ = final_carry

    return output_tokens, output_mask


_original = None
_patched = False


def patch() -> None:
    global _original, _patched

    import openpi.models.pi0 as _pi0

    if not _patched:
        _original = _pi0.Pi0._sample_subtask_tokens
        _patched = True

    _pi0.Pi0._sample_subtask_tokens = _sample_subtask_tokens_cached
    logger.info("Patched Pi0._sample_subtask_tokens with vision cache + pre-allocated KV buffer + fori_loop")


def unpatch() -> None:
    global _original, _patched

    import openpi.models.pi0 as _pi0

    if _original is not None:
        _pi0.Pi0._sample_subtask_tokens = _original
        logger.info("Restored original Pi0._sample_subtask_tokens")
