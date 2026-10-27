import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_tokenize_subtask_training():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    state = np.random.rand(5).astype(np.float32)
    tokens, token_mask, ar_mask, loss_mask = tokenizer.tokenize_subtask_training(
        "pick the box", "grab the handle", state, max_subtask_len=8
    )

    for array in (tokens, token_mask, ar_mask, loss_mask):
        assert array.shape == (64,)

    # Only the subtask tokens (plus its EOS) carry the loss, and they are the only autoregressive tokens.
    loss_idx = np.flatnonzero(loss_mask)
    assert loss_idx[0] > 0
    assert np.all(np.diff(loss_idx) == 1)
    assert tokens[loss_idx[-1]] == _tokenizer.PALIGEMMA_EOS_TOKEN
    assert np.all(ar_mask[: loss_idx[0]] == 0)
    # Everything from the first subtask token up to the last valid token (the action cue) is autoregressive.
    n_valid = int(token_mask.sum())
    assert np.all(ar_mask[loss_idx[0] : n_valid] == 1)
    assert np.all(ar_mask[n_valid:] == 0)
    assert tokenizer.detokenize(tokens[loss_idx], loss_mask[loss_idx]).startswith("grab the handle")


def test_tokenize_subtask_inference():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    state = np.random.rand(5).astype(np.float32)
    tokens, token_mask, ar_mask = tokenizer.tokenize_subtask_inference("pick the box", state, max_subtask_len=8)

    for array in (tokens, token_mask, ar_mask):
        assert array.shape == (64,)

    # Generation starts at the first reserved slot, which is followed by the (valid) action cue.
    start = int(np.argmax(~token_mask))
    reserved = token_mask[start : start + 8]
    assert not reserved.any()
    assert token_mask[start + 8 :].any()
    assert ar_mask[start] == 1


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
