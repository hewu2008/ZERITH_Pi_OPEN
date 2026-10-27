import logging

import numpy as np
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download

# PaliGemma sentencepiece end-of-sequence id (used to stop subtask generation and to strip it when decoding).
PALIGEMMA_EOS_TOKEN = 1

# The text that follows a (predicted) subtask; the action expert is conditioned on both.
SUBTASK_ACTION_CUE = "\nAction: "


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = self._clean_text(prompt)
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            full_prompt = f"Task: {cleaned_text}, State: {self._state_str(state)};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens = self._truncate(tokens, "Token")
        token_mask = [True] * len(tokens)
        return self._pad(tokens, token_mask, [0] * len(tokens), [False] * len(tokens))[:2]

    def tokenize_subtask_training(
        self, prompt: str, subtask: str, state: np.ndarray | None = None, *, max_subtask_len: int = 20
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize pi0.5 inputs for supervised subtask prediction.

        Layout: `Task: .., State: ..;\nSubtask: <subtask>;\nAction: ` where the subtask (plus its EOS) is the
        autoregressive answer: it attends causally and carries the cross-entropy loss, everything else is prefix.
        """
        prefix_tokens = self._tokenizer.encode(self._subtask_prefix(prompt, state), add_bos=True)
        subtask_tokens = self._truncate(
            self._tokenizer.encode(f"{self._clean_text(subtask)};", add_eos=True), "Subtask", limit=max_subtask_len
        )
        if subtask_tokens[-1] != PALIGEMMA_EOS_TOKEN:
            # Truncated subtask: keep an EOS so generation still learns where to stop.
            subtask_tokens[-1] = PALIGEMMA_EOS_TOKEN
        cue_tokens = self._tokenizer.encode(SUBTASK_ACTION_CUE)

        tokens = prefix_tokens + subtask_tokens + cue_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * (len(subtask_tokens) + len(cue_tokens))
        loss_mask = [False] * len(prefix_tokens) + [True] * len(subtask_tokens) + [False] * len(cue_tokens)
        return self._pad(tokens, token_mask, ar_mask, loss_mask)

    def tokenize_subtask_inference(
        self, prompt: str, state: np.ndarray | None = None, *, max_subtask_len: int = 20
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize pi0.5 inputs for staged inference.

        The subtask is generated autoregressively, so `max_subtask_len` masked slots are reserved between the
        `Subtask: ` prefix and the `\nAction: ` cue; generation writes into them in place (no re-alignment needed
        because positions/attention are derived from the mask).
        """
        prefix_tokens = self._tokenizer.encode(self._subtask_prefix(prompt, state), add_bos=True)
        cue_tokens = self._tokenizer.encode(SUBTASK_ACTION_CUE)

        if len(prefix_tokens) + max_subtask_len + len(cue_tokens) > self._max_len:
            logging.warning(
                f"Subtask prompt ({len(prefix_tokens)} + {max_subtask_len} + {len(cue_tokens)}) exceeds max length "
                f"({self._max_len}); raise `max_token_len` or lower `max_subtask_len`."
            )

        tokens = prefix_tokens + ([0] * max_subtask_len) + cue_tokens
        token_mask = ([True] * len(prefix_tokens)) + ([False] * max_subtask_len) + ([True] * len(cue_tokens))
        # The reserved slots are already causal: while they are still masked they are invisible as keys, and once a
        # generated token is written the cumsum ordering keeps it before the `\nAction: ` cue, exactly as in training.
        ar_mask = ([0] * len(prefix_tokens)) + ([1] * max_subtask_len) + ([1] * len(cue_tokens))
        tokens, token_mask, ar_mask, _ = self._pad(tokens, token_mask, ar_mask, [False] * len(tokens))
        return tokens, token_mask, ar_mask

    def detokenize(self, tokens: np.ndarray, mask: np.ndarray | None = None) -> str:
        """Decode generated subtask ids, dropping padding and anything from the EOS token on."""
        token_list = np.asarray(tokens).astype(np.int32).tolist()
        if mask is not None:
            valid = np.asarray(mask).astype(bool).tolist()
            token_list = [token for token, keep in zip(token_list, valid, strict=True) if keep]
        else:
            token_list = [token for token in token_list if token != 0]

        if PALIGEMMA_EOS_TOKEN in token_list:
            token_list = token_list[: token_list.index(PALIGEMMA_EOS_TOKEN)]
        return self._tokenizer.decode(token_list).strip()

    def _clean_text(self, text: str) -> str:
        return text.strip().replace("_", " ").replace("\n", " ")

    def _state_str(self, state: np.ndarray) -> str:
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state))

    def _subtask_prefix(self, prompt: str, state: np.ndarray | None) -> str:
        cleaned_text = self._clean_text(prompt)
        if state is None:
            return f"Task: {cleaned_text};\nSubtask: "
        return f"Task: {cleaned_text}, State: {self._state_str(state)};\nSubtask: "

    def _truncate(self, tokens: list[int], name: str, *, limit: int | None = None) -> list[int]:
        limit = self._max_len if limit is None else min(limit, self._max_len)
        if len(tokens) > limit:
            logging.warning(
                f"{name} length ({len(tokens)}) exceeds max length ({limit}), truncating. "
                "Consider increasing the `max_token_len` in your model config if this happens frequently."
            )
            tokens = tokens[:limit]
        return tokens

    def _pad(
        self, tokens: list[int], token_mask: list[bool], ar_mask: list[int], loss_mask: list[bool]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        def pad(values: list, fill, dtype) -> np.ndarray:
            values = list(values)[: self._max_len]
            return np.asarray(values + [fill] * (self._max_len - len(values)), dtype=dtype)

        return (
            pad(tokens, 0, np.int32),
            pad(token_mask, False, np.bool_),
            pad(ar_mask, 0, np.int32),
            pad(loss_mask, False, np.bool_),
        )


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|")
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
