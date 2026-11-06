"""Home-subtask watchdog: trigger one return-to-home when the high-level subtask stays home."""

from __future__ import annotations

HOME_SUBTASK = "move both arms back to the home position"


def normalize_subtask(text: str) -> str:
    """Compare-ready form of a subtask string.

    Same normalisation as `scripts/open_loop_eval.py`: the tokenizer's training template terminates the
    subtask with `;` right before `\\nAction: `, so the generated text usually carries a trailing semicolon
    that the dataset labels (and the classifier lookup) do not have.
    """
    return text.strip().rstrip(";,. ").strip().lower()


class HomeSubtaskGuard:
    """Counts consecutive home-subtask predictions and fires once per home phase.

    Fed with `(text, seq)` for every finished inference chunk; `seq` deduplicates reads of the same chunk so
    the threshold counts chunks rather than control steps. After firing, the guard latches and only re-arms
    when a different subtask shows up, so a robot parked at home (whose policy keeps predicting home) cannot
    be sent home over and over.
    """

    def __init__(self, home_text: str = HOME_SUBTASK, consecutive: int = 3) -> None:
        self.home_text = normalize_subtask(home_text)
        self.consecutive = int(consecutive)
        self._last_seq = -1
        self._count = 0
        self._latched = False

    @property
    def count(self) -> int:
        """Current run length of consecutive home predictions."""
        return self._count

    @property
    def latched(self) -> bool:
        return self._latched

    def observe(self, text: str | None, seq: int) -> bool:
        """Feed one inference result; returns True exactly once, when the threshold is crossed."""
        if seq == self._last_seq:
            return False
        self._last_seq = seq
        if text is None:
            return False
        if normalize_subtask(str(text)) != self.home_text:
            self._count = 0
            self._latched = False
            return False
        if self._latched:
            return False
        self._count += 1
        if self._count >= self.consecutive:
            self._latched = True
            self._count = 0
            return True
        return False
