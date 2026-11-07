"""Context window manager: keeps conversation context within a token budget.

Tokens are estimated as ``len(text.split())`` (word count), which is a
reasonable proxy for LLM token counts without requiring a tokeniser.

When the accumulated context exceeds ``max_tokens`` the oldest turns are
dropped (FIFO) until the budget is satisfied.
"""

from collections import deque
from typing import Any, Deque, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import Turn  # avoid circular at runtime


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text using word-count heuristic."""
    return max(1, len(text.split()))


class ContextWindow:
    """
    A sliding-window context store that evicts the oldest turns when
    the total estimated token count exceeds the budget.

    Parameters
    ----------
    max_tokens : int
        Maximum number of estimated tokens to retain (default 2048).
    """

    def __init__(self, max_tokens: int = 2048) -> None:
        self.max_tokens = max_tokens
        self._turns: Deque["Turn"] = deque()
        self._token_counts: Deque[int] = deque()
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, turn: "Turn") -> None:
        """
        Add a turn to the window, evicting old turns if necessary.

        Parameters
        ----------
        turn : Turn
            The turn to add.
        """
        tokens = _estimate_tokens(turn.text)
        self._turns.append(turn)
        self._token_counts.append(tokens)
        self._total_tokens += tokens

        # Evict oldest turns until within budget
        while self._total_tokens > self.max_tokens and len(self._turns) > 1:
            self._total_tokens -= self._token_counts.popleft()
            self._turns.popleft()

    def get_context(self) -> List[Dict[str, Any]]:
        """
        Return the current windowed turns as LLM-compatible dicts.

        Returns
        -------
        list of dict
            Each dict has ``role`` and ``content`` keys.
        """
        return [t.to_context_dict() for t in self._turns]

    @property
    def total_tokens(self) -> int:
        """Estimated total tokens currently in the window."""
        return self._total_tokens

    @property
    def turn_count(self) -> int:
        """Number of turns currently retained in the window."""
        return len(self._turns)

    def clear(self) -> None:
        """Remove all turns from the window."""
        self._turns.clear()
        self._token_counts.clear()
        self._total_tokens = 0

    def __repr__(self) -> str:
        return (
            f"<ContextWindow turns={self.turn_count} "
            f"tokens={self._total_tokens}/{self.max_tokens}>"
        )
