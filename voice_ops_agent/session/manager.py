"""Conversation session manager.

Maintains full turn history, speaker labels, and a context window that
keeps accumulated conversation within a configurable token budget.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from .context_window import ContextWindow
from .diarization import DiarizationStub


class Speaker(Enum):
    """Canonical speaker labels."""
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    HUMAN_AGENT = "human_agent"


@dataclass
class Turn:
    """
    A single conversation turn.

    Attributes
    ----------
    turn_id : str
        Unique identifier for this turn.
    index : int
        Zero-based turn index within the session.
    speaker : Speaker
        Who spoke during this turn.
    text : str
        Transcribed or generated text for the turn.
    timestamp : float
        Unix timestamp when the turn was recorded.
    metadata : dict
        Arbitrary metadata (e.g. confidence, tool calls, timings).
    """
    turn_id: str
    index: int
    speaker: Speaker
    text: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable dict representation."""
        return {
            "turn_id": self.turn_id,
            "index": self.index,
            "speaker": self.speaker.value,
            "text": self.text,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_context_dict(self) -> Dict[str, Any]:
        """Return a dict suitable for LLM context injection."""
        return {
            "role": self.speaker.value,
            "content": self.text,
        }


class SessionManager:
    """
    Manages a single voice call session: turn history, diarization, and
    context windowing.

    Parameters
    ----------
    session_id : str, optional
        Unique session identifier.  Auto-generated if not provided.
    max_tokens : int
        Approximate token budget for the context window (default 2048).
        Tokens are estimated as ``len(text.split())``.
    diarization : DiarizationStub, optional
        Speaker diarization stub.  Created with defaults if not provided.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_tokens: int = 2048,
        diarization: Optional[DiarizationStub] = None,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.created_at: float = time.time()
        self.max_tokens: int = max_tokens

        self._turns: List[Turn] = []
        self._context_window = ContextWindow(max_tokens=max_tokens)
        self._diarization = diarization or DiarizationStub()

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_turn(
        self,
        speaker: Speaker,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Turn:
        """
        Record a new turn in the session.

        Parameters
        ----------
        speaker : Speaker
            Who is speaking.
        text : str
            Content of the turn.
        metadata : dict, optional
            Arbitrary metadata to attach.

        Returns
        -------
        Turn
            The newly created turn.
        """
        turn = Turn(
            turn_id=str(uuid.uuid4()),
            index=len(self._turns),
            speaker=speaker,
            text=text,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        # Run diarization stub to assign/validate speaker label
        labelled_speaker = self._diarization.assign_speaker(
            text=text, hint=speaker
        )
        turn.metadata["diarized_speaker"] = labelled_speaker.value

        self._turns.append(turn)
        self._context_window.add(turn)
        return turn

    def add_user_turn(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Turn:
        """Convenience wrapper for ``add_turn(Speaker.USER, ...)``."""
        return self.add_turn(Speaker.USER, text, metadata)

    def add_agent_turn(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Turn:
        """Convenience wrapper for ``add_turn(Speaker.AGENT, ...)``."""
        return self.add_turn(Speaker.AGENT, text, metadata)

    def add_tool_turn(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Turn:
        """Convenience wrapper for ``add_turn(Speaker.TOOL, ...)``."""
        return self.add_turn(Speaker.TOOL, text, metadata)

    # ------------------------------------------------------------------
    # Context / history access
    # ------------------------------------------------------------------

    @property
    def turns(self) -> List[Turn]:
        """All turns recorded in this session (oldest first)."""
        return list(self._turns)

    @property
    def turn_count(self) -> int:
        """Total number of turns recorded."""
        return len(self._turns)

    def get_context(self) -> List[Dict[str, Any]]:
        """
        Return the windowed context suitable for injection into the LLM.

        The context is trimmed so that total estimated tokens remain
        within ``max_tokens``.

        Returns
        -------
        list of dict
            Each dict has ``role`` and ``content`` keys.
        """
        return self._context_window.get_context()

    def get_full_history(self) -> List[Dict[str, Any]]:
        """Return every turn's dict representation (not windowed)."""
        return [t.to_dict() for t in self._turns]

    def last_turn(self) -> Optional[Turn]:
        """Return the most recent turn, or None."""
        return self._turns[-1] if self._turns else None

    def last_user_turn(self) -> Optional[Turn]:
        """Return the most recent USER turn, or None."""
        for turn in reversed(self._turns):
            if turn.speaker == Speaker.USER:
                return turn
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a full serialisable snapshot of the session."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "turn_count": self.turn_count,
            "max_tokens": self.max_tokens,
            "turns": self.get_full_history(),
        }

    def __repr__(self) -> str:
        return (
            f"<SessionManager session_id={self.session_id!r} "
            f"turns={self.turn_count} max_tokens={self.max_tokens}>"
        )
