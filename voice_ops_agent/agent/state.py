"""Agent state machine definitions for the voice ops agent."""

from enum import Enum, auto


class AgentState(Enum):
    """
    States the agent can be in during a conversation turn.

    Transitions
    -----------
    IDLE -> LISTENING
    LISTENING -> TRANSCRIBING
    TRANSCRIBING -> LLM_STREAMING
    LLM_STREAMING -> TTS_PLAYING
    TTS_PLAYING -> IDLE          (normal completion)
    TTS_PLAYING -> INTERRUPTED   (user speaks during playback)
    INTERRUPTED -> LISTENING     (re-enter STT phase)
    Any -> ERROR                 (unrecoverable failure)
    """

    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    LLM_STREAMING = auto()
    TTS_PLAYING = auto()
    INTERRUPTED = auto()
    ERROR = auto()


# Valid state transitions
_VALID_TRANSITIONS: dict = {
    AgentState.IDLE: {AgentState.LISTENING},
    AgentState.LISTENING: {AgentState.TRANSCRIBING, AgentState.INTERRUPTED},
    AgentState.TRANSCRIBING: {AgentState.LLM_STREAMING, AgentState.ERROR},
    AgentState.LLM_STREAMING: {AgentState.TTS_PLAYING, AgentState.INTERRUPTED, AgentState.ERROR},
    AgentState.TTS_PLAYING: {AgentState.IDLE, AgentState.INTERRUPTED, AgentState.ERROR},
    AgentState.INTERRUPTED: {AgentState.LISTENING, AgentState.IDLE},
    AgentState.ERROR: {AgentState.IDLE},
}


class StateMachine:
    """
    A simple state machine that enforces valid transitions.

    Parameters
    ----------
    initial : AgentState
        Starting state (default IDLE).
    """

    def __init__(self, initial: AgentState = AgentState.IDLE) -> None:
        self._state = initial
        self._history: list = [initial]

    @property
    def state(self) -> AgentState:
        """Current state."""
        return self._state

    @property
    def history(self) -> list:
        """List of all states visited (oldest first)."""
        return list(self._history)

    def transition(self, new_state: AgentState) -> None:
        """
        Move to *new_state*, raising ValueError if the transition is invalid.

        Parameters
        ----------
        new_state : AgentState
            Target state.

        Raises
        ------
        ValueError
            If the transition from the current state to *new_state* is not
            in the allowed set.
        """
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {self._state.name} -> {new_state.name}. "
                f"Allowed: {[s.name for s in allowed]}"
            )
        self._state = new_state
        self._history.append(new_state)

    def can_transition(self, new_state: AgentState) -> bool:
        """Return True if transitioning to *new_state* is valid right now."""
        return new_state in _VALID_TRANSITIONS.get(self._state, set())

    def reset(self) -> None:
        """Reset to IDLE."""
        self._state = AgentState.IDLE
        self._history = [AgentState.IDLE]
