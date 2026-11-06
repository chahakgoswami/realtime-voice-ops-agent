"""Escalation rules: configurable conditions that trigger agent handoff."""

import abc
from typing import Any, Dict, Optional


class EscalationRule(abc.ABC):
    """
    Abstract base class for escalation rules.

    Sub-classes implement ``should_escalate`` which inspects the current
    conversation state and returns True when the rule fires.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for this rule."""
        ...

    @property
    @abc.abstractmethod
    def reason(self) -> str:
        """Human-readable description of why escalation was triggered."""
        ...

    @abc.abstractmethod
    def should_escalate(self, state: Dict[str, Any]) -> bool:
        """
        Evaluate the rule against the current conversation *state* dict.

        Parameters
        ----------
        state : dict
            Keys that may be present:
            - ``confidence`` (float): STT confidence score of the last turn.
            - ``failure_count`` (int): consecutive tool/LLM failures so far.
            - ``transcript`` (str): latest user utterance.
            - ``turn_count`` (int): total turns completed.

        Returns
        -------
        bool
            True if this rule decides escalation should occur.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class ConfidenceThresholdRule(EscalationRule):
    """
    Escalates when the STT confidence score falls below *threshold*.

    Parameters
    ----------
    threshold : float
        Confidence below which we hand off to a human (default 0.5).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "low_confidence"

    @property
    def reason(self) -> str:
        return f"STT confidence below threshold ({self.threshold})"

    def should_escalate(self, state: Dict[str, Any]) -> bool:
        confidence = state.get("confidence", 1.0)
        return float(confidence) < self.threshold


class RepeatedFailureRule(EscalationRule):
    """
    Escalates when the number of consecutive failures reaches *max_failures*.

    Parameters
    ----------
    max_failures : int
        How many consecutive failures before escalating (default 3).
    """

    def __init__(self, max_failures: int = 3) -> None:
        self.max_failures = max_failures

    @property
    def name(self) -> str:
        return "repeated_failure"

    @property
    def reason(self) -> str:
        return f"Reached {self.max_failures} consecutive failures"

    def should_escalate(self, state: Dict[str, Any]) -> bool:
        return int(state.get("failure_count", 0)) >= self.max_failures


class ExplicitRequestRule(EscalationRule):
    """
    Escalates when the user's transcript contains an explicit request to
    speak with a human agent.

    Parameters
    ----------
    keywords : list of str
        Phrases that trigger escalation (case-insensitive).
    """

    _DEFAULT_KEYWORDS = [
        "human",
        "agent",
        "representative",
        "operator",
        "person",
        "speak to someone",
        "talk to someone",
        "connect me",
        "real person",
        "live agent",
        "escalate",
    ]

    def __init__(self, keywords: Optional[list] = None) -> None:
        self.keywords = [k.lower() for k in (keywords or self._DEFAULT_KEYWORDS)]

    @property
    def name(self) -> str:
        return "explicit_request"

    @property
    def reason(self) -> str:
        return "User explicitly requested a human agent"

    def should_escalate(self, state: Dict[str, Any]) -> bool:
        transcript = state.get("transcript", "").lower()
        return any(kw in transcript for kw in self.keywords)
