"""Escalation manager: evaluates rules and executes handoff logic."""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from .rules import EscalationRule
from .audit import AuditLogger


class EscalationReason(Enum):
    """Canonical reasons an escalation was triggered."""
    LOW_CONFIDENCE = auto()
    REPEATED_FAILURE = auto()
    EXPLICIT_REQUEST = auto()
    RULE_FIRED = auto()   # generic fallback for custom rules


@dataclass
class EscalationEvent:
    """
    Record of a single escalation occurrence.

    Attributes
    ----------
    timestamp : float
        Unix timestamp (time.time()) when escalation was triggered.
    rule_name : str
        Name of the rule that fired.
    reason : str
        Human-readable reason string.
    state_snapshot : dict
        Copy of the conversation state at the time of escalation.
    resolved : bool
        True once a human (or the simulation) has acknowledged the handoff.
    """
    timestamp: float
    rule_name: str
    reason: str
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "rule_name": self.rule_name,
            "reason": self.reason,
            "state_snapshot": self.state_snapshot,
            "resolved": self.resolved,
        }


# Mapping from rule name to canonical EscalationReason
_RULE_TO_REASON: Dict[str, EscalationReason] = {
    "low_confidence": EscalationReason.LOW_CONFIDENCE,
    "repeated_failure": EscalationReason.REPEATED_FAILURE,
    "explicit_request": EscalationReason.EXPLICIT_REQUEST,
}


class EscalationManager:
    """
    Evaluates a list of :class:`EscalationRule` objects against the current
    conversation state and, when a rule fires, executes the simulated handoff
    to a human agent.

    Parameters
    ----------
    rules : list of EscalationRule
        Ordered list of rules to evaluate.  Evaluation stops at the first
        rule that fires (short-circuit).
    audit_logger : AuditLogger, optional
        If provided, every escalation event is persisted via this logger.
    on_escalate : callable, optional
        ``on_escalate(event: EscalationEvent)`` called synchronously when
        escalation is triggered (useful for UI / testing hooks).
    human_agent_response : str
        Simulated response from the human agent (printed / returned during
        handoff simulation).
    """

    def __init__(
        self,
        rules: Optional[List[EscalationRule]] = None,
        audit_logger: Optional[AuditLogger] = None,
        on_escalate: Optional[Callable[["EscalationEvent"], None]] = None,
        human_agent_response: str = (
            "Thank you for holding. You are now being connected to a "
            "live support agent. Please wait a moment."
        ),
    ) -> None:
        self.rules: List[EscalationRule] = rules or []
        self.audit_logger = audit_logger
        self.on_escalate = on_escalate
        self.human_agent_response = human_agent_response
        self.events: List[EscalationEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_rule(self, rule: EscalationRule) -> None:
        """Append *rule* to the evaluation list."""
        self.rules.append(rule)

    def check(self, state: Dict[str, Any]) -> Optional[EscalationEvent]:
        """
        Evaluate all rules against *state*.

        Returns the first :class:`EscalationEvent` whose rule fired, or
        ``None`` if no escalation is warranted.

        This does **not** perform the handoff automatically; call
        :meth:`escalate` with the returned event to do so.

        Parameters
        ----------
        state : dict
            Current conversation state.

        Returns
        -------
        EscalationEvent or None
        """
        for rule in self.rules:
            if rule.should_escalate(state):
                event = EscalationEvent(
                    timestamp=time.time(),
                    rule_name=rule.name,
                    reason=rule.reason,
                    state_snapshot=dict(state),
                )
                return event
        return None

    def escalate(self, event: EscalationEvent) -> str:
        """
        Execute the handoff for *event*.

        1. Records the event internally.
        2. Logs it via the audit logger (if configured).
        3. Fires the ``on_escalate`` callback (if configured).
        4. Marks the event as resolved.
        5. Returns the simulated human-agent greeting.

        Parameters
        ----------
        event : EscalationEvent

        Returns
        -------
        str
            The simulated human agent response.
        """
        self.events.append(event)

        if self.audit_logger is not None:
            self.audit_logger.log(event)

        if self.on_escalate is not None:
            self.on_escalate(event)

        event.resolved = True
        return self.human_agent_response

    def check_and_escalate(
        self, state: Dict[str, Any]
    ) -> Optional[str]:
        """
        Convenience wrapper: ``check`` then ``escalate`` if warranted.

        Returns
        -------
        str or None
            The human-agent response if escalation occurred, else None.
        """
        event = self.check(state)
        if event is not None:
            return self.escalate(event)
        return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def escalation_count(self) -> int:
        """Total number of escalations that have been executed."""
        return len(self.events)

    def last_event(self) -> Optional[EscalationEvent]:
        """Return the most recent EscalationEvent, or None."""
        return self.events[-1] if self.events else None
