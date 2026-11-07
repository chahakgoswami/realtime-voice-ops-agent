"""Mock telephony adapter sub-package (SIP/PSTN simulation)."""

from .adapter import MockTelephonyAdapter, CallEvent, CallEventType, CallState
from .scenarios import ScenarioPlayer, SCENARIOS

__all__ = [
    "MockTelephonyAdapter",
    "CallEvent",
    "CallEventType",
    "CallState",
    "ScenarioPlayer",
    "SCENARIOS",
]
