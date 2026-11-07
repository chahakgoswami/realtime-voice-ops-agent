"""Configurable call scenarios for end-to-end telephony testing.

Each scenario defines a sequence of caller actions (audio chunks, DTMF
presses, hang-ups) that the ScenarioPlayer replays through a
MockTelephonyAdapter to simulate a complete phone call.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .adapter import CallEventType, MockTelephonyAdapter


@dataclass
class ScenarioStep:
    """
    A single step in a call scenario.

    Attributes
    ----------
    action : str
        One of: ``audio``, ``dtmf``, ``end_call``, ``sleep``, ``assert_response``.
    delay : float
        Seconds to wait before executing this step.
    params : dict
        Action-specific parameters.
    """
    action: str
    delay: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """
    A complete call scenario.

    Attributes
    ----------
    name : str
    description : str
    caller_id : str
    steps : list of ScenarioStep
    """
    name: str
    description: str = ""
    caller_id: str = "+15550001234"
    steps: List[ScenarioStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

def _make_silence_audio(n_bytes: int = 1024) -> bytes:
    """Return n_bytes of silence (all zeros)."""
    return b"\x00\x00" * (n_bytes // 2)


# A minimal tone (sine-wave approximation using repeated pattern)
_TONE_PATTERN = bytes([
    0, 0, 100, 0, 180, 0, 220, 0, 230, 0, 210, 0, 160, 0, 90, 0,
    0, 0, 156, 255, 76, 255, 36, 255, 26, 255, 46, 255, 96, 255, 166, 255,
]) * 32  # ~1 KB of a 440 Hz-ish tone at 16 kHz


SCENARIOS: Dict[str, Scenario] = {
    "happy_path": Scenario(
        name="happy_path",
        description="Caller asks about weather; agent responds; caller hangs up.",
        caller_id="+15550001111",
        steps=[
            ScenarioStep(action="audio", delay=0.1,
                         params={"audio": _TONE_PATTERN,
                                 "transcript_hint": "What is the weather today?"}),
            ScenarioStep(action="sleep", delay=0.2),
            ScenarioStep(action="end_call", delay=0.0),
        ],
    ),
    "escalation": Scenario(
        name="escalation",
        description="Caller immediately requests a human agent.",
        caller_id="+15550002222",
        steps=[
            ScenarioStep(action="audio", delay=0.1,
                         params={"audio": _TONE_PATTERN,
                                 "transcript_hint": "I want to speak to a human agent"}),
            ScenarioStep(action="sleep", delay=0.2),
            ScenarioStep(action="end_call", delay=0.0),
        ],
    ),
    "dtmf_menu": Scenario(
        name="dtmf_menu",
        description="Caller presses DTMF digits to navigate a menu.",
        caller_id="+15550003333",
        steps=[
            ScenarioStep(action="dtmf", delay=0.1, params={"digit": "1"}),
            ScenarioStep(action="dtmf", delay=0.2, params={"digit": "3"}),
            ScenarioStep(action="sleep", delay=0.1),
            ScenarioStep(action="end_call", delay=0.0),
        ],
    ),
    "multi_turn": Scenario(
        name="multi_turn",
        description="Caller exchanges multiple turns with the agent.",
        caller_id="+15550004444",
        steps=[
            ScenarioStep(action="audio", delay=0.05,
                         params={"audio": _TONE_PATTERN,
                                 "transcript_hint": "Hello I need help with my order"}),
            ScenarioStep(action="sleep", delay=0.15),
            ScenarioStep(action="audio", delay=0.05,
                         params={"audio": _TONE_PATTERN,
                                 "transcript_hint": "My order number is 12345"}),
            ScenarioStep(action="sleep", delay=0.15),
            ScenarioStep(action="end_call", delay=0.0),
        ],
    ),
    "silent_caller": Scenario(
        name="silent_caller",
        description="Caller sends only silence (no speech detected).",
        caller_id="+15550005555",
        steps=[
            ScenarioStep(action="audio", delay=0.1,
                         params={"audio": _make_silence_audio(2048),
                                 "transcript_hint": ""}),
            ScenarioStep(action="sleep", delay=0.1),
            ScenarioStep(action="end_call", delay=0.0),
        ],
    ),
}


class ScenarioPlayer:
    """
    Replays a :class:`Scenario` through a :class:`MockTelephonyAdapter`,
    executing each step in sequence.

    Parameters
    ----------
    adapter : MockTelephonyAdapter
    scenario : Scenario
    on_step : callable, optional
        ``on_step(step: ScenarioStep)`` – called before each step executes.
    """

    def __init__(
        self,
        adapter: MockTelephonyAdapter,
        scenario: Scenario,
        on_step: Optional[Callable[[ScenarioStep], None]] = None,
    ) -> None:
        self.adapter = adapter
        self.scenario = scenario
        self.on_step = on_step
        self.completed_steps: List[ScenarioStep] = []

    def play(self, blocking: bool = True) -> None:
        """
        Execute all scenario steps.

        Parameters
        ----------
        blocking : bool
            If True (default), runs synchronously.  If False, runs in a
            background daemon thread.
        """
        if blocking:
            self._run_steps()
        else:
            import threading
            t = threading.Thread(target=self._run_steps, daemon=True)
            t.start()

    def _run_steps(self) -> None:
        """Execute each step in order."""
        # Initiate the call first
        self.adapter.initiate_call(scenario=self.scenario.name)
        time.sleep(0.05)  # brief pause to let adapter process CALL_START

        for step in self.scenario.steps:
            if step.delay > 0:
                time.sleep(step.delay)

            if self.on_step:
                self.on_step(step)

            self._execute_step(step)
            self.completed_steps.append(step)

    def _execute_step(self, step: ScenarioStep) -> None:
        """Dispatch a single scenario step."""
        action = step.action
        params = step.params

        if action == "audio":
            audio_data = params.get("audio", b"")
            self.adapter.send_audio(audio_data)

        elif action == "dtmf":
            digit = params.get("digit", "0")
            self.adapter.send_dtmf(digit)

        elif action == "end_call":
            self.adapter.end_call()

        elif action == "sleep":
            duration = params.get("duration", 0.1)
            time.sleep(duration)

        elif action == "assert_response":
            # Check that the agent has sent at least one response
            keyword = params.get("keyword", "")
            timeout = params.get("timeout", 1.0)
            deadline = time.time() + timeout
            found = False
            while time.time() < deadline:
                for ev in self.adapter.sent_events:
                    text = ev.payload.get("text", "")
                    if keyword.lower() in text.lower():
                        found = True
                        break
                if found:
                    break
                time.sleep(0.01)
            # Store result in step params for callers to inspect
            step.params["_assert_passed"] = found

        else:
            pass  # unknown action – skip gracefully
