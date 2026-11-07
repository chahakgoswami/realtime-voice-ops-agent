"""End-to-end call handler: wires SessionManager + AgentLoop + TelephonyAdapter.

This module provides :class:`CallHandler`, which orchestrates a complete
simulated phone call:

1. Receives :class:`CallEvent` objects from the telephony adapter.
2. Routes audio events through STT → LLM → TTS via :class:`AgentLoop`.
3. Records every turn in a :class:`SessionManager`.
4. Sends agent responses back through the adapter.
5. Applies escalation rules at each turn.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from voice_ops_agent.agent.loop import AgentLoop
from voice_ops_agent.escalation.manager import EscalationManager
from voice_ops_agent.telephony.adapter import (
    CallEvent,
    CallEventType,
    CallState,
    MockTelephonyAdapter,
)
from voice_ops_agent.session.manager import SessionManager, Speaker


class CallHandler:
    """
    Orchestrates a single phone-call session.

    Parameters
    ----------
    adapter : MockTelephonyAdapter
        Telephony adapter to receive events from and send responses to.
    agent_loop : AgentLoop
        Pre-configured agent loop.
    session : SessionManager, optional
        Session manager to use.  Created automatically if not provided.
    escalation_manager : EscalationManager, optional
        Escalation manager evaluated after every user turn.
    on_turn_complete : callable, optional
        ``on_turn_complete(turn_result: dict)`` called after each turn.
    max_turns : int
        Maximum turns before hanging up automatically (default 20).
    """

    def __init__(
        self,
        adapter: MockTelephonyAdapter,
        agent_loop: AgentLoop,
        session: Optional[SessionManager] = None,
        escalation_manager: Optional[EscalationManager] = None,
        on_turn_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_turns: int = 20,
    ) -> None:
        self.adapter = adapter
        self.agent_loop = agent_loop
        self.session = session or SessionManager()
        self.escalation_manager = escalation_manager
        self.on_turn_complete = on_turn_complete
        self.max_turns = max_turns

        self._call_ended = threading.Event()
        self._pending_audio: List[bytes] = []
        self._lock = threading.Lock()
        self.turn_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_call(
        self,
        audio_chunks: Optional[List[bytes]] = None,
        transcript_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle a complete call, processing audio and generating responses.

        This is the high-level entry point for end-to-end scenario testing.
        It runs synchronously and returns when the call ends or ``max_turns``
        is reached.

        Parameters
        ----------
        audio_chunks : list of bytes, optional
            Pre-recorded audio to process.  If None, any audio received via
            the adapter (AUDIO_IN events) is used.
        transcript_override : str, optional
            When set, the STT result is forced to this text (bypasses audio).

        Returns
        -------
        dict
            Summary of the call including session data and turn results.
        """
        # Register event handler on adapter
        original_handler = self.adapter.on_event
        self.adapter.on_event = self._on_telephony_event

        # Mark call as connected
        self.adapter.state = CallState.CONNECTED

        try:
            if audio_chunks is not None:
                # Process provided audio directly
                self._process_audio_turn(audio_chunks, transcript_override)
            else:
                # Wait for events from the adapter
                self._wait_for_call_end()

        finally:
            self.adapter.on_event = original_handler

        return self._build_summary()

    # ------------------------------------------------------------------
    # Internal event handling
    # ------------------------------------------------------------------

    def _on_telephony_event(self, event: CallEvent) -> None:
        """Dispatch incoming telephony events."""
        if event.event_type == CallEventType.CALL_START:
            self.adapter.state = CallState.CONNECTED

        elif event.event_type == CallEventType.AUDIO_IN:
            import base64
            b64 = event.payload.get("data_b64", "")
            try:
                chunk = base64.b64decode(b64)
                with self._lock:
                    self._pending_audio.append(chunk)
            except Exception:  # noqa: BLE001
                pass

        elif event.event_type == CallEventType.CALL_END:
            self._call_ended.set()

    def _wait_for_call_end(self) -> None:
        """Wait for audio events and process them until CALL_END."""
        deadline = time.time() + 30.0  # safety timeout
        processed_turns = 0

        while not self._call_ended.is_set() and time.time() < deadline:
            with self._lock:
                chunks = list(self._pending_audio)
                self._pending_audio.clear()

            if chunks and processed_turns < self.max_turns:
                self._process_audio_turn(chunks)
                processed_turns += 1
            else:
                time.sleep(0.05)

    def _process_audio_turn(
        self,
        audio_chunks: List[bytes],
        transcript_override: Optional[str] = None,
    ) -> None:
        """Run one complete agent turn and record it."""
        context = self.session.get_context()

        # Override STT if requested
        if transcript_override and hasattr(self.agent_loop.stt, 'fixed_transcript'):
            original_transcript = self.agent_loop.stt.fixed_transcript
            self.agent_loop.stt.fixed_transcript = transcript_override
        else:
            original_transcript = None

        try:
            result = self.agent_loop.run(audio_chunks, context=context)
        finally:
            if original_transcript is not None and hasattr(self.agent_loop.stt, 'fixed_transcript'):
                self.agent_loop.stt.fixed_transcript = original_transcript

        transcript = result.get("transcript", "")
        llm_response = result.get("llm_response", "")

        # Record turns in session
        stt_result = result.get("stt_result", {})
        self.session.add_user_turn(
            text=transcript,
            metadata={
                "confidence": stt_result.get("confidence", 1.0),
                "duration_s": stt_result.get("duration_s", 0.0),
            },
        )

        if llm_response:
            self.session.add_agent_turn(
                text=llm_response,
                metadata={"timings": result.get("timings", {})},
            )

        # Escalation check
        escalation_response = None
        if self.escalation_manager is not None:
            state = {
                "confidence": stt_result.get("confidence", 1.0),
                "failure_count": 0,
                "transcript": transcript,
                "turn_count": self.session.turn_count,
            }
            escalation_response = self.escalation_manager.check_and_escalate(state)
            if escalation_response:
                self.session.add_agent_turn(
                    text=escalation_response,
                    metadata={"escalated": True},
                )

        # Send agent response back via adapter
        response_text = escalation_response or llm_response
        if response_text:
            self.adapter.agent_respond(response_text)

        turn_result = {
            "transcript": transcript,
            "llm_response": llm_response,
            "escalation_response": escalation_response,
            "timings": result.get("timings", {}),
            "turn_index": self.session.turn_count,
        }
        self.turn_results.append(turn_result)

        if self.on_turn_complete:
            self.on_turn_complete(turn_result)

    def _build_summary(self) -> Dict[str, Any]:
        """Build the call summary dict."""
        return {
            "session": self.session.to_dict(),
            "turn_results": self.turn_results,
            "turn_count": len(self.turn_results),
            "call_ended": self._call_ended.is_set(),
        }
