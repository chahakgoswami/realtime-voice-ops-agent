"""Mock telephony adapter: SIP/PSTN simulation via socket-based message passing.

Architecture
------------
The adapter creates a pair of connected sockets:

    CallerThread  --[socket A]-->  TelephonyAdapter --[socket B]--> AgentLoop

Messages are line-delimited JSON.  Each message is a ``CallEvent`` with a
type and optional payload.  This allows the agent to be driven entirely from
test code (or configurable scenario scripts) without touching real telephony
hardware.

Event types
-----------
CALL_START      – A new call has been initiated.  Payload: {caller_id, scenario}.
AUDIO_IN        – Caller audio (base64-encoded PCM bytes).  Payload: {data_b64}.
CALL_END        – The caller has hung up.
DTMF            – A DTMF digit was pressed.  Payload: {digit}.
AGENT_RESPONSE  – Agent text response (from agent to telephony layer).  Payload: {text}.
STATUS          – Generic status message.  Payload: {message}.
"""

import base64
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


class CallEventType(str, Enum):
    CALL_START = "CALL_START"
    AUDIO_IN = "AUDIO_IN"
    CALL_END = "CALL_END"
    DTMF = "DTMF"
    AGENT_RESPONSE = "AGENT_RESPONSE"
    STATUS = "STATUS"


class CallState(Enum):
    IDLE = auto()
    RINGING = auto()
    CONNECTED = auto()
    ENDED = auto()


@dataclass
class CallEvent:
    """
    A single telephony event.

    Attributes
    ----------
    event_type : CallEventType
    payload : dict
    timestamp : float
    """
    event_type: CallEventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "event_type": self.event_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
        })

    @classmethod
    def from_json(cls, data: str) -> "CallEvent":
        obj = json.loads(data)
        return cls(
            event_type=CallEventType(obj["event_type"]),
            payload=obj.get("payload", {}),
            timestamp=obj.get("timestamp", time.time()),
        )


class MockTelephonyAdapter:
    """
    Mock SIP/PSTN telephony adapter.

    Creates an in-process socket pair so tests can drive the agent as if
    it were handling a real phone call.

    Parameters
    ----------
    on_event : callable, optional
        ``on_event(event: CallEvent)`` – called for every inbound event
        received by the adapter (from the caller side).
    caller_id : str
        Simulated caller phone number / identifier.
    """

    def __init__(
        self,
        on_event: Optional[Callable[[CallEvent], None]] = None,
        caller_id: str = "+15550001234",
    ) -> None:
        self.on_event = on_event
        self.caller_id = caller_id
        self.state: CallState = CallState.IDLE

        # Received events (from caller perspective)
        self.received_events: List[CallEvent] = []
        # Events sent by agent back to caller
        self.sent_events: List[CallEvent] = []

        # In-process socket pair for message passing
        self._caller_sock, self._agent_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
        self._caller_sock.settimeout(5.0)
        self._agent_sock.settimeout(5.0)

        self._recv_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Buffer for incomplete lines received on agent socket
        self._recv_buffer = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background receive thread."""
        self._stop_event.clear()
        self._recv_thread = threading.Thread(
            target=self._receive_loop,
            name="telephony-recv",
            daemon=True,
        )
        self._recv_thread.start()

    def stop(self) -> None:
        """Stop the adapter and close sockets."""
        self._stop_event.set()
        try:
            self._caller_sock.close()
        except OSError:
            pass
        try:
            self._agent_sock.close()
        except OSError:
            pass
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
            self._recv_thread = None

    # ------------------------------------------------------------------
    # Caller-side API (simulates actions taken by the remote caller)
    # ------------------------------------------------------------------

    def caller_send_event(self, event: CallEvent) -> None:
        """
        Send an event from the *caller* side into the adapter.

        The event travels: caller_sock -> agent_sock, where the receive
        loop picks it up and fires ``on_event``.

        Parameters
        ----------
        event : CallEvent
        """
        message = event.to_json() + "\n"
        try:
            self._caller_sock.sendall(message.encode("utf-8"))
        except OSError:
            pass  # Socket closed during test teardown

    def initiate_call(self, scenario: str = "default") -> None:
        """
        Simulate the caller initiating a call.

        Parameters
        ----------
        scenario : str
            Scenario name passed in the CALL_START event payload.
        """
        self.state = CallState.RINGING
        event = CallEvent(
            event_type=CallEventType.CALL_START,
            payload={"caller_id": self.caller_id, "scenario": scenario},
        )
        self.caller_send_event(event)

    def send_audio(self, audio_bytes: bytes) -> None:
        """
        Simulate the caller sending an audio chunk.

        Parameters
        ----------
        audio_bytes : bytes
            Raw PCM audio bytes.  Will be base64-encoded for transport.
        """
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        event = CallEvent(
            event_type=CallEventType.AUDIO_IN,
            payload={"data_b64": b64},
        )
        self.caller_send_event(event)

    def send_dtmf(self, digit: str) -> None:
        """
        Simulate a DTMF keypress.

        Parameters
        ----------
        digit : str
            Single character (0-9, *, #).
        """
        event = CallEvent(
            event_type=CallEventType.DTMF,
            payload={"digit": digit},
        )
        self.caller_send_event(event)

    def end_call(self) -> None:
        """Simulate the caller hanging up."""
        event = CallEvent(
            event_type=CallEventType.CALL_END,
            payload={},
        )
        self.caller_send_event(event)
        self.state = CallState.ENDED

    # ------------------------------------------------------------------
    # Agent-side API (agent sends responses back to telephony layer)
    # ------------------------------------------------------------------

    def agent_respond(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an agent response (agent -> telephony).

        In a real system this would trigger TTS and stream audio back to
        the caller.  Here we log the event.

        Parameters
        ----------
        text : str
            Agent response text.
        metadata : dict, optional
        """
        event = CallEvent(
            event_type=CallEventType.AGENT_RESPONSE,
            payload={"text": text, **(metadata or {})},
        )
        with self._lock:
            self.sent_events.append(event)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_received_events(self, event_type: Optional[CallEventType] = None) -> List[CallEvent]:
        """Return received events, optionally filtered by type."""
        with self._lock:
            if event_type is None:
                return list(self.received_events)
            return [e for e in self.received_events if e.event_type == event_type]

    def get_audio_chunks(self) -> List[bytes]:
        """Return all audio chunks received from the caller (decoded)."""
        chunks = []
        for event in self.get_received_events(CallEventType.AUDIO_IN):
            b64 = event.payload.get("data_b64", "")
            try:
                chunks.append(base64.b64decode(b64))
            except Exception:  # noqa: BLE001
                pass
        return chunks

    def wait_for_event(
        self,
        event_type: CallEventType,
        timeout: float = 2.0,
    ) -> Optional[CallEvent]:
        """
        Block until an event of the given type is received or timeout expires.

        Parameters
        ----------
        event_type : CallEventType
        timeout : float

        Returns
        -------
        CallEvent or None
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            events = self.get_received_events(event_type)
            if events:
                return events[-1]
            time.sleep(0.01)
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _receive_loop(self) -> None:
        """Background thread: read events from agent socket and dispatch."""
        while not self._stop_event.is_set():
            try:
                data = self._agent_sock.recv(4096)
                if not data:
                    break
                self._recv_buffer += data.decode("utf-8", errors="replace")
                while "\n" in self._recv_buffer:
                    line, self._recv_buffer = self._recv_buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = CallEvent.from_json(line)
                        with self._lock:
                            self.received_events.append(event)
                            if event.event_type == CallEventType.CALL_START:
                                self.state = CallState.CONNECTED
                        if self.on_event:
                            self.on_event(event)
                    except (json.JSONDecodeError, ValueError, KeyError):
                        pass  # malformed message – ignore
            except (socket.timeout, OSError):
                if self._stop_event.is_set():
                    break
                continue
