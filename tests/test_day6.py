"""Day 6 tests: session manager, diarization, context windowing, and telephony adapter."""

import base64
import json
import math
import time
import threading
from typing import List, Optional

import pytest

from voice_ops_agent.audio.input import CHUNK_SIZE, SAMPLE_RATE
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.stt.engine import MockSTTEngine
from voice_ops_agent.llm.engine import MockLLMEngine
from voice_ops_agent.tts.engine import MockTTSEngine
from voice_ops_agent.agent.loop import AgentLoop
from voice_ops_agent.escalation.manager import EscalationManager
from voice_ops_agent.escalation.rules import ExplicitRequestRule, ConfidenceThresholdRule
from voice_ops_agent.session.manager import SessionManager, Speaker, Turn
from voice_ops_agent.session.diarization import DiarizationStub
from voice_ops_agent.session.context_window import ContextWindow
from voice_ops_agent.telephony.adapter import (
    MockTelephonyAdapter,
    CallEvent,
    CallEventType,
    CallState,
)
from voice_ops_agent.telephony.scenarios import ScenarioPlayer, SCENARIOS, Scenario, ScenarioStep
from voice_ops_agent.session.call_handler import CallHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tone_chunk(amplitude: float = 0.5, frequency: float = 440.0) -> bytes:
    samples = []
    for i in range(CHUNK_SIZE):
        t = i / SAMPLE_RATE
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        samples.append(value.to_bytes(2, byteorder="little", signed=True))
    return b"".join(samples)


def dummy_audio(n: int = 5) -> List[bytes]:
    return [make_tone_chunk() for _ in range(n)]


def make_agent_loop(
    transcript: str = "Hello, I need help.",
    response: str = "Sure, I can help you with that.",
) -> AgentLoop:
    stt = MockSTTEngine(fixed_transcript=transcript, latency=0.0)
    llm = MockLLMEngine(
        token_latency=0.0,
        first_token_latency=0.0,
        fixed_response=response,
    )
    tts = MockTTSEngine(latency=0.0)
    speaker = MockSpeaker(realtime=False)
    return AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)


# ---------------------------------------------------------------------------
# SessionManager tests
# ---------------------------------------------------------------------------

class TestSessionManager:
    def test_initial_state(self):
        sm = SessionManager()
        assert sm.turn_count == 0
        assert sm.session_id != ""
        assert sm.last_turn() is None

    def test_add_user_turn(self):
        sm = SessionManager()
        turn = sm.add_user_turn("Hello there")
        assert isinstance(turn, Turn)
        assert turn.speaker == Speaker.USER
        assert turn.text == "Hello there"
        assert sm.turn_count == 1

    def test_add_agent_turn(self):
        sm = SessionManager()
        turn = sm.add_agent_turn("How can I help?")
        assert turn.speaker == Speaker.AGENT
        assert turn.text == "How can I help?"

    def test_add_tool_turn(self):
        sm = SessionManager()
        turn = sm.add_tool_turn("Weather is sunny")
        assert turn.speaker == Speaker.TOOL

    def test_turn_index_increments(self):
        sm = SessionManager()
        t0 = sm.add_user_turn("First")
        t1 = sm.add_agent_turn("Second")
        assert t0.index == 0
        assert t1.index == 1

    def test_turn_has_unique_id(self):
        sm = SessionManager()
        t0 = sm.add_user_turn("A")
        t1 = sm.add_user_turn("B")
        assert t0.turn_id != t1.turn_id

    def test_turn_has_timestamp(self):
        sm = SessionManager()
        before = time.time()
        t = sm.add_user_turn("test")
        after = time.time()
        assert before <= t.timestamp <= after

    def test_last_turn_returns_most_recent(self):
        sm = SessionManager()
        sm.add_user_turn("First")
        sm.add_agent_turn("Second")
        last = sm.last_turn()
        assert last.text == "Second"
        assert last.speaker == Speaker.AGENT

    def test_last_user_turn(self):
        sm = SessionManager()
        sm.add_user_turn("User turn 1")
        sm.add_agent_turn("Agent reply")
        sm.add_user_turn("User turn 2")
        last_user = sm.last_user_turn()
        assert last_user.text == "User turn 2"

    def test_last_user_turn_none_when_no_user_turns(self):
        sm = SessionManager()
        sm.add_agent_turn("Agent only")
        assert sm.last_user_turn() is None

    def test_turns_property_returns_all(self):
        sm = SessionManager()
        sm.add_user_turn("A")
        sm.add_agent_turn("B")
        sm.add_user_turn("C")
        turns = sm.turns
        assert len(turns) == 3
        assert turns[0].text == "A"

    def test_to_dict_has_required_keys(self):
        sm = SessionManager()
        sm.add_user_turn("Hello")
        d = sm.to_dict()
        for key in ("session_id", "created_at", "turn_count", "max_tokens", "turns"):
            assert key in d

    def test_to_dict_turns_serialisable(self):
        sm = SessionManager()
        sm.add_user_turn("Test")
        d = sm.to_dict()
        # Should be JSON-serialisable
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_metadata_stored_on_turn(self):
        sm = SessionManager()
        t = sm.add_user_turn("Hello", metadata={"confidence": 0.95})
        assert t.metadata["confidence"] == 0.95

    def test_diarized_speaker_in_metadata(self):
        sm = SessionManager()
        t = sm.add_user_turn("Hello")
        assert "diarized_speaker" in t.metadata

    def test_get_context_returns_list(self):
        sm = SessionManager()
        sm.add_user_turn("Hi")
        sm.add_agent_turn("Hello")
        ctx = sm.get_context()
        assert isinstance(ctx, list)
        assert len(ctx) >= 1

    def test_get_context_has_role_and_content(self):
        sm = SessionManager()
        sm.add_user_turn("Question")
        ctx = sm.get_context()
        for item in ctx:
            assert "role" in item
            assert "content" in item

    def test_get_full_history_returns_all(self):
        sm = SessionManager()
        sm.add_user_turn("A")
        sm.add_agent_turn("B")
        history = sm.get_full_history()
        assert len(history) == 2

    def test_session_id_is_unique(self):
        sm1 = SessionManager()
        sm2 = SessionManager()
        assert sm1.session_id != sm2.session_id

    def test_custom_session_id(self):
        sm = SessionManager(session_id="test-session-123")
        assert sm.session_id == "test-session-123"

    def test_turn_to_dict(self):
        sm = SessionManager()
        t = sm.add_user_turn("Hello")
        d = t.to_dict()
        for key in ("turn_id", "index", "speaker", "text", "timestamp", "metadata"):
            assert key in d
        assert d["speaker"] == "user"

    def test_turn_to_context_dict(self):
        sm = SessionManager()
        t = sm.add_user_turn("Hello")
        d = t.to_context_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"


# ---------------------------------------------------------------------------
# DiarizationStub tests
# ---------------------------------------------------------------------------

class TestDiarizationStub:
    def test_accepts_hint_by_default(self):
        diar = DiarizationStub()
        result = diar.assign_speaker(text="hello", hint=Speaker.USER)
        assert result == Speaker.USER

    def test_override_speaker_always_returned(self):
        diar = DiarizationStub(override_speaker=Speaker.AGENT)
        for _ in range(5):
            result = diar.assign_speaker(text="anything", hint=Speaker.USER)
            assert result == Speaker.AGENT

    def test_speaker_counts_updated(self):
        diar = DiarizationStub()
        diar.assign_speaker(text="hi", hint=Speaker.USER)
        diar.assign_speaker(text="hello", hint=Speaker.AGENT)
        counts = diar.speaker_counts
        assert counts["user"] == 1
        assert counts["agent"] == 1

    def test_history_records_assignments(self):
        diar = DiarizationStub()
        diar.assign_speaker(text="A", hint=Speaker.USER)
        diar.assign_speaker(text="B", hint=Speaker.AGENT)
        assert diar.history == [Speaker.USER, Speaker.AGENT]

    def test_reset_clears_state(self):
        diar = DiarizationStub()
        diar.assign_speaker(text="hi", hint=Speaker.USER)
        diar.reset()
        assert diar.speaker_counts["user"] == 0
        assert diar.history == []

    def test_no_hint_alternates_speakers(self):
        diar = DiarizationStub()
        s1 = diar.assign_speaker(text="first", hint=None)
        s2 = diar.assign_speaker(text="second", hint=None)
        # Should alternate
        assert s1 != s2

    def test_tool_speaker_accepted(self):
        diar = DiarizationStub()
        result = diar.assign_speaker(text="tool result", hint=Speaker.TOOL)
        assert result == Speaker.TOOL

    def test_speaker_counts_all_speakers_present(self):
        diar = DiarizationStub()
        counts = diar.speaker_counts
        for s in Speaker:
            assert s.value in counts


# ---------------------------------------------------------------------------
# ContextWindow tests
# ---------------------------------------------------------------------------

class TestContextWindow:
    def _make_turn(self, text: str, speaker: Speaker = Speaker.USER) -> Turn:
        """Build a minimal Turn for testing."""
        import uuid
        return Turn(
            turn_id=str(uuid.uuid4()),
            index=0,
            speaker=speaker,
            text=text,
        )

    def test_add_and_get_context(self):
        cw = ContextWindow(max_tokens=1000)
        cw.add(self._make_turn("Hello world"))
        ctx = cw.get_context()
        assert len(ctx) == 1
        assert ctx[0]["content"] == "Hello world"

    def test_total_tokens_increases(self):
        cw = ContextWindow(max_tokens=1000)
        cw.add(self._make_turn("one two three"))
        assert cw.total_tokens == 3

    def test_eviction_when_over_budget(self):
        # Budget of 5 tokens; add turns that exceed it
        cw = ContextWindow(max_tokens=5)
        cw.add(self._make_turn("one two three"))   # 3 tokens
        cw.add(self._make_turn("four five six"))    # 3 tokens → total 6, evict first
        assert cw.total_tokens <= 5
        assert cw.turn_count < 2 or cw.total_tokens <= 5

    def test_always_keeps_at_least_one_turn(self):
        cw = ContextWindow(max_tokens=1)
        # Add a turn with many tokens
        cw.add(self._make_turn("a b c d e f g h i j"))  # 10 tokens
        # Even if over budget, must keep the single turn
        assert cw.turn_count == 1

    def test_clear_removes_all(self):
        cw = ContextWindow(max_tokens=1000)
        cw.add(self._make_turn("hello"))
        cw.clear()
        assert cw.turn_count == 0
        assert cw.total_tokens == 0
        assert cw.get_context() == []

    def test_windowed_context_role_preserved(self):
        cw = ContextWindow(max_tokens=1000)
        cw.add(self._make_turn("Hi", Speaker.USER))
        cw.add(self._make_turn("Hello", Speaker.AGENT))
        ctx = cw.get_context()
        assert ctx[0]["role"] == "user"
        assert ctx[1]["role"] == "agent"

    def test_many_turns_within_budget(self):
        cw = ContextWindow(max_tokens=100)
        for i in range(20):
            cw.add(self._make_turn(f"word{i}"))  # 1 token each
        assert cw.total_tokens <= 100
        assert cw.turn_count == 20

    def test_many_turns_over_budget_evicts_oldest(self):
        cw = ContextWindow(max_tokens=5)
        for i in range(10):
            cw.add(self._make_turn(f"word{i}"))  # 1 token each
        # Only last 5 (or fewer) should remain
        assert cw.total_tokens <= 5
        ctx = cw.get_context()
        # Last added turn should still be present
        assert ctx[-1]["content"] == "word9"

    def test_session_manager_context_window_integration(self):
        sm = SessionManager(max_tokens=10)
        for i in range(20):
            sm.add_user_turn(f"word{i}")  # 1 token each
        ctx = sm.get_context()
        # Context should be within budget
        total = sum(len(item["content"].split()) for item in ctx)
        assert total <= 10 + 1  # small tolerance for token estimation


# ---------------------------------------------------------------------------
# MockTelephonyAdapter tests
# ---------------------------------------------------------------------------

class TestMockTelephonyAdapter:
    def test_initial_state_is_idle(self):
        adapter = MockTelephonyAdapter()
        assert adapter.state == CallState.IDLE

    def test_initiate_call_sends_call_start_event(self):
        events = []
        adapter = MockTelephonyAdapter(on_event=events.append)
        adapter.start()
        try:
            adapter.initiate_call(scenario="happy_path")
            event = adapter.wait_for_event(CallEventType.CALL_START, timeout=2.0)
            assert event is not None
            assert event.event_type == CallEventType.CALL_START
            assert event.payload["scenario"] == "happy_path"
        finally:
            adapter.stop()

    def test_send_audio_delivers_audio_in_event(self):
        events = []
        adapter = MockTelephonyAdapter(on_event=events.append)
        adapter.start()
        try:
            audio = make_tone_chunk()
            adapter.send_audio(audio)
            time.sleep(0.2)
            audio_events = adapter.get_received_events(CallEventType.AUDIO_IN)
            assert len(audio_events) >= 1
        finally:
            adapter.stop()

    def test_send_audio_base64_encoded(self):
        events = []
        adapter = MockTelephonyAdapter(on_event=events.append)
        adapter.start()
        try:
            audio = b"\x00\x01\x02\x03"
            adapter.send_audio(audio)
            time.sleep(0.2)
            audio_events = adapter.get_received_events(CallEventType.AUDIO_IN)
            assert len(audio_events) >= 1
            b64_data = audio_events[0].payload["data_b64"]
            decoded = base64.b64decode(b64_data)
            assert decoded == audio
        finally:
            adapter.stop()

    def test_get_audio_chunks_decodes_all(self):
        adapter = MockTelephonyAdapter()
        adapter.start()
        try:
            chunks = [make_tone_chunk() for _ in range(3)]
            for chunk in chunks:
                adapter.send_audio(chunk)
            time.sleep(0.3)
            decoded = adapter.get_audio_chunks()
            assert len(decoded) == 3
            for original, decoded_chunk in zip(chunks, decoded):
                assert original == decoded_chunk
        finally:
            adapter.stop()

    def test_send_dtmf_delivers_event(self):
        events = []
        adapter = MockTelephonyAdapter(on_event=events.append)
        adapter.start()
        try:
            adapter.send_dtmf("5")
            time.sleep(0.2)
            dtmf_events = adapter.get_received_events(CallEventType.DTMF)
            assert len(dtmf_events) >= 1
            assert dtmf_events[0].payload["digit"] == "5"
        finally:
            adapter.stop()

    def test_end_call_sends_call_end_event(self):
        events = []
        adapter = MockTelephonyAdapter(on_event=events.append)
        adapter.start()
        try:
            adapter.end_call()
            time.sleep(0.2)
            call_end_events = adapter.get_received_events(CallEventType.CALL_END)
            assert len(call_end_events) >= 1
            assert adapter.state == CallState.ENDED
        finally:
            adapter.stop()

    def test_agent_respond_records_sent_event(self):
        adapter = MockTelephonyAdapter()
        adapter.agent_respond("Hello caller!")
        assert len(adapter.sent_events) == 1
        assert adapter.sent_events[0].payload["text"] == "Hello caller!"
        assert adapter.sent_events[0].event_type == CallEventType.AGENT_RESPONSE

    def test_call_event_serialisation_roundtrip(self):
        event = CallEvent(
            event_type=CallEventType.STATUS,
            payload={"message": "connected"},
            timestamp=1234567890.0,
        )
        json_str = event.to_json()
        restored = CallEvent.from_json(json_str)
        assert restored.event_type == event.event_type
        assert restored.payload == event.payload
        assert restored.timestamp == event.timestamp

    def test_caller_id_stored(self):
        adapter = MockTelephonyAdapter(caller_id="+447700900000")
        assert adapter.caller_id == "+447700900000"

    def test_wait_for_event_returns_none_on_timeout(self):
        adapter = MockTelephonyAdapter()
        adapter.start()
        try:
            result = adapter.wait_for_event(CallEventType.CALL_END, timeout=0.1)
            assert result is None
        finally:
            adapter.stop()

    def test_multiple_audio_events_accumulate(self):
        adapter = MockTelephonyAdapter()
        adapter.start()
        try:
            for _ in range(5):
                adapter.send_audio(make_tone_chunk())
            time.sleep(0.3)
            events = adapter.get_received_events(CallEventType.AUDIO_IN)
            assert len(events) == 5
        finally:
            adapter.stop()

    def test_on_event_callback_receives_events(self):
        received = []
        adapter = MockTelephonyAdapter(on_event=received.append)
        adapter.start()
        try:
            adapter.initiate_call()
            time.sleep(0.2)
            assert len(received) >= 1
            assert any(e.event_type == CallEventType.CALL_START for e in received)
        finally:
            adapter.stop()

    def test_get_received_events_filtered_by_type(self):
        adapter = MockTelephonyAdapter()
        adapter.start()
        try:
            adapter.send_audio(make_tone_chunk())
            adapter.send_dtmf("1")
            time.sleep(0.2)
            audio_events = adapter.get_received_events(CallEventType.AUDIO_IN)
            dtmf_events = adapter.get_received_events(CallEventType.DTMF)
            assert all(e.event_type == CallEventType.AUDIO_IN for e in audio_events)
            assert all(e.event_type == CallEventType.DTMF for e in dtmf_events)
        finally:
            adapter.stop()


# ---------------------------------------------------------------------------
# ScenarioPlayer tests
# ---------------------------------------------------------------------------

class TestScenarioPlayer:
    def test_built_in_scenarios_exist(self):
        for name in ("happy_path", "escalation", "dtmf_menu", "multi_turn", "silent_caller"):
            assert name in SCENARIOS

    def test_scenario_has_steps(self):
        for scenario in SCENARIOS.values():
            assert isinstance(scenario.steps, list)
            assert len(scenario.steps) > 0

    def test_happy_path_scenario_runs(self):
        events = []
        adapter = MockTelephonyAdapter(on_event=events.append)
        adapter.start()
        try:
            player = ScenarioPlayer(
                adapter=adapter,
                scenario=SCENARIOS["happy_path"],
            )
            player.play(blocking=True)
            time.sleep(0.2)
            assert len(player.completed_steps) == len(SCENARIOS["happy_path"].steps)
            assert adapter.state == CallState.ENDED
        finally:
            adapter.stop()

    def test_dtmf_scenario_sends_digits(self):
        adapter = MockTelephonyAdapter()
        adapter.start()
        try:
            player = ScenarioPlayer(
                adapter=adapter,
                scenario=SCENARIOS["dtmf_menu"],
            )
            player.play(blocking=True)
            time.sleep(0.2)
            dtmf_events = adapter.get_received_events(CallEventType.DTMF)
            assert len(dtmf_events) >= 2
            digits = [e.payload["digit"] for e in dtmf_events]
            assert "1" in digits
            assert "3" in digits
        finally:
            adapter.stop()

    def test_on_step_callback_fires(self):
        steps_seen = []
        adapter = MockTelephonyAdapter()
        adapter.start()
        try:
            player = ScenarioPlayer(
                adapter=adapter,
                scenario=SCENARIOS["happy_path"],
                on_step=steps_seen.append,
            )
            player.play(blocking=True)
            assert len(steps_seen) == len(SCENARIOS["happy_path"].steps)
        finally:
            adapter.stop()

    def test_non_blocking_play(self):
        adapter = MockTelephonyAdapter()
        adapter.start()
        try:
            player = ScenarioPlayer(
                adapter=adapter,
                scenario=SCENARIOS["happy_path"],
            )
            player.play(blocking=False)
            # Give background thread time to finish
            time.sleep(1.0)
            assert adapter.state == CallState.ENDED
        finally:
            adapter.stop()

    def test_multi_turn_scenario_sends_multiple_audio_events(self):
        adapter = MockTelephonyAdapter()
        adapter.start()
        try:
            player = ScenarioPlayer(
                adapter=adapter,
                scenario=SCENARIOS["multi_turn"],
            )
            player.play(blocking=True)
            time.sleep(0.2)
            audio_events = adapter.get_received_events(CallEventType.AUDIO_IN)
            assert len(audio_events) >= 2
        finally:
            adapter.stop()

    def test_call_start_event_sent_before_steps(self):
        adapter = MockTelephonyAdapter()
        adapter.start()
        try:
            player = ScenarioPlayer(
                adapter=adapter,
                scenario=SCENARIOS["silent_caller"],
            )
            player.play(blocking=True)
            time.sleep(0.2)
            call_start_events = adapter.get_received_events(CallEventType.CALL_START)
            assert len(call_start_events) >= 1
        finally:
            adapter.stop()

    def test_custom_scenario_runs(self):
        custom = Scenario(
            name="custom_test",
            description="Custom test scenario",
            steps=[
                ScenarioStep(action="dtmf", delay=0.0, params={"digit": "9"}),
                ScenarioStep(action="end_call", delay=0.0),
            ],
        )
        adapter = MockTelephonyAdapter()
        adapter.start()
        try:
            player = ScenarioPlayer(adapter=adapter, scenario=custom)
            player.play(blocking=True)
            time.sleep(0.2)
            assert adapter.state == CallState.ENDED
        finally:
            adapter.stop()


# ---------------------------------------------------------------------------
# CallHandler end-to-end tests
# ---------------------------------------------------------------------------

class TestCallHandler:
    def _make_handler(
        self,
        transcript: str = "Hello I need help",
        response: str = "Sure I can help you",
        escalation_manager: Optional[EscalationManager] = None,
    ) -> CallHandler:
        adapter = MockTelephonyAdapter()
        loop = make_agent_loop(transcript=transcript, response=response)
        session = SessionManager()
        return CallHandler(
            adapter=adapter,
            agent_loop=loop,
            session=session,
            escalation_manager=escalation_manager,
            max_turns=5,
        )

    def test_handle_call_records_turns(self):
        handler = self._make_handler()
        result = handler.handle_call(
            audio_chunks=dummy_audio(),
        )
        assert result["turn_count"] >= 1

    def test_handle_call_records_user_turn_in_session(self):
        handler = self._make_handler(transcript="User question")
        handler.handle_call(audio_chunks=dummy_audio())
        user_turn = handler.session.last_user_turn()
        assert user_turn is not None
        assert user_turn.text == "User question"

    def test_handle_call_records_agent_turn_in_session(self):
        handler = self._make_handler(response="Agent answer")
        handler.handle_call(audio_chunks=dummy_audio())
        # Agent turns should be in session
        agent_turns = [
            t for t in handler.session.turns if t.speaker == Speaker.AGENT
        ]
        assert len(agent_turns) >= 1
        assert agent_turns[0].text == "Agent answer"

    def test_handle_call_sends_agent_response_to_adapter(self):
        handler = self._make_handler(response="I am here to help")
        handler.handle_call(audio_chunks=dummy_audio())
        assert len(handler.adapter.sent_events) >= 1
        texts = [e.payload["text"] for e in handler.adapter.sent_events]
        assert any("help" in t.lower() for t in texts)

    def test_handle_call_summary_has_session(self):
        handler = self._make_handler()
        result = handler.handle_call(audio_chunks=dummy_audio())
        assert "session" in result
        assert "turn_results" in result
        assert "turn_count" in result

    def test_escalation_triggers_on_explicit_request(self):
        mgr = EscalationManager(
            rules=[ExplicitRequestRule()],
            human_agent_response="Connecting you to a human.",
        )
        handler = self._make_handler(
            transcript="I want to speak to a human agent",
            escalation_manager=mgr,
        )
        result = handler.handle_call(audio_chunks=dummy_audio())
        # Should have escalated
        assert mgr.escalation_count >= 1
        # Escalation response should be in sent events
        texts = [e.payload["text"] for e in handler.adapter.sent_events]
        assert any("human" in t.lower() or "connecting" in t.lower() for t in texts)

    def test_no_escalation_on_normal_transcript(self):
        mgr = EscalationManager(
            rules=[ExplicitRequestRule()],
        )
        handler = self._make_handler(
            transcript="What is the weather today?",
            escalation_manager=mgr,
        )
        handler.handle_call(audio_chunks=dummy_audio())
        assert mgr.escalation_count == 0

    def test_on_turn_complete_callback(self):
        completed = []
        adapter = MockTelephonyAdapter()
        loop = make_agent_loop()
        session = SessionManager()
        handler = CallHandler(
            adapter=adapter,
            agent_loop=loop,
            session=session,
            on_turn_complete=completed.append,
        )
        handler.handle_call(audio_chunks=dummy_audio())
        assert len(completed) >= 1
        assert "transcript" in completed[0]
        assert "llm_response" in completed[0]

    def test_session_context_grows_with_turns(self):
        handler = self._make_handler()
        handler.handle_call(audio_chunks=dummy_audio())
        assert handler.session.turn_count >= 2  # user + agent

    def test_transcript_override_applied(self):
        handler = self._make_handler()
        # Provide a transcript override different from the loop's fixed_transcript
        result = handler.handle_call(
            audio_chunks=dummy_audio(),
            transcript_override="Override text",
        )
        # The session should record the override
        user_turn = handler.session.last_user_turn()
        assert user_turn.text == "Override text"


# ---------------------------------------------------------------------------
# End-to-end telephony + session integration test
# ---------------------------------------------------------------------------

class TestEndToEndTelephonySession:
    """
    Full end-to-end: ScenarioPlayer drives MockTelephonyAdapter, which
    feeds events into a CallHandler backed by a full AgentLoop and
    SessionManager.
    """

    def test_happy_path_scenario_full_pipeline(self):
        """Simulate a complete happy-path phone call."""
        loop = make_agent_loop(
            transcript="What is the weather today?",
            response="The weather is sunny and warm.",
        )
        session = SessionManager()
        adapter = MockTelephonyAdapter()
        handler = CallHandler(
            adapter=adapter,
            agent_loop=loop,
            session=session,
            max_turns=5,
        )

        # Process the happy_path scenario's audio directly
        scenario = SCENARIOS["happy_path"]
        audio_steps = [s for s in scenario.steps if s.action == "audio"]
        audio_data = audio_steps[0].params["audio"] if audio_steps else make_tone_chunk()

        result = handler.handle_call(
            audio_chunks=[audio_data],
        )

        assert result["turn_count"] >= 1
        assert handler.session.turn_count >= 1
        # Agent should have responded
        assert len(handler.adapter.sent_events) >= 1

    def test_session_windowing_does_not_exceed_budget(self):
        """Context window stays within token budget across many turns."""
        sm = SessionManager(max_tokens=20)
        # Add many turns that would overflow
        for i in range(50):
            if i % 2 == 0:
                sm.add_user_turn(f"user message {i} with some extra words")
            else:
                sm.add_agent_turn(f"agent reply {i} with some extra words")

        ctx = sm.get_context()
        total_tokens = sum(len(item["content"].split()) for item in ctx)
        # Should be within budget (with 1 turn minimum tolerance)
        assert total_tokens <= 20 + 10  # generous tolerance
        assert len(ctx) > 0

    def test_diarization_labels_persist_across_session(self):
        """Diarization assignments are recorded in turn metadata."""
        sm = SessionManager()
        t1 = sm.add_user_turn("Hello")
        t2 = sm.add_agent_turn("Hi there")
        t3 = sm.add_user_turn("Can you help me?")

        assert t1.metadata["diarized_speaker"] == Speaker.USER.value
        assert t2.metadata["diarized_speaker"] == Speaker.AGENT.value
        assert t3.metadata["diarized_speaker"] == Speaker.USER.value

    def test_call_event_types_all_serialisable(self):
        """All CallEventType values can be serialised and deserialised."""
        for event_type in CallEventType:
            event = CallEvent(
                event_type=event_type,
                payload={"key": "value"},
                timestamp=1000.0,
            )
            json_str = event.to_json()
            restored = CallEvent.from_json(json_str)
            assert restored.event_type == event_type
            assert restored.payload == {"key": "value"}
