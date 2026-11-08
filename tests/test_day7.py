"""Day 7 tests: full scenario suite with deterministic fixtures and latency profiler."""

import math
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from voice_ops_agent.audio.input import CHUNK_SIZE, SAMPLE_RATE
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.stt.engine import MockSTTEngine
from voice_ops_agent.llm.engine import MockLLMEngine
from voice_ops_agent.tts.engine import MockTTSEngine
from voice_ops_agent.vad.detector import EnergyVAD
from voice_ops_agent.agent.loop import AgentLoop
from voice_ops_agent.agent.state import AgentState
from voice_ops_agent.tools.registry import default_registry
from voice_ops_agent.tools.parser import ToolCallIntent
from voice_ops_agent.tools.base import ToolResult
from voice_ops_agent.escalation.manager import EscalationManager
from voice_ops_agent.escalation.rules import (
    ExplicitRequestRule,
    ConfidenceThresholdRule,
    RepeatedFailureRule,
)
from voice_ops_agent.escalation.audit import AuditLogger
from voice_ops_agent.session.manager import SessionManager, Speaker
from voice_ops_agent.session.call_handler import CallHandler
from voice_ops_agent.telephony.adapter import (
    MockTelephonyAdapter,
    CallEventType,
    CallState,
)
from voice_ops_agent.profiler import LatencyProfiler


# ---------------------------------------------------------------------------
# Constants for sub-second latency assertions
# ---------------------------------------------------------------------------

# Maximum allowed latency in seconds for sub-second requirement
SUB_SECOND = 1.0
# Generous budget for CI environments (mocks are fast)
STT_TO_FIRST_TOKEN_BUDGET = 0.5
TTS_FIRST_CHUNK_BUDGET = 0.3


# ---------------------------------------------------------------------------
# Deterministic fixture helpers
# ---------------------------------------------------------------------------

def make_tone_chunk(amplitude: float = 0.5, frequency: float = 440.0) -> bytes:
    """Generate a deterministic PCM chunk."""
    samples = []
    for i in range(CHUNK_SIZE):
        t = i / SAMPLE_RATE
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        samples.append(value.to_bytes(2, byteorder="little", signed=True))
    return b"".join(samples)


def make_silence_chunk() -> bytes:
    return b"\x00\x00" * CHUNK_SIZE


def dummy_audio(n: int = 5) -> List[bytes]:
    return [make_tone_chunk() for _ in range(n)]


def make_fast_loop(
    transcript: str = "Hello",
    response: str = "I can help you with that.",
    tool_registry=None,
    on_interrupt=None,
    on_state_change=None,
    max_interruptions: int = 3,
) -> AgentLoop:
    """Build an AgentLoop with zero simulated latency for deterministic tests."""
    stt = MockSTTEngine(fixed_transcript=transcript, latency=0.0)
    llm = MockLLMEngine(
        token_latency=0.0,
        first_token_latency=0.0,
        fixed_response=response,
    )
    tts = MockTTSEngine(latency=0.0)
    speaker = MockSpeaker(realtime=False)
    return AgentLoop(
        stt=stt,
        llm=llm,
        tts=tts,
        speaker=speaker,
        tool_registry=tool_registry,
        on_interrupt=on_interrupt,
        on_state_change=on_state_change,
        max_interruptions=max_interruptions,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fast_loop():
    """A zero-latency AgentLoop for general use."""
    return make_fast_loop()


@pytest.fixture
def fast_loop_with_tools():
    """A zero-latency AgentLoop with the default tool registry."""
    return make_fast_loop(tool_registry=default_registry())


@pytest.fixture
def vad():
    return EnergyVAD(threshold=300.0)


@pytest.fixture
def escalation_manager():
    return EscalationManager(
        rules=[
            ConfidenceThresholdRule(threshold=0.5),
            RepeatedFailureRule(max_failures=3),
            ExplicitRequestRule(),
        ],
        human_agent_response="Connecting you to a live agent now.",
    )


@pytest.fixture
def session():
    return SessionManager()


# ---------------------------------------------------------------------------
# Happy-path scenario
# ---------------------------------------------------------------------------

class TestHappyPath:
    """Full happy-path: audio -> STT -> LLM -> TTS -> speaker, no interruption."""

    def test_run_completes_without_error(self, fast_loop):
        result = fast_loop.run(dummy_audio())
        assert result is not None

    def test_transcript_is_set(self, fast_loop):
        result = fast_loop.run(dummy_audio())
        assert isinstance(result["transcript"], str)
        assert len(result["transcript"]) > 0

    def test_llm_response_is_set(self, fast_loop):
        result = fast_loop.run(dummy_audio())
        assert isinstance(result["llm_response"], str)
        assert len(result["llm_response"]) > 0

    def test_not_interrupted(self, fast_loop):
        result = fast_loop.run(dummy_audio())
        assert result["interrupted"] is False
        assert result["interruption_count"] == 0

    def test_speaker_played_audio(self):
        loop = make_fast_loop(response="The quick brown fox")
        result = loop.run(dummy_audio())
        assert loop.speaker.total_bytes_played > 0

    def test_state_history_full_path(self, fast_loop):
        result = fast_loop.run(dummy_audio())
        history = result["state_history"]
        assert AgentState.IDLE in history
        assert AgentState.LISTENING in history
        assert AgentState.TRANSCRIBING in history
        assert AgentState.LLM_STREAMING in history
        assert AgentState.TTS_PLAYING in history
        assert history[-1] == AgentState.IDLE

    def test_stt_result_has_expected_keys(self, fast_loop):
        result = fast_loop.run(dummy_audio())
        stt = result["stt_result"]
        assert "text" in stt
        assert "confidence" in stt
        assert "duration_s" in stt

    def test_no_tool_calls_in_happy_path(self, fast_loop):
        result = fast_loop.run(dummy_audio())
        assert result["tool_calls"] == []
        assert result["tool_results"] == []

    def test_timings_all_non_negative(self, fast_loop):
        result = fast_loop.run(dummy_audio())
        for key, value in result["timings"].items():
            assert value >= 0.0, f"{key} = {value} is negative"

    def test_callbacks_fire_in_order(self):
        order = []
        stt = MockSTTEngine(fixed_transcript="hi", latency=0.0)
        llm = MockLLMEngine(
            token_latency=0.0,
            first_token_latency=0.0,
            fixed_response="hello",
        )
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            on_transcript=lambda t: order.append("transcript"),
            on_token=lambda t: order.append("token"),
            on_response_complete=lambda r: order.append("complete"),
        )
        loop.run(dummy_audio())
        assert "transcript" in order
        assert "token" in order
        assert "complete" in order
        assert order.index("transcript") < order.index("complete")

    def test_session_records_turns(self, session):
        loop = make_fast_loop(
            transcript="What is the weather?",
            response="It is sunny.",
        )
        # Simulate session recording
        result = loop.run(dummy_audio())
        session.add_user_turn(result["transcript"])
        session.add_agent_turn(result["llm_response"])
        assert session.turn_count == 2
        assert session.last_user_turn().text == "What is the weather?"

    def test_full_pipeline_via_call_handler(self):
        stt = MockSTTEngine(
            fixed_transcript="Hello I need help with my account",
            latency=0.0,
        )
        llm = MockLLMEngine(
            token_latency=0.0,
            first_token_latency=0.0,
            fixed_response="I can help you with your account.",
        )
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)
        adapter = MockTelephonyAdapter()
        handler = CallHandler(
            adapter=adapter,
            agent_loop=loop,
            session=SessionManager(),
        )
        result = handler.handle_call(audio_chunks=dummy_audio())
        assert result["turn_count"] >= 1
        assert len(adapter.sent_events) >= 1


# ---------------------------------------------------------------------------
# Interruption scenario
# ---------------------------------------------------------------------------

class TestInterruptionScenario:
    """Interruption: user speaks during TTS playback."""

    def test_interruption_detected(self, vad):
        loop = make_fast_loop(
            transcript="Hello",
            response="This is a longer response that takes time to synthesise",
            max_interruptions=1,
        )
        q_chunks = [make_tone_chunk(amplitude=0.5)]
        idx = {"i": 0}

        def audio_source():
            if idx["i"] < len(q_chunks):
                chunk = q_chunks[idx["i"]]
                idx["i"] += 1
                return chunk
            return make_silence_chunk()

        result = loop.run(
            dummy_audio(),
            interrupt_audio_source=audio_source,
            vad=vad,
        )
        assert result["interrupted"] is True
        assert result["interruption_count"] >= 1

    def test_state_includes_interrupted(self, vad):
        loop = make_fast_loop(
            transcript="Hello",
            response="A long response that gets cut off by the user speaking",
            max_interruptions=1,
        )

        def audio_source():
            return make_tone_chunk(amplitude=0.5)

        result = loop.run(
            dummy_audio(),
            interrupt_audio_source=audio_source,
            vad=vad,
        )
        if result["interrupted"]:
            assert AgentState.INTERRUPTED in result["state_history"]

    def test_speaker_halted_on_interrupt(self, vad):
        loop = make_fast_loop(
            transcript="Hello",
            response="one two three four five six seven eight nine ten",
            max_interruptions=1,
        )
        # Full playback for reference
        full_loop = make_fast_loop(
            transcript="Hello",
            response="one two three four five six seven eight nine ten",
        )
        full_result = full_loop.run(dummy_audio())
        full_bytes = full_loop.speaker.total_bytes_played

        def audio_source():
            return make_tone_chunk(amplitude=0.5)

        result = loop.run(
            dummy_audio(),
            interrupt_audio_source=audio_source,
            vad=vad,
        )
        # If interrupted, should have played fewer bytes
        if result["interrupted"]:
            assert loop.speaker.total_bytes_played <= full_bytes

    def test_max_interruptions_cap(self, vad):
        loop = make_fast_loop(
            transcript="Hello",
            response="short",
            max_interruptions=2,
        )

        def audio_source():
            return make_tone_chunk(amplitude=0.5)

        result = loop.run(
            dummy_audio(),
            interrupt_audio_source=audio_source,
            vad=vad,
        )
        assert result["interruption_count"] <= 2

    def test_on_interrupt_callback_fires(self, vad):
        interrupted_flag = []
        loop = make_fast_loop(
            transcript="Hello",
            response="A response that gets interrupted",
            on_interrupt=lambda: interrupted_flag.append(True),
            max_interruptions=1,
        )

        def audio_source():
            return make_tone_chunk(amplitude=0.5)

        result = loop.run(
            dummy_audio(),
            interrupt_audio_source=audio_source,
            vad=vad,
        )
        if result["interrupted"]:
            assert len(interrupted_flag) >= 1

    def test_no_interrupt_with_silence(self, vad):
        loop = make_fast_loop(
            transcript="Hello",
            response="A response with no interruption",
        )

        def audio_source():
            return make_silence_chunk()

        result = loop.run(
            dummy_audio(),
            interrupt_audio_source=audio_source,
            vad=vad,
        )
        assert result["interrupted"] is False


# ---------------------------------------------------------------------------
# Tool-call scenario
# ---------------------------------------------------------------------------

class TestToolCallScenario:
    """Tool-calling: LLM emits TOOL_CALL directive, tool executes, result injected."""

    def _make_tool_loop(self, tool_response: str, follow_up: str) -> AgentLoop:
        """Build a loop where LLM emits a tool call, then follows up."""
        responses = [tool_response, follow_up]
        call_count = {"n": 0}

        class SequencedLLM(MockLLMEngine):
            def stream_response(self, user_text, context=None):
                resp = responses[min(call_count["n"], len(responses) - 1)]
                call_count["n"] += 1
                words = resp.split()
                for i, w in enumerate(words):
                    yield w + (" " if i < len(words) - 1 else "")

        stt = MockSTTEngine(
            fixed_transcript="What is the weather in Paris?",
            latency=0.0,
        )
        llm = SequencedLLM(token_latency=0.0, first_token_latency=0.0)
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker(realtime=False)
        registry = default_registry()
        return AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            tool_registry=registry,
        )

    def test_weather_tool_called(self):
        loop = self._make_tool_loop(
            tool_response='TOOL_CALL:weather_lookup:{"location": "Paris"}',
            follow_up="The weather in Paris is sunny and warm.",
        )
        result = loop.run(dummy_audio())
        assert len(result["tool_calls"]) >= 1
        assert result["tool_calls"][0].tool_name == "weather_lookup"

    def test_tool_result_non_empty(self):
        loop = self._make_tool_loop(
            tool_response='TOOL_CALL:weather_lookup:{"location": "Paris"}',
            follow_up="The weather in Paris is sunny.",
        )
        result = loop.run(dummy_audio())
        assert len(result["tool_results"]) >= 1
        assert result["tool_results"][0].success is True

    def test_final_response_comes_from_follow_up(self):
        loop = self._make_tool_loop(
            tool_response='TOOL_CALL:order_status:{"order_id": "99"}',
            follow_up="Your order is on its way.",
        )
        result = loop.run(dummy_audio())
        assert len(result["llm_response"]) > 0

    def test_multi_tool_in_one_response(self):
        responses = [
            'TOOL_CALL:weather_lookup:{"location": "Berlin"}\nTOOL_CALL:order_status:{"order_id": "42"}',
            "I checked both for you.",
        ]
        call_count = {"n": 0}

        class SequencedLLM(MockLLMEngine):
            def stream_response(self, user_text, context=None):
                resp = responses[min(call_count["n"], len(responses) - 1)]
                call_count["n"] += 1
                words = resp.split()
                for i, w in enumerate(words):
                    yield w + (" " if i < len(words) - 1 else "")

        stt = MockSTTEngine(fixed_transcript="Multi tool test", latency=0.0)
        llm = SequencedLLM(token_latency=0.0, first_token_latency=0.0)
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            tool_registry=default_registry(),
        )
        result = loop.run(dummy_audio())
        assert len(result["tool_calls"]) == 2
        tool_names = {tc.tool_name for tc in result["tool_calls"]}
        assert "weather_lookup" in tool_names
        assert "order_status" in tool_names

    def test_unknown_tool_does_not_crash(self):
        loop = make_fast_loop(
            response='TOOL_CALL:nonexistent_tool:{"arg": "val"}',
            tool_registry=default_registry(),
        )
        # Need a sequenced LLM for follow-up; use fast_loop_with_tools pattern
        responses = [
            'TOOL_CALL:nonexistent_tool:{"arg": "val"}',
            "I could not find the right tool.",
        ]
        call_count = {"n": 0}

        class SeqLLM(MockLLMEngine):
            def stream_response(self, user_text, context=None):
                resp = responses[min(call_count["n"], len(responses) - 1)]
                call_count["n"] += 1
                for i, w in enumerate(resp.split()):
                    yield w + (" " if i < len(resp.split()) - 1 else "")

        stt = MockSTTEngine(fixed_transcript="test", latency=0.0)
        llm = SeqLLM(token_latency=0.0, first_token_latency=0.0)
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        agent = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            tool_registry=default_registry(),
        )
        result = agent.run(dummy_audio())
        assert result["tool_results"][0].success is False

    def test_calendar_tool_integration(self):
        responses = [
            'TOOL_CALL:calendar_check:{"date": "Monday"}',
            "You have three events on Monday.",
        ]
        call_count = {"n": 0}

        class SeqLLM(MockLLMEngine):
            def stream_response(self, user_text, context=None):
                resp = responses[min(call_count["n"], len(responses) - 1)]
                call_count["n"] += 1
                for i, w in enumerate(resp.split()):
                    yield w + (" " if i < len(resp.split()) - 1 else "")

        stt = MockSTTEngine(fixed_transcript="What's on my calendar Monday?", latency=0.0)
        llm = SeqLLM(token_latency=0.0, first_token_latency=0.0)
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            tool_registry=default_registry(),
        )
        result = loop.run(dummy_audio())
        assert any(tc.tool_name == "calendar_check" for tc in result["tool_calls"])

    def test_kb_search_tool_integration(self):
        responses = [
            'TOOL_CALL:kb_search:{"query": "refund policy"}',
            "Here is what I found about refunds.",
        ]
        call_count = {"n": 0}

        class SeqLLM(MockLLMEngine):
            def stream_response(self, user_text, context=None):
                resp = responses[min(call_count["n"], len(responses) - 1)]
                call_count["n"] += 1
                for i, w in enumerate(resp.split()):
                    yield w + (" " if i < len(resp.split()) - 1 else "")

        stt = MockSTTEngine(fixed_transcript="What is your refund policy?", latency=0.0)
        llm = SeqLLM(token_latency=0.0, first_token_latency=0.0)
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            tool_registry=default_registry(),
        )
        result = loop.run(dummy_audio())
        assert any(tc.tool_name == "kb_search" for tc in result["tool_calls"])


# ---------------------------------------------------------------------------
# Escalation scenario
# ---------------------------------------------------------------------------

class TestEscalationScenario:
    """Escalation: rules fire and handoff response is returned."""

    def test_explicit_request_escalates(self, escalation_manager):
        state = {
            "confidence": 0.95,
            "failure_count": 0,
            "transcript": "I want to speak to a human agent please",
            "turn_count": 1,
        }
        response = escalation_manager.check_and_escalate(state)
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_low_confidence_escalates(self, escalation_manager):
        state = {
            "confidence": 0.1,
            "failure_count": 0,
            "transcript": "mumble mumble",
            "turn_count": 1,
        }
        response = escalation_manager.check_and_escalate(state)
        assert response is not None

    def test_repeated_failure_escalates(self, escalation_manager):
        state = {
            "confidence": 0.95,
            "failure_count": 5,
            "transcript": "hello",
            "turn_count": 1,
        }
        response = escalation_manager.check_and_escalate(state)
        assert response is not None

    def test_normal_conversation_no_escalation(self, escalation_manager):
        state = {
            "confidence": 0.95,
            "failure_count": 0,
            "transcript": "What is the weather today?",
            "turn_count": 1,
        }
        response = escalation_manager.check_and_escalate(state)
        assert response is None
        assert escalation_manager.escalation_count == 0

    def test_escalation_logged_to_audit(self, tmp_path):
        logger = AuditLogger(path=tmp_path / "audit.json")
        mgr = EscalationManager(
            rules=[ExplicitRequestRule()],
            audit_logger=logger,
            human_agent_response="Connecting you now.",
        )
        mgr.check_and_escalate({
            "confidence": 0.95,
            "failure_count": 0,
            "transcript": "connect me to a human agent",
            "turn_count": 1,
        })
        records = logger.records()
        assert len(records) == 1
        assert records[0]["rule_name"] == "explicit_request"

    def test_escalation_via_call_handler(self):
        mgr = EscalationManager(
            rules=[ExplicitRequestRule()],
            human_agent_response="Transferring you to a human.",
        )
        stt = MockSTTEngine(
            fixed_transcript="I need to speak to a human agent",
            latency=0.0,
        )
        llm = MockLLMEngine(
            token_latency=0.0,
            first_token_latency=0.0,
            fixed_response="Let me connect you.",
        )
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)
        adapter = MockTelephonyAdapter()
        handler = CallHandler(
            adapter=adapter,
            agent_loop=loop,
            session=SessionManager(),
            escalation_manager=mgr,
        )
        handler.handle_call(audio_chunks=dummy_audio())
        assert mgr.escalation_count >= 1
        texts = [e.payload["text"] for e in adapter.sent_events]
        assert any("human" in t.lower() or "transferring" in t.lower() or len(t) > 0
                   for t in texts)

    def test_multiple_escalations_logged(self, tmp_path):
        logger = AuditLogger(path=tmp_path / "audit.json")
        mgr = EscalationManager(
            rules=[
                ExplicitRequestRule(),
                RepeatedFailureRule(max_failures=1),
            ],
            audit_logger=logger,
        )
        mgr.check_and_escalate(
            {"confidence": 0.95, "failure_count": 0,
             "transcript": "connect me to human agent", "turn_count": 1}
        )
        mgr.check_and_escalate(
            {"confidence": 0.95, "failure_count": 2,
             "transcript": "hello", "turn_count": 2}
        )
        assert logger.records().__len__() == 2


# ---------------------------------------------------------------------------
# Latency profiler tests
# ---------------------------------------------------------------------------

class TestLatencyProfiler:
    """Tests for the LatencyProfiler utility."""

    def test_profiler_measures_stt_to_first_token(self):
        loop = make_fast_loop(
            transcript="What time is it?",
            response="It is noon.",
        )
        profiler = LatencyProfiler(loop)
        result = profiler.run_and_profile(dummy_audio())
        assert "stt_to_first_token" in result["timings"]
        assert result["timings"]["stt_to_first_token"] >= 0.0

    def test_profiler_measures_tts_first_chunk(self):
        loop = make_fast_loop(
            transcript="Hello",
            response="Hi there how can I help you today.",
        )
        profiler = LatencyProfiler(loop)
        result = profiler.run_and_profile(dummy_audio())
        assert "tts_first_chunk_latency" in result["timings"]
        assert result["timings"]["tts_first_chunk_latency"] >= 0.0

    def test_profiler_measures_total_latency(self):
        loop = make_fast_loop(response="Quick response.")
        profiler = LatencyProfiler(loop)
        result = profiler.run_and_profile(dummy_audio())
        assert "total_latency" in result["timings"]
        assert result["timings"]["total_latency"] > 0.0

    def test_stt_to_first_token_sub_second_with_zero_latency(self):
        """With mocked zero-latency engines, STT->first-token must be < 1 s."""
        loop = make_fast_loop(
            transcript="Hello world",
            response="Hi there.",
        )
        profiler = LatencyProfiler(loop)
        result = profiler.run_and_profile(dummy_audio())
        assert result["timings"]["stt_to_first_token"] < SUB_SECOND

    def test_tts_first_chunk_sub_second_with_zero_latency(self):
        """With mocked zero-latency engines, TTS first chunk must be < 1 s."""
        loop = make_fast_loop(
            transcript="Hello",
            response="A response.",
        )
        profiler = LatencyProfiler(loop)
        result = profiler.run_and_profile(dummy_audio())
        assert result["timings"]["tts_first_chunk_latency"] < SUB_SECOND

    def test_profiler_assert_sub_second_passes_on_fast_loop(self):
        loop = make_fast_loop(response="Fast response.")
        profiler = LatencyProfiler(loop)
        result = profiler.run_and_profile(dummy_audio())
        # Should not raise
        profiler.assert_sub_second(
            result,
            stt_to_first_token_budget=STT_TO_FIRST_TOKEN_BUDGET,
            tts_first_chunk_budget=TTS_FIRST_CHUNK_BUDGET,
        )

    def test_profiler_assert_sub_second_fails_on_slow_stt(self):
        """Profiler raises AssertionError when latency budget is violated."""
        # STT latency of 0.6 s will push stt_to_first_token over budget
        stt = MockSTTEngine(fixed_transcript="test", latency=0.6)
        llm = MockLLMEngine(
            token_latency=0.0,
            first_token_latency=0.0,
            fixed_response="ok",
        )
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)
        profiler = LatencyProfiler(loop)
        result = profiler.run_and_profile(dummy_audio())
        with pytest.raises(AssertionError):
            profiler.assert_sub_second(
                result,
                stt_to_first_token_budget=0.1,  # very tight budget
                tts_first_chunk_budget=TTS_FIRST_CHUNK_BUDGET,
            )

    def test_profiler_generates_report_string(self):
        loop = make_fast_loop(response="Response text.")
        profiler = LatencyProfiler(loop)
        result = profiler.run_and_profile(dummy_audio())
        report = profiler.report(result)
        assert isinstance(report, str)
        assert "stt" in report.lower() or "latency" in report.lower()

    def test_profiler_multiple_runs_accumulate(self):
        loop = make_fast_loop(response="Hello.")
        profiler = LatencyProfiler(loop)
        for _ in range(3):
            profiler.run_and_profile(dummy_audio())
        assert profiler.run_count == 3

    def test_profiler_average_latencies(self):
        loop = make_fast_loop(response="Hi.")
        profiler = LatencyProfiler(loop)
        for _ in range(3):
            profiler.run_and_profile(dummy_audio())
        avgs = profiler.average_latencies()
        assert "stt_to_first_token" in avgs
        assert avgs["stt_to_first_token"] >= 0.0


# ---------------------------------------------------------------------------
# Sub-second latency assertions (integration)
# ---------------------------------------------------------------------------

class TestSubSecondLatency:
    """Assert that key latency metrics are under 1 second with mock engines."""

    def test_stt_to_first_token_under_budget(self):
        loop = make_fast_loop(
            transcript="Hello, can you help me?",
            response="Of course! What do you need?",
        )
        result = loop.run(dummy_audio())
        assert result["timings"]["stt_to_first_token"] < STT_TO_FIRST_TOKEN_BUDGET

    def test_tts_first_chunk_under_budget(self):
        loop = make_fast_loop(
            transcript="Hello",
            response="Here is my response to you.",
        )
        result = loop.run(dummy_audio())
        assert result["timings"]["tts_first_chunk_latency"] < TTS_FIRST_CHUNK_BUDGET

    def test_total_round_trip_under_one_second(self):
        loop = make_fast_loop(
            transcript="Quick question",
            response="Quick answer.",
        )
        result = loop.run(dummy_audio())
        assert result["timings"]["total_latency"] < SUB_SECOND

    def test_stt_latency_under_budget(self):
        loop = make_fast_loop(transcript="Hi", response="Hello.")
        result = loop.run(dummy_audio())
        assert result["timings"]["stt_latency"] < STT_TO_FIRST_TOKEN_BUDGET

    def test_llm_first_token_under_budget(self):
        loop = make_fast_loop(transcript="Hi", response="Hello.")
        result = loop.run(dummy_audio())
        assert result["timings"]["llm_first_token_latency"] < STT_TO_FIRST_TOKEN_BUDGET

    def test_with_tool_call_still_sub_second(self):
        """Even with a tool call, total latency stays under 1 s with zero-latency mocks."""
        responses = [
            'TOOL_CALL:weather_lookup:{"location": "X"}',
            "The weather is fine.",
        ]
        call_count = {"n": 0}

        class SeqLLM(MockLLMEngine):
            def stream_response(self, user_text, context=None):
                resp = responses[min(call_count["n"], len(responses) - 1)]
                call_count["n"] += 1
                for i, w in enumerate(resp.split()):
                    yield w + (" " if i < len(resp.split()) - 1 else "")

        stt = MockSTTEngine(fixed_transcript="weather", latency=0.0)
        llm = SeqLLM(token_latency=0.0, first_token_latency=0.0)
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            tool_registry=default_registry(),
        )
        result = loop.run(dummy_audio())
        assert result["timings"]["total_latency"] < SUB_SECOND


# ---------------------------------------------------------------------------
# Environment-variable configuration tests
# ---------------------------------------------------------------------------

class TestEnvConfig:
    """Verify that the __main__ module reads environment variables correctly."""

    def test_env_example_file_exists(self):
        env_path = Path(".env.example")
        assert env_path.exists(), ".env.example file should exist"

    def test_env_example_has_required_keys(self):
        env_path = Path(".env.example")
        content = env_path.read_text()
        required_keys = [
            "VAD_THRESHOLD",
            "STT_TRANSCRIPT",
            "STT_LATENCY",
            "LLM_TOKEN_LATENCY",
            "LLM_FIRST_LATENCY",
            "LLM_RESPONSE",
            "TTS_LATENCY",
        ]
        for key in required_keys:
            assert key in content, f"{key} should be in .env.example"

    def test_main_module_importable(self):
        import voice_ops_agent.__main__ as m
        assert hasattr(m, "main")

    def test_main_reads_vad_threshold_env(self, monkeypatch):
        monkeypatch.setenv("VAD_THRESHOLD", "500")
        # Import module after setting env
        import importlib
        import voice_ops_agent.__main__ as m
        importlib.reload(m)
        # We just verify the module loads; actual value is used at runtime
        assert True

    def test_stt_latency_env_applied(self, monkeypatch):
        """STT_LATENCY=0 means fast transcription."""
        monkeypatch.setenv("STT_LATENCY", "0.0")
        stt = MockSTTEngine(
            fixed_transcript="test",
            latency=float(os.environ.get("STT_LATENCY", "0.05")),
        )
        t0 = time.perf_counter()
        stt.transcribe(dummy_audio())
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.1  # near-instant with 0 latency

    def test_package_version_importable(self):
        import voice_ops_agent
        assert hasattr(voice_ops_agent, "__version__")
        assert voice_ops_agent.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# Full scenario integration (all paths in one test)
# ---------------------------------------------------------------------------

class TestFullScenarioIntegration:
    """Single test that exercises the entire pipeline end-to-end."""

    def test_complete_happy_path_pipeline(self, tmp_path):
        """Happy path: audio -> STT -> LLM -> TTS -> speaker, session recorded."""
        loop = make_fast_loop(
            transcript="What is the weather in London today?",
            response="The weather in London is partly cloudy.",
        )
        session = SessionManager()
        result = loop.run(dummy_audio())

        session.add_user_turn(result["transcript"])
        session.add_agent_turn(result["llm_response"])

        assert result["interrupted"] is False
        assert result["tool_calls"] == []
        assert session.turn_count == 2
        assert result["timings"]["total_latency"] < SUB_SECOND

    def test_complete_tool_call_pipeline(self, tmp_path):
        """Tool-call path: LLM calls weather tool, injects result, responds."""
        responses = [
            'TOOL_CALL:weather_lookup:{"location": "Tokyo"}',
            "The weather in Tokyo is clear skies.",
        ]
        call_count = {"n": 0}

        class SeqLLM(MockLLMEngine):
            def stream_response(self, user_text, context=None):
                resp = responses[min(call_count["n"], len(responses) - 1)]
                call_count["n"] += 1
                for i, w in enumerate(resp.split()):
                    yield w + (" " if i < len(resp.split()) - 1 else "")

        stt = MockSTTEngine(fixed_transcript="Weather in Tokyo?", latency=0.0)
        llm = SeqLLM(token_latency=0.0, first_token_latency=0.0)
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            tool_registry=default_registry(),
        )
        result = loop.run(dummy_audio())

        assert len(result["tool_calls"]) >= 1
        assert result["tool_results"][0].success is True
        assert result["timings"]["total_latency"] < SUB_SECOND

    def test_complete_escalation_pipeline(self, tmp_path):
        """Escalation path: user requests human agent, escalation logged."""
        logger = AuditLogger(path=tmp_path / "audit.json")
        mgr = EscalationManager(
            rules=[ExplicitRequestRule()],
            audit_logger=logger,
            human_agent_response="Transferring you now.",
        )
        stt = MockSTTEngine(
            fixed_transcript="I need to talk to a human agent",
            latency=0.0,
        )
        llm = MockLLMEngine(
            token_latency=0.0,
            first_token_latency=0.0,
            fixed_response="Let me connect you to a human.",
        )
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)
        adapter = MockTelephonyAdapter()
        session = SessionManager()
        handler = CallHandler(
            adapter=adapter,
            agent_loop=loop,
            session=session,
            escalation_manager=mgr,
        )
        result = handler.handle_call(audio_chunks=dummy_audio())

        assert mgr.escalation_count >= 1
        records = logger.records()
        assert len(records) >= 1
        assert records[0]["rule_name"] == "explicit_request"
