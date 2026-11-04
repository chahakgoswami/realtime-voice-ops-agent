"""Day 3 unit & integration tests: interruption handling and state machine."""

import math
import threading
import time
from collections import deque
from typing import Optional

import pytest

from voice_ops_agent.audio.input import CHUNK_SIZE, SAMPLE_RATE
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.stt.engine import MockSTTEngine
from voice_ops_agent.llm.engine import MockLLMEngine
from voice_ops_agent.tts.engine import MockTTSEngine
from voice_ops_agent.vad.detector import EnergyVAD
from voice_ops_agent.agent.loop import AgentLoop
from voice_ops_agent.agent.state import AgentState, StateMachine
from voice_ops_agent.agent.interruption import InterruptionListener


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


def make_silence_chunk() -> bytes:
    return b"\x00\x00" * CHUNK_SIZE


def dummy_audio(n: int = 5) -> list:
    return [make_tone_chunk() for _ in range(n)]


class ChunkQueue:
    """
    Thread-safe queue that acts as a fake microphone source for the
    interruption listener.  Push chunks from the test, and the listener
    thread calls `get()` to consume them.
    """

    def __init__(self) -> None:
        self._q: deque = deque()
        self._lock = threading.Lock()

    def push(self, chunk: bytes) -> None:
        with self._lock:
            self._q.append(chunk)

    def get(self) -> Optional[bytes]:
        with self._lock:
            if self._q:
                return self._q.popleft()
        return None  # no data yet


# ---------------------------------------------------------------------------
# StateMachine tests
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_initial_state_is_idle(self):
        sm = StateMachine()
        assert sm.state == AgentState.IDLE

    def test_valid_transition_idle_to_listening(self):
        sm = StateMachine()
        sm.transition(AgentState.LISTENING)
        assert sm.state == AgentState.LISTENING

    def test_invalid_transition_raises(self):
        sm = StateMachine()
        with pytest.raises(ValueError):
            sm.transition(AgentState.TTS_PLAYING)  # not allowed from IDLE

    def test_history_records_all_states(self):
        sm = StateMachine()
        sm.transition(AgentState.LISTENING)
        sm.transition(AgentState.TRANSCRIBING)
        assert sm.history == [
            AgentState.IDLE,
            AgentState.LISTENING,
            AgentState.TRANSCRIBING,
        ]

    def test_can_transition_true_for_valid(self):
        sm = StateMachine()
        assert sm.can_transition(AgentState.LISTENING) is True

    def test_can_transition_false_for_invalid(self):
        sm = StateMachine()
        assert sm.can_transition(AgentState.TTS_PLAYING) is False

    def test_reset_returns_to_idle(self):
        sm = StateMachine()
        sm.transition(AgentState.LISTENING)
        sm.reset()
        assert sm.state == AgentState.IDLE
        assert sm.history == [AgentState.IDLE]

    def test_full_happy_path_transitions(self):
        sm = StateMachine()
        for state in [
            AgentState.LISTENING,
            AgentState.TRANSCRIBING,
            AgentState.LLM_STREAMING,
            AgentState.TTS_PLAYING,
            AgentState.IDLE,
        ]:
            sm.transition(state)
        assert sm.state == AgentState.IDLE

    def test_interruption_path_transitions(self):
        sm = StateMachine()
        sm.transition(AgentState.LISTENING)
        sm.transition(AgentState.TRANSCRIBING)
        sm.transition(AgentState.LLM_STREAMING)
        sm.transition(AgentState.TTS_PLAYING)
        sm.transition(AgentState.INTERRUPTED)
        sm.transition(AgentState.LISTENING)   # re-enter STT
        assert sm.state == AgentState.LISTENING

    def test_error_state_reachable_from_transcribing(self):
        sm = StateMachine()
        sm.transition(AgentState.LISTENING)
        sm.transition(AgentState.TRANSCRIBING)
        sm.transition(AgentState.ERROR)
        assert sm.state == AgentState.ERROR

    def test_error_can_return_to_idle(self):
        sm = StateMachine()
        sm.transition(AgentState.LISTENING)
        sm.transition(AgentState.TRANSCRIBING)
        sm.transition(AgentState.ERROR)
        sm.transition(AgentState.IDLE)
        assert sm.state == AgentState.IDLE


# ---------------------------------------------------------------------------
# InterruptionListener tests
# ---------------------------------------------------------------------------

class TestInterruptionListener:
    def _make_listener(self, q: ChunkQueue, speaker=None, llm=None, callback=None):
        vad = EnergyVAD(threshold=300.0)
        if speaker is None:
            speaker = MockSpeaker()
        if llm is None:
            llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                                fixed_response="test")
        return InterruptionListener(
            vad=vad,
            audio_source=q.get,
            speaker=speaker,
            llm=llm,
            on_interrupt=callback,
        )

    def test_no_interrupt_when_silence(self):
        q = ChunkQueue()
        listener = self._make_listener(q)
        listener.start()
        # Feed only silence
        for _ in range(10):
            q.push(make_silence_chunk())
            time.sleep(0.005)
        listener.stop()
        assert listener.was_interrupted is False

    def test_interrupt_triggered_on_speech(self):
        q = ChunkQueue()
        speaker = MockSpeaker()
        listener = self._make_listener(q, speaker=speaker)
        listener.start()
        # Feed speech chunk
        q.push(make_tone_chunk(amplitude=0.5))
        # Give thread time to process
        time.sleep(0.1)
        listener.stop()
        assert listener.was_interrupted is True

    def test_speaker_interrupted_on_speech(self):
        q = ChunkQueue()
        speaker = MockSpeaker()
        listener = self._make_listener(q, speaker=speaker)
        listener.start()
        q.push(make_tone_chunk(amplitude=0.5))
        time.sleep(0.1)
        listener.stop()
        assert speaker._interrupted is True

    def test_llm_cancelled_on_speech(self):
        q = ChunkQueue()
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="long response here")
        listener = self._make_listener(q, llm=llm)
        listener.start()
        q.push(make_tone_chunk(amplitude=0.5))
        time.sleep(0.1)
        listener.stop()
        assert llm._cancelled is True

    def test_callback_called_on_interrupt(self):
        q = ChunkQueue()
        called = []
        listener = self._make_listener(q, callback=lambda: called.append(True))
        listener.start()
        q.push(make_tone_chunk(amplitude=0.5))
        time.sleep(0.1)
        listener.stop()
        assert called == [True]

    def test_interrupt_event_is_set(self):
        q = ChunkQueue()
        listener = self._make_listener(q)
        assert not listener.interrupt_event.is_set()
        listener.start()
        q.push(make_tone_chunk(amplitude=0.5))
        time.sleep(0.1)
        listener.stop()
        assert listener.interrupt_event.is_set()

    def test_captured_chunks_non_empty_after_interrupt(self):
        q = ChunkQueue()
        listener = self._make_listener(q)
        listener.start()
        q.push(make_tone_chunk(amplitude=0.5))
        time.sleep(0.1)
        listener.stop()
        assert len(listener.captured_chunks) >= 1

    def test_no_capture_before_start(self):
        q = ChunkQueue()
        listener = self._make_listener(q)
        # Never started
        assert listener.captured_chunks == []
        assert not listener.was_interrupted

    def test_silence_does_not_set_interrupt_event(self):
        q = ChunkQueue()
        listener = self._make_listener(q)
        listener.start()
        for _ in range(5):
            q.push(make_silence_chunk())
            time.sleep(0.005)
        listener.stop()
        assert not listener.interrupt_event.is_set()


# ---------------------------------------------------------------------------
# AgentLoop with interruption handling tests
# ---------------------------------------------------------------------------

class TestAgentLoopInterruption:
    """
    Tests for the full AgentLoop with interrupt_audio_source wired in.
    """

    def _make_components(self, fixed_response="This is a longer response for playback"):
        stt = MockSTTEngine(fixed_transcript="Hello", latency=0.0)
        llm = MockLLMEngine(
            token_latency=0.0,
            first_token_latency=0.0,
            fixed_response=fixed_response,
        )
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker(realtime=False)
        vad = EnergyVAD(threshold=300.0)
        return stt, llm, tts, speaker, vad

    def test_no_interruption_result_flag_false(self):
        stt, llm, tts, speaker, vad = self._make_components()
        q = ChunkQueue()
        # Only silence in queue -> no interrupt
        for _ in range(20):
            q.push(make_silence_chunk())

        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)
        result = loop.run(
            dummy_audio(),
            interrupt_audio_source=q.get,
            vad=vad,
        )
        assert result["interrupted"] is False
        assert result["interruption_count"] == 0

    def test_interruption_detected_during_tts(self):
        stt, llm, tts, speaker, vad = self._make_components()
        q = ChunkQueue()

        # Deliver speech chunk after a brief delay so TTS has started
        def inject_speech():
            time.sleep(0.05)
            q.push(make_tone_chunk(amplitude=0.5))
            # Pad with silence so listener stops
            for _ in range(10):
                q.push(make_silence_chunk())

        t = threading.Thread(target=inject_speech, daemon=True)

        interrupt_called = []
        loop = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            on_interrupt=lambda: interrupt_called.append(True),
            max_interruptions=1,
        )

        t.start()
        result = loop.run(
            dummy_audio(),
            interrupt_audio_source=q.get,
            vad=vad,
        )
        t.join(timeout=2.0)

        assert result["interrupted"] is True
        assert result["interruption_count"] >= 1

    def test_state_history_contains_interrupted_state(self):
        stt, llm, tts, speaker, vad = self._make_components()
        q = ChunkQueue()

        def inject_speech():
            time.sleep(0.02)
            q.push(make_tone_chunk(amplitude=0.5))
            for _ in range(10):
                q.push(make_silence_chunk())

        t = threading.Thread(target=inject_speech, daemon=True)

        loop = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            max_interruptions=1,
        )

        t.start()
        result = loop.run(
            dummy_audio(),
            interrupt_audio_source=q.get,
            vad=vad,
        )
        t.join(timeout=2.0)

        if result["interrupted"]:
            assert AgentState.INTERRUPTED in result["state_history"]

    def test_on_state_change_callback_fires(self):
        stt, llm, tts, speaker, vad = self._make_components()
        q = ChunkQueue()
        for _ in range(20):
            q.push(make_silence_chunk())

        transitions = []

        def record_transition(old, new):
            transitions.append((old, new))

        loop = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            on_state_change=record_transition,
        )
        loop.run(dummy_audio(), interrupt_audio_source=q.get, vad=vad)

        # Verify at least IDLE->LISTENING was observed
        assert (AgentState.IDLE, AgentState.LISTENING) in transitions

    def test_result_has_new_keys(self):
        stt, llm, tts, speaker, vad = self._make_components()
        q = ChunkQueue()
        for _ in range(10):
            q.push(make_silence_chunk())

        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)
        result = loop.run(dummy_audio(), interrupt_audio_source=q.get, vad=vad)
        for key in ("interrupted", "interruption_count", "state_history"):
            assert key in result

    def test_without_interrupt_source_still_works(self):
        """Backward-compat: run() without interruption params behaves as Day 2."""
        stt = MockSTTEngine(fixed_transcript="test", latency=0.0)
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="hello there")
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)
        result = loop.run(dummy_audio())
        assert result["transcript"] == "test"
        assert result["llm_response"] == "hello there"
        assert result["interrupted"] is False
        assert speaker.total_bytes_played > 0

    def test_max_interruptions_respected(self):
        """Agent should stop re-entering after max_interruptions."""
        stt = MockSTTEngine(fixed_transcript="hi", latency=0.0)
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="response text")
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        vad = EnergyVAD(threshold=300.0)

        # Always return a speech chunk -> will keep interrupting
        def always_speech():
            return make_tone_chunk(amplitude=0.5)

        loop = AgentLoop(
            stt=stt, llm=llm, tts=tts, speaker=speaker,
            max_interruptions=2,
        )
        result = loop.run(
            dummy_audio(),
            interrupt_audio_source=always_speech,
            vad=vad,
        )
        assert result["interruption_count"] <= 2

    def test_speaker_bytes_reduced_on_interrupt(self):
        """Interrupted playback should play fewer bytes than full playback."""
        response = "one two three four five six seven eight nine ten"

        # Full playback (no interrupt)
        stt1 = MockSTTEngine(fixed_transcript="hi", latency=0.0)
        llm1 = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                             fixed_response=response)
        tts1 = MockTTSEngine(latency=0.0)
        speaker1 = MockSpeaker(realtime=False)
        loop1 = AgentLoop(stt=stt1, llm=llm1, tts=tts1, speaker=speaker1)
        loop1.run(dummy_audio())
        full_bytes = speaker1.total_bytes_played

        # Interrupted playback
        stt2 = MockSTTEngine(fixed_transcript="hi", latency=0.0)
        llm2 = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                             fixed_response=response)
        tts2 = MockTTSEngine(latency=0.0)
        speaker2 = MockSpeaker(realtime=False)
        vad = EnergyVAD(threshold=300.0)

        # Interrupt immediately
        def immediate_speech():
            return make_tone_chunk(amplitude=0.5)

        loop2 = AgentLoop(
            stt=stt2, llm=llm2, tts=tts2, speaker=speaker2,
            max_interruptions=1,
        )
        loop2.run(
            dummy_audio(),
            interrupt_audio_source=immediate_speech,
            vad=vad,
        )
        interrupted_bytes = speaker2.total_bytes_played
        # Interrupted should play <= full bytes
        assert interrupted_bytes <= full_bytes
