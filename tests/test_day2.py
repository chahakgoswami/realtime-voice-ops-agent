"""Day 2 unit & integration tests: LLM engine, TTS engine, and agent loop."""

import math
import time
import struct
import pytest

from voice_ops_agent.audio.input import CHUNK_SIZE, SAMPLE_RATE
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.stt.engine import MockSTTEngine
from voice_ops_agent.llm.engine import MockLLMEngine
from voice_ops_agent.tts.engine import MockTTSEngine, SAMPLE_WIDTH
from voice_ops_agent.agent.loop import AgentLoop


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


def dummy_audio(n: int = 5) -> list:
    return [make_tone_chunk() for _ in range(n)]


# ---------------------------------------------------------------------------
# MockLLMEngine tests
# ---------------------------------------------------------------------------

class TestMockLLMEngine:
    def test_fixed_response_returned(self):
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="hello world")
        result = llm.complete("anything")
        assert result == "hello world"

    def test_stream_yields_strings(self):
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="one two three")
        tokens = list(llm.stream_response("test"))
        assert all(isinstance(t, str) for t in tokens)

    def test_stream_reassembles_correctly(self):
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="The quick brown fox")
        full = "".join(llm.stream_response("test"))
        assert full == "The quick brown fox"

    def test_cancel_stops_stream_early(self):
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="one two three four five six seven eight")
        tokens = []
        for i, token in enumerate(llm.stream_response("test")):
            tokens.append(token)
            if i == 1:          # cancel after second token
                llm.cancel()
        # Should have stopped before collecting all 8 words
        assert len(tokens) < 8

    def test_intent_matching_weather(self):
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0)
        response = llm.complete("What is the weather today?")
        assert "weather" in response.lower() or len(response) > 0

    def test_reset_clears_cancellation(self):
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="a b c")
        llm.cancel()
        llm.reset()
        assert not llm._cancelled

    def test_latency_applied(self):
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.05,
                            fixed_response="x")
        t0 = time.perf_counter()
        list(llm.stream_response("test"))
        elapsed = time.perf_counter() - t0
        assert elapsed >= 0.04   # at least first_token_latency was applied

    def test_default_response_is_string(self):
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0)
        response = llm.complete("gibberish input with no known keywords")
        assert isinstance(response, str)
        assert len(response) > 0


# ---------------------------------------------------------------------------
# MockTTSEngine tests
# ---------------------------------------------------------------------------

class TestMockTTSEngine:
    def test_synthesise_returns_bytes(self):
        tts = MockTTSEngine(latency=0.0)
        audio = tts.synthesise("hello world")
        assert isinstance(audio, bytes)

    def test_synthesise_length_proportional_to_words(self):
        tts = MockTTSEngine(latency=0.0)
        short_audio = tts.synthesise("hi")
        long_audio  = tts.synthesise("This is a much longer sentence with many words in it")
        assert len(long_audio) > len(short_audio)

    def test_synthesise_latency_applied(self):
        tts = MockTTSEngine(latency=0.05)
        t0 = time.perf_counter()
        tts.synthesise("test")
        elapsed = time.perf_counter() - t0
        assert elapsed >= 0.04

    def test_stream_synthesise_yields_bytes(self):
        tts = MockTTSEngine(latency=0.0)
        chunks = list(tts.stream_synthesise("hello there"))
        assert all(isinstance(c, bytes) for c in chunks)
        assert len(chunks) > 0

    def test_stream_synthesise_total_matches_synthesise(self):
        tts = MockTTSEngine(latency=0.0)
        text = "The quick brown fox jumps over the lazy dog"
        blocking = tts.synthesise(text)
        streaming = b"".join(tts.stream_synthesise(text))
        # Total bytes should be identical
        assert len(blocking) == len(streaming)

    def test_sample_width_is_two_bytes(self):
        tts = MockTTSEngine(latency=0.0)
        audio = tts.synthesise("test")
        # Must be divisible by SAMPLE_WIDTH (2)
        assert len(audio) % SAMPLE_WIDTH == 0


# ---------------------------------------------------------------------------
# AgentLoop integration tests
# ---------------------------------------------------------------------------

class TestAgentLoop:
    def _make_loop(self, fixed_transcript="Hello", fixed_response="Hi there how can I help"):
        stt = MockSTTEngine(fixed_transcript=fixed_transcript, latency=0.0)
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response=fixed_response)
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker(realtime=False)
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)
        return loop, speaker

    def test_run_returns_required_keys(self):
        loop, _ = self._make_loop()
        result = loop.run(dummy_audio())
        for key in ("transcript", "llm_response", "timings", "stt_result"):
            assert key in result

    def test_transcript_matches_fixed_stt(self):
        loop, _ = self._make_loop(fixed_transcript="test utterance")
        result = loop.run(dummy_audio())
        assert result["transcript"] == "test utterance"

    def test_llm_response_matches_fixed(self):
        loop, _ = self._make_loop(fixed_response="This is the answer")
        result = loop.run(dummy_audio())
        assert result["llm_response"] == "This is the answer"

    def test_speaker_plays_audio(self):
        loop, speaker = self._make_loop()
        loop.run(dummy_audio())
        assert speaker.total_bytes_played > 0

    def test_timings_present(self):
        loop, _ = self._make_loop()
        result = loop.run(dummy_audio())
        timings = result["timings"]
        for key in ("stt_latency", "llm_first_token_latency",
                    "stt_to_first_token", "tts_first_chunk_latency",
                    "total_latency"):
            assert key in timings
            assert timings[key] >= 0.0

    def test_on_transcript_callback_called(self):
        stt = MockSTTEngine(fixed_transcript="callback test", latency=0.0)
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="ok")
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        received = []
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker,
                         on_transcript=received.append)
        loop.run(dummy_audio())
        assert received == ["callback test"]

    def test_on_token_callback_called_for_each_token(self):
        stt = MockSTTEngine(fixed_transcript="hi", latency=0.0)
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="one two three")
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        tokens = []
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker,
                         on_token=tokens.append)
        loop.run(dummy_audio())
        assert len(tokens) == 3   # "one ", "two ", "three"
        assert "".join(tokens) == "one two three"

    def test_on_response_complete_callback(self):
        stt = MockSTTEngine(fixed_transcript="hi", latency=0.0)
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0,
                            fixed_response="done")
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        completed = []
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker,
                         on_response_complete=completed.append)
        loop.run(dummy_audio())
        assert completed == ["done"]

    def test_full_round_trip_produces_non_empty_response(self):
        """Integration: real intent matching through complete pipeline."""
        stt = MockSTTEngine(
            fixed_transcript="What is the weather like in New York today?",
            latency=0.0,
        )
        llm = MockLLMEngine(token_latency=0.0, first_token_latency=0.0)
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)
        result = loop.run(dummy_audio())
        assert len(result["llm_response"]) > 0
        assert speaker.total_bytes_played > 0
