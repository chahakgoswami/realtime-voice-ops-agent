"""Day 1 unit tests: audio I/O, VAD, and STT mock."""

import math
import struct
import pytest

from voice_ops_agent.audio.input import MockMicrophone, CHUNK_SIZE, SAMPLE_RATE
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.vad.detector import EnergyVAD
from voice_ops_agent.stt.engine import MockSTTEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tone_chunk(amplitude: float = 0.5, frequency: float = 440.0) -> bytes:
    """Build a single CHUNK_SIZE chunk of a sine-wave tone."""
    samples = []
    for i in range(CHUNK_SIZE):
        t = i / SAMPLE_RATE
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        samples.append(value.to_bytes(2, byteorder="little", signed=True))
    return b"".join(samples)


def make_silence_chunk() -> bytes:
    return b"\x00\x00" * CHUNK_SIZE


# ---------------------------------------------------------------------------
# EnergyVAD
# ---------------------------------------------------------------------------

class TestEnergyVAD:
    def test_silence_is_not_speech(self):
        vad = EnergyVAD(threshold=300.0)
        silence = make_silence_chunk()
        assert vad.is_speech(silence) is False

    def test_tone_is_speech(self):
        vad = EnergyVAD(threshold=300.0)
        tone = make_tone_chunk(amplitude=0.5)
        assert vad.is_speech(tone) is True

    def test_rms_energy_silence_is_zero(self):
        vad = EnergyVAD()
        energy = vad.rms_energy(make_silence_chunk())
        assert energy == pytest.approx(0.0)

    def test_rms_energy_tone_is_positive(self):
        vad = EnergyVAD()
        energy = vad.rms_energy(make_tone_chunk(amplitude=0.5))
        assert energy > 0.0

    def test_last_energy_updated(self):
        vad = EnergyVAD()
        assert vad.last_energy is None
        vad.is_speech(make_tone_chunk())
        assert vad.last_energy is not None

    def test_empty_chunk_returns_zero(self):
        vad = EnergyVAD()
        assert vad.rms_energy(b"") == pytest.approx(0.0)

    def test_threshold_boundary(self):
        vad = EnergyVAD(threshold=300.0)
        tone = make_tone_chunk(amplitude=0.5)
        energy = vad.rms_energy(tone)
        # Expect energy well above threshold for amplitude=0.5
        assert energy > 300.0

        vad_high = EnergyVAD(threshold=100_000.0)
        assert vad_high.is_speech(tone) is False


# ---------------------------------------------------------------------------
# MockMicrophone
# ---------------------------------------------------------------------------

class TestMockMicrophone:
    def test_stream_yields_correct_chunk_count(self):
        mic = MockMicrophone(silence_chunks=2, speech_chunks=5, silence_after=3)
        chunks = list(mic.stream())
        assert len(chunks) == 2 + 5 + 3

    def test_all_chunks_are_bytes(self):
        mic = MockMicrophone(silence_chunks=1, speech_chunks=3, silence_after=1)
        for chunk in mic.stream():
            assert isinstance(chunk, bytes)

    def test_speech_chunks_have_energy(self):
        vad = EnergyVAD(threshold=300.0)
        mic = MockMicrophone(silence_chunks=0, speech_chunks=10, silence_after=0,
                             amplitude=0.5)
        for chunk in mic.stream():
            assert vad.is_speech(chunk) is True

    def test_silence_chunks_have_no_energy(self):
        vad = EnergyVAD(threshold=300.0)
        mic = MockMicrophone(silence_chunks=5, speech_chunks=0, silence_after=0)
        for chunk in mic.stream():
            assert vad.is_speech(chunk) is False

    def test_record_until_silence_returns_nonempty(self):
        vad = EnergyVAD(threshold=300.0)
        mic = MockMicrophone(silence_chunks=2, speech_chunks=10, silence_after=10)
        recorded = mic.record_until_silence(vad)
        assert len(recorded) > 0

    def test_record_until_silence_stops_after_trailing_silence(self):
        vad = EnergyVAD(threshold=300.0)
        # 5 speech + 20 silence → should stop well before consuming all
        mic = MockMicrophone(silence_chunks=0, speech_chunks=5, silence_after=20)
        recorded = mic.record_until_silence(vad)
        # Should not return all 25 chunks; stops at ~8 trailing silent chunks
        assert len(recorded) < 25


# ---------------------------------------------------------------------------
# MockSpeaker
# ---------------------------------------------------------------------------

class TestMockSpeaker:
    def test_play_stores_chunks(self):
        speaker = MockSpeaker()
        data = b"\x01\x02" * 1024
        speaker.play(data)
        assert speaker.total_bytes_played > 0

    def test_interrupt_stops_playback(self):
        speaker = MockSpeaker()
        speaker._interrupted = True
        chunk = b"\x00" * 64
        result = speaker.play_chunk(chunk)
        assert result is False
        assert speaker.total_bytes_played == 0

    def test_reset_clears_state(self):
        speaker = MockSpeaker()
        speaker.play(b"\x01" * 512)
        speaker.reset()
        assert speaker.total_bytes_played == 0
        assert speaker._interrupted is False

    def test_interrupt_method(self):
        speaker = MockSpeaker()
        speaker.interrupt()
        assert speaker._interrupted is True


# ---------------------------------------------------------------------------
# MockSTTEngine
# ---------------------------------------------------------------------------

class TestMockSTTEngine:
    def test_fixed_transcript_returned(self):
        stt = MockSTTEngine(fixed_transcript="hello world", latency=0.0)
        result = stt.transcribe([make_tone_chunk()])
        assert result["text"] == "hello world"

    def test_result_has_required_keys(self):
        stt = MockSTTEngine(latency=0.0)
        result = stt.transcribe([make_tone_chunk()])
        assert "text" in result
        assert "confidence" in result
        assert "duration_s" in result

    def test_confidence_is_float(self):
        stt = MockSTTEngine(confidence=0.88, latency=0.0)
        result = stt.transcribe([make_tone_chunk()])
        assert isinstance(result["confidence"], float)
        assert result["confidence"] == pytest.approx(0.88)

    def test_duration_is_positive(self):
        stt = MockSTTEngine(latency=0.0)
        chunks = [make_tone_chunk() for _ in range(5)]
        result = stt.transcribe(chunks)
        assert result["duration_s"] > 0.0

    def test_empty_chunks_gives_zero_duration(self):
        stt = MockSTTEngine(fixed_transcript="empty", latency=0.0)
        result = stt.transcribe([])
        assert result["duration_s"] == pytest.approx(0.0)

    def test_random_transcript_is_string(self):
        stt = MockSTTEngine(fixed_transcript=None, latency=0.0)
        result = stt.transcribe([make_tone_chunk()])
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
