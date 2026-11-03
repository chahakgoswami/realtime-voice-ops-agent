"""Mocked Text-to-Speech engine."""

import math
import time
from typing import Iterator, Optional


SAMPLE_RATE = 16_000   # Hz
SAMPLE_WIDTH = 2       # bytes (16-bit PCM)
_BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH   # 32 000

# Approximate speaking rate used to size synthesised audio
_WORDS_PER_SECOND = 3.0


def _generate_tone_chunk(
    frequency: float,
    num_samples: int,
    sample_offset: int,
    amplitude: float = 0.3,
) -> bytes:
    """Produce *num_samples* of a sine-wave at *frequency* Hz."""
    out = bytearray(num_samples * SAMPLE_WIDTH)
    for i in range(num_samples):
        t = (sample_offset + i) / SAMPLE_RATE
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        struct_bytes = value.to_bytes(SAMPLE_WIDTH, byteorder="little", signed=True)
        out[i * SAMPLE_WIDTH : (i + 1) * SAMPLE_WIDTH] = struct_bytes
    return bytes(out)


class MockTTSEngine:
    """
    A simulated TTS engine that converts text to PCM audio bytes.

    Rather than running a real TTS model the mock generates a sine-wave
    tone whose duration is proportional to the word count of the input
    text, simulating realistic audio length.

    Parameters
    ----------
    latency : float
        Delay in seconds before the first audio byte is produced
        (simulates model warm-up / network RTT).  Default 0.03 s.
    frequency : float
        Sine-wave frequency used for the synthetic speech tone (Hz).
    amplitude : float
        Amplitude of the tone [0.0 – 1.0].
    words_per_second : float
        Used to size the generated audio relative to text length.
    """

    def __init__(
        self,
        latency: float = 0.03,
        frequency: float = 220.0,
        amplitude: float = 0.3,
        words_per_second: float = _WORDS_PER_SECOND,
    ) -> None:
        self.latency = latency
        self.frequency = frequency
        self.amplitude = amplitude
        self.words_per_second = words_per_second

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _text_to_duration(self, text: str) -> float:
        """Estimate speech duration in seconds for *text*."""
        word_count = max(1, len(text.split()))
        return word_count / self.words_per_second

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesise(self, text: str) -> bytes:
        """
        Convert *text* to PCM audio bytes (blocking).

        Parameters
        ----------
        text : str
            Text to be spoken.

        Returns
        -------
        bytes
            Raw 16-bit PCM audio at SAMPLE_RATE Hz.
        """
        if self.latency > 0:
            time.sleep(self.latency)

        duration = self._text_to_duration(text)
        num_samples = int(SAMPLE_RATE * duration)
        return _generate_tone_chunk(self.frequency, num_samples, 0, self.amplitude)

    def stream_synthesise(self, text: str, chunk_samples: int = 512) -> Iterator[bytes]:
        """
        Yield PCM chunks progressively as they are 'synthesised'.

        This mirrors how a real streaming TTS API returns audio: the
        first chunk arrives after *latency* seconds, and subsequent
        chunks arrive immediately (since we pre-compute the full buffer).

        Parameters
        ----------
        text : str
            Text to be spoken.
        chunk_samples : int
            Number of samples per yielded chunk.

        Yields
        ------
        bytes
            PCM audio chunks.
        """
        if self.latency > 0:
            time.sleep(self.latency)

        duration = self._text_to_duration(text)
        total_samples = int(SAMPLE_RATE * duration)
        chunk_bytes = chunk_samples * SAMPLE_WIDTH

        # Generate full audio then stream it in chunks
        full_audio = _generate_tone_chunk(
            self.frequency, total_samples, 0, self.amplitude
        )
        for offset in range(0, len(full_audio), chunk_bytes):
            yield full_audio[offset : offset + chunk_bytes]
