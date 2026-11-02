"""Mock microphone input via simulated audio streams."""

import time
import random
import math
from typing import Iterator, List


CHUNK_SIZE = 512          # samples per chunk
SAMPLE_RATE = 16_000      # Hz
SAMPLE_WIDTH = 2          # bytes (16-bit PCM)


def _sine_wave_chunk(frequency: float, chunk_size: int, offset: int, amplitude: float) -> bytes:
    """Generate a chunk of 16-bit PCM sine-wave samples."""
    samples = []
    for i in range(chunk_size):
        t = (offset + i) / SAMPLE_RATE
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        # little-endian 16-bit
        samples.append(value.to_bytes(2, byteorder="little", signed=True))
    return b"".join(samples)


def _silence_chunk(chunk_size: int) -> bytes:
    """Return a chunk of silence (all zeros)."""
    return b"\x00\x00" * chunk_size


class MockMicrophone:
    """
    Simulated microphone that yields PCM audio chunks.

    Behaviour
    ---------
    * ``silence_chunks``  – number of silent chunks before speech begins.
    * ``speech_chunks``   – number of chunks that contain a simulated speech tone.
    * ``silence_after``   – number of silent chunks after speech (trailing silence).
    * ``frequency``       – tone frequency used for the speech simulation (Hz).
    * ``amplitude``       – relative amplitude of the tone [0.0 – 1.0].
    """

    def __init__(
        self,
        silence_chunks: int = 4,
        speech_chunks: int = 20,
        silence_after: int = 6,
        frequency: float = 440.0,
        amplitude: float = 0.5,
        realtime: bool = False,
    ) -> None:
        self.silence_chunks = silence_chunks
        self.speech_chunks = speech_chunks
        self.silence_after = silence_after
        self.frequency = frequency
        self.amplitude = amplitude
        self.realtime = realtime  # if True, sleep between chunks to simulate real time
        self._chunk_duration = CHUNK_SIZE / SAMPLE_RATE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(self) -> Iterator[bytes]:
        """Yield PCM chunks that simulate a microphone stream."""
        offset = 0

        for _ in range(self.silence_chunks):
            yield _silence_chunk(CHUNK_SIZE)
            if self.realtime:
                time.sleep(self._chunk_duration)

        for i in range(self.speech_chunks):
            chunk = _sine_wave_chunk(self.frequency, CHUNK_SIZE, offset, self.amplitude)
            offset += CHUNK_SIZE
            yield chunk
            if self.realtime:
                time.sleep(self._chunk_duration)

        for _ in range(self.silence_after):
            yield _silence_chunk(CHUNK_SIZE)
            if self.realtime:
                time.sleep(self._chunk_duration)

    def record_until_silence(
        self,
        vad,
        max_chunks: int = 200,
    ) -> List[bytes]:
        """
        Record chunks from the simulated stream, collecting audio while speech
        is detected and stopping after a run of trailing silence.

        Parameters
        ----------
        vad:
            A VAD instance with an ``is_speech(chunk) -> bool`` method.
        max_chunks:
            Safety limit on total chunks collected.

        Returns
        -------
        List of PCM byte chunks that were classified as speech (plus context).
        """
        recorded: List[bytes] = []
        trailing_silence = 0
        SILENCE_LIMIT = 8  # consecutive silent chunks to stop

        in_speech = False
        for chunk in self.stream():
            if len(recorded) >= max_chunks:
                break

            if vad.is_speech(chunk):
                in_speech = True
                trailing_silence = 0
                recorded.append(chunk)
            else:
                if in_speech:
                    trailing_silence += 1
                    recorded.append(chunk)  # keep trailing silence for context
                    if trailing_silence >= SILENCE_LIMIT:
                        break
                # pre-speech silence: skip

        return recorded
