"""Mock speaker output via a simulated stream."""

import time
from typing import List


CHUNK_SIZE = 512
SAMPLE_RATE = 16_000


class MockSpeaker:
    """
    Simulated speaker that accepts PCM chunks and 'plays' them.

    In this mock implementation, playback simply collects chunks and
    optionally sleeps to simulate real-time playback speed.

    Attributes
    ----------
    played_chunks : List[bytes]
        All chunks that were submitted for playback.
    """

    def __init__(self, realtime: bool = False) -> None:
        self.realtime = realtime
        self.played_chunks: List[bytes] = []
        self._chunk_duration = CHUNK_SIZE / SAMPLE_RATE
        self._interrupted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play_chunk(self, chunk: bytes) -> bool:
        """
        'Play' a single PCM chunk.

        Returns
        -------
        bool
            True if playback succeeded, False if it was interrupted.
        """
        if self._interrupted:
            return False
        self.played_chunks.append(chunk)
        if self.realtime:
            time.sleep(self._chunk_duration)
        return True

    def play(self, audio_data: bytes) -> None:
        """
        'Play' a raw PCM buffer by splitting it into chunks.

        Stops early if interrupted.
        """
        self._interrupted = False
        for i in range(0, len(audio_data), CHUNK_SIZE * 2):  # *2 for 16-bit
            chunk = audio_data[i : i + CHUNK_SIZE * 2]
            if not self.play_chunk(chunk):
                break

    def interrupt(self) -> None:
        """Signal that playback should stop immediately."""
        self._interrupted = True

    def reset(self) -> None:
        """Reset state for a new utterance."""
        self._interrupted = False
        self.played_chunks = []

    @property
    def total_bytes_played(self) -> int:
        """Total bytes that have been 'played' so far."""
        return sum(len(c) for c in self.played_chunks)
