"""Mocked Speech-to-Text engine."""

import random
import time
from typing import List, Optional


# A small corpus of plausible user utterances for simulation purposes.
_UTTERANCE_CORPUS = [
    "Hello, I need help with my order.",
    "What's the weather like in New York today?",
    "Can you check my calendar for tomorrow?",
    "I'd like to cancel my subscription please.",
    "Tell me about the latest product updates.",
    "Connect me to a human agent.",
    "What is the status of order number 12345?",
    "I'm having trouble logging into my account.",
]


class MockSTTEngine:
    """
    A simulated STT engine that converts raw PCM audio chunks into text.

    Rather than performing actual speech recognition, the mock either
    returns a caller-specified transcript or picks one at random from a
    built-in corpus.  An optional simulated latency can be added to
    mimic real network/model inference time.

    Parameters
    ----------
    fixed_transcript : str or None
        If provided, ``transcribe`` always returns this text regardless
        of the audio content.
    latency : float
        Simulated processing delay in seconds (default 0.05 s).
    confidence : float
        Reported confidence score [0.0 – 1.0] (default 0.95).
    """

    def __init__(
        self,
        fixed_transcript: Optional[str] = None,
        latency: float = 0.05,
        confidence: float = 0.95,
    ) -> None:
        self.fixed_transcript = fixed_transcript
        self.latency = latency
        self.confidence = confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_chunks: List[bytes]) -> dict:
        """
        'Transcribe' a list of PCM audio chunks.

        Parameters
        ----------
        audio_chunks : List[bytes]
            Recorded audio chunks from the microphone.

        Returns
        -------
        dict with keys:
            ``text``       – transcribed string
            ``confidence`` – float confidence score
            ``duration_s`` – approximate audio duration in seconds
        """
        if self.latency > 0:
            time.sleep(self.latency)

        total_bytes = sum(len(c) for c in audio_chunks)
        # 16-bit PCM, mono, at 16 kHz → 32 000 bytes/s
        duration_s = total_bytes / (16_000 * 2)

        if self.fixed_transcript is not None:
            text = self.fixed_transcript
        else:
            text = random.choice(_UTTERANCE_CORPUS)

        return {
            "text": text,
            "confidence": self.confidence,
            "duration_s": round(duration_s, 3),
        }
