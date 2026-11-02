"""Simple energy-threshold Voice Activity Detector."""

import struct
import math
from typing import Optional


class EnergyVAD:
    """
    A VAD stub that classifies audio chunks as speech or silence based on
    RMS energy compared to a configurable threshold.

    Parameters
    ----------
    threshold : float
        RMS energy threshold.  Chunks with energy above this value are
        classified as speech.  Typical values lie in the range 100–2000
        for 16-bit PCM at 16 kHz.
    sample_width : int
        Number of bytes per sample (default 2 for 16-bit PCM).
    """

    def __init__(self, threshold: float = 300.0, sample_width: int = 2) -> None:
        self.threshold = threshold
        self.sample_width = sample_width
        self._last_energy: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rms_energy(self, chunk: bytes) -> float:
        """
        Compute the Root-Mean-Square energy of a PCM chunk.

        Parameters
        ----------
        chunk : bytes
            Raw PCM audio bytes.

        Returns
        -------
        float
            RMS energy value.
        """
        if len(chunk) < self.sample_width:
            return 0.0

        fmt = "<" + "h" * (len(chunk) // self.sample_width)  # little-endian int16
        samples = struct.unpack(fmt, chunk[: len(chunk) - len(chunk) % self.sample_width])
        if not samples:
            return 0.0

        mean_sq = sum(s * s for s in samples) / len(samples)
        rms = math.sqrt(mean_sq)
        self._last_energy = rms
        return rms

    def is_speech(self, chunk: bytes) -> bool:
        """
        Return True if the chunk is classified as speech.

        Parameters
        ----------
        chunk : bytes
            Raw PCM audio bytes.

        Returns
        -------
        bool
        """
        return self.rms_energy(chunk) > self.threshold

    @property
    def last_energy(self) -> Optional[float]:
        """RMS energy of the most recently processed chunk."""
        return self._last_energy
