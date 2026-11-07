"""Speaker diarization stub.

In a production system this would use an acoustic model (e.g. pyannote.audio)
to identify speakers from raw audio.  Here we provide a lightweight stub that:

1. Accepts a ``hint`` Speaker label (the caller's intent).
2. Applies simple heuristics / configurable overrides to return a final label.
3. Tracks how many turns each speaker has had.

This is enough to wire diarization into the session manager and validate the
interface without requiring a real model.
"""

from typing import Dict, List, Optional

from .manager import Speaker  # type: ignore[import-not-found]


class DiarizationStub:
    """
    Lightweight speaker diarization stub.

    By default the stub accepts the caller-supplied *hint* label as truth.
    Optionally a fixed ``override_speaker`` can be set so that every turn
    is relabelled to that speaker (useful for unit tests).

    Parameters
    ----------
    override_speaker : Speaker, optional
        When provided, every ``assign_speaker`` call returns this value.
    """

    def __init__(self, override_speaker: Optional[Speaker] = None) -> None:
        self.override_speaker = override_speaker
        self._speaker_counts: Dict[str, int] = {s.value: 0 for s in Speaker}
        self._history: List[Speaker] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_speaker(
        self,
        text: str,  # noqa: ARG002  – reserved for future acoustic features
        hint: Optional[Speaker] = None,
    ) -> Speaker:
        """
        Assign a speaker label for the current audio segment.

        Parameters
        ----------
        text : str
            Text content of the segment (reserved for future use).
        hint : Speaker, optional
            Suggested speaker from the caller (e.g. based on turn order).

        Returns
        -------
        Speaker
            The assigned speaker label.
        """
        if self.override_speaker is not None:
            speaker = self.override_speaker
        elif hint is not None:
            speaker = hint
        else:
            # Alternate between USER and AGENT if no hint given
            if self._history and self._history[-1] == Speaker.AGENT:
                speaker = Speaker.USER
            else:
                speaker = Speaker.AGENT

        self._speaker_counts[speaker.value] += 1
        self._history.append(speaker)
        return speaker

    @property
    def speaker_counts(self) -> Dict[str, int]:
        """Return a copy of per-speaker turn counts."""
        return dict(self._speaker_counts)

    @property
    def history(self) -> List[Speaker]:
        """All speaker assignments in order."""
        return list(self._history)

    def reset(self) -> None:
        """Reset state (useful between sessions)."""
        self._speaker_counts = {s.value: 0 for s in Speaker}
        self._history = []

    def __repr__(self) -> str:
        return f"<DiarizationStub counts={self._speaker_counts}>"
