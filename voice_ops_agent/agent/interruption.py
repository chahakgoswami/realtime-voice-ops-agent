"""Parallel listener thread that monitors VAD during TTS playback.

When speech is detected while the agent is speaking, the listener:
  1. Sets the interrupt_event so callers know interruption occurred.
  2. Calls speaker.interrupt() to halt audio playback.
  3. Calls llm.cancel() to abort the in-flight LLM stream.

Usage example::

    listener = InterruptionListener(vad=vad, mic=mic, speaker=speaker, llm=llm)
    listener.start()
    # ... play TTS audio ...
    listener.stop()
    if listener.was_interrupted:
        # re-enter STT phase
        ...
"""

import threading
import time
from typing import Callable, Optional

from voice_ops_agent.vad.detector import EnergyVAD
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.llm.engine import MockLLMEngine


class InterruptionListener:
    """
    A background thread that monitors incoming audio via VAD during TTS
    playback and raises an interrupt signal if user speech is detected.

    Parameters
    ----------
    vad : EnergyVAD
        VAD instance used to classify audio chunks.
    audio_source : callable
        A zero-argument callable that returns the next PCM chunk (bytes)
        from the microphone.  It should block briefly (e.g. read one
        chunk and return it).  Returning ``None`` signals the stream has
        ended.
    speaker : MockSpeaker
        Speaker to interrupt when speech is detected.
    llm : MockLLMEngine
        LLM engine whose stream will be cancelled on interruption.
    on_interrupt : callable, optional
        Optional callback invoked (with no arguments) when an interrupt
        is triggered.
    poll_interval : float
        Seconds to sleep between audio polls when no chunk is available
        (default 0.005 s).
    """

    def __init__(
        self,
        vad: EnergyVAD,
        audio_source: Callable[[], Optional[bytes]],
        speaker: MockSpeaker,
        llm: MockLLMEngine,
        on_interrupt: Optional[Callable[[], None]] = None,
        poll_interval: float = 0.005,
    ) -> None:
        self.vad = vad
        self.audio_source = audio_source
        self.speaker = speaker
        self.llm = llm
        self.on_interrupt = on_interrupt
        self.poll_interval = poll_interval

        self._stop_event = threading.Event()
        self.interrupt_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Chunks collected after the interruption (pre-speech context)
        self.captured_chunks: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def was_interrupted(self) -> bool:
        """True if an interruption was detected."""
        return self.interrupt_event.is_set()

    def start(self) -> None:
        """Start the background listener thread."""
        self._stop_event.clear()
        self.interrupt_event.clear()
        self.captured_chunks = []
        self._thread = threading.Thread(
            target=self._run,
            name="interruption-listener",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Thread target: poll audio source, check VAD, trigger interrupt."""
        while not self._stop_event.is_set():
            chunk = self.audio_source()
            if chunk is None:
                time.sleep(self.poll_interval)
                continue

            if self.vad.is_speech(chunk):
                # Interruption detected
                self.interrupt_event.set()
                self.captured_chunks.append(chunk)
                # Halt speaker
                self.speaker.interrupt()
                # Cancel LLM stream
                self.llm.cancel()
                # Invoke optional callback
                if self.on_interrupt is not None:
                    self.on_interrupt()
                # Keep collecting speech chunks until stopped
                while not self._stop_event.is_set():
                    chunk = self.audio_source()
                    if chunk is None:
                        time.sleep(self.poll_interval)
                        continue
                    if self.vad.is_speech(chunk):
                        self.captured_chunks.append(chunk)
                    else:
                        # First silent chunk after interruption speech ends
                        break
                return  # exit thread after handling interrupt
