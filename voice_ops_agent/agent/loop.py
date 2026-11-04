"""Core agent loop with interruption handling support.

Phase flow (normal):
    IDLE -> LISTENING -> TRANSCRIBING -> LLM_STREAMING -> TTS_PLAYING -> IDLE

Phase flow (interrupted during TTS):
    ... -> TTS_PLAYING -> INTERRUPTED -> LISTENING -> ...
"""

import threading
import time
from typing import List, Optional, Dict, Callable, Iterator

from voice_ops_agent.stt.engine import MockSTTEngine
from voice_ops_agent.llm.engine import MockLLMEngine
from voice_ops_agent.tts.engine import MockTTSEngine
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.agent.state import AgentState, StateMachine
from voice_ops_agent.agent.interruption import InterruptionListener


class AgentLoop:
    """
    Orchestrates a single-turn (or multi-turn via interruption) voice round-trip:

        audio_chunks  ->  STT  ->  LLM stream  ->  TTS stream  ->  speaker

    Interruption handling adds a parallel VAD listener during TTS playback.
    When the user speaks, playback and LLM streaming are cancelled and the
    agent re-enters the STT phase with the newly captured audio.

    Parameters
    ----------
    stt : MockSTTEngine
    llm : MockLLMEngine
    tts : MockTTSEngine
    speaker : MockSpeaker
    on_transcript : callable, optional
        Called with the STT transcript string as soon as it is ready.
    on_token : callable, optional
        Called with each LLM token string as it streams in.
    on_response_complete : callable, optional
        Called with the full assembled LLM response string.
    on_interrupt : callable, optional
        Called (no arguments) when a user interruption is detected.
    on_state_change : callable, optional
        Called with (old_state, new_state) on every state transition.
    max_interruptions : int
        Maximum number of interruptions allowed per ``run`` call before
        giving up (default 3).  Guards against infinite re-entry loops.
    """

    def __init__(
        self,
        stt: MockSTTEngine,
        llm: MockLLMEngine,
        tts: MockTTSEngine,
        speaker: MockSpeaker,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_response_complete: Optional[Callable[[str], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_state_change: Optional[Callable[[AgentState, AgentState], None]] = None,
        max_interruptions: int = 3,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.speaker = speaker
        self.on_transcript = on_transcript
        self.on_token = on_token
        self.on_response_complete = on_response_complete
        self.on_interrupt = on_interrupt
        self.on_state_change = on_state_change
        self.max_interruptions = max_interruptions

        self.state_machine = StateMachine(AgentState.IDLE)
        # Latency bookkeeping (populated after run())
        self.timings: Dict[str, float] = {}
        # Number of interruptions that occurred in the last run()
        self.interruption_count: int = 0

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: AgentState) -> None:
        """Perform a state transition and invoke on_state_change callback."""
        old_state = self.state_machine.state
        self.state_machine.transition(new_state)
        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        audio_chunks: List[bytes],
        context: Optional[List[Dict]] = None,
        interrupt_audio_source: Optional[Callable[[], Optional[bytes]]] = None,
        vad=None,
    ) -> Dict:
        """
        Execute one full voice round-trip (with optional interruption handling).

        Parameters
        ----------
        audio_chunks : list of bytes
            Raw PCM audio from the microphone / VAD.
        context : list of dict, optional
            Conversation history passed to the LLM.
        interrupt_audio_source : callable, optional
            If provided, interruption handling is enabled.  Should be a
            zero-argument callable returning PCM bytes (or None when idle).
        vad : EnergyVAD, optional
            VAD instance used by the interruption listener.  Required when
            *interrupt_audio_source* is provided.

        Returns
        -------
        dict
            {
                "transcript"         : str,
                "llm_response"       : str,
                "timings"            : dict of float (seconds),
                "stt_result"         : dict,
                "interrupted"        : bool,
                "interruption_count" : int,
                "state_history"      : list of AgentState,
            }
        """
        self.timings = {}
        self.interruption_count = 0
        self.state_machine.reset()

        t_start = time.perf_counter()

        # Will hold the final values after (possible) re-entry
        final_transcript = ""
        final_llm_response = ""
        final_stt_result: Dict = {}
        interrupted = False

        current_audio = audio_chunks

        while True:
            # ---------------------------------------------------------- #
            # Phase 1: LISTENING -> TRANSCRIBING (STT)                    #
            # ---------------------------------------------------------- #
            self._transition(AgentState.LISTENING)
            self._transition(AgentState.TRANSCRIBING)

            t_stt_start = time.perf_counter()
            stt_result = self.stt.transcribe(current_audio)
            t_stt_end = time.perf_counter()

            self.timings.setdefault("stt_latency", t_stt_end - t_stt_start)

            transcript = stt_result["text"]
            final_transcript = transcript
            final_stt_result = stt_result

            if self.on_transcript:
                self.on_transcript(transcript)

            # ---------------------------------------------------------- #
            # Phase 2: LLM_STREAMING                                      #
            # ---------------------------------------------------------- #
            self._transition(AgentState.LLM_STREAMING)

            t_llm_start = time.perf_counter()
            first_token_time: Optional[float] = None
            tokens: List[str] = []

            for token in self.llm.stream_response(transcript, context):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    self.timings.setdefault(
                        "stt_to_first_token", first_token_time - t_stt_end
                    )
                    self.timings.setdefault(
                        "llm_first_token_latency", first_token_time - t_llm_start
                    )

                tokens.append(token)

                if self.on_token:
                    self.on_token(token)

            t_llm_end = time.perf_counter()
            self.timings.setdefault(
                "llm_total_latency", t_llm_end - t_llm_start
            )

            llm_response = "".join(tokens)
            final_llm_response = llm_response

            if self.on_response_complete:
                self.on_response_complete(llm_response)

            # ---------------------------------------------------------- #
            # Phase 3: TTS_PLAYING + interruption listener                 #
            # ---------------------------------------------------------- #
            self._transition(AgentState.TTS_PLAYING)
            self.speaker.reset()

            t_tts_start = time.perf_counter()
            first_audio_time: Optional[float] = None

            # Set up interruption listener if audio source provided
            listener: Optional[InterruptionListener] = None
            if interrupt_audio_source is not None and vad is not None:
                listener = InterruptionListener(
                    vad=vad,
                    audio_source=interrupt_audio_source,
                    speaker=self.speaker,
                    llm=self.llm,
                    on_interrupt=self.on_interrupt,
                )
                listener.start()

            try:
                for chunk in self.tts.stream_synthesise(llm_response):
                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()
                        self.timings.setdefault(
                            "tts_first_chunk_latency",
                            first_audio_time - t_tts_start,
                        )
                        self.timings.setdefault(
                            "llm_to_tts_first_chunk",
                            first_audio_time - t_llm_end,
                        )

                    played = self.speaker.play_chunk(chunk)
                    if not played:
                        # Speaker was interrupted; abort TTS streaming
                        break
            finally:
                if listener is not None:
                    listener.stop()

            t_tts_end = time.perf_counter()
            self.timings.setdefault(
                "tts_total_latency", t_tts_end - t_tts_start
            )

            # Check if an interruption occurred
            if listener is not None and listener.was_interrupted:
                interrupted = True
                self.interruption_count += 1
                self._transition(AgentState.INTERRUPTED)

                if self.on_interrupt is None and self.on_state_change:
                    # on_interrupt already called by listener; nothing extra needed
                    pass

                if self.interruption_count >= self.max_interruptions:
                    # Give up re-trying
                    self._transition(AgentState.IDLE)
                    break

                # Re-enter with captured audio
                current_audio = listener.captured_chunks if listener.captured_chunks else current_audio
                # State machine: INTERRUPTED -> LISTENING (handled at loop top)
                continue

            # Normal completion
            self._transition(AgentState.IDLE)
            break

        self.timings["total_latency"] = time.perf_counter() - t_start

        return {
            "transcript": final_transcript,
            "llm_response": final_llm_response,
            "timings": self.timings,
            "stt_result": final_stt_result,
            "interrupted": interrupted,
            "interruption_count": self.interruption_count,
            "state_history": self.state_machine.history,
        }
