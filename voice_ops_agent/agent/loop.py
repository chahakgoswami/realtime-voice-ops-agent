"""Core single-turn agent loop: STT → LLM stream → TTS → speaker."""

import time
from typing import List, Optional, Dict, Callable

from voice_ops_agent.stt.engine import MockSTTEngine
from voice_ops_agent.llm.engine import MockLLMEngine
from voice_ops_agent.tts.engine import MockTTSEngine
from voice_ops_agent.audio.output import MockSpeaker


class AgentLoop:
    """
    Orchestrates a single-turn voice round-trip:

        audio_chunks  →  STT  →  LLM stream  →  TTS stream  →  speaker

    All timings are recorded so callers can assert latency budgets.

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
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.speaker = speaker
        self.on_transcript = on_transcript
        self.on_token = on_token
        self.on_response_complete = on_response_complete

        # Latency bookkeeping (populated after run())
        self.timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        audio_chunks: List[bytes],
        context: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Execute one full voice round-trip.

        Parameters
        ----------
        audio_chunks : list of bytes
            Raw PCM audio from the microphone / VAD.
        context : list of dict, optional
            Conversation history passed to the LLM.

        Returns
        -------
        dict
            {
                "transcript"   : str,
                "llm_response" : str,
                "timings"      : dict of float (seconds),
                "stt_result"   : dict,
            }
        """
        self.timings = {}
        t_start = time.perf_counter()

        # ---------------------------------------------------------------- #
        # Phase 1: STT                                                      #
        # ---------------------------------------------------------------- #
        t_stt_start = time.perf_counter()
        stt_result = self.stt.transcribe(audio_chunks)
        t_stt_end = time.perf_counter()

        transcript = stt_result["text"]
        self.timings["stt_latency"] = t_stt_end - t_stt_start

        if self.on_transcript:
            self.on_transcript(transcript)

        # ---------------------------------------------------------------- #
        # Phase 2: LLM streaming                                            #
        # ---------------------------------------------------------------- #
        t_llm_start = time.perf_counter()
        first_token_time: Optional[float] = None
        tokens: List[str] = []

        for token in self.llm.stream_response(transcript, context):
            if first_token_time is None:
                first_token_time = time.perf_counter()
                self.timings["stt_to_first_token"] = first_token_time - t_stt_end
                self.timings["llm_first_token_latency"] = first_token_time - t_llm_start

            tokens.append(token)

            if self.on_token:
                self.on_token(token)

        t_llm_end = time.perf_counter()
        self.timings["llm_total_latency"] = t_llm_end - t_llm_start

        llm_response = "".join(tokens)

        if self.on_response_complete:
            self.on_response_complete(llm_response)

        # ---------------------------------------------------------------- #
        # Phase 3: TTS + playback                                           #
        # ---------------------------------------------------------------- #
        self.speaker.reset()
        t_tts_start = time.perf_counter()
        first_audio_time: Optional[float] = None

        for chunk in self.tts.stream_synthesise(llm_response):
            if first_audio_time is None:
                first_audio_time = time.perf_counter()
                self.timings["tts_first_chunk_latency"] = first_audio_time - t_tts_start
                self.timings["llm_to_tts_first_chunk"] = first_audio_time - t_llm_end

            played = self.speaker.play_chunk(chunk)
            if not played:
                # Speaker was interrupted; abort TTS streaming
                break

        t_tts_end = time.perf_counter()
        self.timings["tts_total_latency"] = t_tts_end - t_tts_start
        self.timings["total_latency"] = t_tts_end - t_start

        return {
            "transcript": transcript,
            "llm_response": llm_response,
            "timings": self.timings,
            "stt_result": stt_result,
        }
