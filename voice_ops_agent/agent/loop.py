"""Core agent loop with interruption handling and tool-calling support.

Phase flow (normal, no tool calls):
    IDLE -> LISTENING -> TRANSCRIBING -> LLM_STREAMING -> TTS_PLAYING -> IDLE

Phase flow (tool call detected in LLM output):
    ... -> LLM_STREAMING  [tool calls executed, context updated]
                        -> LLM_STREAMING  [resume with tool results]
                        -> TTS_PLAYING -> IDLE

Phase flow (interrupted during TTS):
    ... -> TTS_PLAYING -> INTERRUPTED -> LISTENING -> ...
"""

import threading
import time
from typing import Callable, Dict, Iterator, List, Optional

from voice_ops_agent.stt.engine import MockSTTEngine
from voice_ops_agent.llm.engine import MockLLMEngine
from voice_ops_agent.tts.engine import MockTTSEngine
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.agent.state import AgentState, StateMachine
from voice_ops_agent.agent.interruption import InterruptionListener
from voice_ops_agent.tools.parser import parse_tool_calls, strip_tool_calls, has_tool_call
from voice_ops_agent.tools.registry import ToolRegistry
from voice_ops_agent.agent.tool_executor import ToolExecutor
from voice_ops_agent.tools.base import ToolCallIntent  # re-export for callers


class AgentLoop:
    """
    Orchestrates a single-turn (or multi-turn via interruption) voice round-trip:

        audio_chunks  ->  STT  ->  LLM stream  ->  [tool calls]  ->  TTS  ->  speaker

    Tool-calling
    ------------
    When the LLM response contains ``TOOL_CALL:<name>:<json>`` directives:
    1. The directives are parsed from the accumulated stream output.
    2. Each tool is executed via the ``ToolRegistry``.
    3. Results are injected into the conversation context.
    4. The LLM is called again with the enriched context to produce the
       final spoken response.

    Interruption handling adds a parallel VAD listener during TTS playback.
    When the user speaks, playback and LLM streaming are cancelled and the
    agent re-enters the STT phase with the newly captured audio.

    Parameters
    ----------
    stt : MockSTTEngine
    llm : MockLLMEngine
    tts : MockTTSEngine
    speaker : MockSpeaker
    tool_registry : ToolRegistry, optional
        If provided, tool-calling is enabled.
    on_transcript : callable, optional
    on_token : callable, optional
    on_response_complete : callable, optional
    on_interrupt : callable, optional
    on_tool_call : callable, optional
        ``on_tool_call(intent)`` fired before each tool execution.
    on_tool_result : callable, optional
        ``on_tool_result(result)`` fired after each tool execution.
    on_state_change : callable, optional
    max_interruptions : int
    max_tool_rounds : int
        Maximum rounds of tool-call → LLM re-entry per turn (default 5).
        Prevents infinite tool-calling loops.
    """

    def __init__(
        self,
        stt: MockSTTEngine,
        llm: MockLLMEngine,
        tts: MockTTSEngine,
        speaker: MockSpeaker,
        tool_registry: Optional[ToolRegistry] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_response_complete: Optional[Callable[[str], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_tool_call: Optional[Callable] = None,
        on_tool_result: Optional[Callable] = None,
        on_state_change: Optional[Callable[[AgentState, AgentState], None]] = None,
        max_interruptions: int = 3,
        max_tool_rounds: int = 5,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.speaker = speaker
        self.tool_registry = tool_registry
        self.on_transcript = on_transcript
        self.on_token = on_token
        self.on_response_complete = on_response_complete
        self.on_interrupt = on_interrupt
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_state_change = on_state_change
        self.max_interruptions = max_interruptions
        self.max_tool_rounds = max_tool_rounds

        self.state_machine = StateMachine(AgentState.IDLE)
        self.timings: Dict[str, float] = {}
        self.interruption_count: int = 0

        # Tool executor (created lazily when registry is provided)
        self._tool_executor: Optional[ToolExecutor] = None
        if self.tool_registry is not None:
            self._tool_executor = ToolExecutor(
                registry=self.tool_registry,
                on_tool_call=self.on_tool_call,
                on_tool_result=self.on_tool_result,
            )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: AgentState) -> None:
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
        Execute one full voice round-trip (with optional interruption and
        tool-calling).

        Parameters
        ----------
        audio_chunks : list of bytes
        context : list of dict, optional
        interrupt_audio_source : callable, optional
        vad : EnergyVAD, optional

        Returns
        -------
        dict
            {
                "transcript"         : str,
                "llm_response"       : str,
                "timings"            : dict,
                "stt_result"         : dict,
                "interrupted"        : bool,
                "interruption_count" : int,
                "state_history"      : list,
                "tool_calls"         : list of ToolCallIntent,
                "tool_results"       : list of ToolResult,
            }
        """
        self.timings = {}
        self.interruption_count = 0
        self.state_machine.reset()

        if context is None:
            context = []

        t_start = time.perf_counter()

        final_transcript = ""
        final_llm_response = ""
        final_stt_result: Dict = {}
        interrupted = False
        all_tool_calls: List = []
        all_tool_results: List = []

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

            # Add user turn to context
            context.append({"role": "user", "content": transcript})

            # ---------------------------------------------------------- #
            # Phase 2: LLM_STREAMING (with tool-call loop)                #
            # ---------------------------------------------------------- #
            self._transition(AgentState.LLM_STREAMING)

            t_llm_start = time.perf_counter()
            first_token_time: Optional[float] = None

            llm_response = self._run_llm_with_tools(
                transcript=transcript,
                context=context,
                t_stt_end=t_stt_end,
                t_llm_start=t_llm_start,
                first_token_time_ref=[first_token_time],
                all_tool_calls=all_tool_calls,
                all_tool_results=all_tool_results,
            )

            t_llm_end = time.perf_counter()
            self.timings.setdefault("llm_total_latency", t_llm_end - t_llm_start)

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
                        break
            finally:
                if listener is not None:
                    listener.stop()

            t_tts_end = time.perf_counter()
            self.timings.setdefault("tts_total_latency", t_tts_end - t_tts_start)

            if listener is not None and listener.was_interrupted:
                interrupted = True
                self.interruption_count += 1
                self._transition(AgentState.INTERRUPTED)

                if self.interruption_count >= self.max_interruptions:
                    self._transition(AgentState.IDLE)
                    break

                current_audio = (
                    listener.captured_chunks if listener.captured_chunks else current_audio
                )
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
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
        }

    # ------------------------------------------------------------------
    # Internal: LLM streaming with tool-call loop
    # ------------------------------------------------------------------

    def _run_llm_with_tools(
        self,
        transcript: str,
        context: List[Dict],
        t_stt_end: float,
        t_llm_start: float,
        first_token_time_ref: List,
        all_tool_calls: List,
        all_tool_results: List,
    ) -> str:
        """
        Stream LLM output, detect tool calls, execute them, inject results,
        and re-invoke the LLM as needed.  Returns the final human-facing
        response (tool-call directives stripped).
        """
        tool_round = 0
        accumulated_response = ""

        # The prompt for the first LLM call is the transcript; for
        # subsequent (tool-result) calls we pass the enriched context.
        current_prompt = transcript

        while tool_round <= self.max_tool_rounds:
            tokens: List[str] = []
            first_token_time = first_token_time_ref[0]

            for token in self.llm.stream_response(current_prompt, context):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    first_token_time_ref[0] = first_token_time
                    self.timings.setdefault(
                        "stt_to_first_token", first_token_time - t_stt_end
                    )
                    self.timings.setdefault(
                        "llm_first_token_latency", first_token_time - t_llm_start
                    )

                tokens.append(token)

                # Fire on_token only for non-tool-call tokens (we don't
                # know yet whether this chunk has directives, so we
                # buffer and emit after checking)

            raw_response = "".join(tokens)

            # Detect tool calls in the full response
            if self._tool_executor is not None and has_tool_call(raw_response):
                intents = parse_tool_calls(raw_response)
                all_tool_calls.extend(intents)

                # Execute tools and inject results into context
                context = self._tool_executor.execute_all(intents, context)
                all_tool_results.extend(
                    entry["result"]
                    for entry in self._tool_executor.execution_history[
                        -len(intents):
                    ]
                )

                # Build a follow-up prompt summarising tool results
                summaries = " ".join(
                    entry["result"].text_summary
                    for entry in self._tool_executor.execution_history[-len(intents):]
                )
                current_prompt = (
                    f"The following tool results are available: {summaries}. "
                    "Please provide a helpful spoken response based on these results."
                )
                tool_round += 1
                continue  # re-enter LLM with tool results

            # No tool calls in this response – emit tokens via callback
            clean_response = strip_tool_calls(raw_response)
            if not clean_response:
                clean_response = raw_response.strip()

            for token in clean_response.split(" "):
                if self.on_token:
                    self.on_token(token + " ")

            accumulated_response = clean_response
            break  # done

        return accumulated_response
