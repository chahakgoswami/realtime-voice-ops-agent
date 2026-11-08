"""Latency profiler implementation (importable as voice_ops_agent.profiler).

See voice_ops_agent/profiler.py for the public API documentation.
"""

from typing import Any, Dict, List, Optional

from voice_ops_agent.agent.loop import AgentLoop


class LatencyProfiler:
    """
    Wraps an :class:`AgentLoop` to collect latency measurements across
    one or more runs and assert sub-second SLA targets.

    Parameters
    ----------
    loop : AgentLoop
        The agent loop to profile.
    """

    def __init__(self, loop: AgentLoop) -> None:
        self.loop = loop
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_and_profile(
        self,
        audio_chunks: List[bytes],
        context: Optional[List[Dict]] = None,
        interrupt_audio_source=None,
        vad=None,
    ) -> Dict[str, Any]:
        """
        Run one full agent turn and return the result dict (which includes
        a ``timings`` sub-dict).
        """
        result = self.loop.run(
            audio_chunks=audio_chunks,
            context=context,
            interrupt_audio_source=interrupt_audio_source,
            vad=vad,
        )
        self._results.append(result)
        return result

    def assert_sub_second(
        self,
        result: Dict[str, Any],
        stt_to_first_token_budget: float = 1.0,
        tts_first_chunk_budget: float = 1.0,
    ) -> None:
        """
        Assert that the key latency metrics in *result* are within the
        specified budgets.

        Raises
        ------
        AssertionError
            If any metric exceeds its budget.
        """
        timings = result.get("timings", {})

        stt_to_first = timings.get("stt_to_first_token", 0.0)
        assert stt_to_first < stt_to_first_token_budget, (
            f"stt_to_first_token={stt_to_first:.3f}s exceeds budget "
            f"{stt_to_first_token_budget:.3f}s"
        )

        tts_first = timings.get("tts_first_chunk_latency", 0.0)
        assert tts_first < tts_first_chunk_budget, (
            f"tts_first_chunk_latency={tts_first:.3f}s exceeds budget "
            f"{tts_first_chunk_budget:.3f}s"
        )

    def report(self, result: Dict[str, Any]) -> str:
        """Return a human-readable latency report string for *result*."""
        timings = result.get("timings", {})
        lines = [
            "── Latency Report ─────────────────────────────────────────────",
            f"  STT latency             : {timings.get('stt_latency', 0)*1000:.1f} ms",
            f"  LLM first-token         : {timings.get('llm_first_token_latency', 0)*1000:.1f} ms",
            f"  STT \u2192 first token       : {timings.get('stt_to_first_token', 0)*1000:.1f} ms",
            f"  TTS first-chunk         : {timings.get('tts_first_chunk_latency', 0)*1000:.1f} ms",
            f"  LLM \u2192 TTS first-chunk   : {timings.get('llm_to_tts_first_chunk', 0)*1000:.1f} ms",
            f"  Total round-trip        : {timings.get('total_latency', 0)*1000:.1f} ms",
            "────────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    @property
    def run_count(self) -> int:
        """Number of profiling runs completed."""
        return len(self._results)

    def average_latencies(self) -> Dict[str, float]:
        """
        Return the average of each timing metric across all runs.
        """
        if not self._results:
            return {}

        all_keys: set = set()
        for r in self._results:
            all_keys.update(r.get("timings", {}).keys())

        averages: Dict[str, float] = {}
        for key in all_keys:
            values = [
                r["timings"][key]
                for r in self._results
                if key in r.get("timings", {})
            ]
            if values:
                averages[key] = sum(values) / len(values)

        return averages
