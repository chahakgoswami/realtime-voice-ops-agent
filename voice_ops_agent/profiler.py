"""Latency profiler for the voice ops agent.

Measures and reports key latency metrics for each agent run:
- stt_to_first_token: time from end of STT to first LLM token (must be < 1 s)
- tts_first_chunk_latency: time from TTS start to first audio chunk (must be < 1 s)
- total_latency: wall-clock time for the full round-trip

Usage example::

    profiler = LatencyProfiler(loop)
    result = profiler.run_and_profile(audio_chunks)
    profiler.assert_sub_second(result)
    print(profiler.report(result))
"""

from voice_ops_agent.profiler_impl import LatencyProfiler

__all__ = ["LatencyProfiler"]
