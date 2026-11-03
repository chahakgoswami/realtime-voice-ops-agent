"""CLI entry point: full single-turn voice round-trip.

Usage
-----
    python -m voice_ops_agent

Environment variables (optional)
---------------------------------
    VAD_THRESHOLD       – RMS energy threshold for VAD (default: 300)
    STT_TRANSCRIPT      – fixed transcript returned by mock STT (default: random)
    STT_LATENCY         – simulated STT latency in seconds (default: 0.05)
    LLM_TOKEN_LATENCY   – delay between streamed LLM tokens in seconds (default: 0.02)
    LLM_FIRST_LATENCY   – time-to-first-token latency in seconds (default: 0.05)
    LLM_RESPONSE        – fixed LLM response (default: intent-matched canned reply)
    TTS_LATENCY         – simulated TTS first-chunk latency in seconds (default: 0.03)
"""

import os
import sys
import time

from voice_ops_agent.audio.input import MockMicrophone
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.vad.detector import EnergyVAD
from voice_ops_agent.stt.engine import MockSTTEngine
from voice_ops_agent.llm.engine import MockLLMEngine
from voice_ops_agent.tts.engine import MockTTSEngine
from voice_ops_agent.agent.loop import AgentLoop


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> None:
    # ------------------------------------------------------------------ #
    # Configuration from environment variables                             #
    # ------------------------------------------------------------------ #
    vad_threshold     = float(_env("VAD_THRESHOLD",     "300"))
    stt_transcript    = os.environ.get("STT_TRANSCRIPT") or None
    stt_latency       = float(_env("STT_LATENCY",       "0.05"))
    llm_token_latency = float(_env("LLM_TOKEN_LATENCY", "0.02"))
    llm_first_latency = float(_env("LLM_FIRST_LATENCY", "0.05"))
    llm_response      = os.environ.get("LLM_RESPONSE") or None
    tts_latency       = float(_env("TTS_LATENCY",       "0.03"))

    # ------------------------------------------------------------------ #
    # Initialise components                                                #
    # ------------------------------------------------------------------ #
    mic = MockMicrophone(
        silence_chunks=4,
        speech_chunks=20,
        silence_after=6,
        frequency=440.0,
        amplitude=0.5,
        realtime=False,
    )
    speaker = MockSpeaker(realtime=False)
    vad     = EnergyVAD(threshold=vad_threshold)
    stt     = MockSTTEngine(fixed_transcript=stt_transcript, latency=stt_latency)
    llm     = MockLLMEngine(
        token_latency=llm_token_latency,
        first_token_latency=llm_first_latency,
        fixed_response=llm_response,
    )
    tts     = MockTTSEngine(latency=tts_latency)

    # ------------------------------------------------------------------ #
    # Step 1: Capture audio                                                #
    # ------------------------------------------------------------------ #
    print("[voice-ops] Listening\u2026 (simulated microphone active)")
    t0 = time.perf_counter()
    audio_chunks = mic.record_until_silence(vad)
    recording_duration = time.perf_counter() - t0

    total_bytes = sum(len(c) for c in audio_chunks)
    print(
        f"[voice-ops] Captured {len(audio_chunks)} chunks "
        f"({total_bytes} bytes) in {recording_duration:.3f}s"
    )

    if not audio_chunks:
        print("[voice-ops] No speech detected. Exiting.")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Step 2–4: STT → LLM → TTS via AgentLoop                             #
    # ------------------------------------------------------------------ #
    def on_transcript(text: str) -> None:
        print(f"\n[voice-ops] STT transcript : \"{text}\"")

    collected_tokens: list = []

    def on_token(token: str) -> None:
        collected_tokens.append(token)
        print(token, end="", flush=True)

    def on_response_complete(response: str) -> None:
        print()  # newline after streamed tokens
        print(f"[voice-ops] LLM response complete ({len(response.split())} words)")

    loop = AgentLoop(
        stt=stt,
        llm=llm,
        tts=tts,
        speaker=speaker,
        on_transcript=on_transcript,
        on_token=on_token,
        on_response_complete=on_response_complete,
    )

    print("[voice-ops] Starting agent loop…")
    print("[voice-ops] LLM response: ", end="", flush=True)

    result = loop.run(audio_chunks)

    # ------------------------------------------------------------------ #
    # Step 5: Report timings                                               #
    # ------------------------------------------------------------------ #
    timings = result["timings"]
    print()
    print("[voice-ops] ── Latency report ────────────────────────────────")
    print(f"  STT latency            : {timings.get('stt_latency', 0)*1000:.1f} ms")
    print(f"  LLM first-token        : {timings.get('llm_first_token_latency', 0)*1000:.1f} ms")
    print(f"  STT → first token      : {timings.get('stt_to_first_token', 0)*1000:.1f} ms")
    print(f"  TTS first-chunk        : {timings.get('tts_first_chunk_latency', 0)*1000:.1f} ms")
    print(f"  Total round-trip       : {timings.get('total_latency', 0)*1000:.1f} ms")
    print(f"  Speaker bytes played   : {speaker.total_bytes_played}")
    print("[voice-ops] ─────────────────────────────────────────────────")
    print("[voice-ops] Round-trip complete.")


if __name__ == "__main__":
    main()
