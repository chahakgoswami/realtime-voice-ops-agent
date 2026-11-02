"""CLI entry point: record a spoken utterance and echo it back as text.

Usage
-----
    python -m voice_ops_agent

Environment variables (optional)
---------------------------------
    VAD_THRESHOLD   – RMS energy threshold for VAD (default: 300)
    STT_TRANSCRIPT  – fixed transcript returned by mock STT (default: random)
    STT_LATENCY     – simulated STT latency in seconds (default: 0.05)
"""

import os
import sys
import time

from voice_ops_agent.audio.input import MockMicrophone
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.vad.detector import EnergyVAD
from voice_ops_agent.stt.engine import MockSTTEngine


def main() -> None:
    # ------------------------------------------------------------------ #
    # Configuration from environment variables                             #
    # ------------------------------------------------------------------ #
    vad_threshold = float(os.environ.get("VAD_THRESHOLD", "300"))
    stt_transcript = os.environ.get("STT_TRANSCRIPT", None)  # None → random
    stt_latency = float(os.environ.get("STT_LATENCY", "0.05"))

    # ------------------------------------------------------------------ #
    # Initialise components                                                #
    # ------------------------------------------------------------------ #
    mic = MockMicrophone(
        silence_chunks=4,
        speech_chunks=20,
        silence_after=6,
        frequency=440.0,
        amplitude=0.5,
        realtime=False,   # fast simulation for CLI demo
    )
    speaker = MockSpeaker(realtime=False)
    vad = EnergyVAD(threshold=vad_threshold)
    stt = MockSTTEngine(
        fixed_transcript=stt_transcript,
        latency=stt_latency,
    )

    # ------------------------------------------------------------------ #
    # Step 1: Capture audio                                                #
    # ------------------------------------------------------------------ #
    print("[voice-ops] Listening… (simulated microphone active)")
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
    # Step 2: Transcribe                                                   #
    # ------------------------------------------------------------------ #
    print("[voice-ops] Transcribing…")
    t1 = time.perf_counter()

    result = stt.transcribe(audio_chunks)

    stt_duration = time.perf_counter() - t1
    transcript = result["text"]
    confidence = result["confidence"]
    audio_dur = result["duration_s"]

    print(f"[voice-ops] STT completed in {stt_duration:.3f}s")
    print(f"[voice-ops] Audio duration : {audio_dur:.3f}s")
    print(f"[voice-ops] Confidence     : {confidence:.2f}")
    print()
    print(f"  >>> Transcript: \"{transcript}\"")
    print()

    # ------------------------------------------------------------------ #
    # Step 3: Echo back (mock speaker – just prints confirmation here)     #
    # ------------------------------------------------------------------ #
    # In a real system TTS audio would be generated and sent to the
    # speaker.  For Day 1 we simply acknowledge playback via the mock.
    echo_audio = transcript.encode("utf-8")  # placeholder bytes
    speaker.play(echo_audio)
    print(
        f"[voice-ops] Speaker played {speaker.total_bytes_played} bytes "
        "(echo simulation complete)."
    )


if __name__ == "__main__":
    main()
