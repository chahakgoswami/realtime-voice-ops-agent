# Real-Time Voice Ops Agent

Sub-second voice agent with tool calling, interruption handling, and escalation paths.

**Domain:** Agentic AI
**Language:** python
**Demonstrates:** You can build multimodal agents that feel human.

## 7-day build plan

- [ ] Day 1: Scaffold the project with a Python package structure, mock audio I/O layer (microphone input and speaker output via simulated streams), a simple VAD (voice activity detection) stub using energy thresholds, and a CLI entry point that records a spoken utterance and echoes it back as text using a mocked STT engine.
- [ ] Day 2: Integrate a mocked LLM backend (simulated streaming responses with configurable latency) and build the core agent loop that pipes STT text into the LLM, streams token output, and converts responses to speech via a mocked TTS engine, achieving a full single-turn voice round-trip runnable from the CLI.
- [ ] Day 3: Implement interruption handling by adding a parallel listener thread that monitors VAD during agent speech playback, immediately halts TTS output on user interruption, cancels the in-flight LLM stream, and re-enters the STT phase, with unit tests verifying the interrupt signal propagation and state machine transitions.
- [ ] Day 4: Build the tool-calling subsystem with a registry of mock tools (weather lookup, calendar check, order status, knowledge base search), parse LLM tool-call intents from streamed output, execute the appropriate mock tool, inject results back into context, and resume response generation, with integration tests covering multi-tool conversations.
- [ ] Day 5: Add the escalation path system with configurable rules (confidence threshold, repeated failures, explicit user request) that trigger handoff to a simulated human agent, implement a human-in-the-loop confirmation prompt for any destructive tool actions (order cancellation, data deletion stubs), and log all escalation events to a local JSON audit trail.
- [ ] Day 6: Introduce a conversation session manager that maintains turn history, speaker diarization stubs, context windowing to stay within token limits, and a mock telephony adapter (SIP/PSTN simulation via socket-based message passing) so the agent can be tested as if handling a real phone call end-to-end with configurable scenarios.
- [ ] Day 7: Add a pytest suite covering the full happy path, interruption, tool-call, and escalation scenarios with deterministic mock fixtures; implement a latency profiler that measures and asserts sub-second STT-to-first-token and TTS-start times; and wire everything into a single `python -m voice_ops_agent` command with environment-variable configuration and a sample .env.example file.

_A comprehensive README with an architecture diagram is generated on Day 7._
