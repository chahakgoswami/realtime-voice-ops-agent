# Real-Time Voice Ops Agent

Sub-second voice agent with tool calling, interruption handling, and escalation paths.

**Domain:** Agentic AI  
**Language:** Python  
**Demonstrates:** You can build multimodal agents that feel human.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  python -m voice_ops_agent  (CLI entry point / __main__.py)         │
│                                                                     │
│  MockMicrophone → EnergyVAD → MockSTTEngine                        │
│        │                           │                               │
│        │                     transcript                             │
│        │                           ≐                               │
│  InterruptionListener       MockLLMEngine (streaming)               │
│  (parallel VAD thread)            │                                │
│        │                     tokens/TOOL_CALL directives            │
│        │                           │                               │
│        │                    ToolExecutor → ToolRegistry             │
│        │                     (weather, calendar, order, kb)         │
│        │                           │                               │
│        │                     LLM re-invoke (with tool results)      │
│        │                           │                               │
│        │                     MockTTSEngine → MockSpeaker            │
│        │                           │                               │
│  interrupt signal ───────────────▼                               │
│                          StateMachine (IDLE/LISTENING/TRANSCRIBING/ │
│                           LLM_STREAMING/TTS_PLAYING/INTERRUPTED)    │
│                                                                     │
│  EscalationManager (confidence / failure / explicit request rules)  │
│  HumanInTheLoopConfirmation (destructive tool gate)                 │
│  AuditLogger (JSON file)                                            │
│                                                                     │
│  SessionManager → ContextWindow + DiarizationStub                  │
│  MockTelephonyAdapter (SIP/PSTN socket simulation)                  │
│  ScenarioPlayer (configurable call scripts)                         │
│  LatencyProfiler (sub-second SLA assertions)                        │
└───────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. (Optional) configure via environment
cp .env.example .env
# edit .env as needed

# 3. Run
python -m voice_ops_agent
```

All external integrations are **mocked** — no real microphone, speaker, LLM API key, or telephony hardware required.

---

## Package Structure

```
voice_ops_agent/
├── __init__.py          # package version
├── __main__.py          # CLI entry point (python -m voice_ops_agent)
├── profiler.py          # LatencyProfiler (sub-second SLA)
├── audio/
│   ├── input.py           # MockMicrophone
│   └── output.py          # MockSpeaker
├── vad/
│   └── detector.py        # EnergyVAD
├── stt/
│   └── engine.py          # MockSTTEngine
├── llm/
│   └── engine.py          # MockLLMEngine (streaming)
├── tts/
│   └── engine.py          # MockTTSEngine
├── agent/
│   ├── loop.py            # AgentLoop (core orchestrator)
│   ├── state.py           # StateMachine + AgentState
│   ├── interruption.py    # InterruptionListener
│   └── tool_executor.py   # ToolExecutor
├── tools/
│   ├── base.py            # BaseTool, ToolResult
│   ├── registry.py        # ToolRegistry
│   ├── parser.py          # TOOL_CALL directive parser
│   ├── mock_tools.py      # weather, calendar, order, kb
│   └── destructive_tools.py # order_cancel, data_delete (HITL gated)
├── escalation/
│   ├── rules.py           # ConfidenceThreshold, RepeatedFailure, ExplicitRequest
│   ├── manager.py         # EscalationManager
│   ├── audit.py           # AuditLogger (JSON file)
│   └── hitl.py            # HumanInTheLoopConfirmation
├── session/
│   ├── manager.py         # SessionManager, Turn, Speaker
│   ├── context_window.py  # ContextWindow (token budgeting)
│   ├── diarization.py     # DiarizationStub
│   └── call_handler.py    # CallHandler (end-to-end)
└── telephony/
    ├── adapter.py         # MockTelephonyAdapter (socket-based)
    └── scenarios.py       # ScenarioPlayer + built-in scenarios
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VAD_THRESHOLD` | `300` | RMS energy threshold for speech detection |
| `STT_TRANSCRIPT` | _(random)_ | Fixed STT transcript (empty = random) |
| `STT_LATENCY` | `0.05` | Simulated STT latency (seconds) |
| `LLM_TOKEN_LATENCY` | `0.02` | Delay between streamed LLM tokens (seconds) |
| `LLM_FIRST_LATENCY` | `0.05` | LLM time-to-first-token (seconds) |
| `LLM_RESPONSE` | _(intent-matched)_ | Fixed LLM response (empty = auto) |
| `TTS_LATENCY` | `0.03` | Simulated TTS first-chunk latency (seconds) |

Copy `.env.example` to `.env` and adjust as needed.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 7-Day Build Plan

- [x] Day 1: Audio I/O, VAD, STT mock, CLI scaffold
- [x] Day 2: LLM streaming backend, TTS engine, full agent loop
- [x] Day 3: Interruption handling, state machine
- [x] Day 4: Tool-calling subsystem (weather, calendar, order, KB)
- [x] Day 5: Escalation paths, HITL confirmation, audit logging
- [x] Day 6: Session manager, diarization, context windowing, telephony adapter
- [x] Day 7: Full test suite, latency profiler, `python -m voice_ops_agent`, `.env` config
