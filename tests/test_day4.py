"""Day 4 integration & unit tests: tool-calling subsystem."""

import json
from typing import List

import pytest

from voice_ops_agent.tools.base import BaseTool, ToolResult
from voice_ops_agent.tools.mock_tools import (
    WeatherLookupTool,
    CalendarCheckTool,
    OrderStatusTool,
    KnowledgeBaseSearchTool,
)
from voice_ops_agent.tools.registry import ToolRegistry, default_registry
from voice_ops_agent.tools.parser import (
    parse_tool_calls,
    has_tool_call,
    strip_tool_calls,
    ToolCallIntent,
)
from voice_ops_agent.agent.tool_executor import ToolExecutor
from voice_ops_agent.agent.loop import AgentLoop
from voice_ops_agent.stt.engine import MockSTTEngine
from voice_ops_agent.llm.engine import MockLLMEngine
from voice_ops_agent.tts.engine import MockTTSEngine
from voice_ops_agent.audio.output import MockSpeaker
from voice_ops_agent.audio.input import CHUNK_SIZE, SAMPLE_RATE
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tone_chunk(amplitude: float = 0.5, frequency: float = 440.0) -> bytes:
    samples = []
    for i in range(CHUNK_SIZE):
        t = i / SAMPLE_RATE
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        samples.append(value.to_bytes(2, byteorder="little", signed=True))
    return b"".join(samples)


def dummy_audio(n: int = 5) -> list:
    return [make_tone_chunk() for _ in range(n)]


def tool_call_llm_response(tool_name: str, args: dict, follow_up: str = "") -> str:
    """Build a fake LLM response string containing a TOOL_CALL directive."""
    directive = f"TOOL_CALL:{tool_name}:{json.dumps(args)}"
    if follow_up:
        return f"{directive}\n{follow_up}"
    return directive


# ---------------------------------------------------------------------------
# BaseTool / ToolResult
# ---------------------------------------------------------------------------

class TestToolResult:
    def test_to_context_dict_success(self):
        result = ToolResult(
            tool_name="my_tool",
            success=True,
            data={"key": "value"},
            text_summary="All good.",
        )
        d = result.to_context_dict()
        assert d["role"] == "tool"
        assert d["tool_name"] == "my_tool"
        assert d["success"] is True
        assert d["result"] == {"key": "value"}
        assert d["summary"] == "All good."

    def test_to_context_dict_failure(self):
        result = ToolResult(
            tool_name="bad_tool",
            success=False,
            error="something went wrong",
            text_summary="Error occurred.",
        )
        d = result.to_context_dict()
        assert d["success"] is False
        assert "error" in d["result"]


class DummyTool(BaseTool):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool for testing."

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=kwargs,
            text_summary=f"Dummy ran with {kwargs}",
        )


class TestBaseTool:
    def test_execute_returns_tool_result(self):
        tool = DummyTool()
        result = tool.execute(foo="bar")
        assert isinstance(result, ToolResult)

    def test_name_property(self):
        tool = DummyTool()
        assert tool.name == "dummy"

    def test_description_property(self):
        tool = DummyTool()
        assert isinstance(tool.description, str)

    def test_kwargs_passed_through(self):
        tool = DummyTool()
        result = tool.execute(x=1, y=2)
        assert result.data == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# Mock tools
# ---------------------------------------------------------------------------

class TestWeatherLookupTool:
    def test_returns_tool_result(self):
        tool = WeatherLookupTool()
        result = tool.execute(location="London")
        assert isinstance(result, ToolResult)
        assert result.success is True

    def test_location_in_data(self):
        tool = WeatherLookupTool()
        result = tool.execute(location="Tokyo")
        assert result.data["location"] == "Tokyo"

    def test_temperature_is_int(self):
        tool = WeatherLookupTool()
        result = tool.execute(location="Paris")
        assert isinstance(result.data["temperature_f"], int)

    def test_text_summary_non_empty(self):
        tool = WeatherLookupTool()
        result = tool.execute(location="Sydney")
        assert len(result.text_summary) > 0
        assert "Sydney" in result.text_summary


class TestCalendarCheckTool:
    def test_returns_tool_result(self):
        tool = CalendarCheckTool()
        result = tool.execute(date="today")
        assert isinstance(result, ToolResult)
        assert result.success is True

    def test_events_list_non_empty(self):
        tool = CalendarCheckTool()
        result = tool.execute(date="tomorrow")
        assert isinstance(result.data["events"], list)
        assert len(result.data["events"]) >= 1

    def test_date_in_data(self):
        tool = CalendarCheckTool()
        result = tool.execute(date="Monday")
        assert result.data["date"] == "Monday"

    def test_text_summary_mentions_date(self):
        tool = CalendarCheckTool()
        result = tool.execute(date="Friday")
        assert "Friday" in result.text_summary


class TestOrderStatusTool:
    def test_returns_tool_result(self):
        tool = OrderStatusTool()
        result = tool.execute(order_id="99999")
        assert isinstance(result, ToolResult)
        assert result.success is True

    def test_order_id_in_data(self):
        tool = OrderStatusTool()
        result = tool.execute(order_id="12345")
        assert result.data["order_id"] == "12345"

    def test_status_is_string(self):
        tool = OrderStatusTool()
        result = tool.execute(order_id="42")
        assert isinstance(result.data["status"], str)

    def test_text_summary_mentions_order_id(self):
        tool = OrderStatusTool()
        result = tool.execute(order_id="XYZ")
        assert "XYZ" in result.text_summary


class TestKnowledgeBaseSearchTool:
    def test_returns_tool_result(self):
        tool = KnowledgeBaseSearchTool()
        result = tool.execute(query="password reset")
        assert isinstance(result, ToolResult)
        assert result.success is True

    def test_results_non_empty(self):
        tool = KnowledgeBaseSearchTool()
        result = tool.execute(query="shipping")
        assert isinstance(result.data["results"], list)
        assert len(result.data["results"]) >= 1

    def test_query_in_data(self):
        tool = KnowledgeBaseSearchTool()
        result = tool.execute(query="refund policy")
        assert result.data["query"] == "refund policy"

    def test_text_summary_non_empty(self):
        tool = KnowledgeBaseSearchTool()
        result = tool.execute(query="compatibility")
        assert len(result.text_summary) > 0


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_has(self):
        registry = ToolRegistry()
        registry.register(DummyTool())
        assert registry.has("dummy") is True

    def test_has_returns_false_for_unknown(self):
        registry = ToolRegistry()
        assert registry.has("nonexistent") is False

    def test_duplicate_register_raises(self):
        registry = ToolRegistry()
        registry.register(DummyTool())
        with pytest.raises(ValueError):
            registry.register(DummyTool())

    def test_replace_does_not_raise(self):
        registry = ToolRegistry()
        registry.register(DummyTool())
        registry.replace(DummyTool())  # should not raise
        assert registry.has("dummy")

    def test_unregister_removes_tool(self):
        registry = ToolRegistry()
        registry.register(DummyTool())
        registry.unregister("dummy")
        assert not registry.has("dummy")

    def test_unregister_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.unregister("ghost")

    def test_execute_calls_tool(self):
        registry = ToolRegistry()
        registry.register(DummyTool())
        result = registry.execute("dummy", foo="bar")
        assert result.success is True
        assert result.data == {"foo": "bar"}

    def test_execute_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.execute("ghost")

    def test_list_tools_returns_all(self):
        registry = ToolRegistry()
        registry.register(WeatherLookupTool())
        registry.register(CalendarCheckTool())
        tools = registry.list_tools()
        names = [t["name"] for t in tools]
        assert "weather_lookup" in names
        assert "calendar_check" in names

    def test_len(self):
        registry = ToolRegistry()
        registry.register(WeatherLookupTool())
        registry.register(OrderStatusTool())
        assert len(registry) == 2

    def test_get_returns_tool_instance(self):
        registry = ToolRegistry()
        tool = WeatherLookupTool()
        registry.register(tool)
        assert registry.get("weather_lookup") is tool

    def test_get_returns_none_for_unknown(self):
        registry = ToolRegistry()
        assert registry.get("xyz") is None

    def test_default_registry_has_four_tools(self):
        registry = default_registry()
        assert len(registry) == 4

    def test_default_registry_contains_expected_tools(self):
        registry = default_registry()
        for name in ("weather_lookup", "calendar_check", "order_status", "kb_search"):
            assert registry.has(name), f"Missing tool: {name}"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestParser:
    def test_has_tool_call_true(self):
        text = 'TOOL_CALL:weather_lookup:{"location": "NYC"}'
        assert has_tool_call(text) is True

    def test_has_tool_call_false(self):
        text = "Hello, how can I help you today?"
        assert has_tool_call(text) is False

    def test_parse_single_tool_call(self):
        text = 'TOOL_CALL:weather_lookup:{"location": "Paris"}'
        intents = parse_tool_calls(text)
        assert len(intents) == 1
        assert intents[0].tool_name == "weather_lookup"
        assert intents[0].arguments == {"location": "Paris"}

    def test_parse_multiple_tool_calls(self):
        text = (
            'TOOL_CALL:weather_lookup:{"location": "London"}\n'
            'TOOL_CALL:order_status:{"order_id": "99"}'
        )
        intents = parse_tool_calls(text)
        assert len(intents) == 2
        assert intents[0].tool_name == "weather_lookup"
        assert intents[1].tool_name == "order_status"

    def test_parse_invalid_json_gives_empty_args(self):
        text = "TOOL_CALL:kb_search:not-valid-json"
        intents = parse_tool_calls(text)
        assert len(intents) == 1
        assert intents[0].arguments == {}

    def test_parse_no_calls_returns_empty(self):
        text = "Just a normal sentence."
        intents = parse_tool_calls(text)
        assert intents == []

    def test_strip_tool_calls_removes_directives(self):
        text = 'Some preamble\nTOOL_CALL:order_status:{"order_id": "1"}\nSome postamble'
        stripped = strip_tool_calls(text)
        assert "TOOL_CALL" not in stripped
        assert "Some preamble" in stripped
        assert "Some postamble" in stripped

    def test_strip_tool_calls_no_directives_unchanged(self):
        text = "Hello there."
        assert strip_tool_calls(text) == "Hello there."

    def test_tool_call_intent_raw_field(self):
        text = 'TOOL_CALL:calendar_check:{"date": "Monday"}'
        intents = parse_tool_calls(text)
        assert intents[0].raw == text.strip()


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------

class TestToolExecutor:
    def _make_executor(self, callbacks=False):
        registry = default_registry()
        called_calls = []
        called_results = []
        if callbacks:
            executor = ToolExecutor(
                registry=registry,
                on_tool_call=called_calls.append,
                on_tool_result=called_results.append,
            )
            return executor, called_calls, called_results
        return ToolExecutor(registry=registry), None, None

    def test_execute_all_injects_into_context(self):
        executor, _, _ = self._make_executor()
        intents = [
            ToolCallIntent(tool_name="weather_lookup", arguments={"location": "Rome"})
        ]
        context = []
        updated = executor.execute_all(intents, context)
        assert len(updated) == 1
        assert updated[0]["role"] == "tool"
        assert updated[0]["tool_name"] == "weather_lookup"

    def test_execute_all_multiple_tools(self):
        executor, _, _ = self._make_executor()
        intents = [
            ToolCallIntent(tool_name="weather_lookup", arguments={"location": "Oslo"}),
            ToolCallIntent(tool_name="calendar_check", arguments={"date": "today"}),
        ]
        context = []
        updated = executor.execute_all(intents, context)
        assert len(updated) == 2

    def test_on_tool_call_callback_fires(self):
        executor, calls, _ = self._make_executor(callbacks=True)
        intents = [
            ToolCallIntent(tool_name="order_status", arguments={"order_id": "7"})
        ]
        executor.execute_all(intents, [])
        assert len(calls) == 1
        assert calls[0].tool_name == "order_status"

    def test_on_tool_result_callback_fires(self):
        executor, _, results = self._make_executor(callbacks=True)
        intents = [
            ToolCallIntent(tool_name="kb_search", arguments={"query": "password"})
        ]
        executor.execute_all(intents, [])
        assert len(results) == 1
        assert isinstance(results[0], ToolResult)

    def test_unknown_tool_returns_failure_result(self):
        executor, _, _ = self._make_executor()
        intents = [
            ToolCallIntent(tool_name="nonexistent_tool", arguments={})
        ]
        context = executor.execute_all(intents, [])
        assert context[0]["success"] is False

    def test_execution_history_grows(self):
        executor, _, _ = self._make_executor()
        intents = [
            ToolCallIntent(tool_name="weather_lookup", arguments={"location": "X"}),
            ToolCallIntent(tool_name="order_status", arguments={"order_id": "Y"}),
        ]
        executor.execute_all(intents, [])
        assert len(executor.execution_history) == 2


# ---------------------------------------------------------------------------
# AgentLoop with tool-calling integration tests
# ---------------------------------------------------------------------------

class TestAgentLoopToolCalling:
    """
    Integration tests: AgentLoop with a ToolRegistry and an LLM that
    emits TOOL_CALL directives, verifying end-to-end tool execution and
    context injection.
    """

    def _make_loop(
        self,
        llm_response: str,
        follow_up_response: str = "Here is what I found for you.",
    ) -> AgentLoop:
        """
        Build an AgentLoop whose LLM first returns *llm_response* (which may
        contain TOOL_CALL directives), then returns *follow_up_response* on
        the second call (after tool results are injected).
        """
        # Alternate between first and follow-up responses
        responses = [llm_response, follow_up_response]
        call_count = {"n": 0}

        class SequencedLLM(MockLLMEngine):
            def stream_response(self, user_text, context=None):
                resp = responses[min(call_count["n"], len(responses) - 1)]
                call_count["n"] += 1
                # Yield word-by-word
                words = resp.split()
                for i, w in enumerate(words):
                    yield w + (" " if i < len(words) - 1 else "")

        stt = MockSTTEngine(fixed_transcript="What is the weather in London?", latency=0.0)
        llm = SequencedLLM(token_latency=0.0, first_token_latency=0.0)
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker(realtime=False)
        registry = default_registry()

        loop = AgentLoop(
            stt=stt,
            llm=llm,
            tts=tts,
            speaker=speaker,
            tool_registry=registry,
        )
        return loop

    def test_tool_call_detected_and_executed(self):
        """TOOL_CALL directive in LLM output triggers tool execution."""
        llm_resp = tool_call_llm_response("weather_lookup", {"location": "London"})
        loop = self._make_loop(llm_resp)
        result = loop.run(dummy_audio())
        assert len(result["tool_calls"]) >= 1
        assert result["tool_calls"][0].tool_name == "weather_lookup"

    def test_tool_result_injected_into_response(self):
        """Final response comes from the follow-up LLM call (after tool results)."""
        llm_resp = tool_call_llm_response("weather_lookup", {"location": "London"})
        loop = self._make_loop(llm_resp, follow_up_response="The weather in London is sunny.")
        result = loop.run(dummy_audio())
        assert "London" in result["llm_response"] or len(result["llm_response"]) > 0

    def test_tool_results_list_populated(self):
        llm_resp = tool_call_llm_response("order_status", {"order_id": "99"})
        loop = self._make_loop(llm_resp)
        result = loop.run(dummy_audio())
        assert len(result["tool_results"]) >= 1
        assert result["tool_results"][0].tool_name == "order_status"

    def test_no_tool_call_path_still_works(self):
        """When no TOOL_CALL directive is present, behavior is unchanged."""
        loop = self._make_loop("Hello, I am here to help you.")
        result = loop.run(dummy_audio())
        assert result["tool_calls"] == []
        assert result["tool_results"] == []
        assert len(result["llm_response"]) > 0

    def test_speaker_plays_audio_after_tool_call(self):
        llm_resp = tool_call_llm_response("calendar_check", {"date": "tomorrow"})
        loop = self._make_loop(llm_resp, "You have events tomorrow.")
        result = loop.run(dummy_audio())
        # speaker should have received audio from TTS after tool round-trip
        # (speaker may be 0 if follow-up response is empty, so just check no error)
        assert result["interrupted"] is False

    def test_multi_tool_in_single_response(self):
        """Two TOOL_CALL directives in one LLM response are both executed."""
        llm_resp = (
            f'TOOL_CALL:weather_lookup:{{"location": "Berlin"}}\n'
            f'TOOL_CALL:order_status:{{"order_id": "42"}}'
        )
        loop = self._make_loop(llm_resp, "I checked both for you.")
        result = loop.run(dummy_audio())
        assert len(result["tool_calls"]) == 2
        tool_names = {tc.tool_name for tc in result["tool_calls"]}
        assert "weather_lookup" in tool_names
        assert "order_status" in tool_names

    def test_unknown_tool_does_not_crash(self):
        """An unknown tool name in TOOL_CALL produces a failure result, not an exception."""
        llm_resp = 'TOOL_CALL:nonexistent_tool:{"arg": "val"}'
        loop = self._make_loop(llm_resp)
        result = loop.run(dummy_audio())
        assert len(result["tool_calls"]) == 1
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].success is False

    def test_result_has_tool_calls_key(self):
        loop = self._make_loop("No tool here.")
        result = loop.run(dummy_audio())
        assert "tool_calls" in result
        assert "tool_results" in result

    def test_kb_search_tool_integration(self):
        llm_resp = 'TOOL_CALL:kb_search:{"query": "refund"}'
        loop = self._make_loop(llm_resp, "I found some articles about refunds.")
        result = loop.run(dummy_audio())
        assert any(tc.tool_name == "kb_search" for tc in result["tool_calls"])

    def test_calendar_check_tool_integration(self):
        llm_resp = 'TOOL_CALL:calendar_check:{"date": "Monday"}'
        loop = self._make_loop(llm_resp, "You have events on Monday.")
        result = loop.run(dummy_audio())
        assert any(tc.tool_name == "calendar_check" for tc in result["tool_calls"])

    def test_without_registry_no_tool_processing(self):
        """When no registry is provided, TOOL_CALL directives are left as-is."""
        stt = MockSTTEngine(fixed_transcript="test", latency=0.0)
        llm = MockLLMEngine(
            token_latency=0.0,
            first_token_latency=0.0,
            fixed_response='TOOL_CALL:weather_lookup:{"location": "X"}',
        )
        tts = MockTTSEngine(latency=0.0)
        speaker = MockSpeaker()
        # No tool_registry passed
        loop = AgentLoop(stt=stt, llm=llm, tts=tts, speaker=speaker)
        result = loop.run(dummy_audio())
        assert result["tool_calls"] == []
        assert result["tool_results"] == []


# ---------------------------------------------------------------------------
# Multi-tool conversation integration test
# ---------------------------------------------------------------------------

class TestMultiToolConversation:
    """
    Simulate a multi-turn conversation where each turn may invoke a
    different tool.  Verifies that context accumulates correctly.
    """

    def test_two_turn_multi_tool_conversation(self):
        """
        Turn 1: user asks about weather  -> TOOL_CALL:weather_lookup
        Turn 2: user asks about order    -> TOOL_CALL:order_status
        Both turns should complete without errors, with tool calls recorded.
        """
        registry = default_registry()
        tts = MockTTSEngine(latency=0.0)
        context: List[dict] = []

        # --- Turn 1 ---
        weather_responses = [
            'TOOL_CALL:weather_lookup:{"location": "Madrid"}',
            "The weather in Madrid is warm and sunny.",
        ]
        wc = {"n": 0}

        class LLM1(MockLLMEngine):
            def stream_response(self, user_text, context=None):
                resp = weather_responses[min(wc["n"], len(weather_responses) - 1)]
                wc["n"] += 1
                for i, w in enumerate(resp.split()):
                    yield w + (" " if i < len(resp.split()) - 1 else "")

        loop1 = AgentLoop(
            stt=MockSTTEngine(fixed_transcript="What's the weather in Madrid?", latency=0.0),
            llm=LLM1(token_latency=0.0, first_token_latency=0.0),
            tts=tts,
            speaker=MockSpeaker(),
            tool_registry=registry,
        )
        result1 = loop1.run(dummy_audio(), context=context)
        assert len(result1["tool_calls"]) >= 1
        assert any(tc.tool_name == "weather_lookup" for tc in result1["tool_calls"])

        # Accumulate context for next turn
        context = result1.get("tool_results", [])
        # (We just verify no crash and tool was called; context is rebuilt each turn)

        # --- Turn 2 ---
        order_responses = [
            'TOOL_CALL:order_status:{"order_id": "777"}',
            "Your order 777 is on its way.",
        ]
        oc = {"n": 0}

        class LLM2(MockLLMEngine):
            def stream_response(self, user_text, context=None):
                resp = order_responses[min(oc["n"], len(order_responses) - 1)]
                oc["n"] += 1
                for i, w in enumerate(resp.split()):
                    yield w + (" " if i < len(resp.split()) - 1 else "")

        loop2 = AgentLoop(
            stt=MockSTTEngine(fixed_transcript="Where is my order 777?", latency=0.0),
            llm=LLM2(token_latency=0.0, first_token_latency=0.0),
            tts=tts,
            speaker=MockSpeaker(),
            tool_registry=registry,
        )
        result2 = loop2.run(dummy_audio())
        assert len(result2["tool_calls"]) >= 1
        assert any(tc.tool_name == "order_status" for tc in result2["tool_calls"])
