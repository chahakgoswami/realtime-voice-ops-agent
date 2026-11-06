"""Day 5 tests: escalation paths, HITL confirmation, and audit logging."""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from voice_ops_agent.escalation.rules import (
    ConfidenceThresholdRule,
    EscalationRule,
    ExplicitRequestRule,
    RepeatedFailureRule,
)
from voice_ops_agent.escalation.manager import (
    EscalationEvent,
    EscalationManager,
    EscalationReason,
)
from voice_ops_agent.escalation.audit import AuditLogger
from voice_ops_agent.escalation.hitl import HumanInTheLoopConfirmation, DESTRUCTIVE_TOOLS
from voice_ops_agent.tools.destructive_tools import OrderCancelTool, DataDeleteTool
from voice_ops_agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(
    confidence: float = 1.0,
    failure_count: int = 0,
    transcript: str = "",
    turn_count: int = 1,
) -> Dict[str, Any]:
    return {
        "confidence": confidence,
        "failure_count": failure_count,
        "transcript": transcript,
        "turn_count": turn_count,
    }


# ---------------------------------------------------------------------------
# EscalationRule sub-class tests
# ---------------------------------------------------------------------------

class TestConfidenceThresholdRule:
    def test_fires_below_threshold(self):
        rule = ConfidenceThresholdRule(threshold=0.6)
        assert rule.should_escalate(make_state(confidence=0.3)) is True

    def test_does_not_fire_above_threshold(self):
        rule = ConfidenceThresholdRule(threshold=0.6)
        assert rule.should_escalate(make_state(confidence=0.9)) is False

    def test_fires_exactly_at_threshold_boundary(self):
        rule = ConfidenceThresholdRule(threshold=0.5)
        # confidence == threshold: should NOT escalate (strictly less than)
        assert rule.should_escalate(make_state(confidence=0.5)) is False

    def test_fires_just_below_threshold(self):
        rule = ConfidenceThresholdRule(threshold=0.5)
        assert rule.should_escalate(make_state(confidence=0.499)) is True

    def test_name_property(self):
        rule = ConfidenceThresholdRule()
        assert rule.name == "low_confidence"

    def test_reason_contains_threshold(self):
        rule = ConfidenceThresholdRule(threshold=0.75)
        assert "0.75" in rule.reason

    def test_missing_confidence_key_uses_default_1(self):
        rule = ConfidenceThresholdRule(threshold=0.5)
        # state without 'confidence' key → defaults to 1.0 (no escalation)
        assert rule.should_escalate({}) is False


class TestRepeatedFailureRule:
    def test_fires_at_max_failures(self):
        rule = RepeatedFailureRule(max_failures=3)
        assert rule.should_escalate(make_state(failure_count=3)) is True

    def test_fires_above_max_failures(self):
        rule = RepeatedFailureRule(max_failures=3)
        assert rule.should_escalate(make_state(failure_count=5)) is True

    def test_does_not_fire_below_max(self):
        rule = RepeatedFailureRule(max_failures=3)
        assert rule.should_escalate(make_state(failure_count=2)) is False

    def test_name_property(self):
        rule = RepeatedFailureRule()
        assert rule.name == "repeated_failure"

    def test_missing_failure_count_does_not_escalate(self):
        rule = RepeatedFailureRule(max_failures=1)
        assert rule.should_escalate({}) is False

    def test_zero_failure_count_does_not_escalate(self):
        rule = RepeatedFailureRule(max_failures=1)
        assert rule.should_escalate(make_state(failure_count=0)) is False


class TestExplicitRequestRule:
    def test_fires_on_human_keyword(self):
        rule = ExplicitRequestRule()
        assert rule.should_escalate(make_state(transcript="I want to speak to a human")) is True

    def test_fires_on_agent_keyword(self):
        rule = ExplicitRequestRule()
        assert rule.should_escalate(make_state(transcript="Connect me to an agent please")) is True

    def test_fires_on_representative_keyword(self):
        rule = ExplicitRequestRule()
        assert rule.should_escalate(make_state(transcript="I need a representative")) is True

    def test_does_not_fire_on_normal_transcript(self):
        rule = ExplicitRequestRule()
        assert rule.should_escalate(make_state(transcript="What is the weather today?")) is False

    def test_case_insensitive(self):
        rule = ExplicitRequestRule()
        assert rule.should_escalate(make_state(transcript="CONNECT ME TO A HUMAN NOW")) is True

    def test_custom_keywords(self):
        rule = ExplicitRequestRule(keywords=["banana"])
        assert rule.should_escalate(make_state(transcript="banana please")) is True
        assert rule.should_escalate(make_state(transcript="apple please")) is False

    def test_name_property(self):
        rule = ExplicitRequestRule()
        assert rule.name == "explicit_request"

    def test_empty_transcript_does_not_fire(self):
        rule = ExplicitRequestRule()
        assert rule.should_escalate(make_state(transcript="")) is False


# ---------------------------------------------------------------------------
# EscalationManager tests
# ---------------------------------------------------------------------------

class TestEscalationManager:
    def _make_manager(self, **kwargs) -> EscalationManager:
        rules = [
            ConfidenceThresholdRule(threshold=0.5),
            RepeatedFailureRule(max_failures=3),
            ExplicitRequestRule(),
        ]
        return EscalationManager(rules=rules, **kwargs)

    def test_check_returns_event_when_rule_fires(self):
        mgr = self._make_manager()
        event = mgr.check(make_state(confidence=0.1))
        assert event is not None
        assert isinstance(event, EscalationEvent)

    def test_check_returns_none_when_no_rule_fires(self):
        mgr = self._make_manager()
        event = mgr.check(make_state(confidence=0.9, failure_count=0, transcript="hello"))
        assert event is None

    def test_escalate_records_event(self):
        mgr = self._make_manager()
        event = mgr.check(make_state(confidence=0.1))
        mgr.escalate(event)
        assert mgr.escalation_count == 1

    def test_escalate_returns_human_response(self):
        mgr = self._make_manager(human_agent_response="Connecting you now.")
        event = mgr.check(make_state(confidence=0.1))
        response = mgr.escalate(event)
        assert response == "Connecting you now."

    def test_escalate_marks_event_resolved(self):
        mgr = self._make_manager()
        event = mgr.check(make_state(confidence=0.1))
        mgr.escalate(event)
        assert event.resolved is True

    def test_check_and_escalate_shortcut(self):
        mgr = self._make_manager()
        response = mgr.check_and_escalate(make_state(confidence=0.1))
        assert response is not None
        assert isinstance(response, str)
        assert mgr.escalation_count == 1

    def test_check_and_escalate_returns_none_when_no_rule_fires(self):
        mgr = self._make_manager()
        response = mgr.check_and_escalate(make_state())
        assert response is None
        assert mgr.escalation_count == 0

    def test_on_escalate_callback_fires(self):
        received = []
        mgr = self._make_manager(on_escalate=received.append)
        event = mgr.check(make_state(confidence=0.1))
        mgr.escalate(event)
        assert len(received) == 1
        assert received[0] is event

    def test_rule_name_in_event(self):
        mgr = self._make_manager()
        event = mgr.check(make_state(confidence=0.1))
        assert event.rule_name == "low_confidence"

    def test_explicit_request_rule_fires(self):
        mgr = self._make_manager()
        event = mgr.check(make_state(transcript="please connect me to a human"))
        assert event is not None
        assert event.rule_name == "explicit_request"

    def test_repeated_failure_rule_fires(self):
        mgr = self._make_manager()
        event = mgr.check(make_state(failure_count=5))
        assert event is not None
        assert event.rule_name == "repeated_failure"

    def test_state_snapshot_in_event(self):
        mgr = self._make_manager()
        state = make_state(confidence=0.2, transcript="test")
        event = mgr.check(state)
        assert event.state_snapshot["confidence"] == 0.2
        assert event.state_snapshot["transcript"] == "test"

    def test_first_matching_rule_wins(self):
        """Confidence rule is first; even with explicit request present, low_confidence fires."""
        mgr = self._make_manager()
        event = mgr.check(make_state(confidence=0.1, transcript="connect me to a human"))
        assert event.rule_name == "low_confidence"

    def test_add_rule_increases_rule_count(self):
        mgr = EscalationManager()
        mgr.add_rule(ConfidenceThresholdRule())
        assert len(mgr.rules) == 1

    def test_last_event_returns_most_recent(self):
        mgr = self._make_manager()
        assert mgr.last_event() is None
        mgr.check_and_escalate(make_state(confidence=0.1))
        assert mgr.last_event() is not None

    def test_event_to_dict_has_required_keys(self):
        mgr = self._make_manager()
        event = mgr.check(make_state(confidence=0.1))
        d = event.to_dict()
        for key in ("timestamp", "rule_name", "reason", "state_snapshot", "resolved"):
            assert key in d

    def test_timestamp_is_recent(self):
        mgr = self._make_manager()
        before = time.time()
        event = mgr.check(make_state(confidence=0.1))
        after = time.time()
        assert before <= event.timestamp <= after


# ---------------------------------------------------------------------------
# AuditLogger tests
# ---------------------------------------------------------------------------

class TestAuditLogger:
    def _make_logger(self, tmp_path: Path) -> AuditLogger:
        return AuditLogger(path=tmp_path / "audit.json")

    def test_log_creates_file(self, tmp_path):
        logger = self._make_logger(tmp_path)
        event = EscalationEvent(
            timestamp=time.time(),
            rule_name="test_rule",
            reason="Testing",
            state_snapshot={},
        )
        logger.log(event)
        assert (tmp_path / "audit.json").exists()

    def test_log_persists_record(self, tmp_path):
        logger = self._make_logger(tmp_path)
        event = EscalationEvent(
            timestamp=1234567890.0,
            rule_name="low_confidence",
            reason="Confidence too low",
            state_snapshot={"confidence": 0.1},
        )
        logger.log(event)
        records = logger.records()
        assert len(records) == 1
        assert records[0]["rule_name"] == "low_confidence"

    def test_multiple_logs_all_persisted(self, tmp_path):
        logger = self._make_logger(tmp_path)
        for i in range(5):
            logger.log({"index": i, "rule_name": f"rule_{i}", "reason": "r",
                        "timestamp": time.time(), "state_snapshot": {}, "resolved": False})
        records = logger.records()
        assert len(records) == 5

    def test_file_is_valid_json(self, tmp_path):
        logger = self._make_logger(tmp_path)
        logger.log({"rule_name": "x", "reason": "y", "timestamp": 1.0,
                    "state_snapshot": {}, "resolved": False})
        with open(tmp_path / "audit.json", encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list)

    def test_clear_removes_records(self, tmp_path):
        logger = self._make_logger(tmp_path)
        logger.log({"rule_name": "x", "reason": "y", "timestamp": 1.0,
                    "state_snapshot": {}, "resolved": False})
        logger.clear()
        assert logger.records() == []
        assert not (tmp_path / "audit.json").exists()

    def test_reload_from_existing_file(self, tmp_path):
        logger1 = self._make_logger(tmp_path)
        logger1.log({"rule_name": "x", "reason": "y", "timestamp": 1.0,
                     "state_snapshot": {}, "resolved": False})

        # Create a second logger pointing to the same file
        logger2 = AuditLogger(path=tmp_path / "audit.json")
        assert len(logger2.records()) == 1

    def test_audit_logger_integrated_with_manager(self, tmp_path):
        logger = self._make_logger(tmp_path)
        mgr = EscalationManager(
            rules=[ConfidenceThresholdRule(threshold=0.5)],
            audit_logger=logger,
        )
        mgr.check_and_escalate(make_state(confidence=0.1))
        records = logger.records()
        assert len(records) == 1
        assert records[0]["rule_name"] == "low_confidence"

    def test_log_dict_directly(self, tmp_path):
        logger = self._make_logger(tmp_path)
        logger.log({"custom": True, "rule_name": "manual", "reason": "test",
                    "timestamp": 0.0, "state_snapshot": {}, "resolved": False})
        records = logger.records()
        assert records[0]["custom"] is True


# ---------------------------------------------------------------------------
# HumanInTheLoopConfirmation tests
# ---------------------------------------------------------------------------

class TestHumanInTheLoopConfirmation:
    def test_auto_approve_returns_true(self):
        hitl = HumanInTheLoopConfirmation(auto_approve=True)
        assert hitl.confirm("order_cancel", {"order_id": "99"}) is True

    def test_auto_deny_returns_false(self):
        hitl = HumanInTheLoopConfirmation(auto_deny=True)
        assert hitl.confirm("order_cancel", {"order_id": "99"}) is False

    def test_non_destructive_tool_auto_approved(self):
        hitl = HumanInTheLoopConfirmation(auto_deny=True)
        # weather_lookup is not in destructive_tools, so confirm_if_destructive → True
        assert hitl.confirm_if_destructive("weather_lookup", {}) is True

    def test_destructive_tool_requires_confirm(self):
        hitl = HumanInTheLoopConfirmation(auto_deny=True)
        result = hitl.confirm_if_destructive("order_cancel", {"order_id": "1"})
        assert result is False

    def test_is_destructive_true(self):
        hitl = HumanInTheLoopConfirmation()
        assert hitl.is_destructive("order_cancel") is True

    def test_is_destructive_false(self):
        hitl = HumanInTheLoopConfirmation()
        assert hitl.is_destructive("weather_lookup") is False

    def test_history_records_decision(self):
        hitl = HumanInTheLoopConfirmation(auto_approve=True)
        hitl.confirm("order_cancel", {"order_id": "5"})
        assert len(hitl.history) == 1
        assert hitl.history[0]["approved"] is True
        assert hitl.history[0]["tool_name"] == "order_cancel"

    def test_on_approved_callback_fires(self):
        approved_calls = []
        hitl = HumanInTheLoopConfirmation(
            auto_approve=True,
            on_approved=lambda tn, args: approved_calls.append(tn),
        )
        hitl.confirm("order_cancel", {})
        assert approved_calls == ["order_cancel"]

    def test_on_denied_callback_fires(self):
        denied_calls = []
        hitl = HumanInTheLoopConfirmation(
            auto_deny=True,
            on_denied=lambda tn, args: denied_calls.append(tn),
        )
        hitl.confirm("data_delete", {"record_id": "42"})
        assert denied_calls == ["data_delete"]

    def test_on_approved_not_called_on_deny(self):
        approved_calls = []
        hitl = HumanInTheLoopConfirmation(
            auto_deny=True,
            on_approved=lambda tn, args: approved_calls.append(tn),
        )
        hitl.confirm("order_cancel", {})
        assert approved_calls == []

    def test_interactive_prompt_yes_approves(self):
        hitl = HumanInTheLoopConfirmation(prompt_fn=lambda msg: "yes")
        assert hitl.confirm("order_cancel", {}) is True

    def test_interactive_prompt_no_denies(self):
        hitl = HumanInTheLoopConfirmation(prompt_fn=lambda msg: "no")
        assert hitl.confirm("order_cancel", {}) is False

    def test_interactive_prompt_y_approves(self):
        hitl = HumanInTheLoopConfirmation(prompt_fn=lambda msg: "y")
        assert hitl.confirm("order_cancel", {}) is True

    def test_eof_on_prompt_denies(self):
        def raise_eof(msg):
            raise EOFError()
        hitl = HumanInTheLoopConfirmation(prompt_fn=raise_eof)
        assert hitl.confirm("order_cancel", {}) is False

    def test_custom_destructive_tools_list(self):
        hitl = HumanInTheLoopConfirmation(
            destructive_tools=["my_custom_tool"],
            auto_approve=True,
        )
        assert hitl.is_destructive("my_custom_tool") is True
        assert hitl.is_destructive("order_cancel") is False

    def test_multiple_confirmations_accumulate_in_history(self):
        hitl = HumanInTheLoopConfirmation(auto_approve=True)
        hitl.confirm("order_cancel", {"order_id": "1"})
        hitl.confirm("data_delete", {"record_id": "A"})
        assert len(hitl.history) == 2


# ---------------------------------------------------------------------------
# Destructive mock tools tests
# ---------------------------------------------------------------------------

class TestOrderCancelTool:
    def test_returns_success_result(self):
        tool = OrderCancelTool()
        result = tool.execute(order_id="123")
        assert result.success is True

    def test_order_id_in_data(self):
        tool = OrderCancelTool()
        result = tool.execute(order_id="XYZ")
        assert result.data["order_id"] == "XYZ"
        assert result.data["status"] == "cancelled"

    def test_text_summary_mentions_order(self):
        tool = OrderCancelTool()
        result = tool.execute(order_id="42")
        assert "42" in result.text_summary

    def test_tool_name(self):
        assert OrderCancelTool().name == "order_cancel"


class TestDataDeleteTool:
    def test_returns_success_result(self):
        tool = DataDeleteTool()
        result = tool.execute(record_id="REC001")
        assert result.success is True

    def test_record_id_in_data(self):
        tool = DataDeleteTool()
        result = tool.execute(record_id="ABC")
        assert result.data["record_id"] == "ABC"
        assert result.data["deleted"] is True

    def test_text_summary_mentions_record(self):
        tool = DataDeleteTool()
        result = tool.execute(record_id="99")
        assert "99" in result.text_summary

    def test_tool_name(self):
        assert DataDeleteTool().name == "data_delete"


# ---------------------------------------------------------------------------
# Integration: EscalationManager + AuditLogger in a simulated conversation
# ---------------------------------------------------------------------------

class TestEscalationIntegration:
    """
    End-to-end scenario: a conversation degrades through three turns,
    hitting the repeated-failure escalation path, with every event
    logged to the audit file.
    """

    def test_repeated_failure_escalation_logged(self, tmp_path):
        logger = AuditLogger(path=tmp_path / "audit.json")
        mgr = EscalationManager(
            rules=[
                ConfidenceThresholdRule(threshold=0.4),
                RepeatedFailureRule(max_failures=3),
                ExplicitRequestRule(),
            ],
            audit_logger=logger,
        )

        # Simulate 3 failed turns
        for i in range(3):
            state = make_state(confidence=0.9, failure_count=i)
            response = mgr.check_and_escalate(state)
            assert response is None  # not yet triggered

        # 4th turn hits repeated_failure
        state = make_state(confidence=0.9, failure_count=3)
        response = mgr.check_and_escalate(state)
        assert response is not None
        assert "agent" in response.lower() or "support" in response.lower() or len(response) > 0

        records = logger.records()
        assert len(records) == 1
        assert records[0]["rule_name"] == "repeated_failure"

    def test_explicit_request_escalation_logged(self, tmp_path):
        logger = AuditLogger(path=tmp_path / "audit.json")
        mgr = EscalationManager(
            rules=[ExplicitRequestRule()],
            audit_logger=logger,
        )
        mgr.check_and_escalate(
            make_state(transcript="I want to speak with a human agent right now")
        )
        records = logger.records()
        assert len(records) == 1
        assert records[0]["rule_name"] == "explicit_request"

    def test_no_escalation_produces_no_audit_record(self, tmp_path):
        logger = AuditLogger(path=tmp_path / "audit.json")
        mgr = EscalationManager(
            rules=[ConfidenceThresholdRule(threshold=0.5)],
            audit_logger=logger,
        )
        mgr.check_and_escalate(make_state(confidence=0.99))
        assert logger.records() == []

    def test_hitl_blocks_destructive_tool_without_approval(self):
        hitl = HumanInTheLoopConfirmation(auto_deny=True)
        tool = OrderCancelTool()
        approved = hitl.confirm_if_destructive(tool.name, {"order_id": "99"})
        assert approved is False
        # Tool.execute() should NOT be called when not approved
        # (caller is responsible for checking approved; we just verify the gate)

    def test_hitl_allows_destructive_tool_with_approval(self):
        hitl = HumanInTheLoopConfirmation(auto_approve=True)
        tool = OrderCancelTool()
        approved = hitl.confirm_if_destructive(tool.name, {"order_id": "99"})
        assert approved is True
        result = tool.execute(order_id="99")
        assert result.success is True

    def test_full_audit_trail_json_structure(self, tmp_path):
        logger = AuditLogger(path=tmp_path / "audit.json")
        mgr = EscalationManager(
            rules=[
                ExplicitRequestRule(),
                RepeatedFailureRule(max_failures=2),
            ],
            audit_logger=logger,
        )
        mgr.check_and_escalate(make_state(transcript="I need a real person"))
        mgr.check_and_escalate(make_state(failure_count=2))

        with open(tmp_path / "audit.json", encoding="utf-8") as fh:
            data = json.load(fh)

        assert isinstance(data, list)
        assert len(data) == 2
        for record in data:
            assert "timestamp" in record
            assert "rule_name" in record
            assert "reason" in record
            assert "state_snapshot" in record
            assert "resolved" in record
