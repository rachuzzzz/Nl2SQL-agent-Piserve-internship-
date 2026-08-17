"""
Regression tests for loop.py's Stage 4b (2026-08-07 generation-quality plan,
qid=e63f03af): a deprecation warning fires when the reasoning LLM emits a
direct execute_sql call, but the path itself is left fully functional —
Stage 4a only removed the tool's declaration/worked examples from the
system prompts, it did not touch dispatch. These are the first tests to
ever drive the LLM-authored direct execute_sql path through
parse_tool_call -> dispatch -> loop.py (confirmed by prior audit: no
existing test exercised this before).
"""

import json
import logging

from src.agent.orchestrator.loop import AgentOrchestrator
from src.agent.orchestrator.session import ConversationSession


class _FakeRegistry:
    """Records every execute_sql call's args, returns a canned benign
    result (no rows, so synthesis's early "no results" branches handle it
    without needing further orchestrator fixture setup)."""

    def __init__(self):
        self.execute_sql_calls: list[dict] = []

    def call(self, tool_name, args):
        if tool_name == "execute_sql":
            self.execute_sql_calls.append(dict(args))
            return {"success": True, "result": {"rows": [], "columns": []}}
        raise AssertionError(f"unexpected tool call in test: {tool_name}")


def _fake_orchestrator(registry: _FakeRegistry, sql: str = "SELECT 1") -> AgentOrchestrator:
    fake = object.__new__(AgentOrchestrator)
    fake.domain = "inspection"
    fake._system_prompt = "SYSTEM PROMPT"
    fake._builder_module = type(
        "B", (), {"build_dynamic_prompt_additions": staticmethod(lambda q, score_fn=None: "")}
    )()
    fake.embed_model = None
    fake._scope_primary_table = "inspection_report"
    fake._scope_primary_alias = "ir"
    fake._debug = None
    fake.registry = registry
    _raw = json.dumps({
        "thought": "running the sql I already have",
        "tool": "execute_sql",
        "args": {"sql": sql},
    })
    fake._call_llm = lambda messages, stats, iteration=None: _raw
    return fake


def test_direct_execute_sql_still_works_and_logs_deprecation_warning(caplog):
    registry = _FakeRegistry()
    orch = _fake_orchestrator(registry)
    session = ConversationSession()

    with caplog.at_level(logging.WARNING, logger="inspectai.query"):
        result = orch.query("what does this return", session=session, visibility_scope={})

    assert result.success is True
    assert registry.execute_sql_calls, "execute_sql was never dispatched"
    # question backfilled from the original question (loop.py's pre-existing
    # fix, confirmed still intact — the LLM's own call never supplies one).
    assert registry.execute_sql_calls[0]["question"] == "what does this return"
    assert any("[deprecated]" in r.message and "execute_sql" in r.message for r in caplog.records)


def test_direct_execute_sql_scope_still_enforced():
    """The enforce_scope safety net (loop.py, unrelated to Stage 4b) must
    still apply to a direct execute_sql call after the deprecation warning
    was added — the warning must not have been inserted in a way that
    short-circuits or reorders the existing scope-enforcement branch."""
    registry = _FakeRegistry()
    orch = _fake_orchestrator(registry, sql="SELECT ir.id FROM inspection_report ir")
    session = ConversationSession()

    from src.sql.validation import SQLValidator
    orch.validator = SQLValidator(domain="inspection")

    result = orch.query(
        "what does this return", session=session,
        visibility_scope={"sub": "123", "permissions": ["conduct an inspection"]},
    )

    assert result.success is True
    assert registry.execute_sql_calls, "execute_sql was never dispatched"
    assert "ir.inspector_user_id" in registry.execute_sql_calls[0]["sql"]
