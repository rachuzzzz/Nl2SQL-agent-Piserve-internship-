"""
End-to-end regression test for the resolve_selected_rows deterministic
backstop in loop.py — 2026-08-07 fix. A user checked specific row
checkboxes in the UI, then asked "internal stakeholders for these"; the
reasoning LLM ignored the selection and answered over the FULL previous
result set instead, because a competing "this is the COMPLETE result
set... Filter by X IN (...)" context line offered a ready-made filter that
needed no extra tool call.

Same object.__new__(AgentOrchestrator) + real ToolRegistry pattern as
test_ordinal_backstop_redirect.py — resolve_selected_rows is a pure
in-memory session lookup, no DB needed.
"""

import json

from src.agent.orchestrator.loop import AgentOrchestrator
from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn
from src.agent.tools import ToolRegistry


def _fake_orchestrator_with_registry(domain: str, canned_tool_calls: list[dict]) -> AgentOrchestrator:
    fake = object.__new__(AgentOrchestrator)
    fake.domain = domain
    fake._system_prompt = "SYSTEM PROMPT"
    fake._builder_module = type(
        "B", (), {"build_dynamic_prompt_additions": staticmethod(lambda q, score_fn=None: "")}
    )()
    fake.embed_model = None
    fake._scope_primary_table = "audit_report"
    fake._scope_primary_alias = "arep"
    fake._debug = None
    fake.registry = ToolRegistry(
        db_engine=None, semantic_index=None, validator=None, sql_llm=None, domain=domain,
    )

    _calls = {"n": 0}

    def _call_llm(messages, stats, iteration=None):
        idx = min(_calls["n"], len(canned_tool_calls) - 1)
        _calls["n"] += 1
        return json.dumps(canned_tool_calls[idx])

    fake._call_llm = _call_llm
    return fake


def _session_with_selection(positions: list[int]) -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show details those", answer="5 audits found", tool_used="execute_sql",
        sql="SELECT arep.audit_id FROM audit_report arep WHERE arep.status IN ('CLOSED','CLOSED_WITH_EXCEPTION')",
        total_count=5, audit_ids=["AUD-1", "AUD-2", "AUD-3", "AUD-4", "AUD-5"],
        row_titles=["AUD-1", "AUD-2", "AUD-3", "AUD-4", "AUD-5"],
        row_ids=["uuid-1", "uuid-2", "uuid-3", "uuid-4", "uuid-5"],
        row_id_columns=["id", "id", "id", "id", "id"], row_entity_type="audit",
        domain="audit",
    ))
    session.pending_selected_positions = positions
    return session


def test_generate_sql_is_blocked_before_resolve_selected_rows_is_called():
    """The core guarantee: generate_sql must never dispatch while a
    checkbox selection is unresolved."""
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "using the full list from context", "tool": "generate_sql",
         "args": {"question": "internal stakeholders for these",
                   "schema_hint": "SELECT ... WHERE arep.audit_id IN ('AUD-1','AUD-2','AUD-3','AUD-4','AUD-5')"}},
        {"thought": "ok, resolving selection first", "tool": "resolve_selected_rows", "args": {}},
        {"thought": "done", "tool": "final_answer", "args": {"answer": "Resolved to AUD-1 and AUD-3."}},
    ])
    session = _session_with_selection([1, 3])

    result = orch.query(
        "internal stakeholders for these",
        session=session, visibility_scope={},
    )

    assert result.success is True
    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" in tool_sequence
    assert "resolve_selected_rows" in tool_sequence
    # generate_sql must never have been DISPATCHED (only attempted, then
    # blocked) — no step named "generate_sql" should carry a real result.
    assert not any(s.tool == "generate_sql" for s in result.steps)

    blocked_step = next(s for s in result.steps if s.tool == "[blocked]")
    assert blocked_step.result["error"] == "redirected to resolve_selected_rows"

    resolve_step = next(s for s in result.steps if s.tool == "resolve_selected_rows")
    assert resolve_step.result["success"] is True
    assert resolve_step.result["result"]["resolved"] is True
    assert resolve_step.result["result"]["target_ids"] == ["uuid-1", "uuid-3"]


def test_backstop_does_not_fire_once_resolve_selected_rows_already_ran():
    """If the LLM calls resolve_selected_rows FIRST (the intended,
    non-backstop-triggered path), a subsequent generate_sql call for the
    SAME turn must be allowed through."""
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "resolving selection first", "tool": "resolve_selected_rows", "args": {}},
        {"thought": "now answering", "tool": "final_answer",
         "args": {"answer": "Resolved to uuid-1, uuid-3."}},
    ])
    session = _session_with_selection([1, 3])

    result = orch.query(
        "internal stakeholders for these",
        session=session, visibility_scope={},
    )

    assert result.success is True
    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" not in tool_sequence
    assert tool_sequence[0] == "resolve_selected_rows"


def test_backstop_does_not_fire_without_pending_selection():
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "no selection pending, using context", "tool": "generate_sql",
         "args": {"question": "internal stakeholders for these", "schema_hint": "SELECT 1"}},
        {"thought": "done", "tool": "final_answer", "args": {"answer": "N/A"}},
    ])
    session = _session_with_selection([])  # nothing selected

    result = orch.query(
        "internal stakeholders for these",
        session=session, visibility_scope={},
    )

    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" not in tool_sequence


def test_backstop_does_not_fire_for_clearly_fresh_question():
    """A pending selection alongside a question with no backward-reference
    marker must not hijack an unrelated fresh question — guards against a
    stale/leftover selection incorrectly blocking generate_sql."""
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "fresh question, unrelated to any selection", "tool": "generate_sql",
         "args": {"question": "list all audit plans", "schema_hint": "SELECT 1"}},
        {"thought": "done", "tool": "final_answer", "args": {"answer": "N/A"}},
    ])
    session = _session_with_selection([1, 3])

    result = orch.query(
        "list all audit plans",
        session=session, visibility_scope={},
    )

    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" not in tool_sequence
