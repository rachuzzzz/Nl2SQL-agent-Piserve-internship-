"""
End-to-end regression test for the resolve_ordinal_reference deterministic
backstop in loop.py — the fix for the reasoning LLM silently ignoring the
tool and guessing via title/ILIKE instead (the actual live bug this whole
ordinal-resolution-as-a-tool effort exists to close).

Drives AgentOrchestrator.query() with a stubbed _call_llm (no real Ollama/
Anthropic call) but a REAL ToolRegistry/tool_defs dispatch path (no DB
needed — resolve_ordinal_reference is a pure in-memory session lookup),
mirroring the object.__new__(AgentOrchestrator) fixture pattern already
established in test_domain_redirect.py for exercising query() without a
live DB/embedding/Ollama connection.

Scenario: the LLM first tries to skip straight to generate_sql on a
question that references a prior result by position — the backstop must
intercept BEFORE dispatch (no SQL generation attempted at all), redirect
it, and only then let resolve_ordinal_reference actually run.
"""

import json

from src.agent.orchestrator.loop import AgentOrchestrator
from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn
from src.agent.tools import ToolRegistry


def _fake_orchestrator_with_registry(domain: str, canned_tool_calls: list[dict]) -> AgentOrchestrator:
    """Same minimal-fixture pattern as test_domain_redirect.py's
    _fake_orchestrator, extended with a real ToolRegistry (db_engine=None
    is fine — resolve_ordinal_reference never touches the DB) so the
    resolve_ordinal_reference dispatch in this test is genuine, not
    stubbed. _call_llm returns canned_tool_calls in sequence, one per
    call, cycling to the last entry if called more times than provided."""
    fake = object.__new__(AgentOrchestrator)
    fake.domain = domain
    fake._system_prompt = "SYSTEM PROMPT"
    fake._builder_module = type(
        "B", (), {"build_dynamic_prompt_additions": staticmethod(lambda q, score_fn=None: "")}
    )()
    fake.embed_model = None
    fake._scope_primary_table = "audit_corrective_action"
    fake._scope_primary_alias = "aca"
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


def _session_after_plan_listing() -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="list all audit plans", answer="3 plans found", tool_used="generate_sql",
        total_count=3, columns=["title", "status"],
        row_titles=["Plan A", "Plan B", "Plan C"],
        row_ids=["uuid-a", "uuid-b", "uuid-c"],
        row_id_columns=["id", "id", "id"], row_entity_type="plan",
    ))
    return session


def test_generate_sql_is_blocked_before_resolve_ordinal_reference_is_called():
    """The core guarantee: generate_sql must never dispatch (no SQL
    generation attempted) while an ordinal reference is unresolved."""
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "guessing", "tool": "generate_sql",
         "args": {"question": "for the first one show corrective actions",
                   "schema_hint": "SELECT ... WHERE ap.title ILIKE '%Plan A%'"}},
        {"thought": "ok, resolving first", "tool": "resolve_ordinal_reference",
         "args": {"position": 1, "entity_type_hint": "plan"}},
        {"thought": "done", "tool": "final_answer", "args": {"answer": "Resolved to Plan A."}},
    ])
    session = _session_after_plan_listing()

    result = orch.query(
        "for the first one show corrective actions",
        session=session, visibility_scope={},
    )

    assert result.success is True
    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" in tool_sequence
    assert "resolve_ordinal_reference" in tool_sequence
    # generate_sql must never have been DISPATCHED (only attempted, then
    # blocked) — no step named "generate_sql" should carry a real result.
    assert not any(s.tool == "generate_sql" for s in result.steps)

    blocked_step = next(s for s in result.steps if s.tool == "[blocked]")
    assert blocked_step.result["error"] == "redirected to resolve_ordinal_reference"

    resolve_step = next(s for s in result.steps if s.tool == "resolve_ordinal_reference")
    assert resolve_step.result["success"] is True
    assert resolve_step.result["result"]["target_id"] == "uuid-a"
    assert resolve_step.result["result"]["resolved"] is True


def test_backstop_does_not_fire_once_resolve_ordinal_reference_already_ran():
    """If the LLM calls resolve_ordinal_reference FIRST (the intended,
    non-backstop-triggered path), a subsequent generate_sql call for the
    SAME turn must be allowed through — the backstop only blocks an
    UNRESOLVED reference, not generate_sql itself."""
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "resolving first", "tool": "resolve_ordinal_reference",
         "args": {"position": 2, "entity_type_hint": "plan"}},
        {"thought": "now querying", "tool": "final_answer",
         "args": {"answer": "Resolved to Plan B, uuid-b."}},
    ])
    session = _session_after_plan_listing()

    result = orch.query(
        "for the second one show corrective actions",
        session=session, visibility_scope={},
    )

    assert result.success is True
    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" not in tool_sequence
    assert tool_sequence[0] == "resolve_ordinal_reference"


def test_backstop_does_not_fire_without_prior_turns():
    """A fresh, single-turn conversation has nothing to resolve an ordinal
    reference against — the backstop must not fire (efficiency guard, not
    correctness: resolve_ordinal_reference would just report
    needs_clarification anyway, but this saves the extra round-trip)."""
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "no history to resolve against", "tool": "generate_sql",
         "args": {"question": "for the first one show corrective actions",
                   "schema_hint": "SELECT 1"}},
        {"thought": "done", "tool": "final_answer", "args": {"answer": "N/A"}},
    ])
    session = ConversationSession()  # no turns at all

    result = orch.query(
        "for the first one show corrective actions",
        session=session, visibility_scope={},
    )

    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" not in tool_sequence
