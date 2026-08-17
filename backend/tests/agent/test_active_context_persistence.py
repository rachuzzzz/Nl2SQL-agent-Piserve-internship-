"""
Live-flow regression test for active_subject_type/current_scope persisting
across MULTIPLE real AgentOrchestrator.query() calls, not just within a
single build_context_block() call (which the other test_active_context_*
files already cover directly). Mirrors test_disambiguation_backstop_redirect.py's
object.__new__(AgentOrchestrator) fixture pattern.

Turn 1 is seeded directly onto the session (bypassing real SQL execution,
which needs a live DB) — this mirrors how synthesis.py's _finish() would
have set active_subject_type after a real generate_sql/execute_sql
success; the actual mechanics of THAT are covered elsewhere
(test_row_identity_capture.py, test_disambiguation_reply_synthesis_reconciliation.py).
This test's job is narrower and specifically new coverage: proving
session.current_scope set by turn 1 is still correctly read and rendered
into the system prompt TWO turns later (turn 3), across real
AgentOrchestrator.query() calls — not just a single build_context_block()
call in isolation.
"""

import json

from src.agent.orchestrator.loop import AgentOrchestrator
from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn
from src.agent.tools import ToolRegistry


def _fake_orchestrator(domain: str, canned_tool_calls: list[dict]) -> AgentOrchestrator:
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
    _seen_system_contents = []

    def _call_llm(messages, stats, iteration=None):
        _seen_system_contents.append(messages[0].content)
        idx = min(_calls["n"], len(canned_tool_calls) - 1)
        _calls["n"] += 1
        return json.dumps(canned_tool_calls[idx])

    fake._call_llm = _call_llm
    fake._seen_system_contents = _seen_system_contents
    return fake


def test_facility_scope_persists_across_three_real_query_calls():
    session = ConversationSession()
    # Turn 1: seeded directly, as if generate_sql/execute_sql + synthesis.py's
    # _finish() had already resolved a single facility for this turn.
    session.add_turn(ConversationTurn(
        question="which facility has the most overdue actions", answer="Golf Gardens",
        tool_used="execute_sql", domain="audit",
        active_subject_type="facility", active_subject_value="Golf Gardens",
        active_subject_label="Golf Gardens", active_subject_domain="audit",
        active_subject_id="fac-1",
    ))
    assert session.current_scope["value"] == "Golf Gardens"

    # Turn 2: no facility named — must inherit scope. Canned response answers
    # directly via final_answer (no real SQL needed for THIS test's purpose).
    orch2 = _fake_orchestrator("audit", [
        {"thought": "using active context", "tool": "final_answer",
         "args": {"answer": "3 of them are still open."}},
    ])
    result2 = orch2.query("how many of them are still open", session=session, visibility_scope={})
    assert result2.success is True
    assert "Golf Gardens" in orch2._seen_system_contents[0]
    assert "fac-1" in orch2._seen_system_contents[0]

    # Turn 2's own final_answer doesn't itself re-set active_subject_type
    # (that's synthesis.py's terminal-tool path, not exercised here — see
    # module docstring), so re-seed current_scope exactly as it already was
    # to isolate what THIS test actually verifies: does a THIRD query() call,
    # two turns after the original resolution, still see it?
    assert session.current_scope["value"] == "Golf Gardens"

    orch3 = _fake_orchestrator("audit", [
        {"thought": "using active context", "tool": "final_answer",
         "args": {"answer": "Inspector Jane is assigned."}},
    ])
    result3 = orch3.query("who is the inspector assigned to them", session=session, visibility_scope={})
    assert result3.success is True
    assert "Golf Gardens" in orch3._seen_system_contents[0]
    assert "fac-1" in orch3._seen_system_contents[0]


def test_scope_correctly_absent_from_system_prompt_for_a_fresh_unrelated_question():
    """Regression guard alongside the persistence test above: a question
    with no followup marker and no fresh-subject conflict simply doesn't
    reference the facility at all — confirms the positive assertions above
    are actually meaningful (the text isn't just always present)."""
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="which facility has the most overdue actions", answer="Golf Gardens",
        tool_used="execute_sql", domain="audit",
        active_subject_type="facility", active_subject_value="Golf Gardens",
        active_subject_label="Golf Gardens", active_subject_domain="audit",
        active_subject_id="fac-1",
    ))
    orch = _fake_orchestrator("audit", [
        {"thought": "fresh question", "tool": "final_answer",
         "args": {"answer": "12 audit plans total."}},
    ])
    orch.query("how many audit plans exist in total", session=session, visibility_scope={})
    # No followup marker present ("in total" isn't a followup cue) — the
    # ACTIVE CONTEXT facility line should not appear in this turn's prompt.
    assert "ACTIVE CONTEXT" not in orch._seen_system_contents[0]
