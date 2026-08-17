"""
End-to-end regression test for the resolve_disambiguation_reply
deterministic backstop in loop.py — mirrors test_ordinal_backstop_redirect.py's
fixture pattern exactly (object.__new__(AgentOrchestrator) + a REAL
ToolRegistry/tool_defs dispatch path, db_engine=None since
resolve_disambiguation_reply is a pure in-memory session lookup, no DB
needed) — proving the backstop actually intercepts a real dispatch
sequence, not just that the underlying resolution function is correct in
isolation (already covered by test_disambiguation_reply.py).

Unlike an ordinal reference, a disambiguation reply ("George Smith") has
no distinguishing lexical marker — the backstop's condition is purely
conversation STATE (a pending disambiguation on the previous turn), so
these tests are the only place that state-based condition itself gets
exercised against the real loop.
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


def _session_with_pending_people_disambiguation() -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="who is george", answer="...", tool_used="generate_sql",
        disambiguation_name="george", total_count=2,
        disambiguation_candidates=[
            {"name": "George Smith", "user_id": "uid-1"},
            {"name": "George Joseph", "user_id": "uid-2"},
        ],
    ))
    return session


def _session_with_duplicate_name_candidates() -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="who is george joseph", answer="...", tool_used="generate_sql",
        disambiguation_name="george joseph", total_count=2,
        disambiguation_candidates=[
            {"name": "George Joseph", "user_id": "uid-2"},
            {"name": "George Joseph", "user_id": "uid-3"},
        ],
    ))
    return session


def test_generate_sql_is_blocked_before_resolve_disambiguation_reply_is_called():
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "guessing", "tool": "generate_sql",
         "args": {"question": "corrective actions for george",
                   "schema_hint": "SELECT ... WHERE u.first_name ILIKE '%George%'"}},
        {"thought": "ok, resolving", "tool": "resolve_disambiguation_reply",
         "args": {"picked_name": "George Smith"}},
        {"thought": "done", "tool": "final_answer", "args": {"answer": "Resolved to George Smith."}},
    ])
    session = _session_with_pending_people_disambiguation()

    result = orch.query(
        "George Smith", session=session, visibility_scope={},
    )

    assert result.success is True
    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" in tool_sequence
    assert "resolve_disambiguation_reply" in tool_sequence
    assert not any(s.tool == "generate_sql" for s in result.steps)

    blocked_step = next(s for s in result.steps if s.tool == "[blocked]")
    assert blocked_step.result["error"] == "redirected to resolve_disambiguation_reply"

    resolve_step = next(s for s in result.steps if s.tool == "resolve_disambiguation_reply")
    assert resolve_step.result["success"] is True
    assert resolve_step.result["result"]["target_id"] == "uid-1"
    assert resolve_step.result["result"]["resolved"] is True


def test_final_answer_without_question_mark_is_also_blocked():
    """The backstop blocks a direct assertion ('I think you mean X') just
    as much as it blocks generate_sql — only a genuine clarifying question
    (containing '?') is allowed through unresolved."""
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "guessing without resolving", "tool": "final_answer",
         "args": {"answer": "I think you mean George Smith, uid-1."}},
        {"thought": "ok, resolving", "tool": "resolve_disambiguation_reply",
         "args": {"picked_name": "George Smith"}},
        {"thought": "done", "tool": "final_answer", "args": {"answer": "Resolved to George Smith."}},
    ])
    session = _session_with_pending_people_disambiguation()

    result = orch.query("George Smith", session=session, visibility_scope={})

    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" in tool_sequence
    assert "resolve_disambiguation_reply" in tool_sequence


def test_final_answer_with_question_mark_passes_through_unblocked():
    """A legitimate 'still ambiguous, please clarify further' answer
    (containing '?') must NOT be blocked — the backstop should only catch
    an unresolved ASSERTION, not a genuine re-ask."""
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "still too vague, asking again", "tool": "final_answer",
         "args": {"answer": "Which George did you mean — Smith or Joseph?"}},
    ])
    session = _session_with_pending_people_disambiguation()

    result = orch.query("tell me about george", session=session, visibility_scope={})

    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" not in tool_sequence
    assert result.answer == "Which George did you mean — Smith or Joseph?"


def test_duplicate_name_stop_path_leads_to_final_answer_not_generate_sql():
    """The ambiguous-duplicate-name STOP: resolve_disambiguation_reply
    itself returns needs_clarification (two candidates share the picked
    name), and the LLM correctly follows up with final_answer rather than
    guessing via generate_sql."""
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "guessing", "tool": "generate_sql",
         "args": {"question": "corrective actions for george joseph", "schema_hint": "..."}},
        {"thought": "resolving", "tool": "resolve_disambiguation_reply",
         "args": {"picked_name": "George Joseph"}},
        {"thought": "still ambiguous, asking for email", "tool": "final_answer",
         "args": {"answer": "Two people are named George Joseph — could you provide an email address?"}},
    ])
    session = _session_with_duplicate_name_candidates()

    result = orch.query("George Joseph", session=session, visibility_scope={})

    assert result.success is True
    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" in tool_sequence
    assert not any(s.tool == "generate_sql" for s in result.steps)

    resolve_step = next(s for s in result.steps if s.tool == "resolve_disambiguation_reply")
    assert resolve_step.result["result"]["needs_clarification"] is True
    assert resolve_step.result["result"]["resolved"] is False

    assert tool_sequence[-1] == "final_answer"
    assert "?" in result.answer


def test_backstop_does_not_fire_once_resolve_disambiguation_reply_already_ran():
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "resolving first", "tool": "resolve_disambiguation_reply",
         "args": {"picked_position": 2}},
        {"thought": "now answering", "tool": "final_answer",
         "args": {"answer": "Resolved to George Joseph, uid-2."}},
    ])
    session = _session_with_pending_people_disambiguation()

    result = orch.query("the second one", session=session, visibility_scope={})

    assert result.success is True
    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" not in tool_sequence
    assert tool_sequence[0] == "resolve_disambiguation_reply"


def test_backstop_does_not_fire_without_pending_disambiguation():
    """No pending disambiguation on the previous turn — the backstop must
    not fire on an ordinary question."""
    orch = _fake_orchestrator_with_registry("audit", [
        {"thought": "no pending disambiguation", "tool": "generate_sql",
         "args": {"question": "list all audit plans", "schema_hint": "SELECT 1"}},
        {"thought": "done", "tool": "final_answer", "args": {"answer": "N/A"}},
    ])
    session = ConversationSession()  # no turns, nothing pending
    session.add_turn(ConversationTurn(
        question="list all audit plans", answer="...", tool_used="generate_sql",
    ))

    result = orch.query("list all audit plans", session=session, visibility_scope={})

    tool_sequence = [s.tool for s in result.steps]
    assert "[blocked]" not in tool_sequence
