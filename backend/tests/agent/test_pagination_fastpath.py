"""
Regression tests for loop.py's pagination fast-path (`_PAGINATION_ONLY_RE`/
`_HAS_NEW_SUBJECT_RE`, "Path A" for execute_sql-sourced turns and "Path B"
for get_answers-sourced turns) — evaluated for tool conversion in the same
effort that produced resolve_ordinal_reference/resolve_disambiguation_reply
and deliberately REJECTED (unlike those two): the regex pair's whole-message
anchoring already fails closed (false negatives fall through to the safe,
full LLM path; there is no dangerous false-positive direction converting to
a tool would improve on — see the tool-conversion plan history).

Real work instead: this file is the actual regression coverage (zero
existed before), plus Path B's genuine bug fix — get_answers previously had
no `offset` parameter at all, so repeated "show more" on a get_answers-
sourced turn silently re-fetched the identical first 500 rows forever.

Uses a fake `self.registry` (not a real ToolRegistry/DB) since the
fast-path's own logic — the regex guard, args construction, offset
threading, turn construction — is what's under test here, not real SQL
execution (already covered by live verification separately).
"""

from src.agent.orchestrator.loop import AgentOrchestrator
from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


class _FakeRegistry:
    def __init__(self, canned: dict):
        self.canned = canned
        self.calls: list[tuple[str, dict]] = []

    def call(self, name, args):
        self.calls.append((name, dict(args)))
        return self.canned.get(name, {"success": False, "result": None, "error": "no canned response"})


def _fake_orchestrator(domain: str, registry: _FakeRegistry) -> AgentOrchestrator:
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
    fake.registry = registry

    def _call_llm(messages, stats, iteration=None):
        raise AssertionError("pagination fast-path must never call the LLM")

    fake._call_llm = _call_llm
    return fake


# ── Path A (execute_sql-sourced turns) ──────────────────────────────────

def _session_with_sql_turn(page_offset=0) -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="list all audit plans", answer="25 plans found",
        tool_used="execute_sql", domain="audit",
        sql="SELECT ap.title FROM audit_plan ap WHERE ap.active = true ORDER BY ap.created_on DESC",
        total_count=25, page_offset=page_offset,
        columns=["title"],
    ))
    return session


def test_path_a_fires_on_show_more_and_advances_offset():
    registry = _FakeRegistry({"execute_sql": {
        "success": True,
        "result": {"rows": [{"title": f"Plan {i}"} for i in range(10)],
                   "columns": ["title"], "total_count": 25, "row_count": 10},
    }})
    orch = _fake_orchestrator("audit", registry)
    session = _session_with_sql_turn(page_offset=0)

    result = orch.query("show more", session=session, visibility_scope={})

    assert result.success is True
    assert registry.calls[0][0] == "execute_sql"
    assert registry.calls[0][1]["offset"] == 0
    assert session.turns[-1].page_offset == 10


def test_path_a_offset_accumulates_across_repeated_show_more():
    registry = _FakeRegistry({"execute_sql": {
        "success": True,
        "result": {"rows": [{"title": f"Plan {i}"} for i in range(10)],
                   "columns": ["title"], "total_count": 25, "row_count": 10},
    }})
    orch = _fake_orchestrator("audit", registry)
    session = _session_with_sql_turn(page_offset=10)  # as if one "show more" already happened

    orch.query("show more", session=session, visibility_scope={})

    assert registry.calls[0][1]["offset"] == 10
    assert session.turns[-1].page_offset == 20


def test_path_a_empty_page_says_no_more_results():
    registry = _FakeRegistry({"execute_sql": {
        "success": True,
        "result": {"rows": [], "columns": ["title"], "total_count": 25, "row_count": 0},
    }})
    orch = _fake_orchestrator("audit", registry)
    session = _session_with_sql_turn(page_offset=25)

    result = orch.query("show more", session=session, visibility_scope={})

    assert result.answer == "No more results to show."
    assert result.success is True


def test_path_a_does_not_fire_when_new_subject_word_present():
    """_HAS_NEW_SUBJECT_RE guard — 'show more facilities' names a new topic
    word ('facility'), so this must fall through to the normal LLM path,
    not be treated as pure pagination."""
    registry = _FakeRegistry({})
    orch = _fake_orchestrator("audit", registry)
    session = _session_with_sql_turn()

    try:
        orch.query("show more facilities", session=session, visibility_scope={})
    except AssertionError as e:
        assert "must never call the LLM" in str(e)
    else:
        raise AssertionError("expected the fake _call_llm assertion to fire")
    assert registry.calls == []


def test_path_a_does_not_fire_without_a_pageable_prior_turn():
    registry = _FakeRegistry({})
    orch = _fake_orchestrator("audit", registry)
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="hi", answer="hello", tool_used="final_answer", domain="audit",
    ))
    try:
        orch.query("show more", session=session, visibility_scope={})
    except AssertionError as e:
        assert "must never call the LLM" in str(e)
    else:
        raise AssertionError("expected the fake _call_llm assertion to fire")


# ── Path B (get_answers-sourced turns) ──────────────────────────────────

def _session_with_get_answers_turn(page_offset=0) -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show me the answers for inspection X", answer="500 answers found",
        tool_used="get_answers", domain="inspection",
        inspection_ids=["2026/04/ST085/INS003"], page_offset=page_offset,
    ))
    return session


def test_path_b_fires_on_show_more_and_requests_offset():
    """The actual regression test for the bug fix: get_answers previously
    had no offset parameter at all — this asserts the fast-path now
    threads prev.page_offset through to the tool call."""
    registry = _FakeRegistry({"get_answers": {
        "success": True,
        "result": {"rows": [{"question_label": f"Q{i}"} for i in range(50)],
                   "columns": ["question_label"], "row_count": 50},
    }})
    orch = _fake_orchestrator("inspection", registry)
    session = _session_with_get_answers_turn(page_offset=0)

    orch.query("show more", session=session, visibility_scope={})

    assert registry.calls[0][0] == "get_answers"
    assert registry.calls[0][1]["offset"] == 0
    assert session.turns[-1].page_offset == 50


def test_path_b_offset_advances_on_second_show_more():
    """Confirms repeated 'show more' calls no longer re-fetch the same
    first page — the core bug this phase fixes."""
    registry = _FakeRegistry({"get_answers": {
        "success": True,
        "result": {"rows": [{"question_label": f"Q{i}"} for i in range(50)],
                   "columns": ["question_label"], "row_count": 50},
    }})
    orch = _fake_orchestrator("inspection", registry)
    session = _session_with_get_answers_turn(page_offset=50)  # one page already fetched

    orch.query("show more", session=session, visibility_scope={})

    assert registry.calls[0][1]["offset"] == 50
    assert session.turns[-1].page_offset == 100


def test_path_b_empty_page_says_no_more_results():
    registry = _FakeRegistry({"get_answers": {
        "success": True,
        "result": {"rows": [], "columns": [], "row_count": 0},
    }})
    orch = _fake_orchestrator("inspection", registry)
    session = _session_with_get_answers_turn(page_offset=500)

    result = orch.query("show more", session=session, visibility_scope={})

    assert result.answer == "No more results to show."
    assert result.success is True
    assert session.turns[-1].page_offset == 500
