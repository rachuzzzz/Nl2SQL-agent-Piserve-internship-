"""
Regression tests for loop.py's final_answer backstop for unqueried count
follow-ups — 2026-08-07 fix (debug_llm_20260807_153128, qid=3dbcdaf3): "how
many of the ones listed above are completed" was answered directly via
final_answer with ZERO tool calls, based on a fabricated recollection of
the previous turn's filter.

Same _FakeRegistry/_fake_orchestrator pattern as
test_refuse_or_clarify_after_failures.py.
"""

import json

from src.agent.orchestrator.loop import AgentOrchestrator
from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


class _FakeRegistry:
    """Returns canned generate_sql/execute_sql results in order, records
    every execute_sql call — lets tests assert whether the backstop
    actually forced a real query."""

    def __init__(self, gen_sql_results=None, exec_sql_results=None):
        self._gen_sql_results = list(gen_sql_results or [])
        self._exec_sql_results = list(exec_sql_results or [])
        self.execute_sql_calls: list[dict] = []
        self.generate_sql_calls: list[dict] = []

    def call(self, tool_name, args):
        if tool_name == "generate_sql":
            self.generate_sql_calls.append(args)
            return (self._gen_sql_results.pop(0)
                    if len(self._gen_sql_results) > 1 else self._gen_sql_results[0])
        if tool_name == "execute_sql":
            self.execute_sql_calls.append(args)
            return (self._exec_sql_results.pop(0)
                    if len(self._exec_sql_results) > 1 else self._exec_sql_results[0])
        raise AssertionError(f"unexpected tool call in test: {tool_name}")


def _seed_session() -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="list all conducted audits",
        answer="18 audits found",
        tool_used="execute_sql",
        sql="SELECT arep.audit_id FROM audit_report arep WHERE arep.status = 'SUBMITTED'",
        total_count=18,
        audit_ids=["A1", "A2", "A3", "A4", "A5"],
        row_titles=["A1", "A2", "A3", "A4", "A5"],
        domain="audit",
    ))
    return session


def _fake_orchestrator(registry: _FakeRegistry, answers: list[str]) -> AgentOrchestrator:
    """_call_llm returns a final_answer for each item in `answers`, in
    order (repeating the last once exhausted) — lets a test simulate the
    model calling final_answer more than once."""
    fake = object.__new__(AgentOrchestrator)
    fake.domain = "audit"
    fake._system_prompt = "SYSTEM PROMPT"
    fake._builder_module = type(
        "B", (), {"build_dynamic_prompt_additions": staticmethod(lambda q, score_fn=None: "")}
    )()
    fake.embed_model = None
    fake._scope_primary_table = "audit_report"
    fake._scope_primary_alias = "arep"
    fake._debug = None
    fake.registry = registry
    _calls = {"n": 0}

    def _call_llm(messages, stats, iteration=None):
        idx = min(_calls["n"], len(answers) - 1)
        _calls["n"] += 1
        return json.dumps({
            "thought": "answering directly",
            "tool": "final_answer",
            "args": {"answer": answers[idx]},
        })

    fake._call_llm = _call_llm
    return fake


def test_direct_final_answer_is_blocked_for_live_phrasing():
    registry = _FakeRegistry()
    orch = _fake_orchestrator(registry, ["All 18 audits listed above are completed."])
    session = _seed_session()

    result = orch.query(
        "how many of the ones listed above are completed",
        session=session, visibility_scope={},
    )

    # Blocked at least once — the loop then keeps re-offering the same
    # final_answer (this fake never actually calls generate_sql), so it
    # runs to max_iterations and falls into the forced-summary path. The
    # assertion that matters is that the backstop fired and no unqueried
    # answer was accepted as the final result.
    assert result.success is True
    assert any(s.tool == "[blocked]" for s in result.steps)
    assert registry.execute_sql_calls == []


def test_final_answer_with_question_mark_is_not_blocked():
    registry = _FakeRegistry()
    orch = _fake_orchestrator(registry, ["Did you mean the completed ones or all of them?"])
    session = _seed_session()

    result = orch.query(
        "how many of the ones listed above are completed",
        session=session, visibility_scope={},
    )

    assert result.success is True
    assert not any(s.tool == "[blocked]" for s in result.steps)
    assert result.answer == "Did you mean the completed ones or all of them?"


def test_fresh_question_with_no_matching_previous_turn_is_not_blocked():
    registry = _FakeRegistry()
    orch = _fake_orchestrator(registry, ["There were no conducted audits last month."])
    session = _seed_session()

    result = orch.query(
        "how many audits were conducted last month",
        session=session, visibility_scope={},
    )

    assert result.success is True
    assert not any(s.tool == "[blocked]" for s in result.steps)


def test_blocked_at_most_once_per_turn():
    """Without the one-shot _count_followup_redirected flag, a model that
    keeps calling final_answer would get redirected every single time,
    looping to max_iterations on the SAME block repeatedly."""
    registry = _FakeRegistry()
    orch = _fake_orchestrator(registry, [
        "All 18 audits listed above are completed.",
        "All 18 audits listed above are completed.",
        "All 18 audits listed above are completed.",
    ])
    session = _seed_session()

    result = orch.query(
        "how many of the ones listed above are completed",
        session=session, visibility_scope={},
    )

    assert result.success is True
    blocked_steps = [s for s in result.steps if s.tool == "[blocked]"
                      and "count follow-up" in s.thought]
    assert len(blocked_steps) == 1
