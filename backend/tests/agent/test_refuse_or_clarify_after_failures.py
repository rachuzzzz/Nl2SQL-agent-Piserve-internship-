"""
Regression tests for loop.py's refuse/clarify terminal path — replaces the
old "force=True execute_sql after 3 validation failures, return the result
as a normal answer with no caveat, and never reset the counter so every
subsequent failure force-executes again" behavior. Confirmed-live incidents
(see loop.py's comments) shipped over-broad/wrong answers to the user this
way with no error surfaced anywhere.

New behavior: after 3 failures of EITHER kind (validation OR execution —
both now count toward the same gen_sql_failures threshold), the loop
refuses (same failure signature every attempt — a genuine technical problem
rephrasing won't fix) or asks a clarifying question (varying signatures —
the model itself was unsure how to interpret the question), and returns
immediately without ever executing unvalidated SQL.
"""

import json

from src.agent.orchestrator.loop import AgentOrchestrator
from src.agent.orchestrator.session import ConversationSession
from src.sql.validation.models import ValidationIssue, ValidationResult


class _FakeRegistry:
    """Returns canned generate_sql/execute_sql results in order (repeating
    the last one once exhausted) and records every execute_sql call's args
    — lets tests assert the never-validated SQL is never force-executed."""

    def __init__(self, gen_sql_results=None, exec_sql_results=None):
        self._gen_sql_results = list(gen_sql_results or [])
        self._exec_sql_results = list(exec_sql_results or [])
        self.execute_sql_calls: list[dict] = []

    def call(self, tool_name, args):
        if tool_name == "generate_sql":
            return (self._gen_sql_results.pop(0)
                    if len(self._gen_sql_results) > 1 else self._gen_sql_results[0])
        if tool_name == "execute_sql":
            self.execute_sql_calls.append(args)
            return (self._exec_sql_results.pop(0)
                    if len(self._exec_sql_results) > 1 else self._exec_sql_results[0])
        raise AssertionError(f"unexpected tool call in test: {tool_name}")


def _fake_orchestrator(registry: _FakeRegistry) -> AgentOrchestrator:
    """Minimal AgentOrchestrator bypassing __init__ (same pattern as
    test_domain_redirect.py's _fake_orchestrator) — _call_llm always
    requests generate_sql, and self.registry is the canned-result stub
    above, so the real generate_sql/execute_sql/validator machinery is
    never touched."""
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
        "thought": "generating SQL",
        "tool": "generate_sql",
        "args": {"question": "placeholder", "schema_hint": ""},
    })
    fake._call_llm = lambda messages, stats, iteration=None: _raw
    return fake


def _gen_sql_ok(sql="SELECT 1"):
    return {"success": True, "result": {"sql": sql, "validation": {"passed": True}}}


def _gen_sql_validation_fail(*codes):
    return {
        "success": True,
        "result": {"sql": "SELECT 1", "validation": {"passed": False, "errors": list(codes)}},
    }


def test_stable_validation_failure_refuses_without_forced_execute():
    """3 validation failures sharing the SAME error code must refuse, not
    force-execute the never-validated SQL — the exact bug this fix closes."""
    registry = _FakeRegistry(gen_sql_results=[
        _gen_sql_validation_fail("UNDEFINED_ALIAS"),
        _gen_sql_validation_fail("UNDEFINED_ALIAS"),
        _gen_sql_validation_fail("UNDEFINED_ALIAS"),
    ])
    orch = _fake_orchestrator(registry)
    session = ConversationSession()
    result = orch.query("how many inspections happened", session=session, visibility_scope={})

    assert result.success is True
    assert "wasn't able" in result.answer
    assert session.turns[-1].tool_used == "refuse_unvalidated_sql"
    assert registry.execute_sql_calls == []  # never force-executed


def test_varying_validation_failure_signature_asks_for_clarification():
    """3 validation failures with DIFFERENT error codes each time mean the
    model was flip-flopping on intent, not stuck on one wall — clarify
    instead of refusing."""
    registry = _FakeRegistry(gen_sql_results=[
        _gen_sql_validation_fail("UNDEFINED_ALIAS"),
        _gen_sql_validation_fail("CAPEX_LEFT_JOIN_MULT"),
        _gen_sql_validation_fail("MISSING_GROUP_BY"),
    ])
    orch = _fake_orchestrator(registry)
    session = ConversationSession()
    result = orch.query("how many inspections happened", session=session, visibility_scope={})

    assert result.success is True
    assert "rephrase" in result.answer.lower()
    assert session.turns[-1].tool_used == "clarify_after_failures"
    assert registry.execute_sql_calls == []


def test_stable_execution_failure_signature_refuses():
    """SQL that always passes validation but always fails against Postgres
    with the SAME error class must also refuse after 3 attempts — this
    branch previously didn't count toward gen_sql_failures at all and could
    loop all the way to max_iterations before falling into the (separate,
    still-unfixed) forced-summary fallback."""
    registry = _FakeRegistry(
        gen_sql_results=[_gen_sql_ok()],
        exec_sql_results=[{
            "success": False,
            "error": "(psycopg2.errors.UndefinedColumn) column ip.inspector_id does not exist",
        }],
    )
    orch = _fake_orchestrator(registry)
    session = ConversationSession()
    result = orch.query("how many inspections happened", session=session, visibility_scope={})

    assert result.success is True
    assert session.turns[-1].tool_used == "refuse_unvalidated_sql"
    # No forced-execution escape hatch exists for this failure class either.
    assert all("force" not in call for call in registry.execute_sql_calls)


def _gen_sql_validation_fail_vresult(*issues: ValidationIssue):
    """Same shape as _gen_sql_validation_fail, but with a real ValidationResult
    under "_vresult" — the production shape (see generation.py), which is what
    loop.py:~1218 actually reads via _validation_issue_signature(). The plain
    _gen_sql_validation_fail helper above only ever populates "errors" (the
    legacy fallback), so it never exercises that code path at all."""
    vresult = ValidationResult(passed=False, issues=list(issues))
    return {
        "success": True,
        "result": {
            "sql": "SELECT 1",
            "validation": {
                "passed": False,
                "errors": [str(i) for i in issues],
                "_vresult": vresult,
            },
        },
    }


def test_undefined_alias_different_alias_each_attempt_clarifies_not_refuses():
    """Regression test for qid=e63f03af: 3 UNDEFINED_ALIAS failures with a
    DIFFERENT missing alias each time are 3 different bugs, not the model
    stuck on one wall — must clarify, not refuse. Before the
    _validation_issue_signature() fix, comparing by bare .code alone treated
    all three as identical and refused instead."""
    registry = _FakeRegistry(gen_sql_results=[
        _gen_sql_validation_fail_vresult(ValidationIssue(
            "HARD_FAIL", "UNDEFINED_ALIAS",
            "Table alias 'ir' is used in WHERE/SELECT but not introduced by any FROM/JOIN",
        )),
        _gen_sql_validation_fail_vresult(ValidationIssue(
            "HARD_FAIL", "UNDEFINED_ALIAS",
            "Table alias 'proj' is used in WHERE/SELECT but not introduced by any FROM/JOIN",
        )),
        _gen_sql_validation_fail_vresult(ValidationIssue(
            "HARD_FAIL", "UNDEFINED_ALIAS",
            "Table alias 'xyz' is used in WHERE/SELECT but not introduced by any FROM/JOIN",
        )),
    ])
    orch = _fake_orchestrator(registry)
    session = ConversationSession()
    result = orch.query("capex and opex breakdown for mitigative actions", session=session, visibility_scope={})

    assert result.success is True
    assert session.turns[-1].tool_used == "clarify_after_failures"
    assert registry.execute_sql_calls == []


def test_undefined_alias_same_alias_each_attempt_still_refuses():
    """Guards against over-correcting the fix into never refusing: 3
    UNDEFINED_ALIAS failures on the SAME alias every time is a genuinely
    stable failure signature and must still refuse."""
    registry = _FakeRegistry(gen_sql_results=[
        _gen_sql_validation_fail_vresult(ValidationIssue(
            "HARD_FAIL", "UNDEFINED_ALIAS",
            "Table alias 'ir' is used in WHERE/SELECT but not introduced by any FROM/JOIN",
        )),
        _gen_sql_validation_fail_vresult(ValidationIssue(
            "HARD_FAIL", "UNDEFINED_ALIAS",
            "Table alias 'ir' is used in WHERE/SELECT but not introduced by any FROM/JOIN",
        )),
        _gen_sql_validation_fail_vresult(ValidationIssue(
            "HARD_FAIL", "UNDEFINED_ALIAS",
            "Table alias 'ir' is used in WHERE/SELECT but not introduced by any FROM/JOIN",
        )),
    ])
    orch = _fake_orchestrator(registry)
    session = ConversationSession()
    result = orch.query("capex and opex breakdown for mitigative actions", session=session, visibility_scope={})

    assert result.success is True
    assert session.turns[-1].tool_used == "refuse_unvalidated_sql"
    assert registry.execute_sql_calls == []
