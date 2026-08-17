"""
Regression tests for src/sql/generation.py's post-generation table/pattern
guard (the "[table-guard]" block, ~generation.py:380-459) — forces one
corrective retry when the generated SQL violates a fired INJECT rule's
required_tables/forbidden_patterns.

The main regression here is a confirmed real bug found during Phase 6's
evaluation: the guard hardcoded `get_engine()` (the INSPECTION RulesEngine
singleton) regardless of which domain was actually running, so for the
audit domain — whose `_inject_rule_ids` come from a completely separate
`get_audit_engine()` registry — required_tables_for()/forbidden_patterns_for()
always returned [] (the audit rule ids don't exist in the inspection
registry), and the guard silently never enforced anything for audit-domain
questions. Fixed by threading the domain-correct RulesEngine through as
generate_sql()'s new `_rules_engine` param (mirrors the existing
`_join_graph_module` threading pattern exactly).

Uses the same _FakeSqlLlm/_FakeResponse harness as
test_generation_token_accounting.py — no real LLM/DB needed.
"""

from types import SimpleNamespace

from src.sql.generation import generate_sql
from src.sql.validation import SQLValidator
from src.sql.rules import get_engine, get_audit_engine


class _FakeResponse:
    def __init__(self, text, raw):
        self._text = text
        self.raw = raw

    def __str__(self):
        return self._text


class _FakeSqlLlm:
    def __init__(self, responses):
        self.model = "fake-model"
        self._responses = list(responses)
        self.calls = 0

    def complete(self, prompt):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _usage(prompt_tokens=100, completion_tokens=10):
    return {"usage": SimpleNamespace(input_tokens=prompt_tokens, output_tokens=completion_tokens)}


def test_required_table_missing_fires_a_retry_inspection_domain():
    """Sanity case using the default (no _rules_engine passed) path — the
    inspection registry's own inject_impact_scale rule requires the
    'impact' table."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(  # missing the required 'impact' table
            "SELECT ica.risk_level FROM inspection_corrective_action ica;", _usage(),
        ),
        _FakeResponse(  # retry includes it
            "SELECT ica.risk_level, imp.name FROM inspection_corrective_action ica "
            "JOIN impact imp ON ica.impact_id = imp.id;", _usage(150, 15),
        ),
    ])
    result = generate_sql(
        sql_llm, SQLValidator(domain="inspection"),
        question="what is the impact scale for this action",
        _inject_rule_ids=["inject_impact_scale"],
    )
    assert sql_llm.calls == 2
    assert result["success"] is True


def test_forbidden_pattern_present_fires_a_retry():
    sql_llm = _FakeSqlLlm([
        _FakeResponse(  # violates inject_portfolio_modification's forbidden RETURN_FOR_MODIFICATION
            "SELECT * FROM inspector_portfolio WHERE status = 'RETURN_FOR_MODIFICATION';", _usage(),
        ),
        _FakeResponse(
            "SELECT * FROM inspector_portfolio WHERE status = 'PENDING';", _usage(150, 15),
        ),
    ])
    result = generate_sql(
        sql_llm, SQLValidator(domain="inspection"),
        question="show portfolios needing modification",
        _inject_rule_ids=["inject_portfolio_modification"],
    )
    assert sql_llm.calls == 2
    assert result["success"] is True


def test_no_inject_rule_ids_means_no_guard_at_all():
    sql_llm = _FakeSqlLlm([_FakeResponse("SELECT 1;", _usage())])
    generate_sql(sql_llm, SQLValidator(domain="inspection"), question="anything")
    assert sql_llm.calls == 1


# ── The cross-domain regression ──────────────────────────────────────────

def test_audit_domain_table_guard_uses_audit_engine_when_threaded():
    """The fix: passing _rules_engine=get_audit_engine() lets the guard
    correctly look up audit_inject_auditor_auditee_from_ca's
    required_tables/forbidden_patterns (audit_report/audit_schedule,
    forbidding a direct aca.audit_id = asch.id join) and fire a retry."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(  # violates forbidden_patterns: direct aca.audit_id = asch.id join
            "SELECT u.first_name FROM audit_corrective_action aca "
            "JOIN audit_schedule asch ON aca.audit_id = asch.id "
            "JOIN users u ON asch.auditor_id = u.id;", _usage(),
        ),
        _FakeResponse(  # retry correctly routes through audit_report
            "SELECT u.first_name FROM audit_corrective_action aca "
            "JOIN audit_report arep ON aca.audit_id = arep.id "
            "JOIN audit_schedule asch ON arep.schedule_id = asch.id "
            "JOIN users u ON asch.auditor_id = u.id;", _usage(150, 15),
        ),
    ])
    result = generate_sql(
        sql_llm, SQLValidator(domain="audit"),
        question="who is the auditor for corrective action CA004",
        _inject_rule_ids=["audit_inject_auditor_auditee_from_ca"],
        _rules_engine=get_audit_engine(),
    )
    assert sql_llm.calls == 2, "table-guard must fire a retry — forbidden pattern present"
    assert result["success"] is True


def test_without_domain_engine_threaded_the_guard_silently_no_ops_the_old_bug():
    """Confirms the BUG, not just the fix: omitting _rules_engine (the
    exact call shape every caller had before this phase) falls back to the
    inspection singleton, which has no 'audit_inject_auditor_auditee_from_ca'
    id — required_tables_for/forbidden_patterns_for return [] for it, so the
    guard finds nothing to enforce and never retries, even though the SQL
    violates the rule's forbidden_patterns."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT u.first_name FROM audit_corrective_action aca "
            "JOIN audit_schedule asch ON aca.audit_id = asch.id "
            "JOIN users u ON asch.auditor_id = u.id;", _usage(),
        ),
    ])
    result = generate_sql(
        sql_llm, SQLValidator(domain="audit"),
        question="who is the auditor for corrective action CA004",
        _inject_rule_ids=["audit_inject_auditor_auditee_from_ca"],
        # _rules_engine intentionally omitted — reproduces the pre-fix bug
    )
    assert sql_llm.calls == 1, "old buggy behavior: guard silently no-ops, no retry"
    assert result["success"] is True


def test_inspection_engine_lookup_of_audit_only_id_returns_nothing():
    """Direct confirmation at the engine level (not just through
    generate_sql) that the two registries are genuinely disjoint — this is
    the root cause the table-guard fix depends on."""
    insp = get_engine()
    audit_ids = ["audit_inject_auditor_auditee_from_ca"]
    assert insp.required_tables_for(audit_ids) == []
    assert insp.forbidden_patterns_for(audit_ids) == []
    audit = get_audit_engine()
    assert audit.required_tables_for(audit_ids) == ["audit_report", "audit_schedule"] \
        or set(audit.required_tables_for(audit_ids)) == {"audit_report", "audit_schedule"}
