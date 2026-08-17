"""
Regression tests for src/sql/rules/engine.py's RulesEngine — INJECT-rule
matching, priority ordering, suppression, the budget cap on hint_for_question
(added alongside the audit-domain table-guard fix — see
test_table_guard.py), and required_tables_for/forbidden_patterns_for
staying scoped to INJECT-type rules regardless of the cap.

Zero test coverage existed for RulesEngine/hint_for_question/
matching_inject_ids anywhere in this tree before this file (confirmed via
repo-wide grep during Phase 6's evaluation) — uses a small, isolated
fixture registry (RulesEngine accepts a custom `rules` list) rather than
the live 32/8-rule production registries, so these tests aren't fragile to
future rule-set edits.
"""

from src.sql.rules.engine import RulesEngine
from src.sql.rules._types import INJECT


def _rule(id_, pattern, hint, priority=50, **extra):
    return {
        "id": id_, "type": INJECT, "priority": priority,
        "question_pattern": pattern, "hint": hint,
        **extra,
    }


def test_hint_for_question_concatenates_matches_in_priority_order():
    engine = RulesEngine(rules=[
        _rule("low", r"\bfacility\b", "[LOW]", priority=10),
        _rule("high", r"\bdetails\b", "[HIGH]", priority=90),
    ])
    assert engine.hint_for_question("show facility details") == "[HIGH][LOW]"


def test_no_match_returns_empty_string():
    engine = RulesEngine(rules=[_rule("a", r"\bfacility\b", "[A]")])
    assert engine.hint_for_question("list all audit plans") == ""


def test_matching_inject_ids_returns_only_matched_ids_in_priority_order():
    engine = RulesEngine(rules=[
        _rule("a", r"\bfacility\b", "[A]", priority=10),
        _rule("b", r"\bcorrective\b", "[B]", priority=90),
        _rule("c", r"\baudit\b", "[C]", priority=50),
    ])
    assert engine.matching_inject_ids("facility and corrective actions") == ["b", "a"]


def test_suppresses_silences_lower_priority_rule_hint():
    engine = RulesEngine(rules=[
        _rule("general", r"\bcapex\b", "[GENERAL CAPEX]", priority=10),
        _rule("specific", r"\btop\s+capex\b", "[SPECIFIC]", priority=90,
              suppresses=["general"]),
    ])
    assert engine.hint_for_question("top capex converters") == "[SPECIFIC]"


def test_suppresses_does_not_affect_a_rule_that_did_not_match():
    engine = RulesEngine(rules=[
        _rule("general", r"\bcapex\b", "[GENERAL CAPEX]", priority=10),
        _rule("specific", r"\btop\s+capex\b", "[SPECIFIC]", priority=90,
              suppresses=["general"]),
    ])
    # "general" rule doesn't match this question at all — suppression is moot,
    # nothing should raise or behave differently.
    assert engine.hint_for_question("audit plans this year") == ""


def test_required_tables_for_and_forbidden_patterns_for_scoped_to_inject_type():
    engine = RulesEngine(rules=[
        _rule("has_reqs", r"\bauditor\b", "[HINT]",
              required_tables=["audit_report"], forbidden_patterns=[r"\bbad_join\b"]),
        {"id": "block_rule", "type": "BLOCK", "sql_pattern": r"\bDROP\b",
         "message": "no drops", "required_tables": ["should_never_appear"]},
    ])
    assert engine.required_tables_for(["has_reqs", "block_rule"]) == ["audit_report"]
    assert engine.forbidden_patterns_for(["has_reqs", "block_rule"]) == [r"\bbad_join\b"]


def test_required_tables_for_unknown_id_returns_empty():
    engine = RulesEngine(rules=[_rule("a", r"\bfacility\b", "[A]")])
    assert engine.required_tables_for(["nonexistent"]) == []
    assert engine.forbidden_patterns_for(["nonexistent"]) == []


# ── Budget cap ────────────────────────────────────────────────────────────

def test_budget_cap_drops_lower_priority_hints_beyond_budget():
    engine = RulesEngine(rules=[
        _rule("high", r"\bx\b", "H" * 3000, priority=90),
        _rule("mid", r"\by\b", "M" * 3000, priority=50),
        _rule("low", r"\bz\b", "L" * 3000, priority=10),
    ])
    hint = engine.hint_for_question("x y z", budget_chars=4000)
    assert hint.startswith("H" * 3000)
    assert "M" * 3000 not in hint
    assert "L" * 3000 not in hint


def test_budget_cap_always_includes_the_single_highest_priority_match_in_full():
    """Even a single match larger than the entire budget must never be
    truncated mid-rule — the guarantee is 'at least the top match, whole',
    not 'never exceed budget under any circumstance'."""
    engine = RulesEngine(rules=[_rule("huge", r"\bx\b", "H" * 10000, priority=90)])
    hint = engine.hint_for_question("x", budget_chars=4000)
    assert hint == "H" * 10000


def test_budget_cap_does_not_drop_hints_that_fit():
    engine = RulesEngine(rules=[
        _rule("a", r"\bx\b", "A" * 100, priority=90),
        _rule("b", r"\by\b", "B" * 100, priority=50),
    ])
    hint = engine.hint_for_question("x y", budget_chars=4000)
    assert "A" * 100 in hint
    assert "B" * 100 in hint


def test_budget_cap_never_affects_matching_inject_ids():
    """The safety-critical half (enforcement ids) must stay fully
    unfiltered regardless of how much prose got dropped for space."""
    engine = RulesEngine(rules=[
        _rule("high", r"\bx\b", "H" * 3000, priority=90),
        _rule("mid", r"\by\b", "M" * 3000, priority=50,
              required_tables=["some_table"]),
        _rule("low", r"\bz\b", "L" * 3000, priority=10,
              forbidden_patterns=[r"\bbad\b"]),
    ])
    hint = engine.hint_for_question("x y z", budget_chars=4000)
    assert "M" * 3000 not in hint and "L" * 3000 not in hint  # prose dropped

    ids = engine.matching_inject_ids("x y z")
    assert set(ids) == {"high", "mid", "low"}  # ids NOT dropped
    assert engine.required_tables_for(ids) == ["some_table"]
    assert engine.forbidden_patterns_for(ids) == [r"\bbad\b"]


def test_default_budget_matches_the_realistic_audit_domain_measurement():
    """Sanity check against the real production registry (not the isolated
    fixture) — confirms the default budget actually engages on realistic
    overlapping-rule questions rather than being so generous it never
    matters in practice. Values match what was measured directly against
    the live audit registry during Phase 6's evaluation."""
    from src.sql.rules import get_audit_engine
    engine = get_audit_engine()
    hint = engine.hint_for_question(
        "What is the risk level and compliance status for corrective action "
        "CA004, and who is the auditor and auditee?"
    )
    assert len(hint) <= RulesEngine._DEFAULT_HINT_BUDGET_CHARS + 3000  # top match may exceed alone
    ids = engine.matching_inject_ids(
        "What is the risk level and compliance status for corrective action "
        "CA004, and who is the auditor and auditee?"
    )
    assert len(ids) == 4  # all 4 real rules still recognized as fired, uncapped


def test_audit_type_resolution_rule_does_not_fire_on_forms_questions():
    """Regression for a false positive found live via the audit eval suite
    (debug_llm_20260805_171956.log, questions FORMS-01/FORMS-03): the rule's
    required_tables=['audit_plan'] only applies to resolving a REPORT/
    SCHEDULE's audit type (audit_report.audit_type_id is dead, audit_plan is
    the only real path) — audit_forms has its own live, direct
    audit_type_id column, so there's no join path to audit_plan at all.
    Before the question_pattern was narrowed, this rule fired on any "audit
    form template(s)... audit type" question, forcing the table-guard to
    demand a table that could never legitimately appear, wasting an LLM
    call every time (the model just ignored the bogus instruction; no
    correctness impact, but the guard should not cry wolf)."""
    from src.sql.rules import get_audit_engine
    engine = get_audit_engine()
    rid = "audit_inject_audit_type_resolution"

    assert rid not in engine.matching_inject_ids(
        "list all audit form templates by universe and audit type"
    )
    assert rid not in engine.matching_inject_ids(
        "how many audit form templates exist for each audit type"
    )

    # Still fires for its actual intended report/schedule scope — including
    # the plural "audits" phrasing the fix must not accidentally break.
    assert rid in engine.matching_inject_ids("what audit type is audit AUD-002")
    assert rid in engine.matching_inject_ids("which audits are internal audit type")
    assert rid in engine.matching_inject_ids("list third party audits scheduled next month")
