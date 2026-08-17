"""
Regression tests for the glossary HARD_FAIL enforcement allowlist —
2026-08-07 fix (debug_llm_20260807_153128, qid=88b9c71e): "list all
conducted audits" generated status IN ('SUBMITTED','CLOSED',
'CLOSED_WITH_EXCEPTION') — wrong per AUDIT_REPORT_SEMANTICS's own
conducted=SUBMITTED-only definition. The glossary check caught it, but was
hardcoded to SOFT_WARN regardless of the entry's own severity field, so
nothing blocked or corrected the query.

Direct SQLValidator.validate_semantic() calls — no LLM, no DB.
"""

from src.knowledge.business_glossary import GLOSSARY
from src.sql.validation import SQLValidator
from src.sql.validation.semantic import _ENFORCED_GLOSSARY_IDS


def _validator():
    return SQLValidator(domain="audit")


def _severities(warns, code="GLOSSARY_VIOLATION"):
    return [w.severity for w in warns if w.code == code]


def test_live_wrong_sql_is_hard_fail():
    warns = _validator().validate_semantic(
        "arep.status IN ('SUBMITTED', 'CLOSED', 'CLOSED_WITH_EXCEPTION')",
        "list all conducted audits",
    )
    assert "HARD_FAIL" in _severities(warns)


def test_correct_equals_form_no_violation():
    warns = _validator().validate_semantic(
        "arep.status = 'SUBMITTED'", "list all conducted audits",
    )
    assert _severities(warns) == []


def test_correct_in_single_form_no_violation():
    """Locks in the A1 regex fix — fails without it."""
    warns = _validator().validate_semantic(
        "arep.status IN ('SUBMITTED')", "list all conducted audits",
    )
    assert _severities(warns) == []


def test_two_bucket_case_form_is_exempted_to_soft_warn():
    sql = (
        "SELECT CASE WHEN arep.status IN ('CLOSED','CLOSED_WITH_EXCEPTION') "
        "THEN 'completed' ELSE 'conducted' END AS category, COUNT(*) "
        "FROM audit_report arep"
    )
    warns = _validator().validate_semantic(sql, "list conducted audits by category")
    assert _severities(warns) == ["SOFT_WARN"]


def test_completed_inspections_violation_stays_soft_warn():
    """Not in the allowlist — locks in allowlist scoping."""
    warns = _validator().validate_semantic(
        "ir.status = 'SUBMITTED'", "how many completed inspections",
    )
    sevs = _severities(warns)
    assert sevs, "expected a completed_inspections violation to fire"
    assert "HARD_FAIL" not in sevs


def test_capex_hard_fail_tagged_entry_stays_soft_warn():
    """capex_converted is tagged severity=HARD_FAIL in business_glossary.py
    but is deliberately NOT in the enforcement allowlist yet."""
    warns = _validator().validate_semantic(
        "SUM(ica.capex)", "capex converted this month",
    )
    sevs = _severities(warns)
    assert sevs, "expected a capex_converted-family violation to fire"
    assert "HARD_FAIL" not in sevs


def test_every_enforced_id_exists_in_glossary():
    """The _CONSUMED_RULE_IDS-style typo trap: an id in the allowlist that
    doesn't exist in GLOSSARY would silently enforce nothing."""
    real_ids = {e.id for e in GLOSSARY}
    for enforced_id in _ENFORCED_GLOSSARY_IDS:
        assert enforced_id in real_ids, f"{enforced_id!r} not in GLOSSARY"
