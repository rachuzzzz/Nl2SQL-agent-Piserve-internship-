"""
Regression tests for src/sql/validation/engine.py's LIMIT_MISMATCH check
(SOFT_WARN, not HARD_FAIL) — added in the same effort that fixed loop.py's
LIMIT-preservation hint (the "use LIMIT N" prose injected into schema_hint
before generate_sql, previously unenforced anywhere downstream).

Deliberately SOFT_WARN: a top-level LIMIT mismatch isn't always wrong — a
legitimate query can need a LIMIT inside a CTE/subquery to control join
fan-out, with no matching top-level LIMIT at all. HARD_FAILing here would
block correct SQL, which this whole validator redesign exists to avoid —
these tests lock in that a mismatch is visible (for observability) but
NEVER blocks execution or appears in a retry prompt.
"""

from src.sql.validation.engine import SQLValidator


def _validator(domain="inspection"):
    return SQLValidator(domain=domain)


def test_limit_mismatch_produces_a_soft_warn_not_a_hard_fail():
    v = _validator()
    r = v.validate(
        "SELECT ir.inspection_id FROM inspection_report ir "
        "ORDER BY ir.submitted_on DESC LIMIT 10",
        question="show the last 5 inspections",
    )
    warn_codes = [i.code for i in r.issues if i.severity == "SOFT_WARN"]
    hard_codes = [i.code for i in r.issues if i.severity == "HARD_FAIL"]
    assert "LIMIT_MISMATCH" in warn_codes
    assert "LIMIT_MISMATCH" not in hard_codes
    assert r.passed is True


def test_soft_warn_never_appears_in_retry_message():
    v = _validator()
    r = v.validate(
        "SELECT ir.inspection_id FROM inspection_report ir "
        "ORDER BY ir.submitted_on DESC LIMIT 10",
        question="show the last 5 inspections",
    )
    assert "LIMIT_MISMATCH" not in r.retry_message()


def test_matching_limit_produces_no_warning():
    v = _validator()
    r = v.validate(
        "SELECT ir.inspection_id FROM inspection_report ir "
        "ORDER BY ir.submitted_on DESC LIMIT 5",
        question="show the last 5 inspections",
    )
    warn_codes = [i.code for i in r.issues if i.severity == "SOFT_WARN"]
    assert "LIMIT_MISMATCH" not in warn_codes


def test_date_window_question_never_triggers_the_check_at_all():
    """Reuses the same (already date-window-excluding) regex loop.py's hint
    itself uses — a pure date filter must never be flagged, regardless of
    what LIMIT the SQL happens to have."""
    v = _validator()
    r = v.validate(
        "SELECT ir.inspection_id FROM inspection_report ir "
        "WHERE ir.submitted_on > CURRENT_DATE - 30 LIMIT 100",
        question="show inspections from the last 30 days",
    )
    warn_codes = [i.code for i in r.issues if i.severity == "SOFT_WARN"]
    assert "LIMIT_MISMATCH" not in warn_codes


def test_cte_inner_limit_does_not_falsely_trigger_when_outer_limit_matches():
    """The check must compare against the OUTER/top-level LIMIT, not a
    LIMIT buried inside a CTE controlling join fan-out."""
    v = _validator()
    r = v.validate(
        "WITH recent AS (SELECT id FROM inspection_report ORDER BY submitted_on DESC LIMIT 3) "
        "SELECT r.id, ir.inspection_id FROM recent r "
        "JOIN inspection_report ir ON ir.id = r.id LIMIT 5",
        question="show the last 5 inspections",
    )
    warn_codes = [i.code for i in r.issues if i.severity == "SOFT_WARN"]
    assert "LIMIT_MISMATCH" not in warn_codes


def test_no_requested_count_in_question_never_triggers_the_check():
    v = _validator()
    r = v.validate(
        "SELECT ir.inspection_id FROM inspection_report ir LIMIT 100",
        question="list all audit plans",
    )
    warn_codes = [i.code for i in r.issues if i.severity == "SOFT_WARN"]
    assert "LIMIT_MISMATCH" not in warn_codes
