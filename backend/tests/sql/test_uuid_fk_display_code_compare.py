"""
Regression tests for src/sql/validation/ast_checks.py's
_check_uuid_fk_compared_to_display_code_ast() — the inverse of
_check_display_code_uuid_compare_ast (test_display_code_uuid_compare.py).

Confirmed live TWICE: the reasoning model generated
WHERE aca.audit_id = '2026/06/ST041/AUD003' (2026-07-29, 4 iterations to
self-correct) and WHERE aca.audit_id = '2026/06/ST074/AUD001' (2026-08-04,
2 iterations to self-correct) — audit_corrective_action.audit_id is a
genuine UUID FK to audit_report.id, not a display code itself. Both failed
at DB execution with psycopg2.errors.InvalidTextRepresentation: invalid
input syntax for type uuid.

Must stay scoped to the exact (alias, column) pair, not the bare column
name — the negative-case tests below lock in that a display-code string
compared against the display-code column ON ITS OWN TABLE (e.g.
arep.audit_id, ir.inspection_id) is legitimate and must not be flagged by
this check (that's the ordinary, correct usage — see
test_display_code_uuid_compare.py's own positive-usage test).
"""

from src.sql.validation.engine import SQLValidator

_CODE = "2026/06/ST074/AUD001"
_LIVE_UUID = "e5e62e10-7519-49d2-932b-0abe44678abb"


def _validator(domain="audit"):
    return SQLValidator(domain=domain)


def test_blocks_the_exact_live_bug_20260804():
    v = _validator()
    sql = (
        "SELECT COUNT(*) AS ca_count FROM audit_corrective_action aca "
        f"WHERE aca.audit_id = '{_CODE}' AND aca.active = true "
        "AND aca.adequacy_status = 'APPROVE';"
    )
    result = v.validate(sql, "how many corrective actions were raised for this audit")
    assert not result.passed
    assert "UUID_FK_COMPARED_TO_DISPLAY_CODE" in [i.code for i in result.hard_fails]


def test_blocks_the_exact_live_bug_20260729():
    v = _validator()
    sql = (
        "SELECT aca.corrective_action_id, aca.status FROM audit_corrective_action aca "
        "WHERE aca.audit_id = '2026/06/ST041/AUD003' AND aca.active = true "
        "AND aca.adequacy_status = 'APPROVE';"
    )
    result = v.validate(sql, "all corrective actions for audit report 2026/06/ST041/AUD003")
    assert not result.passed
    assert "UUID_FK_COMPARED_TO_DISPLAY_CODE" in [i.code for i in result.hard_fails]


def test_blocks_schedule_id_uuid_fk_display_code_compare():
    """audit_report.schedule_id is a real UUID FK to audit_schedule.id —
    comparing it to a display-code string is the same trap."""
    v = _validator()
    sql = "SELECT arep.status FROM audit_report arep WHERE arep.schedule_id = '2026/06/ST074/SCH001';"
    result = v.validate(sql, "audit report for this schedule")
    assert not result.passed
    assert "UUID_FK_COMPARED_TO_DISPLAY_CODE" in [i.code for i in result.hard_fails]


def test_blocks_inspection_id_uuid_fk_display_code_compare():
    """inspection_corrective_action.inspection_id is a real UUID FK to
    inspection_report.id — the exact inspection-domain mirror of the
    live audit-domain bug (never yet observed live, but same exposure)."""
    v = _validator(domain="inspection")
    sql = "SELECT ica.status FROM inspection_corrective_action ica WHERE ica.inspection_id = '2026/04/ST001/INS001';"
    result = v.validate(sql, "mitigative actions for this inspection")
    assert not result.passed
    assert "UUID_FK_COMPARED_TO_DISPLAY_CODE" in [i.code for i in result.hard_fails]


def test_does_not_false_positive_on_legitimate_uuid_compare():
    """The correct usage of these same columns — a real UUID, resolved from
    session/ordinal context — must not be flagged by this check."""
    v = _validator()
    sql = f"SELECT aca.status FROM audit_corrective_action aca WHERE aca.audit_id = '{_LIVE_UUID}';"
    result = v.validate(sql, "corrective actions for this audit")
    assert "UUID_FK_COMPARED_TO_DISPLAY_CODE" not in [i.code for i in result.hard_fails]


def test_does_not_false_positive_on_display_code_own_table_usage():
    """audit_report.audit_id compared to its OWN display code is ordinary,
    correct usage — must not be flagged by this check (it's the other
    check's job, DISPLAY_CODE_UUID_COMPARE, to guard THAT column, and only
    against a UUID-shaped literal, which this is not)."""
    v = _validator()
    sql = f"SELECT arep.status FROM audit_report arep WHERE arep.audit_id = '{_CODE}';"
    result = v.validate(sql, "audit status")
    assert "UUID_FK_COMPARED_TO_DISPLAY_CODE" not in [i.code for i in result.hard_fails]


def test_correct_query_joining_through_parent_report_passes():
    """The fix the model should have generated instead of the live bug."""
    v = _validator()
    sql = (
        "SELECT COUNT(*) AS ca_count FROM audit_corrective_action aca "
        "JOIN audit_report arep ON aca.audit_id = arep.id "
        f"WHERE arep.audit_id = '{_CODE}' AND aca.active = true "
        "AND arep.active = true AND aca.adequacy_status = 'APPROVE';"
    )
    result = v.validate(sql, "how many corrective actions were raised for this audit")
    assert "UUID_FK_COMPARED_TO_DISPLAY_CODE" not in [i.code for i in result.hard_fails]
