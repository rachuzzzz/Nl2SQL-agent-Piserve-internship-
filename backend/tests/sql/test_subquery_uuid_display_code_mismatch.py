"""
Regression tests for src/sql/validation/ast_checks.py's
_check_subquery_uuid_display_code_mismatch_ast() — a fourth failure shape
in the same audit_id/schedule_id/corrective_action_id/inspection_id family
as test_display_code_uuid_compare.py, test_uuid_fk_display_code_compare.py,
and test_display_code_wrong_table.py, but through a scalar subquery instead
of a literal, so none of those three catch it (each requires a Literal on
one side of the comparison).

Confirmed live TWICE, ~5 days apart:

  2026-07-30 (qid=d5423641): a JOIN...ON compared audit_corrective_action's
  genuine UUID FK (aca.audit_id -> audit_report.id) against a subquery that
  selected audit_report's OWN varchar display code (arep.audit_id) instead
  of arep.id. Failed with "operator does not exist: uuid = character
  varying".

  2026-08-04 (qid=9caecfb2): a WHERE compared audit_schedule's OWN varchar
  display code (asch.schedule_id — a direct literal filter would have
  sufficed) against a subquery that resolved audit_schedule's UUID id
  instead. Failed with "operator does not exist: character varying = uuid".

Unlike the literal-based checks, a uuid/varchar type mismatch can never
implicitly cast in Postgres either way, so there is no legitimate query
shape this should ever false-positive on.
"""

from src.sql.validation.engine import SQLValidator


def _validator(domain="audit"):
    return SQLValidator(domain=domain)


def test_blocks_the_exact_live_bug_20260730():
    v = _validator()
    sql = (
        "SELECT DISTINCT u_stk.first_name || ' ' || u_stk.last_name AS stakeholder_name, "
        "aca.corrective_action_id, aca.status "
        "FROM audit_plan AS ap "
        "JOIN audit_schedule AS asch ON asch.audit_plan_id = ap.id "
        "JOIN audit_corrective_action AS aca ON aca.audit_id = "
        "(SELECT arep.audit_id FROM audit_report arep WHERE arep.schedule_id = asch.id) "
        "JOIN users u_stk ON aca.internal_stakeholder_id = u_stk.id "
        "WHERE ap.title ILIKE '%Delegation retirement plan%' AND ap.active = true "
        "AND asch.active = true AND aca.active = true AND u_stk.active = true "
        "ORDER BY stakeholder_name;"
    )
    result = v.validate(sql, "stakeholders for corrective actions under Delegation retirement plan")
    assert not result.passed
    assert "SUBQUERY_UUID_DISPLAY_CODE_MISMATCH" in [i.code for i in result.hard_fails]


def test_blocks_the_exact_live_bug_20260804():
    v = _validator()
    sql = (
        "SELECT asch.schedule_status FROM audit_schedule asch "
        "WHERE asch.schedule_id = "
        "(SELECT id FROM audit_schedule WHERE schedule_id = '2026/08/ST040/SCH001');"
    )
    result = v.validate(sql, "show observations under 2026/08/ST040/SCH001")
    assert not result.passed
    assert "SUBQUERY_UUID_DISPLAY_CODE_MISMATCH" in [i.code for i in result.hard_fails]


def test_blocks_inspection_domain_mirror():
    v = _validator(domain="inspection")
    sql = (
        "SELECT ica.status FROM inspection_corrective_action ica "
        "WHERE ica.inspection_id = "
        "(SELECT ir.inspection_id FROM inspection_report ir WHERE ir.facility_id = ica.id);"
    )
    result = v.validate(sql, "mitigative actions for this inspection")
    assert not result.passed
    assert "SUBQUERY_UUID_DISPLAY_CODE_MISMATCH" in [i.code for i in result.hard_fails]


def test_does_not_false_positive_on_correct_subquery_form_case_1():
    """The fix for the 2026-07-30 bug — subquery correctly selects .id."""
    v = _validator()
    sql = (
        "SELECT aca.status FROM audit_corrective_action aca "
        "WHERE aca.audit_id = "
        "(SELECT arep.id FROM audit_report arep WHERE arep.schedule_id = '<uuid>');"
    )
    result = v.validate(sql, "corrective actions for this schedule")
    assert "SUBQUERY_UUID_DISPLAY_CODE_MISMATCH" not in [i.code for i in result.hard_fails]


def test_does_not_false_positive_on_correct_subquery_form_case_2():
    """The fix for the 2026-08-04 bug — outer column is also the UUID id."""
    v = _validator()
    sql = (
        "SELECT asch.schedule_status FROM audit_schedule asch "
        "WHERE asch.id = "
        "(SELECT id FROM audit_schedule WHERE schedule_id = '2026/08/ST040/SCH001');"
    )
    result = v.validate(sql, "schedule status")
    assert "SUBQUERY_UUID_DISPLAY_CODE_MISMATCH" not in [i.code for i in result.hard_fails]


def test_does_not_false_positive_on_direct_literal_filter():
    """The simplest, correct fix for the 2026-08-04 bug — no subquery at all."""
    v = _validator()
    sql = "SELECT asch.schedule_status FROM audit_schedule asch WHERE asch.schedule_id = '2026/08/ST040/SCH001';"
    result = v.validate(sql, "schedule status")
    assert "SUBQUERY_UUID_DISPLAY_CODE_MISMATCH" not in [i.code for i in result.hard_fails]


def test_does_not_false_positive_on_unrelated_aggregate_subquery():
    """A scalar subquery whose select expression isn't a bare column at all
    (an aggregate) — must never be misclassified."""
    v = _validator()
    sql = (
        "SELECT arep.status FROM audit_report arep "
        "WHERE arep.submitted_on = (SELECT MAX(submitted_on) FROM audit_report);"
    )
    result = v.validate(sql, "most recently submitted audit report")
    assert "SUBQUERY_UUID_DISPLAY_CODE_MISMATCH" not in [i.code for i in result.hard_fails]
