"""
Regression tests for src/sql/validation/ast_checks.py's
_check_display_code_uuid_compare_ast() — the inverse of the existing
bare-uuid-pk-in-SELECT check (test_bare_uuid_pk_autocorrect.py).

Confirmed live via direct DB query: ordinal-reference resolution ("the 1st
one in the list") correctly resolved a row's real UUID from session
context, but the reasoning model generated
WHERE arep.audit_id = '61b73532-bf5c-481c-b49a-355226737b9c' instead of
WHERE arep.id = '61b73532-bf5c-481c-b49a-355226737b9c' — audit_report.audit_id
is a varchar business code ('2026/06/ST001/AUD001'), never a UUID. The query
executed without error and silently returned 0 rows instead of the 4 real
"Findings / Remarks" observations that row actually has.

Must stay scoped to the exact (alias, column) pair, not the bare column
name — audit_id/schedule_id/corrective_action_id/inspection_id are each a
genuine UUID foreign key on a DIFFERENT table while being the display code
on their own table (audit_corrective_action.audit_id -> audit_report.id,
audit_report.schedule_id -> audit_schedule.id are both real UUID FKs) — the
negative-case tests below lock that in.
"""

from src.sql.validation.engine import SQLValidator

_LIVE_UUID = "61b73532-bf5c-481c-b49a-355226737b9c"


def _validator(domain="audit"):
    return SQLValidator(domain=domain)


def test_blocks_the_exact_live_bug():
    v = _validator()
    sql = (
        f"SELECT aaa.submission_id, aq.label, aaa.answer_text "
        f"FROM ai_audit_answers aaa "
        f"JOIN audit_report arep ON aaa.audit_report_id = arep.id "
        f"JOIN ai_questions aq ON aaa.element_id = aq.element_id "
        f"WHERE arep.audit_id = '{_LIVE_UUID}' AND aq.label = 'Findings / Remarks';"
    )
    result = v.validate(sql, "list all observations for the 1st one in the list")
    assert not result.passed
    codes = [i.code for i in result.hard_fails]
    assert "DISPLAY_CODE_UUID_COMPARE" in codes


def test_blocks_schedule_id_display_code_compare():
    v = _validator()
    sql = f"SELECT asch.schedule_status FROM audit_schedule asch WHERE asch.schedule_id = '{_LIVE_UUID}';"
    result = v.validate(sql, "schedule status")
    assert not result.passed
    assert "DISPLAY_CODE_UUID_COMPARE" in [i.code for i in result.hard_fails]


def test_blocks_corrective_action_id_display_code_compare_audit():
    v = _validator()
    sql = f"SELECT aca.status FROM audit_corrective_action aca WHERE aca.corrective_action_id = '{_LIVE_UUID}';"
    result = v.validate(sql, "corrective action status")
    assert not result.passed
    assert "DISPLAY_CODE_UUID_COMPARE" in [i.code for i in result.hard_fails]


def test_blocks_corrective_action_id_display_code_compare_inspection():
    v = _validator(domain="inspection")
    sql = f"SELECT ica.status FROM inspection_corrective_action ica WHERE ica.corrective_action_id = '{_LIVE_UUID}';"
    result = v.validate(sql, "corrective action status")
    assert not result.passed
    assert "DISPLAY_CODE_UUID_COMPARE" in [i.code for i in result.hard_fails]


def test_blocks_inspection_id_display_code_compare():
    v = _validator(domain="inspection")
    sql = f"SELECT ir.status FROM inspection_report ir WHERE ir.inspection_id = '{_LIVE_UUID}';"
    result = v.validate(sql, "inspection status")
    assert not result.passed
    assert "DISPLAY_CODE_UUID_COMPARE" in [i.code for i in result.hard_fails]


def test_does_not_flag_display_code_compared_to_a_real_business_code():
    """The common, correct case — audit_id used for its actual purpose."""
    v = _validator()
    sql = "SELECT arep.status FROM audit_report arep WHERE arep.audit_id = '2026/06/ST001/AUD001';"
    result = v.validate(sql, "audit status")
    assert "DISPLAY_CODE_UUID_COMPARE" not in [i.code for i in result.hard_fails]


def test_does_not_false_positive_on_legitimate_audit_id_fk_on_a_different_table():
    """audit_corrective_action.audit_id is a REAL uuid FK to audit_report.id —
    must not be confused with audit_report's OWN varchar audit_id column."""
    v = _validator()
    sql = (
        f"SELECT aca.status FROM audit_corrective_action aca "
        f"WHERE aca.audit_id = '{_LIVE_UUID}';"
    )
    result = v.validate(sql, "corrective actions for this audit")
    assert "DISPLAY_CODE_UUID_COMPARE" not in [i.code for i in result.hard_fails]


def test_does_not_false_positive_on_legitimate_schedule_id_fk_on_a_different_table():
    """audit_report.schedule_id is a REAL uuid FK to audit_schedule.id —
    must not be confused with audit_schedule's OWN varchar schedule_id column."""
    v = _validator()
    sql = (
        f"SELECT arep.status FROM audit_report arep "
        f"WHERE arep.schedule_id = '{_LIVE_UUID}';"
    )
    result = v.validate(sql, "audits for this schedule")
    assert "DISPLAY_CODE_UUID_COMPARE" not in [i.code for i in result.hard_fails]


def test_correct_query_using_real_primary_key_passes():
    """The fix the model should have generated instead of the live bug."""
    v = _validator()
    sql = (
        f"SELECT aaa.submission_id, aq.label, aaa.answer_text "
        f"FROM ai_audit_answers aaa "
        f"JOIN audit_report arep ON aaa.audit_report_id = arep.id "
        f"JOIN ai_questions aq ON aaa.element_id = aq.element_id "
        f"WHERE arep.id = '{_LIVE_UUID}' AND aq.label = 'Findings / Remarks';"
    )
    result = v.validate(sql, "list all observations for the 1st one in the list")
    assert "DISPLAY_CODE_UUID_COMPARE" not in [i.code for i in result.hard_fails]
