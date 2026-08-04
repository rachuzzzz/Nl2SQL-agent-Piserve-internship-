"""
Regression tests for src/sql/validation/ast_checks.py's
_check_display_code_wrong_table_ast() — a third failure shape in the same
display-code/UUID family as test_display_code_uuid_compare.py and
test_uuid_fk_display_code_compare.py, but neither side here is a UUID: two
ordinary, well-formed display codes from DIFFERENT tables get confused for
each other.

Confirmed live (2026-08-04): "list all schedules under First party" listed
audit_schedule rows, then "show the observations for the second one"
correctly resolved the ordinal to schedule '2026/08/ST040/SCH001', but the
reasoning model filtered audit_report.audit_id with that SCHEDULE code
instead of joining audit_schedule -> audit_report. The query executed
without error and silently returned 0 rows. Verified directly against the
DB: the schedule has a real linked report, audit_id='2026/08/ST040/AUD001'
— a genuinely different, non-interchangeable code.
"""

from src.sql.validation.engine import SQLValidator


def _validator(domain="audit"):
    return SQLValidator(domain=domain)


def test_blocks_the_exact_live_bug():
    v = _validator()
    sql = (
        "SELECT aaa.submission_id, aq.label, aaa.answer_text "
        "FROM audit_corrective_action aca "
        "JOIN audit_report arep ON aca.audit_id = arep.id "
        "JOIN audit_schedule asch ON arep.schedule_id = asch.id "
        "JOIN ai_audit_answers aaa ON aaa.audit_report_id = arep.id "
        "JOIN ai_questions aq ON aaa.element_id = aq.element_id "
        "WHERE arep.audit_id = '2026/08/ST040/SCH001' AND aq.label = 'Findings / Remarks';"
    )
    result = v.validate(sql, "show the observations for the second one")
    assert not result.passed
    assert "DISPLAY_CODE_WRONG_TABLE" in [i.code for i in result.hard_fails]


def test_blocks_schedule_id_compared_to_audit_code():
    """The reverse confusion — an audit_report code fed into schedule_id."""
    v = _validator()
    sql = "SELECT asch.schedule_status FROM audit_schedule asch WHERE asch.schedule_id = '2026/08/ST040/AUD001';"
    result = v.validate(sql, "schedule status")
    assert not result.passed
    assert "DISPLAY_CODE_WRONG_TABLE" in [i.code for i in result.hard_fails]


def test_blocks_inspection_id_compared_to_audit_code():
    """Cross-domain confusion — an audit code fed into inspection_report.inspection_id."""
    v = _validator(domain="inspection")
    sql = "SELECT ir.status FROM inspection_report ir WHERE ir.inspection_id = '2026/08/ST040/AUD001';"
    result = v.validate(sql, "inspection status")
    assert not result.passed
    assert "DISPLAY_CODE_WRONG_TABLE" in [i.code for i in result.hard_fails]


def test_does_not_flag_correct_own_table_code():
    """The ordinary, correct case for each of the three own-code columns."""
    v = _validator()
    sql = "SELECT arep.status FROM audit_report arep WHERE arep.audit_id = '2026/08/ST040/AUD001';"
    result = v.validate(sql, "audit status")
    assert "DISPLAY_CODE_WRONG_TABLE" not in [i.code for i in result.hard_fails]

    sql2 = "SELECT asch.schedule_status FROM audit_schedule asch WHERE asch.schedule_id = '2026/06/ST001/SCH002';"
    result2 = v.validate(sql2, "schedule status")
    assert "DISPLAY_CODE_WRONG_TABLE" not in [i.code for i in result2.hard_fails]

    v2 = _validator(domain="inspection")
    sql3 = "SELECT ir.status FROM inspection_report ir WHERE ir.inspection_id = '2026/07/ST006/INS001';"
    result3 = v2.validate(sql3, "inspection status")
    assert "DISPLAY_CODE_WRONG_TABLE" not in [i.code for i in result3.hard_fails]


def test_does_not_false_positive_on_corrective_action_id_own_values():
    """audit_corrective_action.corrective_action_id / inspection's own values
    embed a parent code plus a _CAnnn/_MAnnn suffix and must never fullmatch
    the own-display-code shape used by this check — they're not in the map,
    but confirm the regex itself would never wrongly match them either."""
    v = _validator()
    sql = "SELECT aca.status FROM audit_corrective_action aca WHERE aca.corrective_action_id = '2026/06/ST041/AUD003_CA002';"
    result = v.validate(sql, "corrective action status")
    assert "DISPLAY_CODE_WRONG_TABLE" not in [i.code for i in result.hard_fails]


def test_does_not_false_positive_on_non_code_string_literal():
    """An ordinary string filter unrelated to display codes at all."""
    v = _validator()
    sql = "SELECT arep.status FROM audit_report arep WHERE arep.status = 'SUBMITTED';"
    result = v.validate(sql, "submitted audit reports")
    assert "DISPLAY_CODE_WRONG_TABLE" not in [i.code for i in result.hard_fails]


def test_correct_query_joining_through_schedule_passes():
    """The fix the model should have generated instead of the live bug."""
    v = _validator()
    sql = (
        "SELECT aaa.submission_id, aq.label, aaa.answer_text "
        "FROM audit_schedule asch "
        "JOIN audit_report arep ON arep.schedule_id = asch.id "
        "JOIN ai_audit_answers aaa ON aaa.audit_report_id = arep.id "
        "JOIN ai_questions aq ON aaa.element_id = aq.element_id "
        "WHERE asch.schedule_id = '2026/08/ST040/SCH001' AND aq.label = 'Findings / Remarks';"
    )
    result = v.validate(sql, "show the observations for the second one")
    assert "DISPLAY_CODE_WRONG_TABLE" not in [i.code for i in result.hard_fails]
