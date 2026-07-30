"""
Audit-domain counterpart to tests/core/test_business_rules_consumption.py —
regression tests for the audit-specific validator rules built up over this
project's live-log debugging sessions (sql/rules/audit_rules.py,
validation/engine.py's HALLUCINATED_AC_COLUMNS/HALLUCINATED_AP_COLUMNS,
and the two AUTOCORRECT rules).

Unlike test_business_rules_consumption.py's print-and-collect `_check()`
helper (which never raises, so a false condition still shows PASSED under
pytest), these use plain `assert` — a regression here actually fails the
suite, which is the point: formalizing the ad-hoc `python -c` verification
scripts used to confirm each of these fixes into a permanent, fast check.

No LLM, no live DB — pure regex/string matching against fixture SQL,
constructed with SQLValidator(domain="audit").
"""

from src.sql.validation import SQLValidator

validator = SQLValidator(domain="audit")


def _hard_fail_codes(sql, question=""):
    vresult = validator.validate(sql, question)
    return {i.code for i in vresult.hard_fails}, {i.code for i in vresult.soft_warns}, vresult.passed


# ---------------------------------------------------------------------------
# BLOCK rules
# ---------------------------------------------------------------------------

def test_block_dml_delete():
    hard, _, passed = _hard_fail_codes(
        "DELETE FROM audit_corrective_action WHERE corrective_action_id = 'CA_DRAFT';"
    )
    assert "DML_BLOCKED" in hard
    assert not passed


def test_block_dml_drop():
    hard, _, passed = _hard_fail_codes("DROP TABLE audit_report;")
    assert "DML_BLOCKED" in hard
    assert not passed


def test_block_dml_update():
    hard, _, passed = _hard_fail_codes(
        "UPDATE audit_corrective_action SET status = 'CLOSED' WHERE active = true;"
    )
    assert "DML_BLOCKED" in hard
    assert not passed


def test_block_dml_stays_quiet_for_select():
    hard, _, _ = _hard_fail_codes(
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "WHERE aca.active = true AND aca.adequacy_status = 'APPROVE' LIMIT 100;"
    )
    assert "DML_BLOCKED" not in hard


def test_block_closed_with_deferred_mislabel():
    hard, _, passed = _hard_fail_codes(
        "SELECT COUNT(*) FILTER (WHERE status = 'CLOSED' AND completed_on <= target_close_out_date) "
        "AS closed_on_time, "
        "COUNT(*) FILTER (WHERE status = 'CLOSED' AND completed_on > target_close_out_date) "
        "AS closed_with_deferred "
        "FROM audit_corrective_action WHERE active = true AND adequacy_status = 'APPROVE';"
    )
    assert "AUDIT_CLOSED_WITH_DEFERRED_MISLABEL" in hard
    assert not passed


def test_block_closed_with_deferred_mislabel_stays_quiet_for_real_status_use():
    hard, _, _ = _hard_fail_codes(
        "SELECT COUNT(*) FILTER (WHERE status = 'CLOSED') AS closed_on_time, "
        "COUNT(*) FILTER (WHERE status = 'CLOSE_WITH_DEFERRED') AS closed_with_deferred "
        "FROM audit_corrective_action WHERE active = true AND adequacy_status = 'APPROVE';"
    )
    assert "AUDIT_CLOSED_WITH_DEFERRED_MISLABEL" not in hard


def test_block_forms_status_active_hallucination():
    hard, _, passed = _hard_fail_codes(
        "SELECT COUNT(DISTINCT id) AS form_count FROM audit_forms WHERE active = true "
        "AND status = 'ACTIVE' LIMIT 100;"
    )
    assert "AUDIT_FORMS_STATUS_ACTIVE_HALLUCINATION" in hard
    assert not passed


def test_block_forms_status_active_hallucination_stays_quiet_without_status_filter():
    hard, _, _ = _hard_fail_codes(
        "SELECT COUNT(DISTINCT id) AS form_count FROM audit_forms WHERE active = true LIMIT 100;"
    )
    assert "AUDIT_FORMS_STATUS_ACTIVE_HALLUCINATION" not in hard


def test_block_ca_draft_not_excluded():
    hard, _, passed = _hard_fail_codes(
        "SELECT aca.corrective_action_id, aca.correction FROM audit_corrective_action aca "
        "WHERE aca.active = true LIMIT 100;",
        question="list all corrective actions",
    )
    assert "AUDIT_CA_DRAFT_NOT_EXCLUDED" in hard
    assert not passed


def test_block_ca_draft_not_excluded_stays_quiet_when_question_asks_for_pending():
    # A question that explicitly asks about pending/draft/approval status must
    # NOT be forced to add the adequacy_status filter — see the rule's own
    # question_pattern negative-lookahead.
    hard, _, _ = _hard_fail_codes(
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "WHERE aca.active = true LIMIT 100;",
        question="which corrective actions are still pending adequacy approval",
    )
    assert "AUDIT_CA_DRAFT_NOT_EXCLUDED" not in hard


def test_block_review_entity_id_corrective_action_id():
    hard, _, passed = _hard_fail_codes(
        "SELECT * FROM review rev JOIN audit_corrective_action aca "
        "ON rev.entity_id = aca.corrective_action_id "
        "WHERE rev.entity_type = 'AUDIT_CORRECTIVE_ACTION';"
    )
    assert "AUDIT_REVIEW_ENTITY_ID_WRONG_COL" in hard
    assert not passed


def test_ca_existence_check_adequacy_filtered_stays_quiet_for_plain_status_filter():
    """
    Confirmed live 2026-07-30 (eval_audit.py WORKFLOW-08, "corrective actions
    with both implementation and adequacy approved but not yet closed"): a
    plain, correct LISTING query — aca.status NOT IN ('CLOSED', ...) is a
    normal value-list status filter, nothing to do with checking whether a
    CA exists — got blocked anyway because the rule's old sql_pattern
    matched ANY "NOT IN (" text, and the query also (correctly, per
    audit_block_ca_draft_not_excluded) has adequacy_status = 'APPROVE'.
    NOT IN now requires a subquery (the actual existence-check shape) to
    distinguish it from a literal value-list filter.
    """
    hard, _, passed = _hard_fail_codes(
        "SELECT aca.corrective_action_id, aca.status, aca.implementation_status, "
        "aca.adequacy_status FROM audit_corrective_action aca WHERE aca.active = true "
        "AND aca.implementation_status = 'APPROVE' AND aca.adequacy_status = 'APPROVE' "
        "AND aca.status NOT IN ('CLOSED', 'CLOSE_WITH_DEFERRED') ORDER BY aca.created_on DESC;",
        question="corrective actions with both implementation and adequacy approved but not yet closed",
    )
    assert "AUDIT_CA_EXISTENCE_CHECK_ADEQUACY_FILTERED" not in hard
    assert passed


def test_ca_existence_check_adequacy_filtered_still_blocks_subquery_form():
    """The rule must still catch the genuine bug it exists for: a real
    existence check (NOT IN a subquery) combined with an adequacy_status
    filter, which would wrongly exclude CA_DRAFT rows from counting as
    'raised'."""
    hard, _, passed = _hard_fail_codes(
        "SELECT arep.audit_id FROM audit_report arep WHERE arep.active = true AND arep.id NOT IN "
        "(SELECT aca.audit_id FROM audit_corrective_action aca WHERE aca.adequacy_status = 'APPROVE' "
        "AND aca.active = true);",
        question="audits with no approved corrective actions",
    )
    assert "AUDIT_CA_EXISTENCE_CHECK_ADEQUACY_FILTERED" in hard
    assert not passed


def test_ca_existence_check_adequacy_filtered_still_blocks_is_null_form():
    hard, _, passed = _hard_fail_codes(
        "SELECT arep.audit_id FROM audit_report arep LEFT JOIN audit_corrective_action aca "
        "ON aca.audit_id = arep.id AND aca.adequacy_status = 'APPROVE' "
        "WHERE aca.id IS NULL AND arep.active = true;",
        question="audits with no approved corrective actions raised",
    )
    assert "AUDIT_CA_EXISTENCE_CHECK_ADEQUACY_FILTERED" in hard
    assert not passed


def test_block_review_completed_not_approved():
    hard, _, passed = _hard_fail_codes(
        "SELECT u.first_name FROM review rev JOIN users u ON rev.reviewed_by = u.id "
        "WHERE rev.status = 'COMPLETED' AND rev.entity_type = 'AUDIT_PLAN';",
        question="which auditors approved audit plans",
    )
    assert "REVIEW_COMPLETED_NOT_APPROVED" in hard
    assert not passed


def test_block_review_completed_not_approved_stays_quiet_for_rejection_questions():
    hard, _, _ = _hard_fail_codes(
        "SELECT * FROM review rev WHERE rev.status = 'COMPLETED';",
        question="which reviews were rejected or returned for modification",
    )
    assert "REVIEW_COMPLETED_NOT_APPROVED" not in hard


def test_block_report_missing_facility_project():
    hard, _, passed = _hard_fail_codes(
        "SELECT aca.corrective_action_id, fac.name FROM audit_corrective_action aca "
        "LEFT JOIN facility fac ON aca.facility_id = fac.id "
        "WHERE aca.active = true LIMIT 100;"
    )
    assert "AUDIT_REPORT_NO_FACILITY_PROJECT" in hard
    assert not passed


def test_block_responsible_join_pending_with():
    hard, _, passed = _hard_fail_codes(
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "LEFT JOIN users u_pending ON aca.pending_with = u_pending.id "
        "WHERE aca.active = true LIMIT 100;"
    )
    assert "AUDIT_RESPONSIBLE_NOT_FK" in hard
    assert not passed


def test_block_responsible_join_responsible_column():
    hard, _, passed = _hard_fail_codes(
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "LEFT JOIN users u ON aca.responsible = u.id "
        "WHERE aca.active = true LIMIT 100;"
    )
    assert "AUDIT_RESPONSIBLE_NOT_FK" in hard
    assert not passed


def test_block_overdue_status_overbroad():
    hard, _, passed = _hard_fail_codes(
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "WHERE aca.status IN ('OPEN', 'OVERDUE') AND aca.active = true LIMIT 100;",
        question="list all corrective actions that are overdue",
    )
    assert "AUDIT_OVERDUE_INCLUDES_OPEN" in hard
    assert not passed


def test_block_overdue_status_overbroad_stays_quiet_when_question_asks_for_open_too():
    hard, _, _ = _hard_fail_codes(
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "WHERE aca.status IN ('OPEN', 'OVERDUE') AND aca.active = true LIMIT 100;",
        question="list open overdue corrective actions",
    )
    assert "AUDIT_OVERDUE_INCLUDES_OPEN" not in hard


def test_block_overdue_status_overbroad_stays_quiet_for_overdue_only_sql():
    hard, _, passed = _hard_fail_codes(
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "WHERE aca.status = 'OVERDUE' AND aca.active = true "
        "AND aca.adequacy_status = 'APPROVE' LIMIT 100;",
        question="list all corrective actions that are overdue",
    )
    assert "AUDIT_OVERDUE_INCLUDES_OPEN" not in hard
    assert passed


def test_block_sub_category_inner_join():
    hard, _, passed = _hard_fail_codes(
        "SELECT u.first_name, acat.name, asub.name FROM audit_report arep "
        "JOIN audit_schedule asch ON arep.schedule_id = asch.id "
        "JOIN users u ON asch.auditor_id = u.id "
        "JOIN audit_plan ap ON asch.audit_plan_id = ap.id "
        "JOIN audit_category acat ON ap.category_id = acat.id "
        "JOIN audit_sub_category asub ON asch.audit_sub_category_id = asub.id "
        "WHERE arep.audit_id = '2026/07/ST033/AUD001';"
    )
    assert "AUDIT_SUB_CATEGORY_INNER_JOIN" in hard
    assert not passed


def test_block_sub_category_inner_join_stays_quiet_for_left_join():
    hard, _, _ = _hard_fail_codes(
        "SELECT u.first_name, asub.name FROM audit_schedule asch "
        "JOIN users u ON asch.auditor_id = u.id "
        "LEFT JOIN audit_sub_category asub ON asch.audit_sub_category_id = asub.id "
        "WHERE asch.active = true LIMIT 100;"
    )
    assert "AUDIT_SUB_CATEGORY_INNER_JOIN" not in hard


def test_block_stakeholder_auditor_grain_mismatch():
    hard, _, passed = _hard_fail_codes(
        "SELECT u_aud.first_name AS auditor_name, u_stk.first_name AS stakeholder_name "
        "FROM audit_corrective_action aca "
        "JOIN audit_report arep ON aca.audit_id = arep.id "
        "JOIN audit_schedule asch ON arep.schedule_id = asch.id "
        "JOIN users u_aud ON asch.auditor_id = u_aud.id "
        "JOIN users u_stk ON aca.internal_stakeholder_id = u_stk.id "
        "WHERE arep.audit_id = '2026/06/ST041/AUD003';"
    )
    assert "AUDIT_STAKEHOLDER_GRAIN_MISMATCH" in hard
    assert not passed


def test_block_stakeholder_auditor_grain_mismatch_stays_quiet_with_ca_id():
    hard, _, _ = _hard_fail_codes(
        "SELECT aca.corrective_action_id, u_aud.first_name AS auditor_name, "
        "u_stk.first_name AS stakeholder_name "
        "FROM audit_corrective_action aca "
        "JOIN audit_report arep ON aca.audit_id = arep.id "
        "JOIN audit_schedule asch ON arep.schedule_id = asch.id "
        "JOIN users u_aud ON asch.auditor_id = u_aud.id "
        "JOIN users u_stk ON aca.internal_stakeholder_id = u_stk.id "
        "WHERE arep.audit_id = '2026/06/ST041/AUD003';"
    )
    assert "AUDIT_STAKEHOLDER_GRAIN_MISMATCH" not in hard


def test_block_portfolio_ap_alias_collision():
    hard, _, passed = _hard_fail_codes(
        "SELECT ap.title FROM auditor_portfolio ap WHERE ap.active = true LIMIT 100;"
    )
    assert "PORTFOLIO_AP_ALIAS_COLLISION" in hard
    assert not passed


def test_hallucinated_audit_cycle_name_column():
    hard, _, passed = _hard_fail_codes(
        "SELECT acyc.name FROM audit_cycle acyc WHERE acyc.active = true LIMIT 100;"
    )
    assert "WRONG_COLUMN" in hard
    assert not passed


def test_hallucinated_audit_plan_universe_id_column():
    hard, _, passed = _hard_fail_codes(
        "SELECT ap.id FROM audit_plan ap WHERE ap.audit_universe_id = '00000000-0000-0000-0000-000000000000';"
    )
    assert "WRONG_COLUMN" in hard
    assert not passed


# ---------------------------------------------------------------------------
# AUTOCORRECT rules
# ---------------------------------------------------------------------------

def test_autocorrect_implementation_status_approved():
    out = validator.clean_sql(
        "SELECT * FROM audit_corrective_action WHERE implementation_status = 'APPROVED';"
    )
    assert "implementation_status = 'APPROVE'" in out
    assert "'APPROVED'" not in out


def test_autocorrect_adequacy_status_approved():
    out = validator.clean_sql(
        "SELECT * FROM audit_corrective_action WHERE adequacy_status = 'APPROVED';"
    )
    assert "adequacy_status = 'APPROVE'" in out
    assert "'APPROVED'" not in out


# ---------------------------------------------------------------------------
# Regression guard: SQLValidator(domain=...) loads a genuinely different rule
# engine per domain (see engine.py: `get_audit_engine() if domain == "audit"
# else get_engine()`) — confirm an audit-only BLOCK rule stays inert on the
# inspection-domain validator, proving the domain split is load-bearing and
# not just two constants that happen to both get consumed everywhere.
# ---------------------------------------------------------------------------

def test_audit_only_rule_does_not_fire_on_inspection_validator():
    inspection_validator = SQLValidator(domain="inspection")
    vresult = inspection_validator.validate(
        "SELECT ap.title FROM auditor_portfolio ap WHERE ap.active = true LIMIT 100;"
    )
    hard = {i.code for i in vresult.hard_fails}
    assert "PORTFOLIO_AP_ALIAS_COLLISION" not in hard
