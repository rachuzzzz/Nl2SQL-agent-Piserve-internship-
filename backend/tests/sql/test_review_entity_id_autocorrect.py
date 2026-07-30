"""
Regression tests for the review.entity_id AUTOCORRECT rules (audit_rules.py):
audit_autocorrect_review_entity_id_corrective_action_id and its reversed-
operand twin. review.entity_id must always join to audit_corrective_action.id
(UUID PK), never .corrective_action_id (VARCHAR display code) — this
substitution has no legitimate alternate reading, so it's fixed silently in
clean_sql() before validate() ever runs, instead of costing a retry.

Confirmed live via eval (APPR-09, "who is assigned to review this audit
corrective action"): even with the pre-existing BLOCK rule
(audit_block_review_entity_id_corrective_action_id) catching this mistake,
it still cost a full retry cycle as part of a 4-call/54K-token turn.
"""

from src.sql.validation.engine import SQLValidator


def _validator():
    return SQLValidator(domain="audit")


def test_forward_direction_autocorrected():
    v = _validator()
    sql = (
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "JOIN review rev ON rev.entity_id = aca.corrective_action_id "
        "AND rev.entity_type = 'AUDIT_CORRECTIVE_ACTION' WHERE aca.active = true;"
    )
    cleaned = v.clean_sql(sql)
    assert "entity_id = aca.id" in cleaned
    assert "corrective_action_id" not in cleaned.split("JOIN review")[1].split("AND")[0]


def test_reversed_direction_autocorrected():
    v = _validator()
    sql = (
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "JOIN review rev ON aca.corrective_action_id = rev.entity_id "
        "WHERE aca.active = true;"
    )
    cleaned = v.clean_sql(sql)
    assert "aca.id = rev.entity_id" in cleaned


def test_cast_suffix_is_dropped_correctly():
    v = _validator()
    sql = (
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "JOIN review rev ON rev.entity_id = aca.corrective_action_id::uuid "
        "WHERE aca.active = true;"
    )
    cleaned = v.clean_sql(sql)
    assert "entity_id = aca.id" in cleaned
    assert "::uuid" not in cleaned


def test_full_pipeline_passes_validation_on_first_attempt():
    """End-to-end: the exact APPR-09-style query that previously needed a
    validation-fail-then-retry cycle now passes on the first attempt,
    stacking with the adequacy_status autocorrect too."""
    v = _validator()
    raw_sql = (
        "SELECT aca.corrective_action_id, rev.status, rev.level "
        "FROM audit_corrective_action aca "
        "JOIN review rev ON rev.entity_id = aca.corrective_action_id "
        "AND rev.entity_type = 'AUDIT_CORRECTIVE_ACTION' WHERE aca.active = true;"
    )
    question = "who is assigned to review this audit corrective action"

    sql = v.clean_sql(raw_sql)
    sql, _ = v.ast_repair(sql)
    sql, _ = v.inject_ca_adequacy_filter(sql, question)
    sql, _ = v.downgrade_ca_multi_axis_joins(sql, question)
    result = v.validate(sql, question)

    assert "entity_id = aca.id" in sql
    assert result.passed, [str(i) for i in result.hard_fails]
