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
