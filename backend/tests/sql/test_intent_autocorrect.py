"""
Regression tests for src/sql/validation/autocorrect_intent.py's
inject_ca_adequacy_filter() — turns a "BLOCK, hope the LLM remembers to add
the filter on retry" round trip into a deterministic fix, for the single
most common retry cause in the audit domain (per
audit_block_ca_draft_not_excluded's own docstring).

Confirmed live 2026-07-30: "show all corrective actions raised for the
above mentioned audits" needed 2 LLM round trips (~9s, ~14K tokens) purely
to add `AND aca.adequacy_status = 'APPROVE'`.
"""

from src.sql.validation.engine import SQLValidator


def _validator():
    return SQLValidator(domain="audit")


def test_injects_filter_for_general_listing_question():
    v = _validator()
    sql = (
        "SELECT aca.corrective_action_id, aca.status, asch.schedule_id "
        "FROM audit_corrective_action aca "
        "JOIN audit_report arep ON aca.audit_id = arep.id "
        "JOIN audit_schedule asch ON arep.schedule_id = asch.id "
        "WHERE aca.active = true;"
    )
    question = "corrective actions for the 3 schedules from the previous turn"
    fixed_sql, mutations = v.inject_ca_adequacy_filter(sql, question)
    assert "adequacy_status = 'APPROVE'" in fixed_sql
    assert len(mutations) == 1
    # Fixed SQL must itself pass validation now (no retry needed)
    result = v.validate(fixed_sql, question)
    assert result.passed, [str(i) for i in result.hard_fails]


def test_does_not_inject_when_question_explicitly_asks_pending():
    v = _validator()
    sql = "SELECT aca.corrective_action_id, aca.status FROM audit_corrective_action aca WHERE aca.active = true;"
    question = "show corrective actions still pending adequacy approval"
    fixed_sql, mutations = v.inject_ca_adequacy_filter(sql, question)
    assert mutations == []
    assert fixed_sql == sql


def test_does_not_inject_when_filter_already_present():
    v = _validator()
    sql = (
        "SELECT aca.corrective_action_id FROM audit_corrective_action aca "
        "WHERE aca.active = true AND aca.adequacy_status = 'APPROVE';"
    )
    question = "list all corrective actions"
    fixed_sql, mutations = v.inject_ca_adequacy_filter(sql, question)
    assert mutations == []
    assert fixed_sql == sql


def test_does_not_inject_when_question_is_not_about_corrective_actions():
    v = _validator()
    sql = "SELECT ap.title FROM audit_plan ap WHERE ap.active = true;"
    question = "list all plans"
    fixed_sql, mutations = v.inject_ca_adequacy_filter(sql, question)
    assert mutations == []
    assert fixed_sql == sql


def test_pipeline_wiring_prevents_the_block_from_ever_firing():
    """End-to-end: the exact scenario that previously needed iter=1 (BLOCK)
    + iter=2 (LLM retry) now passes validate() on the very first SQL."""
    v = _validator()
    raw_sql = (
        "SELECT aca.corrective_action_id, aca.status, aca.cause, fac.name AS facility_name "
        "FROM audit_corrective_action aca "
        "JOIN audit_report arep ON aca.audit_id = arep.id "
        "JOIN audit_schedule asch ON arep.schedule_id = asch.id "
        "JOIN facility fac ON asch.facility_id = fac.id "
        "WHERE aca.active = true;"
    )
    question = "corrective actions for the above mentioned audits"

    sql = v.clean_sql(raw_sql)
    sql, _ = v.ast_repair(sql)
    sql, mutations = v.inject_ca_adequacy_filter(sql, question)
    result = v.validate(sql, question)

    assert mutations  # the autocorrect actually fired
    assert result.passed, [str(i) for i in result.hard_fails]


def test_downgrades_multi_axis_inner_joins_to_left():
    """
    Confirmed live 2026-07-30: "now for the first one show all corrective
    actions" needed 2 full reasoning-model retries (~15s, ~30K tokens)
    purely because audit_block_ca_multi_axis_inner_join kept firing — not
    from the reasoning model padding its request, but from generation.py's
    OWN uuid-guard retry silently downgrading previously-safe LEFT JOINs to
    plain INNER JOINs while regenerating to fix an unrelated WHERE-clause
    issue.
    """
    v = _validator()
    sql = (
        "SELECT aca.corrective_action_id, rl.name AS risk_level_name, "
        "imp.name AS impact_name, comp.name AS compliance_name "
        "FROM audit_corrective_action aca "
        "JOIN risk_level rl ON aca.risk_level_id = rl.id "
        "JOIN impact imp ON aca.impact_id = imp.id "
        "JOIN compliance comp ON aca.compliance_id = comp.id "
        "WHERE aca.audit_id = '67b60528-07ec-4eb2-98c1-522e8ea0b877' AND aca.active = true "
        "AND aca.adequacy_status = 'APPROVE';"
    )
    question = "all corrective actions for audit 2026/07/ST031/AUD001"

    fixed_sql, mutations = v.downgrade_ca_multi_axis_joins(sql, question)
    assert len(mutations) == 3
    assert "LEFT JOIN" in fixed_sql
    assert "JOIN risk_level" not in fixed_sql.replace("LEFT JOIN risk_level", "")
    result = v.validate(fixed_sql, question)
    assert result.passed, [str(i) for i in result.hard_fails]


def test_does_not_downgrade_when_question_explicitly_names_two_axes():
    v = _validator()
    sql = (
        "SELECT aca.corrective_action_id, rl.name AS risk_level_name, imp.name AS impact_name "
        "FROM audit_corrective_action aca "
        "JOIN risk_level rl ON aca.risk_level_id = rl.id "
        "JOIN impact imp ON aca.impact_id = imp.id "
        "WHERE aca.active = true;"
    )
    question = "corrective actions grouped by risk level and impact"
    fixed_sql, mutations = v.downgrade_ca_multi_axis_joins(sql, question)
    assert mutations == []
    assert fixed_sql == sql


def test_does_not_touch_already_left_joined_axes():
    v = _validator()
    sql = (
        "SELECT aca.corrective_action_id, rl.name AS risk_level_name, imp.name AS impact_name "
        "FROM audit_corrective_action aca "
        "LEFT JOIN risk_level rl ON aca.risk_level_id = rl.id "
        "LEFT JOIN impact imp ON aca.impact_id = imp.id "
        "WHERE aca.active = true;"
    )
    question = "all corrective actions"
    fixed_sql, mutations = v.downgrade_ca_multi_axis_joins(sql, question)
    assert mutations == []
    assert fixed_sql == sql


def test_does_not_touch_single_axis_inner_join():
    """Only 1 axis INNER-joined — the BLOCK rule's own sql_pattern requires
    2+, so nothing should be downgraded here (a single axis INNER join is
    fine when the question DOES ask about that one axis)."""
    v = _validator()
    sql = (
        "SELECT aca.corrective_action_id, rl.name AS risk_level_name "
        "FROM audit_corrective_action aca "
        "JOIN risk_level rl ON aca.risk_level_id = rl.id "
        "WHERE aca.active = true AND rl.name = 'High';"
    )
    question = "corrective actions with high risk level"
    fixed_sql, mutations = v.downgrade_ca_multi_axis_joins(sql, question)
    assert mutations == []
    assert fixed_sql == sql
