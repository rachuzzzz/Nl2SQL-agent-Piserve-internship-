"""
Regression tests for src/sql/validation/identity.py's
strip_hallucinated_no_substitute_columns() — deterministically removes a
hallucinated display-name column (e.g. acyc.name, po.portfolio_name) from
the outermost SELECT when engine.py's HALLUCINATED_AC_COLUMNS/
HALLUCINATED_PO_COLUMNS/HALLUCINATED_IC_COLUMNS maps already confirm that
table has no such column at all.

Confirmed live and already documented in engine.py's own comments as
high-cost BEFORE this fix: PLAN-08 burned 7 LLM calls / 21s repeatedly
confabulating audit_cycle.name; PORT-03 hallucinated
auditor_portfolio.portfolio_name TWICE, once before and once after a
get_schema round-trip.
"""

from src.sql.validation.engine import SQLValidator


def test_removes_hallucinated_audit_cycle_name():
    v = SQLValidator(domain="audit")
    sql = (
        "SELECT acyc.name AS cycle_name, acyc.start_date, acyc.end_date, acyc.status "
        "FROM audit_cycle acyc WHERE acyc.active = true;"
    )
    fixed, mutations = v.strip_hallucinated_no_substitute_columns(sql)
    assert "acyc.name" not in fixed
    assert len(mutations) == 1
    result = v.validate(fixed, "all audit cycles with date range and status")
    assert result.passed, [str(i) for i in result.hard_fails]


def test_removes_hallucinated_auditor_portfolio_name():
    v = SQLValidator(domain="audit")
    sql = "SELECT po.portfolio_name, po.status FROM auditor_portfolio po WHERE po.active = true;"
    fixed, mutations = v.strip_hallucinated_no_substitute_columns(sql)
    assert "portfolio_name" not in fixed
    assert len(mutations) == 1
    result = v.validate(fixed, "which category does this auditor portfolio cover")
    assert result.passed, [str(i) for i in result.hard_fails]


def test_stacks_with_bare_uuid_pk_autocorrect():
    """CYCLE-01's exact live shape has BOTH a hallucinated acyc.name AND a
    bare acyc.id in the same SELECT — both fixes must compose cleanly."""
    v = SQLValidator(domain="audit")
    sql = (
        "SELECT acyc.id AS cycle_id, acyc.name AS cycle_name, acyc.start_date, "
        "acyc.end_date, acyc.due_date, acyc.status FROM audit_cycle acyc "
        "WHERE acyc.active = true ORDER BY acyc.start_date DESC LIMIT 100;"
    )
    fixed, mut1 = v.strip_hallucinated_no_substitute_columns(sql)
    fixed, mut2 = v.strip_bare_uuid_pk_select(fixed)
    assert mut1 and mut2
    result = v.validate(fixed, "all audit cycles with date range and status")
    assert result.passed, [str(i) for i in result.hard_fails]


def test_does_not_touch_where_clause_hallucination():
    """Deliberately scoped to SELECT only — a hallucinated column in WHERE
    must be left for the reactive check + retry, not silently dropped
    (which could widen the result set to the wrong rows)."""
    v = SQLValidator(domain="audit")
    sql = (
        "SELECT acyc.start_date, acyc.status FROM audit_cycle acyc "
        "WHERE acyc.name = 'Q1 Audit' AND acyc.active = true;"
    )
    fixed, mutations = v.strip_hallucinated_no_substitute_columns(sql)
    assert mutations == []
    assert fixed == sql


def test_gated_to_correct_domain():
    """HALLUCINATED_AC_COLUMNS/HALLUCINATED_PO_COLUMNS are audit-only;
    HALLUCINATED_IC_COLUMNS (inspection_cycle) is inspection-only."""
    v_audit = SQLValidator(domain="audit")
    ic_sql = "SELECT ic.name, ic.start_date FROM inspection_cycle ic WHERE ic.active = true;"
    fixed, mutations = v_audit.strip_hallucinated_no_substitute_columns(ic_sql)
    assert mutations == []  # wrong domain for this table's map

    v_inspection = SQLValidator(domain="inspection")
    fixed, mutations = v_inspection.strip_hallucinated_no_substitute_columns(ic_sql)
    assert len(mutations) == 1
    assert "ic.name" not in fixed


def test_skips_when_removal_would_empty_select_list():
    v = SQLValidator(domain="audit")
    sql = "SELECT acyc.name FROM audit_cycle acyc WHERE acyc.active = true;"
    fixed, mutations = v.strip_hallucinated_no_substitute_columns(sql)
    assert mutations == []
    assert fixed == sql


def test_also_cleans_up_orphaned_group_by_reference():
    """
    Confirmed live 2026-07-30 (CYCLE-05, 'count of audit plans, schedules,
    and reports in the current audit cycle'): removing acyc.name from
    SELECT alone still left `GROUP BY acyc.id, acyc.name, acyc.status`
    referencing it — validation still failed on the orphaned GROUP BY
    entry even though the SELECT-list fix was itself correct.
    """
    v = SQLValidator(domain="audit")
    sql = (
        "SELECT acyc.name AS cycle_name, acyc.status AS cycle_status, "
        "COUNT(DISTINCT ap.id) AS plan_count "
        "FROM audit_cycle acyc "
        "LEFT JOIN audit_plan ap ON ap.cycle_id = acyc.id AND ap.active = true "
        "WHERE acyc.status = 'ONGOING' AND acyc.active = true "
        "GROUP BY acyc.id, acyc.name, acyc.status;"
    )
    question = "count of audit plans in the current audit cycle"

    fixed, mutations = v.strip_hallucinated_no_substitute_columns(sql)
    assert "acyc.name" not in fixed
    assert any("GROUP BY" in m for m in mutations)
    result = v.validate(fixed, question)
    assert result.passed, [str(i) for i in result.hard_fails]


def test_also_cleans_up_orphaned_order_by_reference():
    v = SQLValidator(domain="audit")
    sql = (
        "SELECT acyc.status FROM audit_cycle acyc "
        "WHERE acyc.active = true ORDER BY acyc.name, acyc.status;"
    )
    fixed, mutations = v.strip_hallucinated_no_substitute_columns(sql)
    assert "acyc.name" not in fixed
    assert any("ORDER BY" in m for m in mutations)
    result = v.validate(fixed, "list audit cycles")
    assert result.passed, [str(i) for i in result.hard_fails]
