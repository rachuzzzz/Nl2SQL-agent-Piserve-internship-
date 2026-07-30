"""
Regression tests for src/sql/validation/identity.py's
strip_bare_uuid_pk_select() — deterministic fix for a bare `<alias>.id`
selected for display, the single most common validation retry in the audit
domain (confirmed live via eval_audit.py, reproducible at temperature=0.0:
PLAN-02, STAT-02, CYCLE-01, CYCLE-05, WORKFLOW-08 all cost an extra LLM
round trip for exactly this reason).

Verified NOT caused by conflicting guidance in our own prompts:
AUDIT_USEFUL_COLUMNS (schema_introspector.py) already omits "id" from every
table's curated column list, and no seed/worked example anywhere selects a
bare id — this is the model's own generic SQL-writing prior, not a
self-inflicted contradiction.
"""

from src.sql.validation.engine import SQLValidator


def _validator():
    return SQLValidator(domain="audit")


def test_removes_bare_id_when_display_column_already_present():
    v = _validator()
    sql = (
        "SELECT ap.id, ap.title, ap.status FROM audit_plan ap "
        "WHERE ap.status = 'DRAFT' AND ap.active = true;"
    )
    fixed, mutations = v.strip_bare_uuid_pk_select(sql)
    assert "ap.id" not in fixed.split("FROM")[0]
    assert "ap.title" in fixed
    assert len(mutations) == 1
    result = v.validate(fixed, "audit plans still in draft status")
    assert result.passed, [str(i) for i in result.hard_fails]


def test_replaces_bare_id_with_display_column_when_absent():
    v = _validator()
    sql = "SELECT ap.id, ap.status FROM audit_plan ap WHERE ap.active = true;"
    fixed, mutations = v.strip_bare_uuid_pk_select(sql)
    assert "ap.title" in fixed
    assert "ap.id" not in fixed.split("FROM")[0]
    assert len(mutations) == 1


def test_removes_bare_id_with_no_display_substitute():
    """audit_cycle (acyc) and auditor_portfolio (po) have NO display column
    at all — the only safe fix is removal, matching the BLOCK rule's own
    'identify this row by its other columns instead' message."""
    v = _validator()
    sql = (
        "SELECT acyc.id, acyc.start_date, acyc.end_date, acyc.status "
        "FROM audit_cycle acyc WHERE acyc.active = true;"
    )
    fixed, mutations = v.strip_bare_uuid_pk_select(sql)
    assert "acyc.id" not in fixed.split("FROM")[0]
    assert len(mutations) == 1
    result = v.validate(fixed, "show all audit cycles")
    assert result.passed, [str(i) for i in result.hard_fails]


def test_skips_when_removal_would_empty_the_select_list():
    """Pathological case: acyc.id is the ONLY expression and acyc has no
    display substitute — removing it would produce invalid SQL (empty
    SELECT), so this must be left untouched, falling back to the existing
    BLOCK+retry safety net instead of risking corrupt SQL."""
    v = _validator()
    sql = "SELECT acyc.id FROM audit_cycle acyc WHERE acyc.active = true;"
    fixed, mutations = v.strip_bare_uuid_pk_select(sql)
    assert mutations == []
    assert fixed == sql


def test_swaps_sole_bare_id_for_its_display_column():
    v = _validator()
    sql = "SELECT ap.id FROM audit_plan ap WHERE ap.active = true;"
    fixed, mutations = v.strip_bare_uuid_pk_select(sql)
    assert "ap.title" in fixed
    assert len(mutations) == 1


def test_never_touches_row_identity_injected_columns():
    """A __id-suffixed column is inject_row_identity()'s own legitimate
    output (see identity.py's inject_row_identity docstring) — must never
    be removed or altered by this unrelated fix."""
    v = _validator()
    sql = (
        "SELECT ap.title, ap.id AS title__id FROM audit_plan ap "
        "WHERE ap.active = true;"
    )
    fixed, mutations = v.strip_bare_uuid_pk_select(sql)
    assert mutations == []
    assert fixed == sql


def test_gated_to_audit_domain_only():
    v = SQLValidator(domain="inspection")
    sql = "SELECT ap.id, ap.title FROM audit_plan ap WHERE ap.active = true;"
    fixed, mutations = v.strip_bare_uuid_pk_select(sql)
    assert mutations == []
    assert fixed == sql


def test_full_pipeline_prevents_retry_on_exact_eval_case():
    """End-to-end: PLAN-02's exact shape now passes validate() on the first
    attempt instead of costing a retry."""
    v = _validator()
    raw_sql = (
        "SELECT ap.id, ap.title, ap.status, acat.name AS category_name, "
        "aty.name AS audit_type_name, acyc.start_date AS cycle_start, "
        "acyc.end_date AS cycle_end, acyc.status AS cycle_status "
        "FROM audit_plan ap "
        "LEFT JOIN audit_category acat ON ap.category_id = acat.id "
        "LEFT JOIN audit_type aty ON ap.audit_type_id = aty.id "
        "LEFT JOIN audit_cycle acyc ON ap.cycle_id = acyc.id "
        "WHERE ap.status = 'DRAFT' AND ap.active = true ORDER BY ap.created_on ASC;"
    )
    question = "audit plans still in draft status"

    sql = v.clean_sql(raw_sql)
    sql, _ = v.ast_repair(sql)
    sql, mutations = v.strip_bare_uuid_pk_select(sql)
    result = v.validate(sql, question)

    assert mutations
    assert result.passed, [str(i) for i in result.hard_fails]
