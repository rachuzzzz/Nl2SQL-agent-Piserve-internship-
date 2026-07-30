"""
Regression tests for src/sql/validation/identity.py's inject_row_identity().

Confirmed live 2026-07-30: audit_plan.title has NO uniqueness constraint —
two different plans in the same cycle can share a title ('Delegation
testing'). This injection is what lets a later conversational follow-up
("the third one") resolve to a real row id instead of re-deriving one from
a possibly-duplicated display string.
"""

from src.sql.validation.identity import IdentityMixin
from src.sql.validation.engine import SQLValidator


class _Validator(IdentityMixin):
    pass


def test_injected_column_survives_revalidation():
    """
    Confirmed live 2026-07-30: generation.py's pipeline calls
    inject_row_identity() AFTER validator.validate() has already passed on
    the model's own SQL — but execute_sql() (src/sql/execution.py) runs a
    SEPARATE, independent re-validation gate on the SQL it's about to
    actually run against the DB, right before execution. Without an
    exemption in _check_uuid_pk_in_select_ast (ast_checks.py) for the
    __id-suffixed convention, that SECOND validation call HARD_FAILs on the
    injected ap.id AS title__id column — exactly like a genuine "bare .id
    leaked into display" mistake — even though the reasoning LLM never wrote
    it and has no way to remove something it didn't select. Every retry
    regenerates clean SQL, generation.py re-injects the same column, and the
    query fails every iteration up to the retry cap (confirmed live: 9 LLM
    calls, 6 SQL errors, ultimately unsuccessful, for a plain "list all
    plans" question).
    """
    validator = SQLValidator(domain="audit")
    sql, mutations = validator.inject_row_identity(
        "SELECT ap.title, ap.status FROM audit_plan ap WHERE ap.active = true;"
    )
    assert mutations  # sanity: injection actually happened
    assert "title__id" in sql

    # This is the exact re-validation execute_sql() performs before running
    # the SQL against the DB — must NOT fail on the injected column.
    result = validator.validate(sql, "list all plans")
    assert result.passed, [str(i) for i in result.hard_fails]


def test_genuine_bare_id_selection_is_still_blocked():
    """The exemption must be narrow — a real bare `ap.id` selection (no __id
    suffix) is still exactly the mistake UUID_IN_SELECT exists to catch."""
    validator = SQLValidator(domain="audit")
    result = validator.validate(
        "SELECT ap.id, ap.title FROM audit_plan ap WHERE ap.active = true;",
        "list all plans",
    )
    assert not result.passed
    assert any("UUID_IN_SELECT" in str(i) for i in result.hard_fails)


def _inject(sql: str):
    return _Validator().inject_row_identity(sql)


def test_injects_id_for_simple_display_column():
    sql = "SELECT ap.title, ap.status FROM audit_plan ap WHERE ap.active = true;"
    result_sql, mutations = _inject(sql)
    assert "ap.id AS title__id" in result_sql
    assert len(mutations) == 1


def test_injects_id_using_output_alias_not_source_column():
    sql = (
        "SELECT ap.title AS audit_plan_title, ap.status "
        "FROM audit_plan ap WHERE ap.active = true;"
    )
    result_sql, mutations = _inject(sql)
    assert "ap.id AS audit_plan_title__id" in result_sql
    assert "title__id" not in result_sql.replace("audit_plan_title__id", "")
    assert len(mutations) == 1


def test_multi_table_join_injects_per_matching_expression_not_from_table():
    """review JOIN audit_plan, selecting rev.status and ap.title — the FROM
    table is `review` (not in ROW_IDENTITY_TABLES), but ap.title still gets
    its own id injected, independent of which table is FROM vs JOIN."""
    sql = (
        "SELECT ap.title, rev.status, rev.level FROM review rev "
        "JOIN audit_plan ap ON rev.entity_id = ap.id "
        "WHERE rev.status = 'COMPLETED';"
    )
    result_sql, mutations = _inject(sql)
    assert "ap.id AS title__id" in result_sql
    assert len(mutations) == 1


def test_skips_group_by_queries():
    sql = "SELECT ap.title, COUNT(*) AS cnt FROM audit_plan ap GROUP BY ap.title;"
    result_sql, mutations = _inject(sql)
    assert result_sql == sql
    assert mutations == []


def test_skips_when_no_nameable_table_present():
    sql = "SELECT schedule_status, COUNT(*) FROM audit_schedule GROUP BY schedule_status;"
    result_sql, mutations = _inject(sql)
    assert result_sql == sql
    assert mutations == []


def test_no_injection_when_query_selects_no_display_column():
    sql = "SELECT ap.status FROM audit_plan ap WHERE ap.active = true;"
    result_sql, mutations = _inject(sql)
    assert result_sql == sql
    assert mutations == []


def test_skips_when_output_name_already_taken():
    # ap.title's natural target ("title__id") collides with an unrelated
    # column that already happens to output that exact name — skip rather
    # than guess/overwrite. ap.status doesn't match ROW_IDENTITY_TABLES'
    # display column ("title"), so it's not itself an injection candidate.
    sql = "SELECT ap.title, ap.status AS title__id FROM audit_plan ap;"
    result_sql, mutations = _inject(sql)
    assert result_sql == sql
    assert mutations == []


def test_facility_display_column_also_supported():
    sql = "SELECT fac.name, fac.city FROM facility fac WHERE fac.active = true;"
    result_sql, mutations = _inject(sql)
    assert "fac.id AS name__id" in result_sql
    assert len(mutations) == 1


def test_left_joined_nameable_table_still_gets_injected():
    """Unlike cleaner.py's _direct_tables (which excludes LEFT-joined tables
    from active=true injection since they're nullable), row-identity
    injection must NOT exclude them — a LEFT-joined facility's name is just
    as valid a row identity as one from an inner join."""
    sql = (
        "SELECT asch.schedule_id, fac.name FROM audit_schedule asch "
        "LEFT JOIN facility fac ON asch.facility_id = fac.id;"
    )
    result_sql, mutations = _inject(sql)
    assert "asch.id AS schedule_id__id" in result_sql
    assert "fac.id AS name__id" in result_sql
    assert len(mutations) == 2
