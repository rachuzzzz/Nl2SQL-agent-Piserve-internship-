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


def test_skips_group_by_query_when_id_not_grouped():
    """GROUP BY ap.title (not ap.id) — injecting ap.id here would be a real
    Postgres 'column must appear in the GROUP BY clause' error, so this
    must still skip."""
    sql = "SELECT ap.title, COUNT(*) AS cnt FROM audit_plan ap GROUP BY ap.title;"
    result_sql, mutations = _inject(sql)
    assert result_sql == sql
    assert mutations == []


def test_injects_group_by_query_when_id_already_grouped():
    """Confirmed live 2026-08-03: "list all auditors" (grouped by
    u.id/first_name/last_name, with a COUNT(schedule) aggregate) got NO
    id-injection at all under the old blanket "skip any GROUP BY" rule,
    despite u.id being safely groupable (already in the GROUP BY clause) —
    a later "the 6th auditor" ordinal follow-up had no id to resolve to and
    fell back to re-searching by name, surfacing unrelated same-named
    people instead of the specific auditor already shown. Since ap.id is
    ALREADY in the GROUP BY here, adding a second SELECT expression
    referencing that same grouped column is always valid SQL — must inject
    normally, not skip."""
    sql = "SELECT ap.title, COUNT(*) AS cnt FROM audit_plan ap GROUP BY ap.id, ap.title;"
    result_sql, mutations = _inject(sql)
    assert "ap.id AS title__id" in result_sql
    assert len(mutations) == 1


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


def test_injects_cycle_composite_id_for_audit_cycle():
    """audit_cycle has NO name/title column at all — the display value is a
    synthetic composite of 3 separate SELECT expressions (start_date,
    end_date, status), so this can't reuse ROW_IDENTITY_TABLES' single-column
    matching. Confirmed live 2026-08-03 (qid=f2ec1739/f13ab827/4d1d0bf4): an
    ordinal follow-up ("the 4th one") to a cycle listing had nothing to
    resolve against, and fell back to the SQL-gen model hallucinating a
    fictitious column trying to implement LIMIT 1 OFFSET N itself."""
    sql = "SELECT acyc.start_date, acyc.end_date, acyc.status FROM audit_cycle acyc WHERE acyc.active = true;"
    result_sql, mutations = _inject(sql)
    assert "acyc.id AS acyc_cycle_row__id" in result_sql
    assert len(mutations) == 1


def test_injects_cycle_composite_id_for_inspection_cycle():
    """Same gap, same fix, inspection domain's cycle table — fixed
    symmetrically rather than leaving a known-identical gap on purpose."""
    sql = "SELECT ic.start_date, ic.end_date, ic.status FROM inspection_cycle ic WHERE ic.active = true;"
    result_sql, mutations = _inject(sql)
    assert "ic.id AS ic_cycle_row__id" in result_sql
    assert len(mutations) == 1


def test_skips_cycle_injection_when_status_missing():
    """All three of start_date/end_date/status are required — a query only
    selecting 2 of the 3 doesn't get the composite display injected, since
    synthesis.py's row_titles capture needs all three to build the display
    string."""
    sql = "SELECT acyc.start_date, acyc.end_date FROM audit_cycle acyc WHERE acyc.active = true;"
    result_sql, mutations = _inject(sql)
    assert result_sql == sql
    assert mutations == []


def test_cycle_injection_fires_on_unaliased_single_table_query():
    """Confirmed live 2026-08-03 (qid=ca987d9a): DeepSeek sometimes emits a
    single-table query with NO alias at all — `SELECT id, start_date,
    end_date, due_date, status FROM audit_cycle WHERE active = true ...` —
    leaving every column's .table empty. Since there's only one table in the
    whole query, an unqualified column can only ever mean that table; this
    must still fire the injection instead of silently skipping (which is
    what happened live: row_titles stayed empty for that turn, and both
    follow-up ordinal references had to fall back to the SQL-gen model
    guessing at LIMIT/OFFSET or ROW_NUMBER() itself)."""
    sql = "SELECT id, start_date, end_date, due_date, status FROM audit_cycle WHERE active = true ORDER BY start_date DESC LIMIT 100;"
    result_sql, mutations = _inject(sql)
    assert "audit_cycle.id AS audit_cycle_cycle_row__id" in result_sql
    assert len(mutations) == 1


def test_unaliased_column_skipped_when_query_is_ambiguous():
    """The single-table fallback must NOT apply when 2+ tables are present —
    an unqualified column there could belong to any of them, so col.table is
    still required."""
    sql = (
        "SELECT title, status FROM audit_plan ap "
        "JOIN audit_cycle acyc ON ap.cycle_id = acyc.id;"
    )
    result_sql, mutations = _inject(sql)
    assert result_sql == sql
    assert mutations == []


def test_users_composite_fires_on_unaliased_single_table_query():
    """Same fallback, users branch — an unaliased `FROM users` with a bare
    `first_name || ' ' || last_name` concat must still get the id injected."""
    sql = "SELECT first_name || ' ' || last_name AS auditor_name FROM users WHERE active = true;"
    result_sql, mutations = _inject(sql)
    assert "users.id AS auditor_name__id" in result_sql
    assert len(mutations) == 1


def test_injects_users_composite_when_group_by_includes_id():
    """The exact real-world failing shape: 'list all auditors' as a GROUP BY
    with a COUNT(schedule) aggregate — u.id is already grouped, so injecting
    the composite-name id sibling is safe."""
    sql = (
        "SELECT u.first_name || ' ' || u.last_name AS name, COUNT(asch.id) AS schedule_count "
        "FROM users u LEFT JOIN audit_schedule asch ON asch.auditor_id = u.id "
        "GROUP BY u.id, u.first_name, u.last_name;"
    )
    result_sql, mutations = _inject(sql)
    assert "u.id AS name__id" in result_sql
    assert len(mutations) == 1


def test_skips_users_composite_when_group_by_excludes_id():
    sql = (
        "SELECT u.first_name || ' ' || u.last_name AS name, COUNT(*) AS cnt "
        "FROM users u GROUP BY u.first_name, u.last_name;"
    )
    result_sql, mutations = _inject(sql)
    assert result_sql == sql
    assert mutations == []


def test_injects_cycle_composite_when_group_by_includes_id():
    sql = (
        "SELECT acyc.start_date, acyc.end_date, acyc.status, COUNT(ap.id) AS plan_count "
        "FROM audit_cycle acyc LEFT JOIN audit_plan ap ON ap.cycle_id = acyc.id "
        "GROUP BY acyc.id, acyc.start_date, acyc.end_date, acyc.status;"
    )
    result_sql, mutations = _inject(sql)
    assert "acyc.id AS acyc_cycle_row__id" in result_sql
    assert len(mutations) == 1


def test_cycle_injection_fires_when_only_touching_cycle_table():
    """Regression test for the early-return gate at the top of
    inject_row_identity(): a query that touches ONLY audit_cycle — no
    ROW_IDENTITY_TABLES table, no users — must still reach the cycle branch
    instead of returning early. The gate must OR in _CYCLE_TABLES membership,
    not just ROW_IDENTITY_TABLES/users."""
    sql = (
        "SELECT acyc.start_date, acyc.end_date, acyc.status FROM audit_cycle acyc "
        "WHERE acyc.active = true ORDER BY acyc.start_date LIMIT 100;"
    )
    result_sql, mutations = _inject(sql)
    assert "acyc.id AS acyc_cycle_row__id" in result_sql
    assert len(mutations) == 1
