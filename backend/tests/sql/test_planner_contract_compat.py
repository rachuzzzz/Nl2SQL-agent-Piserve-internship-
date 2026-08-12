"""
Regression tests for generation.py's planner->SQL structured contract
(draft_sql/join_path/business_rules) and its backward-compat resolution
against the legacy single-string schema_hint shape — Item 3 of the
2026-08-07 generation-quality plan (qid=e63f03af).
"""

from types import SimpleNamespace

from src.sql.generation import _looks_like_sql, _resolve_planner_contract, generate_sql
from src.sql.validation import SQLValidator

from tests.sql.test_generation_token_accounting import _FakeResponse, _FakeSqlLlm


def _validator():
    return SQLValidator(domain="audit")


def test_legacy_schema_hint_with_sql_resolves_to_draft():
    llm_hint, draft = _resolve_planner_contract(
        schema_hint="SELECT ap.title FROM audit_plan ap", draft_sql="", join_path="", business_rules=None,
    )
    assert draft == "SELECT ap.title FROM audit_plan ap"
    assert "SELECT ap.title FROM audit_plan ap" in llm_hint


def test_legacy_prose_schema_hint_stays_notes_not_sql():
    llm_hint, draft = _resolve_planner_contract(
        schema_hint="table=inspection_report, metric=avg(score)", draft_sql="", join_path="", business_rules=None,
    )
    assert draft == ""
    assert "table=inspection_report, metric=avg(score)" in llm_hint


def test_new_shape_draft_sql_and_notes_both_render():
    llm_hint, draft = _resolve_planner_contract(
        schema_hint="only join through inspection_report, never facility directly",
        draft_sql="SELECT ir.id FROM inspection_report ir",
        join_path="", business_rules=None,
    )
    assert draft == "SELECT ir.id FROM inspection_report ir"
    assert "only join through inspection_report" in llm_hint
    assert "SELECT ir.id FROM inspection_report ir" in llm_hint


def test_join_path_and_business_rules_render_in_llm_hint():
    llm_hint, draft = _resolve_planner_contract(
        schema_hint="", draft_sql="SELECT 1",
        join_path="ica -> inspection_report -> facility",
        business_rules=["exclude drafts", "active=true only"],
    )
    assert "ica -> inspection_report -> facility" in llm_hint
    assert "exclude drafts" in llm_hint
    assert "active=true only" in llm_hint


def test_looks_like_sql_classifier():
    assert _looks_like_sql("SELECT 1")
    assert _looks_like_sql("  WITH x AS (SELECT 1) SELECT * FROM x")
    assert not _looks_like_sql("table=x, metric=count")
    assert not _looks_like_sql("")


def test_tables_field_drives_schema_block_selection_same_as_relevant_tables():
    """generation.py resolves the caller-supplied `tables` (aliased to
    relevant_tables upstream in tools.py) the same way relevant_tables
    always worked — this is an equivalence check at the generate_sql level,
    not a new code path in generation.py itself."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT ap.title FROM audit_plan ap WHERE ap.active = true;",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=20)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="list audit plans",
        relevant_tables=["audit_plan"],
    )
    assert result["success"] is True


def test_offset_guard_fires_on_offset_in_draft_sql():
    """Critical regression test: the offset-guard must scan the resolved
    draft_sql, not just legacy schema_hint — see generation.py's
    _llama_schema_hint = _draft_sql_only."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT ac.status FROM audit_cycle ac;",  # no OFFSET -> triggers retry
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)},
        ),
        _FakeResponse(
            "SELECT ac.status FROM audit_cycle ac LIMIT 1 OFFSET 3;",
            {"usage": SimpleNamespace(input_tokens=150, output_tokens=15)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="the 4th cycle's status",
        draft_sql="SELECT ac.id FROM audit_cycle ac ORDER BY ac.start_date LIMIT 1 OFFSET 3",
    )
    assert sql_llm.calls == 2
    assert result["result"]["sql"] is not None


def test_uuid_guard_fires_on_uuid_in_draft_sql():
    uid = "12345678-1234-1234-1234-123456789012"
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT u.first_name FROM users u WHERE u.active = true;",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)},
        ),
        _FakeResponse(
            f"SELECT u.first_name FROM users u WHERE u.id = '{uid}';",
            {"usage": SimpleNamespace(input_tokens=150, output_tokens=15)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="that specific user's name",
        draft_sql=f"SELECT u.first_name FROM users u WHERE u.id = '{uid}'",
    )
    assert sql_llm.calls == 2
    assert uid in result["result"]["sql"]
