"""
Regression tests for generation.py's _diagnostics attribution data — Item 3
of the 2026-08-07 generation-quality plan (qid=e63f03af). Makes "did the
planner or the coder model make the mistake" answerable from one log line
instead of manually diffing two prompt blobs.
"""

from types import SimpleNamespace

from src.retrieval.examples import SeedExampleIndex
from src.sql.generation import generate_sql
from src.sql.validation import SQLValidator

from tests.retrieval.test_seed_example_index_skip_top_n import _make_index, _QUERY
from tests.sql.test_generation_token_accounting import _FakeResponse, _FakeSqlLlm


def _validator():
    return SQLValidator(domain="audit")


def test_diagnostics_reports_table_drift():
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT arep.id FROM audit_report arep;",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="anything",
        relevant_tables=["inspection_report", "users"],
    )
    diag = result["result"]["_diagnostics"]
    assert diag["planner_tables"] == ["inspection_report", "users"]
    assert diag["generated_tables"] == ["audit_report"]
    assert diag["table_drift"] == ["audit_report"]


def test_diagnostics_no_drift_when_tables_match():
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT ir.id FROM inspection_report ir;",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="anything",
        relevant_tables=["inspection_report"],
    )
    diag = result["result"]["_diagnostics"]
    assert diag["table_drift"] == []


def test_diagnostics_reports_primary_example_key():
    index = _make_index(3)
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT 1;",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question=_QUERY,
        _seed_index=index,
    )
    diag = result["result"]["_diagnostics"]
    expected_key = SeedExampleIndex._example_key({"nl": "example zero nl text", "sql": "SELECT 0"})
    assert diag["primary_example_key"] == expected_key


def test_diagnostics_reports_draft_sql_supplied_flag():
    sql_llm = _FakeSqlLlm([
        _FakeResponse("SELECT 1;", {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)}),
    ])
    result_with = generate_sql(sql_llm, _validator(), question="q", draft_sql="SELECT 1")
    assert result_with["result"]["_diagnostics"]["draft_sql_supplied"] is True

    sql_llm2 = _FakeSqlLlm([
        _FakeResponse("SELECT 1;", {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)}),
    ])
    result_without = generate_sql(sql_llm2, _validator(), question="q")
    assert result_without["result"]["_diagnostics"]["draft_sql_supplied"] is False


def test_diagnostics_never_appears_in_llm_facing_errors_or_warnings():
    """Validator-poisoning guard: _diagnostics must never leak into
    validation["errors"]/["warnings"] — the same rule sem_warns already
    follows (see generation.py's comment on the return dict)."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse("SELECT 1;", {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)}),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="anything",
        relevant_tables=["a", "b"], draft_sql="SELECT * FROM c",
    )
    validation = result["result"]["validation"]
    combined = " ".join(validation.get("errors", []) + validation.get("warnings", []))
    assert "_diagnostics" not in combined
    assert "table_drift" not in combined
    assert "primary_example_key" not in combined
