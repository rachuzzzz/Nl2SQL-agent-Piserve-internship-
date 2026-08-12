"""
Regression tests proving an allowlisted semantic HARD_FAIL actually reaches
generate_sql()'s caller via the SAME retry path a validate() HARD_FAIL
already uses (loop.py reads _vresult/retry_message()) — the 2026-08-07 fix
(debug_llm_20260807_153128). generate_sql() has no retry loop of its own;
this merge is the only way a semantic issue can ever cause a retry.
"""

from types import SimpleNamespace

from src.sql.generation import generate_sql
from src.sql.validation import SQLValidator

from tests.sql.test_generation_token_accounting import _FakeResponse, _FakeSqlLlm


def _validator():
    return SQLValidator(domain="audit")


def test_live_wrong_sql_blocks_and_reaches_errors_and_retry_message():
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT arep.audit_id FROM audit_report arep "
            "WHERE arep.status IN ('SUBMITTED', 'CLOSED', 'CLOSED_WITH_EXCEPTION');",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=20)},
        ),
    ])
    result = generate_sql(sql_llm, _validator(), question="list all conducted audits")
    validation = result["result"]["validation"]
    assert validation["passed"] is False
    assert any("conducted audits" in e for e in validation["errors"])
    assert validation["warnings"] == []
    vresult = validation["_vresult"]
    assert "conducted audits" in vresult.retry_message()


def test_unrelated_soft_warns_survive_the_merge():
    """A HARD_FAIL semantic merge must not clobber validate()'s own soft
    warns or other semantic soft warns."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT arep.audit_id FROM audit_report arep "
            "WHERE arep.status IN ('SUBMITTED', 'CLOSED', 'CLOSED_WITH_EXCEPTION');",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=20)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(),
        question="list all conducted audits and how many completed inspections too",
    )
    validation = result["result"]["validation"]
    # The completed_inspections violation (not allowlisted) should still be
    # a soft warn, not lost.
    assert any("completed" in w.lower() for w in validation["warnings"])


def test_no_id_columns_injected_into_a_blocked_query():
    """The merge must happen BEFORE row-identity injection — a query about
    to be retried/discarded must never get a synthetic __id column."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT arep.audit_id FROM audit_report arep "
            "WHERE arep.status IN ('SUBMITTED', 'CLOSED', 'CLOSED_WITH_EXCEPTION');",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=20)},
        ),
    ])
    result = generate_sql(sql_llm, _validator(), question="list all conducted audits")
    assert "__id" not in result["result"]["sql"]
