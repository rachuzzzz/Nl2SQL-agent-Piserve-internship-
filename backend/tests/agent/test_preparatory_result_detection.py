"""
Regression tests for disambiguation.py's _is_preparatory_result() — the
guard that stops the orchestrator loop from treating an intermediate
identity-resolution result as the final answer.

Two confirmed-live bug shapes covered here:

1. Bare-id-only result (qid=f13ab827, 2026-08-03): a question about
   "portfolios" resolves down to a single id column with nothing else — the
   reasoning model's own plan said "get the id, then use it in a separate
   query", but the loop returned the bare id as if it answered the question.

2. Entity-noun/SQL-table mismatch (qid=081c7a81, 2026-08-03): "list all
   audit plans under the 6th auditor" resolved only the auditor's identity
   (columns=['id','name','name__id']) and never touched audit_plan at all —
   not caught by the bare-id-only check since 'name' is a real display
   column, not an id.
"""

from src.agent.orchestrator.disambiguation import DisambiguationMixin


class _FakeOrchestrator(DisambiguationMixin):
    domain = "audit"


def _result(columns: list, row_count: int) -> dict:
    return {"result": {"columns": columns, "row_count": row_count, "rows": [{}] * row_count}}


def test_bare_id_only_result_is_preparatory():
    orch = _FakeOrchestrator()
    question = "list all portfolios submitted for the 4th one in the list"
    sql = "SELECT id AS cycle_id FROM audit_cycle WHERE active = true ORDER BY start_date LIMIT 1 OFFSET 3;"
    assert orch._is_preparatory_result(question, _result(["cycle_id"], 1), sql) is True


def test_bare_id_only_result_not_preparatory_when_many_rows():
    """A genuine 'list all portfolio ids' request returning 50 rows is not
    an ordinal-resolution artifact — only small id-only results are."""
    orch = _FakeOrchestrator()
    sql = "SELECT id FROM auditor_portfolio;"
    assert orch._is_preparatory_result(
        "list all portfolio ids", _result(["id"], 50), sql
    ) is False


def test_entity_noun_mismatch_is_preparatory():
    """The exact live-caught shape: question asks for audit plans, SQL only
    resolved the auditor's identity (id/name), never touched audit_plan."""
    orch = _FakeOrchestrator()
    question = "list all audit plans under the 6th auditor from the lsit"
    sql = (
        "SELECT u.id, u.first_name || ' ' || u.last_name AS name, "
        "u.id AS name__id FROM users AS u WHERE u.active = TRUE "
        "ORDER BY u.id ASC LIMIT 1 OFFSET 5;"
    )
    assert orch._is_preparatory_result(
        question, _result(["id", "name", "name__id"], 1), sql
    ) is True


def test_entity_noun_match_not_preparatory():
    """Same question, but the SQL actually joins audit_plan — a genuine
    answer, must not be flagged."""
    orch = _FakeOrchestrator()
    question = "list all audit plans under the 6th auditor from the lsit"
    sql = (
        "SELECT ap.title AS plan_title, ap.status AS plan_status "
        "FROM audit_plan ap JOIN users u ON ap.user_id = u.id "
        "WHERE u.id = 'xxx';"
    )
    assert orch._is_preparatory_result(
        question, _result(["plan_title", "plan_status"], 3), sql
    ) is False


def test_no_entity_noun_in_question_not_flagged():
    """A question that never names a specific entity table has nothing to
    compare against — must not misfire."""
    orch = _FakeOrchestrator()
    sql = "SELECT start_date, end_date, status FROM audit_cycle WHERE active = true;"
    assert orch._is_preparatory_result(
        "list all cycles", _result(["start_date", "end_date", "status"], 6), sql
    ) is False


def test_portfolio_question_touching_auditor_portfolio_not_flagged():
    orch = _FakeOrchestrator()
    question = "show all portfolios submitted for the 4th cycle"
    sql = "SELECT po.status, po.submitted_date FROM auditor_portfolio po WHERE po.audit_cycle_id = 'xxx';"
    assert orch._is_preparatory_result(
        question, _result(["status", "submitted_date"], 8), sql
    ) is False


def test_auditor_noun_does_not_trigger_table_membership_check():
    """A follow-up already scoped by a resolved auditor id (no JOIN users
    needed at all) must NOT be flagged as preparatory just because the
    question contains the word 'auditor' — the auditor/inspector entries in
    _entity_noun_tables are deliberately type-only (table=None), consumed
    only by the ordinal noun-type lookup in session.py, not by this
    table-membership check. Otherwise a legitimate, complete answer like
    `WHERE po.auditor_id = '<uuid>'` (which never needs to literally say
    "users" anywhere) would get wrongly flagged as incomplete."""
    orch = _FakeOrchestrator()
    question = "show all portfolios submitted by that auditor"
    sql = "SELECT po.status, po.submitted_date FROM auditor_portfolio po WHERE po.auditor_id = 'xxx';"
    assert orch._is_preparatory_result(
        question, _result(["status", "submitted_date"], 5), sql
    ) is False


def test_inspection_domain_uses_its_own_entity_table_map():
    """Domain-gated: an audit-domain 'audit plans' phrase must not apply
    inspection-domain table names, and vice versa."""
    orch = _FakeOrchestrator()
    orch.domain = "inspection"
    question = "list all mitigative actions under the 3rd one"
    sql = "SELECT u.id, u.first_name AS name FROM users u LIMIT 1 OFFSET 2;"
    assert orch._is_preparatory_result(
        question, _result(["id", "name"], 1), sql
    ) is True
