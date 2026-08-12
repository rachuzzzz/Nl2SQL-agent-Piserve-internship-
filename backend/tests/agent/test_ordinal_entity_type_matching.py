"""
Regression tests for type-gated ordinal resolution — the fix for the
confirmed-live bug (qid=081c7a81, and a same-day manual replay against the
real DB/LLM backend): "list all cycles" (6 rows) -> "for the 4th one... show
portfolios" (correctly resolved) -> "list all audit plans under the 6th
auditor from the list". No auditor list was ever shown anywhere in that
conversation; the old resolver silently grabbed the most recent list turn
(the cycles list) and resolved "the 6th" to a cycle's UUID as if it were an
auditor's id — a well-formed but empty result, wrong with no error.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn
from src.agent.orchestrator.ordinal import resolve_ordinal_reference

_CYCLE_TITLES = [f"cycle {i}" for i in range(1, 7)]
_CYCLE_IDS = [f"cccc000{i}-0000-0000-0000-00000000000{i}" for i in range(1, 7)]

_AUDITOR_NAMES = ["Amir A", "Bilal B", "Chandra C", "Deepa D", "Elena E", "Farid F"]
_AUDITOR_6TH_ID = "bbbb0006-0000-0000-0000-000000000006"
_AUDITOR_IDS = [f"bbbb000{i}-0000-0000-0000-00000000000{i}" for i in range(1, 7)]


def _cycles_turn():
    return ConversationTurn(
        question="list all cycles", answer="6 cycles found", tool_used="generate_sql",
        total_count=6, columns=["status"],
        row_titles=list(_CYCLE_TITLES), row_ids=list(_CYCLE_IDS),
        row_id_columns=["id"] * 6, row_entity_type="cycle",
    )


def _auditors_turn():
    return ConversationTurn(
        question="list all auditors", answer="6 auditors found", tool_used="generate_sql",
        total_count=6, columns=["auditor_name"],
        row_titles=list(_AUDITOR_NAMES), row_ids=list(_AUDITOR_IDS),
        row_id_columns=["id"] * 6, row_entity_type="person",
    )


def test_bug_repro_mismatched_noun_requests_clarification_not_wrong_resolution():
    """Only a CYCLES list exists; the entity_type_hint names 'auditor'.
    Must clarify, never silently resolve against the cycles list."""
    session = ConversationSession()
    session.add_turn(_cycles_turn())
    r = resolve_ordinal_reference(session, "audit", position=6, entity_type_hint="auditor")
    result = r["result"]
    assert result["resolved"] is False
    assert result["needs_clarification"] is True
    assert result["target_id"] not in _CYCLE_IDS  # must never leak a cycle UUID as if it were the auditor


def test_positive_case_resolves_against_matching_type_list_further_back():
    """A real auditors list exists, but NOT as the most recent listing — a
    cycles list was shown after it. Must search back far enough to find the
    auditors list, not just check the most recent turn."""
    session = ConversationSession()
    session.add_turn(_auditors_turn())
    session.add_turn(_cycles_turn())
    r = resolve_ordinal_reference(session, "audit", position=6, entity_type_hint="auditor")
    result = r["result"]
    assert result["resolved"] is True
    assert result["target_id"] == _AUDITOR_6TH_ID
    assert result["target_id"] not in _CYCLE_IDS


def test_bare_ordinal_behavior_unchanged():
    """No hint given ('the 4th one') — must resolve against the most recent
    list turn exactly as before, ignoring row_entity_type entirely."""
    session = ConversationSession()
    session.add_turn(_auditors_turn())
    session.add_turn(_cycles_turn())
    r = resolve_ordinal_reference(session, "audit", position=4, entity_type_hint=None)
    result = r["result"]
    assert result["resolved"] is True
    assert result["target_id"] == _CYCLE_IDS[3]  # most recent list = cycles, unaffected by type


def test_no_matching_type_list_and_no_noun_falls_through_silently():
    """Pre-existing gap, deliberately untouched: bare ordinal with NO
    row_titles anywhere -> no clarification, no resolution, silent skip."""
    session = ConversationSession()
    ctx = session.build_context_block(
        current_question="what about the 3rd one",
        current_domain="audit",
    )
    assert "ORDINAL RESOLVED" not in ctx
    assert "NEEDS CLARIFICATION" not in ctx


def test_snake_case_table_name_hint_actually_type_checks():
    """Confirmed live 2026-08-05: driving the real orchestrator end-to-end
    (real qwen3.5:9b, real DB) for "list all audit plans" -> "for the first
    one show all corrective actions raised", the model sent
    entity_type_hint="audit_plan" (echoing the schema's real table name,
    not "plan" or "audit plan"). _normalize_entity_type_hint's alias table
    didn't cover the underscored table-name form, so the hint silently
    failed to normalize and resolution fell through to the untyped bare-
    ordinal path — it happened to land on the right answer only because
    there was exactly one prior turn with row_titles in that session, not
    because the type check actually ran. With a second, wrong-type turn in
    history (as here), the old behavior would have risked matching the
    wrong list; the fix (normalizing underscores to spaces + a substring
    fallback) must resolve against the PLANS turn specifically."""
    session = ConversationSession()
    session.add_turn(_cycles_turn())
    session.add_turn(ConversationTurn(
        question="list all audit plans", answer="3 plans found", tool_used="generate_sql",
        total_count=3, columns=["title"],
        row_titles=["Plan A", "Plan B", "Plan C"],
        row_ids=["plan-a", "plan-b", "plan-c"],
        row_id_columns=["id"] * 3, row_entity_type="plan",
    ))
    r = resolve_ordinal_reference(session, "audit", position=1, entity_type_hint="audit_plan")
    result = r["result"]
    assert result["resolved"] is True
    assert result["entity_type"] == "plan"
    assert result["target_id"] == "plan-a"
    for cid in _CYCLE_IDS:
        assert result["target_id"] != cid
