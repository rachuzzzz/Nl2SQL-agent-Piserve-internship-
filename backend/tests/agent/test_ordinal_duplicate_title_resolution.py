"""
Regression test for the exact live-caught bug: a 7-plan listing for one
ONGOING cycle contains 'Delegation testing' TWICE (two different plans,
different audit_type/category, different UUIDs) at list positions 3 and 6.

Before the row-identity fix (M1-M3), an ordinal follow-up like "the third
one" resolved to the STRING 'Delegation testing', and a subsequent
WHERE ap.title = 'Delegation testing' filter would silently match BOTH
rows, merging two different plans' data into one answer.

Ported onto resolve_ordinal_reference (agent/orchestrator/ordinal.py) —
the LLM now supplies an explicit 1-based `position` directly as a tool
argument instead of the retired session.py regex parsing an ordinal word
out of question text; the resolution logic itself (real row id over
ambiguous title, duplicate-title WARNING) is otherwise unchanged.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn
from src.agent.orchestrator.ordinal import resolve_ordinal_reference

# Real titles/order from the live-caught 7-plan list.
_PLAN_TITLES = [
    "Auto-Generated Plan – External Audit – FLS Audit",
    "Delegation retirement plan",
    "Delegation testing",       # position 3 (index 2) — plan A
    "Delegation plan -2",
    "Delegation plan -1",
    "Delegation testing",       # position 6 (index 5) — plan B, SAME title
    "Delegation testing - First party",
]
_PLAN_A_ID = "e5ba20a2-3192-45ab-8215-7d548b626e1b"
_PLAN_B_ID = "cf5a8da3-f03b-4192-aa2f-1ee43443e3bc"
_ROW_IDS = ["", "", _PLAN_A_ID, "", "", _PLAN_B_ID, ""]


def _session_after_plan_listing() -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="list all plans under this cycle",
        answer="7 plans found",
        tool_used="generate_sql",
        total_count=7,
        columns=["title", "status"],
        row_titles=list(_PLAN_TITLES),
        row_ids=list(_ROW_IDS),
    ))
    return session


def test_third_one_resolves_to_the_specific_duplicate_titled_plan():
    session = _session_after_plan_listing()
    r = resolve_ordinal_reference(session, "audit", position=3, entity_type_hint=None)
    result = r["result"]
    assert result["resolved"] is True
    assert result["target_id"] == _PLAN_A_ID
    assert result["target_id"] != _PLAN_B_ID
    assert result["target_column"] == "id"
    assert result["warning"] is not None
    assert "MORE THAN ONCE" in result["warning"]


def test_sixth_one_resolves_to_the_other_duplicate_titled_plan():
    # Swap plan A/B into positions 1 and 2 to confirm POSITION, not title,
    # drives resolution.
    session2 = ConversationSession()
    session2.add_turn(ConversationTurn(
        question="list all plans under this cycle",
        answer="7 plans found",
        tool_used="generate_sql",
        total_count=2,
        columns=["title", "status"],
        row_titles=["Delegation testing", "Delegation testing"],
        row_ids=[_PLAN_A_ID, _PLAN_B_ID],
    ))
    r_first = resolve_ordinal_reference(session2, "audit", position=1, entity_type_hint=None)
    r_second = resolve_ordinal_reference(session2, "audit", position=2, entity_type_hint=None)
    assert r_first["result"]["target_id"] == _PLAN_A_ID
    assert r_second["result"]["target_id"] == _PLAN_B_ID


def test_falls_back_to_warning_when_no_id_was_captured():
    """Same duplicate-title shape, but row_ids empty (e.g. the query had a
    GROUP BY and inject_row_identity() skipped it) — must still surface the
    duplicate-detection WARNING, not silently claim a resolved id."""
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="list all plans under this cycle",
        answer="7 plans found",
        tool_used="generate_sql",
        total_count=7,
        columns=["title", "status"],
        row_titles=list(_PLAN_TITLES),
        row_ids=["", "", "", "", "", "", ""],
    ))
    r = resolve_ordinal_reference(session, "audit", position=3, entity_type_hint=None)
    result = r["result"]
    assert result["resolved"] is True
    assert result["target_id"] == ""
    assert result["target_column"] == ""
    assert result["warning"] is not None
    assert "MORE THAN ONCE" in result["warning"]
