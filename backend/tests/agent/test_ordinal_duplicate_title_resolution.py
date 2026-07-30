"""
Regression test for the exact live-caught bug: a 7-plan listing for one
ONGOING cycle contains 'Delegation testing' TWICE (two different plans,
different audit_type/category, different UUIDs) at list positions 3 and 6.

Before the row-identity fix (M1-M3), an ordinal follow-up like "the third
one" resolved to the STRING 'Delegation testing', and a subsequent
WHERE ap.title = 'Delegation testing' filter would silently match BOTH
rows, merging two different plans' data into one answer.

After the fix, build_context_block() must resolve "the third one" straight
to the SPECIFIC row's real id (captured via row_ids, paired 1:1 with
row_titles by inject_row_identity()'s __id columns) — not to the
ambiguous title.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn

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
    ctx = session.build_context_block(
        current_question="for the third one what is the review status",
        current_domain="audit",
    )
    assert f"id '{_PLAN_A_ID}'" in ctx
    assert "ORDINAL RESOLVED" in ctx
    assert _PLAN_B_ID not in ctx  # must not conflate with the OTHER duplicate
    assert "WHERE ... = 'Delegation testing'" not in ctx  # no title-based filter


def test_sixth_one_resolves_to_the_other_duplicate_titled_plan():
    session = _session_after_plan_listing()
    # No "sixth" in _ORDINAL_WORD_TO_IDX (only 1st-5th) — use an in-range
    # ordinal instead to confirm position, not just title, drives resolution.
    # Swap plan A/B into positions 1 and 2 for this check.
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
    ctx_first = session2.build_context_block(
        current_question="for the first one what is the review status",
        current_domain="audit",
    )
    ctx_second = session2.build_context_block(
        current_question="for the second one what is the review status",
        current_domain="audit",
    )
    assert f"id '{_PLAN_A_ID}'" in ctx_first and _PLAN_B_ID not in ctx_first
    assert f"id '{_PLAN_B_ID}'" in ctx_second and _PLAN_A_ID not in ctx_second


def test_falls_back_to_warning_when_no_id_was_captured():
    """Same duplicate-title shape, but row_ids empty (e.g. the query had a
    GROUP BY and inject_row_identity() skipped it) — must fall back to the
    duplicate-detection WARNING, not silently produce a title filter."""
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
    ctx = session.build_context_block(
        current_question="for the third one what is the review status",
        current_domain="audit",
    )
    assert "ORDINAL RESOLVED" not in ctx
    assert "ORDINAL REFERENCE DETECTED" in ctx
    assert "WARNING" in ctx
    assert "appears MORE THAN ONCE" in ctx
