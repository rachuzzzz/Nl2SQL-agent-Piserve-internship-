"""
Regression test for the confirmed-live ordinal-resolution failure on
audit_cycle listings (qid=f2ec1739, qid=f13ab827, qid=4d1d0bf4).

audit_cycle has no name/title column at all — before this fix, a "list all
cycles" turn left row_titles empty (none of synthesis.py's column-name
checks matched a cycle row), so build_context_block()'s ordinal resolver had
nothing to point at for a follow-up like "show portfolios for the 4th one in
the list". The reasoning model fell back to asking the SQL-gen model to
implement "the 4th one" itself via LIMIT 1 OFFSET 3 — something it has never
once been shown a worked example of, and hallucinated on nearly every
attempt (fabricated cycle_id/cycle_number columns, even a nonsensical
`LIKE '%CYCLE_4%'` against a date column). After repeated failures, the
reasoning model gave up and returned a fully UNFILTERED result across all 6
cycles as if it had answered the question — a confidently wrong answer, no
error of any kind.

After the fix (src/sql/validation/identity.py's _CYCLE_TABLES branch +
src/agent/orchestrator/synthesis.py's Priority 1.5 capture), a cycle listing
populates row_titles/row_ids with a synthetic "<start_date> to <end_date>
(<status>)" composite display, and this exact same ordinal-resolution
machinery — already proven correct for audit_plan (see
test_ordinal_duplicate_title_resolution.py) — resolves "the 4th one"
deterministically, with zero LLM guessing required.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn

# Mirrors the real 6-cycle listing shape from the live-caught failure.
_CYCLE_TITLES = [
    "2026-01-01 to 2026-01-31 (COMPLETED)",
    "2026-02-01 to 2026-02-28 (COMPLETED)",
    "2026-03-01 to 2026-03-31 (COMPLETED)",
    "2026-04-01 to 2026-04-30 (ONGOING)",
    "2026-05-01 to 2026-05-31 (UPCOMING)",
    "2026-06-01 to 2026-06-30 (UPCOMING)",
]
_CYCLE_4_ID = "d14d5a29-15de-4671-b8d3-82231f7c3546"
_ROW_IDS = [
    "aaaa0001-0000-0000-0000-000000000001",
    "aaaa0002-0000-0000-0000-000000000002",
    "aaaa0003-0000-0000-0000-000000000003",
    _CYCLE_4_ID,
    "aaaa0005-0000-0000-0000-000000000005",
    "aaaa0006-0000-0000-0000-000000000006",
]


def _session_after_cycle_listing() -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="list all cycles",
        answer="6 cycles found",
        tool_used="generate_sql",
        total_count=6,
        columns=["start_date", "end_date", "status"],
        row_titles=list(_CYCLE_TITLES),
        row_ids=list(_ROW_IDS),
        row_id_columns=["id"] * 6,
    ))
    return session


def test_fourth_one_resolves_to_the_specific_cycle_uuid():
    session = _session_after_cycle_listing()
    ctx = session.build_context_block(
        current_question="show all auditors who submitted portfolios for the 4th one in the list",
        current_domain="audit",
    )
    assert "ORDINAL RESOLVED" in ctx
    assert f"id '{_CYCLE_4_ID}'" in ctx
    assert "WHERE <alias>.id" in ctx
    # None of the other 5 cycle UUIDs should appear
    for other_id in _ROW_IDS:
        if other_id != _CYCLE_4_ID:
            assert other_id not in ctx


def test_does_not_fall_back_to_offset_based_guidance():
    """The whole point of the fix: once ordinal resolution succeeds
    deterministically, nothing in the injected context should nudge the
    model toward a LIMIT/OFFSET-based SQL approach — that's the pattern the
    SQL-gen model has never seen a working example of and hallucinates on
    almost every attempt."""
    session = _session_after_cycle_listing()
    ctx = session.build_context_block(
        current_question="show portfolios for the 4th one in the list",
        current_domain="audit",
    )
    assert "OFFSET" not in ctx.upper()
