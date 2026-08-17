"""
Tests for resolve_selected_rows (agent/orchestrator/row_selection.py) —
the checkbox-selection counterpart to resolve_ordinal_reference. See that
module's docstring for the deliberate deviations from ordinal.py this test
suite exists to pin down: target turns[-1] ONLY (never search backward),
and any single invalid/missing-id/ambiguous item hard-fails the WHOLE
selection rather than resolving the good ones and dropping/guessing the
bad one.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn
from src.agent.orchestrator.row_selection import resolve_selected_rows

_SCHEDULE_TITLES = ["Schedule A", "Schedule B", "Schedule C", "Schedule D"]
_SCHEDULE_IDS = [
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
    "33333333-3333-3333-3333-333333333333",
    "44444444-4444-4444-4444-444444444444",
]


def _turn(**overrides) -> ConversationTurn:
    defaults = dict(
        question="list all audit schedules",
        answer="4 schedules found",
        tool_used="generate_sql",
        total_count=4,
        columns=["schedule_status"],
        row_titles=list(_SCHEDULE_TITLES),
        row_ids=list(_SCHEDULE_IDS),
        row_id_columns=["id"] * 4,
        row_entity_type="schedule",
    )
    defaults.update(overrides)
    return ConversationTurn(**defaults)


def _session_after_listing(**overrides) -> ConversationSession:
    session = ConversationSession()
    session.add_turn(_turn(**overrides))
    return session


def test_resolves_multiple_selected_positions():
    session = _session_after_listing()
    session.pending_selected_positions = [1, 3]
    r = resolve_selected_rows(session, "audit")
    result = r["result"]
    assert result["resolved"] is True
    assert result["needs_clarification"] is False
    assert result["positions"] == [1, 3]
    assert result["display_titles"] == ["Schedule A", "Schedule C"]
    assert result["target_ids"] == [_SCHEDULE_IDS[0], _SCHEDULE_IDS[2]]
    assert result["target_column"] == "id"
    assert result["target_table"] == "audit_schedule"
    assert result["entity_type"] == "schedule"


def test_no_selection_needs_clarification():
    session = _session_after_listing()
    session.pending_selected_positions = []
    r = resolve_selected_rows(session, "audit")
    assert r["result"]["needs_clarification"] is True


def test_out_of_range_position_hard_fails_whole_selection():
    session = _session_after_listing()
    session.pending_selected_positions = [1, 99]
    r = resolve_selected_rows(session, "audit")
    result = r["result"]
    assert result["resolved"] is False
    assert result["needs_clarification"] is True
    assert "99" in result["clarification_message"]
    # The valid position (1) must NOT have been silently resolved either.
    assert result["target_ids"] == []


def test_stale_turn_never_falls_back_to_an_older_list():
    """The most recent turn has NO row_titles (e.g. a get_answer_stats
    reply) while an earlier turn does — unlike resolve_ordinal_reference's
    backward search, checkbox positions are physically tied to whatever
    table is on screen right NOW, so this must hard-fail rather than
    silently resolving against the older, wrong table."""
    session = ConversationSession()
    session.add_turn(_turn())  # older turn WITH row_titles
    session.add_turn(ConversationTurn(
        question="what's the average score",
        answer="82.5",
        tool_used="get_answer_stats",
        row_titles=[],
        row_ids=[],
    ))
    session.pending_selected_positions = [1]
    r = resolve_selected_rows(session, "audit")
    result = r["result"]
    assert result["resolved"] is False
    assert result["needs_clarification"] is True
    assert result["target_ids"] == []


def test_missing_target_id_hard_fails_no_title_fallback():
    """A GROUP BY turn with no identity injection — row_ids all empty.
    Unlike resolve_ordinal_reference's single-pick title fallback, a
    multi-select must never fall back to WHERE title IN (...), since that
    risks silently pulling in unchecked sibling rows sharing a title."""
    session = _session_after_listing(row_ids=["", "", "", ""])
    session.pending_selected_positions = [1, 2]
    r = resolve_selected_rows(session, "audit")
    result = r["result"]
    assert result["resolved"] is False
    assert result["needs_clarification"] is True
    assert result["target_ids"] == []


def test_duplicate_title_among_unselected_siblings_warns_but_resolves():
    """Position 1's title ('Delegation testing') also appears at position 3,
    which the user did NOT select. Since ids are available, WHERE id IN
    (...) is exact regardless — this must resolve normally with a
    non-blocking warning, not hard-fail. (A prior version of this function
    hard-failed here; live testing showed audit_corrective_action rows
    routinely share their displayed corrective_action_id, which made
    ordinary multi-row corrective-action selections unusable.)"""
    session = _session_after_listing(
        row_titles=["Delegation testing", "Schedule B", "Delegation testing", "Schedule D"],
    )
    session.pending_selected_positions = [1]
    r = resolve_selected_rows(session, "audit")
    result = r["result"]
    assert result["resolved"] is True
    assert result["needs_clarification"] is False
    assert result["target_ids"] == [_SCHEDULE_IDS[0]]
    assert result["duplicate_title_positions"] == [1]


def test_no_turns_at_all_needs_clarification():
    session = ConversationSession()
    session.pending_selected_positions = [1]
    r = resolve_selected_rows(session, "audit")
    assert r["result"]["needs_clarification"] is True


# ── build_context_block() ordering — 2026-08-07 fix ─────────────────────────
# Confirmed live: SELECTED ROWS used to render AFTER the ACTIVE CONTEXT
# "this is the COMPLETE result set... Filter by X IN (...)" block, which gave
# the model a ready-made filter needing no extra tool call — the model took
# that over the extra-step SELECTED ROWS instruction and silently ignored a
# real checkbox selection. SELECTED ROWS now renders FIRST, with an explicit
# priority statement.

def test_selected_rows_renders_before_active_context_complete_set_block():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show details those", answer="5 audits found", tool_used="execute_sql",
        sql="SELECT arep.audit_id FROM audit_report arep WHERE arep.status IN ('CLOSED','CLOSED_WITH_EXCEPTION')",
        total_count=5, audit_ids=["A1", "A2", "A3", "A4", "A5"],
        row_titles=["A1", "A2", "A3", "A4", "A5"], domain="audit",
    ))
    session.pending_selected_positions = [1, 3]

    ctx = session.build_context_block("internal stakeholders for these", current_domain="audit")

    assert "SELECTED ROWS" in ctx
    assert "CONTEXT" in ctx
    assert ctx.index("SELECTED ROWS") < ctx.index("CONTEXT")
    assert "TAKES PRIORITY" in ctx


def test_selected_rows_absent_when_nothing_pending():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show details those", answer="5 audits found", tool_used="execute_sql",
        sql="SELECT 1", total_count=5, audit_ids=["A1"], row_titles=["A1"], domain="audit",
    ))
    session.pending_selected_positions = []

    ctx = session.build_context_block("internal stakeholders for these", current_domain="audit")
    assert "SELECTED ROWS" not in ctx
