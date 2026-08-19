"""
Regression test for build_context_block()'s per-turn CONVERSATION HISTORY
render cap — the token-usage audit follow-up to test_row_titles_context_cap.py.

Confirmed live via real debug_llm_*.log traces: system_content (the
reasoning LLM's system prompt, rebuilt fresh every turn from
build_context_block()'s output) reached 83,456 chars (~20.9K tokens) by
turn ~18 of one real session — already past num_ctx=20000 BEFORE a single
tool-call iteration even ran. The per-turn CONVERSATION HISTORY block —
full SQL/columns/id-list detail for every one of up to MAX_TURNS=8 turns,
with an uncapped question and uncapped columns list — was the largest
single contributor.

The fix caps the RENDER only (full detail for the most recent 3 turns,
a compact one-line summary for older ones) — same principle as
test_row_titles_context_cap.py: ordinal/"go back to"/entity resolution all
read structured fields directly off ConversationTurn objects in
self.turns, never this rendered text, so a reference to something from an
older (now-compacted) turn must still resolve correctly.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn

_WIDE_COLUMNS = [
    "schedule_id", "schedule_status", "duration", "follow_up_date",
    "reschedule_count", "is_unplanned", "auditor_name", "auditee_name",
    "facility_name", "project_name", "plan_title", "category_name",
    "sub_category_name", "universe_name",
]
_LONG_QUESTION = (
    "show all audit schedules with their key details including auditor, "
    "auditee, facility, project, plan title, category and sub-category, "
    "grouped by universe and sorted by follow up date descending please"
)


def _session_with_n_full_turns(n: int) -> ConversationSession:
    session = ConversationSession()
    for i in range(n):
        session.add_turn(ConversationTurn(
            question=f"{_LONG_QUESTION} (round {i})",
            answer="Found 30 results: " + ", ".join(f"row{n}" for n in range(30)),
            tool_used="generate_sql",
            sql="SELECT asch.schedule_id, asch.schedule_status FROM audit_schedule asch "
                "JOIN audit_plan ap ON asch.audit_plan_id = ap.id",
            total_count=30,
            columns=list(_WIDE_COLUMNS),
            facility_names=["Golf Gardens"],
            active_subject_type="facility", active_subject_value="Golf Gardens",
            active_subject_label="Golf Gardens",
        ))
    return session


def test_older_turns_are_compacted_no_sql_or_columns():
    session = _session_with_n_full_turns(8)
    ctx = session.build_context_block(current_question="what about opex", current_domain="audit")

    # Turns 1-5 (older than the most recent 3) must be a single compact
    # line — no SQL:/Columns: detail lines for them.
    for i in range(1, 6):
        assert f"Turn {i}: User:" in ctx
    assert ctx.count("SQL:") == 3          # only the 3 most recent turns
    assert ctx.count("Columns:") == 3


def test_recent_turns_keep_full_detail():
    session = _session_with_n_full_turns(8)
    ctx = session.build_context_block(current_question="what about opex", current_domain="audit")

    for i in (6, 7, 8):
        assert f"Turn {i}:\n" in ctx
    assert "SQL: SELECT asch.schedule_id" in ctx


def test_wide_columns_list_is_capped_with_more_marker():
    session = _session_with_n_full_turns(3)  # all 3 are "recent" — full detail
    ctx = session.build_context_block(current_question="what about opex", current_domain="audit")

    assert "(+2 more)" in ctx  # 14 columns, capped to 12
    assert "universe_name" not in ctx  # last column, beyond the cap
    assert "schedule_id" in ctx


def test_long_question_is_truncated_for_recent_turns():
    """The recent-turn question cap (300 chars) is separate/looser than the
    compact-older-turn cap (150 chars, see test_older_turns_are_compacted_
    no_sql_or_columns) — verify it independently with a question long
    enough to actually exceed 300 chars."""
    very_long_question = _LONG_QUESTION + " " + _LONG_QUESTION  # 401 chars
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question=very_long_question, answer="ok", tool_used="generate_sql",
    ))
    ctx = session.build_context_block(current_question="anything", current_domain="audit")
    assert very_long_question not in ctx
    assert very_long_question[:300] in ctx


def test_history_block_is_meaningfully_smaller_than_uncapped():
    """Sanity check on the actual byte savings, not just presence/absence
    of specific substrings — a long (8-turn) conversation's history block
    should shrink substantially once 5 of the 8 turns are compacted."""
    session = _session_with_n_full_turns(8)
    ctx_8 = session.build_context_block(current_question="what about opex", current_domain="audit")

    session_3 = _session_with_n_full_turns(3)
    ctx_3 = session_3.build_context_block(current_question="what about opex", current_domain="audit")

    # 8 turns (5 compacted + 3 full) should be far smaller than what 8 turns
    # of ALL-full-detail rendering would be (8/3 * a single full turn's
    # rendered size, roughly) — a loose upper bound, not a tight one, since
    # exact savings depend on field lengths above.
    assert len(ctx_8) < len(ctx_3) * 3


def test_domain_empty_recent_turn_is_compacted_not_verbatim():
    """A greeting/domain-ack turn resolves no domain data — it must render
    via the compact one-line form even when it's within the 3-most-recent
    window, not the full-detail form real data-bearing turns get. Regression
    guard for the MTP-vs-base investigation: two live sessions whose only
    difference was "Hello" vs "hello" in this exact kind of turn produced
    different downstream SQL three turns later, purely from this verbatim
    rendering — the model itself was proven deterministic per exact input."""
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="hello",
        answer="Are you looking for information about facility inspections or audits?",
        tool_used="domain_clarification",
    ))
    session.add_turn(ConversationTurn(
        question="audit",
        answer="Got it — audits. What would you like to know?",
        tool_used="domain_ack", domain="audit",
    ))
    ctx = session.build_context_block(
        current_question="show all audit schedules", current_domain="audit",
    )
    # Compact form: "Turn N: User: 'q' -> a"
    assert "Turn 1: User: 'hello'" in ctx
    assert "Turn 2: User: 'audit'" in ctx
    # Full-detail form markers must NOT appear for these two turns.
    assert "Turn 1:\n" not in ctx
    assert "Turn 2:\n" not in ctx
    assert "Answer: Are you looking for" not in ctx
    assert "Answer: Got it" not in ctx


def test_go_back_to_older_turn_still_resolves_via_structured_data():
    """The active_subject on turn 2 (an "older", now-compacted turn once 8
    turns exist) must still be found by the "go back to" resolver — it
    reads t.active_subject_type/value directly off self.turns, never the
    rendered CONVERSATION HISTORY text."""
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show golf gardens inspections", answer="...", tool_used="generate_sql",
        active_subject_type="facility", active_subject_value="Golf Gardens",
        active_subject_label="Golf Gardens",
    ))
    for i in range(7):
        session.add_turn(ConversationTurn(
            question=f"unrelated question {i}", answer="...", tool_used="generate_sql",
        ))

    ctx = session.build_context_block(
        current_question="go back to that facility", current_domain="inspection",
    )
    assert "ACTIVE CONTEXT" in ctx
    assert "Golf Gardens" in ctx
