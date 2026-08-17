"""
Regression test for build_context_block()'s row_titles preview cap.

Confirmed live: unlike its sibling per-turn lists (inspection_ids[:5],
audit_ids[:5], cycle_ids[:3], facility_names[:5], inspector_names[:5] — all
capped when rendered), row_titles was rendered into the CONVERSATION HISTORY
block in full, with no count or per-item length cap. synthesis.py caps
CAPTURE at last_rows[:20] (matching max_display_rows), so a turn listing up
to 20 rows stores up to 20 row_titles entries — one capture path
(observation/submission_id) stores full free-text answer_text values, not
short business codes, so a turn with several observations echoed a large,
uncapped block into EVERY subsequent turn's context for up to MAX_TURNS
turns. Confirmed as a real contributor to a reasoning-LLM prompt hitting
num_ctx=20000 and silently truncating its own tool-call JSON mid-object
(the exact "show all mitigative actions" query that triggered this, which
returned 50 rows total but only captures/displays the first 20).

The fix caps the RENDER only (10 items, 80 chars each, "+N more" suffix) —
ordinal resolution reads the full stored row_titles/row_ids on the
ConversationTurn directly, never this rendered preview, so referencing an
item beyond position 10 must still resolve correctly.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn
from src.agent.orchestrator.ordinal import resolve_ordinal_reference

# 20 items — matches synthesis.py's actual capture cap (last_rows[:20],
# tied to max_display_rows), not an arbitrary large number.
_TITLES = [f"2026/06/ST0{i:02d}/AUD001" for i in range(1, 21)]
_IDS = [f"00000000-0000-0000-0000-{i:012d}" for i in range(1, 21)]

_LONG_OBSERVATION = (
    "The organization has defined competency requirements for personnel "
    "performing work affecting quality performance; however, there was "
    "limited evidence demonstrating a systematic process for evaluating "
    "and maintaining competence based on education, training, and experience."
)


def _session_with_20_row_listing() -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show all mitigative actions",
        answer="20 results found",
        tool_used="generate_sql",
        total_count=139,
        columns=["corrective_action_id", "status"],
        row_titles=list(_TITLES),
        row_ids=list(_IDS),
    ))
    return session


def test_render_previews_only_ten_with_a_more_marker():
    session = _session_with_20_row_listing()
    ctx = session.build_context_block(
        current_question="what about the overdue ones", current_domain="audit",
    )
    assert "(+10 more)" in ctx
    # Only the first 10 titles should appear verbatim in the preview.
    for title in _TITLES[:10]:
        assert title in ctx
    for title in _TITLES[10:]:
        assert title not in ctx


def test_ordinal_beyond_the_render_cap_still_resolves():
    """The render preview stops at 10, but position=15 must still resolve
    — resolve_ordinal_reference reads the full stored list on the
    ConversationTurn directly, never the render preview."""
    session = _session_with_20_row_listing()
    r = resolve_ordinal_reference(session, "audit", position=15, entity_type_hint=None)
    result = r["result"]
    assert result["resolved"] is True
    assert result["target_id"] == _IDS[14]


def test_long_free_text_title_is_truncated_in_the_render():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show observations for the first one",
        answer="3 observations found",
        tool_used="generate_sql",
        total_count=3,
        columns=["submission_id", "answer_text"],
        row_titles=[_LONG_OBSERVATION, "short one", "another short one"],
        row_ids=["id-1", "id-2", "id-3"],
    ))
    ctx = session.build_context_block(
        current_question="what corrective actions came from that",
        current_domain="audit",
    )
    assert _LONG_OBSERVATION not in ctx
    assert _LONG_OBSERVATION[:80] in ctx
