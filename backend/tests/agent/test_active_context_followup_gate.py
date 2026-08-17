"""
Regression tests for build_context_block()'s followup/fresh-subject gate
(session.py:109-139) — the pre-processing step that decides whether the
ACTIVE CONTEXT block (and the legacy single-entity fallback) are even
eligible to render this turn.

This is part of the area evaluated for tool conversion in Phase 3 and
deliberately kept as prose (see session.py:219's comment for the
reasoning) — these tests close a near-zero prior coverage gap rather than
accompanying a behavior change; no functional code was touched here.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


def _session_with_facility_scope() -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="which facility has the most overdue actions", answer="Aldar HQ",
        tool_used="generate_sql",
        active_subject_type="facility", active_subject_value="Aldar HQ",
        active_subject_label="Aldar HQ", active_subject_domain="audit",
        active_subject_id="52611ecb-8be8-4507-a25c-df5807ccf2b4",
    ))
    return session


def _renders_active_context(question: str) -> bool:
    session = _session_with_facility_scope()
    ctx = session.build_context_block(question, current_domain="audit")
    return "ACTIVE CONTEXT" in ctx


def test_pronoun_it_triggers_followup():
    assert _renders_active_context("what is its status") is True


def test_demonstrative_that_facility_triggers_followup():
    assert _renders_active_context("show me that facility's inspections") is True


def test_the_same_triggers_followup():
    assert _renders_active_context("show the same for last month") is True


def test_continue_triggers_followup():
    assert _renders_active_context("continue") is True


def test_bare_that_or_this_does_not_trigger_followup():
    """Deliberately excluded (session.py:109-114) — bare 'that'/'this' double
    as relative pronouns/determiners in ordinary questions and would
    false-positive-match plain global questions."""
    assert _renders_active_context("mitigative actions that are overdue") is False
    assert _renders_active_context("show me this month's data") is False


def test_fresh_subject_alone_suppresses_active_context():
    assert _renders_active_context("what is the most recent audit") is False
    assert _renders_active_context("show the most common finding") is False


def test_fresh_subject_combined_with_followup_still_triggers():
    """is_clearly_fresh = ... and not is_followup (session.py:136-139) — a
    followup marker takes precedence over a fresh-subject marker in the
    same question."""
    assert _renders_active_context("what is its most recent status") is True


def test_most_recently_is_not_a_fresh_subject_match():
    """_FRESH_SUBJECT_RE's negative lookahead (session.py:132,
    'most\\s+recent(?!ly)') deliberately excludes 'most recently' — a
    different phrase from the fresh-subject marker 'most recent'."""
    session = _session_with_facility_scope()
    ctx = session.build_context_block(
        "which facility was inspected most recently", current_domain="audit",
    )
    # No followup marker present either, so ACTIVE CONTEXT correctly stays
    # absent here — this test only confirms the negative lookahead doesn't
    # crash/misbehave, not that this phrasing renders ACTIVE CONTEXT.
    assert "ACTIVE CONTEXT" not in ctx


def test_plain_question_with_no_markers_does_not_trigger():
    assert _renders_active_context("list all audit plans") is False


def test_empty_session_returns_empty_context():
    session = ConversationSession()
    assert session.build_context_block("its status", current_domain="audit") == ""


# ── 2026-08-07 (debug_llm_20260807_153128) broadening — these/those/ones/
# "listed above" variants added to _FOLLOWUP_RE. Blast-radius test for that
# change: it must widen ACTIVE CONTEXT eligibility for these new tokens
# without touching the existing pins above.

def test_these_triggers_followup():
    assert _renders_active_context("how many of these are completed") is True


def test_those_triggers_followup():
    assert _renders_active_context("how many of those are completed") is True


def test_plural_ones_triggers_followup():
    assert _renders_active_context("which ones are completed") is True
    assert _renders_active_context("show the open ones") is True


def test_listed_above_variant_triggers_followup():
    assert _renders_active_context("how many of the ones listed above are completed") is True
    assert _renders_active_context("what about the ones shown above") is True
