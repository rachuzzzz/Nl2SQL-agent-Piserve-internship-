"""
Regression tests for build_context_block()'s explicit-name entity recall
(session.py:654-682) — reaches further back than current_scope (which
only ever holds the MOST RECENT focus) by scanning the full entity_memory
for any label the user's question names directly by exact word match.

Unconditional relative to the followup gate — this block runs regardless
of is_followup/is_clearly_fresh, unlike the main ACTIVE CONTEXT block.

This is part of the area evaluated for tool conversion in Phase 3 and
deliberately kept as prose (see session.py:219's comment) — this test
closes a prior zero-coverage gap, no functional code was touched.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


def _facility_turn(value: str, domain: str = "audit") -> ConversationTurn:
    return ConversationTurn(
        question=f"about {value}", answer="...", tool_used="generate_sql",
        active_subject_type="facility", active_subject_value=value,
        active_subject_label=value, active_subject_domain=domain,
        active_subject_id=f"id-{value}",
    )


def test_recalls_entity_named_explicitly_from_earlier_in_conversation():
    session = ConversationSession()
    session.add_turn(_facility_turn("Facility A"))
    session.add_turn(_facility_turn("Facility B"))  # current_scope now points at B

    ctx = session.build_context_block("what about Facility A", current_domain="audit")

    assert "RECALLED FROM EARLIER IN THIS CONVERSATION" in ctx
    assert "Facility A" in ctx


def test_does_not_recall_the_currently_active_entity_twice():
    """Dedup: the currently-active entity (already surfaced via
    current_scope/ACTIVE CONTEXT) must not also print under RECALLED."""
    session = ConversationSession()
    session.add_turn(_facility_turn("Facility A"))
    session.add_turn(_facility_turn("Facility B"))  # current_scope now points at B

    ctx = session.build_context_block("what about Facility B", current_domain="audit")

    assert "RECALLED FROM EARLIER IN THIS CONVERSATION" not in ctx


def test_short_labels_are_never_recalled():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="q", answer="a", tool_used="generate_sql",
        active_subject_type="facility", active_subject_value="Al",
        active_subject_label="Al", active_subject_domain="audit",
        active_subject_id="id-al",
    ))
    session.add_turn(_facility_turn("Facility B"))

    ctx = session.build_context_block("what about Al", current_domain="audit")

    assert "RECALLED FROM EARLIER IN THIS CONVERSATION" not in ctx


def test_domain_incompatible_entity_is_not_recalled():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="about inspection ABC", answer="...", tool_used="generate_sql",
        active_subject_type="inspection", active_subject_value="INS-ABC",
        active_subject_label="INS-ABC", active_subject_domain="inspection",
        active_subject_id="id-abc",
    ))
    session.add_turn(_facility_turn("Facility B"))

    ctx = session.build_context_block("what about INS-ABC", current_domain="audit")

    assert "RECALLED FROM EARLIER IN THIS CONVERSATION" not in ctx


def test_recall_requires_whole_word_match_not_substring():
    session = ConversationSession()
    session.add_turn(_facility_turn("Golf"))
    session.add_turn(_facility_turn("Facility B"))

    ctx = session.build_context_block("what about Golfball Arena", current_domain="audit")

    assert "RECALLED FROM EARLIER IN THIS CONVERSATION" not in ctx


def test_recall_is_unconditional_regardless_of_followup_markers():
    """Unlike the main ACTIVE CONTEXT block, this recall block has no
    is_followup gate — a plain question with no pronoun/followup marker
    still triggers recall if it names an earlier entity by exact word."""
    session = ConversationSession()
    session.add_turn(_facility_turn("Facility A"))
    session.add_turn(_facility_turn("Facility B"))

    ctx = session.build_context_block("list all audit plans for Facility A", current_domain="audit")

    assert "RECALLED FROM EARLIER IN THIS CONVERSATION" in ctx
