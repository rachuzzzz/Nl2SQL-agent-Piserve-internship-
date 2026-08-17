"""
Regression test for the PAGINATION/REFINEMENT/AUGMENTATION block in
ConversationSession.build_context_block() (session.py) becoming conditional
(token-usage audit, 2026-08-06) — it used to be unconditional on every turn
from turn 2 onward regardless of question content. Confirms the gate fires
for genuine follow-up phrasing and stays silent otherwise, without touching
whether the block's CONTENT is correct (unchanged text, already covered by
live usage).
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


def _session_with_one_turn() -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="list all audit schedules",
        answer="25 schedules found",
        tool_used="generate_sql",
        total_count=25,
        columns=["schedule_status"],
    ))
    return session


def test_pagination_block_present_for_show_more():
    session = _session_with_one_turn()
    block = session.build_context_block("show more", current_domain="audit")
    assert "PAGINATION:" in block
    assert "REFINEMENT:" in block
    assert "AUGMENTATION:" in block


def test_pagination_block_present_for_augmentation_phrasing():
    session = _session_with_one_turn()
    block = session.build_context_block("also show the auditee", current_domain="audit")
    assert "AUGMENTATION:" in block


def test_pagination_block_absent_for_unrelated_question():
    session = _session_with_one_turn()
    block = session.build_context_block("what is the average inspection score", current_domain="audit")
    assert "PAGINATION:" not in block
    assert "REFINEMENT:" not in block
    assert "AUGMENTATION:" not in block


def test_pagination_block_absent_when_question_omitted():
    session = _session_with_one_turn()
    block = session.build_context_block(current_domain="audit")
    assert "PAGINATION:" not in block
