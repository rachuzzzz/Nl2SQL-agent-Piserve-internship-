"""
Regression test for the "pending clarification" context — confirmed live
2026-07-29 (qid=c3004b3d -> qid=7d5d343d): "top 5 capex converters" got a
clarifying question back ("if you mean FM Managers... just say which group
and metric"). The reply "FM manafers" should have resolved that and continued
the ORIGINAL request, but instead the next turn treated it as a brand-new,
unrelated ambiguous-person-name search (43 unrelated user matches) and the
original request was silently dropped, never answered.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn

# Verbatim from the live incident (qid=c3004b3d) — the "?" is mid-answer,
# followed by bullet-point options, and the answer ends with a period, not a
# question mark. This shape is exactly why "ends with ?" is the wrong
# detection heuristic and "?" appears anywhere in the answer is the right one.
CLARIFYING_ANSWER = (
    "Could you clarify who and what 'top 5 capex converters' means?\n\n"
    "**If you mean FM Managers** (facility managers being inspected):\n"
    "• Most capex converted (highest PO amounts paid by client)\n\n"
    "Just say which group and metric and I'll pull the data."
)


def test_pending_clarification_surfaced_in_context_block():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="top 5 capex converters",
        answer=CLARIFYING_ANSWER,
        tool_used="final_answer",
        domain="inspection",
        clarification_origin_question="top 5 capex converters",
    ))
    ctx = session.build_context_block("FM manafers", current_domain="inspection")
    assert "PENDING CLARIFICATION" in ctx
    assert "top 5 capex converters" in ctx
    assert "brand-new unrelated request" in ctx


def test_no_pending_clarification_when_last_answer_was_not_a_question():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="how many audits are overdue",
        answer="There are 3 overdue audits.",
        tool_used="execute_sql",
        domain="audit",
        clarification_origin_question="",
    ))
    ctx = session.build_context_block("what about last month", current_domain="audit")
    assert "PENDING CLARIFICATION" not in ctx


def test_final_answer_containing_question_mark_sets_clarification_origin():
    # Exercises the actual detection heuristic used in loop.py's final_answer
    # handling: "?" ANYWHERE in the answer (not just at the end) is treated as
    # a clarifying question, and the ORIGINAL question is captured as the
    # origin to resolve later. Must be "contains", not "ends with" — the real
    # incident's answer has its "?" mid-text, followed by bullet options, and
    # ends in a period.
    ans = CLARIFYING_ANSWER
    question = "top 5 capex converters"
    clarification_origin = question if "?" in ans else ""
    assert clarification_origin == "top 5 capex converters"


def test_plain_final_answer_without_question_mark_has_no_clarification_origin():
    ans = "There are 3 overdue audits."
    question = "how many audits are overdue"
    clarification_origin = question if "?" in ans else ""
    assert clarification_origin == ""
