"""
Regression tests for session.py's standalone re-aggregate context block
(the "you have not seen the individual rows, do not answer from Total
rows/memory" warning) — 2026-08-07 fix (debug_llm_20260807_153128,
qid=3dbcdaf3).

Deliberately decoupled from captured id lists (is_count_followup +
last.sql + last.total_count>0 only) — the pre-existing id-list-specific
backstop (session.py's audit_ids/inspection_ids branches) structurally
cannot fire when a previous turn captured no id column at all.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn

_MARKER = "You have NOT seen the individual rows"


def _session_with_prior_listing(total_count=18, audit_ids=None, row_titles=None, sql=None):
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="list all conducted audits",
        answer="18 audits found",
        tool_used="execute_sql",
        sql=sql or "SELECT arep.audit_id FROM audit_report arep WHERE arep.status = 'SUBMITTED'",
        total_count=total_count,
        audit_ids=audit_ids if audit_ids is not None else ["A1", "A2", "A3", "A4", "A5"],
        row_titles=row_titles if row_titles is not None else ["A1", "A2", "A3", "A4", "A5"],
        domain="audit",
    ))
    return session


def test_reaggregate_block_renders_for_live_phrasing():
    session = _session_with_prior_listing()
    ctx = session.build_context_block(
        "how many of the ones listed above are completed", current_domain="audit",
    )
    assert _MARKER in ctx
    assert "Previous SQL:" in ctx


def test_reaggregate_block_renders_even_with_no_captured_ids():
    """The case the old id-list-specific backstop structurally cannot
    cover: a previous turn that listed by title only, no id column."""
    session = _session_with_prior_listing(audit_ids=[], row_titles=[])
    ctx = session.build_context_block(
        "how many of the ones listed above are completed", current_domain="audit",
    )
    assert _MARKER in ctx


def test_reaggregate_block_still_renders_alongside_complete_set_in_list_branch():
    """total_count == len(captured ids) still gets the new block too — it's
    deliberately unconditional on "complete vs sample" (a fresh COUNT(*) is
    safe either way); this coexists with the pre-existing IN-list branch's
    own guidance for the complete-set case, not a replacement for it."""
    session = _session_with_prior_listing(total_count=3, audit_ids=["A1", "A2", "A3"], row_titles=["A1", "A2", "A3"])
    ctx = session.build_context_block(
        "how many of these are completed", current_domain="audit",
    )
    assert _MARKER in ctx
    assert "COMPLETE result set" in ctx


def test_reaggregate_block_absent_for_fresh_question():
    session = _session_with_prior_listing()
    ctx = session.build_context_block(
        "how many audits were conducted last month", current_domain="audit",
    )
    assert _MARKER not in ctx


def test_reaggregate_block_absent_without_prior_sql():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="hi", answer="hello", tool_used="final_answer",
        sql="", total_count=0, domain="audit",
    ))
    ctx = session.build_context_block(
        "how many of the ones listed above are completed", current_domain="audit",
    )
    assert _MARKER not in ctx
