"""
Regression test for the ACTIVE CONTEXT plan branch's title-ambiguity
warning (session.py:398-416) — confirmed live 2026-07-30: more than one
ACTIVE audit_plan row can share an identical title ('Delegation plan -1',
'Delegation testing' both had 2 active rows). When no resolved plan id is
available (GROUP BY query, or injection didn't run) and the ORIGINAL turn
that set this active_value was flagged plan_title_ambiguous, the system
must warn that a title/ILIKE filter isn't guaranteed to hit the right row
— never silently pick one.

This is part of the area evaluated for tool conversion in Phase 3 and
deliberately kept as prose (see session.py:219's comment) — this test
closes a prior zero-coverage gap, no functional code was touched.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


def test_ambiguous_plan_title_warns_when_no_id_is_available():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show the review status", answer="...", tool_used="generate_sql",
        active_subject_type="plan", active_subject_value="Delegation plan -1",
        active_subject_label="Delegation plan -1", active_subject_domain="audit",
        active_subject_id="",  # unresolved — no plan_ids captured either
        plan_title_ambiguous=True,
    ))
    ctx = session.build_context_block("what is its status", current_domain="audit")

    assert "WARNING" in ctx
    assert "more than one ACTIVE audit_plan" in ctx
    assert "ap.title = 'Delegation plan -1'" in ctx  # still falls back to title filter


def test_unambiguous_plan_title_does_not_warn():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show the review status", answer="...", tool_used="generate_sql",
        active_subject_type="plan", active_subject_value="Unique Plan Name",
        active_subject_label="Unique Plan Name", active_subject_domain="audit",
        active_subject_id="",
        plan_title_ambiguous=False,
    ))
    ctx = session.build_context_block("what is its status", current_domain="audit")

    assert "WARNING" not in ctx


def test_resolved_plan_id_skips_ambiguity_warning_entirely():
    """When a real id IS available, the branch uses it directly and never
    reaches the title-ambiguity fallback/warning at all — even if the
    title happens to also be ambiguous elsewhere."""
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show the review status", answer="...", tool_used="generate_sql",
        active_subject_type="plan", active_subject_value="Delegation plan -1",
        active_subject_label="Delegation plan -1", active_subject_domain="audit",
        active_subject_id="plan-uuid-1",
        plan_title_ambiguous=True,
    ))
    ctx = session.build_context_block("what is its status", current_domain="audit")

    assert "WARNING" not in ctx
    assert "ap.id = 'plan-uuid-1'" in ctx
