"""
Regression tests for build_context_block()'s domain gate (_domain_ok,
session.py:234-239) — domain-exclusive active_subject_type values
(inspection/audit/cycle/plan) only carry forward when they originated in
the domain currently being answered; facility/inspector/entity are shared
across both domains and always carry forward.

Confirmed live (session.py:224-226): without this gate, an
inspection-scoped subject got handed to the audit LLM as "use WHERE
ir.inspection_id = ..." — ir/inspection_report doesn't exist in the audit
schema at all.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


def _session_with_scope(active_type: str, domain: str, **extra) -> ConversationSession:
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="seed turn", answer="...", tool_used="generate_sql",
        active_subject_type=active_type, active_subject_value="X",
        active_subject_label="X", active_subject_domain=domain,
        active_subject_id="id-1",
        **extra,
    ))
    return session


def test_domain_exclusive_inspection_scope_does_not_cross_into_audit():
    session = _session_with_scope("inspection", domain="inspection")
    ctx = session.build_context_block("what is its status", current_domain="audit")
    assert "ACTIVE CONTEXT" not in ctx


def test_domain_exclusive_inspection_scope_renders_in_own_domain():
    session = _session_with_scope("inspection", domain="inspection")
    ctx = session.build_context_block("what is its status", current_domain="inspection")
    assert "ACTIVE CONTEXT" in ctx


def test_domain_exclusive_audit_scope_does_not_cross_into_inspection():
    session = _session_with_scope("audit", domain="audit")
    ctx = session.build_context_block("what is its status", current_domain="inspection")
    assert "ACTIVE CONTEXT" not in ctx


def test_domain_exclusive_cycle_scope_does_not_cross_domains():
    session = _session_with_scope("cycle", domain="audit")
    ctx = session.build_context_block("what is its status", current_domain="inspection")
    assert "ACTIVE CONTEXT" not in ctx


def test_domain_exclusive_plan_scope_does_not_cross_domains():
    session = _session_with_scope("plan", domain="audit")
    ctx = session.build_context_block("what is its status", current_domain="inspection")
    assert "ACTIVE CONTEXT" not in ctx


def test_facility_scope_carries_across_both_domains():
    session = _session_with_scope("facility", domain="audit")
    ctx_audit = session.build_context_block("what is its status", current_domain="audit")
    ctx_inspection = session.build_context_block("what is its status", current_domain="inspection")
    assert "ACTIVE CONTEXT" in ctx_audit
    assert "ACTIVE CONTEXT" in ctx_inspection


def test_entity_scope_carries_across_both_domains():
    session = _session_with_scope(
        "entity", domain="audit",
        active_subject_table="project", active_subject_alias="proj",
    )
    ctx_audit = session.build_context_block("what is its status", current_domain="audit")
    ctx_inspection = session.build_context_block("what is its status", current_domain="inspection")
    assert "ACTIVE CONTEXT" in ctx_audit
    assert "ACTIVE CONTEXT" in ctx_inspection


def test_domain_exclusive_scope_carries_when_current_domain_unknown():
    """_domain_ok's `not current_domain` clause — an empty/unset current
    domain doesn't block a domain-exclusive scope from carrying forward."""
    session = _session_with_scope("inspection", domain="inspection")
    ctx = session.build_context_block("what is its status", current_domain="")
    assert "ACTIVE CONTEXT" in ctx
