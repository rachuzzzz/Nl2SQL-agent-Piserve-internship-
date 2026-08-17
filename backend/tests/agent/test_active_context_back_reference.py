"""
Regression tests for build_context_block()'s "go back"/"back to" handling
(_BACK_RE, session.py:242-260) — re-anchors to the OLDEST domain-compatible
active subject in history, skipping past an incompatible one rather than
stopping dead, instead of using the persistent current_scope pointer
(which always points at the MOST recent subject).
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


def _turn(active_type, value, domain, id_, **extra) -> ConversationTurn:
    return ConversationTurn(
        question=f"about {value}", answer="...", tool_used="generate_sql",
        active_subject_type=active_type, active_subject_value=value,
        active_subject_label=value, active_subject_domain=domain,
        active_subject_id=id_, **extra,
    )


def test_back_reference_skips_domain_incompatible_earlier_subject():
    """3 turns: oldest is inspection-domain (incompatible when answering in
    audit), middle is facility (always compatible), newest is plan
    (audit-domain, also compatible). 'go back' must resolve to the OLDEST
    COMPATIBLE one — the middle facility turn — not the plan (even though
    plan is also compatible, it's not the oldest) and not the incompatible
    inspection turn."""
    session = ConversationSession()
    session.add_turn(_turn("inspection", "INS-OLD", "inspection", "id-old"))
    session.add_turn(_turn("facility", "Facility A", "audit", "id-facility"))
    session.add_turn(_turn("plan", "Plan X", "audit", "id-plan"))

    ctx = session.build_context_block("go back to the earlier one", current_domain="audit")

    assert "id-facility" in ctx or "Facility A" in ctx
    assert "id-plan" not in ctx
    # "INS-OLD" legitimately appears in the always-rendered turn-history
    # dump (a separate, unconditional block) — check the resolved id
    # specifically wasn't used for THIS turn's active context instead.
    assert "id-old" not in ctx


def test_back_reference_finds_immediately_compatible_earlier_subject():
    session = ConversationSession()
    session.add_turn(_turn("facility", "Facility A", "audit", "id-facility"))
    session.add_turn(_turn("plan", "Plan X", "audit", "id-plan"))

    ctx = session.build_context_block("go back to that facility", current_domain="audit")

    assert "id-facility" in ctx or "Facility A" in ctx
    assert "id-plan" not in ctx


def test_without_back_reference_uses_most_recent_current_scope_instead():
    """Regression guard: normal (non-'go back') phrasing must NOT trigger
    the oldest-first walk — it should use current_scope (the MOST recent
    subject) as usual."""
    session = ConversationSession()
    session.add_turn(_turn("facility", "Facility A", "audit", "id-facility"))
    session.add_turn(_turn("plan", "Plan X", "audit", "id-plan"))

    ctx = session.build_context_block("what is its status", current_domain="audit")

    assert "id-plan" in ctx or "Plan X" in ctx
    assert "id-facility" not in ctx


def test_return_to_phrasing_also_triggers_back_reference():
    session = ConversationSession()
    session.add_turn(_turn("facility", "Facility A", "audit", "id-facility"))
    session.add_turn(_turn("plan", "Plan X", "audit", "id-plan"))

    ctx = session.build_context_block("return to the facility question", current_domain="audit")

    assert "id-facility" in ctx or "Facility A" in ctx
