"""
Regression tests for the ACTIVE CONTEXT block's ambiguous-duplicate-name
STOP guards (facility: session.py:451-460, entity: session.py:497-505) —
safety-critical: when _lookup_entity_id() finds MORE THAN ONE row sharing
the active label's display name, the system must hard-stop and ask the
user to disambiguate further rather than silently picking one arbitrarily
(the same class of bug as audit_plan.title duplicates — two different
facility/project/client/organisation rows can share a display name).

This is part of the area evaluated for tool conversion in Phase 3 and
deliberately kept as prose (see session.py:219's comment) — this test
closes a prior zero-coverage gap, no functional code was touched.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


def test_facility_stop_guard_fires_on_duplicate_name():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="mitigative actions at golf gardens", answer="...", tool_used="generate_sql",
        active_subject_type="facility", active_subject_value="Golf Gardens",
        active_subject_label="Golf Gardens", active_subject_domain="audit",
        active_subject_id="",  # unresolved — forces fallback to _lookup_entity_id
        entity_disambiguation_candidates=[
            {"name": "Golf Gardens", "count": 5, "id": "fac-1"},
            {"name": "Golf Gardens", "count": 3, "id": "fac-2"},
        ],
    ))
    ctx = session.build_context_block("what about its overdue actions", current_domain="audit")

    assert "STOP" in ctx
    assert "fac-1" in ctx and "fac-2" in ctx
    # Mutually exclusive branches — the normal filter line must NOT also render.
    assert "ALL follow-up questions" not in ctx


def test_entity_stop_guard_fires_on_duplicate_name():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="audits for acme corp", answer="...", tool_used="generate_sql",
        active_subject_type="entity", active_subject_value="Acme Corp",
        active_subject_label="Acme Corp", active_subject_domain="audit",
        active_subject_table="project", active_subject_alias="proj",
        active_subject_id="",
        entity_disambiguation_candidates=[
            {"name": "Acme Corp", "count": 2, "id": "proj-1"},
            {"name": "Acme Corp", "count": 1, "id": "proj-2"},
        ],
    ))
    ctx = session.build_context_block("what about its status", current_domain="audit")

    assert "STOP" in ctx
    assert "proj-1" in ctx and "proj-2" in ctx


def test_facility_resolves_normally_when_name_is_unique():
    """Regression guard: the STOP guard must not false-positive when only
    one candidate matches the active label."""
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="mitigative actions at golf gardens", answer="...", tool_used="generate_sql",
        active_subject_type="facility", active_subject_value="Golf Gardens",
        active_subject_label="Golf Gardens", active_subject_domain="audit",
        active_subject_id="",
        entity_disambiguation_candidates=[
            {"name": "Golf Gardens", "count": 5, "id": "fac-1"},
        ],
    ))
    ctx = session.build_context_block("what about its overdue actions", current_domain="audit")

    assert "STOP" not in ctx
    assert "fac-1" in ctx


def test_facility_resolves_via_active_id_without_needing_lookup():
    """When active_subject_id is already resolved (the common case — set
    directly by synthesis.py's turn-finish capture), the STOP-guard lookup
    path isn't even needed; the id is used directly."""
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="mitigative actions at golf gardens", answer="...", tool_used="generate_sql",
        active_subject_type="facility", active_subject_value="Golf Gardens",
        active_subject_label="Golf Gardens", active_subject_domain="audit",
        active_subject_id="fac-1",
    ))
    ctx = session.build_context_block("what about its overdue actions", current_domain="audit")

    assert "STOP" not in ctx
    assert "fac-1" in ctx
