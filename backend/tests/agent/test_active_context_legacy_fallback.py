"""
Regression tests for build_context_block()'s legacy single-entity fallback
context (session.py:528-652) — fires only when the primary ACTIVE CONTEXT
block (session.py:317-527) didn't set an active_type/active_value (i.e.
no resolved active_subject_type on the current scope), reading
session.turns[-1] directly instead.

This is part of the area evaluated for tool conversion in Phase 3 and
deliberately kept as prose (see session.py:219's comment) — this test
closes a prior zero-coverage gap, no functional code was touched.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


def _turn(**kwargs) -> ConversationTurn:
    base = dict(question="q", answer="a", tool_used="generate_sql", domain="inspection")
    base.update(kwargs)
    return ConversationTurn(**base)


def test_single_inspection_id_gives_exact_filter():
    session = ConversationSession()
    session.add_turn(_turn(inspection_ids=["2026/04/ST085/INS003"], total_count=1))
    ctx = session.build_context_block("what is its status", current_domain="inspection")
    assert "ir.inspection_id = '2026/04/ST085/INS003'" in ctx


def test_multiple_inspection_ids_sample_of_larger_set_warns_against_in_list():
    session = ConversationSession()
    session.add_turn(_turn(
        inspection_ids=["INS-1", "INS-2", "INS-3"], total_count=195,
        sql="SELECT * FROM inspection_report WHERE status = 'OPEN'",
    ))
    ctx = session.build_context_block("how many of them are overdue", current_domain="inspection")
    assert "NOT the full set" in ctx
    assert "do NOT filter by" in ctx
    assert "fresh COUNT" in ctx or "complete recount" in ctx


def test_multiple_inspection_ids_complete_set_allows_in_list():
    session = ConversationSession()
    session.add_turn(_turn(
        inspection_ids=["INS-1", "INS-2", "INS-3"], total_count=3,
    ))
    ctx = session.build_context_block("show details for them", current_domain="inspection")
    assert "COMPLETE result set" in ctx
    assert "inspection_id IN" in ctx


def test_single_audit_id_gives_exact_filter():
    session = ConversationSession()
    session.add_turn(_turn(domain="audit", audit_ids=["2026/07/ST033/AUD001"], total_count=1))
    ctx = session.build_context_block("what is its status", current_domain="audit")
    assert "arep.audit_id = '2026/07/ST033/AUD001'" in ctx


def test_multiple_audit_ids_sample_warns_against_in_list():
    session = ConversationSession()
    session.add_turn(_turn(
        domain="audit", audit_ids=["AUD-1", "AUD-2"], total_count=50,
        sql="SELECT * FROM audit_report",
    ))
    ctx = session.build_context_block("how many of them are closed", current_domain="audit")
    assert "NOT the full set" in ctx


def test_multiple_audit_ids_complete_set_allows_in_list():
    session = ConversationSession()
    session.add_turn(_turn(domain="audit", audit_ids=["AUD-1", "AUD-2"], total_count=2))
    ctx = session.build_context_block("show details for the above two audits", current_domain="audit")
    assert "COMPLETE result set" in ctx
    assert "arep.audit_id IN" in ctx


def test_single_facility_name_gives_ilike_filter():
    session = ConversationSession()
    session.add_turn(_turn(facility_names=["Golf Gardens"]))
    ctx = session.build_context_block("what about it", current_domain="inspection")
    assert "fac.name ILIKE '%Golf Gardens%'" in ctx


def test_single_row_aggregate_result_points_at_row_level_columns():
    session = ConversationSession()
    session.add_turn(_turn(
        single_row_values={"count": 42},
        sql="SELECT COUNT(*) FROM inspection_corrective_action WHERE adequacy_status != 'APPROVE'",
    ))
    ctx = session.build_context_block("show more details regarding those", current_domain="inspection")
    assert "SCALAR AGGREGATE" in ctx
    assert "INDIVIDUAL ROWS" in ctx


def test_single_row_specific_record_result_gives_generic_message():
    session = ConversationSession()
    session.add_turn(_turn(
        single_row_values={"corrective_action_id": "CA001", "status": "OPEN"},
        sql="SELECT corrective_action_id, status FROM inspection_corrective_action WHERE id = 'x'",
    ))
    ctx = session.build_context_block("tell me more about it", current_domain="inspection")
    assert "SCALAR AGGREGATE" not in ctx
    assert "Last query returned a single result" in ctx


def test_aggregate_with_group_by_is_not_misclassified_as_scalar_aggregate():
    """COUNT(*) WITH a GROUP BY is a per-group breakdown, not a single
    scalar summarizing many rows — must not trigger the aggregate branch."""
    session = ConversationSession()
    session.add_turn(_turn(
        single_row_values={"status": "OPEN", "count": 5},
        sql="SELECT status, COUNT(*) FROM inspection_report GROUP BY status",
    ))
    ctx = session.build_context_block("show more about that", current_domain="inspection")
    assert "SCALAR AGGREGATE" not in ctx


def test_domain_incompatible_last_turn_does_not_leak_inspection_ids():
    """_domain_compat gate — a last turn from a DIFFERENT domain than the
    one currently being answered must not leak its inspection_ids/audit_ids
    (confirmed live as a second, independent leak path from the primary
    gate — session.py:535-538)."""
    session = ConversationSession()
    session.add_turn(_turn(domain="inspection", inspection_ids=["INS-1"], total_count=1))
    ctx = session.build_context_block("what is its status", current_domain="audit")
    assert "INS-1" not in ctx or "ir.inspection_id" not in ctx


def test_no_fallback_when_primary_active_context_already_resolved():
    """The elif gate — when the primary block already set active_type/
    active_value, the legacy fallback must not ALSO render (mutually
    exclusive, not additive)."""
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="q", answer="a", tool_used="generate_sql", domain="audit",
        active_subject_type="facility", active_subject_value="Golf Gardens",
        active_subject_label="Golf Gardens", active_subject_domain="audit",
        active_subject_id="fac-1",
        inspection_ids=["INS-1"], total_count=1,
    ))
    ctx = session.build_context_block("what is its status", current_domain="audit")
    assert "scoped to facility 'Golf Gardens'" in ctx
    assert "ir.inspection_id" not in ctx
