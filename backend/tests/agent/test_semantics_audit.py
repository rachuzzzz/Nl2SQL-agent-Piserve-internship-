"""
Regression tests confirming AUDIT_SEMANTICS-derived content actually reaches
the built prompts — the audit-domain consumption check for
knowledge/semantics_registry.py's AUDIT_SEMANTICS (paired constant to
SEMANTICS, same pattern as schema_introspector.py's
BUSINESS_RULES/AUDIT_BUSINESS_RULES). Companion to
tests/audit/test_audit_rules_consumption.py, which covers the validator side.

No LLM, no live DB — pure string assembly, asserting on the built module-level
constants (SYSTEM_PROMPT_AUDIT, CORE_CONTEXT_AUDIT) and the underlying
registry functions directly.
"""

from src.knowledge.semantics_registry import (
    SEMANTICS,
    AUDIT_SEMANTICS,
    constraint_note_status,
    constraint_note_cast,
    audit_status_block,
    audit_ca_workflow_block,
    audit_columns_block,
)


def test_audit_semantics_has_own_status_lifecycle_distinct_from_inspection():
    assert AUDIT_SEMANTICS["status"]["label"] == "audit_plan.status"
    assert SEMANTICS["status"]["label"] == "inspection_report.status"
    assert AUDIT_SEMANTICS["status"]["groupings"]["completed"]["statuses"] == ["APPROVED"]
    assert SEMANTICS["status"]["groupings"]["completed"]["statuses"] == ["CLOSED"]


def test_constraint_note_status_is_data_driven_per_domain():
    inspection_note = constraint_note_status()
    audit_note = constraint_note_status(AUDIT_SEMANTICS)
    assert "status = 'CLOSED'" in inspection_note
    assert "status = 'APPROVED'" in audit_note
    assert "ap.status" in audit_note
    assert "ir.status" in inspection_note


def test_constraint_note_cast_is_data_driven_per_domain():
    inspection_note = constraint_note_cast()
    audit_note = constraint_note_cast(AUDIT_SEMANTICS)
    assert "OBSERVATION" in inspection_note
    assert "REVIEW" in audit_note and "AUDIT_CORRECTIVE_ACTION" in audit_note
    assert "aa.submission_id" in inspection_note
    assert "rev.entity_id" in audit_note


def test_audit_status_block_covers_all_four_values():
    block = audit_status_block()
    for value in ("DRAFT", "SUBMITTED", "APPROVED", "RETURNED_FOR_MODIFICATION"):
        assert value in block


def test_audit_ca_workflow_block_covers_all_four_tracks():
    block = audit_ca_workflow_block()
    assert "progress_stage" in block
    assert "implementation_status" in block
    assert "adequacy_status" in block
    assert "NO inspection equivalent" in block  # adequacy_status has no inspection analog


def test_audit_columns_block_states_universal_active_rule():
    block = audit_columns_block()
    assert "active" in block
    assert "audit_plan" in block and "audit_corrective_action" in block


def test_core_context_audit_includes_registry_derived_blocks():
    from src.sql.prompts_audit import CORE_CONTEXT_AUDIT
    assert "audit_plan.status" in CORE_CONTEXT_AUDIT
    assert "AUDIT CORRECTIVE ACTION WORKFLOW" in CORE_CONTEXT_AUDIT
    assert "REVIEW" in CORE_CONTEXT_AUDIT and "AUDIT_CORRECTIVE_ACTION JOIN" in CORE_CONTEXT_AUDIT


def test_system_prompt_audit_includes_registry_derived_blocks():
    from src.agent.prompts.builder_audit import SYSTEM_PROMPT_AUDIT
    assert "<<<" not in SYSTEM_PROMPT_AUDIT  # every sentinel must be spliced
    assert "audit_plan.status" in SYSTEM_PROMPT_AUDIT
    assert "AUDIT CORRECTIVE ACTION WORKFLOW" in SYSTEM_PROMPT_AUDIT
