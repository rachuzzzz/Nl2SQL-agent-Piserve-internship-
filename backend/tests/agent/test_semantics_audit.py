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

import re

from src.knowledge.semantics_registry import (
    SEMANTICS,
    AUDIT_SEMANTICS,
    AUDIT_REPORT_SEMANTICS,
    constraint_note_status,
    constraint_note_cast,
    audit_status_block,
    audit_ca_workflow_block,
    audit_columns_block,
    conducted_required_regex,
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


def test_conducted_required_regex_accepts_both_single_status_forms():
    """2026-08-07 fix (debug_llm_20260807_153128): AUDIT_REPORT_SEMANTICS's
    conducted bucket is a single status (SUBMITTED). The regex must accept
    both the natural '=' form and the 'IN (single)' form the glossary's own
    corrective message renders — before the fix, only '=' matched, so the
    message could never satisfy its own required_pattern."""
    pattern = conducted_required_regex(AUDIT_REPORT_SEMANTICS)
    assert re.search(pattern, "arep.status = 'SUBMITTED'")
    assert re.search(pattern, "arep.status IN ('SUBMITTED')")
    assert not re.search(pattern, "arep.status IN ('SUBMITTED','CLOSED')")


def test_system_prompt_audit_includes_count_followup_worked_example():
    """2026-08-07 fix (debug_llm_20260807_153128): the audit domain had no
    equivalent of the inspection domain's "how many of these are X"
    re-aggregate worked example — guards against it silently disappearing
    again."""
    from src.agent.prompts.builder_audit import SYSTEM_PROMPT_AUDIT
    assert "How many of these are X" in SYSTEM_PROMPT_AUDIT
    assert "completed_count" in SYSTEM_PROMPT_AUDIT
    # Targets the second half of the live failure — the model conflating
    # audit_report.status / audit_schedule.schedule_status / audit_plan.status.
    assert "three separate columns" in SYSTEM_PROMPT_AUDIT


def test_every_glossary_entrys_own_correct_sql_satisfies_its_own_required_pattern():
    """Standing invariant: this exact class of self-contradiction (a
    corrective message that fails its own required_pattern) was silent
    under SOFT_WARN and only surfaced once an entry was escalated to
    HARD_FAIL. Catch it for every entry going forward, not just the two
    fixed in this session."""
    from src.knowledge.business_glossary import GLOSSARY
    failures = []
    for entry in GLOSSARY:
        if not entry.required_pattern:
            continue
        if not re.search(entry.required_pattern, entry.correct_sql, re.IGNORECASE):
            failures.append(entry.id)
    assert not failures, f"entries whose correct_sql fails their own required_pattern: {failures}"
