"""
Audit-domain equivalent of sql/prompts.py — SQL_GENERATION_PROMPT_AUDIT (base
template) and CORE_CONTEXT_AUDIT (audit-specific NL vocabulary facts),
consumed by sql/generation.py::generate_sql() the same way sql/prompts.py's
SQL_GENERATION_PROMPT/CORE_CONTEXT are for inspection.

STATUS GROUPINGS + CA WORKFLOW + CAST RULES come from
src.knowledge.semantics_registry's AUDIT_SEMANTICS (the audit-domain sibling
of SEMANTICS) via audit_status_block()/audit_ca_workflow_block()/
audit_columns_block()/core_context_cast_block(AUDIT_SEMANTICS) — the same
registry inspection's sql/prompts.py consumes, just fed audit's data. Edit
AUDIT_SEMANTICS there, not here. Still no CORE_CONTEXT_CONDITIONAL blocks —
audit's per-question routing hints live in sql/rules/audit_rules.py's INJECT
rules instead, applied by AgentOrchestrator.query() before generate_sql runs,
not by this module.
"""

from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "examples" / "templates_audit"


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


SQL_GENERATION_PROMPT_AUDIT = _read_template("sql_generation_prompt.txt")

# ── Registry-generated blocks (single source of truth) ──────────────────────
try:
    from src.knowledge.semantics_registry import (
        AUDIT_SEMANTICS,
        audit_status_block as _status_block,
        audit_report_status_block as _report_status_block,
        audit_ca_workflow_block as _ca_workflow_block,
        audit_columns_block as _columns_block,
        core_context_cast_block as _cast_block,
    )
    # Two SEPARATE status lifecycles, both rendered so DeepSeek sees the
    # audit_plan-vs-audit_report distinction directly — audit_status_block()
    # alone (audit_plan only) previously left "a bare 'conducted/completed
    # audit'" underspecified; see audit_report_status_block()'s docstring for
    # the live-caught bug this fixes.
    _STATUS_BLOCK = _status_block() + "\n\n" + _report_status_block()
    _CA_WORKFLOW_BLOCK = _ca_workflow_block()
    _COLUMNS_BLOCK = _columns_block()
    # Two non-primary-displacing cast traps: review->CA (still "primary") and
    # observation->CA (added 2026-07-29 once ai_audit_answers was confirmed
    # live and populated — see AUDIT_SEMANTICS["joins"]["observation_to_ca"]).
    _CAST_BLOCK = (
        _cast_block(AUDIT_SEMANTICS) + "\n\n"
        + _cast_block(AUDIT_SEMANTICS, join_key="observation_to_ca")
    )
except ImportError:
    # Fallback — keep in sync with knowledge/semantics_registry.py
    _STATUS_BLOCK = _read_template("status_block_fallback.txt")
    _CA_WORKFLOW_BLOCK = _read_template("ca_workflow_block_fallback.txt")
    _COLUMNS_BLOCK = _read_template("columns_block_fallback.txt")
    _CAST_BLOCK = _read_template("cast_block_fallback.txt")

CORE_CONTEXT_AUDIT = (
    f"{_STATUS_BLOCK}\n\n"
    f"{_COLUMNS_BLOCK}\n\n"
    f"{_CAST_BLOCK}\n\n"
    f"{_CA_WORKFLOW_BLOCK}\n\n"
    "-- ap.id/asch.id/arep.id/aca.id/po.id = UUID (JOIN only, never SELECT for display)\n"
    "-- Display IDs instead: asch.schedule_id, arep.audit_id, aca.corrective_action_id\n"
    "--   (corrective_action_id is 'CA_DRAFT' until adequacy_status = 'APPROVE' — confirmed\n"
    "--   live this is NOT about submission status: a SUBMITTED-status CA can still be\n"
    "--   'CA_DRAFT'. Not a unique identifier while in that state — see below. CRITICAL: a\n"
    "--   'CA_DRAFT' row is still a REAL, EXISTING corrective action that WAS raised, just not\n"
    "--   yet adequacy-approved. Excluding it applies ONLY to what's DISPLAYED in a general\n"
    "--   listing of corrective actions. For an EXISTENCE check — 'audits/reports with no\n"
    "--   corrective actions', 'has a CA been raised against X' — a 'CA_DRAFT' row counts as\n"
    "--   EXISTS/raised = true. Check row existence (aca.id IS NULL / NOT EXISTS), never\n"
    "--   adequacy_status or corrective_action_id, for that kind of question.)\n"
    "-- audit_corrective_action JOIN: ON aca.audit_id = arep.id (NOT aca.id = arep.audit_id)\n"
    "-- audit_schedule JOIN from report: ON arep.schedule_id = asch.id\n"
    "-- risk_level_id/impact_id/compliance_id are UUID FKs: JOIN {table} x ON aca.{col} = x.id\n"
)


def build_core_context_additions(question: str) -> str:
    """
    Same call signature as sql/prompts.py's build_core_context_additions() —
    sql/generation.py calls this generically per domain (see agent/domains.py's
    DOMAIN_REGISTRY). Audit has no CORE_CONTEXT_CONDITIONAL blocks yet (see
    module docstring) — always returns "".
    """
    return ""
