"""
Audit-domain equivalent of agent/prompts/builder.py — assembles SYSTEM_PROMPT_AUDIT
from templates_audit/system_prompt.txt and exposes build_dynamic_prompt_additions_audit()
for per-question Tier-2 worked-example injection (examples_tier2_audit.py).

STATUS_BLOCK/CA_WORKFLOW_BLOCK come from src.knowledge.semantics_registry's
AUDIT_SEMANTICS via audit_status_block()/audit_ca_workflow_block() — same
registry inspection's builder.py consumes, just fed audit's data (see
semantics_registry.py's module docstring). Still no CONDITIONAL_BLOCKS —
audit's per-question routing hints live in sql/rules/audit_rules.py's INJECT
rules instead, consumed via the rules engine, not this module.
FORCE_SUMMARY_PROMPT is shared with the inspection domain
(agent/prompts/templates/force_summary_prompt.txt) since its content
("max iterations reached, summarize") is domain-agnostic.
"""

import logging
import re
from pathlib import Path

_log_prompts = logging.getLogger("inspectai.prompts")

_TEMPLATES_DIR = Path(__file__).parent / "templates_audit"


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


# ── Registry-generated blocks (single source of truth) ──────────────────────
try:
    from src.knowledge.semantics_registry import (
        audit_status_block as _audit_status_block,
        audit_ca_workflow_block as _audit_ca_workflow_block,
    )
    _STATUS_BLOCK = _audit_status_block()
    _CA_WORKFLOW_BLOCK = _audit_ca_workflow_block()
except ImportError:
    # Fallback — keep in sync with knowledge/semantics_registry.py
    _STATUS_BLOCK = _read_template("status_block_fallback.txt")
    _CA_WORKFLOW_BLOCK = _read_template("ca_workflow_block_fallback.txt")

SYSTEM_PROMPT_AUDIT = (
    _read_template("system_prompt.txt")
    .replace("<<<STATUS_BLOCK>>>", _STATUS_BLOCK)
    .replace("<<<CA_WORKFLOW_BLOCK>>>", _CA_WORKFLOW_BLOCK)
)

from src.agent.prompts.examples_tier2_audit import EXAMPLES_TIER2_AUDIT

_DEFAULT_BLOCK_BUDGET_CHARS = 6000


def _select_within_budget(candidates, question, score_fn, budget_chars) -> list[str]:
    """Identical logic to builder.py's _select_within_budget — see there for
    full rationale. Duplicated (not imported) to keep the two domains'
    prompt-assembly modules independently reviewable, matching the rest of
    this domain's file layout."""
    if score_fn is None or len(candidates) <= 1:
        return [text for _, text in candidates]
    scored = sorted(candidates, key=lambda c: score_fn(question, c[0]), reverse=True)
    kept: list[str] = []
    total = 0
    for _, text in scored:
        if kept and total + len(text) > budget_chars:
            break
        kept.append(text)
        total += len(text)
    return kept


def build_worked_examples_audit(
    question: str, score_fn=None, budget_chars: int = _DEFAULT_BLOCK_BUDGET_CHARS
) -> str:
    """Audit-domain equivalent of builder.py's build_worked_examples()."""
    if not question:
        return ""
    matched = []
    for ex in EXAMPLES_TIER2_AUDIT:
        try:
            if re.search(ex["question_pattern"], question):
                matched.append((ex["title"], ex["text"]))
        except re.error:
            continue
    if not matched:
        return ""
    kept = _select_within_budget(matched, question, score_fn, budget_chars)
    return (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ADDITIONAL WORKED EXAMPLES (topic-matched for this question)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        + "\n".join(kept)
    )


def build_dynamic_prompt_additions(question: str, score_fn=None) -> str:
    """
    Same call signature as builder.py's build_dynamic_prompt_additions() —
    AgentOrchestrator calls this generically via self._builder_module (see
    agent/domains.py's DOMAIN_REGISTRY), so both domains' builder modules
    must match this shape. No CONDITIONAL_BLOCKS half for audit (see module
    docstring) — this is worked-examples only.
    """
    worked = build_worked_examples_audit(question, score_fn=score_fn)
    if question and not worked:
        _log_prompts.debug(
            "[PROMPTS] no audit worked-example matched: %r", question[:120]
        )
    return worked
