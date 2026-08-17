"""
Audit-domain equivalent of agent/prompts/builder.py — assembles SYSTEM_PROMPT_AUDIT
from templates_audit/system_prompt.txt and exposes build_dynamic_prompt_additions_audit()
for per-question Tier-2 worked-example injection (examples_tier2_audit.py).

STATUS_BLOCK/CA_WORKFLOW_BLOCK come from src.knowledge.semantics_registry's
AUDIT_SEMANTICS via audit_status_block()/audit_ca_workflow_block() — same
registry inspection's builder.py consumes, just fed audit's data (see
semantics_registry.py's module docstring). CONDITIONAL_BLOCKS_AUDIT (2026-08 token audit) surfaces the topic-gated
remainder of AUDIT_BUSINESS_RULES that orchestrator/loop.py no longer
splices in unconditionally — see knowledge/schema_introspector.py's
"Topic-gated core/topical split". Separate from audit's SQL-routing INJECT
rules in sql/rules/audit_rules.py, which still work as before.
FORCE_SUMMARY_PROMPT is shared with the inspection domain
(agent/prompts/templates/force_summary_prompt.txt) since its content
("max iterations reached, summarize") is domain-agnostic.
"""

import logging
import re
from pathlib import Path

from src.agent.tool_defs import declared_tools as _declared_tools
from src.knowledge.schema_introspector import AUDIT_RULE_TOPIC_TEXT

_log_prompts = logging.getLogger("inspectai.prompts")

_TEMPLATES_DIR = Path(__file__).parent / "templates_audit"


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


# ── Registry-generated blocks (single source of truth) ──────────────────────
try:
    from src.knowledge.semantics_registry import (
        audit_status_block as _audit_status_block,
        audit_report_status_block as _audit_report_status_block,
        audit_ca_workflow_block as _audit_ca_workflow_block,
    )
    # Both lifecycles rendered so the reasoning-stage (Llama) prompt sees the
    # audit_plan-vs-audit_report distinction directly — matches the fix
    # already applied on the SQL-generation-stage (Deepseek) side, see
    # sql/prompts_audit.py's identical comment for the live-caught bug
    # (bare "conducted audit" routed to audit_plan instead of audit_report)
    # this closes on the reasoning side too.
    _STATUS_BLOCK = _audit_status_block() + "\n\n" + _audit_report_status_block()
    _CA_WORKFLOW_BLOCK = _audit_ca_workflow_block()
except ImportError:
    # Fallback — keep in sync with knowledge/semantics_registry.py
    _STATUS_BLOCK = _read_template("status_block_fallback.txt")
    _CA_WORKFLOW_BLOCK = _read_template("ca_workflow_block_fallback.txt")

# See builder.py's identical comment — tools declared via agent/tool_defs
# get appended after the static numbered tool list, which is never itself
# regenerated.
_DYNAMIC_TOOLS_BLOCK = "".join(
    t.doc for t in _declared_tools() if "audit" in t.domains
)

SYSTEM_PROMPT_AUDIT = (
    _read_template("system_prompt.txt")
    .replace("<<<STATUS_BLOCK>>>", _STATUS_BLOCK)
    .replace("<<<CA_WORKFLOW_BLOCK>>>", _CA_WORKFLOW_BLOCK)
    .replace("<<<DYNAMIC_TOOLS_BLOCK>>>", _DYNAMIC_TOOLS_BLOCK)
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


# ── Topic-gated business-rules remainder (2026-08 token audit) — see
# knowledge/schema_introspector.py's "Topic-gated core/topical split" for
# what moved here and why. Patterns are deliberately broad — a false
# positive only costs tokens, a false negative silently drops a
# "confirmed live" bug-fix rule.
_FORM_ANSWERS_TOPIC = (
    r"(?i)\bobserv|\bfinding|\brisk\s+level\b|\bimpact\b|\bcompliance\b"
    r"|\bcheckpoint\b|\brequirement\b|\bform\s+answer|\bsubmission\b"
)
_APPROVAL_WORKFLOW_TOPIC = (
    r"(?i)\bapprov|\breview\b|\breject|\breschedul|\bcompleted\b"
)
_WHO_QUESTIONS_TOPIC = r"(?i)\bwho\b|\bstakeholder\b|\bproposed\b|\bowns?\b|\bassigned\b"
_CYCLE_TOPIC = r"(?i)\bcycle\b"
_PORTFOLIO_TOPIC = r"(?i)\bportfolio\b|\bmodification\b"
_UNIVERSE_TOPIC = r"(?i)\buniverse\b"
_DATE_CONDUCTED_TOPIC = (
    r"(?i)\bconduct|\bwhen\b.{0,20}\baudit\b|\baudit\b.{0,20}\bdate\b|\bdate\b.{0,20}\baudit\b"
)

CONDITIONAL_BLOCKS_AUDIT = [
    {"title": "Form answers / observation / risk-impact-compliance",
     "question_pattern": _FORM_ANSWERS_TOPIC,
     "get_text": lambda: AUDIT_RULE_TOPIC_TEXT["form_answers"]},
    {"title": "Approval workflow",
     "question_pattern": _APPROVAL_WORKFLOW_TOPIC,
     "get_text": lambda: AUDIT_RULE_TOPIC_TEXT["approval_workflow"]},
    {"title": "Who questions",
     "question_pattern": _WHO_QUESTIONS_TOPIC,
     "get_text": lambda: AUDIT_RULE_TOPIC_TEXT["who_questions"]},
    {"title": "Audit cycle",
     "question_pattern": _CYCLE_TOPIC,
     "get_text": lambda: AUDIT_RULE_TOPIC_TEXT["cycle"]},
    {"title": "Portfolio",
     "question_pattern": _PORTFOLIO_TOPIC,
     "get_text": lambda: AUDIT_RULE_TOPIC_TEXT["portfolio"]},
    {"title": "Audit universe",
     "question_pattern": _UNIVERSE_TOPIC,
     "get_text": lambda: AUDIT_RULE_TOPIC_TEXT["universe"]},
    {"title": "Date conducted",
     "question_pattern": _DATE_CONDUCTED_TOPIC,
     "get_text": lambda: AUDIT_RULE_TOPIC_TEXT["date_conducted"]},
]


def build_conditional_blocks_audit(
    question: str, score_fn=None, budget_chars: int = _DEFAULT_BLOCK_BUDGET_CHARS
) -> str:
    """Audit-domain equivalent of builder.py's build_conditional_blocks()."""
    if not question:
        return ""
    matched = []
    for blk in CONDITIONAL_BLOCKS_AUDIT:
        try:
            if re.search(blk["question_pattern"], question):
                matched.append((blk["title"], blk["get_text"]()))
        except re.error:
            continue
    if not matched:
        return ""
    kept = _select_within_budget(matched, question, score_fn, budget_chars)
    return (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ADDITIONAL CONTEXT (topic-matched for this question)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        + "\n\n".join(kept)
    )


def build_dynamic_prompt_additions(question: str, score_fn=None) -> str:
    """
    Same call signature as builder.py's build_dynamic_prompt_additions() —
    AgentOrchestrator calls this generically via self._builder_module (see
    agent/domains.py's DOMAIN_REGISTRY), so both domains' builder modules
    must match this shape.
    """
    conditional = build_conditional_blocks_audit(question, score_fn=score_fn)
    worked = build_worked_examples_audit(question, score_fn=score_fn)
    if question and not conditional and not worked:
        _log_prompts.debug(
            "[PROMPTS] no audit conditional block/worked-example matched: %r", question[:120]
        )
    return conditional + worked
