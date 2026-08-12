"""
SQL-generation prompt assembly ("Deepseek-side"): SQL_GENERATION_PROMPT (the
base template) and CORE_CONTEXT (registry-derived status/cast/columns facts
plus conditional counting/MA-workflow blocks), both consumed by
sql/generation.py::generate_sql(). Companion to agent/prompts/builder.py,
which assembles the reasoning-LLM's SYSTEM_PROMPT instead — SYSTEM_PROMPT
and SQL_GENERATION_PROMPT are different prompts for different models and
were merged from two separate builder.py files (agent/prompts,
former agent/examples) that used to live in the agent package.

STATUS GROUPINGS + CAST RULES come from src.knowledge.semantics_registry -- the
single source of truth. Edit there, not here.
"""

import logging
import re
from pathlib import Path

from src.knowledge.question_patterns import MA_WORKFLOW_TOPIC, FINANCIAL_ROUTING_TOPIC

_log_prompts = logging.getLogger("inspectai.prompts")

_TEMPLATES_DIR = Path(__file__).parent / "examples" / "templates"


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


SQL_GENERATION_PROMPT = _read_template("sql_generation_prompt.txt")


# ── Registry-generated blocks (single source of truth) ──────────────────────
try:
    from src.knowledge.semantics_registry import (
        core_context_status_block as _status_block,
        core_context_cast_block as _cast_block,
        core_context_counting_block as _counting_block,
        core_context_columns_block as _columns_block,
        core_context_ma_workflow_block as _ma_workflow_block,
    )
    _STATUS_BLOCK = _status_block()
    _CAST_BLOCK = _cast_block()
    _COUNTING_BLOCK = _counting_block()
    _COLUMNS_BLOCK = _columns_block()
    _MA_WORKFLOW_BLOCK = _ma_workflow_block()
except ImportError:
    # Fallback — keep in sync with knowledge/semantics_registry.py
    _STATUS_BLOCK = _read_template("status_block_fallback.txt")
    _CAST_BLOCK = _read_template("cast_block_fallback.txt")
    _COUNTING_BLOCK = _read_template("counting_block_fallback.txt")
    _COLUMNS_BLOCK = _read_template("columns_block_fallback.txt")
    _MA_WORKFLOW_BLOCK = _read_template("ma_workflow_block_fallback.txt")

CORE_CONTEXT = (
    _read_template("core_context.txt")
    .replace("<<<STATUS_BLOCK>>>", _STATUS_BLOCK)
    .replace("<<<CAST_BLOCK>>>", _CAST_BLOCK)
    .replace("<<<COLUMNS_BLOCK>>>", _COLUMNS_BLOCK)
)

# Moved out of the always-on SQL_GENERATION_PROMPT skeleton (token-usage
# audit, 2026-08-06) — was baked directly into sql_generation_prompt.txt
# ("### CAPEX CONVERSION" / "### P&L (PROFIT & LOSS) MODEL", 76 lines) and
# sent on EVERY inspection-domain generate_sql call regardless of whether
# the question touched finance at all — the single biggest reason the
# inspection skeleton was ~1.8x the size of audit's (which has no
# CAPEX/OPEX/P&L content to begin with, so nothing comparable to move).
# Gated on FINANCIAL_ROUTING_TOPIC, the SAME pattern inject_rules.py's
# inject_financial_routing and examples_tier2.py's "Total capex generated
# converted and lost" example already gate on — single source of truth for
# "is this question about CAPEX/OPEX/P&L", not a new regex.
_CAPEX_PNL_BLOCK = """\
### CAPEX CONVERSION — CRITICAL DEFINITIONS:
  CAPEX Generated  = SUM(ica.capex)              — the FM manager's quoted/estimated amount on each ICA
  CAPEX Converted  = SUM(po_details.po_amount)   — what the client actually paid via PO (money HAS moved)
  CAPEX Lost       = SUM(ica.capex) - COALESCE(SUM(po.po_amount), 0)
                     — the arithmetic gap between quoted and paid. This is NOT capex_status='REJECTED'.
                     — user phrases that mean CAPEX Lost: "didn't get converted", "not converted",
                       "unconverted capex", "capex gap", "capex shortfall", "wasted capex",
                       "how much capex was lost", "capex that wasn't paid"
  OPEX Incurred    = SUM(pr_details.pr_amount)   — Khidmah's own internal spend via PR (bad: org paying itself)
  Conversion Rate  = SUM(po.po_amount) / NULLIF(SUM(ica.capex), 0) * 100

  CAPEX GROUPING — "entity" and "facility" are DIFFERENT tables, NEVER treat as synonyms:
    Facility: JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id
    Entity:   JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN entity ent ON ir.entity_id = ent.id
    Region:   ... → facility → region
    Project:  JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN project proj ON ir.project_id = proj.id
    FM Mgr:   JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN users u ON ir.inspectee_user_id = u.id
  If the user says "by entity", GROUP BY ent.name (entity table) — do NOT substitute facility.

  ica.expenditure is a CATEGORY FLAG ('CAPEX','OPEX','CAPEX_AND_OPEX','NO_EXPENDITURE')
  NEVER use ica.expenditure to compute any financial amount — it is a category string, not a number
  NEVER use capex_status='ACCEPTED' as a proxy for conversion — use po_details
  NEVER use capex_status='REJECTED' to compute CAPEX Lost — use the arithmetic difference above
  CAPEX Lost ≠ rejected CAPEX: a user asking about "wasted" or "unconverted" capex means the
    arithmetic gap, not the subset where capex_status='REJECTED'. Only use capex_status='REJECTED'
    when the user explicitly says "rejected" and wants to see those specific MA rows.

  FAN-OUT RULE — ALWAYS pre-aggregate po_details and pr_details in CTEs before joining to ica.
  One ICA can have MULTIPLE POs and MULTIPLE PRs — a direct LEFT JOIN multiplies rows silently.
  NEVER: LEFT JOIN po_details p ON p.mitigative_action_uuid = ica.id  (causes fan-out)
  ALWAYS use CTE pre-aggregation (or scalar subqueries for single-row totals):
    WITH po_agg AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
                    FROM po_details WHERE active = true GROUP BY 1)
    LEFT JOIN po_agg po ON po.ica_id = ica.id

### P&L (PROFIT & LOSS) MODEL — CRITICAL:
  INCOME / CAPEX Converted = SUM(po_details.po_amount)   — client paid Khidmah (PO = Purchase Order)
  COST   / OPEX Incurred   = SUM(pr_details.pr_amount)   — Khidmah paid out internally (PR = Payment Request)
  PROFIT                   = SUM(po.po_amount) - SUM(pr.pr_amount)
  LOSS                     = when COST > INCOME (negative profit)

  "OPEX" in user language always means SUM(pr_details.pr_amount) — the PR spend.
  "income" / "revenue" in user language always means SUM(po_details.po_amount).
  NEVER use ica.capex as income or revenue — it is a QUOTE/ESTIMATE, no money has moved.

  These totals are NOT fixed — they change as new POs/PRs are entered.
  ALWAYS compute Income/Cost/Profit LIVE via SUM(). NEVER state remembered/previous totals.

  P&L JOIN PATTERN — always pre-aggregate in CTEs to avoid row fan-out:
    WITH po_agg AS (
        SELECT mitigative_action_uuid AS ica_id,
               SUM(po_amount) AS total_income
        FROM po_details WHERE active = true GROUP BY 1
    ),
    pr_agg AS (
        SELECT mitigative_action_uuid AS ica_id,
               SUM(pr_amount) AS total_cost
        FROM pr_details WHERE active = true GROUP BY 1
    )
    SELECT ...,
           COALESCE(SUM(po.total_income), 0) AS income,
           COALESCE(SUM(pr.total_cost),  0) AS cost,
           COALESCE(SUM(po.total_income), 0) - COALESCE(SUM(pr.total_cost), 0) AS profit
    FROM inspection_corrective_action ica
    LEFT JOIN po_agg po ON po.ica_id = ica.id
    LEFT JOIN pr_agg pr ON pr.ica_id = ica.id
    ...GROUP BY ...

  P&L GROUPING — can be broken down by:
    Facility:   JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id
    FM Manager: JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN users u ON ir.inspectee_user_id = u.id
    Region:     ... → facility → region
    Entity:     JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN entity ent ON ir.entity_id = ent.id
    Project:    JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN project proj ON ir.project_id = proj.id
    Country:    JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id JOIN country c ON fac.country_id = c.id\
"""

# ── Conditional CORE_CONTEXT additions — included only when relevant ────────
# Deepseek-side counterpart to agent/prompts/builder.py's CONDITIONAL_BLOCKS /
# EXAMPLES_TIER2. _COUNTING_BLOCK (good practice/deviation/observation
# counting, ~148 tok) and _MA_WORKFLOW_BLOCK (MA workflow columns, ~805 tok)
# were always-on parts of CORE_CONTEXT before 2026-06-13's truncation fix —
# both grew this session and are narrowly topical, so they're now conditional
# like their Llama-side counterparts (CONDITIONAL_BLOCKS["MA workflow
# columns"] and the FORM-pattern Tier-2 examples in agent/prompts/builder.py).
#
# The MA pattern is shared with agent/prompts/builder.py's CONDITIONAL_BLOCKS
# via src.knowledge.question_patterns.MA_WORKFLOW_TOPIC (single source of truth --
# edit there, not here) so both sides treat "is this question about
# mitigative actions" the same way. The counting pattern below has no
# cross-file counterpart today and stays defined inline.
CORE_CONTEXT_CONDITIONAL = [
    {
        "title": "MA workflow columns (Deepseek)",
        "question_pattern": MA_WORKFLOW_TOPIC,
        "get_text": lambda: _MA_WORKFLOW_BLOCK,
    },
    {
        "title": "CAPEX/OPEX/P&L definitions (Deepseek)",
        "question_pattern": FINANCIAL_ROUTING_TOPIC,
        "get_text": lambda: _CAPEX_PNL_BLOCK,
    },
    {
        "title": "Good practice / deviation / observation counting (Deepseek)",
        # Tightened (token-usage audit, 2026-08-06) — the old pattern matched
        # any bare "form(s)"/"answer(s)"/"question(s)"/"observation(s)"/
        # "risk level(s)", which fires on most real questions regardless of
        # whether they're actually about the COUNT(*) vs COUNT(DISTINCT ...)
        # trap this block exists to fix (see core_context_counting_block()
        # above — it's specifically about good-practice/deviation/
        # observation COUNTING, nothing broader). Scoped to an actual
        # counting verb/noun paired with one of those three topics.
        "question_pattern": (
            r"(?i)\b(good\s+practices?|deviations?|observations?)\b.{0,60}"
            r"\b(count|counts?|how\s+many|number\s+of|total)\b"
            r"|\b(count|counts?|how\s+many|number\s+of|total)\b.{0,60}"
            r"\b(good\s+practices?|deviations?|observations?)\b"
            r"|\bobservations?\s+per\s+inspection\b"
        ),
        "get_text": lambda: _COUNTING_BLOCK,
    },
]


def build_core_context_additions(question: str) -> str:
    """
    Return additional CORE_CONTEXT blocks whose question_pattern matches the
    current question. Called from sql/generation.py::generate_sql(): append
    the result to CORE_CONTEXT before building dynamic_examples.

    Returns "" if nothing matches — no size penalty for unrelated questions.
    """
    if not question:
        return ""
    matched = []
    for blk in CORE_CONTEXT_CONDITIONAL:
        try:
            if re.search(blk["question_pattern"], question):
                matched.append(blk["get_text"]())
        except re.error:
            continue
    if not matched:
        # Silent-miss signal — see agent/prompts/builder.py's build_dynamic_prompt_additions
        # for the reasoning-side counterpart and the full rationale.
        _log_prompts.debug(
            "[PROMPTS] no SQL-side conditional block matched: %r", question[:120]
        )
        return ""
    return "\n--\n" + "\n--\n".join(matched)
