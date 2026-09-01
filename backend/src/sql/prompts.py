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

from src.knowledge.question_patterns import MA_WORKFLOW_TOPIC, FINANCIAL_ROUTING_TOPIC, AI_ANSWERS_TOPIC

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

# Moved out of the always-on SQL_GENERATION_PROMPT skeleton (Phase 3.7
# latency profiling, 2026-08-19) — was baked directly into
# sql_generation_prompt.txt's "### AI FLAT TABLES" section and sent on
# EVERY inspection-domain generate_sql call regardless of whether the
# question touched form-answer data at all (~1250 tokens, unlike the
# CAPEX/P&L block above, which was already gated). Gated on AI_ANSWERS_TOPIC
# (question_patterns.py — see its docstring for the validation methodology:
# ground-truthed against generate_sql()'s actual output on all 99 real
# single-turn eval.py questions, TP=20 TN=76 FP=1 FN=2).
_AI_ANSWERS_BLOCK = """\
### AI FLAT TABLES — use these for all form question/answer data:

  ai_questions EXACT columns (use ONLY these — nothing else exists):
    element_id   uuid
    label        text   ← this is the question text / display name
    entity_type  varchar
    module_id    uuid
    module_name  text
    created_at   timestamp
  DO NOT USE: question_label, page_name, required, question_type, form_name
              (none of these columns exist on ai_questions)

  ai_answers EXACT columns (use ONLY these — nothing else exists):
    id                   uuid
    element_id           uuid  → JOIN to ai_questions.element_id
    inspection_report_id uuid  → JOIN to inspection_report.id
    inspection_id        varchar
    answer_text          text   ← normalized plain string (use this for all filtering/display)
    answer_text_raw      text   ← original pre-normalization value (JSON arrays/uuids) — for audit only
    answer_numeric       numeric
    score                numeric  ← weighted points earned for this answer (range 1–27 in data)
    score_type           varchar
    max_score            numeric  ← maximum possible points for this question
    module_name          text
    submission_id        text     ← groups ALL answers for one repeatable entry (e.g. one Observation)
    created_on            timestamp without time zone  ← confirmed live 2026-08-10, needed for
                          ORDER BY when reconstructing a whole form (see below) — id tiebreak
                          required too: a submission's answers are written in one batch and
                          commonly share the same created_on

  aa.score vs inspection_report.inspection_score:
    aa.score              = points earned for ONE answer to ONE question (weighted per question)
    inspection_report.inspection_score = overall % score for the whole inspection report
    SUM(aa.score) per inspection ≠ inspection_report.inspection_score (different scales)

  aa.score USE CASES:
    CRITICAL: aa.score is ONLY populated for these question labels:
      'Impact'           — MULTI-SELECT category field (comma-separated when multiple chosen)
                           answer_text values: 'Client Relationship', 'Health & Safety (HS)',
                           'Financial', 'Environmental', 'Legal / Regulatory', 'Reputational',
                           'Operational - Asset Downtime', 'Operational - Manpower/Resource Impact',
                           'Operational - Service Failure', 'Quality'
                           aa.score = weighted points based on which categories were selected (1–27)
                           Higher score = more/higher-weight categories selected
                           NEVER treat Impact as High/Medium/Low severity — it is category-based
                           To filter by category: aa.answer_text ILIKE '%Health & Safety%'
                           NEVER inspection_corrective_action.impact_id for "impact of this
                           OBSERVATION" — that's a SEPARATE, later value on the mitigative
                           action itself (only exists once a MA is raised). Use ica.impact_id
                           only for "impact of this MA" — a question about the ACTION, not the
                           observation. Confirmed live 2026-08-10 alongside the identical
                           Risk Level fix below.
      'Risk Level'       — FORM CONFIG ARTIFACT: score is set at form publication time, NOT by the inspector's answer
                           Current forms always score=4; legacy forms always score=1 regardless of answer
                           NEVER use aa.score WHERE label='Risk Level' for severity analysis — it is meaningless
                           To analyse actual risk level: use aa.answer_text ILIKE '%High%' / '%Medium%' etc.
                           answer_text values: 'High', 'Medium', 'Low', 'No Active Risk'
                           NEVER inspection_corrective_action.risk_level_id for this — that's a
                           SEPARATE, later value on the mitigative action itself (only exists once
                           a MA is raised), not the observation's own Risk Level answer.
                           To pull the paired Observation TEXT alongside the Risk Level filter,
                           self-join ai_answers to itself matched on submission_id (the value that
                           identifies "same repeatable entry") — NEVER on inspection_report_id,
                           which pairs answers from ANY entry in the report, not necessarily the
                           SAME one a High-risk Risk Level answer came from. Confirmed live
                           2026-08-10 (see BUSINESS_RULES' RISK LEVEL note): two seed examples in
                           this codebase had exactly this bug before being fixed.
      'Observation Type' — type score (range observed: 1, also a form config artifact — not meaningful)
                           answer_text values: 'Deviation', 'Good Practice'
    'Observation' (free-text) has NO score — NEVER use WHERE aq.label = 'Observation' AND aa.score IS NOT NULL

    SEVERITY SIGNAL — aa.score WHERE label='Impact' is the ONLY valid severity metric from ai_answers:
    "observation with highest impact score" → WHERE aq.label = 'Impact' ORDER BY aa.score DESC LIMIT N
    "highest scoring answer for inspector X" → ORDER BY aa.score DESC WHERE aq.label = 'Impact'
    "inspector with best form scores"        → GROUP BY inspector, AVG(aa.score) WHERE aq.label = 'Impact'
    "facility with best scored answers"      → GROUP BY fac.name, AVG(aa.score) WHERE aq.label = 'Impact'
    "total score for an inspection"          → SUM(aa.score) WHERE inspection_report_id = X AND aq.label = 'Impact'
    ALWAYS filter: WHERE aa.score IS NOT NULL AND aa.module_name = 'Inspection Form'
    module_id            uuid
    submitted_on         timestamptz
    status               varchar
  DO NOT USE: answer_value, form_name, question_id, source_table
              (none of these columns exist on ai_answers)

  Standard JOIN:
    FROM ai_answers aa
  LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id

  Join to inspection metadata:
    JOIN inspection_report ir ON aa.inspection_report_id = ir.id

  CRITICAL — answer_text encoding:
    Dropdown/checkbox answers are stored as plain strings (normalized).
    e.g. Risk Level:          'High', 'Medium', 'Low', 'No Active Risk'  (single value — = works)
         Observation Type:    'Deviation', 'Good Practice'               (single value — = works)
         Impact (multi-select): 'Health & Safety (HS), Environmental'   (comma-joined — ILIKE required)
    Single-value fields: exact match aa.answer_text = 'High' works; ILIKE '%High%' also safe.
    Impact MUST use ILIKE: aa.answer_text ILIKE '%Health & Safety%'  (value is a comma-joined list)
    For distribution/grouping: GROUP BY aa.answer_text returns clean readable values.
    Raw pre-normalization values (JSON arrays with uuids) are in aa.answer_text_raw.
    Known values: 'High', 'Medium', 'Low', 'No Active Risk', 'Deviation', 'Good Practice'

  CRITICAL — ai_answers spans ALL 16 modules:
    When user says 'the form', 'the inspection form', 'what they filled' without
    specifying a module — filter to the main form only:
      AND aa.module_name = 'Inspection Form'
    Without this, queries return answers from all 16 modules (~10,000 rows instead of ~50).
    Only omit this filter when user explicitly asks about another module.

  CRITICAL — most recent inspection subquery (use aa.submitted_on, NOT ir.submitted_on):
    CORRECT:   WHERE aa.inspection_report_id = (
                   SELECT aa2.inspection_report_id FROM ai_answers aa2
                   ORDER BY aa2.submitted_on DESC LIMIT 1)
    WRONG:     WHERE ir.id = (SELECT id FROM inspection_report ORDER BY ir.submitted_on DESC LIMIT 1)
               (ir.submitted_on and aa.submitted_on differ — use aa.submitted_on for most-recently-answered)\
"""

# Moved out of the always-on SQL_GENERATION_PROMPT skeleton (Phase 3.19
# token audit) — same rationale as _CAPEX_PNL_BLOCK/_AI_ANSWERS_BLOCK above:
# these two rules only matter for questions touching ai_answers/ai_questions
# form-answer data, but were sent on EVERY inspection-domain generate_sql
# call regardless. Gated on AI_ANSWERS_TOPIC, the same already-validated
# pattern _AI_ANSWERS_BLOCK above uses.
_EXACT_QUESTION_LABELS_BLOCK = """- EXACT QUESTION LABELS (use = not ILIKE for known labels):
  'Observation'      — free-text field written by inspector. NOT a type/category.
  'Observation Type' — dropdown: 'Deviation' | 'Good Practice'  ← use THIS to count deviations or GP
  'Risk Level'       — dropdown: High, Medium, Low, No Active Risk  (form answers only — NOT ICA risk)
  'Impact'           — multi-select impact categories

  CRITICAL: NEVER use aq.label ILIKE '%Observation%' — matches BOTH labels and overcounts.
  Count Deviations:    WHERE aq.label = 'Observation Type' AND aa.answer_text ILIKE '%Deviation%'
  Count Good Practice: WHERE aq.label = 'Observation Type' AND aa.answer_text ILIKE '%Good Practice%'
  Read free-text:      WHERE aq.label = 'Observation'
"""

_FACILITY_LEVEL_NEGATION_BLOCK = """- FACILITY-LEVEL NEGATION (has NEVER had X across ALL history):
  WRONG — only checks per inspection:
    GROUP BY fac.name HAVING COUNT(CASE WHEN ... THEN 1 END) = 0
  CORRECT — NOT IN subquery at facility level:
    WHERE fac.id NOT IN (
        SELECT DISTINCT ir.facility_id FROM ai_answers aa
        JOIN ai_questions aq ON aa.element_id = aq.element_id
        JOIN inspection_report ir ON aa.inspection_report_id = ir.id
        WHERE aq.label = 'Observation Type'
          AND aa.answer_text ILIKE '%Good Practice%'
          AND aa.module_name = 'Inspection Form'
          AND ir.active = true AND ir.status != 'DRAFT'
    )
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
        "title": "ai_answers / ai_questions form-answer schema (Deepseek)",
        "question_pattern": AI_ANSWERS_TOPIC,
        "get_text": lambda: _AI_ANSWERS_BLOCK,
    },
    # Phase 3.19 — sql_generation_prompt.txt skeleton audit's two remaining
    # topic-narrow rules, same AI_ANSWERS_TOPIC gate as the block above.
    {
        "title": "Exact question labels (Deepseek)",
        "question_pattern": AI_ANSWERS_TOPIC,
        "get_text": lambda: _EXACT_QUESTION_LABELS_BLOCK,
    },
    {
        "title": "Facility-level negation pattern (Deepseek)",
        "question_pattern": AI_ANSWERS_TOPIC,
        "get_text": lambda: _FACILITY_LEVEL_NEGATION_BLOCK,
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


def profile_core_context_composition(base_url: str) -> dict:
    """
    Phase 3.17 Part B: real /tokenize breakdown of CORE_CONTEXT's always-on
    static base and SQL_GENERATION_PROMPT's fixed skeleton — the two pieces
    of the SQL-gen prompt Phase 3.8 measured only as opaque top-level totals
    (core_context_tokens / template_or_other). core_context.txt mixes 3
    registry-block placeholders with prose in a way that doesn't cleanly
    slice by marker the way system_prompt.txt's boxed headings do (the
    prose runs continuously around each placeholder) — reported by
    subtraction instead: STATUS_BLOCK/CAST_BLOCK/COLUMNS_BLOCK's real costs
    are measured directly (they're already isolated string constants), and
    the remaining prose is CORE_CONTEXT's real total minus those three, same
    reconciliation-by-subtraction convention this project already uses
    elsewhere (e.g. schema_introspector.py's schema_hint "boilerplate").
    Also reports each CORE_CONTEXT_CONDITIONAL block's real cost (replacing
    the code-comment token estimates like "~805 tok" with measured numbers).
    All of this text is invariant per process — computed once.
    """
    from src.llm.tokenize import count_tokens_llamacpp
    static_total = count_tokens_llamacpp(CORE_CONTEXT, base_url)
    named_blocks = {"status_block": _STATUS_BLOCK, "cast_block": _CAST_BLOCK, "columns_block": _COLUMNS_BLOCK}
    named_blocks_tok = {
        name: {"tokens": count_tokens_llamacpp(text, base_url), "chars": len(text)}
        for name, text in named_blocks.items()
    }
    conditional_tok = {}
    for blk in CORE_CONTEXT_CONDITIONAL:
        txt = blk["get_text"]()
        conditional_tok[blk["title"]] = {
            "tokens": count_tokens_llamacpp(txt, base_url),
            "chars": len(txt),
            "question_pattern": blk["question_pattern"],
        }
    return {
        "core_context_static_total_tokens": static_total,
        "named_blocks": named_blocks_tok,
        "prose_only_tokens": static_total - sum(v["tokens"] for v in named_blocks_tok.values()),
        "conditional_blocks": conditional_tok,
        "sql_generation_prompt_skeleton_tokens": count_tokens_llamacpp(SQL_GENERATION_PROMPT, base_url),
    }
