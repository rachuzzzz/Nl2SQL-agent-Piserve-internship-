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

from src.knowledge.question_patterns import MA_WORKFLOW_TOPIC

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
        "title": "Good practice / deviation / observation counting (Deepseek)",
        "question_pattern": (
            r"(?i)\b(forms?|answer\w*|questions?|observations?|risk\s*levels?"
            r"|deviations?|good\s+practice|total\s+score|ai_answers)\b"
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
