"""
CORE_CONTEXT (Deepseek/SQL-generation-side prompt additions) and the
question-triggered conditional blocks that supplement it.

STATUS GROUPINGS + CAST RULES come from src.core.semantics_registry — the
single source of truth. Edit there, not here.

CORE_CONTEXT's body lives in templates/core_context.txt with
<<<STATUS_BLOCK>>> / <<<CAST_BLOCK>>> / <<<COLUMNS_BLOCK>>> sentinels marking
where the registry-generated blocks below get spliced in. SEED_EXAMPLES (the
actual nl/sql example list, ~3900 lines) is pure Python data and lives in
seed_data.py instead — see its module docstring for the field schema.
"""

import re
from pathlib import Path

from src.core.question_patterns import MA_WORKFLOW_TOPIC

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


# ── Registry-generated blocks (single source of truth) ──────────────────────
try:
    from src.core.semantics_registry import (
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
    # Fallback — keep in sync with core/semantics_registry.py
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

from src.agent.examples.seed_data import SEED_EXAMPLES

# ── Conditional CORE_CONTEXT additions — included only when relevant ────────
# Deepseek-side counterpart to prompts.py's CONDITIONAL_BLOCKS / EXAMPLES_TIER2.
# _COUNTING_BLOCK (good practice/deviation/observation counting, ~148 tok) and
# _MA_WORKFLOW_BLOCK (MA workflow columns, ~805 tok) were always-on parts of
# CORE_CONTEXT before 2026-06-13's truncation fix — both grew this session and
# are narrowly topical, so they're now conditional like their Llama-side
# counterparts (CONDITIONAL_BLOCKS["MA workflow columns"] and the FORM-pattern
# Tier-2 examples in prompts.py).
#
# The MA pattern is shared with prompts.py's CONDITIONAL_BLOCKS via
# src.core.question_patterns.MA_WORKFLOW_TOPIC (single source of truth —
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
    current question. Call from tools.py's generate_sql: append the result to
    CORE_CONTEXT before building dynamic_examples.

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
        return ""
    return "\n--\n" + "\n--\n".join(matched)
