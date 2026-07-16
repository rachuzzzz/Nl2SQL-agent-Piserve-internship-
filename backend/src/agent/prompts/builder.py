"""
Prompt templates for the agentic NL2SQL system.

STATUS GROUPINGS come from src.core.semantics_registry — the single source of
truth. Edit there, not here.

The large prompt bodies (SYSTEM_PROMPT, SQL_GENERATION_PROMPT,
FORCE_SUMMARY_PROMPT) live as plain text in templates/*.txt rather than
Python triple-quoted strings — same content, easier to review/diff as
prose. SYSTEM_PROMPT's template has <<<STATUS_BLOCK>>> / <<<COLUMNS_BLOCK>>>
sentinels marking where the registry-generated blocks below get spliced in.
EXAMPLES_TIER2 (worked examples, ~550 lines) is pure Python data (regex
question_patterns + text) and lives in examples_tier2.py instead.
"""

import re
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


# ── Registry-generated status block (single source of truth) ─────────────────
try:
    from src.core.semantics_registry import (
        llama_status_block as _llama_status_block,
        llama_ma_workflow_block as _llama_ma_workflow_block,
        llama_columns_block as _llama_columns_block,
    )
    _STATUS_BLOCK = _llama_status_block()
    _MA_WORKFLOW_BLOCK = _llama_ma_workflow_block()
    _COLUMNS_BLOCK = _llama_columns_block()
except ImportError:
    # Fallback — keep in sync with core/semantics_registry.py
    _STATUS_BLOCK = _read_template("status_block_fallback.txt")
    _MA_WORKFLOW_BLOCK = _read_template("ma_workflow_block_fallback.txt")
    _COLUMNS_BLOCK = _read_template("columns_block_fallback.txt")

SYSTEM_PROMPT = (
    _read_template("system_prompt.txt")
    .replace("<<<STATUS_BLOCK>>>", _STATUS_BLOCK)
    .replace("<<<COLUMNS_BLOCK>>>", _COLUMNS_BLOCK)
)

SQL_GENERATION_PROMPT = _read_template("sql_generation_prompt.txt")

FORCE_SUMMARY_PROMPT = _read_template("force_summary_prompt.txt")

# ── Tier-2 worked examples — included only when question_pattern matches ────
# These are narrow-topic examples (CAPEX/FM-manager, cycle-frequency, MA
# approval workflow, schedule+cycle joins) that were previously part of the
# always-on SYSTEM_PROMPT WORKED EXAMPLES section (~5K chars across 6 blocks).
# Each maps to (or reuses) a corresponding business_rules.py INJECT rule's
# question_pattern where one exists, so Deepseek's "USE THIS EXACT SQL" hint
# and Llama's few-shot fire on the SAME questions — single source of truth
# for "is this question about topic X".
#
# question_pattern=None would mean "always include" (Tier 1) — none of these
# are None; Tier-1 examples remain inline in SYSTEM_PROMPT as before.
from src.agent.prompts.examples_tier2 import EXAMPLES_TIER2

# ── Per-question budget cap for multi-match Tier-2/conditional blocks ───────
# Without this, 2+ simultaneously-matching EXAMPLES_TIER2/CONDITIONAL_BLOCKS
# entries concatenate unbounded onto SYSTEM_PROMPT — observed as multiple
# regexes (e.g. "capex" + "approval workflow") legitimately co-firing and
# match the same question (see _select_within_budget) — single-match questions
# are never capped, so the common case is untouched. EXAMPLES_TIER2 blocks
# range from ~700 to ~12,900 chars; this budget is sized so a multi-match
# question costs roughly what ONE large block already costs alone, rather
# than letting matches stack unbounded (observed: 2 blocks = 8,764 chars for
# a single "capex" query, contributing to a reasoning-LLM prompt that nearly
# filled its 20K-token context window and produced an empty completion).
_DEFAULT_BLOCK_BUDGET_CHARS = 6000


def _select_within_budget(
    candidates: list[tuple[str, str]],
    question: str,
    score_fn,
    budget_chars: int,
) -> list[str]:
    """
    candidates: [(title, text), ...] — already topic-gated by regex match.

    Without score_fn (the default everywhere this is called unless the
    caller opts in), or when there's only one candidate, returns every text
    unchanged — today's behavior, zero cost, zero risk.

    With score_fn(question, title) -> float supplied, ranks candidates by
    score descending and greedily keeps them within budget_chars, ALWAYS
    keeping at least the single best-scored block regardless of its size
    (so one large, highly-relevant block is never dropped) — additional
    blocks are included only while they still fit.
    """
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


def build_worked_examples(
    question: str, score_fn=None, budget_chars: int = _DEFAULT_BLOCK_BUDGET_CHARS
) -> str:
    """
    Return additional WORKED EXAMPLES text for narrow topics whose
    question_pattern matches the current question. Append the result to
    SYSTEM_PROMPT at per-call prompt-assembly time (see orchestrator.py
    query()). Tier-1 (broadly-applicable) examples remain inline in
    SYSTEM_PROMPT and are NOT duplicated here.

    Returns "" if no Tier-2 example matches — i.e. no size penalty for
    questions outside these narrow topics.

    `score_fn`, if supplied, ranks and budget-caps matches when 2+ fire
    together — see _select_within_budget.
    """
    if not question:
        return ""
    matched = []
    for ex in EXAMPLES_TIER2:
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


# ── Conditional registry-derived blocks — included only when relevant ───────
# Unlike EXAMPLES_TIER2 (Turn-1 worked examples), these are semantic FACT
# blocks generated by core.semantics_registry. MA_WORKFLOW_BLOCK (521 tokens)
# was previously always-on; moved here 2026-06-13 as part of the num_ctx
# truncation fix — it's entirely about inspection_corrective_action (ica)
# workflow columns (status/progress_stage/implementation_status/capex_status/
# timeliness) and is only relevant when the question is about mitigative
# actions at all.
CONDITIONAL_BLOCKS = [
    {
        "title": "MA workflow columns",
        "question_pattern": (
            r"(?i)\bmitigative\s+actions?\b|\bMAs?\b"
            r"|\bcapex\b|\bopex\b|\bdeferred\b"
            r"|\bprogress[_\s]stage\b|\bimplementation[_\s]status\b"
            r"|\bneeds?\s+attention\b|\bcritical\s+issues?\b"
        ),
        "get_text": lambda: _MA_WORKFLOW_BLOCK,
    },
]


def build_conditional_blocks(
    question: str, score_fn=None, budget_chars: int = _DEFAULT_BLOCK_BUDGET_CHARS
) -> str:
    """
    Return additional registry-derived semantic blocks whose question_pattern
    matches the current question. Companion to build_worked_examples — call
    both at per-call prompt-assembly time (see orchestrator.py query()).

    Returns "" if nothing matches — no size penalty for unrelated questions.

    `score_fn`, if supplied, ranks and budget-caps matches when 2+ fire
    together — see _select_within_budget. Only one entry exists today, so
    this is a no-op in practice until CONDITIONAL_BLOCKS grows.
    """
    if not question:
        return ""
    matched = []
    for blk in CONDITIONAL_BLOCKS:
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
    Single integration point: combines build_worked_examples (Tier-2 few-shots)
    and build_conditional_blocks (registry semantic facts) into one string to
    append to SYSTEM_PROMPT. Call this from orchestrator.py instead of either
    function individually, so future conditional-injection categories only
    need to be added here.

    `score_fn(question, title) -> float`, if supplied, ranks and budget-caps
    same-question multi-block matches (see _select_within_budget) instead of
    concatenating every regex match unconditionally — without it, behavior is
    unchanged from before this parameter existed.
    """
    return (
        build_conditional_blocks(question, score_fn=score_fn)
        + build_worked_examples(question, score_fn=score_fn)
    )
