"""
Prompt templates for the reasoning ("Llama-side") half of the agentic NL2SQL
system. The SQL-generation ("Deepseek-side") prompt, SQL_GENERATION_PROMPT,
lives in src.sql.prompts instead — a different prompt for a different model,
consumed by sql/generation.py, not by the orchestrator directly.

STATUS GROUPINGS come from src.knowledge.semantics_registry — the single source of
truth. Edit there, not here.

The large prompt bodies (SYSTEM_PROMPT, FORCE_SUMMARY_PROMPT) live as plain
text in templates/*.txt rather than Python triple-quoted strings — same
content, easier to review/diff as prose. SYSTEM_PROMPT's template has
<<<STATUS_BLOCK>>> / <<<COLUMNS_BLOCK>>> sentinels marking where the
registry-generated blocks below get spliced in. EXAMPLES_TIER2 (worked
examples, ~550 lines) is pure Python data (regex question_patterns + text)
and lives in examples_tier2.py instead.
"""

import logging
import re
from pathlib import Path

from src.knowledge.question_patterns import (
    MA_WORKFLOW_TOPIC,
    USER_ROLES_TOPIC as _USER_ROLES_TOPIC,
    APPROVAL_WORKFLOW_TOPIC as _APPROVAL_WORKFLOW_TOPIC,
    COMMUNICATION_GROUPS_TOPIC as _COMMUNICATION_GROUPS_TOPIC,
    FINANCIAL_ROUTING_TOPIC as _FINANCIAL_ROUTING_TOPIC,
)
from src.agent.tool_defs import declared_tools as _declared_tools
from src.knowledge.schema_introspector import INSPECTION_RULE_TOPIC_TEXT

_log_prompts = logging.getLogger("inspectai.prompts")

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


# ── Registry-generated status block (single source of truth) ─────────────────
try:
    from src.knowledge.semantics_registry import (
        llama_status_block as _llama_status_block,
        llama_ma_workflow_block as _llama_ma_workflow_block,
        llama_columns_block as _llama_columns_block,
    )
    _STATUS_BLOCK = _llama_status_block()
    _MA_WORKFLOW_BLOCK = _llama_ma_workflow_block()
    _COLUMNS_BLOCK = _llama_columns_block()
except ImportError:
    # Fallback — keep in sync with knowledge/semantics_registry.py
    _STATUS_BLOCK = _read_template("status_block_fallback.txt")
    _MA_WORKFLOW_BLOCK = _read_template("ma_workflow_block_fallback.txt")
    _COLUMNS_BLOCK = _read_template("columns_block_fallback.txt")

# Tools declared via agent/tool_defs (see that package's docstring) instead
# of hand-edited into system_prompt.txt directly — appended after the
# static numbered tool list, which is never itself regenerated.
_DYNAMIC_TOOLS_BLOCK = "".join(
    t.doc for t in _declared_tools() if "inspection" in t.domains
)

# Phase 3.18: P&L financial model moved out of the always-on SYSTEM_PROMPT
# template into CONDITIONAL_BLOCKS below (same gating mechanism as
# MA_WORKFLOW_BLOCK) — see that list's "P&L financial model" entry.
# ("Form answers" was also tried here and reverted — see CONDITIONAL_BLOCKS'
# comment for why; that content stays in system_prompt.txt, always-on.)
# system_prompt.txt's separate "OBSERVATION -> MITIGATIVE ACTION" restatement
# (previously ~line 300-330) was deleted outright rather than gated — it was
# a duplicate of knowledge/schema_introspector.py's BUSINESS_RULES_CORE
# entry of the same name, which stays core/always-on in its own (already
# proven reliable — see that file's "observation_ma" gating-reverted note)
# position; removing the duplicate doesn't touch that copy at all.
_PNL_FINANCIAL_MODEL_BLOCK = _read_template("pnl_financial_model_block.txt")

SYSTEM_PROMPT = (
    _read_template("system_prompt.txt")
    .replace("<<<STATUS_BLOCK>>>", _STATUS_BLOCK)
    .replace("<<<COLUMNS_BLOCK>>>", _COLUMNS_BLOCK)
    .replace("<<<DYNAMIC_TOOLS_BLOCK>>>", _DYNAMIC_TOOLS_BLOCK)
)

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
    # Evidence for a future retrieval-aware "bucketed" prefix cache
    # (PrefixCache MVP's Option B — not built yet): which EXAMPLES_TIER2
    # titles fired for this question, so offline log analysis can measure
    # how often different questions match the same block set before anyone
    # builds caching logic around it. Logs the pre-budget match set (not
    # _select_within_budget's post-cap survivors) — good enough signal for
    # "which topics this question touched" without threading titles through
    # _select_within_budget's return contract.
    _log_prompts.debug(
        "[PROMPTS] worked-example titles matched: %r -> %s",
        question[:120], [title for title, _ in matched],
    )
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
# ── Topic-gated business-rules/table-relationships remainder (2026-08 token
# audit) — see knowledge/schema_introspector.py's "Topic-gated core/topical
# split" for what moved here and why (BUSINESS_RULES/TABLE_RELATIONSHIPS
# were previously spliced in FULL and unconditionally by orchestrator/loop.py;
# now only the CORE subset is always-on and this narrow-topic remainder is
# surfaced only when the question matches, same as MA_WORKFLOW_BLOCK below).
# Patterns are deliberately broad — a false positive only costs tokens, a
# false negative silently drops a "confirmed live" bug-fix rule.
#
# NOTE: "observation_ma" (OBSERVATION -> MITIGATIVE ACTION join rule) was
# gated here originally but reverted to core 2026-08-10 — see
# knowledge/schema_introspector.py's matching note for the eval regression
# (FLT-02/FLT-03) that motivated the revert.
#
# _USER_ROLES_TOPIC/_APPROVAL_WORKFLOW_TOPIC/_COMMUNICATION_GROUPS_TOPIC now
# live in knowledge/question_patterns.py (imported above) — Phase 3.16 reuses
# the same three patterns to topic-gate DatabaseSchema.for_schema_hint's
# table_relationships content on the SQL-gen side, so both prompt layers
# agree on which niche topics a question touches.

CONDITIONAL_BLOCKS = [
    {
        "title": "MA workflow columns",
        "question_pattern": MA_WORKFLOW_TOPIC,
        "get_text": lambda: _MA_WORKFLOW_BLOCK,
    },
    {
        "title": "User roles",
        "question_pattern": _USER_ROLES_TOPIC,
        "get_text": lambda: INSPECTION_RULE_TOPIC_TEXT["user_roles"],
    },
    {
        "title": "Approval workflow",
        "question_pattern": _APPROVAL_WORKFLOW_TOPIC,
        "get_text": lambda: INSPECTION_RULE_TOPIC_TEXT["approval_workflow"],
    },
    {
        "title": "Communication groups",
        "question_pattern": _COMMUNICATION_GROUPS_TOPIC,
        "get_text": lambda: INSPECTION_RULE_TOPIC_TEXT["communication_groups"],
    },
    # Phase 3.18 — token audit's REASONING PROMPT COMPOSITION findings.
    # FINANCIAL_ROUTING_TOPIC is the same pattern already validated with
    # real TP/TN/FP/FN numbers against the eval corpus for the SQL-gen
    # side's identical-topic CORE_CONTEXT_CONDITIONAL block.
    #
    # "Form answers" was ALSO gated here (behind AI_ANSWERS_TOPIC) and then
    # reverted the same phase — full-suite validation caught a real
    # regression: MT-01 ("show me the most recently answered inspection")
    # needs this section's ai_answers routing guidance but doesn't match
    # AI_ANSWERS_TOPIC (\banswers?\b requires the exact word "answer(s)";
    # "answered" is a different word to a word-boundary regex), silently
    # dropping it and cascading into a broken MT-01→04 multi-turn chain.
    # AI_ANSWERS_TOPIC is shared (question_patterns.py) with 3 other
    # consumers that already have their own validated firing-rate data for
    # THEIR phrasing patterns — broadening it to also catch "answered" was
    # judged too risky to do blind, un-validated, for those other 3
    # consumers just to fix this one. form_answers stays always-on in
    # system_prompt.txt instead; see that file's FORM ANSWERS section.
    {
        "title": "P&L financial model",
        "question_pattern": _FINANCIAL_ROUTING_TOPIC,
        "get_text": lambda: _PNL_FINANCIAL_MODEL_BLOCK,
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
    # See build_worked_examples' matching comment above — same Option-B
    # evidence-gathering rationale, applied to CONDITIONAL_BLOCKS instead.
    _log_prompts.debug(
        "[PROMPTS] conditional-block titles matched: %r -> %s",
        question[:120], [title for title, _ in matched],
    )
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
    conditional = build_conditional_blocks(question, score_fn=score_fn)
    worked = build_worked_examples(question, score_fn=score_fn)
    if question and not conditional and not worked:
        # Silent-miss signal: no CONDITIONAL_BLOCKS/EXAMPLES_TIER2 pattern matched
        # this question at all. Not an error — most questions match nothing — but
        # tracking it in logs lets topic-pattern drift (new phrasings real users
        # try that the regexes never anticipated) become visible over time instead
        # of invisible, since a miss here degrades context silently otherwise.
        _log_prompts.debug(
            "[PROMPTS] no reasoning-side conditional block/worked example matched: %r",
            question[:120],
        )
    return conditional + worked


# ── Phase 3.17 — real /tokenize composition audit ────────────────────────────
# Companion to knowledge/schema_introspector.py's SQL-gen-side profiling
# (profile_slim_schema_tokens/profile_targeted_schema_tables/
# profile_fixed_overhead_relevance) — same "computed once, invariant text"
# rationale, applied to the reasoning-LLM side, which had ZERO token
# instrumentation before this phase despite being the larger prompt (its
# always-on static base alone is ~4-5x schema_hint's pre-3.15 size).
#
# system_prompt.txt mixes a boxed ("━━━...") heading convention for its
# major sections with plain "TITLE:" sub-headers in between (unlike
# TABLE_RELATIONSHIPS's uniform column-0 heading style
# schema_introspector._split_relationship_sections parses generically) — so
# boundaries are hand-listed here rather than regex-derived. Each marker is
# a distinctive substring confirmed (by reading templates/system_prompt.txt
# directly) to appear exactly once — a ValueError here means the template
# changed and this list needs updating, which is the intended failure mode
# (loud, not a silent miscount).
_SYSTEM_PROMPT_SECTION_MARKERS = [
    ("preamble_response_format", None),  # start of string
    ("ambiguity_rule", "AMBIGUITY RULE — Ask the user only when it genuinely matters:"),
    ("domain_mismatch_check", "DOMAIN-MISMATCH CHECK — CRITICAL:"),
    # Spans the {inspection_schema} placeholder — when split against the
    # orchestrator's already-assembled self._system_prompt (not the raw
    # SYSTEM_PROMPT constant), this bucket naturally includes the real
    # rendered for_reasoning_prompt()/business_rules_core/table_relationships_core
    # content, since the placeholder sits inside this section's span.
    ("schema_routing_lookup", "INSPECTION WORKFLOW — plain relational tables"),
    ("form_answers", "FORM ANSWERS — flat ai_* tables"),
    # Phase 3.18: "pnl_financial_model" and "observation_ma_paths" were
    # removed from this list — the former moved out of the always-on
    # template into CONDITIONAL_BLOCKS ("P&L financial model"; "form_answers"
    # was ALSO tried there and reverted, see that entry's comment — it stays
    # here, always-on), and the latter was deleted outright as a confirmed
    # duplicate of BUSINESS_RULES_CORE's own "OBSERVATION -> MITIGATIVE
    # ACTION" entry (which remains, unaffected, inside "schema_routing_lookup"
    # below). Neither marker exists in system_prompt.txt any more.
    ("available_tools", "AVAILABLE TOOLS\n"),
    ("follow_up_query_rules", "FOLLOW-UP QUERY RULES — CRITICAL:"),
    ("resolved_reference_disclosure", "RESOLVED-REFERENCE DISCLOSURE RULE — CRITICAL"),
    ("ambiguous_reference_rule", "AMBIGUOUS REFERENCE RULE — check before fetching by partial name"),
    ("important_distinctions", "IMPORTANT DISTINCTIONS\n"),
    ("worked_examples_tier1", "WORKED EXAMPLES\n"),
    ("closing_reminder", "Remember: respond with ONLY valid JSON. No other text."),
]


def _split_system_prompt_sections(text: str) -> list[tuple[str, str]]:
    """Cut `text` (SYSTEM_PROMPT, or an orchestrator's already-assembled
    self._system_prompt) into the sections _SYSTEM_PROMPT_SECTION_MARKERS
    names, in order. Returns [(label, section_text), ...]."""
    idxs = [(label, 0 if marker is None else text.index(marker))
            for label, marker in _SYSTEM_PROMPT_SECTION_MARKERS]
    return [
        (label, text[start:(idxs[n + 1][1] if n + 1 < len(idxs) else len(text))])
        for n, (label, start) in enumerate(idxs)
    ]


def profile_prompt_composition(system_prompt_text: str, base_url: str) -> dict:
    """
    Phase 3.17: real /tokenize breakdown of the reasoning-LLM prompt's
    always-on static base (system_prompt_text — pass the orchestrator's
    live self._system_prompt so the schema/rules/relationships content
    inside "schema_routing_lookup" is the REAL rendered text, not a
    reconstruction) plus the real per-entry cost of every
    CONDITIONAL_BLOCKS/EXAMPLES_TIER2 candidate. All of this text is
    invariant per process — computed once, same rationale as
    schema_introspector.py's SQL-gen profiling methods.

    Per-call variable cost (which CONDITIONAL_BLOCKS/EXAMPLES_TIER2 entries
    actually fired for a given question, plus conversation-history cost) is
    NOT reconstructed from these per-entry numbers — eval.py instead calls
    build_dynamic_prompt_additions(question) directly and tokenizes its
    real return value, avoiding the exact "reconstruction drifts from
    production logic" trap 3.13 hit once already.
    """
    from src.llm.tokenize import count_tokens_llamacpp
    sections = {
        label: {"text": text, "tokens": count_tokens_llamacpp(text, base_url), "chars": len(text)}
        for label, text in _split_system_prompt_sections(system_prompt_text)
    }
    named_blocks = {
        "status_block": _STATUS_BLOCK,
        "columns_block": _COLUMNS_BLOCK,
        "dynamic_tools_block": _DYNAMIC_TOOLS_BLOCK,
        "ma_workflow_block": _MA_WORKFLOW_BLOCK,
    }
    named_blocks_tok = {
        name: {"tokens": count_tokens_llamacpp(text, base_url), "chars": len(text)}
        for name, text in named_blocks.items()
    }
    conditional_blocks_tok = {}
    for blk in CONDITIONAL_BLOCKS:
        txt = blk["get_text"]()
        conditional_blocks_tok[blk["title"]] = {
            "tokens": count_tokens_llamacpp(txt, base_url),
            "chars": len(txt),
            "question_pattern": blk["question_pattern"],
        }
    examples_tier2_tok = {}
    for ex in EXAMPLES_TIER2:
        examples_tier2_tok[ex["title"]] = {
            "tokens": count_tokens_llamacpp(ex["text"], base_url),
            "chars": len(ex["text"]),
            "question_pattern": ex["question_pattern"],
        }
    return {
        "system_prompt_sections": sections,
        "system_prompt_total_tokens": count_tokens_llamacpp(system_prompt_text, base_url),
        "named_blocks": named_blocks_tok,
        "conditional_blocks": conditional_blocks_tok,
        "examples_tier2": examples_tier2_tok,
    }
