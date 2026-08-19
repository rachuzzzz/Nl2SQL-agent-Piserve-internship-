"""
SQL generation pipeline: schema-block selection, retrieval-based few-shot
injection, the LLM call itself, table/UUID retry guards, AST repair, and
validation. Extracted from agent/tools.py, where this pipeline was filed as
a single "tool" among several simpler query tools — generation is the
system's actual core NL2SQL capability, not a tool wrapper.
"""

import re
from typing import Any, Optional

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

from src.sql.validation import SQLValidator, ValidationResult
from src.llm.base import extract_token_counts, extract_llamacpp_timings, TokenUsage
from src.llm.prefix_cache import PrefixCacheManager


_schema_cache: Optional[str] = None


def _fetch_key_schemas(db_engine: Engine) -> str:
    """Auto-inject actual column names for ai_* and inspection_* tables."""
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache
    try:
        inspector = sa_inspect(db_engine)
        key_tables = ["ai_questions", "ai_answers",
                      "inspection_report", "inspection_corrective_action"]
        parts = ["### Actual database columns (use these exactly):"]
        for table in key_tables:
            try:
                columns = inspector.get_columns(table)
                col_list = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
                parts.append(f"  {table}: {col_list}")
            except Exception:
                pass
        _schema_cache = "\n".join(parts)
        return _schema_cache
    except Exception:
        return ""


_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _extract_table_names_from_sql(text: str) -> list[str]:
    """
    Pull candidate table names out of a raw (possibly draft/invalid) SQL
    string — used to derive a join-path-hint fallback from the reasoning
    model's OWN schema_hint when it didn't separately supply relevant_tables
    (see _select_schema_block's fallback_tables param for why this exists).

    Deliberately permissive: dedups preserving first-seen order, doesn't
    validate against a real table list (path_as_hint/find_join_path already
    degrade gracefully — return None — for any name that isn't a real graph
    node, confirmed by test_bogus_table_in_pair_degrades_gracefully), and
    may occasionally pick up a CTE alias as if it were a table. Worst case
    is a missed hint for that pair, never a crash or a fabricated one — the
    join graph itself is the source of truth for whether a pair resolves.
    """
    seen: list[str] = []
    for m in _TABLE_REF_RE.finditer(text or ""):
        name = m.group(1).lower()
        if name not in seen:
            seen.append(name)
    return seen


_SQL_START_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


def _looks_like_sql(text: str) -> bool:
    """Cheap classifier for the legacy schema_hint compat path: does this
    freeform string look like a runnable SQL draft, or is it prose notes
    (e.g. "table=inspection_report, metric=avg(...)")? Deliberately
    permissive — a false positive just means a non-SQL string renders
    under the "Planner draft" heading instead of being treated as notes,
    which is harmless (the coder model already treats that section as
    editable/possibly-wrong)."""
    return bool(_SQL_START_RE.match(text or ""))


def _resolve_planner_contract(
    schema_hint: str, draft_sql: str, join_path: str, business_rules,
) -> tuple[str, str]:
    """
    Resolve the legacy single-string schema_hint shape and the new
    structured planner fields (draft_sql/join_path/business_rules) into one
    combined llm_hint string, plus the actual SQL draft text alone (for
    table-name extraction and guard-scanning — join_path/business_rules
    prose is never valid SQL and shouldn't be scanned as if it were).

    Compat rule:
      - draft_sql supplied  -> that's the draft; schema_hint (if any) is
        additional freeform notes alongside it.
      - draft_sql absent, schema_hint looks like SQL -> schema_hint IS the
        draft (today's only shape, before this plan's Item 3).
      - neither -> no draft; schema_hint (if any) is notes-only, e.g. the
        "table=X, metric=avg(...)" shorthand some worked examples use.

    Returns (llm_hint, draft_sql_only) — llm_hint is what renders under
    "Planner draft" in the coder prompt; draft_sql_only is the resolved SQL
    text alone, used for _extract_table_names_from_sql and (by the caller)
    guard-scan fidelity checks.
    """
    if draft_sql:
        resolved_draft = draft_sql
        notes = schema_hint
    elif _looks_like_sql(schema_hint):
        resolved_draft = schema_hint
        notes = ""
    else:
        resolved_draft = ""
        notes = schema_hint

    lines = []
    if notes:
        lines.append(notes)
    if join_path:
        lines.append(f"Suggested join path: {join_path}")
    if business_rules:
        _rules_text = (business_rules if isinstance(business_rules, str)
                        else "; ".join(str(r) for r in business_rules))
        lines.append(
            "Planner-noted business rules (verify against the schema "
            f"reference and Required section above): {_rules_text}"
        )
    if resolved_draft:
        lines.append(resolved_draft)

    return "\n".join(lines), resolved_draft


def _build_additional_context(auto_schema: str, llm_hint: str, mandatory_hint: str) -> str:
    """
    Render the three registers that make up "{schema_hint}" in the prompt
    template as separately labelled sub-sections, instead of one
    undifferentiated blob.

    Confirmed live 2026-08-07 (qid=e63f03af): before this split, a genuinely
    mandatory INJECT-rule hint ("USE THIS EXACT SQL ... do not deviate")
    and the reasoning model's own freeform guess sat in the same
    concatenated string with no way for the coder model to tell which was
    which. A blanket "this is all editable" note would have told the model
    it could ignore the mandatory part too — the actual fix is telling it
    which part is which, not softening everything uniformly.

    auto_schema     — factual, always correct (curated schema/join block).
    llm_hint        — the reasoning model's own draft; may be wrong.
    mandatory_hint  — deterministic requirements (INJECT rules, LIMIT-N);
                       must be followed exactly.
    """
    parts: list[str] = []
    if auto_schema:
        parts.append(f"#### Schema reference — factual\n{auto_schema}")
    if mandatory_hint:
        parts.append(f"#### Required — follow exactly, do not deviate\n{mandatory_hint}")
    if llm_hint:
        parts.append(
            "#### Planner draft — editable, may be wrong\n"
            "The block below is the planning model's first attempt at this "
            "query. It is a STARTING POINT, not a specification. You may "
            "rewrite its joins, substitute different tables, add or remove "
            "predicates, and restructure it entirely if the schema "
            "reference above or the PRIMARY EXAMPLE below shows a better "
            "fit. Resolved literals in it (UUIDs, exact IDs, LIMIT/OFFSET "
            "values) were resolved deliberately — keep those.\n"
            f"{llm_hint}"
        )
    return "\n\n".join(parts) if parts else "No additional context provided."


def _select_schema_block(
    db_schema, relevant_tables: list | None, db_engine=None, join_graph_module=None,
    fallback_tables: list | None = None,
) -> str:
    """
    Prefer a targeted, full-column-detail block for the tables the reasoning
    model said are relevant (DatabaseSchema.for_schema_hint) over the static
    curated slim block (for_sql_prompt_slim) that's sent on every call
    regardless of question — replacement, not blend, since for_schema_hint
    already includes its own lookup-tables + TABLE_RELATIONSHIPS sections.

    relevant_tables is LLM-supplied and may be malformed (e.g. a bare string
    instead of a JSON array, since this is the only list-typed tool argument
    in this codebase) — the isinstance guard prevents silently iterating
    individual characters as "table names" if that happens.

    join_graph_module: the domain's join-graph module (sql/join_graph.py for
    inspection, sql/audit_join_graph.py for audit — see agent/domains.py's
    DOMAIN_REGISTRY). None defaults to inspection's, preserving behavior for
    every existing caller.

    fallback_tables: confirmed live (2026-07-30, 2026-08-04) that
    relevant_tables is essentially NEVER populated in practice — checked 58
    real generate_sql calls across 5 sessions spanning 5 days, zero hits.
    Root cause: the system prompt frames it as needed only for tables
    "beyond the default schema context", but the tables where the join-path
    hint matters most (audit_report/audit_schedule/audit_corrective_action)
    are ALREADY in that default context, so the reasoning model has no
    reason to ever list them. Rather than rely on prompt-compliance for a
    fix, fallback_tables carries table names extracted from the reasoning
    model's OWN schema_hint (see _extract_table_names_from_sql) — supplied
    on nearly every call, unlike relevant_tables — and is used ONLY to
    still append the join-path hint onto the slim block; it deliberately
    does NOT switch to the larger for_schema_hint() per-table block (that
    stays gated on explicit relevant_tables) to preserve the token-budget
    tradeoff for_sql_prompt_slim() exists for.
    """
    if db_schema is not None:
        if isinstance(relevant_tables, list) and relevant_tables:
            clean = [t for t in relevant_tables if isinstance(t, str) and t.strip()]
            if clean:
                block = db_schema.for_schema_hint(clean)
                if len(clean) >= 2:
                    block += _join_path_hints(clean, join_graph_module)
                return block
        block = db_schema.for_sql_prompt_slim()
        if fallback_tables and len(fallback_tables) >= 2:
            block += _join_path_hints(fallback_tables, join_graph_module)
        return block
    if db_engine:
        return _fetch_key_schemas(db_engine)
    return ""


def _join_graph_source(table_name: str) -> str:
    """
    Inspection's JOIN_GRAPH (sql/join_graph.py) has no "users" key — only
    role-specific "users_inspector"/"users_inspectee" pseudo-nodes. Default to
    the inspector role when a caller says "users" with no role context; this
    is a heuristic, not a real disambiguation — informational hint text only,
    never used to filter/scope data. Audit's graph (sql/audit_join_graph.py)
    similarly has no bare "users" key (users_auditor/users_auditee/
    users_proposer/users_internal_stakeholder instead) — default to the
    auditor role there for the same reason.
    """
    return "users_inspector" if table_name == "users" else table_name


def _join_graph_source_audit(table_name: str) -> str:
    return "users_auditor" if table_name == "users" else table_name


def _join_path_hints(relevant_tables: list, join_graph_module=None) -> str:
    """Chain path_as_hint() across consecutive relevant_tables pairs, using
    the given domain's join-graph module (default: inspection's)."""
    if join_graph_module is not None:
        path_as_hint = join_graph_module.path_as_hint
        _source = _join_graph_source_audit if join_graph_module.__name__.endswith("audit_join_graph") else _join_graph_source
    else:
        from src.sql.join_graph import path_as_hint
        _source = _join_graph_source
    hints = []
    for a, b in zip(relevant_tables, relevant_tables[1:]):
        hint = path_as_hint(_source(a), _source(b))
        if hint:
            hints.append(hint)
    return "".join(hints)


def _build_guard_retry_prompt(question: str, additional_context: str, raw_sql: str, mandatory_block: str) -> str:
    """
    Minimal reconstructed prompt for a guard-forced retry (offset/table/uuid
    guards below) — deliberately NOT the full original `prompt`. A guard
    fires because the model dropped or violated one specific, already-
    diagnosed instruction; fixing that doesn't need the full skeleton
    template's naming/style rules, CORE_CONTEXT, or the 4 retrieved few-shot
    examples again — the model already used all of that correctly enough to
    produce `raw_sql` in the first place, it just needs the schema/join
    context (additional_context — the same three-sub-section rendering
    built by _build_additional_context for the main prompt, see
    generate_sql above) and the one specific fix.

    Confirmed live (token-usage audit, 2026-08-06): reusing the FULL
    original prompt on every retry was the single largest contributor to
    this app's worst-case 56-80K-token queries — 2-3 guard retries in one
    turn meant re-sending the same ~7-8K-token prompt 2-3 times over for
    what's usually a single dropped clause. Previously each guard also
    re-appended its instruction onto the tail of the ORIGINAL prompt (one
    guard's own "collapse schema_hint" step was silently dead code — its
    regex matched a header, "### Schema Hint (follow this guidance)", that
    neither real template actually uses; both say "### Additional context"
    — so that retry was sending MORE tokens than the initial call, not
    fewer). This replaces all three guards' prompt-reuse with one shared,
    genuinely small reconstruction instead of patching that regex, since
    the smaller prompt is strictly better once schema_hint is included.
    """
    return (
        f"### Schema context\n{additional_context or 'No additional context provided.'}\n\n"
        f"### Question\n{question}\n\n"
        f"### Your previous SQL (needs correction)\n{raw_sql}\n\n"
        f"{mandatory_block}\n"
        f"[SQL]\n"
    )


def generate_sql(
    sql_llm, validator: SQLValidator, question: str,
    schema_hint: str = "", db_engine=None,
    sql_prompt_template: str = "",
    draft_sql: str = "",
    join_path: str = "",
    business_rules: list | str | None = None,
    _mandatory_hint: str = "",
    _debug_logger=None,
    _inject_rule_ids: list | None = None,
    _seed_index=None,
    _db_schema=None,
    _active_cycle_uuid: str = "",
    _retry_attempt: int = 0,
    # True when the caller (loop.py) knows this call, if it also fails
    # validation/execution, will trip the outer 3-attempt cap and end the
    # turn in a refusal/clarification — see loop.py's _last_attempt comment.
    # When True, the offset/table/uuid guard retries below skip firing:
    # whatever they'd produce is still judged by that same terminal cap on
    # the next pass, so the extra sql_llm.complete() call(s) they'd make
    # can only add latency/tokens, never change the outcome. Guards on
    # attempts 1-2 (the default, False) are unaffected — a guard fixing the
    # SQL there can still save the query, so they still fire.
    _last_attempt: bool = False,
    relevant_tables: list | None = None,
    _core_context: str | None = None,
    _build_core_context_fn=None,
    _join_graph_module=None,
    _rules_engine=None,
    _domain: str = "inspection",
) -> dict[str, Any]:
    try:
        # Planner->SQL structured contract (generation-quality plan Item 3):
        # resolve draft_sql/join_path/business_rules plus the legacy
        # single-string schema_hint shape into one combined llm_hint text
        # (schema_hint local var is reassigned to it below, so every
        # downstream use of "schema_hint" — additional_context, the guard
        # scans, the debug prompt log — automatically covers the new
        # fields with no further plumbing) and the resolved SQL draft
        # alone, for table extraction and fidelity checks.
        schema_hint, _draft_sql_only = _resolve_planner_contract(
            schema_hint, draft_sql, join_path, business_rules,
        )

        # Captured BEFORE the auto_schema concatenation below so the
        # schema-hint-fidelity guard (further down) can extract table names
        # from exactly what the reasoning model itself proposed — the
        # curated auto_schema block appended next is a general column
        # reference, not a specific table requirement, and would pollute
        # the extraction with unrelated tables if included. Uses the
        # resolved SQL draft alone (not the combined llm_hint, which may
        # also carry join_path/business_rules prose that isn't SQL and
        # shouldn't be scanned for FROM/JOIN table references).
        _llama_schema_hint = _draft_sql_only

        # Schema block — the slim version keeps context under ~600 tokens.
        # A full introspected column dump for every table (~2000+ tokens) inflates
        # deepseek's context 3x and degrades 7B model generation quality.
        # When the reasoning model supplies relevant_tables, use a targeted
        # full-detail block for just those tables instead (see _select_schema_block).
        #
        # fallback_tables: relevant_tables is confirmed almost never supplied
        # in practice (see _select_schema_block's docstring) — derive a
        # fallback from the tables the reasoning model's OWN schema_hint
        # already names, so the join-path hint still has a chance to fire
        # without depending on that separate, effectively-unused argument.
        auto_schema = _select_schema_block(
            _db_schema, relevant_tables, db_engine, _join_graph_module,
            fallback_tables=_extract_table_names_from_sql(_llama_schema_hint),
        )

        # Three registers, three authorities — rendered as separately
        # labelled sub-sections instead of one concatenated blob (see
        # _build_additional_context's docstring for the qid=e63f03af
        # incident this fixes). _guard_scan_text is what the offset/uuid
        # guards below scan for resolved literals the coder model must
        # preserve — auto_schema is deliberately excluded (it's a factual
        # column/table reference, never carries a resolved OFFSET/UUID).
        #
        # schema_hint.startswith("Use"): preserves the pre-existing
        # suppression of auto_schema when the reasoning model's own draft
        # already opens with its own "Use ..." guidance — behavior-
        # preserving, not new: inject/mandatory hints were always appended
        # after the LLM's own text (never prepended), so this check has
        # only ever tested the LLM's own hint, both before and after the
        # _mandatory_hint split.
        _auto_schema_for_context = "" if schema_hint.startswith("Use") else auto_schema
        additional_context = _build_additional_context(_auto_schema_for_context, schema_hint, _mandatory_hint)
        _guard_scan_text = "\n".join(filter(None, [schema_hint, _mandatory_hint]))

        if not sql_prompt_template:
            from src.sql.prompts import SQL_GENERATION_PROMPT
            sql_prompt_template = SQL_GENERATION_PROMPT

        # Retrieval-based example injection via SeedExampleIndex (cosine similarity)
        # Falls back to empty string if index not available or seed_examples not deployed.
        # _core_context/_build_core_context_fn let the caller (ToolRegistry, per
        # domain — see agent/domains.py's DOMAIN_REGISTRY) supply the audit
        # domain's own vocabulary block instead of inspection's; None means "use
        # inspection defaults", preserving behavior for every existing caller.
        if _core_context is not None:
            _fn = _build_core_context_fn or (lambda q: "")
            try:
                _core_ctx = _core_context + _fn(question)
            except Exception:
                _core_ctx = _core_context
        else:
            try:
                from src.sql.prompts import CORE_CONTEXT, build_core_context_additions
                _core_ctx = CORE_CONTEXT + build_core_context_additions(question)
            except ImportError:
                _core_ctx = (
                    "-- ir.id = UUID (JOIN only, never SELECT), ir.inspection_id = VARCHAR display\n"
                    "-- ICA join: ON ica.inspection_id = ir.id | completed = IN ('CLOSED','SUBMITTED')\n"
                    "-- risk_level_id is UUID FK: JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
                )
        if _seed_index is not None:
            try:
                # On a retry, a weak SQL-gen model tends to reproduce its
                # closest seed example almost verbatim — including whatever
                # fragment triggered the validator failure — ignoring the
                # planner's corrected schema_hint entirely. Dropping examples
                # forces the model to actually use schema_hint instead of
                # re-copying the same broken pattern (see validator retry
                # loop in orchestrator.py which increments gen_sql_failures).
                #
                # Escalate the drop count with each retry rather than always
                # dropping just the #1 match — confirmed live: a persistently
                # contaminating example that happens to rank #2/#3/#4 (never
                # #1) survives a flat "drop #1" on every single retry, since
                # that only ever removes whichever example is #1 THAT round.
                # One real trace burned 8 retries because the same seed
                # example (missing a JOIN its WHERE clause still referenced)
                # reappeared in the prompt every time. Capped at 3 so at
                # least one example always survives at top_k=4.
                retrieved, _primary_example_key = _seed_index.retrieve_with_primary_key(
                    question, top_k=4, skip_top_n=min(_retry_attempt, 3)
                )
            except Exception:
                retrieved = ""
                _primary_example_key = ""
        else:
            retrieved = ""
            _primary_example_key = ""
        # Substitute <CYCLE_UUID> placeholder with actual active cycle UUID if available
        if _active_cycle_uuid and retrieved:
            retrieved = retrieved.replace("<CYCLE_UUID>", _active_cycle_uuid)
        dynamic_examples = _core_ctx + ("\n" + retrieved if retrieved else "")

        prompt = sql_prompt_template.replace(
            "{question}", question
        ).replace(
            "{schema_hint}", additional_context
        ).replace(
            "{dynamic_examples}", dynamic_examples
        )

        # "deepseek" in these labels is a hardcoded legacy name, not the
        # actual configured model — this project moved off DeepSeek to Ollama
        # (sqlcoder/qwen) and now optionally Claude (see llm_provider) long
        # ago without the label ever being updated, so it stopped reflecting
        # what really produced the output. sql_llm.model is a real field on
        # both llm/ollama.py's and llm/anthropic.py's client objects, so this
        # stays accurate across any future provider swap too.
        _sql_model_name = getattr(sql_llm, "model", "?")

        # Log the full prompt sent to the SQL-gen model if debug enabled
        if _debug_logger:
            _debug_logger.debug(
                f"\n{'='*70}\n>>> SQL-GEN PROMPT ({_sql_model_name}) (question={question!r})\n"
                f"{'='*70}\n{prompt[-3000:]}\n"  # last 3000 chars — schema_hint now sits right before the question (dynamic_examples moved earlier, see loop.py's _sql_static_prefix comment)
            )

        # Accumulated across every sql_llm.complete() call this invocation
        # makes (initial + up to 3 guard retries below) — each is a real,
        # separately-billed API call, so the total returned to the caller
        # must be the sum, not just the last call's count. Cache fields are
        # Anthropic-only (0 on Ollama/vLLM) — see llm/base.py's TokenUsage.
        _prompt_tok_total = 0
        _completion_tok_total = 0
        _cache_read_tok_total = 0
        _cache_creation_tok_total = 0
        # llama.cpp-only prefill/decode split, summed across every
        # sql_llm.complete() call this invocation makes — same
        # accumulate-across-retries rationale as the token counters above.
        # See llm/base.py's LlamaCppTimings and Phase 3.2 profiling.
        _prefill_ms_total = 0.0
        _decode_ms_total = 0.0
        _prefix_cache = PrefixCacheManager(sql_llm, domain=_domain, role="sql")

        response = sql_llm.complete(prompt)
        raw_sql = str(response).strip()
        _raw_response = getattr(response, "raw", {}) or {}
        _usage = extract_token_counts(_raw_response)
        _prompt_tok_total += _usage.prompt_tokens
        _completion_tok_total += _usage.completion_tokens
        _cache_read_tok_total += _usage.cache_read_tokens
        _cache_creation_tok_total += _usage.cache_creation_tokens
        _prefix_cache.record(_raw_response)
        _timing = extract_llamacpp_timings(_raw_response)
        _prefill_ms_total += _timing.prompt_ms
        _decode_ms_total += _timing.predicted_ms

        # Log raw SQL-gen model output
        if _debug_logger:
            _debug_logger.debug(
                f"\n{'='*70}\n>>> SQL-GEN RAW OUTPUT ({_sql_model_name})\n{'='*70}\n{raw_sql}\n"
            )

        # ── Ordinal-offset guard: when schema_hint resolves "the Nth one"
        # via LIMIT 1 OFFSET <n> (the fallback the reasoning model uses when
        # it can't pull a literal UUID for the ordinal straight out of
        # conversation history — see the uuid-guard below for the sibling
        # case where it CAN), DeepSeek must preserve that row-selection
        # mechanism somewhere in its query. Dropping it silently discards
        # "the 4th one" entirely, and — unlike a missing UUID literal, which
        # is a single findable substring — there's no way to check for its
        # *substance* being preserved under an expanded/rewritten query, so
        # this only checks that an OFFSET clause survives at all.
        #
        # Confirmed live 2026-08-03 (qid=4d1d0bf4): given schema_hint
        # "SELECT ac.id AS cycle_id, ... FROM audit_cycle ac ... LIMIT 1
        # OFFSET 3" (the reasoning model's own thought said "get the 4th
        # cycle's id, then use it in a separate query"), DeepSeek instead
        # reproduced an unrelated retrieved example nearly verbatim ("which
        # category and type does this auditor portfolio cover") — joining
        # audit_cycle, auditor_portfolio, users, auditor_portfolio_detail,
        # facility, and audit_category with NO LIMIT/OFFSET at all, so the
        # query ran against every cycle's portfolios at once instead of just
        # the 4th one, and dropped the status filter as a side effect of
        # copying the wrong example. It validated and executed cleanly (a
        # wrong-scope query rarely errors the way a wrong-column one does),
        # so nothing caught it until a human compared the result against the
        # original question — this class of bug is strictly worse than an
        # execution failure, since it returns a confident, plausible-
        # looking, entirely wrong-scope answer instead of triggering a
        # retry. A plain table-overlap check would have MISSED this: the
        # bad query still legitimately touched audit_cycle, it just dropped
        # the row-selector that pinned it to one specific row. ────────────
        try:
            if re.search(r'\bOFFSET\s+\d+', _guard_scan_text or "", re.IGNORECASE):
                if not re.search(r'\bOFFSET\s+\d+', raw_sql, re.IGNORECASE) and _last_attempt:
                    if _debug_logger:
                        _debug_logger.debug(
                            "  [offset-guard] skipped — last permitted attempt, "
                            "a retry here can't change the outcome"
                        )
                elif not re.search(r'\bOFFSET\s+\d+', raw_sql, re.IGNORECASE):
                    _mandatory = (
                        "### MANDATORY (do not deviate)\n"
                        "Your schema_hint above resolves \"the Nth one\" via a LIMIT 1 OFFSET "
                        "clause on audit_cycle — your query MUST include that same LIMIT/OFFSET "
                        "row-selection (directly, or via a subquery/CTE using it to pin the "
                        "specific cycle). Do NOT drop it and query across every cycle instead, "
                        "and do NOT substitute a different table structure copied from one of "
                        "the examples."
                    )
                    forced_prompt = _build_guard_retry_prompt(question, additional_context, raw_sql, _mandatory)
                    import logging as _logging
                    _logging.getLogger("inspectai").warning(
                        "[offset-guard] retry fired — schema_hint has OFFSET, raw_sql doesn't"
                    )
                    if _debug_logger:
                        _debug_logger.debug(
                            "  [offset-guard] firing retry — schema_hint has OFFSET, raw_sql doesn't"
                        )
                    response_og = sql_llm.complete(forced_prompt)
                    raw_sql = str(response_og).strip()
                    _usage = extract_token_counts(getattr(response_og, "raw", {}) or {})
                    _prompt_tok_total += _usage.prompt_tokens
                    _completion_tok_total += _usage.completion_tokens
                    _cache_read_tok_total += _usage.cache_read_tokens
                    _cache_creation_tok_total += _usage.cache_creation_tokens
                    _prefix_cache.record(getattr(response_og, "raw", {}) or {})
                    _timing = extract_llamacpp_timings(getattr(response_og, "raw", {}) or {})
                    _prefill_ms_total += _timing.prompt_ms
                    _decode_ms_total += _timing.predicted_ms
                    if _debug_logger:
                        _debug_logger.debug(
                            f"\n{'='*70}\n>>> SQL-GEN RAW OUTPUT ({_sql_model_name}) (offset-guard retry)"
                            f"\n{'='*70}\n{raw_sql}\n"
                        )

                    if not re.search(r'\bOFFSET\s+\d+', raw_sql, re.IGNORECASE):
                        _logging.getLogger("inspectai").warning(
                            "[offset-guard] retry did NOT fix it — still no OFFSET"
                        )
                        if _debug_logger:
                            _debug_logger.debug(
                                "  [offset-guard] retry unsuccessful — still no OFFSET"
                            )
        except Exception as _og_exc:
            if _debug_logger:
                _debug_logger.debug(f"  [offset-guard] skipped: {_og_exc}")
        # ── end ordinal-offset guard ─────────────────────────────────────────

        # ── Table guard: if inject rules required specific tables and they're
        # missing from the generated SQL, force one retry with the requirement
        # prepended directly to the question (not just in schema_hint).
        # This turns INJECT rules from suggestions into contracts. ───────────
        if _inject_rule_ids:
            try:
                # Domain-correct engine, threaded from the caller (loop.py
                # already computes self._rules_engine per-domain via
                # DomainConfig.rules_engine_getter — see domains.py). Falls
                # back to the inspection singleton only when no engine was
                # passed at all, preserving old behavior for any caller that
                # doesn't thread this through yet. Confirmed live bug this
                # fixes: _inject_rule_ids for the AUDIT domain (e.g.
                # "audit_inject_auditor_auditee_from_ca") were being looked
                # up in the INSPECTION registry, which has no such ids —
                # required_tables_for/forbidden_patterns_for silently
                # returned [] every time, so the guard never enforced
                # anything for audit-domain questions.
                _guard_engine = _rules_engine
                if _guard_engine is None:
                    from src.sql.rules import get_engine as _get_rules_engine
                    _guard_engine = _get_rules_engine()   # singleton — no re-init cost
                required = _guard_engine.required_tables_for(_inject_rule_ids)
                forbidden = _guard_engine.forbidden_patterns_for(_inject_rule_ids)
                missing_req = [t for t in required if t.lower() not in raw_sql.lower()]
                violated_forbidden = []
                for fpat in forbidden:
                    try:
                        if re.search(fpat, raw_sql, re.IGNORECASE):
                            violated_forbidden.append(fpat)
                    except re.error:
                        pass

                if (missing_req or violated_forbidden) and _last_attempt:
                    if _debug_logger:
                        _debug_logger.debug(
                            "  [table-guard] skipped — last permitted attempt, "
                            "a retry here can't change the outcome"
                        )
                elif missing_req or violated_forbidden:
                    guard_msg_parts = []
                    if missing_req:
                        guard_msg_parts.append(f"MUST USE TABLES: {', '.join(missing_req)}")
                    if violated_forbidden:
                        guard_msg_parts.append(f"DO NOT USE: {', '.join(violated_forbidden)}")
                    guard_prefix = " | ".join(guard_msg_parts)
                    forced_question = f"[{guard_prefix}] {question}"
                    # Minimal reconstructed prompt (see _build_guard_retry_prompt) —
                    # replaces the old approach of reusing the full original `prompt`
                    # and trying to "slim" its schema_hint section via a regex that
                    # matched a header ("### Schema Hint (follow this guidance)")
                    # neither real template actually has (both say "### Additional
                    # context") — that regex silently no-op'd, so the old retry sent
                    # MORE tokens than the initial call, not fewer. An f-string here
                    # (not a regex-replacement template) also sidesteps the old
                    # backslash-escape concern entirely: guard_prefix can contain raw
                    # regex text from a rule's forbidden_patterns (e.g.
                    # "\baca\.audit_id\s*=..."), which only breaks re.sub's template
                    # parsing, not plain string interpolation.
                    _mandatory = f"### MANDATORY (do not deviate)\n{guard_prefix}"
                    forced_prompt = _build_guard_retry_prompt(forced_question, additional_context, raw_sql, _mandatory)
                    # Always log table-guard fires — doubles deepseek latency, important to track
                    import logging as _logging
                    _logging.getLogger("inspectai").warning(
                        f"[table-guard] retry fired — missing={missing_req} forbidden={violated_forbidden}"
                    )
                    if _debug_logger:
                        _debug_logger.debug(
                            f"  [table-guard] firing retry — missing={missing_req} forbidden_hit={violated_forbidden}"
                        )
                    response2 = sql_llm.complete(forced_prompt)
                    raw_sql = str(response2).strip()
                    _usage = extract_token_counts(getattr(response2, "raw", {}) or {})
                    _prompt_tok_total += _usage.prompt_tokens
                    _completion_tok_total += _usage.completion_tokens
                    _cache_read_tok_total += _usage.cache_read_tokens
                    _cache_creation_tok_total += _usage.cache_creation_tokens
                    _prefix_cache.record(getattr(response2, "raw", {}) or {})
                    _timing = extract_llamacpp_timings(getattr(response2, "raw", {}) or {})
                    _prefill_ms_total += _timing.prompt_ms
                    _decode_ms_total += _timing.predicted_ms

                    # Re-check: did the retry actually fix what triggered it?
                    # Without this, a small model ignoring the forced instruction
                    # a second time silently passes the still-broken SQL through
                    # to validation/execution as if the guard had succeeded.
                    _still_missing = [t for t in required if t.lower() not in raw_sql.lower()]
                    _still_violated = [
                        fpat for fpat in violated_forbidden
                        if re.search(fpat, raw_sql, re.IGNORECASE)
                    ]
                    if _still_missing or _still_violated:
                        _logging.getLogger("inspectai").warning(
                            f"[table-guard] retry did NOT fix it — still missing="
                            f"{_still_missing} still_violated={_still_violated}"
                        )
                        if _debug_logger:
                            _debug_logger.debug(
                                f"  [table-guard] retry unsuccessful — still_missing="
                                f"{_still_missing} still_violated={_still_violated}"
                            )
            except Exception as _guard_exc:
                if _debug_logger:
                    _debug_logger.debug(f"  [table-guard] skipped: {_guard_exc}")
        # ── end table guard ──────────────────────────────────────────────────

        # ── UUID-filter guard: schema_hint sometimes carries an already-resolved
        # identity filter, e.g. "WHERE u.id = '<uuid>'" after a name has been
        # disambiguated to a single user. DeepSeek can silently discard this and
        # invent its own ILIKE/name-based filter (or wrong FK column) instead —
        # same failure mode as the table-guard above, but for exact-identity
        # filters. Force one retry pinning the UUID if it's missing from the
        # generated SQL. ─────────────────────────────────────────────────────
        try:
            _UUID_RE_FRAG = (
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
            )
            _hint_uuid_filters = [
                (col, uid, "eq") for col, uid in re.findall(
                    r"([a-zA-Z_]\w*\.\w+)\s*=\s*'(" + _UUID_RE_FRAG + r")'",
                    _guard_scan_text,
                )
            ]
            # Same identity-filter shape, but IN (...) — resolve_selected_rows
            # (agent/orchestrator/row_selection.py) emits "WHERE x.id IN
            # ('u1','u2',...)" for a multi-row checkbox selection, which the
            # "=" pattern above never matched. Confirmed live 2026-08-06: the
            # coder model silently dropped the entire IN clause and fell back
            # to a generic unfiltered listing (18 rows instead of the 2 the
            # user actually selected) — this guard never fired to catch/retry
            # it because it only recognized single-uuid "=" filters.
            for _col, _uid_blob in re.findall(
                r"([a-zA-Z_]\w*\.\w+)\s*IN\s*\(\s*((?:'" + _UUID_RE_FRAG
                + r"'\s*,?\s*)+)\)",
                _guard_scan_text, re.IGNORECASE,
            ):
                for _uid in re.findall(_UUID_RE_FRAG, _uid_blob):
                    _hint_uuid_filters.append((_col, _uid, "in"))

            _missing_uuid = [
                (col, uid, src) for col, uid, src in _hint_uuid_filters
                if uid.lower() not in raw_sql.lower()
            ]
            if _missing_uuid and _last_attempt:
                if _debug_logger:
                    _debug_logger.debug(
                        "  [uuid-guard] skipped — last permitted attempt, "
                        "a retry here can't change the outcome"
                    )
            elif _missing_uuid:
                # Multiple FK columns can resolve to the SAME uuid (e.g. "assigned
                # to X" meaning X could be the inspector OR the inspectee) — those
                # must be OR'd, not AND'd, or the filter demands a row where X is
                # simultaneously both roles and the query returns 0 rows. Different
                # uuids (genuinely separate resolved entities) still combine with AND.
                #
                # IN-sourced pairs are handled separately and grouped by COLUMN,
                # not by uuid — several DIFFERENT uuids for the SAME column (a
                # multi-row selection) mean "any of these", i.e. a single
                # "col IN (u1, u2, ...)" clause, never "col = u1 AND col = u2"
                # (which can't match any row at all).
                _eq_missing = [(c, u) for c, u, s in _missing_uuid if s == "eq"]
                _in_missing = [(c, u) for c, u, s in _missing_uuid if s == "in"]

                _cols_by_uuid: dict[str, list[str]] = {}
                for col, uid in _eq_missing:
                    _cols_by_uuid.setdefault(uid, []).append(col)
                _clause_parts = [
                    f"{cols[0]} = '{uid}'" if len(cols) == 1
                    else "(" + " OR ".join(f"{c} = '{uid}'" for c in cols) + ")"
                    for uid, cols in _cols_by_uuid.items()
                ]

                _uuids_by_col: dict[str, list[str]] = {}
                for col, uid in _in_missing:
                    _uuids_by_col.setdefault(col, []).append(uid)
                _clause_parts += [
                    f"{col} = '{uids[0]}'" if len(uids) == 1
                    else f"{col} IN (" + ", ".join(f"'{u}'" for u in uids) + ")"
                    for col, uids in _uuids_by_col.items()
                ]

                _filter_clause = " AND ".join(_clause_parts)
                # Minimal reconstructed prompt (see _build_guard_retry_prompt) —
                # not the full original `prompt` (see that helper's docstring for
                # why: the full-prompt-reuse pattern used across all three guards
                # was the single largest contributor to this app's worst-case
                # 56-80K-token queries).
                _mandatory = (
                    "### MANDATORY FILTER (do not deviate)\n"
                    f"Your query MUST include this exact filter, verbatim: {_filter_clause}\n"
                    "This identity has already been resolved to a UUID — do NOT substitute "
                    "an ILIKE/name search, and do NOT change the column/table it's joined on."
                )
                forced_prompt = _build_guard_retry_prompt(question, additional_context, raw_sql, _mandatory)
                import logging as _logging
                _logging.getLogger("inspectai").warning(
                    f"[uuid-guard] retry fired — missing={_missing_uuid}"
                )
                if _debug_logger:
                    _debug_logger.debug(f"  [uuid-guard] firing retry — missing={_missing_uuid}")
                response3 = sql_llm.complete(forced_prompt)
                raw_sql = str(response3).strip()
                _usage = extract_token_counts(getattr(response3, "raw", {}) or {})
                _prompt_tok_total += _usage.prompt_tokens
                _completion_tok_total += _usage.completion_tokens
                _cache_read_tok_total += _usage.cache_read_tokens
                _cache_creation_tok_total += _usage.cache_creation_tokens
                _prefix_cache.record(getattr(response3, "raw", {}) or {})
                _timing = extract_llamacpp_timings(getattr(response3, "raw", {}) or {})
                _prefill_ms_total += _timing.prompt_ms
                _decode_ms_total += _timing.predicted_ms
                if _debug_logger:
                    _debug_logger.debug(
                        f"\n{'='*70}\n>>> SQL-GEN RAW OUTPUT ({_sql_model_name}) (uuid-guard retry)\n{'='*70}\n{raw_sql}\n"
                    )

                # Re-check: did the retry actually pin the required UUID filter(s)?
                # Without this, a still-broken retry (e.g. the model substitutes
                # an ILIKE name search again) silently proceeds to validation and
                # execution as if the guard had succeeded — potentially returning
                # a different person's data than the one already resolved.
                _still_missing_uuid = [
                    (col, uid, src) for col, uid, src in _missing_uuid
                    if uid.lower() not in raw_sql.lower()
                ]
                if _still_missing_uuid:
                    _logging.getLogger("inspectai").warning(
                        f"[uuid-guard] retry did NOT fix it — still missing={_still_missing_uuid}"
                    )
                    if _debug_logger:
                        _debug_logger.debug(
                            f"  [uuid-guard] retry unsuccessful — still missing={_still_missing_uuid}"
                        )
        except Exception as _uuid_guard_exc:
            if _debug_logger:
                _debug_logger.debug(f"  [uuid-guard] skipped: {_uuid_guard_exc}")
        # ── end UUID-filter guard ────────────────────────────────────────────

        sql = validator.clean_sql(raw_sql)

        # AST-level deterministic repairs (sqlglot): drops hallucinated
        # {alias}.active on no-active tables, injects missing soft-delete
        # filters, fixes date_trunc(literal) ambiguity, and renames
        # sched.scheduled_date -> sched.schedule_date. Falls back to the
        # unmodified SQL on parse failure or if sqlglot isn't installed.
        sql, _ast_mutations = validator.ast_repair(sql)
        if _ast_mutations and _debug_logger:
            _debug_logger.debug(f"  [ast-repair] {'; '.join(_ast_mutations)}")

        # Deterministic fix for a bare <alias>.id selected for display on a
        # table with no (or a known) display substitute — the single most
        # common validation retry in the audit domain (confirmed live via
        # eval_audit.py, reproducible at temperature=0.0). Same "fix it
        # ourselves instead of blocking and hoping the LLM remembers"
        # principle as ast_repair above; unconditional (not question-gated)
        # since dropping/replacing a bare id never changes which rows come
        # back, only which columns are displayed.
        sql, _bare_id_mutations = validator.strip_bare_uuid_pk_select(sql)
        if _bare_id_mutations and _debug_logger:
            _debug_logger.debug(f"  [bare-id-autocorrect] {'; '.join(_bare_id_mutations)}")

        # Same principle, for a hallucinated display-name column (e.g.
        # acyc.name, po.portfolio_name) on a table confirmed to have NO
        # substitute at all — see identity.py's strip_hallucinated_no_
        # substitute_columns docstring for the PLAN-08 (7 calls) / PORT-03
        # (hallucinated twice, even after a get_schema check) history this
        # closes.
        sql, _halluc_mutations = validator.strip_hallucinated_no_substitute_columns(sql)
        if _halluc_mutations and _debug_logger:
            _debug_logger.debug(f"  [bare-id-autocorrect] {'; '.join(_halluc_mutations)}")

        # Question-aware intent autocorrects — same "fix it ourselves instead
        # of blocking and hoping the LLM remembers" principle as ast_repair
        # above, but gated on the ORIGINAL QUESTION TEXT too (not just SQL
        # shape), reusing the exact compiled regexes their corresponding
        # BLOCK rule already defines. Runs BEFORE validate() specifically so
        # it prevents that BLOCK from ever firing, unlike identity injection
        # below (which deliberately runs AFTER).
        sql, _intent_mutations = validator.inject_ca_adequacy_filter(sql, question)
        if _intent_mutations and _debug_logger:
            _debug_logger.debug(f"  [intent-autocorrect] {'; '.join(_intent_mutations)}")

        sql, _axis_mutations = validator.downgrade_ca_multi_axis_joins(sql, question)
        if _axis_mutations and _debug_logger:
            _debug_logger.debug(f"  [intent-autocorrect] {'; '.join(_axis_mutations)}")

        # Question-independent (SQL-shape-only, like ast_repair) — see its
        # own docstring for why dropping a WHERE term is safe here despite
        # cleaner.py's general rule against doing so.
        sql, _name_filter_mutations = validator.strip_redundant_name_filter_on_undefined_alias(sql)
        if _name_filter_mutations and _debug_logger:
            _debug_logger.debug(f"  [intent-autocorrect] {'; '.join(_name_filter_mutations)}")

        # validate() returns ValidationResult — hard fails block, soft warns are advisory only.
        vresult = validator.validate(sql, question)

        # validate_semantic() returns SOFT_WARN issues by default.
        # These are logged for observability but NEVER injected into the retry prompt.
        # This is the core fix for validator poisoning: semantic heuristics must not
        # cause retries or appear in the errors list that the orchestrator reads.
        #
        # ONE narrow, allowlisted exception, confirmed live 2026-08-07
        # (debug_llm_20260807_153128): glossary entries in semantic.py's
        # _ENFORCED_GLOSSARY_IDS (conducted_audits/completed_audits) return
        # HARD_FAIL — exact-literal checks against a 4-value enum sourced
        # from semantics_registry.py's single source of truth, not the
        # regex-heuristic-matching validator poisoning was originally about
        # (BLOCK rules in sql/rules/block_rules.py are regex-based too —
        # "regex" was never the real discriminator). See the merge below.
        sem_warns = validator.validate_semantic(sql, question)

        # validate_shape() — AST shape enforcement (HARD_FAIL if violated).
        # Only fires when question intent implies a specific result cardinality.
        # e.g. "this month vs last month" must return ONE row, not grouped rows.
        if vresult.passed:
            shape_result = validator.validate_shape(sql, question)
            if not shape_result.passed:
                # Merge shape failure into vresult
                vresult = shape_result

        # Merge any allowlisted semantic HARD_FAILs into vresult — same
        # pattern as the validate_shape() merge just above, reusing the
        # SAME orchestrator retry path (loop.py reads _vresult/
        # retry_message()); generate_sql() has no retry loop of its own.
        # Must run AFTER the validate_shape() merge (so a shape failure
        # isn't silently overwritten) and BEFORE the row-identity injection
        # block below (so a synthetic __id column is never added to SQL
        # that's about to be retried/discarded — see that block's own
        # invariant, documented in its comment).
        _sem_hard = [w for w in sem_warns if w.severity == "HARD_FAIL"]
        _sem_soft = [w for w in sem_warns if w.severity != "HARD_FAIL"]
        if _sem_hard and vresult.passed:
            vresult = ValidationResult(passed=False, issues=vresult.issues + _sem_hard)

        # Row-identity injection (sqlglot AST, see validation/identity.py):
        # append a hidden <alias>.id sibling alongside any display column
        # already selected from a "nameable" table (audit_plan.title,
        # facility.name, ...), so a later conversational follow-up can
        # resolve "the third one"/"that plan" to a real row id instead of
        # re-deriving one from a possibly-duplicated display string.
        # Deliberately runs AFTER validate()/validate_shape() have already
        # passed on the MODEL'S OWN sql — never before — so the synthetic
        # column is invisible to every validation pass (in particular
        # UUID_IN_SELECT, which would otherwise flag a bare `.id` selection)
        # and never risks being injected into a draft that's about to be
        # retried/discarded. Every consumer of the resulting row data (see
        # synthesis.py / response_formatting.py) is responsible for stripping
        # these __id-suffixed columns before they reach the end user or the
        # reasoning LLM's own next-turn prompt.
        if vresult.passed:
            sql, _identity_mutations = validator.inject_row_identity(sql)
            if _identity_mutations and _debug_logger:
                _debug_logger.debug(f"  [row-identity] {'; '.join(_identity_mutations)}")

            # Same rationale, different shape: ai_audit_answers/ai_answers.
            # submission_id is already a directly-filterable value (not a
            # primary key needing a synthetic __id sibling), but confirmed
            # live 2026-07-31 the SQL-gen model doesn't reliably include it
            # alongside answer_text on its own — this makes it deterministic.
            sql, _obs_mutations = validator.inject_observation_submission_id(sql)
            if _obs_mutations and _debug_logger:
                _debug_logger.debug(f"  [row-identity] {'; '.join(_obs_mutations)}")

        # Log cleaned SQL and validation result
        if _debug_logger:
            if vresult.passed:
                status = "✓ PASSED"
            else:
                status = f"✗ FAILED: {vresult.retry_message()}"
            _debug_logger.debug(
                f"\n{'='*70}\n>>> SQL-GEN CLEANED SQL ({_sql_model_name}) — {status}\n"
                f"{'='*70}\n{sql}\n"
            )
            if _sem_soft:
                _debug_logger.debug(
                    f"  Semantic warns (not forwarded to LLM): "
                    f"{'; '.join(str(w) for w in _sem_soft)}"
                )
            if _sem_hard:
                _debug_logger.debug(
                    f"  Semantic HARD_FAILs (forwarded to LLM via retry, allowlisted): "
                    f"{'; '.join(str(w) for w in _sem_hard)}"
                )

        # Planner/generator attribution — Item 3 of the generation-quality
        # plan (qid=e63f03af). planner_tables comes from whatever the
        # reasoning model actually declared (relevant_tables/tables, or a
        # fallback extraction from its own draft_sql if it didn't); this is
        # NOT a validation gate, purely a diagnostic — see the docstring on
        # "_diagnostics" below for why it's never forwarded to the LLM.
        _planner_tables = relevant_tables or _extract_table_names_from_sql(_draft_sql_only)
        _generated_tables = _extract_table_names_from_sql(sql)
        _table_drift = [t for t in _generated_tables if t not in _planner_tables]
        _table_overlap = len(set(_planner_tables) & set(_generated_tables))

        if _debug_logger:
            _debug_logger.debug(
                f"\n{'='*70}\n>>> SQL-GEN PLANNER CONTRACT ({_sql_model_name})\n"
                f"{'='*70}\n"
                f"draft_sql_supplied={bool(draft_sql)} "
                f"resolved_from='{'draft_sql' if draft_sql else ('schema_hint' if _draft_sql_only else 'none')}'\n"
                f"join_path={join_path!r}\n"
                f"business_rules={business_rules!r}\n"
                f"primary_example={_primary_example_key or '(none matched)'}\n"
                f"planner_tables={_planner_tables} generated_tables={_generated_tables} "
                f"overlap={_table_overlap} drift={_table_drift}\n"
            )

        # errors: concise hard-fail codes only — no soft warn prose, no semantic hints.
        # warnings: soft warns + soft semantic warns — logged by caller, never in
        # retry prompt. Uses _sem_soft, not sem_warns — an escalated HARD_FAIL
        # already landed in vresult.hard_fails via the merge above; including it
        # here too would duplicate it into both errors AND warnings.
        return {
            "success": True,
            "result": {
                "sql": sql,
                "validation": {
                    "passed": vresult.passed,
                    "errors": [str(i) for i in vresult.hard_fails],
                    "warnings": [str(i) for i in vresult.soft_warns + _sem_soft],
                    # _vresult: structured object for orchestrator retry_message()
                    "_vresult": vresult,
                },
                # Internal-only, mirrors _vresult above — summed across every
                # sql_llm.complete() call this invocation made (initial + any
                # offset/table/uuid-guard retries), so the caller (loop.py)
                # can fold real SQL-gen spend into QueryStats alongside the
                # reasoning role's tokens.
                "_token_usage": {
                    "prompt_tokens": _prompt_tok_total,
                    "completion_tokens": _completion_tok_total,
                    "cache_read_tokens": _cache_read_tok_total,
                    "cache_creation_tokens": _cache_creation_tok_total,
                },
                # llama.cpp-only prefill/decode split — see llm/base.py's
                # LlamaCppTimings and Phase 3.2 profiling. 0/0 on any other
                # provider (extract_llamacpp_timings' zero-valued default).
                "_llm_timings": {
                    "prefill_ms": _prefill_ms_total,
                    "decode_ms": _decode_ms_total,
                },
                # Internal-only, human-debugging attribution data (Item 3 of
                # the generation-quality plan) — answers "did the planner or
                # the coder model make the mistake" from a single log line
                # instead of manually diffing two prompt blobs. Deliberately
                # NEVER forwarded into validation["errors"/"warnings"] or any
                # LLM-facing retry message — same hard-won rule as semantic
                # warnings below (see sem_warns: "logged for observability
                # but NEVER injected into the retry prompt... the core fix
                # for validator poisoning"). Diagnostics are for humans only.
                "_diagnostics": {
                    "primary_example_key": _primary_example_key,
                    "planner_tables": _planner_tables,
                    "generated_tables": _generated_tables,
                    "table_drift": _table_drift,
                    "draft_sql_supplied": bool(draft_sql),
                    "join_path": join_path,
                },
            },
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "result": {
                "sql": "", "validation": {"passed": False, "errors": [str(exc)]},
                "_token_usage": {
                    "prompt_tokens": _prompt_tok_total if "_prompt_tok_total" in locals() else 0,
                    "completion_tokens": _completion_tok_total if "_completion_tok_total" in locals() else 0,
                    "cache_read_tokens": _cache_read_tok_total if "_cache_read_tok_total" in locals() else 0,
                    "cache_creation_tokens": _cache_creation_tok_total if "_cache_creation_tok_total" in locals() else 0,
                },
            },
            "error": str(exc),
        }
