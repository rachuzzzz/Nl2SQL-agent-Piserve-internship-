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

from src.sql.validation import SQLValidator


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


def _select_schema_block(db_schema, relevant_tables: list | None, db_engine=None) -> str:
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
    """
    if db_schema is not None:
        if isinstance(relevant_tables, list) and relevant_tables:
            clean = [t for t in relevant_tables if isinstance(t, str) and t.strip()]
            if clean:
                block = db_schema.for_schema_hint(clean)
                if len(clean) >= 2:
                    block += _join_path_hints(clean)
                return block
        return db_schema.for_sql_prompt_slim()
    if db_engine:
        return _fetch_key_schemas(db_engine)
    return ""


def _join_graph_source(table_name: str) -> str:
    """
    JOIN_GRAPH (sql/join_graph.py) has no "users" key — only role-specific
    "users_inspector"/"users_inspectee" pseudo-nodes. Default to the inspector
    role when a caller says "users" with no role context; this is a heuristic,
    not a real disambiguation — informational hint text only, never used to
    filter/scope data.
    """
    return "users_inspector" if table_name == "users" else table_name


def _join_path_hints(relevant_tables: list) -> str:
    """Chain path_as_hint() across consecutive relevant_tables pairs."""
    from src.sql.join_graph import path_as_hint
    hints = []
    for a, b in zip(relevant_tables, relevant_tables[1:]):
        hint = path_as_hint(_join_graph_source(a), _join_graph_source(b))
        if hint:
            hints.append(hint)
    return "".join(hints)


def generate_sql(
    sql_llm, validator: SQLValidator, question: str,
    schema_hint: str = "", db_engine=None,
    sql_prompt_template: str = "",
    _debug_logger=None,
    _inject_rule_ids: list | None = None,
    _seed_index=None,
    _db_schema=None,
    _active_cycle_uuid: str = "",
    _retry_attempt: int = 0,
    relevant_tables: list | None = None,
) -> dict[str, Any]:
    try:
        # Schema block — the slim version keeps context under ~600 tokens.
        # A full introspected column dump for every table (~2000+ tokens) inflates
        # deepseek's context 3x and degrades 7B model generation quality.
        # When the reasoning model supplies relevant_tables, use a targeted
        # full-detail block for just those tables instead (see _select_schema_block).
        auto_schema = _select_schema_block(_db_schema, relevant_tables, db_engine)

        if auto_schema and not schema_hint.startswith("Use"):
            schema_hint = f"{auto_schema}\n{schema_hint}" if schema_hint else auto_schema

        if not sql_prompt_template:
            from src.sql.prompts import SQL_GENERATION_PROMPT
            sql_prompt_template = SQL_GENERATION_PROMPT

        # Retrieval-based example injection via SeedExampleIndex (cosine similarity)
        # Falls back to empty string if index not available or seed_examples not deployed.
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
                # On a retry, a weak SQL-gen model tends to reproduce its single
                # closest seed example almost verbatim — including whatever
                # fragment triggered the validator failure — ignoring the
                # planner's corrected schema_hint entirely. Dropping that one
                # example forces the model to actually use schema_hint instead
                # of re-copying the same broken pattern (see validator retry
                # loop in orchestrator.py which increments gen_sql_failures).
                retrieved = _seed_index.format_for_prompt(
                    question, top_k=4, skip_top_match=bool(_retry_attempt)
                )
            except Exception:
                retrieved = ""
        else:
            retrieved = ""
        # Substitute <CYCLE_UUID> placeholder with actual active cycle UUID if available
        if _active_cycle_uuid and retrieved:
            retrieved = retrieved.replace("<CYCLE_UUID>", _active_cycle_uuid)
        dynamic_examples = _core_ctx + ("\n" + retrieved if retrieved else "")

        prompt = sql_prompt_template.replace(
            "{question}", question
        ).replace(
            "{schema_hint}", schema_hint or "No additional context provided."
        ).replace(
            "{dynamic_examples}", dynamic_examples
        )

        # Log the full prompt sent to deepseek if debug enabled
        if _debug_logger:
            _debug_logger.debug(
                f"\n{'='*70}\n>>> DEEPSEEK PROMPT (question={question!r})\n"
                f"{'='*70}\n{prompt[-3000:]}\n"  # last 3000 chars — schema_hint is at end
            )

        response = sql_llm.complete(prompt)
        raw_sql = str(response).strip()

        # Log raw deepseek output
        if _debug_logger:
            _debug_logger.debug(
                f"\n{'='*70}\n>>> DEEPSEEK RAW OUTPUT\n{'='*70}\n{raw_sql}\n"
            )

        # ── Table guard: if inject rules required specific tables and they're
        # missing from the generated SQL, force one retry with the requirement
        # prepended directly to the question (not just in schema_hint).
        # This turns INJECT rules from suggestions into contracts. ───────────
        if _inject_rule_ids:
            try:
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

                if missing_req or violated_forbidden:
                    guard_msg_parts = []
                    if missing_req:
                        guard_msg_parts.append(f"MUST USE TABLES: {', '.join(missing_req)}")
                    if violated_forbidden:
                        guard_msg_parts.append(f"DO NOT USE: {', '.join(violated_forbidden)}")
                    guard_prefix = " | ".join(guard_msg_parts)
                    forced_question = f"[{guard_prefix}] {question}"
                    # Replace both the question AND collapse the schema_hint to
                    # just the guard message — avoids token overflow on retry
                    # when the original schema_hint is long (e.g. FM CTE hints).
                    forced_prompt = prompt.replace(
                        f"[QUESTION]{question}[/QUESTION]",
                        f"[QUESTION]{forced_question}[/QUESTION]",
                    )
                    # Also slim the schema_hint section to the guard message only
                    # so deepseek focuses on the constraint, not the original hint
                    forced_prompt = re.sub(
                        r'(### Schema Hint \(follow this guidance\)\n).*?(\n### Examples)',
                        r'\1-- ' + guard_prefix + r'\2',
                        forced_prompt,
                        flags=re.DOTALL,
                    )
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
            _hint_uuid_filters = re.findall(
                r"([a-zA-Z_]\w*\.\w+)\s*=\s*'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})'",
                schema_hint,
            )
            _missing_uuid = [
                (col, uid) for col, uid in _hint_uuid_filters
                if uid.lower() not in raw_sql.lower()
            ]
            if _missing_uuid:
                # Multiple FK columns can resolve to the SAME uuid (e.g. "assigned
                # to X" meaning X could be the inspector OR the inspectee) — those
                # must be OR'd, not AND'd, or the filter demands a row where X is
                # simultaneously both roles and the query returns 0 rows. Different
                # uuids (genuinely separate resolved entities) still combine with AND.
                _cols_by_uuid: dict[str, list[str]] = {}
                for col, uid in _missing_uuid:
                    _cols_by_uuid.setdefault(uid, []).append(col)
                _clause_parts = [
                    f"{cols[0]} = '{uid}'" if len(cols) == 1
                    else "(" + " OR ".join(f"{c} = '{uid}'" for c in cols) + ")"
                    for uid, cols in _cols_by_uuid.items()
                ]
                _filter_clause = " AND ".join(_clause_parts)
                # Strip the base template's trailing "[SQL]" completion marker before
                # appending our own instructions — otherwise the prompt ends up with
                # two "[SQL]" markers (one mid-prompt, one at the true end), which can
                # confuse a small completion-style model about where to actually start
                # writing SQL.
                _base_prompt = re.sub(r"\[SQL\]\s*$", "", prompt).rstrip()
                forced_prompt = (
                    _base_prompt
                    + f"\n\n### MANDATORY FILTER (do not deviate)\n"
                    f"Your query MUST include this exact filter, verbatim: {_filter_clause}\n"
                    f"This identity has already been resolved to a UUID — do NOT substitute "
                    f"an ILIKE/name search, and do NOT change the column/table it's joined on.\n"
                    f"[SQL]\n"
                )
                import logging as _logging
                _logging.getLogger("inspectai").warning(
                    f"[uuid-guard] retry fired — missing={_missing_uuid}"
                )
                if _debug_logger:
                    _debug_logger.debug(f"  [uuid-guard] firing retry — missing={_missing_uuid}")
                response3 = sql_llm.complete(forced_prompt)
                raw_sql = str(response3).strip()
                if _debug_logger:
                    _debug_logger.debug(
                        f"\n{'='*70}\n>>> DEEPSEEK RAW OUTPUT (uuid-guard retry)\n{'='*70}\n{raw_sql}\n"
                    )

                # Re-check: did the retry actually pin the required UUID filter(s)?
                # Without this, a still-broken retry (e.g. the model substitutes
                # an ILIKE name search again) silently proceeds to validation and
                # execution as if the guard had succeeded — potentially returning
                # a different person's data than the one already resolved.
                _still_missing_uuid = [
                    (col, uid) for col, uid in _missing_uuid
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

        # validate() returns ValidationResult — hard fails block, soft warns are advisory only.
        vresult = validator.validate(sql, question)

        # validate_semantic() returns SOFT_WARN issues only.
        # These are logged for observability but NEVER injected into the retry prompt.
        # This is the core fix for validator poisoning: semantic heuristics must not
        # cause retries or appear in the errors list that the orchestrator reads.
        sem_warns = validator.validate_semantic(sql, question)

        # validate_shape() — AST shape enforcement (HARD_FAIL if violated).
        # Only fires when question intent implies a specific result cardinality.
        # e.g. "this month vs last month" must return ONE row, not grouped rows.
        if vresult.passed:
            shape_result = validator.validate_shape(sql, question)
            if not shape_result.passed:
                # Merge shape failure into vresult
                vresult = shape_result

        # Log cleaned SQL and validation result
        if _debug_logger:
            if vresult.passed:
                status = "✓ PASSED"
            else:
                status = f"✗ FAILED: {vresult.retry_message()}"
            _debug_logger.debug(
                f"\n{'='*70}\n>>> DEEPSEEK CLEANED SQL — {status}\n"
                f"{'='*70}\n{sql}\n"
            )
            if sem_warns:
                _debug_logger.debug(
                    f"  Semantic warns (not forwarded to LLM): "
                    f"{'; '.join(str(w) for w in sem_warns)}"
                )

        # errors: concise hard-fail codes only — no soft warn prose, no semantic hints.
        # warnings: soft warns + semantic warns — logged by caller, never in retry prompt.
        return {
            "success": True,
            "result": {
                "sql": sql,
                "validation": {
                    "passed": vresult.passed,
                    "errors": [str(i) for i in vresult.hard_fails],
                    "warnings": [str(i) for i in vresult.soft_warns + sem_warns],
                    # _vresult: structured object for orchestrator retry_message()
                    "_vresult": vresult,
                },
            },
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "result": {"sql": "", "validation": {"passed": False, "errors": [str(exc)]}},
            "error": str(exc),
        }
