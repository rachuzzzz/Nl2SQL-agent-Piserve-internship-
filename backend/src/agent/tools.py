"""
Tool implementations for the NL2SQL agent.

The form builder complexity (fb_* JSONB tables, dynamic fb_{uuid} answer tables)
is fully hidden behind the ai_question and ai_answer flat tables.
This file only exposes tools that work with those flat tables + inspection_* tables.

Tool return contract:
  Every tool returns {"success": bool, "result": <any>, "error": str | None}
"""

import logging
from typing import Any, Optional

from sqlalchemy import inspect as sa_inspect, text as sql_text
from sqlalchemy.engine import Engine

from src.retrieval import SemanticQuestionIndex
from src.retrieval.questions import module_names_for_domain
from src.sql.validation import SQLValidator
from src.sql.generation import generate_sql
from src.sql.execution import execute_sql

_log_security = logging.getLogger("inspectai.security")
_log_sql = logging.getLogger("inspectai.sql")

_SEMANTIC_MIN_SCORE = 0.55


# ---------------------------------------------------------------------------
# Tool: list_forms
# Lists distinct forms known to the system via ai_question (the flat table).
# Falls back to fb_forms if ai_question has no form_name column.
# ---------------------------------------------------------------------------

def list_forms(db_engine: Engine, form_name: Optional[str] = None,
                domain: str = "inspection") -> dict[str, Any]:
    """
    List all forms. Sourced from ai_question.form_name (distinct values)
    so it only shows forms that actually have questions indexed.
    Optionally filter by partial name. Domain-scoped (added 2026-07-29) so an
    audit-domain call never lists inspection's own modules and vice versa —
    see retrieval/questions.py's AUDIT_MODULE_NAMES/INSPECTION_MODULE_NAMES.
    """
    try:
        allowed = module_names_for_domain(domain)
        params: dict[str, Any] = {"allowed": list(allowed)}
        filters = ["module_name IS NOT NULL", "module_name != ''",
                   "module_name = ANY(:allowed)"]
        if form_name:
            filters.append("module_name ILIKE :pattern")
            params["pattern"] = f"%{form_name}%"

        sql = sql_text(f"""
            SELECT DISTINCT module_name,
                   COUNT(*) AS question_count
            FROM ai_questions
            WHERE {' AND '.join(filters)}
            GROUP BY module_name
            ORDER BY module_name
            LIMIT 100
        """)

        with db_engine.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        data = [{"form_name": r[0], "question_count": r[1]} for r in rows]
        return {"success": True, "result": data, "error": None}
    except Exception as exc:
        return {"success": False, "result": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: semantic_search
# Search for question labels by meaning using the embedding index.
# ---------------------------------------------------------------------------

def semantic_search(
    semantic_index: SemanticQuestionIndex,
    query: str,
    form_name: Optional[str] = None,
    top_k: int = 10,
    domain: str = "inspection",
) -> dict[str, Any]:
    """
    Semantic search over ai_question labels.
    Returns matching questions sorted by semantic similarity.
    Use this when the user describes a topic and you need to find
    which questions/forms are related to it.
    Domain-scoped (added 2026-07-29) — see list_forms()'s docstring.
    """
    try:
        matches = semantic_index.search(
            query=query,
            entity_type="QUESTION",
            form_name=form_name,
            top_k=top_k,
            allowed_forms=module_names_for_domain(domain),
        )
        good = [m for m in matches if m.score >= _SEMANTIC_MIN_SCORE]
        data = [
            {
                "question_label": m.text,
                "form_name": m.form_name,
                "element_id": m.element_id,
                "score": round(m.score, 3),
            }
            for m in good
        ]
        return {"success": True, "result": data, "error": None}
    except Exception as exc:
        return {"success": False, "result": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: search_questions
# Keyword search over ai_question.question_label.
# ---------------------------------------------------------------------------

def search_questions(
    db_engine: Engine,
    query: str,
    form_name: Optional[str] = None,
    module_name: Optional[str] = None,
    question_type: Optional[str] = None,
    limit: int = 30,
    domain: str = "inspection",
) -> dict[str, Any]:
    """
    ILIKE search over ai_question.question_label.
    Use when the user knows part of the question text exactly.
    Complements semantic_search for exact keyword matches.
    Domain-scoped (added 2026-07-29) — see list_forms()'s docstring.
    """
    try:
        params: dict[str, Any] = {"limit": limit,
                                   "allowed": list(module_names_for_domain(domain))}
        filters = ["aq.module_name = ANY(:allowed)"]
        # query is optional in practice — a form-only lookup ("what questions
        # are in form X") legitimately passes no keyword at all. Skipping the
        # ILIKE clause when empty avoids matching on a literal '' or (via the
        # dispatcher coercing a JSON null) the literal string 'None'.
        if query:
            filters.append("aq.label ILIKE :pattern")
            params["pattern"] = f"%{query}%"

        if form_name:
            filters.append("aq.module_name ILIKE :form_name")
            params["form_name"] = f"%{form_name}%"
        if module_name:
            filters.append("aq.module_name ILIKE :module_name")
            params["module_name"] = f"%{module_name}%"
        if question_type:
            filters.append("aq.entity_type ILIKE :qtype")
            params["qtype"] = f"%{question_type}%"

        sql = sql_text(f"""
            SELECT aq.label          AS question_label,
                   aq.module_name,
                   aq.entity_type    AS question_type,
                   aq.element_id
            FROM ai_questions aq
            WHERE {' AND '.join(filters)}
            ORDER BY aq.module_name, aq.label
            LIMIT :limit
        """)

        with db_engine.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            cols = ["question_label", "module_name", "question_type", "element_id"]

        data = [dict(zip(cols, row)) for row in rows]
        return {"success": True, "result": data, "error": None}
    except Exception as exc:
        return {"success": False, "result": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: get_answers
# Query submitted answers from ai_answer, joined to ai_question for labels.
# ---------------------------------------------------------------------------

def get_answers(
    db_engine: Engine,
    form_name: Optional[str] = None,
    module_name: Optional[str] = None,
    question_label: Optional[str] = None,
    inspection_id: Optional[str] = None,
    answer_value: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    scope_filter: str = '',
    domain: str = "inspection",
    _qid: str = "",
) -> dict[str, Any]:
    """
    Query submitted answers from ai_answers (inspection) or ai_audit_answers
    (audit, added 2026-07-29 once confirmed live/populated — see
    schema_introspector.py's AUDIT_BUSINESS_RULES "AUDIT FORM-ANSWER LAYER").
    Automatically joins ai_questions for the question label.
    All filters are optional — combine as needed.
    scope_filter: SQL predicate from _build_scope_filter(); '' = unrestricted, 'BLOCK' = deny.
    inspection_id: for audit domain this matches audit_report.audit_id (the
      display code) instead — same parameter name kept for one call-site
      contract shared by both domains' ToolRegistry dispatch.
    offset: for the "show more" pagination fast-path (loop.py's Path B) —
      previously absent entirely (no param, no OFFSET clause), so every
      repeated "show more" on a get_answers-sourced turn silently re-fetched
      the identical first `limit` rows forever instead of advancing.
    """
    try:
        if scope_filter == 'BLOCK':
            _log_security.warning("[SCOPE] qid=%s outcome=BLOCKED tool=get_answers domain=%s",
                                  _qid, domain)
            return {"success": False, "result": None, "error": "SCOPE_BLOCKED"}

        is_audit = domain == "audit"
        answers_table  = "ai_audit_answers" if is_audit else "ai_answers"
        report_table   = "audit_report"     if is_audit else "inspection_report"
        report_alias   = "arep"             if is_audit else "ir"
        report_id_col  = "audit_report_id"  if is_audit else "inspection_report_id"
        display_id_col = "audit_id"         if is_audit else "inspection_id"

        filters = ["1=1"]
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if form_name:
            filters.append("aq.module_name ILIKE :form_name")
            params["form_name"] = f"%{form_name}%"
        if module_name:
            filters.append("aa.module_name ILIKE :module_name")
            params["module_name"] = f"%{module_name}%"
        if question_label:
            filters.append("aq.label ILIKE :q_label")
            params["q_label"] = f"%{question_label}%"
        if inspection_id:
            # aa.<display_id_col> may store the UUID, not the human-readable varchar.
            # Safest path: join via the report table to match its display id (varchar).
            # This handles both cases — if aa.<display_id_col> IS the varchar it works too.
            filters.append(f"""aa.{report_id_col} = (
                SELECT {report_alias}.id FROM {report_table} {report_alias}
                WHERE {report_alias}.{display_id_col} = :insp_id
                LIMIT 1
            )""")
            params["insp_id"] = inspection_id
            # When fetching by report id with no explicit module filter, default
            # to the domain's primary conduct-form module — ai_answers/
            # ai_audit_answers contains rows from multiple modules for the same report.
            if not module_name and not form_name and not is_audit:
                filters.append("aa.module_name = 'Inspection Form'")
        if answer_value:
            filters.append(
                "(aa.answer_text ILIKE :ans_val OR aa.answer_numeric::text ILIKE :ans_val)"
            )
            params["ans_val"] = f"%{answer_value}%"

        # Scope predicate — server-generated SQL fragment, always safe to inline.
        # NOTE: scope_filter is generated for the inspection domain's ir alias
        # (see sql/validation/scope.py) — audit callers should pass scope_filter=''
        # until an audit-domain scope predicate exists.
        if scope_filter:
            filters.append(scope_filter)

        where = " AND ".join(filters)

        # Use INNER JOIN when scope is active so the predicate cannot be bypassed
        # by NULL report rows (a LEFT JOIN with WHERE on the right side acts as
        # inner, but INNER JOIN makes the intent explicit and avoids NULL propagation risk).
        _report_join = (
            f"{'INNER' if scope_filter else 'LEFT'} JOIN {report_table} {report_alias} "
            f"ON aa.{report_id_col} = {report_alias}.id"
        )

        sql = sql_text(f"""
            SELECT {report_alias}.{display_id_col},
                   aa.module_name,
                   aq.label                                          AS question_label,
                   COALESCE(aa.answer_text, aa.answer_numeric::text) AS answer_value,
                   aa.score,
                   aa.submitted_on
            FROM {answers_table} aa
            LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id
            {_report_join}
            WHERE {where}
            ORDER BY aa.submitted_on DESC, aa.id
            LIMIT :limit OFFSET :offset
        """)

        with db_engine.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            cols = ["inspection_id", "module_name", "question_label",
                    "answer_value", "score", "submitted_on"]

        data = [dict(zip(cols, row)) for row in rows]
        truncated = len(data) == limit
        total_count = len(data)

        if truncated:
            try:
                count_sql = sql_text(f"""
                    SELECT COUNT(*) FROM {answers_table} aa
                    LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id
                    {_report_join}
                    WHERE {where}
                """)
                with db_engine.connect() as conn:
                    total_count = conn.execute(count_sql, params).fetchone()[0]
            except Exception:
                pass

        return {
            "success": True,
            "result": {
                "rows": data,
                "row_count": len(data),
                "total_count": total_count,
                "truncated": truncated,
                "columns": cols,
            },
            "error": None,
        }
    except Exception as exc:
        _log_sql.debug("qid=%s get_answers raised %s", _qid, type(exc).__name__, exc_info=True)
        return {"success": False, "result": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: get_answer_stats
# Aggregate stats (avg, distribution) over ai_answer.answer_value.
# ---------------------------------------------------------------------------

def get_answer_stats(
    db_engine: Engine,
    form_name: Optional[str] = None,
    module_name: Optional[str] = None,
    question_label: Optional[str] = None,
    breakdown_limit: Optional[int] = None,
    domain: str = "inspection",
) -> dict[str, Any]:
    """
    Aggregate statistics over ai_answers (inspection) or ai_audit_answers
    (audit, added 2026-07-29 — see get_answers()'s docstring for context).
    Returns numeric stats (avg/min/max) + categorical value breakdown.
    Use for "most common answer", "average score", "distribution" questions.

    breakdown_limit — if set, truncates value_breakdown to the top N entries.
                      Pass 1 when user asks for "most common", "most chosen",
                      "top" or "highest" single value.
    """
    try:
        answers_table = "ai_audit_answers" if domain == "audit" else "ai_answers"

        filters = ["1=1"]
        params: dict[str, Any] = {}

        if form_name:
            filters.append("aq.module_name ILIKE :form_name")
            params["form_name"] = f"%{form_name}%"
        if module_name:
            filters.append("aa.module_name ILIKE :module_name")
            params["module_name"] = f"%{module_name}%"
        if question_label:
            filters.append("aq.label ILIKE :q_label")
            params["q_label"] = f"%{question_label}%"

        where = " AND ".join(filters)

        # Numeric stats — aggregates over answer_numeric (for text-entered numbers)
        # AND over score (for questions with scoring config from the form builder).
        # answer_numeric: NULL for dropdowns (Risk Level, Impact etc.) — use score instead.
        # score: populated by migration backfill for scored questions only.
        num_sql = sql_text(f"""
            SELECT COUNT(*)               AS total_answers,
                   AVG(aa.answer_numeric) AS avg_answer_numeric,
                   MIN(aa.answer_numeric) AS min_answer_numeric,
                   MAX(aa.answer_numeric) AS max_answer_numeric,
                   AVG(aa.score)          AS avg_score,
                   MIN(aa.score)          AS min_score,
                   MAX(aa.score)          AS max_score,
                   COUNT(aa.score)        AS scored_count
            FROM {answers_table} aa
            LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id
            WHERE {where}
              AND (
                (aa.answer_numeric IS NOT NULL AND aa.answer_numeric BETWEEN -1e12 AND 1e12)
                OR aa.score IS NOT NULL
              )
        """)

        # Categorical breakdown — use answer_text
        cat_sql = sql_text(f"""
            SELECT aa.answer_text, COUNT(*) AS frequency
            FROM {answers_table} aa
            LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id
            WHERE {where}
              AND aa.answer_text IS NOT NULL
              AND aa.answer_text != ''
            GROUP BY aa.answer_text
            ORDER BY frequency DESC
            LIMIT 15
        """)

        with db_engine.connect() as conn:
            # Only run numeric aggregation when filtered to a specific question.
            # Even then, skip if the question label suggests text answers
            # (observations, regulations, actions, notes, descriptions).
            _TEXT_QUESTION_HINTS = (
                "observation", "regulation", "action", "recommend",
                "comment", "note", "description", "identifier", "name",
                "email", "contact", "designation", "stakeholder",
                "attach", "image", "file", "upload", "cause", "correction",
                "mitigative", "completion",
            )
            _is_text_question = question_label and any(
                hint in question_label.lower() for hint in _TEXT_QUESTION_HINTS
            )
            if question_label and not _is_text_question:
                num_row = conn.execute(num_sql, params).fetchone()
            else:
                num_row = None
            cat_rows = conn.execute(cat_sql, params).fetchall()

        if num_row:
            total = num_row[0]
            avg_an, mn_an, mx_an = num_row[1], num_row[2], num_row[3]
            avg_sc, mn_sc, mx_sc, scored_count = num_row[4], num_row[5], num_row[6], num_row[7]
        else:
            total = 0
            avg_an = mn_an = mx_an = None
            avg_sc = mn_sc = mx_sc = None
            scored_count = 0

        # Prefer score stats when available (scored questions like Risk Level, Impact)
        # Fall back to answer_numeric stats (text-input numeric questions like CAPEX Amount)
        if avg_sc is not None:
            avg, mn, mx = avg_sc, mn_sc, mx_sc
            stats_source = "score"
        else:
            avg, mn, mx = avg_an, mn_an, mx_an
            stats_source = "answer_numeric" 

        def _round(v):
            try:
                return round(float(v), 4) if v is not None else None
            except (TypeError, ValueError):
                return None

        return {
            "success": True,
            "result": {
                "filters": {
                    "form_name": form_name,
                    "module_name": module_name,
                    "question_label": question_label,
                },
                "total_answers": total,
                "average_value": _round(avg),
                "min_value": _round(mn),
                "max_value": _round(mx),
                "stats_source": stats_source,   # "score" or "answer_numeric"
                "scored_count": scored_count,   # how many answers have a score
                "value_breakdown": [
                    {"value": r[0], "count": r[1]} for r in cat_rows
                ][:breakdown_limit] if breakdown_limit else [
                    {"value": r[0], "count": r[1]} for r in cat_rows
                ],
            },
            "error": None,
        }
    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# get_schema
# ---------------------------------------------------------------------------
#
# generate_sql/execute_sql (and their schema-selection/join-hint helpers)
# now live in src.sql.generation / src.sql.execution — that pipeline is the
# system's core NL2SQL capability, not a query tool like the ones above.
# ToolRegistry below still dispatches "generate_sql"/"execute_sql" tool calls
# by importing those functions, so the agent-facing tool contract is unchanged.


def get_schema(db_engine: Engine, table_name: Optional[str] = None,
                db_schema=None) -> dict[str, Any]:
    try:
        if table_name is None:
            inspector = sa_inspect(db_engine)
            tables = inspector.get_table_names()
            return {"success": True, "result": {"tables": tables}, "error": None}

        # Prefer the curated schema (FK-resolution JOINs, enum values, useful-column
        # notes) already built for the SQL model — richer than a bare column dump,
        # and the same level of detail the reasoning LLM's system prompt used to
        # always carry before it was slimmed (see DatabaseSchema.for_reasoning_prompt).
        # Only used when the table is one db_schema actually introspected — for_schema_hint()
        # unconditionally appends the lookup/relationships boilerplate even for an
        # unrecognized name, so membership is checked here first rather than relying
        # on its return value being empty.
        if db_schema is not None:
            known = (db_schema.tables.get(table_name)
                     or db_schema.ai_tables.get(table_name)
                     or db_schema.lookup_tables.get(table_name))
            if known:
                return {"success": True,
                        "result": {"detail": db_schema.for_schema_hint([table_name])},
                        "error": None}

        # Fallback: raw introspection — covers tables outside the curated sets
        # (e.g. a typo, or a table db_schema doesn't track) so a valid table name
        # never comes back empty just because it wasn't pre-curated.
        inspector = sa_inspect(db_engine)
        columns = inspector.get_columns(table_name)
        col_info = [{"name": c["name"], "type": str(c["type"])} for c in columns]
        return {"success": True, "result": {"columns": col_info}, "error": None}
    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}



# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    KNOWN_TOOLS = {
        # Form discovery
        "list_forms",
        "semantic_search",
        "search_questions",
        # Answer retrieval
        "get_answers",
        "get_answer_stats",
        # SQL path (inspection_* and complex ai_* joins)
        "generate_sql",
        "execute_sql",
        "get_schema",
    }

    def __init__(self, db_engine: Engine, semantic_index: SemanticQuestionIndex,
                 validator: SQLValidator, sql_llm, sql_prompt: str = "",
                 debug_logger=None, seed_index=None, db_schema=None,
                 core_context: str | None = None, build_core_context_fn=None,
                 join_graph_module=None, rules_engine=None,
                 domain: str = "inspection", profile: bool = False) -> None:
        self._db_engine = db_engine
        self._semantic_index = semantic_index
        self._validator = validator
        self._sql_llm = sql_llm
        self._sql_prompt = sql_prompt
        self._debug_logger = debug_logger
        self._seed_index = seed_index  # SeedExampleIndex for retrieval-based injection
        self._db_schema = db_schema    # DatabaseSchema with live enum_values
        # Domain-specific pieces threaded down to generate_sql() — None means
        # "use inspection defaults" (see sql/generation.py's generate_sql()),
        # so an orchestrator that doesn't pass these gets unchanged behavior.
        self._core_context = core_context
        self._build_core_context_fn = build_core_context_fn
        self._join_graph_module = join_graph_module
        # The SAME per-domain RulesEngine instance loop.py already computes
        # (self._rules_engine, via DomainConfig.rules_engine_getter) — passed
        # through to generate_sql()'s post-generation table/pattern guard so
        # it checks fired INJECT rule ids against the domain they actually
        # came from, not always the inspection registry (see generation.py's
        # _rules_engine param docstring for the bug this fixes).
        self._rules_engine = rules_engine
        # Threaded into get_answers/get_answer_stats so they route to
        # ai_answers/inspection_report or ai_audit_answers/audit_report —
        # added 2026-07-29 when ai_audit_answers was confirmed live for audit.
        self._domain = domain
        # Phase 3.8 profiling flag — see sql/generation.py's generate_sql()
        # _profile param docstring. False (default) means every downstream
        # generate_sql call skips the extra /tokenize round-trips entirely,
        # so production/non-profiling callers pay nothing for this.
        self._profile = profile

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        # Tools declared via agent/tool_defs (see that package's docstring)
        # take a separate, additive dispatch path — checked first so a
        # declared tool's name never needs to also join KNOWN_TOOLS below.
        # The 8 tools in the if/elif chain further down stay exactly as-is:
        # several (get_answers, generate_sql, search_questions) carry
        # tool-specific arg-cleaning (alias fallbacks, "none"-string
        # normalization, limit clamping) that a generic fn(**args) call
        # would silently break, so they are deliberately NOT migrated.
        from src.agent.tool_defs import get_declared
        declared = get_declared(name)
        if declared is not None:
            return self._call_declared(declared, args)

        if name not in self.KNOWN_TOOLS:
            return {
                "success": False, "result": None,
                "error": f"Unknown tool '{name}'. Available: {sorted(self.KNOWN_TOOLS)}",
            }
        try:
            if name == "list_forms":
                return list_forms(self._db_engine,
                                  form_name=args.get("form_name"),
                                  domain=self._domain)

            if name == "semantic_search":
                _q = args.get("query")
                if not isinstance(_q, str) or not _q.strip() or _q.lower() in ('none', 'null'):
                    return {"success": False, "result": [],
                            "error": "semantic_search requires a non-empty 'query'"}
                return semantic_search(
                    self._semantic_index,
                    query=_q,
                    form_name=args.get("form_name"),
                    top_k=int(args.get("top_k", 10)),
                    domain=self._domain,
                )

            if name == "search_questions":
                def _clean(v):
                    if isinstance(v, str) and v.lower() in ('none', 'null', ''):
                        return None
                    return v
                def _limit(v, default=30):
                    try:
                        return max(1, min(int(v), 200))
                    except (TypeError, ValueError):
                        return default
                # args.get (not args[]) + _clean: a JSON null query (a valid,
                # common form-only lookup with no keyword) must become None/'',
                # not the literal string 'None' — str(args['query']) previously
                # corrupted the ILIKE filter into '%None%', silently returning
                # zero rows. Caught live 2026-07-29 testing "what questions are
                # in the Quality Audit form" — the LLM correctly passed
                # query=null, form_name='Quality Audit', but got 0 results back.
                q = _clean(args.get("query")) or ""
                return search_questions(
                    self._db_engine,
                    query=q,
                    form_name=_clean(args.get("form_name")),
                    module_name=_clean(args.get("module_name")),
                    question_type=_clean(args.get("question_type")),
                    limit=_limit(args.get("limit", 30)),
                    domain=self._domain,
                )

            if name == "get_answers":
                def _clean(v):
                    if isinstance(v, str) and v.lower() in ('none', 'null', ''):
                        return None
                    return v
                # Accept both 'label' and 'question_label' — LLM uses both names
                q_label = args.get("question_label") or args.get("label")
                # Accept both 'answer_text' (the key the tool schema in
                # agent/prompts.py actually documents) and 'answer_value' (the
                # function parameter's own name) — without this the documented
                # key silently drops the filter, since get_answers() never sees it.
                ans_filter = args.get("answer_text") or args.get("answer_value")
                def _limit(v, default=50):
                    try:
                        n = int(v)
                        return max(1, min(n, 500))
                    except (TypeError, ValueError):
                        return default
                def _offset(v, default=0):
                    try:
                        return max(0, int(v))
                    except (TypeError, ValueError):
                        return default
                return get_answers(
                    self._db_engine,
                    form_name=_clean(args.get("form_name")),
                    module_name=_clean(args.get("module_name")),
                    question_label=_clean(q_label),
                    inspection_id=_clean(args.get("inspection_id")),
                    answer_value=_clean(ans_filter),
                    limit=_limit(args.get("limit", 50)),
                    offset=_offset(args.get("offset", 0)),
                    scope_filter=str(args.get("scope_filter", "")),
                    domain=self._domain,
                )

            if name == "get_answer_stats":
                def _clean(v):
                    """Convert string 'None'/'null'/'none' to actual None."""
                    if isinstance(v, str) and v.lower() in ('none', 'null', ''):
                        return None
                    return v
                # Accept both 'label' and 'question_label' — LLM uses both names
                q_label = args.get("question_label") or args.get("label")
                # Parse limit — 1 = top only, None = full distribution
                _blimit = args.get("limit")
                if _blimit is not None:
                    try:
                        _blimit = int(_blimit)
                    except (TypeError, ValueError):
                        _blimit = None
                return get_answer_stats(
                    self._db_engine,
                    form_name=_clean(args.get("form_name")),
                    module_name=_clean(args.get("module_name")),
                    question_label=_clean(q_label),
                    breakdown_limit=_blimit,
                    domain=self._domain,
                )

            if name == "generate_sql":
                try:
                    _retry_attempt = int(args.get("_retry_attempt", 0))
                except (TypeError, ValueError):
                    _retry_attempt = 0
                # tables supersedes relevant_tables in name only (Item 3 of
                # the generation-quality plan) — accept either, tables wins
                # if both are somehow present. Same defensive isinstance
                # guard as relevant_tables always had: this is the only
                # list-typed tool argument in this codebase and arrives
                # malformed often enough that _select_schema_block needs
                # the None-if-not-a-list fallback rather than trusting it.
                _tables = args.get("tables")
                if not isinstance(_tables, list):
                    _tables = args.get("relevant_tables")
                if not isinstance(_tables, list):
                    _tables = None
                _business_rules = args.get("business_rules")
                if not isinstance(_business_rules, (list, str)):
                    _business_rules = None
                return generate_sql(
                    self._sql_llm, self._validator,
                    question=str(args["question"]),
                    schema_hint=str(args.get("schema_hint", "")),
                    draft_sql=str(args.get("draft_sql", "")),
                    join_path=str(args.get("join_path", "")),
                    business_rules=_business_rules,
                    _mandatory_hint=str(args.get("_mandatory_hint", "")),
                    db_engine=self._db_engine,
                    sql_prompt_template=self._sql_prompt,
                    _debug_logger=self._debug_logger,
                    _inject_rule_ids=args.get("_inject_rule_ids", []),
                    _seed_index=self._seed_index,
                    _db_schema=self._db_schema,
                    _active_cycle_uuid=str(args.get("_active_cycle_uuid", "")),
                    _retry_attempt=_retry_attempt,
                    _last_attempt=bool(args.get("_last_attempt", False)),
                    relevant_tables=_tables,
                    _core_context=self._core_context,
                    _build_core_context_fn=self._build_core_context_fn,
                    _join_graph_module=self._join_graph_module,
                    _rules_engine=self._rules_engine,
                    _domain=self._domain,
                    _profile=self._profile,
                    _qid=str(args.get("_qid", "")),
                )

            if name == "execute_sql":
                try:
                    _offset = int(args.get("offset", 0))
                except (TypeError, ValueError):
                    _offset = 0
                return execute_sql(self._db_engine, self._validator,
                                   sql=str(args["sql"]),
                                   question=str(args.get("question", "")),
                                   offset=_offset,
                                   force=bool(args.get("force", False)),
                                   qid=str(args.get("_qid", "")))

            if name == "get_schema":
                return get_schema(self._db_engine,
                                  table_name=args.get("table_name"),
                                  db_schema=self._db_schema)

        except KeyError as exc:
            return {"success": False, "result": None,
                    "error": f"Missing required argument for '{name}': {exc}"}
        except Exception as exc:
            return {"success": False, "result": None,
                    "error": f"Tool '{name}' raised an exception: {exc}"}
        return {"success": False, "result": None, "error": "Unhandled routing case"}

    def _call_declared(self, tool, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch for tools registered via agent/tool_defs. `args` here is
        already the dispatch_args dict built by loop.py — it may carry extra
        keys (e.g. `_session`, `_domain`) merged in via the tool's
        context_provider on top of the LLM-supplied args; tool.fn's own
        signature is the source of truth for which keys it actually reads."""
        try:
            return tool.fn(**args)
        except TypeError as exc:
            return {"success": False, "result": None,
                    "error": f"Bad arguments for '{tool.name}': {exc}"}
        except Exception as exc:
            return {"success": False, "result": None,
                    "error": f"Tool '{tool.name}' raised an exception: {exc}"}