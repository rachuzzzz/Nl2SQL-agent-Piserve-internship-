"""
Tool implementations for the NL2SQL agent.

The form builder complexity (fb_* JSONB tables, dynamic fb_{uuid} answer tables)
is fully hidden behind the ai_question and ai_answer flat tables.
This file only exposes tools that work with those flat tables + inspection_* tables.

Tool return contract:
  Every tool returns {"success": bool, "result": <any>, "error": str | None}
"""

import re
from typing import Any, Optional

from sqlalchemy import inspect as sa_inspect, text as sql_text
from sqlalchemy.engine import Engine

from core.semantic import SemanticQuestionIndex
from core.validator import SQLValidator


_SEMANTIC_MIN_SCORE = 0.55
_SEMANTIC_FALLBACK_MIN_SCORE = 0.50


# ---------------------------------------------------------------------------
# Tool: list_forms
# Lists distinct forms known to the system via ai_question (the flat table).
# Falls back to fb_forms if ai_question has no form_name column.
# ---------------------------------------------------------------------------

def list_forms(db_engine: Engine, form_name: Optional[str] = None) -> dict[str, Any]:
    """
    List all forms. Sourced from ai_question.form_name (distinct values)
    so it only shows forms that actually have questions indexed.
    Optionally filter by partial name.
    """
    try:
        if form_name:
            sql = sql_text("""
                SELECT DISTINCT module_name,
                       COUNT(*) AS question_count
                FROM ai_questions
                WHERE module_name ILIKE :pattern
                  AND module_name IS NOT NULL AND module_name != ''
                GROUP BY module_name
                ORDER BY module_name
                LIMIT 100
            """)
            params = {"pattern": f"%{form_name}%"}
        else:
            sql = sql_text("""
                SELECT DISTINCT module_name,
                       COUNT(*) AS question_count
                FROM ai_questions
                WHERE module_name IS NOT NULL AND module_name != ''
                GROUP BY module_name
                ORDER BY module_name
                LIMIT 100
            """)
            params = {}

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
) -> dict[str, Any]:
    """
    Semantic search over ai_question labels.
    Returns matching questions sorted by semantic similarity.
    Use this when the user describes a topic and you need to find
    which questions/forms are related to it.
    """
    try:
        matches = semantic_index.search(
            query=query,
            entity_type="QUESTION",
            form_name=form_name,
            top_k=top_k,
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
) -> dict[str, Any]:
    """
    ILIKE search over ai_question.question_label.
    Use when the user knows part of the question text exactly.
    Complements semantic_search for exact keyword matches.
    """
    try:
        params: dict[str, Any] = {"limit": limit}
        filters = ["aq.label ILIKE :pattern"]
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
) -> dict[str, Any]:
    """
    Query submitted answers from ai_answer.
    Automatically joins ai_question for the question label.
    All filters are optional — combine as needed.
    """
    try:
        filters = ["1=1"]
        params: dict[str, Any] = {"limit": limit}

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
            filters.append("aa.inspection_id = :insp_id")
            params["insp_id"] = inspection_id
        if answer_value:
            filters.append(
                "(aa.answer_text ILIKE :ans_val OR aa.answer_numeric::text ILIKE :ans_val)"
            )
            params["ans_val"] = f"%{answer_value}%"

        where = " AND ".join(filters)

        # answer_text for categorical, answer_numeric for scores/numbers.
        # COALESCE so we always get one displayable value.
        sql = sql_text(f"""
            SELECT aa.inspection_id,
                   aa.module_name,
                   aq.label                                          AS question_label,
                   COALESCE(aa.answer_text, aa.answer_numeric::text) AS answer_value,
                   aa.score,
                   aa.submitted_on
            FROM ai_answers aa
            LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id
            WHERE {where}
            ORDER BY aa.submitted_on DESC
            LIMIT :limit
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
                    SELECT COUNT(*) FROM ai_answers aa
                    LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id
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
) -> dict[str, Any]:
    """
    Aggregate statistics over ai_answer.answer_value.
    Returns numeric stats (avg/min/max) + categorical value breakdown.
    Use for "most common answer", "average score", "distribution" questions.
    """
    try:
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
            FROM ai_answers aa
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
            FROM ai_answers aa
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
                ],
            },
            "error": None,
        }
    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Core tools: generate_sql, execute_sql, get_schema
# ---------------------------------------------------------------------------

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


def generate_sql(
    sql_llm, validator: SQLValidator, question: str,
    schema_hint: str = "", db_engine=None,
    sql_prompt_template: str = "",
    _debug_logger=None,
) -> dict[str, Any]:
    try:
        if db_engine and not schema_hint.startswith("Use"):
            auto_schema = _fetch_key_schemas(db_engine)
            schema_hint = f"{auto_schema}\n{schema_hint}" if schema_hint else auto_schema

        if not sql_prompt_template:
            from agent.prompts import SQL_GENERATION_PROMPT
            sql_prompt_template = SQL_GENERATION_PROMPT

        prompt = sql_prompt_template.replace(
            "{question}", question
        ).replace(
            "{schema_hint}", schema_hint or "No additional context provided."
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

        sql = validator.clean_sql(raw_sql)

        # validate() returns ValidationResult — hard fails block, soft warns are advisory only.
        vresult = validator.validate(sql)

        # validate_semantic() returns SOFT_WARN issues only.
        # These are logged for observability but NEVER injected into the retry prompt.
        # This is the core fix for validator poisoning: semantic heuristics must not
        # cause retries or appear in the errors list that the orchestrator reads.
        sem_warns = validator.validate_semantic(sql, question)

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


def execute_sql(db_engine: Engine, validator: SQLValidator, sql: str) -> dict[str, Any]:
    try:
        # validate() returns ValidationResult.
        # Only hard fails block execution — soft warns are advisory and ignored here.
        vresult = validator.validate(sql)
        if not vresult.passed:
            return {
                "success": False,
                "result": None,
                "error": "SQL failed validation: " + vresult.retry_message(),
            }

        with db_engine.connect() as conn:
            result = conn.execute(sql_text(sql))
            rows = result.fetchmany(50)
            columns = list(result.keys())

        data = [dict(zip(columns, row)) for row in rows]
        truncated = False
        total_count = len(data)

        if len(data) == 50:
            try:
                stripped = re.sub(r'\bLIMIT\s+\d+\s*;?\s*$', '', sql,
                                  flags=re.IGNORECASE).rstrip('; ')
                count_sql = f"SELECT COUNT(*) FROM ({stripped}) AS _sub"
                with db_engine.connect() as conn:
                    count_row = conn.execute(sql_text(count_sql)).fetchone()
                    if count_row:
                        total_count = count_row[0]
                        truncated = total_count > 50
            except Exception:
                truncated = True
                total_count = -1

        return {
            "success": True,
            "result": {
                "rows": data, "row_count": len(data),
                "total_count": total_count,
                "truncated": truncated, "columns": columns,
            },
            "error": None,
        }
    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


def get_schema(db_engine: Engine, table_name: Optional[str] = None) -> dict[str, Any]:
    try:
        inspector = sa_inspect(db_engine)
        if table_name is None:
            tables = inspector.get_table_names()
            return {"success": True, "result": {"tables": tables}, "error": None}
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
                 debug_logger=None) -> None:
        self._db_engine = db_engine
        self._semantic_index = semantic_index
        self._validator = validator
        self._sql_llm = sql_llm
        self._sql_prompt = sql_prompt
        self._debug_logger = debug_logger

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in self.KNOWN_TOOLS:
            return {
                "success": False, "result": None,
                "error": f"Unknown tool '{name}'. Available: {sorted(self.KNOWN_TOOLS)}",
            }
        try:
            if name == "list_forms":
                return list_forms(self._db_engine,
                                  form_name=args.get("form_name"))

            if name == "semantic_search":
                return semantic_search(
                    self._semantic_index,
                    query=str(args["query"]),
                    form_name=args.get("form_name"),
                    top_k=int(args.get("top_k", 10)),
                )

            if name == "search_questions":
                def _limit(v, default=30):
                    try:
                        return max(1, min(int(v), 200))
                    except (TypeError, ValueError):
                        return default
                return search_questions(
                    self._db_engine,
                    query=str(args["query"]),
                    form_name=args.get("form_name"),
                    module_name=args.get("module_name"),
                    question_type=args.get("question_type"),
                    limit=_limit(args.get("limit", 30)),
                )

            if name == "get_answers":
                def _clean(v):
                    if isinstance(v, str) and v.lower() in ('none', 'null', ''):
                        return None
                    return v
                # Accept both 'label' and 'question_label' — LLM uses both names
                q_label = args.get("question_label") or args.get("label")
                def _limit(v, default=50):
                    try:
                        n = int(v)
                        return max(1, min(n, 500))
                    except (TypeError, ValueError):
                        return default
                return get_answers(
                    self._db_engine,
                    form_name=_clean(args.get("form_name")),
                    module_name=_clean(args.get("module_name")),
                    question_label=_clean(q_label),
                    inspection_id=_clean(args.get("inspection_id")),
                    answer_value=_clean(args.get("answer_value")),
                    limit=_limit(args.get("limit", 50)),
                )

            if name == "get_answer_stats":
                def _clean(v):
                    """Convert string 'None'/'null'/'none' to actual None."""
                    if isinstance(v, str) and v.lower() in ('none', 'null', ''):
                        return None
                    return v
                # Accept both 'label' and 'question_label' — LLM uses both names
                q_label = args.get("question_label") or args.get("label")
                return get_answer_stats(
                    self._db_engine,
                    form_name=_clean(args.get("form_name")),
                    module_name=_clean(args.get("module_name")),
                    question_label=_clean(q_label),
                )

            if name == "generate_sql":
                return generate_sql(
                    self._sql_llm, self._validator,
                    question=str(args["question"]),
                    schema_hint=str(args.get("schema_hint", "")),
                    db_engine=self._db_engine,
                    sql_prompt_template=self._sql_prompt,
                    _debug_logger=self._debug_logger,
                )

            if name == "execute_sql":
                return execute_sql(self._db_engine, self._validator,
                                   sql=str(args["sql"]))

            if name == "get_schema":
                return get_schema(self._db_engine,
                                  table_name=args.get("table_name"))

        except KeyError as exc:
            return {"success": False, "result": None,
                    "error": f"Missing required argument for '{name}': {exc}"}
        except Exception as exc:
            return {"success": False, "result": None,
                    "error": f"Tool '{name}' raised an exception: {exc}"}
        return {"success": False, "result": None, "error": "Unhandled routing case"}