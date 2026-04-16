"""
Tool implementations for the NL2SQL agent.

Each tool function is a pure function that receives its dependencies
explicitly — no hidden global state, so every tool is independently
testable.

The ToolRegistry class binds a set of dependencies (db_engine, semantic
index, etc.) and exposes a single call(name, kwargs) interface used by
the dispatcher.

Tool return contract:
  Every tool returns {"success": bool, "result": <any>, "error": str | None}
"""

import json
import re
from typing import Any, Optional

from sqlalchemy import inspect as sa_inspect, text as sql_text
from sqlalchemy.engine import Engine

from core.semantic import SemanticQuestionIndex
from core.validator import SQLValidator
from agent.prompts import SQL_GENERATION_PROMPT


# Minimum similarity score to accept a semantic match. Raised from the
# original 0.4 because bge-small tends to produce noisy matches for short
# queries — "Responsible" at 0.74 for "safety" is a false positive, but
# "Emergency Contact Name" at 0.91 for "emergency contact" is solid.
# 0.55 is a compromise between recall and precision.
_SEMANTIC_MIN_SCORE = 0.55

# Slightly lower threshold when falling back from form-name search to
# question-label search inside lookup_form. The match is indirect (we're
# finding a form via one of its questions) so we accept weaker signals.
_SEMANTIC_FALLBACK_MIN_SCORE = 0.50


# ---------------------------------------------------------------------------
# Individual tool functions (pure, dependency-injected)
# ---------------------------------------------------------------------------

def list_forms(db_engine: Engine, status: Optional[str] = None) -> dict[str, Any]:
    try:
        if status:
            query = sql_text(
                "SELECT name, status, active FROM fb_forms "
                "WHERE status = :status AND name IS NOT NULL AND name != '' "
                "ORDER BY name LIMIT 100"
            )
            params = {"status": status.upper()}
        else:
            query = sql_text(
                "SELECT name, status, active FROM fb_forms "
                "WHERE name IS NOT NULL AND name != '' "
                "ORDER BY name LIMIT 100"
            )
            params = {}

        with db_engine.connect() as conn:
            result = conn.execute(query, params)
            rows = result.fetchall()
            columns = list(result.keys())

        data = [dict(zip(columns, row)) for row in rows]
        return {"success": True, "result": data, "error": None}
    except Exception as exc:
        return {"success": False, "result": [], "error": str(exc)}


def lookup_form(
    db_engine: Engine,
    semantic_index: SemanticQuestionIndex,
    fuzzy_name: str,
) -> dict[str, Any]:
    """
    Resolve a fuzzy/partial form name to actual form names in the DB.

    Three-step cascade — stops at the first step that yields matches:

      1. ILIKE '%fuzzy_name%' against fb_forms.name  (fast, exact substring)
      2. Semantic search over indexed FORM names     (handles paraphrases)
      3. Semantic search over QUESTION labels →      (indirect: the form that
         collect distinct form_name values            *contains* a question
                                                      about this topic)

    Returns a list of form-name strings (up to 10). `_source` in the payload
    indicates which cascade step matched, useful for logging.
    """
    try:
        # --- Step 1: ILIKE ---
        query = sql_text(
            "SELECT name FROM fb_forms WHERE name ILIKE :pattern "
            "AND name IS NOT NULL AND name != '' "
            "ORDER BY name LIMIT 10"
        )
        with db_engine.connect() as conn:
            result = conn.execute(query, {"pattern": f"%{fuzzy_name}%"})
            names = [row[0] for row in result.fetchall()]

        if names:
            return {
                "success": True,
                "result": names,
                "error": None,
                "_source": "ilike",
            }

        # --- Step 2: semantic over FORM entries ---
        form_matches = semantic_index.search(
            fuzzy_name, entity_type="FORM", top_k=10
        )
        good_forms = [m for m in form_matches if m.score >= _SEMANTIC_MIN_SCORE]
        if good_forms:
            seen, names = set(), []
            for m in good_forms:
                if m.text and m.text not in seen:
                    seen.add(m.text)
                    names.append(m.text)
            return {
                "success": True,
                "result": names,
                "error": None,
                "_source": "semantic_form",
            }

        # --- Step 3: semantic over QUESTION labels → collect form_names ---
        q_matches = semantic_index.search(
            fuzzy_name, entity_type="QUESTION", top_k=20
        )
        good_q = [m for m in q_matches if m.score >= _SEMANTIC_FALLBACK_MIN_SCORE]
        if good_q:
            seen, names = set(), []
            for m in good_q:
                fn = (m.form_name or "").strip()
                if fn and fn not in seen:
                    seen.add(fn)
                    names.append(fn)
                    if len(names) >= 10:
                        break
            if names:
                return {
                    "success": True,
                    "result": names,
                    "error": None,
                    "_source": "semantic_question",
                }

        # --- No matches from any step ---
        return {
            "success": True,
            "result": [],
            "error": None,
            "_source": "none",
        }

    except Exception as exc:
        return {"success": False, "result": [], "error": str(exc)}


def semantic_search(
    semantic_index: SemanticQuestionIndex,
    query: str,
    form_name: Optional[str] = None,
    entity_type: Optional[str] = None,
    top_k: int = 10,
) -> dict[str, Any]:
    try:
        # Defensive: the tool-facing API advertises QUESTION|PAGE|null only.
        # If the LLM passes FORM or MODULE, normalise to None rather than
        # silently returning 0 matches (those entity types are reserved for
        # internal lookup_form use).
        if entity_type and entity_type.upper() in ("FORM", "MODULE"):
            entity_type = None

        matches = semantic_index.search(
            query=query,
            entity_type=entity_type if entity_type else None,
            form_name=form_name,
            top_k=top_k,
        )
        # Filter to QUESTION/PAGE only even when caller passed entity_type=None,
        # so the LLM-facing tool surface never leaks FORM/MODULE entries.
        good = [
            m for m in matches
            if m.score >= _SEMANTIC_MIN_SCORE
            and m.entity_type in ("QUESTION", "PAGE")
        ]
        data = [
            {
                "text": m.text,
                "form_name": m.form_name,
                "entity_type": m.entity_type,
                "score": round(m.score, 3),
            }
            for m in good
        ]
        return {"success": True, "result": data, "error": None}
    except Exception as exc:
        return {"success": False, "result": [], "error": str(exc)}


# Schema cache
_schema_cache: Optional[str] = None


def _fetch_key_schemas(db_engine: Engine) -> str:
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache
    try:
        inspector = sa_inspect(db_engine)
        key_tables = ["fb_forms", "fb_modules", "fb_translation_json"]
        parts = ["### Actual database columns (use these, not guesses):"]
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
    schema_hint: str = "", db_engine: Optional[Engine] = None,
) -> dict[str, Any]:
    try:
        if db_engine and not schema_hint.startswith("Use"):
            auto_schema = _fetch_key_schemas(db_engine)
            schema_hint = f"{auto_schema}\n{schema_hint}" if schema_hint else auto_schema

        prompt = SQL_GENERATION_PROMPT.format(
            question=question,
            schema_hint=schema_hint or "No additional context provided.",
        )
        response = sql_llm.complete(prompt)
        raw_sql = str(response).strip()
        sql = validator.clean_sql(raw_sql)
        passed, errors = validator.validate(sql)
        semantic_warnings = validator.validate_semantic(sql, question)
        all_errors = errors + semantic_warnings
        return {
            "success": True,
            "result": {
                "sql": sql,
                "validation": {"passed": passed, "errors": all_errors},
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
        passed, errors = validator.validate(sql)
        real_errors = [e for e in errors if not e.startswith("WARNING:")]
        if real_errors:
            return {
                "success": False, "result": None,
                "error": "SQL failed validation: " + "; ".join(real_errors),
            }
        with db_engine.connect() as conn:
            result = conn.execute(sql_text(sql))
            rows = result.fetchmany(50)
            columns = list(result.keys())
        data = [dict(zip(columns, row)) for row in rows]
        return {
            "success": True,
            "result": {"rows": data, "row_count": len(data), "columns": columns},
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
        col_info = [{"name": col["name"], "type": str(col["type"])} for col in columns]
        return {"success": True, "result": {"columns": col_info}, "error": None}
    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Answer data tools — dynamic module tables
# ---------------------------------------------------------------------------

_ANSWER_TABLE_RE = re.compile(
    r'^fb_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}$'
)

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)

_REQUIRED_ANSWER_COLUMNS = {"form_name", "status", "answer_data", "created_on", "modified_on"}


def _validate_answer_table_name(table_name: str) -> Optional[str]:
    if not _ANSWER_TABLE_RE.match(table_name):
        return f"Invalid answer table name: '{table_name}'. Expected fb_<uuid> pattern."
    return None


def _safe_array(expr: str) -> str:
    """NULL-safe jsonb_array_elements fragment — see query_answers docstring."""
    return (
        f"jsonb_array_elements("
        f"CASE WHEN jsonb_typeof({expr}) = 'array' "
        f"THEN {expr} ELSE NULL END)"
    )


def resolve_answer_table(db_engine: Engine, form_name: str) -> dict[str, Any]:
    try:
        query = sql_text("""
            SELECT f.id AS form_id, f.name AS form_name,
                   f.module_id, m.name AS module_name
            FROM fb_forms f
            JOIN fb_modules m ON f.module_id = m.id
            WHERE f.name ILIKE :pattern
              AND f.name IS NOT NULL AND f.name != ''
            ORDER BY f.name
            LIMIT 5
        """)
        with db_engine.connect() as conn:
            result = conn.execute(query, {"pattern": f"%{form_name}%"})
            rows = result.fetchall()
            columns = list(result.keys())

        if not rows:
            return {
                "success": False, "result": None,
                "error": f"No form found matching '{form_name}'. Use lookup_form first.",
            }

        inspector = sa_inspect(db_engine)
        matches = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            module_id = str(row_dict["module_id"])
            table_name = "fb_" + module_id.replace("-", "_")

            if not _ANSWER_TABLE_RE.match(table_name):
                continue

            try:
                with db_engine.connect() as conn:
                    conn.execute(sql_text(f"SELECT 1 FROM {table_name} LIMIT 1"))
                table_exists = True
            except Exception:
                table_exists = False

            table_columns: list[str] = []
            missing_required: list[str] = []
            if table_exists:
                try:
                    cols = inspector.get_columns(table_name)
                    table_columns = [c["name"] for c in cols]
                    missing_required = sorted(
                        _REQUIRED_ANSWER_COLUMNS - set(table_columns)
                    )
                except Exception:
                    pass

            matches.append({
                "answer_table": table_name,
                "attachment_table": table_name + "_attachments",
                "table_exists": table_exists,
                "module_id": module_id,
                "module_name": row_dict["module_name"],
                "form_name": row_dict["form_name"],
                "form_id": str(row_dict["form_id"]),
                "columns": table_columns,
                "missing_required_columns": missing_required,
            })

        fully_usable = [
            m for m in matches
            if m["table_exists"] and not m["missing_required_columns"]
        ]
        merely_existing = [m for m in matches if m["table_exists"]]

        if fully_usable:
            best = fully_usable[0]
        elif merely_existing:
            best = merely_existing[0]
        else:
            best = matches[0]

        result = {
            **best,
            "matches": matches,
            "ambiguous": len(matches) > 1,
        }
        return {"success": True, "result": result, "error": None}

    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


def query_answers(
    db_engine: Engine, validator: SQLValidator,
    answer_table: str,
    form_name: Optional[str] = None,
    question_id: Optional[str] = None,
    status: Optional[str] = None,
    include_scores: bool = False,
    limit: int = 50,
) -> dict[str, Any]:
    try:
        err = _validate_answer_table_name(answer_table)
        if err:
            return {"success": False, "result": None, "error": err}

        if status and status.upper() not in ("DRAFT", "PUBLISHED", "DELETED"):
            return {"success": False, "result": None,
                    "error": f"Invalid status: '{status}'. Use DRAFT, PUBLISHED, or DELETED."}
        if question_id and not _UUID_RE.match(question_id):
            return {"success": False, "result": None,
                    "error": f"Invalid question_id: '{question_id}'. Must be a UUID."}
        limit = min(max(int(limit), 1), 200)

        where_parts = ["fa.answer_data IS NOT NULL"]
        params = {}

        if form_name:
            where_parts.append("fa.form_name ILIKE :form_pattern")
            params["form_pattern"] = f"%{form_name}%"
        if status:
            where_parts.append("fa.status = :status")
            params["status"] = status.upper()

        where_clause = "WHERE " + " AND ".join(where_parts)

        forms_elem = _safe_array("fa.answer_data->'forms'")
        subs_elem  = _safe_array("form_elem->'submissions'")
        answers_elem = _safe_array("sub_elem->'answers'")
        scores_elem  = _safe_array("sub_elem->'scores'")

        if question_id:
            params["question_id"] = question_id
            params["limit_val"] = limit
            sql = f"""
                SELECT fa.form_name, fa.status,
                       sub_elem->>'submissionId' AS submission_id,
                       ans_elem->>'questionId'   AS question_id,
                       ans_elem->>'answer'        AS answer,
                       ans_elem->>'maximumPossibleScore' AS max_score,
                       fa.created_on, fa.modified_on
                FROM {answer_table} fa,
                     {forms_elem} AS form_elem,
                     {subs_elem}  AS sub_elem,
                     {answers_elem} AS ans_elem
                {where_clause}
                  AND ans_elem->>'questionId' = :question_id
                LIMIT :limit_val
            """
        elif include_scores:
            params["limit_val"] = limit
            sql = f"""
                SELECT fa.form_name, fa.status,
                       sub_elem->>'submissionId' AS submission_id,
                       score_elem->>'score'      AS score,
                       score_elem->>'scoreType'  AS score_type,
                       fa.created_on
                FROM {answer_table} fa,
                     {forms_elem} AS form_elem,
                     {subs_elem}  AS sub_elem,
                     {scores_elem} AS score_elem
                {where_clause}
                LIMIT :limit_val
            """
        else:
            params["limit_val"] = limit
            sql = f"""
                SELECT fa.form_name, fa.status,
                       sub_elem->>'submissionId' AS submission_id,
                       ans_elem->>'questionId'   AS question_id,
                       ans_elem->>'answer'        AS answer,
                       fa.created_on, fa.modified_on
                FROM {answer_table} fa,
                     {forms_elem} AS form_elem,
                     {subs_elem}  AS sub_elem,
                     {answers_elem} AS ans_elem
                {where_clause}
                LIMIT :limit_val
            """

        with db_engine.connect() as conn:
            result = conn.execute(sql_text(sql), params)
            rows = result.fetchmany(limit)
            columns = list(result.keys())

        data = [dict(zip(columns, row)) for row in rows]
        return {
            "success": True,
            "result": {
                "rows": data, "row_count": len(data),
                "columns": columns, "table": answer_table,
            },
            "error": None,
        }
    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


def get_answer_summary(db_engine: Engine, answer_table: str) -> dict[str, Any]:
    try:
        err = _validate_answer_table_name(answer_table)
        if err:
            return {"success": False, "result": None, "error": err}

        sql = f"""
            SELECT COUNT(*) AS total_rows,
                   COUNT(DISTINCT form_name) AS distinct_forms,
                   MIN(created_on) AS earliest,
                   MAX(modified_on) AS latest
            FROM {answer_table};
        """
        with db_engine.connect() as conn:
            result = conn.execute(sql_text(sql))
            row = result.fetchone()
            columns = list(result.keys())
        data = dict(zip(columns, row)) if row else {}

        sql2 = f"""
            SELECT status, COUNT(*) AS count
            FROM {answer_table}
            GROUP BY status ORDER BY count DESC;
        """
        with db_engine.connect() as conn:
            result2 = conn.execute(sql_text(sql2))
            status_rows = result2.fetchall()
        data["status_breakdown"] = {r[0]: r[1] for r in status_rows}

        return {"success": True, "result": data, "error": None}
    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    KNOWN_TOOLS = {
        "list_forms", "lookup_form", "semantic_search",
        "generate_sql", "execute_sql", "get_schema",
        "resolve_answer_table", "query_answers", "get_answer_summary",
    }

    def __init__(self, db_engine: Engine, semantic_index: SemanticQuestionIndex,
                 validator: SQLValidator, sql_llm) -> None:
        self._db_engine = db_engine
        self._semantic_index = semantic_index
        self._validator = validator
        self._sql_llm = sql_llm

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in self.KNOWN_TOOLS:
            return {
                "success": False, "result": None,
                "error": f"Unknown tool '{name}'. Available: {sorted(self.KNOWN_TOOLS)}",
            }
        try:
            if name == "list_forms":
                return list_forms(self._db_engine, status=args.get("status"))
            if name == "lookup_form":
                return lookup_form(
                    self._db_engine, self._semantic_index,
                    fuzzy_name=str(args["fuzzy_name"]),
                )
            if name == "semantic_search":
                return semantic_search(
                    self._semantic_index, query=str(args["query"]),
                    form_name=args.get("form_name"),
                    entity_type=args.get("entity_type"),
                    top_k=int(args.get("top_k", 10)),
                )
            if name == "generate_sql":
                return generate_sql(
                    self._sql_llm, self._validator,
                    question=str(args["question"]),
                    schema_hint=str(args.get("schema_hint", "")),
                    db_engine=self._db_engine,
                )
            if name == "execute_sql":
                return execute_sql(self._db_engine, self._validator, sql=str(args["sql"]))
            if name == "get_schema":
                return get_schema(self._db_engine, table_name=args.get("table_name"))

            if name == "resolve_answer_table":
                return resolve_answer_table(self._db_engine, form_name=str(args["form_name"]))
            if name == "query_answers":
                return query_answers(
                    self._db_engine, self._validator,
                    answer_table=str(args["answer_table"]),
                    form_name=args.get("form_name"),
                    question_id=args.get("question_id"),
                    status=args.get("status"),
                    include_scores=bool(args.get("include_scores", False)),
                    limit=int(args.get("limit", 50)),
                )
            if name == "get_answer_summary":
                return get_answer_summary(self._db_engine, answer_table=str(args["answer_table"]))

        except KeyError as exc:
            return {"success": False, "result": None,
                    "error": f"Missing required argument for '{name}': {exc}"}
        except Exception as exc:
            return {"success": False, "result": None,
                    "error": f"Tool '{name}' raised an exception: {exc}"}
        return {"success": False, "result": None, "error": "Unhandled routing case"}