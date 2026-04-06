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
from typing import Any, Optional

from sqlalchemy import inspect as sa_inspect, text as sql_text
from sqlalchemy.engine import Engine

from core.semantic import SemanticQuestionIndex
from core.validator import SQLValidator
from agent.prompts import SQL_GENERATION_PROMPT


# ---------------------------------------------------------------------------
# Individual tool functions (pure, dependency-injected)
# ---------------------------------------------------------------------------

def list_forms(db_engine: Engine, status: Optional[str] = None) -> dict[str, Any]:
    """
    List forms from the database.
    status — optional filter: 'DRAFT', 'PUBLISHED', 'DELETED', or None for all.
    """
    try:
        if status:
            query = sql_text(
                "SELECT name, status, active FROM fb_forms "
                "WHERE status = :status ORDER BY name LIMIT 100"
            )
            params = {"status": status.upper()}
        else:
            query = sql_text(
                "SELECT name, status, active FROM fb_forms ORDER BY name LIMIT 100"
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


def lookup_form(db_engine: Engine, fuzzy_name: str) -> dict[str, Any]:
    """
    Find forms whose name contains fuzzy_name (case-insensitive).
    Returns a list of actual form name strings so the agent can
    pick the right one before constructing SQL.
    """
    try:
        query = sql_text(
            "SELECT name FROM fb_forms WHERE name ILIKE :pattern ORDER BY name LIMIT 10"
        )
        with db_engine.connect() as conn:
            result = conn.execute(query, {"pattern": f"%{fuzzy_name}%"})
            names = [row[0] for row in result.fetchall()]
        return {"success": True, "result": names, "error": None}

    except Exception as exc:
        return {"success": False, "result": [], "error": str(exc)}


def semantic_search(
    semantic_index: SemanticQuestionIndex,
    query: str,
    form_name: Optional[str] = None,
    entity_type: Optional[str] = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Embedding-based search over question/page/section labels.
    Returns only matches with score >= 0.4.
    """
    try:
        matches = semantic_index.search(
            query=query,
            entity_type=entity_type if entity_type else None,
            form_name=form_name,
            top_k=top_k,
        )
        good = [m for m in matches if m.score >= 0.4]
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


def generate_sql(
    sql_llm,
    validator: SQLValidator,
    question: str,
    schema_hint: str = "",
) -> dict[str, Any]:
    """
    Ask sqlcoder:7b to produce a PostgreSQL SELECT for the given question.
    Applies SQLValidator.clean_sql() and validate() on the output.

    sql_llm — an llama_index Ollama client initialised with sqlcoder:7b.
    """
    try:
        prompt = SQL_GENERATION_PROMPT.format(
            question=question,
            schema_hint=schema_hint or "No additional context provided.",
        )
        response = sql_llm.complete(prompt)
        raw_sql = str(response).strip()
        sql = validator.clean_sql(raw_sql)
        passed, errors = validator.validate(sql)
        # Semantic cross-check: does the SQL match what the question asks for?
        semantic_warnings = validator.validate_semantic(sql, question)
        all_errors = errors + semantic_warnings
        # A semantic warning about missing JSONB is treated as a hard error so
        # the agent is forced to regenerate rather than returning a wrong count.
        if semantic_warnings:
            passed = False
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


def execute_sql(
    db_engine: Engine,
    validator: SQLValidator,
    sql: str,
) -> dict[str, Any]:
    """
    Validate then execute a SQL SELECT.  Returns up to 50 rows as a list
    of dicts.  Refuses to run SQL that fails validation (non-WARNING errors).
    """
    try:
        passed, errors = validator.validate(sql)
        real_errors = [e for e in errors if not e.startswith("WARNING:")]
        warnings = [e for e in errors if e.startswith("WARNING:")]
        # Validator warnings are returned as observations, not execution gates.
        # Only a database-level exception stops execution.

        with db_engine.connect() as conn:
            result = conn.execute(sql_text(sql))
            rows = result.fetchmany(51)
            truncated = len(rows) > 50
            if truncated:
                rows = rows[:50]
            columns = list(result.keys())

        data = [dict(zip(columns, row)) for row in rows]
        return {
            "success": True,
            "result": {
                "rows": data,
                "row_count": len(data),
                "columns": columns,
                "truncated": truncated,
                "validator_warnings": real_errors + warnings,
            },
            "error": None,
        }

    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


def validate_sql(validator: SQLValidator, sql: str) -> dict[str, Any]:
    """
    Run the SQL validator on a query without executing it.
    Returns the full validation result including all errors and warnings.
    The model uses this to decide whether to execute or regenerate.
    """
    try:
        passed, errors = validator.validate(sql)
        real_errors = [e for e in errors if not e.startswith("WARNING:")]
        warnings = [e for e in errors if e.startswith("WARNING:")]
        return {
            "success": True,
            "result": {
                "passed": passed,
                "real_errors": real_errors,
                "warnings": warnings,
                "sql_reviewed": sql,
            },
            "error": None,
        }
    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


def get_schema(db_engine: Engine, table_name: Optional[str] = None) -> dict[str, Any]:
    """
    Inspect the database schema.
    table_name=None → return list of all table names.
    table_name given → return [{name, type}, ...] for that table's columns.
    """
    try:
        inspector = sa_inspect(db_engine)

        if table_name is None:
            tables = inspector.get_table_names()
            return {"success": True, "result": {"tables": tables}, "error": None}

        columns = inspector.get_columns(table_name)
        col_info = [
            {"name": col["name"], "type": str(col["type"])}
            for col in columns
        ]
        return {"success": True, "result": {"columns": col_info}, "error": None}

    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# ToolRegistry — binds deps, exposes call(name, kwargs) -> dict
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Holds all runtime dependencies and routes agent tool calls to the correct
    function.  Constructed once by AgentOrchestrator; passed to the dispatcher.

    All tool names map directly to agent/prompts.py SYSTEM_PROMPT descriptions.
    """

    KNOWN_TOOLS = {
        "list_forms",
        "lookup_form",
        "semantic_search",
        "generate_sql",
        "validate_sql",
        "execute_sql",
        "get_schema",
    }

    def __init__(
        self,
        db_engine: Engine,
        semantic_index: SemanticQuestionIndex,
        validator: SQLValidator,
        sql_llm,  # llama_index Ollama client (sqlcoder:7b)
    ) -> None:
        self._db_engine = db_engine
        self._semantic_index = semantic_index
        self._validator = validator
        self._sql_llm = sql_llm

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Route a tool name + args dict to the appropriate function.
        Returns a standardised {"success", "result", "error"} dict.
        Unknown tool names return an error without raising.
        """
        if name not in self.KNOWN_TOOLS:
            return {
                "success": False,
                "result": None,
                "error": f"Unknown tool '{name}'. Available: {sorted(self.KNOWN_TOOLS)}",
            }

        try:
            if name == "list_forms":
                return list_forms(self._db_engine, status=args.get("status"))

            if name == "lookup_form":
                return lookup_form(self._db_engine, fuzzy_name=str(args["fuzzy_name"]))

            if name == "semantic_search":
                return semantic_search(
                    self._semantic_index,
                    query=str(args["query"]),
                    form_name=args.get("form_name"),
                    entity_type=args.get("entity_type"),
                    top_k=int(args.get("top_k", 10)),
                )

            if name == "generate_sql":
                return generate_sql(
                    self._sql_llm,
                    self._validator,
                    question=str(args["question"]),
                    schema_hint=str(args.get("schema_hint", "")),
                )

            if name == "validate_sql":
                return validate_sql(
                    self._validator,
                    sql=str(args["sql"]),
                )

            if name == "execute_sql":
                return execute_sql(
                    self._db_engine,
                    self._validator,
                    sql=str(args["sql"]),
                )

            if name == "get_schema":
                return get_schema(
                    self._db_engine,
                    table_name=args.get("table_name"),
                )

        except KeyError as exc:
            return {
                "success": False,
                "result": None,
                "error": f"Missing required argument for '{name}': {exc}",
            }
        except Exception as exc:
            return {
                "success": False,
                "result": None,
                "error": f"Tool '{name}' raised an exception: {exc}",
            }

        # Should never reach here
        return {"success": False, "result": None, "error": "Unhandled routing case"}
