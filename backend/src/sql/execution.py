"""
SQL execution against the live database. Extracted from agent/tools.py,
alongside sql/generation.py.
"""

import re
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

from src.sql.validation import SQLValidator


def _strip_limit_offset(sql: str) -> str:
    """Remove a trailing LIMIT/OFFSET clause so a fresh one can be appended.
    Single source of truth — was previously duplicated near-identically in
    api/deps.py's refetch_for_calendar and api/response_formatting.py's
    paginate_sql."""
    return re.sub(
        r'\bLIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*;?\s*$', '',
        sql.strip(), flags=re.IGNORECASE
    ).strip().rstrip(';')


def execute_sql(db_engine: Engine, validator: SQLValidator, sql: str, question: str = "",
                 offset: int = 0, limit: int = 50, force: bool = False) -> dict[str, Any]:
    try:
        # validate() returns ValidationResult.
        # Only hard fails block execution — soft warns are advisory and ignored here.
        # question is threaded through so question-scoped HARD_FAILs (e.g.
        # OVERDUE_DATE_ARITHMETIC, CYCLE_UUID_NO_JOIN) still apply here — this is
        # also the re-validation gate used by the "forced execution after 3
        # failures" fallback in the orchestrator, which must not silently bypass them.
        vresult = validator.validate(sql, question)
        if not vresult.passed:
            # force=True is the orchestrator's "forced execution after 3
            # validation failures" safety valve (loop.py) — but calling
            # execute_sql() again on the SAME sql just re-runs this same
            # validate() and fails again identically, since nothing here
            # bypasses it. Confirmed live: a real capex query burned 8
            # iterations / 76s / ~172K tokens because "forced execution"
            # never actually forced anything — every retry hit this same
            # HARD_FAIL return, the loop kept treating it as an ordinary
            # failure, and gen_sql_failures just kept climbing past 3.
            # Real fix: when force=True, let every RETRYABLE issue through to
            # Postgres itself (a genuine bug still fails there, just via one
            # real DB error instead of an endless validation loop) — but
            # NEVER bypass a non-retryable issue (retryable=False), which is
            # reserved for actual safety blocks like DML_BLOCKED, not
            # logical-correctness guards.
            _unsafe = [i for i in vresult.hard_fails if not i.retryable]
            if not force or _unsafe:
                return {
                    "success": False,
                    "result": None,
                    "error": "SQL failed validation: " + vresult.retry_message(),
                }

        # offset > 0 means a "show more" pagination re-fetch (see orchestrator
        # loop.py's Path A) — rewrite the query for the requested window instead
        # of always re-running the same first `limit` rows. offset == 0 keeps the
        # original fetchmany-only behavior byte-for-byte (fetchmany is cheaper
        # than a rewritten LIMIT/OFFSET query for the common non-paginated case).
        if offset > 0:
            stripped = _strip_limit_offset(sql)
            windowed_sql = f"{stripped}\nLIMIT {limit} OFFSET {offset};"
            with db_engine.connect() as conn:
                result = conn.execute(sql_text(windowed_sql))
                rows = result.fetchall()
                columns = list(result.keys())
        else:
            with db_engine.connect() as conn:
                result = conn.execute(sql_text(sql))
                rows = result.fetchmany(limit)
                columns = list(result.keys())

        data = [dict(zip(columns, row)) for row in rows]
        truncated = False
        total_count = len(data)

        if len(data) == limit:
            try:
                stripped = _strip_limit_offset(sql)
                count_sql = f"SELECT COUNT(*) FROM ({stripped}) AS _sub"
                with db_engine.connect() as conn:
                    count_row = conn.execute(sql_text(count_sql)).fetchone()
                    if count_row:
                        total_count = count_row[0]
                        truncated = total_count > (offset + limit)
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
