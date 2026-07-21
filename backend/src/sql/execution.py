"""
SQL execution against the live database. Extracted from agent/tools.py,
alongside sql/generation.py.
"""

import re
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

from src.sql.validation import SQLValidator


def execute_sql(db_engine: Engine, validator: SQLValidator, sql: str, question: str = "") -> dict[str, Any]:
    try:
        # validate() returns ValidationResult.
        # Only hard fails block execution — soft warns are advisory and ignored here.
        # question is threaded through so question-scoped HARD_FAILs (e.g.
        # OVERDUE_DATE_ARITHMETIC, CYCLE_UUID_NO_JOIN) still apply here — this is
        # also the re-validation gate used by the "forced execution after 3
        # failures" fallback in the orchestrator, which must not silently bypass them.
        vresult = validator.validate(sql, question)
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
