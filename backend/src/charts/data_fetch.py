"""
Chart-specific data fetching — decouples chart rendering from execute_sql's
50-row cache (sess_data["last_raw_rows"]).

Generalizes the old api/deps.py's refetch_for_calendar (calendar-only) into a
single helper every chart type/trigger path uses, so chart data volume is no
longer inconsistent depending on which of the three chart-producing code
paths (auto-embed dashboard, chart-followup replay, /api/chart) happened to
run — see the chart redesign plan for the full "disoriented charts" root
cause.
"""

from typing import Any

from sqlalchemy import text as sql_text

from src.sql.execution import _strip_limit_offset


def fetch_chart_data(sql: str, orch, cap: int | None = None,
                      params: dict[str, Any] | None = None) -> list[dict]:
    """Re-run sql with a higher row limit than execute_sql's 50-row cache, for
    building a chart against the real dataset instead of whatever fraction
    happened to be cached from the original table-display fetch.

    Returns [] on any error (no last_sql, orch not ready, bad SQL, DB error)
    so every caller can fall back to its existing capped sample rather than
    failing the request — same contract the calendar-only version had.

    UUID-scrubbed via response_formatting.strip_uuid_columns() before return —
    added 2026-07-29: the regular chat table already scrubs raw UUID columns
    (chat.py calls strip_uuid_columns twice) but this function, the single
    choke point all three chart-producing paths share, never did. A query
    that SELECTs both a UUID (e.g. u.id AS user_id) and its human-readable
    name (e.g. inspector_name) passes SQL validation fine — but the chart
    advisor picks its category/x-axis column positionally, so if the UUID
    column happens to come first in the SELECT list, the chart renders UUIDs
    on the axis instead of names. Caught live 2026-07-29 on "how many
    inspectors have not submitted their portfolios" (SELECT u.id AS user_id,
    u.first_name || ' ' || u.last_name AS inspector_name, ...).
    """
    if not sql or orch is None:
        return []
    from src.config.settings import get_settings
    cap = cap or get_settings().chart_fetch_cap
    try:
        stripped = _strip_limit_offset(sql)
        full_sql = f"{stripped}\nLIMIT {cap};"
        with orch.db_engine.connect() as conn:
            result = conn.execute(sql_text(full_sql), params or {})
            rows = result.fetchall()
            cols = list(result.keys())
        data = [dict(zip(cols, r)) for r in rows]
        if data:
            from src.api.response_formatting import strip_uuid_columns
            _, data = strip_uuid_columns(cols, data)
        return data
    except Exception:
        return []
