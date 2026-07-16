"""
Shared FastAPI dependencies: app-state accessors, the session store, and the
repo-root path constant.

Exists as its own module (not inside app.py or any routes/*.py file) so all
of them can import it without a circular dependency — app.py includes the
routers defined in routes/*.py, so those modules can never import back from
app.py.

Replaces the old `_orch = None` module-level global (only valid after
lifespan() ran — an import-order footgun) with FastAPI's standard app.state
pattern, read fresh on every request via dependency injection.
"""

import os

from fastapi import HTTPException, Request

# __file__ is backend/src/api/deps.py — repo root is 4 levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_orch(request: Request):
    """FastAPI dependency: the live AgentOrchestrator instance, or 503 if the
    server hasn't finished starting up yet (set on app.state.orch in
    app.py's lifespan)."""
    orch = getattr(request.app.state, "orch", None)
    if orch is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return orch


def get_orch_optional(request: Request):
    """Non-raising variant for call sites that degrade gracefully instead of
    failing the whole request when the orchestrator isn't ready yet (e.g. a
    best-effort calendar-heatmap refetch, or an informational model-name
    display)."""
    return getattr(request.app.state, "orch", None)


# ── Session store ──────────────────────────────────────────────────────────
# Shared mutable state read and written by routes/chat.py (chat, get_page,
# clear_session, get_last_sql, get_last_columns) and routes/charts.py
# (get_chart).

_sessions: dict = {}


def get_or_create_session(session_id: str) -> dict:
    from src.agent.orchestrator import ConversationSession
    if session_id not in _sessions:
        _sessions[session_id] = {
            "session": ConversationSession(),
            "message_count": 0,
            "last_raw_rows": [],
            "last_cols": [],
            "last_sql": None,
            "last_sql_params": {},   # bound params when last_sql uses :placeholder style
            "last_total": None,
        }
    return _sessions[session_id]


def refetch_for_calendar(sql: str, orch, limit: int = 1000, params: dict | None = None) -> list[dict]:
    """Re-run sql with a higher row limit for calendar heatmap charts.
    Calendar charts need the full date range — the default 50-row cache
    only covers the most recent 50 inspections, leaving the rest of the
    calendar blank. Returns [] on any error so callers can fall back."""
    import re as _re
    from sqlalchemy import text as sql_text
    try:
        stripped = _re.sub(
            r'\bLIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*;?\s*$', '',
            sql.strip(), flags=_re.IGNORECASE
        ).strip().rstrip(';')
        full_sql = f"{stripped}\nLIMIT {limit};"
        with orch.db_engine.connect() as conn:
            result = conn.execute(sql_text(full_sql), params or {})
            rows = result.fetchall()
            cols = list(result.keys())
        return [dict(zip(cols, r)) for r in rows]
    except Exception:
        return []
