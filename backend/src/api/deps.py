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


def get_orch_inspection(request: Request):
    """FastAPI dependency: the live inspection-domain AgentOrchestrator, or
    503 if not ready. Same instance as get_orch() (app.state.orch is an
    alias) — this name exists so domain-routing call sites (chat()) are
    explicit about which instance they mean once a second domain exists."""
    orch = getattr(request.app.state, "orch_inspection", None)
    if orch is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return orch


def get_orch_audit(request: Request):
    """FastAPI dependency: the live audit-domain AgentOrchestrator, or 503 if
    not ready."""
    orch = getattr(request.app.state, "orch_audit", None)
    if orch is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return orch


def get_router(request: Request):
    """FastAPI dependency: the live DomainRouter, or 503 if not ready."""
    router = getattr(request.app.state, "router", None)
    if router is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return router


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
    """Deprecated — calendar was the first chart type to need more than the
    50-row execute_sql cache, but every chart type needs this now. Use
    charts.data_fetch.fetch_chart_data instead (all three chart-producing
    routes have been migrated off this). Kept only until nothing imports it."""
    from src.charts.data_fetch import fetch_chart_data
    return fetch_chart_data(sql, orch, cap=limit, params=params)
