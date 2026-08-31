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
import time

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
#
# Phase 5 hardening: in-memory dict that previously grew without bound (one
# ConversationSession + last rows/cols/sql per "session_id:sub", forever).
# Now every touch stamps _last_seen and prunes: drop entries idle longer than
# SESSION_TTL_S, and if still over SESSION_MAX, LRU-evict the oldest. Still
# process-local and lost on restart — acceptable for a chat store, and the
# only thing lost is conversational context, not data.

_sessions: dict = {}
_session_seen: dict[str, float] = {}


def _prune_sessions() -> None:
    from src.config.settings import get_settings
    s = get_settings()
    now = time.time()
    stale = [k for k, t in _session_seen.items() if now - t > s.session_ttl_s]
    for k in stale:
        _sessions.pop(k, None)
        _session_seen.pop(k, None)
    over = len(_sessions) - s.session_max
    if over > 0:
        # oldest-first LRU eviction
        for k, _ in sorted(_session_seen.items(), key=lambda kv: kv[1])[:over]:
            _sessions.pop(k, None)
            _session_seen.pop(k, None)


def forget_session(session_id: str) -> None:
    """Drop a session from both the store and the LRU index (used by
    DELETE /api/sessions/{id})."""
    _sessions.pop(session_id, None)
    _session_seen.pop(session_id, None)


def get_or_create_session(session_id: str) -> dict:
    from src.agent.orchestrator import ConversationSession
    is_new = session_id not in _sessions
    if is_new:
        _sessions[session_id] = {
            "session": ConversationSession(),
            "message_count": 0,
            "last_raw_rows": [],
            "last_cols": [],
            "last_sql": None,
            "last_sql_params": {},   # bound params when last_sql uses :placeholder style
            "last_total": None,
        }
    _session_seen[session_id] = time.time()
    if is_new:
        # prune AFTER inserting (with the fresh timestamp above) so the count
        # settles to exactly SESSION_MAX and the just-created entry is never
        # the one evicted.
        _prune_sessions()
    return _sessions[session_id]


def refetch_for_calendar(sql: str, orch, limit: int = 1000, params: dict | None = None) -> list[dict]:
    """Deprecated — calendar was the first chart type to need more than the
    50-row execute_sql cache, but every chart type needs this now. Use
    charts.data_fetch.fetch_chart_data instead (all three chart-producing
    routes have been migrated off this). Kept only until nothing imports it."""
    from src.charts.data_fetch import fetch_chart_data
    return fetch_chart_data(sql, orch, cap=limit, params=params)
