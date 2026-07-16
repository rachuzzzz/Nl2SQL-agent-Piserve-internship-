"""
Shared FastAPI app-state accessor.

Exists as its own module (not inside fastapi_app.py or auth_routes.py) so
both can import it without a circular dependency — fastapi_app.py includes
the router defined in auth_routes.py, so auth_routes.py can never import
back from fastapi_app.py.

Replaces the old `_orch = None` module-level global (only valid after
lifespan() ran — an import-order footgun) with FastAPI's standard app.state
pattern, read fresh on every request via dependency injection.
"""

from fastapi import HTTPException, Request


def get_orch(request: Request):
    """FastAPI dependency: the live AgentOrchestrator instance, or 503 if the
    server hasn't finished starting up yet (set on app.state.orch in
    fastapi_app.py's lifespan)."""
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
