"""
Health/readiness/diagnostics routes — /api/health, /api/ready,
/api/validator-stats.
"""

import os

from fastapi import APIRouter, Request

from khidmah.api.deps import get_orch_optional

router = APIRouter()


@router.get("/api/health")
async def health(request: Request):
    _health_orch = get_orch_optional(request)
    if _health_orch is not None:
        model_label = f"{_health_orch.chat_model_name} + {_health_orch.sql_model_name}"
    else:
        # Intentionally not khidmah.config.settings here — this display-only
        # fallback uses "—" as its default (shown when the orchestrator hasn't
        # started yet), not settings.py's functional default for these vars.
        chat = os.getenv("OLLAMA_CHAT_MODEL", "—")
        sql  = os.getenv("OLLAMA_SQL_MODEL",  "—")
        model_label = f"{chat} + {sql}"
    return {"status": "ok", "model": model_label}

@router.get("/api/ready")
async def ready(request: Request):
    """Used by start.sh to wait until the orchestrator is fully loaded."""
    from fastapi.responses import JSONResponse
    if get_orch_optional(request) is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready"}

@router.get("/api/validator-stats")
async def validator_stats():
    """
    Read-only diagnostic: how often each validator rule has fired since process
    start. In-memory only (resets on restart, same as _sessions) — enough to
    notice a SOFT_WARN firing constantly without needing to grep debug logs.
    """
    from khidmah.core.validator import ValidatorStats
    return ValidatorStats.snapshot()
