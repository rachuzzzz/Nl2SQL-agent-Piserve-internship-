"""
InspectAI — FastAPI Backend
===========================
App factory: lifespan (orchestrator warm-up + semantics consistency check),
CORS, router registration, and the Angular SPA static mount.

Install:  pip install fastapi uvicorn
Run:      uvicorn src.api.app:app --app-dir backend --reload --port 8000
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.logging import setup_logging
from src.api.deps import REPO_ROOT

log = logging.getLogger("inspectai")

# ── Eager startup ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up both domain orchestrators + the domain router at server start
    so the first request is instant. The audit instance shares the
    inspection instance's DB connection pool, embedding model, and reasoning
    LLM client rather than building its own (one pool/model/client for both
    domains — see agent/domains.py's DOMAIN_REGISTRY and the domain-router
    plan's §3/§6)."""
    app.state.orch = None
    app.state.orch_inspection = None
    app.state.orch_audit = None
    app.state.router = None
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging(log_level="INFO")
    log.info("[SERVER] Starting InspectAI — loading orchestrators (LLM + embeddings + schema)…")
    t0 = time.time()
    from src.agent.orchestrator import AgentOrchestrator
    from src.agent.domain_router import DomainRouter
    app.state.orch_inspection = AgentOrchestrator.from_env(domain="inspection")
    app.state.orch_audit = AgentOrchestrator.from_env(
        domain="audit",
        db_engine=app.state.orch_inspection.db_engine,
        embed_model=app.state.orch_inspection.embed_model,
        reasoning_llm=app.state.orch_inspection.reasoning_llm,
    )
    # Back-compat alias — the ~5 call sites that aren't domain-aware yet
    # (get_page()'s best-effort calendar refetch, health/chart-advisor spots)
    # keep working against the inspection instance unchanged.
    app.state.orch = app.state.orch_inspection
    app.state.router = DomainRouter(
        reasoning_llm=app.state.orch_inspection.reasoning_llm,
        debug_logger=app.state.orch_inspection._debug,
    )
    log.info("[SERVER] Orchestrators ready in %.1fs", time.time() - t0)

    # Catch definition desyncs between knowledge/semantics_registry.py (the
    # declared single source of truth) and the prompt/example files that
    # actually feed the SQL-generating LLM. This previously existed as a
    # CLI-only tool nobody ran automatically — the "overdue" bug (fixed
    # 2026-07-07) sat undetected for an unknown period because of exactly
    # that gap.
    from src.knowledge.semantics_registry import verify_consistency
    _src_dir = os.path.join(REPO_ROOT, "backend", "src")
    # prompts.py/seed_examples.py/business_rules.py/orchestrator.py are now
    # packages (see the restructure) — check every file that can actually
    # carry hardcoded status/cast prose or SQL text, not just one file per
    # former module.
    _desyncs = verify_consistency([
        os.path.join(_src_dir, "agent", "prompts", "templates", "system_prompt.txt"),
        os.path.join(_src_dir, "sql", "examples", "templates", "sql_generation_prompt.txt"),
        os.path.join(_src_dir, "agent", "prompts", "templates", "force_summary_prompt.txt"),
        os.path.join(_src_dir, "agent", "prompts", "examples_tier2.py"),
        os.path.join(_src_dir, "sql", "examples", "templates", "core_context.txt"),
        os.path.join(_src_dir, "sql", "examples", "seed_data.py"),
        os.path.join(_src_dir, "sql", "rules", "block_rules.py"),
        os.path.join(_src_dir, "sql", "rules", "warn_rules.py"),
        os.path.join(_src_dir, "sql", "rules", "autocorrect_rules.py"),
        os.path.join(_src_dir, "sql", "rules", "inject_rules.py"),
        os.path.join(_src_dir, "agent", "orchestrator", "loop.py"),
        os.path.join(_src_dir, "agent", "orchestrator", "synthesis.py"),
    ])
    if _desyncs:
        log.warning("[SERVER] %d semantics desync(s) detected — see knowledge.semantics_registry:", len(_desyncs))
        for _d in _desyncs:
            log.warning("[SERVER]   %s", _d)
    else:
        log.info("[SERVER] semantics_registry consistency check passed")

    yield
    log.info("[SERVER] Shutting down")


app = FastAPI(title="InspectAI NL2SQL API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.routes.auth import auth_router
from src.api.routes.chat import router as chat_router
from src.api.routes.charts import router as charts_router
from src.api.routes.audio import router as audio_router
from src.api.routes.health import router as health_router

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(charts_router)
app.include_router(audio_router)
app.include_router(health_router)


# ── Serve Angular build (must be last — catches all non-API routes) ───────────
_DIST = os.path.join(REPO_ROOT, "frontend", "dist", "inspectai-chat", "browser")
if os.path.isdir(_DIST):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    @app.get("/", include_in_schema=False)
    async def spa_root():
        return FileResponse(os.path.join(_DIST, "index.html"))

    app.mount("/", StaticFiles(directory=_DIST, html=True), name="spa")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
