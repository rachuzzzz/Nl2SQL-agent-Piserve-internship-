"""
Centralized environment-variable configuration.

Declares every environment variable this backend reads and its default, in
one place, instead of scattered `os.getenv(...)` calls across api/app.py,
api/auth_routes.py, charts/advisor.py, retrieval/, and
agent/orchestrator.py.

This module does NOT call load_dotenv() itself. The existing load_dotenv()
call sites (api/app.py's lifespan, agent/orchestrator.py's from_env,
backend/eval/eval.py) are left as-is — get_settings() just reads whatever
is in os.environ at call time, so it works whichever of those already ran.
get_settings() is intentionally uncached (each call re-reads os.environ)
rather than memoized, so it can never return stale values depending on
which load_dotenv() call site happens to run first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Database
    db_user: str
    db_password: str
    db_host: str
    db_port: str
    db_name: str
    db_schema: str
    # SQLAlchemy connection-pool resilience (Phase 5 hardening). pool_pre_ping
    # is always on in create_engine; these tune size / staleness. pool_recycle
    # in seconds — recycle a connection older than this before handing it out,
    # so an idle-timed-out or server-restarted connection never reaches a query.
    db_pool_size: int
    db_max_overflow: int
    db_pool_recycle: int

    # ── API service hardening (Phase 5) — consumed in api/routes/chat.py and
    # api/deps.py. All have safe defaults; override via env for load tuning.
    # request_timeout_s: hard deadline on one /api/chat turn (the whole
    #   reasoning+SQL pipeline). On expiry the request returns 504 and the
    #   event loop is freed (the worker thread finishes in the background —
    #   Python can't force-kill it — but the semaphore below bounds how many
    #   can pile up).
    # max_concurrent_pipelines: how many /api/chat turns may run the LLM
    #   pipeline at once (the llama-servers have finite slots; unbounded
    #   fan-out just makes every request slow). Excess requests wait up to
    #   sem_acquire_timeout_s then get 503.
    # session_ttl_s / session_max: eviction for the in-memory _sessions store.
    request_timeout_s: float
    max_concurrent_pipelines: int
    sem_acquire_timeout_s: float
    session_ttl_s: float
    session_max: int

    # Ollama / models
    ollama_url: str
    ollama_sql_model: str
    ollama_chat_model: str
    embedding_model: str

    # How long Ollama keeps a model (and its cached KV slots) resident after
    # the last request — see llm/ollama.py's build_ollama_client keep_alive
    # docstring. Server default is 5m, too short to survive normal gaps
    # between user questions without losing prefix-cache benefit.
    ollama_keep_alive: str

    # Anthropic (Claude) — optional alternative LLM provider, see llm/anthropic.py
    anthropic_api_key: str
    anthropic_model: str

    # vLLM — optional alternative LLM provider, see llm/vllm.py. vLLM's server
    # exposes an OpenAI-compatible REST API at vllm_url (e.g. http://host:8000/v1);
    # vllm_api_key is typically a dummy value unless the server enforces auth.
    vllm_url: str
    vllm_reasoning_model: str
    vllm_sql_model: str
    vllm_api_key: str

    # llama.cpp — optional alternative LLM provider, see llm/factory.py's
    # get_reasoning_llm_llamacpp/get_sql_llm_llamacpp. Unlike vLLM/Ollama,
    # reasoning and sql are two separate llama-server processes/ports (one
    # model per process is a llama-server constraint), hence two URLs
    # instead of one shared vllm_url/ollama_url.
    llamacpp_reasoning_url: str
    llamacpp_reasoning_model: str
    llamacpp_sql_url: str
    llamacpp_sql_model: str
    llamacpp_api_key: str

    # Which provider AgentOrchestrator.from_env() builds reasoning_llm/sql_llm
    # from, independently per role — each is "ollama" (default), "anthropic",
    # or "vllm". Replaces the old single llm_provider switch that flipped both
    # roles together; e.g. reasoning_llm_provider="vllm" + sql_llm_provider=
    # "ollama" runs the reasoning model on vLLM and the SQL model on Ollama
    # at the same time.
    reasoning_llm_provider: str
    sql_llm_provider: str

    # Debug
    debug_llm: bool

    # OAuth / permissions (api/auth_routes.py)
    oauth_url: str
    client_id: str
    client_secret: str
    oauth_ca_cert: str
    oauth_skip_tls_verify: bool
    permissions_api_url: str

    # Semantic reranker
    reranker_model: str
    reranker_device: str

    # Charts — safety cap for the fuller re-fetch charts use instead of the
    # 50-row execute_sql cache (see charts/data_fetch.py's fetch_chart_data)
    chart_fetch_cap: int


def get_settings() -> Settings:
    g = os.getenv
    return Settings(
        db_user=g("DB_USER", "postgres"),
        db_password=g("DB_PASSWORD", ""),
        db_host=g("DB_HOST", "localhost"),
        db_port=g("DB_PORT", "5432"),
        db_name=g("DB_NAME", "postgres"),
        db_schema=g("DB_SCHEMA", "public"),
        db_pool_size=int(g("DB_POOL_SIZE", "10")),
        db_max_overflow=int(g("DB_MAX_OVERFLOW", "5")),
        db_pool_recycle=int(g("DB_POOL_RECYCLE", "1800")),
        request_timeout_s=float(g("REQUEST_TIMEOUT_S", "300")),
        max_concurrent_pipelines=int(g("MAX_CONCURRENT_PIPELINES", "4")),
        sem_acquire_timeout_s=float(g("SEM_ACQUIRE_TIMEOUT_S", "5")),
        session_ttl_s=float(g("SESSION_TTL_S", "86400")),
        session_max=int(g("SESSION_MAX", "5000")),
        ollama_url=g("OLLAMA_URL", "http://localhost:11434"),
        ollama_sql_model=g("OLLAMA_SQL_MODEL", "sqlcoder:7b"),
        ollama_chat_model=g("OLLAMA_CHAT_MODEL", "qwen2.5:14b-instruct-q4_K_M"),
        ollama_keep_alive=g("OLLAMA_KEEP_ALIVE", "30m"),
        embedding_model=g("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        anthropic_api_key=g("ANTHROPIC_API_KEY", "").strip(),
        # Same model for both the reasoning and sql roles (llm/factory.py's
        # get_reasoning_llm_anthropic/get_sql_llm_anthropic) — user's explicit
        # choice; Claude has no equivalent to Ollama's separate small
        # SQL-specialized model, so one model does both jobs.
        anthropic_model=g("ANTHROPIC_MODEL", "claude-sonnet-5"),
        vllm_url=g("VLLM_URL", "http://localhost:8000/v1"),
        vllm_reasoning_model=g("VLLM_REASONING_MODEL", ""),
        vllm_sql_model=g("VLLM_SQL_MODEL", ""),
        vllm_api_key=g("VLLM_API_KEY", "dummy").strip(),
        llamacpp_reasoning_url=g("LLAMACPP_REASONING_URL", "http://localhost:10000/v1"),
        llamacpp_reasoning_model=g("LLAMACPP_REASONING_MODEL", ""),
        llamacpp_sql_url=g("LLAMACPP_SQL_URL", "http://localhost:10001/v1"),
        llamacpp_sql_model=g("LLAMACPP_SQL_MODEL", ""),
        llamacpp_api_key=g("LLAMACPP_API_KEY", "dummy").strip(),
        reasoning_llm_provider=g("REASONING_LLM_PROVIDER", "ollama").strip().lower(),
        sql_llm_provider=g("SQL_LLM_PROVIDER", "ollama").strip().lower(),
        # NOTE: no .lower() here — matches the original orchestrator.py check
        # exactly, which only recognizes lowercase "true"/"yes" (or "1").
        debug_llm=g("DEBUG_LLM", "").strip() in ("1", "true", "yes"),
        oauth_url=g("OAUTH_URL", "https://10.77.20.51/oauth/token?_format=json"),
        client_id=g("CLIENT_ID", "").strip(),
        client_secret=g("CLIENT_SECRET", "").strip(),
        oauth_ca_cert=g("OAUTH_CA_CERT", "").strip(),
        oauth_skip_tls_verify=g("OAUTH_SKIP_TLS_VERIFY", "").strip().lower() in ("1", "true", "yes"),
        permissions_api_url=g("PERMISSIONS_API_URL", "http://10.77.20.54:8083"),
        reranker_model=g("RERANKER_MODEL", "").strip(),
        reranker_device=g("RERANKER_DEVICE", "cpu"),
        chart_fetch_cap=int(g("CHART_FETCH_CAP", "2000")),
    )
