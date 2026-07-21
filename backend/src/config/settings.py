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

    # Ollama / models
    ollama_url: str
    ollama_sql_model: str
    ollama_chat_model: str
    embedding_model: str

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


def get_settings() -> Settings:
    g = os.getenv
    return Settings(
        db_user=g("DB_USER", "postgres"),
        db_password=g("DB_PASSWORD", ""),
        db_host=g("DB_HOST", "localhost"),
        db_port=g("DB_PORT", "5432"),
        db_name=g("DB_NAME", "postgres"),
        db_schema=g("DB_SCHEMA", "public"),
        ollama_url=g("OLLAMA_URL", "http://localhost:11434"),
        ollama_sql_model=g("OLLAMA_SQL_MODEL", "sqlcoder:7b"),
        ollama_chat_model=g("OLLAMA_CHAT_MODEL", "qwen2.5:14b-instruct-q4_K_M"),
        embedding_model=g("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
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
    )
