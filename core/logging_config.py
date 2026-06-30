"""
Logging configuration for InspectAI.
Call setup_logging() once at server startup (lifespan).

Sub-loggers used across the app:
  inspectai           — top-level / server lifecycle
  inspectai.auth      — login, token validation, permission fetch
  inspectai.query     — per-query lifecycle (entry, scope, exit)
  inspectai.llm       — LLM calls, tool selection, reasoning steps
  inspectai.sql       — SQL generation, validation, execution, retries

By default writes to debug_llm_YYYYMMDD_HHMMSS.log — the same file that
the orchestrator's raw LLM debug logger uses, so all logs appear in one place.
"""

import logging
import logging.handlers
from datetime import datetime

_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_LOG_BACKUP_COUNT = 5

# Generated once at import time so lifespan and orchestrator use the same filename.
_DEFAULT_LOG_FILE = f"debug_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def get_log_file() -> str:
    """Return the active log file path (useful for other modules to reuse it)."""
    root = logging.getLogger("inspectai")
    for h in root.handlers:
        if isinstance(h, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
            return h.baseFilename
    return _DEFAULT_LOG_FILE


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure the inspectai logger hierarchy with:
      - StreamHandler  → console at log_level
      - RotatingFileHandler → log_file at DEBUG (captures everything)

    Defaults to debug_llm_YYYYMMDD_HHMMSS.log so structured logs land in the
    same file that the orchestrator's raw LLM debug logger writes to.

    Silences noisy third-party loggers (httpx, ollama, transformers…).
    Safe to call multiple times — clears existing handlers first.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)
    target_file = log_file or _DEFAULT_LOG_FILE

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)

    rotating = logging.handlers.RotatingFileHandler(
        target_file,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    rotating.setFormatter(formatter)
    rotating.setLevel(logging.DEBUG)

    root = logging.getLogger("inspectai")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(rotating)

    for noisy in (
        "httpx", "httpcore", "httpcore.connection", "httpcore.http11",
        "ollama", "llama_index", "llama_index.core",
        "sentence_transformers", "transformers", "torch",
        "uvicorn.access", "sqlalchemy.engine", "sqlalchemy.pool",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
