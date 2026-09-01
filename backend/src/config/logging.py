"""
Logging configuration for InspectAI.
Call setup_logging() once at server startup (lifespan).

Sub-loggers used across the app, and the label each renders as:
  inspectai                 — top-level / server lifecycle          [SERVER]
  inspectai.auth             — login, token validation, perm fetch  [AUTH]
  inspectai.query             — per-query lifecycle (entry/exit)    [QUERY]
  inspectai.llm                 — LLM calls, tool selection         [LLM]
  inspectai.sql                   — SQL gen, validation, retries    [SQL]
  inspectai.security                — RBAC/scope decisions          [SECURITY]
                                       (ALLOWED/SCOPED/BLOCKED),
                                       tagged [SCOPE] for grepping
  inspectai.router                    — domain routing              [ROUTER]
  inspectai.prompts                     — prompt assembly           [PROMPTS]
  inspectai.turn_classifier               — turn classification     [TURN_CLS]
  inspectai.charts                          — chart advisor          [CHARTS]
  inspectai.audio                             — ASR/TTS routes       [AUDIO]

Console output is colorized by level (INFO/WARNING/ERROR/...) and by
component (AUTH/QUERY/LLM/SQL/SECURITY/...) so a scrolling terminal is
scannable at a glance — you can tell severity and subsystem apart without
reading the message. RBAC outcome words (ALLOWED/SCOPED/BLOCKED) are
highlighted inline wherever they appear, independent of log level, since
those decisions matter regardless of whether they were logged at INFO or
WARNING.

The log file stays plain text (same [LABEL] tags, no ANSI codes) since
color escapes just show up as garbage in an editor or a log viewer without
color support.

Color can be disabled with NO_COLOR=1, or forced on a non-tty stream with
FORCE_COLOR=1 (both standard env var conventions).

By default writes to debug_llm_YYYYMMDD_HHMMSS.log — the same file that
the orchestrator's raw LLM debug logger uses, so all logs appear in one place.
"""

import ctypes
import logging
import logging.handlers
import os
import sys
from datetime import datetime

_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_LOG_BACKUP_COUNT = 5

# Generated once at import time so lifespan and orchestrator use the same filename.
_DEFAULT_LOG_FILE = f"debug_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


# ---------------------------------------------------------------------------
# Component labels — short, fixed-width tags derived from the logger name so
# every line reads "who logged this" at a glance, in the console AND the file.
# ---------------------------------------------------------------------------

_LABELS = {
    "inspectai": "SERVER",
    "inspectai.auth": "AUTH",
    "inspectai.query": "QUERY",
    "inspectai.llm": "LLM",
    "inspectai.sql": "SQL",
    "inspectai.security": "SECURITY",
    "inspectai.router": "ROUTER",
    "inspectai.prompts": "PROMPTS",
    "inspectai.turn_classifier": "TURN_CLS",
    "inspectai.charts": "CHARTS",
    "inspectai.audio": "AUDIO",
}
_LABEL_WIDTH = max(len(v) for v in _LABELS.values())


def _label_for(logger_name: str) -> str:
    if logger_name in _LABELS:
        return _LABELS[logger_name]
    # Unregistered inspectai.* sub-logger (or a third-party one that slipped
    # through) — fall back to its last dotted segment, uppercased.
    return logger_name.rsplit(".", 1)[-1].upper()


# ---------------------------------------------------------------------------
# ANSI colors — basic 16-color codes for broad terminal compatibility.
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_FG = {
    "gray": "\033[90m", "red": "\033[31m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_red": "\033[91m", "bright_green": "\033[92m",
    "bright_yellow": "\033[93m", "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m", "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}

# Severity → color. Warnings and errors also tint the whole message body
# (see _ColorFormatter) so a problem is visible without reading closely.
_LEVEL_COLOR = {
    logging.DEBUG: _FG["gray"],
    logging.INFO: _FG["bright_green"],
    logging.WARNING: _FG["bright_yellow"],
    logging.ERROR: _FG["bright_red"],
    logging.CRITICAL: _BOLD + _FG["bright_red"],
}

# Subsystem → color, independent of severity, so e.g. SQL lines are always
# yellow-tagged whether they're DEBUG or ERROR.
_COMPONENT_COLOR = {
    "SERVER": _FG["bright_white"],
    "AUTH": _FG["bright_magenta"],
    "QUERY": _FG["bright_blue"],
    "LLM": _FG["bright_cyan"],
    "SQL": _FG["yellow"],
    "SECURITY": _FG["bright_red"],
    "ROUTER": _FG["blue"],
    "PROMPTS": _FG["magenta"],
    "TURN_CLS": _FG["cyan"],
    "CHARTS": _FG["bright_blue"],
    "AUDIO": _FG["bright_magenta"],
}
_DEFAULT_COMPONENT_COLOR = _FG["white"]

# RBAC outcome words get their own color wherever they appear in a message,
# regardless of log level, so a scope decision reads at a glance (matches
# the existing [SCOPE] grep convention used by inspectai.security).
_OUTCOME_COLOR = {
    "BLOCKED": _BOLD + _FG["bright_red"],
    "SCOPED": _FG["bright_yellow"],
    "ALLOWED": _FG["bright_green"],
    "UNRESTRICTED": _FG["bright_green"],
}


def _colorize(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return sys.stderr.isatty()


def _enable_windows_ansi() -> None:
    """Turn on VT100 escape processing for legacy Windows consoles.

    Windows Terminal and modern PowerShell already do this by default;
    older cmd.exe / conhost windows need it enabled explicitly via
    SetConsoleMode. No-op on any other platform, and safe to call even if
    it fails (e.g. stdout redirected to a file) — colors just won't render.
    """
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


class _PlainFormatter(logging.Formatter):
    """File formatter — same fields as the console (timestamp, level,
    component label), no ANSI codes."""

    def format(self, record: logging.LogRecord) -> str:
        label = _label_for(record.name)
        base = (
            f"{self.formatTime(record, _DATE_FMT)}.{int(record.msecs):03d} | "
            f"{record.levelname:<5} | {label:<{_LABEL_WIDTH}} | {record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


class _ColorFormatter(logging.Formatter):
    """Console formatter — colors the level tag and component label, tints
    the message body for WARNING+ severity, and highlights RBAC outcome
    words inline. Falls back to identical plain-text output automatically
    when the stream isn't a color terminal (piped output, NO_COLOR, CI)."""

    def __init__(self):
        super().__init__()
        self._color = _supports_color()

    def format(self, record: logging.LogRecord) -> str:
        label = _label_for(record.name)
        message = record.getMessage()
        timestamp = f"{self.formatTime(record, _DATE_FMT)}.{int(record.msecs):03d}"

        if not self._color:
            base = f"{timestamp} | {record.levelname:<5} | {label:<{_LABEL_WIDTH}} | {message}"
            if record.exc_info:
                base += "\n" + self.formatException(record.exc_info)
            return base

        level_color = _LEVEL_COLOR.get(record.levelno, _FG["white"])
        comp_color = _COMPONENT_COLOR.get(label, _DEFAULT_COMPONENT_COLOR)
        msg_color = level_color if record.levelno >= logging.WARNING else ""

        for outcome, color in _OUTCOME_COLOR.items():
            if outcome in message:
                resume = msg_color or ""
                message = message.replace(outcome, f"{color}{outcome}{_RESET}{resume}")

        colored_message = f"{msg_color}{message}{_RESET}" if msg_color else message
        base = (
            f"{_colorize(timestamp, _DIM)} | "
            f"{_colorize(f'{record.levelname:<5}', level_color)} | "
            f"{_colorize(f'{label:<{_LABEL_WIDTH}}', comp_color)} | "
            f"{colored_message}"
        )
        if record.exc_info:
            base += "\n" + _colorize(self.formatException(record.exc_info), _FG["bright_red"])
        return base


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
      - StreamHandler       → console at log_level, colorized
      - RotatingFileHandler → log_file at DEBUG (captures everything), plain text

    Defaults to debug_llm_YYYYMMDD_HHMMSS.log so structured logs land in the
    same file that the orchestrator's raw LLM debug logger writes to.

    Silences noisy third-party loggers (httpx, ollama, transformers…).
    Safe to call multiple times — clears existing handlers first.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    target_file = log_file or _DEFAULT_LOG_FILE

    _enable_windows_ansi()

    console = logging.StreamHandler()
    console.setFormatter(_ColorFormatter())
    console.setLevel(level)

    rotating = logging.handlers.RotatingFileHandler(
        target_file,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    rotating.setFormatter(_PlainFormatter())
    rotating.setLevel(logging.DEBUG)

    root = logging.getLogger("inspectai")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(rotating)

    logging.getLogger("inspectai.security").handlers.clear()

    for noisy in (
        "httpx", "httpcore", "httpcore.connection", "httpcore.http11",
        "ollama", "llama_index", "llama_index.core",
        "sentence_transformers", "transformers", "torch",
        "uvicorn.access", "sqlalchemy.engine", "sqlalchemy.pool",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
