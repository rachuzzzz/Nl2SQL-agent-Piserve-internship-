"""
InspectAI — FastAPI Backend
===========================
Place this file in the project root (same folder as run.py).

Install:  pip install fastapi uvicorn
Run:      uvicorn fastapi_app:app --reload --port 8000
"""

import asyncio
import time
import time as _time
import os
import tempfile
import subprocess
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import re
import uuid
import logging
from contextlib import asynccontextmanager

from khidmah.core.logging_config import setup_logging
from khidmah.core.rbac import DATA_VISIBILITY_PERMS, build_scope_filter
from khidmah.core.disambiguation import is_disambiguation_result, is_entity_disambiguation_result, extract_search_term
from khidmah.api.deps import get_orch, get_orch_optional

# ── TTS — lazy-loaded on first /api/tts call ──────────────────────────────────
_tts_model = None

# __file__ is backend/khidmah/api/app.py — repo root is 4 levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_TTS_DIR    = os.path.join(_REPO_ROOT, "models")
_TTS_MODEL  = os.path.join(_TTS_DIR, "kokoro-v1.0.onnx")
_TTS_VOICES = os.path.join(_TTS_DIR, "voices-v1.0.bin")

def _get_tts():
    global _tts_model
    if _tts_model is None:
        from kokoro_onnx import Kokoro
        logging.info("Loading Kokoro TTS …")
        _tts_model = Kokoro(_TTS_MODEL, _TTS_VOICES)
        logging.info("Kokoro TTS ready.")
    return _tts_model

_MD_STRIP = re.compile(
    r'\*\*(.+?)\*\*'       # **bold**
    r'|\*(.+?)\*'          # *italic*
    r'|`[^`]*`'            # `code`
    r'|#{1,6}\s+'          # ## headings
    r'|\|[^\n]*'           # | table cells
    r'|^[-|=]+$'           # table separator rows
    r'|\[([^\]]*)\]\([^)]*\)',  # [link](url) → keep label
    re.MULTILINE,
)

def _clean_for_tts(text: str) -> str:
    text = _MD_STRIP.sub(lambda m: m.group(1) or m.group(2) or m.group(3) or '', text)
    # Expand common patterns that confuse espeak word-count alignment
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    text = re.sub(r'\$(\d[\d,]*)', r'\1 dollars', text)
    text = re.sub(r'(\d+)/(\d+)', r'\1 out of \2', text)
    text = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1 \2 \3', text)  # ISO dates
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:800]

# ── ASR — lazy-loaded on first /api/transcribe call ───────────────────────────
_asr_pipeline = None

def _get_asr():
    global _asr_pipeline
    if _asr_pipeline is None:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        logging.info("Loading openai/whisper-tiny on %s …", "GPU" if device == 0 else "CPU")
        _asr_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=device,
            generate_kwargs={"language": "en", "task": "transcribe"},
        )
        logging.info("ASR model ready.")
    return _asr_pipeline

from khidmah.charts.advisor import advise, force_config, get_alternatives, classify_columns, build_kpi_cards, should_show_dashboard

# ── Chart follow-up fast-path ──────────────────────────────────────────────────
# Detects pure chart requests that need no new SQL — handled before the LLM.

_VISUAL_KW = re.compile(
    r'\b(chart|graph|plot|visual(i[sz]e|ization)?|bar|pie|line|doughnut|donut|scatter|draw)\b'
    r'|\bshow\s+as\s+chart\b|\bas\s+(a\s+)?chart\b',
    re.IGNORECASE,
)
_DOMAIN_KW = re.compile(
    r'\b(inspection|facility|capex|opex|score|mitigative|action|overdue|report|month|'
    r'year|week|average|total|count|inspector|manager|status|pending|approved|rejected|'
    r'deferred|finding|observation|po|purchase|order|amount|budget|responsible|closed|open)\b',
    re.IGNORECASE,
)
_TYPE_RE = re.compile(
    r'\b(horizontal\s+bar|bar|line|trend|pie|doughnut|donut|scatter)\b',
    re.IGNORECASE,
)
_TYPE_MAP = {
    'bar': 'bar', 'horizontal bar': 'bar', 'line': 'line', 'trend': 'line',
    'pie': 'pie', 'doughnut': 'doughnut', 'donut': 'doughnut', 'scatter': 'scatter',
}

def _is_chart_followup(q: str) -> bool:
    """True when message is purely a chart request with no new data intent."""
    return (bool(_VISUAL_KW.search(q))
            and not bool(_DOMAIN_KW.search(q))
            and len(q.split()) <= 20)

def _extract_requested_type(q: str) -> str | None:
    m = _TYPE_RE.search(q)
    return _TYPE_MAP.get(m.group(1).lower().strip()) if m else None

log       = logging.getLogger("inspectai")
log_auth  = logging.getLogger("inspectai.auth")
log_query = logging.getLogger("inspectai.query")
log_security = logging.getLogger("inspectai.security")

# ── Eager startup ──────────────────────────────────────────────────────────────

_sessions: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the orchestrator at server start so the first request is instant."""
    app.state.orch = None
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging(log_level="INFO")
    log.info("[SERVER] Starting InspectAI — loading orchestrator (LLM + embeddings + schema)…")
    t0 = time.time()
    from khidmah.agent.orchestrator import AgentOrchestrator
    app.state.orch = AgentOrchestrator.from_env()
    log.info("[SERVER] Orchestrator ready in %.1fs", time.time() - t0)

    # Catch definition desyncs between core/semantics_registry.py (the declared
    # single source of truth) and the prompt/example files that actually feed
    # the SQL-generating LLM. This previously existed as a CLI-only tool nobody
    # ran automatically — the "overdue" bug (fixed 2026-07-07) sat undetected
    # for an unknown period because of exactly that gap.
    from khidmah.core.semantics_registry import verify_consistency
    _khidmah_dir = os.path.join(_REPO_ROOT, "backend", "khidmah")
    _desyncs = verify_consistency([
        os.path.join(_khidmah_dir, "agent", "prompts.py"),
        os.path.join(_khidmah_dir, "agent", "seed_examples.py"),
        os.path.join(_khidmah_dir, "core", "business_rules.py"),
        os.path.join(_khidmah_dir, "agent", "orchestrator.py"),
    ])
    if _desyncs:
        log.warning("[SERVER] %d semantics desync(s) detected — see core.semantics_registry:", len(_desyncs))
        for _d in _desyncs:
            log.warning("[SERVER]   %s", _d)
    else:
        log.info("[SERVER] semantics_registry consistency check passed")

    yield
    log.info("[SERVER] Shutting down")


# ── Pydantic models ────────────────────────────────────────────────────────────

app = FastAPI(title="InspectAI NL2SQL API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from khidmah.core.auth import require_auth
from khidmah.api.auth_routes import auth_router

app.include_router(auth_router)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class PageRequest(BaseModel):
    session_id: str
    page:       int           # 0-based page index
    page_size:  int = Field(default=50, ge=1, le=500)

class ChartRequest(BaseModel):
    session_id: str
    chart_type: str | None = None   # None → auto-advise; string → force that type
    x_col:      str | None = None   # override label column (column picker)
    y_cols:     list[str] | None = None  # override value columns (column picker)

class StepInfo(BaseModel):
    tool: str
    thought: str
    sql: str | None = None
    result_preview: str | None = None
    success: bool = True

class DisambiguationMeta(BaseModel):
    kind: str          # "person" | "facility" | "project" | "client" | "organisation"
    search_term: str   # literal text from the STEP 1 ILIKE clause, e.g. "adnoc"

class ResultData(BaseModel):
    rows: list[dict]
    columns: list[str]
    row_count: int
    total_count: int | None = None
    page: int = 0
    page_size: int = 50
    has_more: bool = False
    sql: str | None = None   # base SQL — sent back with every /api/page call
    disambiguation: DisambiguationMeta | None = None

class ChartConfig(BaseModel):
    chart_type: str | None
    reason: str
    confidence: str
    config: dict | None = None
    conflict: bool = False
    requested_type: str | None = None
    requested_config: dict | None = None
    alternatives: list[dict] = []
    x_label: str | None = None      # category / x-axis column name
    y_labels: list[str] = []        # value column name(s)

class DashboardData(BaseModel):
    show: bool = False
    cards: list[dict] = []   # [{label, value, kind, icon}]

class ChatResponse(BaseModel):
    answer: str
    sql: str | None
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_ms: int
    llm_calls: int
    sql_errors: int
    steps: list[StepInfo]
    session_id: str
    result_data: ResultData | None = None
    chart_config: ChartConfig | None = None
    dashboard: DashboardData | None = None


# ── Session management ────────────────────────────────────────────────────────

def get_or_create_session(session_id: str) -> dict:
    from khidmah.agent.orchestrator import ConversationSession
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


# ── UUID column scrubber ──────────────────────────────────────────────────────

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE
)


def _strip_uuid_columns(columns: list, rows: list) -> tuple[list, list]:
    """
    Drop any column whose values are raw UUIDs before the result table reaches
    the frontend. UUIDs (user_id, facility_id, etc.) are needed server-side to
    resolve/join entities but are meaningless — and look like a bug — to a
    person reading the chat table, so they're scrubbed here regardless of
    which tool/path produced the rows.
    """
    if not columns or not rows:
        return columns, rows
    uuid_cols = {
        col for col in columns
        if any(_UUID_RE.match(str(row.get(col, "")).strip()) for row in rows if row.get(col))
    }
    if not uuid_cols:
        return columns, rows
    kept_cols = [c for c in columns if c not in uuid_cols]
    kept_rows = [{k: v for k, v in row.items() if k not in uuid_cols} for row in rows]
    return kept_cols, kept_rows


_SCOPE_PREDICATE_RE = re.compile(
    r"ir\.(?:inspector|inspectee)_user_id\s*=\s*\(\s*SELECT\s+id\s+FROM\s+users\s+WHERE\s+id_fk\s*=\s*\d+[^)]*\)",
    re.IGNORECASE,
)
_ID_FK_RE = re.compile(r"id_fk\s*=\s*\d+", re.IGNORECASE)
# No trailing \)? — consuming an optional closing paren here ate it on redact,
# leaving unbalanced markdown like "(Drupal UID [redacted]" in the response.
_DRUPAL_UID_RE = re.compile(r"Drupal\s+UID[\s:(]*\d+", re.IGNORECASE)
# Catches the model explaining the scope mechanism in prose instead of the
# literal SQL shape _SCOPE_PREDICATE_RE expects — e.g. "Inspectors see
# inspections they conduct (via inspector_user_id)" or "WHERE ur.user_id =
# 11223" — observed live in production logs from an ordinary, non-adversarial
# question ("how do you determine what I'm allowed to see"), which the SQL-
# shaped regex above does not match at all.
_FK_COLUMN_MENTION_RE = re.compile(
    r"`?\b(?:inspector_user_id|inspectee_user_id)\b`?", re.IGNORECASE
)
_USER_ID_VALUE_RE = re.compile(r"\buser_id\s*=\s*\d+", re.IGNORECASE)


def _redact_security_internals(text: str | None) -> str | None:
    """Deterministic backstop for the system prompt's "never expose this
    constraint" instruction, which the model doesn't reliably honor when
    directly asked about its filters/mechanism. Applied to LLM-authored
    text right before it reaches the client — enforce_scope() itself
    doesn't care what the LLM says, so this is purely about not handing an
    attacker the enforcement mechanism to study.
    """
    if not text:
        return text
    text = _SCOPE_PREDICATE_RE.sub("[access-control filter]", text)
    text = _ID_FK_RE.sub("id_fk = [redacted]", text)
    text = _DRUPAL_UID_RE.sub("[redacted]", text)
    text = _FK_COLUMN_MENTION_RE.sub("[access-control column]", text)
    text = _USER_ID_VALUE_RE.sub("user_id = [redacted]", text)
    return text


# ── Answer cleaner ────────────────────────────────────────────────────────────

def _clean_answer(answer: str, result_rows: list) -> str:
    """
    Keep the LLM's natural-language INSIGHT but strip raw data lists/tables
    since the UI renders the data table separately.

    Strips:
    - Markdown tables (| col | col |)
    - Numbered data rows: "1. facility_name: X | total_capex: Y"
    - Bullet data rows: "- X: 123.00" or "• X: 123"
    - "Found N result(s):" / "Here are the results:" preamble lines
    - "... and N more shown" trailing lines
    - Excessive blank lines

    Keeps: natural language sentences, insights, summaries.
    """
    import re as _re

    if not answer:
        return answer

    lines = answer.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines at this stage (re-add single blank later)
        if not stripped:
            cleaned.append("")
            continue

        # Skip markdown table rows and separators
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        if _re.match(r"^[\|\-\s:]+$", stripped) and "|" in stripped:
            continue

        # Skip numbered data rows: "1. col: val | col: val" or "  1. col: val"
        if _re.match(r"^\d+\.\s+\w.*\|", stripped):
            continue
        # Also catch "1. facility_name: X" without pipes
        if _re.match(r"^\d+\.\s+\w+[\w_]+:\s+", stripped):
            continue

        # Skip bullet data rows: "- X: 123" or "• X: 123" or "* X: 123"
        if _re.match(r"^[-•*]\s+\S.*:\s+[\d,\.]+\s*$", stripped):
            continue
        # Skip bullet rows with pipes: "• facility: X | capex: Y"
        if _re.match(r"^[-•*]\s+\S.*\|", stripped):
            continue

        # Skip "Found N result(s)..." lines — count is shown in table meta
        if _re.match(r"^found\s+\d+\s+result", stripped, _re.IGNORECASE):
            continue
        if _re.match(r"^results?\s*\(\d+", stripped, _re.IGNORECASE):
            continue
        if _re.match(r"^showing\s+(first\s+)?\d+", stripped, _re.IGNORECASE):
            continue

        # Skip "... and N more shown" trailing lines
        if _re.match(r"^\.\.\.\s+and\s+\d+\s+more", stripped, _re.IGNORECASE):
            continue

        # Skip "Here are the results:" / "Here is the data:" preambles
        if _re.match(r"^here (are|is) the (results?|data|following|capex|list)", stripped, _re.IGNORECASE):
            continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # Collapse 3+ consecutive blank lines to 1
    result = _re.sub(r"\n{3,}", "\n\n", result)

    # If after cleaning we have nothing meaningful, fall back gracefully
    if not result or len(result.strip()) < 15:
        n = len(result_rows)
        return f"Found {n} result{'s' if n != 1 else ''}."

    return result.strip()


# ── Summary generator ────────────────────────────────────────────────────────

def _generate_summary(question: str, columns: list, rows: list) -> str:
    """Pure Python insight summary — zero LLM calls."""
    if not rows or not columns or len(rows) < 2:
        return ""
    try:
        def _coerce(v):
            try:
                return float(str(v).replace(",", "").strip())
            except Exception:
                return None

        def fmt(v):
            if v >= 1_000_000: return f"{v/1_000_000:.2f}M"
            if v >= 1_000: return f"{v:,.0f}"
            return f"{v:.2f}"

        def _fmt_label(v) -> str:
            """Return a human-readable label, formatting datetimes nicely."""
            import re
            from datetime import datetime, date

            def _fmt_dt(dt: datetime) -> str:
                # day==1 at midnight → DATE_TRUNC('month', ...) pattern → show month only
                if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.day == 1:
                    return dt.strftime("%b %Y")
                if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                    return dt.strftime("%b %d, %Y").replace(" 0", " ")
                return dt.strftime("%b %d, %Y %H:%M").replace(" 0", " ")

            s = str(v).strip()
            # Already a date/datetime object
            if isinstance(v, datetime):
                return _fmt_dt(v)
            if isinstance(v, date):
                if v.day == 1:
                    return v.strftime("%b %Y")
                return v.strftime("%b %d, %Y").replace(" 0", " ")
            # ISO timestamp strings: 2026-01-01 00:00:00+00:00 or 2026-01-01T00:00:00Z
            if re.match(r'^\d{4}-\d{2}-\d{2}[ T]', s):
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                    return _fmt_dt(dt)
                except Exception:
                    pass
            # Plain date string: 2026-01-01
            if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
                try:
                    dt = datetime.strptime(s, "%Y-%m-%d")
                    return _fmt_dt(dt)
                except Exception:
                    pass
            return s

        num_cols, cat_cols = [], []
        for col in columns:
            sample = [rows[i].get(col) for i in range(min(5, len(rows))) if rows[i].get(col) is not None]
            if sample and all(_coerce(v) is not None for v in sample):
                num_cols.append(col)
            else:
                cat_cols.append(col)

        n = len(rows)
        parts = []
        for nc in num_cols[:2]:
            vals = [(row, _coerce(row.get(nc))) for row in rows if _coerce(row.get(nc)) is not None]
            if not vals:
                continue
            vals.sort(key=lambda x: x[1], reverse=True)
            top_row, top_val = vals[0]
            total = sum(v for _, v in vals)
            label = None
            for cc in cat_cols:
                lv = top_row.get(cc)
                if lv and str(lv).strip() not in ("None", ""):
                    label = _fmt_label(lv)
                    break
            col_label = nc.replace("_", " ").title()
            if label:
                parts.append(f"{label} leads with {fmt(top_val)} in {col_label}.")
                if n > 2 and abs(total - top_val) > 0.01:
                    entity = "facilities" if any("facilit" in c for c in cat_cols) else "records"
                    parts.append(f"Total across {n} {entity} is {fmt(total)}.")
            else:
                parts.append(f"Highest {col_label} is {fmt(top_val)} across {n} records.")

        if not parts and cat_cols:
            parts.append(f"{n} records returned.")

        return " ".join(parts)
    except Exception:
        return ""


# ── Chart suggestion sentence ─────────────────────────────────────────────────

_TYPE_LABELS = {
    'bar': 'Bar Chart', 'line': 'Line Chart', 'pie': 'Pie Chart',
    'doughnut': 'Doughnut Chart', 'scatter': 'Scatter Plot',
}
_ALT_MAP = {
    'bar':      ['line', 'pie'],
    'line':     ['bar', 'scatter'],
    'pie':      ['doughnut', 'bar'],
    'doughnut': ['pie', 'bar'],
    'scatter':  ['line', 'bar'],
}

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, token_data: dict = Depends(require_auth), orch=Depends(get_orch)):
    qid = uuid.uuid4().hex[:8]
    sub = token_data['sub']
    try:
        sess_key = f"{req.session_id}:{sub}"
        sess_data = get_or_create_session(sess_key)

        log_query.info("[QUERY] START qid=%s sub=%s session=%s q=%r",
                       qid, sub, req.session_id, req.message[:120])

        # Build visibility scope. None means BLOCK — fail closed.
        # An empty permission list (permissions not yet fetched, or Java API down)
        # resolves to None → BLOCK. Only a confirmed non-empty list from the
        # Java API that contains at least one DATA_VISIBILITY_PERM grants access.
        _user_perms = token_data['permissions']
        if _user_perms:
            _data_vis = [p for p in _user_perms if p in DATA_VISIBILITY_PERMS]
            visibility_scope: dict | None = {
                'sub': sub,
                'permissions': _data_vis,
            }
            log_security.info("[SCOPE] qid=%s sub=%s data_visibility_perms=%s",
                               qid, sub, _data_vis or "(none — will BLOCK)")
        else:
            visibility_scope = None  # → BLOCK in orchestrator
            log_security.warning("[SCOPE] qid=%s sub=%s — no permissions fetched, BLOCKING", qid, sub)

        # ── Chart follow-up fast-path — skip LLM entirely ────────────────────
        if _is_chart_followup(req.message) and sess_data.get("last_raw_rows"):
            cols = sess_data["last_cols"]
            rows = sess_data["last_raw_rows"]
            requested = _extract_requested_type(req.message)

            # Calendar heatmap needs the full date range, not just the cached 50 rows.
            _cal_needed = (requested == "calendar_heatmap") or (not requested and advise(cols, rows).get("chart_type") == "calendar_heatmap")
            if _cal_needed and sess_data.get("last_sql"):
                _full = _refetch_for_calendar(sess_data["last_sql"], orch, params=sess_data.get("last_sql_params", {}))
                if _full:
                    rows = _full
                    cols = list(_full[0].keys())

            rec = advise(cols, rows)
            if rec.get("chart_type"):
                if requested:
                    # User explicitly typed /chart <type> — show it directly,
                    # no conflict banner. The advisor config is the fallback only
                    # if force_config returns nothing (e.g. unsupported combination).
                    req_cfg = force_config(cols, rows, requested) or rec["config"]
                    cc = ChartConfig(
                        chart_type=requested,
                        reason=f"Showing {requested.replace('_', ' ').title()} as requested",
                        confidence="medium",
                        config=req_cfg,
                        conflict=False,
                        requested_type=None,
                        requested_config=None,
                        alternatives=rec.get("alternatives", []),
                        x_label=rec.get("labels_column"),
                        y_labels=rec.get("values_columns") or [],
                    )
                else:
                    cc = ChartConfig(
                        chart_type=rec["chart_type"], reason=rec["reason"],
                        confidence=rec["confidence"], config=rec["config"],
                        conflict=False,
                        requested_type=None,
                        requested_config=None,
                        alternatives=rec.get("alternatives", []),
                        x_label=rec.get("labels_column"),
                        y_labels=rec.get("values_columns") or [],
                    )
                return ChatResponse(
                    answer="Here's the chart.",
                    sql=None, reasoning=None,
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    time_ms=1, llm_calls=0, sql_errors=0, steps=[],
                    session_id=req.session_id,
                    result_data=None,
                    chart_config=cc,
                )

        t0 = time.time()
        result = await asyncio.to_thread(
            orch.query, req.message, session=sess_data["session"],
            visibility_scope=visibility_scope, query_id=qid,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        sess_data["message_count"] += 1

        sql       = None
        reasoning = None
        steps: list[StepInfo] = []
        sql_errors = 0
        _result_rows: list = []
        _result_cols: list = []
        _row_count: int = 0
        _chart_config: ChartConfig | None = None
        _dashboard: DashboardData | None = None
        _skip_dashboard: bool = False  # set True for non-SQL tool results (get_answers, get_answer_stats)

        for step in result.steps:
            thought = getattr(step, "thought", "") or ""
            info = StepInfo(tool=step.tool, thought=thought)

            if step.tool == "generate_sql":
                generated = (step.result or {}).get("result", {}).get("sql", "")
                if generated:
                    sql = generated
                info.sql = generated
                info.success = (step.result or {}).get("success", True)

            elif step.tool == "execute_sql":
                info.success = (step.result or {}).get("success", True)
                if not info.success:
                    sql_errors += 1
                inner = (step.result or {}).get("result") or {}
                rows = inner.get("rows", [])
                info.result_preview = str(rows[:3])[:200] if rows else "No rows"
                if rows and isinstance(rows[0], dict):
                    sess_data["last_raw_rows"] = rows       # cache for fast-path
                    sess_data["last_cols"] = list(rows[0].keys())
                    _result_rows = [{k: str(v) for k, v in row.items()} for row in rows]
                    _result_cols = list(rows[0].keys())
                    _row_count = inner.get("row_count", len(rows))

                    # Cache SQL for pagination — prefer this execute_sql step's own
                    # args, which is the ACTUALLY EXECUTED SQL (agent/orchestrator.py
                    # rewrites it via enforce_scope() before execution). Falling back
                    # to `sql` (the raw, pre-scope-enforcement generate_sql result)
                    # would cache unscoped SQL, which /api/page and the calendar
                    # refetch then re-run with no scope re-application — a bypass for
                    # a scoped user paging through results or opening a calendar chart.
                    exec_sql = (step.args or {}).get("sql") or sql
                    if exec_sql:
                        sess_data["last_sql"]        = exec_sql
                        sess_data["last_sql_params"] = {}
                        sess_data["last_total"]      = inner.get("total_count", _row_count)

            elif step.tool == "get_answers":
                # get_answers returns same shape as execute_sql — extract rows for frontend
                info.success = (step.result or {}).get("success", True)
                inner = (step.result or {}).get("result") or {}
                rows = inner.get("rows", [])
                info.result_preview = str(rows[:3])[:200] if rows else "No rows"
                if rows and isinstance(rows[0], dict):
                    sess_data["last_raw_rows"] = rows
                    sess_data["last_cols"] = inner.get("columns", list(rows[0].keys()))
                    _result_rows = [{k: str(v) for k, v in row.items()} for row in rows]
                    _result_cols = inner.get("columns", list(rows[0].keys()))
                    _row_count = inner.get("row_count", len(rows))
                    # Rebuild the tool's query as a parameterised SQL template so
                    # /api/page can paginate safely. All LLM-supplied values go through
                    # :param placeholders — never f-string-interpolated.
                    # The scope predicate is inlined directly (not via enforce_scope) to
                    # preserve the :param placeholders through the string concat path.
                    _ga_args    = step.args or {}
                    _ga_params: dict = {}
                    _ga_clauses = ["1=1"]
                    _ga_label   = _ga_args.get("question_label") or _ga_args.get("label")
                    if _ga_args.get("form_name"):
                        _ga_clauses.append("aq.module_name ILIKE :ga_form_name")
                        _ga_params["ga_form_name"] = f"%{_ga_args['form_name']}%"
                    if _ga_args.get("module_name"):
                        _ga_clauses.append("aa.module_name ILIKE :ga_module_name")
                        _ga_params["ga_module_name"] = f"%{_ga_args['module_name']}%"
                    if _ga_label:
                        _ga_clauses.append("aq.label ILIKE :ga_label")
                        _ga_params["ga_label"] = f"%{_ga_label}%"
                    if _ga_args.get("inspection_id"):
                        _ga_clauses.append(
                            "aa.inspection_report_id = (SELECT ir_s.id"
                            " FROM inspection_report ir_s"
                            " WHERE ir_s.inspection_id = :ga_insp_id LIMIT 1)"
                        )
                        _ga_params["ga_insp_id"] = _ga_args["inspection_id"]
                        if not _ga_args.get("module_name") and not _ga_args.get("form_name"):
                            _ga_clauses.append("aa.module_name = 'Inspection Form'")
                    if _ga_args.get("answer_text"):
                        _ga_clauses.append("aa.answer_text ILIKE :ga_ans_text")
                        _ga_params["ga_ans_text"] = f"%{_ga_args['answer_text']}%"
                    # Inline the same scope predicate the tool applied at query time.
                    # Uses build_scope_filter (server-trusted SQL, int(sub) cast).
                    # Added to clauses before SQL string is built to preserve :params.
                    _ga_scope = build_scope_filter(visibility_scope) if visibility_scope else 'BLOCK'
                    if _ga_scope and _ga_scope != 'BLOCK':
                        _ga_clauses.append(_ga_scope)
                    _ga_where = " AND ".join(_ga_clauses)
                    # Match agent/tools.py's get_answers(): INNER JOIN when scope is
                    # active so the RBAC predicate can't be bypassed by NULL ir rows
                    # (a LEFT JOIN with a WHERE-side scope predicate still behaves
                    # like an inner join for matched rows, but INNER JOIN makes the
                    # intent explicit and avoids relying on that NULL-propagation
                    # behavior — the two code paths for the same logical query must
                    # share this enforcement, not diverge).
                    _ga_ir_join = (
                        "INNER JOIN inspection_report ir ON aa.inspection_report_id = ir.id"
                        if _ga_scope and _ga_scope != 'BLOCK'
                        else "LEFT JOIN inspection_report ir ON aa.inspection_report_id = ir.id"
                    )
                    _ga_sql = (
                        "SELECT ir.inspection_id, aa.module_name,"
                        " aq.label AS question_label,"
                        " COALESCE(aa.answer_text, aa.answer_numeric::text) AS answer_value,"
                        " aa.score, aa.submitted_on"
                        " FROM ai_answers aa"
                        " LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id"
                        f" {_ga_ir_join}"
                        f" WHERE {_ga_where} ORDER BY aa.submitted_on DESC"
                    )
                    sess_data["last_sql"]        = _ga_sql if _ga_scope != 'BLOCK' else None
                    sess_data["last_sql_params"] = _ga_params
                    sess_data["last_total"]      = inner.get("total_count", _row_count)
                    _skip_dashboard = True

            elif step.tool == "get_answer_stats":
                # get_answer_stats returns value_breakdown — convert to table rows for frontend
                info.success = (step.result or {}).get("success", True)
                inner = (step.result or {}).get("result") or {}
                breakdown = inner.get("value_breakdown", [])
                info.result_preview = str(breakdown[:3])[:200] if breakdown else "No data"
                if breakdown:
                    # Normalise to [{answer: ..., count: ...}] table shape
                    rows = [{"answer": b.get("value", ""), "count": str(b.get("count", 0))} for b in breakdown]
                    # Prepend aggregate stats row if available
                    avg = inner.get("average_value")
                    if avg is not None:
                        rows.insert(0, {"answer": "— average score —", "count": str(round(avg, 2))})
                    sess_data["last_raw_rows"] = rows
                    _result_rows = rows
                    _result_cols = list(rows[0].keys()) if rows else []
                    _result_cols = ["answer", "count"]
                    _row_count = len(breakdown)
                    sess_data["last_sql"]        = None
                    sess_data["last_sql_params"] = {}
                    sess_data["last_total"]      = None
                    _skip_dashboard = True

            elif step.tool == "[parse_error]":
                info.success = False

            steps.append(info)

        if result.steps:
            reasoning = getattr(result.steps[0], "thought", None) or None

        # Resolve pagination metadata now that last_total is populated
        _total_count = sess_data.get("last_total")
        if _total_count is not None and _total_count != -1:
            _has_more = _total_count > _row_count
        else:
            _has_more = _row_count >= 50

        # Name-disambiguation count-check queries exist purely so the model can
        # ask "which person/facility did you mean?" — showing them as a full
        # animated heatmap/KPI dashboard is misleading (looks like an analytics
        # result, not a pick-one list). Uses the same shared shape-checks
        # (core.disambiguation) that agent/orchestrator.py's _synthesize() and
        # _extract_disambiguation()/_extract_entity_disambiguation() use, so the
        # chat text and the dashboard/chart shown for the same result never
        # disagree. Based on the FINAL result shape only, so an earlier
        # disambiguation step in the same turn doesn't suppress the dashboard
        # for a later, legitimate analytical result. Covers both the people
        # shape (user_id+name) and the entity shape (facility/project/client/
        # organisation — id+name+inspection_count).
        _last_sql = sess_data.get("last_sql") or ""
        _disambiguation_meta: DisambiguationMeta | None = None
        if _result_cols:
            if is_disambiguation_result(_result_cols, _last_sql):
                _skip_dashboard = True
                _term = extract_search_term(_last_sql)
                if _term:
                    _disambiguation_meta = DisambiguationMeta(kind="person", search_term=_term)
            else:
                _entity = is_entity_disambiguation_result(_result_cols, _last_sql)
                if _entity:
                    _skip_dashboard = True
                    _term = extract_search_term(_last_sql)
                    if _term:
                        _disambiguation_meta = DisambiguationMeta(kind=_entity[0], search_term=_term)

        # Generate answer text; decide whether to show dashboard or plain chart suggestion
        if _result_rows and len(_result_rows) > 1 and not _chart_config and not _skip_dashboard:
            summary = _generate_summary(req.message, _result_cols, _result_rows)
            display_answer = summary if summary else _clean_answer(result.answer, _result_rows)
            raw_for_chart = sess_data["last_raw_rows"] if sess_data.get("last_raw_rows") else _result_rows

            rec = advise(_result_cols, raw_for_chart)
            if should_show_dashboard(_result_cols, raw_for_chart, rec):
                # Dashboard path: embed chart_config directly; KPI cards computed here.
                # The frontend renders dashboard-display (cards + chart + table) instead of
                # the plain "Show as chart" button — no __CHART_AVAILABLE__ signal needed.
                _chart_config = ChartConfig(
                    chart_type=rec["chart_type"],
                    reason=rec["reason"],
                    confidence=rec["confidence"],
                    config=rec["config"],
                    conflict=False,
                    requested_type=None,
                    requested_config=None,
                    alternatives=rec.get("alternatives", []),
                    x_label=rec.get("labels_column"),
                    y_labels=rec.get("values_columns") or [],
                )
                _dashboard = DashboardData(
                    show=True,
                    cards=build_kpi_cards(_result_cols, raw_for_chart),
                )
            else:
                _dashboard = DashboardData(show=False, cards=[])
                if rec.get("chart_type"):
                    rtype  = rec["chart_type"]
                    rlabel = _TYPE_LABELS.get(rtype, rtype.title())
                    alts   = _ALT_MAP.get(rtype, [])[:2]
                    hint   = rec["reason"]
                    signal = f"\n📊__CHART_AVAILABLE__:{rtype}:{hint}"
                    alt_str = " or a ".join(f"**{_TYPE_LABELS.get(a, a.title())}**" for a in alts)
                    if alts:
                        display_answer += f"{signal}\n\nI can visualize this as a **{rlabel}** ({hint}), or as a {alt_str} if you prefer — just say the word."
                    else:
                        display_answer += f"{signal}\n\nI can visualize this as a **{rlabel}** ({hint}) — just ask."
        else:
            display_answer = _clean_answer(result.answer, _result_rows)

        # Single-row P&L waterfall: auto-embed chart even when len == 1.
        # The len > 1 gate above skips single-row results, but a high-confidence
        # waterfall (income/cost/profit columns) should always show a chart.
        if (not _chart_config and not _skip_dashboard
                and _result_rows and len(_result_rows) == 1):
            raw_for_chart = sess_data["last_raw_rows"] if sess_data.get("last_raw_rows") else _result_rows
            _pl_rec = advise(_result_cols, raw_for_chart)
            if _pl_rec.get("chart_type") == "waterfall" and _pl_rec.get("confidence") == "high":
                _chart_config = ChartConfig(
                    chart_type=_pl_rec["chart_type"],
                    reason=_pl_rec["reason"],
                    confidence=_pl_rec["confidence"],
                    config=_pl_rec["config"],
                    conflict=False,
                    requested_type=None,
                    requested_config=None,
                    alternatives=_pl_rec.get("alternatives", []),
                    x_label=_pl_rec.get("labels_column"),
                    y_labels=_pl_rec.get("values_columns") or [],
                )
                _dashboard = _dashboard or DashboardData(show=False, cards=[])

        # Scrub raw UUID columns (user_id, facility_id, ...) right before the
        # table reaches the frontend — done last so it never affects chart/KPI
        # column selection or summary generation upstream, only the payload.
        _result_cols, _result_rows = _strip_uuid_columns(_result_cols, _result_rows)
        display_answer = _redact_security_internals(display_answer)
        reasoning = _redact_security_internals(reasoning)

        log_query.info(
            "[QUERY] END   qid=%s sub=%s elapsed=%dms llm_calls=%d "
            "sql_errors=%d rows=%d tokens=%d/%d success=%s",
            qid, sub, elapsed_ms,
            result.stats.total_llm_calls,
            sql_errors,
            _row_count,
            result.stats.total_prompt_tokens,
            result.stats.total_completion_tokens,
            result.success,
        )

        return ChatResponse(
            answer=display_answer,
            sql=sql,
            reasoning=reasoning,
            prompt_tokens=result.stats.total_prompt_tokens,
            completion_tokens=result.stats.total_completion_tokens,
            total_tokens=result.stats.total_prompt_tokens + result.stats.total_completion_tokens,
            time_ms=elapsed_ms,
            llm_calls=result.stats.total_llm_calls,
            sql_errors=sql_errors,
            steps=steps,
            session_id=req.session_id,
            result_data=ResultData(
                rows=_result_rows,
                columns=_result_cols,
                row_count=_row_count,
                total_count=_total_count,
                page=0,
                page_size=50,
                has_more=_has_more,
                sql=sql,
                disambiguation=_disambiguation_meta,
            ) if _result_rows else None,
            chart_config=_chart_config,
            dashboard=_dashboard,
        )

    except Exception as exc:
        log_query.exception("[QUERY] ERROR qid=%s sub=%s exc=%s", qid, sub, exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _refetch_for_calendar(sql: str, orch, limit: int = 1000, params: dict | None = None) -> list[dict]:
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


def _paginate_sql(sql: str, page: int, page_size: int) -> str:
    """
    Replace (or append) LIMIT/OFFSET in an SQL query for the requested page.
    page is 0-based.
    """
    import re as _re
    offset = page * page_size
    # Remove existing LIMIT / OFFSET
    stripped = _re.sub(
        r'\bLIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*;?\s*$', '',
        sql.strip(), flags=_re.IGNORECASE
    ).strip().rstrip(';')
    return f"{stripped}\nLIMIT {page_size} OFFSET {offset};"


@app.post("/api/page")
async def get_page(req: PageRequest, token_data: dict = Depends(require_auth), orch=Depends(get_orch)):
    """
    Paginate the last server-generated SQL for this session.
    Client-supplied SQL is never accepted — pagination always re-runs the
    scope-enforced SQL that was cached when the original query executed.
    """
    try:
        # Only the server-generated, scope-enforced SQL is trusted here.
        # The session key is scoped to (session_id, sub) so a user cannot
        # paginate another user's session even with a known session_id.
        sess_data = _sessions.get(f"{req.session_id}:{token_data['sub']}")
        if not sess_data or not sess_data.get("last_sql"):
            raise HTTPException(
                status_code=404,
                detail="No cached query for this session. Ask a question first."
            )
        sql   = sess_data["last_sql"]
        total = sess_data.get("last_total")

        paged_sql = _paginate_sql(sql, req.page, req.page_size)

        from sqlalchemy import text as sql_text
        _page_params = sess_data.get("last_sql_params", {})
        with orch.db_engine.connect() as conn:
            result  = conn.execute(sql_text(paged_sql), _page_params)
            columns = list(result.keys())
            rows    = [{k: str(v) for k, v in dict(row._mapping).items()} for row in result.fetchall()]

        has_more = (
            (total is not None and (req.page + 1) * req.page_size < total)
            or len(rows) >= req.page_size
        )

        # Recompute the same disambiguation metadata /api/chat attaches, so
        # paging into a large pick-list (e.g. 112 name matches, 50/page)
        # doesn't silently lose click-to-resolve on page 2+ — goToPage()
        # replaces the whole result object with this response.
        _page_disambiguation = None
        if is_disambiguation_result(columns, sql):
            _term = extract_search_term(sql)
            if _term:
                _page_disambiguation = {"kind": "person", "search_term": _term}
        else:
            _page_entity = is_entity_disambiguation_result(columns, sql)
            if _page_entity:
                _term = extract_search_term(sql)
                if _term:
                    _page_disambiguation = {"kind": _page_entity[0], "search_term": _term}

        columns, rows = _strip_uuid_columns(columns, rows)

        return {
            "rows":           rows,
            "columns":        columns,
            "row_count":      len(rows),
            "total_count":    total,
            "page":           req.page,
            "page_size":      req.page_size,
            "has_more":       has_more,
            "disambiguation": _page_disambiguation,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str, token_data: dict = Depends(require_auth)):
    _sessions.pop(f"{session_id}:{token_data['sub']}", None)
    return {"status": "cleared", "session_id": session_id}


@app.post("/api/chart")
async def get_chart(req: ChartRequest, request: Request, token_data: dict = Depends(require_auth)):
    """
    Slash-command chart endpoint — /chart [type?]
    Generates or regenerates a chart from the session's last result.
    Zero LLM calls. Zero SQL calls. Pure chart advisor.

    - chart_type=None      → auto-advise best type
    - chart_type="bar"     → force that specific type
    - x_col / y_cols       → column picker overrides (user selected columns)
    """
    sess_data = _sessions.get(f"{req.session_id}:{token_data['sub']}")
    if not sess_data or not sess_data.get("last_raw_rows"):
        raise HTTPException(
            status_code=404,
            detail="No data in session. Run a query first, then use /chart."
        )

    cols = sess_data["last_cols"]
    rows = sess_data["last_raw_rows"]

    # Calendar heatmap needs the full date range, not just the cached 50 rows.
    _cal_needed = (req.chart_type == "calendar_heatmap") or (not req.chart_type and advise(cols, rows).get("chart_type") == "calendar_heatmap")
    _chart_orch = get_orch_optional(request)
    if _cal_needed and sess_data.get("last_sql") and _chart_orch:
        _full = _refetch_for_calendar(sess_data["last_sql"], _chart_orch, params=sess_data.get("last_sql_params", {}))
        if _full:
            rows = _full
            cols = list(_full[0].keys())

    # Column picker overrides — user explicitly selected which cols to chart
    lc_override = req.x_col
    num_cols_override = req.y_cols

    if req.chart_type:
        cfg = force_config(cols, rows, req.chart_type, lc=lc_override, num_cols=num_cols_override)
        logging.info(f"Forcing chart {req.chart_type} for lc={lc_override}, nums={num_cols_override}. Config generated: {bool(cfg)}")
        if cfg is None:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot build '{req.chart_type}' chart for this data shape."
            )
        # Determine real confidence by comparing forced type to what the advisor recommends
        _COMPAT = [
            {"bar", "horizontal_bar", "stacked_bar", "waterfall"},
            {"line", "area", "calendar_heatmap"},
            {"pie", "doughnut", "nightingale"},
        ]
        rec = advise(cols, rows)
        rec_type = rec.get("chart_type")
        if req.chart_type == rec_type:
            forced_confidence = "high"
        elif any(req.chart_type in g and rec_type in g for g in _COMPAT if rec_type):
            forced_confidence = "medium"
        else:
            forced_confidence = "low"

        alts = get_alternatives(cols, rows, n=3)
        # x_label / y_labels from the advisor rec (covers forced type too)
        _x = rec.get("labels_column") if rec else (lc_override or (cols[0] if cols else None))
        _y = rec.get("values_columns") or (num_cols_override or [])
        return ChartConfig(
            chart_type=req.chart_type,
            reason=f"Requested by user via /chart {req.chart_type}",
            confidence=forced_confidence,
            config=cfg,
            conflict=False,
            requested_type=None,
            requested_config=None,
            alternatives=alts,
            x_label=_x,
            y_labels=_y,
        )
    else:
        rec = advise(cols, rows)
        if not rec.get("chart_type"):
            raise HTTPException(
                status_code=422,
                detail="Cannot determine a chart for this data. Try /chart bar or select specific columns."
            )
        return ChartConfig(
            chart_type=rec["chart_type"],
            reason=rec["reason"],
            confidence=rec["confidence"],
            config=rec["config"],
            conflict=False,
            requested_type=None,
            requested_config=None,
            alternatives=rec.get("alternatives", []),
            x_label=rec.get("labels_column"),
            y_labels=rec.get("values_columns") or [],
        )


@app.get("/api/sql/{session_id}")
async def get_last_sql(session_id: str, token_data: dict = Depends(require_auth)):
    """
    Slash-command /sql endpoint.
    Returns the last generated SQL for this session.
    """
    sess_data = _sessions.get(f"{session_id}:{token_data['sub']}")
    if not sess_data or not sess_data.get("last_sql"):
        raise HTTPException(
            status_code=404,
            detail="No SQL in session. Ask a data question first."
        )
    return {
        "sql": sess_data["last_sql"],
        "session_id": session_id,
    }


@app.get("/api/columns/{session_id}")
async def get_last_columns(session_id: str, token_data: dict = Depends(require_auth)):
    """
    Returns available columns and their types from the last result.
    Used by the column picker when user wants to customise a chart.
    """
    sess_data = _sessions.get(f"{session_id}:{token_data['sub']}")
    if not sess_data or not sess_data.get("last_cols"):
        raise HTTPException(status_code=404, detail="No result in session.")

    cols = sess_data["last_cols"]
    rows = sess_data.get("last_raw_rows", [])

    # Classify each column so the frontend can show sensible picker options
    t = classify_columns(cols, rows)

    return {
        "columns":    cols,
        "categories": t.get("category", []),
        "numerics":   t.get("numeric", []),
        "dates":      t.get("date", []),
        "session_id": session_id,
    }


@app.get("/api/health")
async def health(request: Request):
    _health_orch = get_orch_optional(request)
    if _health_orch is not None:
        model_label = f"{_health_orch.chat_model_name} + {_health_orch.sql_model_name}"
    else:
        import os
        chat = os.getenv("OLLAMA_CHAT_MODEL", "—")
        sql  = os.getenv("OLLAMA_SQL_MODEL",  "—")
        model_label = f"{chat} + {sql}"
    return {"status": "ok", "model": model_label}

@app.get("/api/ready")
async def ready(request: Request):
    """Used by start.sh to wait until the orchestrator is fully loaded."""
    from fastapi.responses import JSONResponse
    if get_orch_optional(request) is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready"}

@app.get("/api/validator-stats")
async def validator_stats():
    """
    Read-only diagnostic: how often each validator rule has fired since process
    start. In-memory only (resets on restart, same as _sessions) — enough to
    notice a SOFT_WARN firing constantly without needing to grep debug logs.
    """
    from khidmah.core.validator import ValidatorStats
    return ValidatorStats.snapshot()


# ── ASR ───────────────────────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...), _: None = Depends(require_auth)):
    """
    Accepts any browser audio blob (WebM/Opus, WAV, MP4 …),
    converts to 16 kHz mono WAV via ffmpeg, then transcribes
    with dimavz/whisper-tiny.  Returns {"text": "..."}.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    suffix = ".webm"
    ct = audio.content_type or ""
    if "wav" in ct:   suffix = ".wav"
    elif "mp4" in ct: suffix = ".mp4"
    elif "ogg" in ct: suffix = ".ogg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fin:
        fin.write(raw)
        in_path = fin.name

    out_path = in_path.replace(suffix, "_16k.wav")
    try:
        # Decode & resample to 16 kHz mono PCM WAV
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=422,
                                detail="Audio conversion failed: " + result.stderr.decode()[-300:])

        import soundfile as sf
        data, sr = sf.read(out_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo → mono (safety)

        pipe = _get_asr()
        out  = pipe({"array": data, "sampling_rate": sr})
        text = out.get("text", "").strip()
        return {"text": text}

    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ── TTS ───────────────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str

@app.post("/api/tts")
async def tts(req: TTSRequest, _: None = Depends(require_auth)):
    """
    Converts answer text to speech using Kokoro-ONNX (CPU).
    Strips markdown, caps at 800 chars, returns audio/wav.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    clean = _clean_for_tts(req.text)
    if not clean:
        raise HTTPException(status_code=400, detail="Nothing to speak after stripping markdown")

    try:
        import io
        import soundfile as sf
        from fastapi.responses import Response

        import warnings
        kokoro = _get_tts()
        def _synthesize():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*words count mismatch.*")
                return kokoro.create(clean, voice="af_bella", speed=1.05, lang="en-us")
        samples, sample_rate = await asyncio.to_thread(_synthesize)
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")

    except Exception as exc:
        logging.exception("TTS error")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Serve Angular build (must be last — catches all non-API routes) ───────────
_DIST = os.path.join(_REPO_ROOT, "inspectai-chat", "dist", "inspectai-chat", "browser")
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