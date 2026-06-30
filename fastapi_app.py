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
import sys
import tempfile
import subprocess
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer as _HTTPBearer, HTTPAuthorizationCredentials as _HTTPCreds
from pydantic import BaseModel, Field

import re
import uuid
import logging
from contextlib import asynccontextmanager

from core.logging_config import setup_logging

# ── TTS — lazy-loaded on first /api/tts call ──────────────────────────────────
_tts_model = None

_TTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chart_advisor import advise, force_config, get_alternatives, classify_columns, build_kpi_cards, should_show_dashboard

# ── Chart follow-up fast-path ──────────────────────────────────────────────────
# Detects pure chart requests that need no new SQL — handled before the LLM.

_VISUAL_KW = re.compile(
    r'\b(chart|graph|plot|visual(i[sz]e|ization)?|bar|pie|line|doughnut|donut|scatter|draw)\b'
    r'|\bshow\s+as\s+chart\b|\bas\s+(a\s+)?chart\b',
    re.IGNORECASE,
)
_DOMAIN_KW = re.compile(
    r'\b(inspection|facility|capex|opex|score|corrective|action|overdue|report|month|'
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

# ── Eager startup ──────────────────────────────────────────────────────────────

_orch = None
_sessions: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the orchestrator at server start so the first request is instant."""
    global _orch
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging(log_level="INFO")
    log.info("[SERVER] Starting InspectAI — loading orchestrator (LLM + embeddings + schema)…")
    t0 = time.time()
    from agent.orchestrator import AgentOrchestrator
    _orch = AgentOrchestrator.from_env()
    log.info("[SERVER] Orchestrator ready in %.1fs", time.time() - t0)
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

# ── Token store ────────────────────────────────────────────────────────────────
# Maps access_token → {expiry, sub, permissions[]}. Populated on /api/login
# (sub + expiry only) and hydrated by /api/me/permissions. Cleared on restart.
_valid_tokens: dict[str, dict] = {}
_bearer_scheme = _HTTPBearer(auto_error=False)

# Inspection data-visibility permissions — the only ones the orchestrator is
# allowed to know about. All other permissions are stripped before the
# visibility_scope dict reaches the SQL layer. This is the boundary wall.
DATA_VISIBILITY_PERMS: frozenset = frozenset({
    'view list of all inspections',
    'view list of own inspections',
    'view own inspection dashboard',
    'view own inspection portfolio list',
    'view list of own inspection schedules',
    'view list of own inspection mitigative actions',
    'view list of all inspection mitigative actions',
    'conduct an inspection',
    'can be an inspectee',
    # Grants scoped access to PO details linked to the user's own CAs
    'view po details in any mitigative action',
    # Scoped cycle and project dashboard access via the inspection chain
    'view inspection cycle list',
    'view project inspection dashboard',
})


def _decode_jwt_payload(token: str) -> dict:
    """Decode the JWT payload segment (base64url) without signature verification."""
    import base64, json as _json
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return {}
        payload = parts[1]
        payload += '=' * (4 - len(payload) % 4)
        return _json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _register_token(token: str, sub: str, permissions: list, expires_in: int = 3600) -> None:
    _valid_tokens[token] = {
        'expiry': _time.time() + expires_in,
        'sub': sub,
        'permissions': permissions,
    }
    now = _time.time()
    expired = [t for t, data in list(_valid_tokens.items()) if data['expiry'] < now]
    for t in expired:
        _valid_tokens.pop(t, None)


async def require_auth(
    credentials: _HTTPCreds | None = Depends(_bearer_scheme),
) -> dict:
    token = credentials.credentials if credentials else None
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    data = _valid_tokens.get(token)
    if not data or data['expiry'] < _time.time():
        raise HTTPException(status_code=401, detail="Authentication required")
    return data


def require_scope(*scopes: str):
    """Dependency factory — raises 403 if the token holds none of the given scopes."""
    async def _dep(token_data: dict = Depends(require_auth)) -> dict:
        if scopes and not any(s in token_data['permissions'] for s in scopes):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return token_data
    return _dep


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

class ResultData(BaseModel):
    rows: list[dict]
    columns: list[str]
    row_count: int
    total_count: int | None = None
    page: int = 0
    page_size: int = 50
    has_more: bool = False
    sql: str | None = None   # base SQL — sent back with every /api/page call

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

class SessionInfo(BaseModel):
    session_id: str
    message_count: int


# ── Session management ────────────────────────────────────────────────────────

def get_or_create_session(session_id: str) -> dict:
    from agent.orchestrator import ConversationSession
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
            s = str(v).strip()
            # Already a date/datetime object
            if isinstance(v, datetime):
                return v.strftime("%b %-d, %Y")
            if isinstance(v, date):
                return v.strftime("%b %-d, %Y")
            # ISO timestamp strings: 2026-01-01 00:00:00+00:00 or 2026-01-01T00:00:00Z
            if re.match(r'^\d{4}-\d{2}-\d{2}[ T]', s):
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                    # If time is midnight, show date only
                    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                        return dt.strftime("%b %-d, %Y")
                    return dt.strftime("%b %-d, %Y %H:%M")
                except Exception:
                    pass
            # Plain date string: 2026-01-01
            if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
                try:
                    dt = datetime.strptime(s, "%Y-%m-%d")
                    return dt.strftime("%b %-d, %Y")
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

def _chart_suggestion_sentence(columns: list, rows: list) -> str:
    """Build a conversational chart suggestion to append to the answer text."""
    try:
        from chart_advisor import advise
        rec = advise(columns, rows)
        if not rec["chart_type"]:
            return ""
        rtype = rec["chart_type"]
        rlabel = _TYPE_LABELS.get(rtype, rtype.title())
        alts = _ALT_MAP.get(rtype, [])[:2]
        alt_str = " or a ".join(f"**{_TYPE_LABELS.get(a, a.title())}**" for a in alts)
        hint = rec["reason"]

        signal = f"\n📊__CHART_AVAILABLE__:{rtype}:{hint}"
        if alts:
            return f"{signal}\n\nI can visualize this as a **{rlabel}** ({hint}), or as a {alt_str} if you prefer — just say the word."
        return f"{signal}\n\nI can visualize this as a **{rlabel}** ({hint}) — just ask."
    except Exception:
        return ""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, token_data: dict = Depends(require_auth)):
    qid = uuid.uuid4().hex[:8]
    sub = token_data['sub']
    try:
        orch = _orch
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
            log_query.info("[SCOPE] qid=%s sub=%s data_visibility_perms=%s",
                           qid, sub, _data_vis or "(none — will BLOCK)")
        else:
            visibility_scope = None  # → BLOCK in orchestrator
            log_query.warning("[SCOPE] qid=%s sub=%s — no permissions fetched, BLOCKING", qid, sub)

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

                    # Cache SQL for pagination — if it's a new or auto-paginated query
                    exec_sql = sql or (step.args or {}).get("sql")
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
                    # Uses _build_scope_filter (server-trusted SQL, int(sub) cast).
                    # Added to clauses before SQL string is built to preserve :params.
                    _ga_scope = _orch._build_scope_filter(visibility_scope) if visibility_scope else 'BLOCK'
                    if _ga_scope and _ga_scope != 'BLOCK':
                        _ga_clauses.append(_ga_scope)
                    _ga_where = " AND ".join(_ga_clauses)
                    _ga_sql = (
                        "SELECT ir.inspection_id, aa.module_name,"
                        " aq.label AS question_label,"
                        " COALESCE(aa.answer_text, aa.answer_numeric::text) AS answer_value,"
                        " aa.score, aa.submitted_on"
                        " FROM ai_answers aa"
                        " LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id"
                        " LEFT JOIN inspection_report ir ON aa.inspection_report_id = ir.id"
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


def _count_total(sql: str, orch) -> int | None:
    """
    Count total rows for the SQL query without LIMIT/OFFSET.
    Works for both plain SELECT and GROUP BY queries by wrapping in a subquery.
    Returns None only on DB error.
    """
    import re as _re
    # Strip trailing LIMIT / OFFSET / semicolon
    stripped = _re.sub(
        r'\bLIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*;?\s*$', '',
        sql.strip(), flags=_re.IGNORECASE
    ).strip().rstrip(';')

    # Also strip any WITH ... AS CTEs that are already in the main query —
    # they're preserved inside the subquery so the count is accurate
    count_sql = f"SELECT COUNT(*) AS _total FROM ({stripped}) _sub"
    try:
        from sqlalchemy import text as sql_text
        with orch.db_engine.connect() as conn:
            row = conn.execute(sql_text(count_sql)).fetchone()
            return int(row[0]) if row else None
    except Exception:
        return None


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
async def get_page(req: PageRequest, token_data: dict = Depends(require_auth)):
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
        with _orch.db_engine.connect() as conn:
            result  = conn.execute(sql_text(paged_sql), _page_params)
            columns = list(result.keys())
            rows    = [{k: str(v) for k, v in dict(row._mapping).items()} for row in result.fetchall()]

        has_more = (
            (total is not None and (req.page + 1) * req.page_size < total)
            or len(rows) >= req.page_size
        )

        return {
            "rows":        rows,
            "columns":     columns,
            "row_count":   len(rows),
            "total_count": total,
            "page":        req.page,
            "page_size":   req.page_size,
            "has_more":    has_more,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



async def get_session_info(session_id: str):
    data = _sessions.get(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(session_id=session_id, message_count=data["message_count"])


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str, token_data: dict = Depends(require_auth)):
    _sessions.pop(f"{session_id}:{token_data['sub']}", None)
    return {"status": "cleared", "session_id": session_id}


@app.post("/api/chart")
async def get_chart(req: ChartRequest, token_data: dict = Depends(require_auth)):
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
    if _cal_needed and sess_data.get("last_sql") and _orch:
        _full = _refetch_for_calendar(sess_data["last_sql"], _orch, params=sess_data.get("last_sql_params", {}))
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


class LoginRequest(BaseModel):
    username: str
    password: str



@app.post("/api/login")
async def login(req: LoginRequest):
    """Proxy OAuth password-grant to the identity server.
    Client credentials never leave the backend. Returns the OAuth response
    plus the decoded Drupal sub so Angular can display user info."""
    oauth_url     = os.getenv("OAUTH_URL", "https://10.77.20.51/oauth/token?_format=json")
    client_id     = os.getenv("CLIENT_ID", "").strip()
    client_secret = os.getenv("CLIENT_SECRET", "").strip()
    ca_cert = os.getenv("OAUTH_CA_CERT", "").strip()
    skip_tls = os.getenv("OAUTH_SKIP_TLS_VERIFY", "").strip().lower() in ("1", "true", "yes")
    verify: str | bool = False if skip_tls else (ca_cert if ca_cert else True)

    payload = {
        "grant_type":    "password",
        "client_id":     client_id,
        "client_secret": client_secret,
        "username":      req.username,
        "password":      req.password,
    }
    log_auth.info("[AUTH] Login attempt user=%r", req.username)
    try:
        async with httpx.AsyncClient(verify=verify, timeout=15) as client:
            resp = await client.post(oauth_url, data=payload)
        if resp.status_code == 200:
            data = resp.json()
            access_token = data["access_token"]
            token_hint = access_token[:8] + "…"

            # Decode JWT to extract Drupal UID (sub claim).
            # Validate at decode time — a malformed sub must never enter the token store.
            claims = _decode_jwt_payload(access_token)
            sub = str(claims.get("sub", "")).strip()
            log_auth.debug("[AUTH] JWT decoded sub=%r claims_keys=%s", sub, list(claims.keys()))
            if not re.match(r'^\d+$', sub) or sub == "0":
                log_auth.warning("[AUTH] Login FAIL user=%r — invalid sub=%r", req.username, sub)
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token: sub must be a positive integer"
                )

            # Verify user is synced in the local DB (OAuth already validated credentials)
            if _orch is None:
                raise HTTPException(status_code=503, detail="Service not ready")
            from sqlalchemy import text as _sql_text
            try:
                with _orch.db_engine.connect() as conn:
                    row = conn.execute(
                        _sql_text(
                            "SELECT id_fk FROM users WHERE id_fk = :sub LIMIT 1"
                        ),
                        {"sub": int(sub)},
                    ).fetchone()
                    log_auth.debug("[AUTH] DB lookup id_fk=%r → found=%s", sub, row is not None)
            except (ValueError, TypeError):
                log_auth.exception("[AUTH] sub=%r could not be cast to int", sub)
                raise HTTPException(status_code=401, detail="Invalid token: sub is not a valid user ID")
            except Exception:
                log_auth.exception("[AUTH] DB user sync check failed for sub=%r", sub)
                raise HTTPException(status_code=502, detail="Database lookup failed")

            if not row:
                log_auth.warning("[AUTH] Login FAIL user=%r sub=%r — not in local DB", req.username, sub)
                raise HTTPException(status_code=403, detail="User not found in system")
            if row.id_fk == -1:
                log_auth.warning("[AUTH] Login FAIL user=%r sub=%r — account not synced", req.username, sub)
                raise HTTPException(status_code=403, detail="User account not synchronized")

            # Register token with empty permissions — client fetches via /api/me/permissions
            _register_token(access_token, sub, [], int(data.get("expires_in", 3600)))
            log_auth.info("[AUTH] Login OK user=%r sub=%s token=%s expires_in=%ss",
                          req.username, sub, token_hint, data.get("expires_in", 3600))
            return {**data, "sub": sub}

        try:
            detail = resp.json().get("error_description") or resp.json().get("message") or "Invalid credentials"
        except Exception:
            detail = "Invalid credentials"
        log_auth.warning("[AUTH] Login FAIL user=%r — OAuth status=%d detail=%r",
                         req.username, resp.status_code, detail)
        # Never forward a 5xx from the OAuth server — it looks like our fault.
        fwd_status = resp.status_code if resp.status_code < 500 else 502
        raise HTTPException(status_code=fwd_status, detail=detail)
    except HTTPException:
        raise
    except Exception as _exc:
        import traceback
        _tb = traceback.format_exc()
        log_auth.error("[AUTH] Login ERROR user=%r exc=%s\n%s", req.username, _exc, _tb)
        raise HTTPException(status_code=502, detail=f"Authentication service unavailable: {_exc}")


@app.post("/api/logout")
async def logout(
    credentials: _HTTPCreds | None = Depends(_bearer_scheme),
    token_data: dict = Depends(require_auth),
):
    """Invalidate the caller's access token server-side."""
    token = credentials.credentials  # type: ignore[union-attr]
    _valid_tokens.pop(token, None)
    log_auth.info("[AUTH] Logout sub=%s", token_data.get("sub"))
    return {"status": "logged_out"}


@app.get("/api/me/permissions")
async def get_my_permissions(
    credentials: _HTTPCreds | None = Depends(_bearer_scheme),
    token_data: dict = Depends(require_auth),
):
    """Fetch (and cache) the permission list for the authenticated user.
    The full list is stored in the token store but only DATA_VISIBILITY_PERMS
    is ever forwarded to the orchestrator — nothing else reaches the SQL layer."""
    sub = token_data["sub"]  # Drupal integer UID from JWT

    # Return cached list if already populated (avoids hammering the Java API)
    if token_data.get("permissions"):
        return {"sub": sub, "permissions": token_data["permissions"]}

    # The permissions API expects the internal UUID, not the Drupal integer.
    # Resolve: JWT sub (int) → users.id_fk → users.id (UUID)
    if _orch is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    from sqlalchemy import text as _sql_text
    try:
        with _orch.db_engine.connect() as conn:
            row = conn.execute(
                _sql_text("SELECT id FROM users WHERE id_fk = :sub LIMIT 1"),
                {"sub": int(sub)},
            ).fetchone()
    except Exception:
        logging.exception("UUID lookup failed for sub=%s", sub)
        raise HTTPException(status_code=502, detail="User UUID lookup failed")

    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    user_uuid = str(row.id)

    perms_base = os.getenv("PERMISSIONS_API_URL", "http://10.77.20.54:8083")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{perms_base}/api/users/{user_uuid}/permissions")
        if resp.status_code == 404:
            log_auth.warning("[AUTH] Permissions API 404 sub=%s uuid=%s — no permissions assigned", sub, user_uuid)
            raise HTTPException(status_code=403, detail="You have no permission to access this feature")
        if resp.status_code != 200:
            log_auth.warning("[AUTH] Permissions API returned %s for sub=%s uuid=%s",
                             resp.status_code, sub, user_uuid)
            raise HTTPException(status_code=502, detail="Permissions service unavailable")
        result = resp.json()
        raw = result.get("result", [])
        # Deduplicate while preserving order (API sometimes repeats entries)
        permissions = list(dict.fromkeys(raw))

        if not permissions:
            log_auth.warning("[AUTH] No permissions assigned sub=%s uuid=%s", sub, user_uuid)
            raise HTTPException(status_code=403, detail="You have no permission to access this feature")

        data_vis = [p for p in permissions if p in DATA_VISIBILITY_PERMS]

        if not data_vis:
            log_auth.warning(
                "[AUTH] Access DENIED sub=%s uuid=%s — total=%d permissions but 0 data-visibility grants",
                sub, user_uuid, len(permissions),
            )
            raise HTTPException(status_code=403, detail="You have no permission to access this feature")

        # Hydrate the token store so future require_auth calls carry permissions
        token_str = credentials.credentials if credentials else None
        if token_str and token_str in _valid_tokens:
            _valid_tokens[token_str]["permissions"] = permissions

        log_auth.info("[AUTH] Permissions loaded sub=%s uuid=%s total=%d data_visibility=%d: %s",
                      sub, user_uuid, len(permissions), len(data_vis), data_vis or "(none)")
        return {"sub": sub, "permissions": permissions}
    except HTTPException:
        raise
    except Exception:
        log_auth.exception("[AUTH] Permissions fetch error for sub=%s uuid=%s", sub, user_uuid)
        raise HTTPException(status_code=502, detail="Permissions service unavailable")


@app.get("/api/health")
async def health():
    if _orch is not None:
        model_label = f"{_orch.chat_model_name} + {_orch.sql_model_name}"
    else:
        import os
        chat = os.getenv("OLLAMA_CHAT_MODEL", "—")
        sql  = os.getenv("OLLAMA_SQL_MODEL",  "—")
        model_label = f"{chat} + {sql}"
    return {"status": "ok", "model": model_label}

@app.get("/api/ready")
async def ready():
    """Used by start.sh to wait until the orchestrator is fully loaded."""
    from fastapi.responses import JSONResponse
    if _orch is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready"}


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
_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "inspectai-chat", "dist", "inspectai-chat", "browser")
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