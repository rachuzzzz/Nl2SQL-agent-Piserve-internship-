"""
InspectAI — FastAPI Backend
===========================
Place this file in the project root (same folder as run.py).

Install:  pip install fastapi uvicorn
Run:      uvicorn fastapi_app:app --reload --port 8000
"""

import time
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import re
import logging
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chart_advisor import advise, force_config

# ── Chart follow-up fast-path ──────────────────────────────────────────────────
# Detects pure chart requests that need no new SQL — handled before the LLM.

_VISUAL_KW = re.compile(
    r'\b(chart|graph|plot|visual(i[sz]e|ization)?|bar|pie|line|doughnut|donut|scatter|draw)\b',
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

log = logging.getLogger("inspectai")

# ── Eager startup ──────────────────────────────────────────────────────────────

_orch = None
_sessions: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the orchestrator at server start so the first request is instant."""
    global _orch
    from dotenv import load_dotenv
    load_dotenv()
    log.info("⏳  Loading orchestrator (LLM + embeddings + schema)…")
    t0 = time.time()
    from agent.orchestrator import AgentOrchestrator
    _orch = AgentOrchestrator.from_env()
    log.info(f"✅  Orchestrator ready in {time.time() - t0:.1f}s")
    yield
    # shutdown — nothing to clean up
    log.info("👋  Shutting down")


# ── Pydantic models ────────────────────────────────────────────────────────────

app = FastAPI(title="InspectAI NL2SQL API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

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

class ChartConfig(BaseModel):
    chart_type: str | None
    reason: str
    confidence: str
    config: dict | None = None
    conflict: bool = False
    requested_type: str | None = None
    requested_config: dict | None = None

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
            "last_raw_rows": [],   # cached for chart follow-up fast-path
            "last_cols": [],
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
                    label = str(lv).strip()
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
        if alts:
            return f"\n\nI can visualize this as a **{rlabel}** ({hint}), or as a {alt_str} if you prefer — just say the word."
        return f"\n\nI can visualize this as a **{rlabel}** ({hint}) — just ask."
    except Exception:
        return ""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        orch = _orch
        sess_data = get_or_create_session(req.session_id)

        # ── Chart follow-up fast-path — skip LLM entirely ────────────────────
        if _is_chart_followup(req.message) and sess_data.get("last_raw_rows"):
            cols = sess_data["last_cols"]
            rows = sess_data["last_raw_rows"]
            requested = _extract_requested_type(req.message)
            rec = advise(cols, rows)
            if rec.get("chart_type"):
                conflict = bool(requested and requested != rec["chart_type"])
                cc = ChartConfig(
                    chart_type=rec["chart_type"], reason=rec["reason"],
                    confidence=rec["confidence"], config=rec["config"],
                    conflict=conflict,
                    requested_type=requested if conflict else None,
                    requested_config=force_config(cols, rows, requested) if conflict else None,
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
        result = orch.query(req.message, session=sess_data["session"])
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

            elif step.tool == "[parse_error]":
                info.success = False

            steps.append(info)

        if result.steps:
            reasoning = getattr(result.steps[0], "thought", None) or None

        # Generate a natural-language summary + chart suggestion for multi-row results
        if _result_rows and len(_result_rows) > 1 and not _chart_config:
            # Use _result_rows (stringified) — _coerce() in _generate_summary handles string numbers
            # Avoids the sess_data.get() empty-list bug with last_raw_rows
            summary = _generate_summary(req.message, _result_cols, _result_rows)
            display_answer = summary if summary else _clean_answer(result.answer, _result_rows)
            raw_for_chart = sess_data["last_raw_rows"] if sess_data.get("last_raw_rows") else _result_rows
            suggestion = _chart_suggestion_sentence(_result_cols, raw_for_chart)
            if suggestion:
                display_answer += suggestion
        else:
            display_answer = _clean_answer(result.answer, _result_rows)

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
            ) if _result_rows else None,
            chart_config=_chart_config,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    data = _sessions.get(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(session_id=session_id, message_count=data["message_count"])


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "qwen3.5:9b + qwen2.5-coder"}

@app.get("/api/ready")
async def ready():
    """Used by start.sh to wait until the orchestrator is fully loaded."""
    from fastapi.responses import JSONResponse
    if _orch is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)