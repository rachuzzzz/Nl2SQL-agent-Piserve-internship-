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

# Ensure project root is on the path so agent/ and core/ are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="InspectAI NL2SQL API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────

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

class SessionInfo(BaseModel):
    session_id: str
    message_count: int


# ── Lazy singletons ────────────────────────────────────────────────────────────

_orch = None
_sessions: dict = {}


def get_orchestrator():
    global _orch
    if _orch is None:
        from agent.orchestrator import AgentOrchestrator
        from dotenv import load_dotenv
        load_dotenv()
        _orch = AgentOrchestrator.from_env()
    return _orch


def get_or_create_session(session_id: str) -> dict:
    from agent.orchestrator import ConversationSession
    if session_id not in _sessions:
        _sessions[session_id] = {
            "session": ConversationSession(),
            "message_count": 0,
        }
    return _sessions[session_id]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        orch = get_orchestrator()
        sess_data = get_or_create_session(req.session_id)

        t0 = time.time()
        result = orch.query(req.message, session=sess_data["session"])
        elapsed_ms = int((time.time() - t0) * 1000)
        sess_data["message_count"] += 1

        # ── Extract SQL, reasoning, and per-step info ──────────────────────────
        # step.result structure:
        #   generate_sql → {"success": bool, "result": {"sql": "...", "validation": {...}}, "error": ...}
        #   execute_sql  → {"success": bool, "result": {"rows": [...], "row_count": N, ...}, "error": ...}

        sql = None
        reasoning = None
        steps: list[StepInfo] = []
        sql_errors = 0
        _result_rows: list = []
        _result_cols: list = []
        _row_count: int = 0

        for step in result.steps:
            thought = getattr(step, "thought", "") or ""
            info = StepInfo(tool=step.tool, thought=thought)

            if step.tool == "generate_sql":
                # SQL lives at result["result"]["sql"]
                generated = (step.result or {}).get("result", {}).get("sql", "")
                if generated:
                    sql = generated          # keep the last successful SQL
                info.sql = generated
                info.success = (step.result or {}).get("success", True)

            elif step.tool == "execute_sql":
                info.success = (step.result or {}).get("success", True)
                if not info.success:
                    sql_errors += 1
                inner = (step.result or {}).get("result") or {}
                rows = inner.get("rows", [])
                info.result_preview = str(rows[:3])[:200] if rows else "No rows"
                # Capture for table rendering
                if rows and isinstance(rows[0], dict):
                    _result_rows = [
                        {k: str(v) for k, v in row.items()}
                        for row in rows
                    ]
                    _result_cols = list(rows[0].keys())
                    _row_count = inner.get('row_count', len(rows))

            elif step.tool == "[parse_error]":
                info.success = False

            steps.append(info)

        # First step thought = Qwen's planning reasoning
        if result.steps:
            reasoning = getattr(result.steps[0], "thought", None) or None


        # SHORT_ANSWER_PATCH
        # If we have structured rows, shorten the answer to avoid duplication
        display_answer = result.answer
        if _result_rows:
            # Extract just the first line e.g. "Found 292 result(s)..."
            first_line = result.answer.split("\n")[0].strip()
            # If it looks like a count line, use it; otherwise generic
            if first_line.lower().startswith("found") or first_line.lower().startswith("result"):
                display_answer = first_line
            else:
                display_answer = f"Showing {_row_count} result{'' if _row_count == 1 else 's'}."
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
    return {"status": "ok", "model": "llama3.1:8b + deepseek:6.7b"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)