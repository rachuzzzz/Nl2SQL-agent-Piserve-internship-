"""
Chat, pagination, and session routes — /api/chat, /api/page,
/api/sessions/{id}, /api/sql/{id}, /api/columns/{id}.
"""

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from src.charts.advisor import advise, force_config, classify_columns, build_kpi_cards, should_show_dashboard
from src.security.auth import require_auth
from src.sql.result_shape import is_disambiguation_result, is_entity_disambiguation_result, extract_search_term
from src.security.rbac import DATA_VISIBILITY_PERMS, build_scope_filter
from src.api.deps import get_orch, get_or_create_session, refetch_for_calendar, _sessions
from src.api.response_formatting import (
    is_chart_followup, extract_requested_type, strip_uuid_columns,
    redact_security_internals, clean_answer, generate_summary, paginate_sql,
    TYPE_LABELS, ALT_MAP,
)
from src.api.schemas import (
    ChatRequest, ChatResponse, PageRequest, ResultData, ChartConfig, DashboardData,
    DisambiguationMeta, StepInfo,
)

log_query = logging.getLogger("inspectai.query")
log_security = logging.getLogger("inspectai.security")

router = APIRouter()

# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/api/chat", response_model=ChatResponse)
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
        if is_chart_followup(req.message) and sess_data.get("last_raw_rows"):
            cols = sess_data["last_cols"]
            rows = sess_data["last_raw_rows"]
            requested = extract_requested_type(req.message)

            # Calendar heatmap needs the full date range, not just the cached 50 rows.
            _cal_needed = (requested == "calendar_heatmap") or (not requested and advise(cols, rows).get("chart_type") == "calendar_heatmap")
            if _cal_needed and sess_data.get("last_sql"):
                _full = refetch_for_calendar(sess_data["last_sql"], orch, params=sess_data.get("last_sql_params", {}))
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
            summary = generate_summary(req.message, _result_cols, _result_rows)
            display_answer = summary if summary else clean_answer(result.answer, _result_rows)
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
                    rlabel = TYPE_LABELS.get(rtype, rtype.title())
                    alts   = ALT_MAP.get(rtype, [])[:2]
                    hint   = rec["reason"]
                    signal = f"\n📊__CHART_AVAILABLE__:{rtype}:{hint}"
                    alt_str = " or a ".join(f"**{TYPE_LABELS.get(a, a.title())}**" for a in alts)
                    if alts:
                        display_answer += f"{signal}\n\nI can visualize this as a **{rlabel}** ({hint}), or as a {alt_str} if you prefer — just say the word."
                    else:
                        display_answer += f"{signal}\n\nI can visualize this as a **{rlabel}** ({hint}) — just ask."
        else:
            display_answer = clean_answer(result.answer, _result_rows)

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
        _result_cols, _result_rows = strip_uuid_columns(_result_cols, _result_rows)
        display_answer = redact_security_internals(display_answer)
        reasoning = redact_security_internals(reasoning)

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


@router.post("/api/page")
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

        paged_sql = paginate_sql(sql, req.page, req.page_size)

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

        columns, rows = strip_uuid_columns(columns, rows)

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


@router.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str, token_data: dict = Depends(require_auth)):
    _sessions.pop(f"{session_id}:{token_data['sub']}", None)
    return {"status": "cleared", "session_id": session_id}


@router.get("/api/sql/{session_id}")
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


@router.get("/api/columns/{session_id}")
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
