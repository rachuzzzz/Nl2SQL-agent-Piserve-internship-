"""
Chat, pagination, and session routes — /api/chat, /api/page,
/api/sessions/{id}, /api/sql/{id}, /api/columns/{id}.
"""

import asyncio
import logging
import re
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from src.charts.advisor import (
    advise, advise_full, force_config_full, classify_columns,
    build_kpi_cards, should_show_dashboard,
)
from src.security.auth import require_auth
from src.sql.result_shape import is_disambiguation_result, is_entity_disambiguation_result, extract_search_term
from src.security.rbac import DATA_VISIBILITY_PERMS, build_scope_filter
from src.agent.orchestrator.models import ConversationTurn
from src.api.intent_recap import start_intent_recap, resolve_intent_recap
from src.api.deps import (
    get_orch, get_orch_inspection, get_orch_audit, get_router,
    get_or_create_session, _sessions,
)
from src.api.response_formatting import (
    is_chart_followup, extract_requested_type, strip_uuid_columns,
    strip_row_identity_columns,
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

# A bare greeting is never a real data question worth carrying forward as a
# domain-clarification's "origin" — see the domain-pick carryover logic below.
_GREETING_RE = re.compile(r"^(hi|hello|hey|hiya|yo|sup|greetings)\b", re.IGNORECASE)

# A "bare domain pick" names ONLY the domain itself ("Audit", "the audit
# one", "i mean inspection") with no other content — meaningless standalone,
# so the ORIGINAL question gets re-run against it (see _is_pending_domain_pick
# below). A reply like "i mean mitigative actions" is NOT bare — "mitigative
# actions" is itself a Tier-1 routing keyword (see agent/domain_router.py)
# that's MORE specific than the vague original question ("Show all actions").
# Confirmed live: qid=08fa6c7d discarded "i mean mitigative actions" in favor
# of replaying "Show all actions" verbatim, so the reasoning LLM saw the same
# vague question a second time (plus a confusing echo of the router's
# inspection-vs-audit clarifying question in history) and invented a SECOND,
# unprompted clarifying question about action sub-types instead of just
# generating SQL — which it does correctly when given "mitigative actions"
# directly (see the working "Show all mitigative actions" turn in the same
# session). The old heuristic (word count <= 4) mistook "mitigative actions"
# for a bare pick purely because it was short.
_BARE_DOMAIN_PICK_RE = re.compile(
    r"^\s*(?:i\s+mean\s+|the\s+|it'?s\s+|that'?s\s+)?"
    r"(?:audit|audits|auditing|inspection|inspections|inspecting)"
    r"(?:\s+one)?\s*[.!]?\s*$",
    re.IGNORECASE,
)

# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest, token_data: dict = Depends(require_auth),
    orch_inspection=Depends(get_orch_inspection), orch_audit=Depends(get_orch_audit),
    router_=Depends(get_router),
):
    qid = uuid.uuid4().hex[:8]
    sub = token_data['sub']
    try:
        sess_key = f"{req.session_id}:{sub}"
        sess_data = get_or_create_session(sess_key)
        # Chart-followup fast-path below is a pure UI replay against the
        # LAST result — domain-agnostic (both orchestrators share one
        # db_engine, see api/app.py's lifespan) — default to inspection here;
        # real domain routing happens further below, before the LLM pipeline.
        orch = orch_inspection

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
            sample_rows = sess_data["last_raw_rows"]
            requested = extract_requested_type(req.message)
            _sql_params = sess_data.get("last_sql_params", {})

            if requested:
                # User explicitly typed /chart <type> — show it directly, no
                # conflict banner. force_config_full fetches a fuller dataset
                # (any type, not just calendar) before building/aggregating,
                # same flow every other chart-producing path now uses.
                req_cfg, rows_total, rows_shown = force_config_full(
                    cols, sample_rows, requested,
                    sql=sess_data.get("last_sql"), orch=orch, params=_sql_params,
                )
                rec = advise(cols, sample_rows)  # only for alternatives/x_label/y_labels
                if req_cfg is None:
                    req_cfg = rec["config"]
                    rows_total = rec.get("rows_total")
                    rows_shown = rec.get("rows_shown")
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
                    rows_shown=rows_shown, rows_total=rows_total,
                    is_aggregated=bool(rows_total and rows_shown and rows_shown < rows_total),
                    can_view_all=bool(rows_total and rows_shown and rows_shown < rows_total),
                )
                return ChatResponse(
                    answer="Here's the chart.",
                    sql=None, reasoning=None,
                    domain=sess_data["session"].last_domain or None,
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    time_ms=1, llm_calls=0, sql_errors=0, steps=[],
                    session_id=req.session_id,
                    result_data=None,
                    chart_config=cc,
                )

            rec = advise_full(
                cols, sample_rows,
                sql=sess_data.get("last_sql"), orch=orch, params=_sql_params,
            )
            if rec.get("chart_type"):
                cc = ChartConfig(
                    chart_type=rec["chart_type"], reason=rec["reason"],
                    confidence=rec["confidence"], config=rec["config"],
                    conflict=False,
                    requested_type=None,
                    requested_config=None,
                    alternatives=rec.get("alternatives", []),
                    x_label=rec.get("labels_column"),
                    y_labels=rec.get("values_columns") or [],
                    rows_shown=rec.get("rows_shown"), rows_total=rec.get("rows_total"),
                    is_aggregated=rec.get("is_aggregated", False),
                    can_view_all=rec.get("can_view_all", False),
                )
                return ChatResponse(
                    answer="Here's the chart.",
                    sql=None, reasoning=None,
                    domain=sess_data["session"].last_domain or None,
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    time_ms=1, llm_calls=0, sql_errors=0, steps=[],
                    session_id=req.session_id,
                    result_data=None,
                    chart_config=cc,
                )

        # ── Domain routing — 3-tier hybrid (keyword → session stickiness →
        # LLM judge), see agent/domain_router.py. Runs after the chart-followup
        # fast-path (irrelevant to domain) and before the LLM pipeline. ──────
        #
        # If the PREVIOUS turn was itself an unresolved domain clarification
        # (route.domain == "AMBIGUOUS"), a short reply now ("Audit", "the
        # audit one") is a DOMAIN PICK, not a new question — classify on the
        # reply text (Tier 1 resolves the bare domain word directly) but then
        # run the ORIGINAL question, not the bare pick. Without this, "list
        # all schedules" -> asked to clarify -> "Audit" -> the orchestrator
        # receives the literal string "Audit" as its question, which is
        # meaningless on its own and just asks the user to clarify AGAIN —
        # confirmed live: qid=94c31bf3 did exactly this. Mirrors the existing
        # disambiguation_origin_question pattern used for people/entity name
        # picks (see agent/orchestrator/session.py's _origin_note).
        _sess_obj = sess_data["session"]
        _prev = _sess_obj.last_turn
        # Only carry the origin forward when it actually LOOKS like a data
        # question — confirmed live: "Hi" -> Tier 3 correctly said AMBIGUOUS
        # (reasonable for a bare greeting) -> user replied "audits" -> this
        # carryover fired and substituted "Hi" back in as the question,
        # so the orchestrator just re-greeted the user instead of treating
        # "audits" as their actual (if vague) request. A greeting is never
        # worth re-running verbatim, unlike a genuine question like "list
        # all schedules" that just needed a domain picked for it.
        _origin_is_real_question = bool(
            _prev and not _GREETING_RE.match(_prev.question.strip())
            and len(_prev.question.strip().split()) >= 2
        )
        _is_pending_domain_pick = bool(
            _prev and _prev.tool_used == "domain_clarification"
            and _BARE_DOMAIN_PICK_RE.match(req.message.strip())
            and _origin_is_real_question
        )
        # Tier 1/2/3 always classify on the REPLY text itself (Tier 1 needs
        # to see the actual domain word, e.g. "Audit") — only the message
        # actually sent to the orchestrator gets swapped for the origin
        # question, once resolved.
        route = router_.classify(req.message, session=_sess_obj)
        log_query.info("[ROUTER] qid=%s tier=%d domain=%s confidence=%.2f%s",
                        qid, route.tier, route.domain, route.confidence,
                        " (origin-question carryover)" if _is_pending_domain_pick and route.domain != "AMBIGUOUS" else "")
        if route.domain == "AMBIGUOUS":
            _sess_obj.add_turn(ConversationTurn(
                question=req.message, answer=route.clarifying_question,
                tool_used="domain_clarification",
            ))
            return ChatResponse(
                answer=route.clarifying_question,
                sql=None, reasoning=None,
                domain=None,  # ambiguous — no orchestrator ran, nothing accurate to label
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                time_ms=0, llm_calls=0, sql_errors=0, steps=[],
                session_id=req.session_id,
                result_data=None,
            )
        orch = orch_audit if route.domain == "AUDIT" else orch_inspection
        _effective_message = _prev.question if _is_pending_domain_pick else req.message

        # A message that's STILL just a bare domain word after the carryover
        # swap above means there was no real question to resume (the prior
        # turn was a bare greeting, or this is a session's very first
        # message) — sending "audit" to the reasoning model as if it were a
        # real question left the outcome to model judgment, and confirmed
        # live that judgment isn't consistent: the SAME bare "audit" input
        # produced two different real outcomes across two sessions — once a
        # fast clarifying question (~5s, 1 LLM call), once an unrequested
        # default "recent audit reports" query (~16.5s, 3 LLM calls, a
        # validation retry). Short-circuit deterministically instead: just
        # acknowledge the domain and ask what they'd like to know, with no
        # LLM call at all — cheap, fast, and never wrong.
        if _BARE_DOMAIN_PICK_RE.match(_effective_message.strip()):
            _domain_label = "audits" if orch.domain == "audit" else "facility inspections"
            _ack = f"Got it — {_domain_label}. What would you like to know?"
            _sess_obj.add_turn(ConversationTurn(
                question=req.message, answer=_ack,
                tool_used="domain_ack", domain=orch.domain,
            ))
            return ChatResponse(
                answer=_ack,
                sql=None, reasoning=None,
                domain=orch.domain,
                intent_recap=None,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                time_ms=0, llm_calls=0, sql_errors=0, steps=[],
                session_id=req.session_id,
                result_data=None,
            )

        # Fresh every request — set BEFORE orch.query() so build_context_block()
        # (called inside the reasoning loop) can prime the LLM to call
        # resolve_selected_rows. Defaults to [] when the client selected
        # nothing this turn; never explicitly cleared elsewhere since this
        # always overwrites whatever the previous request left behind.
        sess_data["session"].pending_selected_positions = req.selected_positions or []

        # Fired now, resolved after orch.query() returns (below) — runs
        # concurrently with the SQL-gen/execution loop so its latency
        # overlaps instead of adding on top. See api/intent_recap.py.
        _recap_future = start_intent_recap(
            orch.reasoning_llm, _effective_message, sess_data["session"],
        )

        # Only trust the router's decision as authoritative for THIS message
        # when it resolved via Tier 1 (an unambiguous domain keyword actually
        # present in the classified text) AND that's the same text being sent
        # to the orchestrator — the domain-pick carryover path above swaps in
        # the ORIGIN question, whose wording the router never actually saw
        # (that's precisely why it needed clarification in the first place).
        _router_locked_domain = route.tier == 1 and not _is_pending_domain_pick

        t0 = time.time()
        result = await asyncio.to_thread(
            orch.query, _effective_message, session=sess_data["session"],
            visibility_scope=visibility_scope, query_id=qid,
            router_locked_domain=_router_locked_domain,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        sess_data["message_count"] += 1
        intent_recap = resolve_intent_recap(_recap_future, _effective_message, sess_data["session"])

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
                    # Strip __id columns injected by identity.py (see
                    # inject_row_identity()) before caching — this cache feeds
                    # the chart advisor and the chart-followup fast-path, not
                    # just the response, so it must be clean at the source.
                    _clean_cols, _clean_rows = strip_row_identity_columns(
                        list(rows[0].keys()), rows
                    )
                    sess_data["last_raw_rows"] = _clean_rows  # cache for fast-path
                    sess_data["last_cols"] = _clean_cols
                    _result_rows = [{k: str(v) for k, v in row.items()} for row in _clean_rows]
                    _result_cols = _clean_cols
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
                    _ga_cols_raw = inner.get("columns", list(rows[0].keys()))
                    _clean_cols, _clean_rows = strip_row_identity_columns(_ga_cols_raw, rows)
                    sess_data["last_raw_rows"] = _clean_rows
                    sess_data["last_cols"] = _clean_cols
                    _result_rows = [{k: str(v) for k, v in row.items()} for row in _clean_rows]
                    _result_cols = _clean_cols
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

            # Two-phase fetch: this is the previously-missing wiring — the
            # calendar-only refetch_for_calendar bypass used to skip this path
            # entirely, so a calendar chart embedded right after the initial
            # answer could look sparser than the same chart requested a moment
            # later via the chart-followup path below (which DID refetch).
            # advise_full now fetches a fuller dataset for every chart type
            # that benefits from it, consistently across all three paths.
            rec = advise_full(
                _result_cols, raw_for_chart,
                sql=sess_data.get("last_sql"), orch=orch,
                params=sess_data.get("last_sql_params", {}),
            )
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
                    rows_shown=rec.get("rows_shown"), rows_total=rec.get("rows_total"),
                    is_aggregated=rec.get("is_aggregated", False),
                    can_view_all=rec.get("can_view_all", False),
                )
                # Use the fuller fetch for KPI sums/averages when advise_full
                # got one — otherwise a card like "Total CAPEX" would silently
                # report a partial sum over just the 50-row sample as if it
                # were the true total.
                _dashboard = DashboardData(
                    show=True,
                    cards=build_kpi_cards(_result_cols, rec.get("_full_rows") or raw_for_chart),
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
                    rows_shown=_pl_rec.get("rows_shown"), rows_total=_pl_rec.get("rows_total"),
                    is_aggregated=_pl_rec.get("is_aggregated", False),
                    can_view_all=_pl_rec.get("can_view_all", False),
                )
                _dashboard = _dashboard or DashboardData(show=False, cards=[])

        # Scrub raw UUID columns (user_id, facility_id, ...) right before the
        # table reaches the frontend — done last so it never affects chart/KPI
        # column selection or summary generation upstream, only the payload.
        # __id columns are already stripped at cache time above (the
        # execute_sql/get_answers branches) — this is defense-in-depth only,
        # in case some other path populated _result_cols/_result_rows without
        # going through that cache step.
        _result_cols, _result_rows = strip_row_identity_columns(_result_cols, _result_rows)
        _result_cols, _result_rows = strip_uuid_columns(_result_cols, _result_rows)
        display_answer = redact_security_internals(display_answer)
        reasoning = redact_security_internals(reasoning)
        intent_recap = redact_security_internals(intent_recap)

        log_query.info(
            "[QUERY] END   qid=%s sub=%s elapsed=%dms llm_calls=%d "
            "sql_errors=%d rows=%d tokens=%d/%d cache_read=%d cache_write=%d success=%s",
            qid, sub, elapsed_ms,
            result.stats.total_llm_calls,
            sql_errors,
            _row_count,
            result.stats.total_prompt_tokens,
            result.stats.total_completion_tokens,
            result.stats.total_cache_read_tokens,
            result.stats.total_cache_creation_tokens,
            result.success,
        )

        # Ground-truth flag, not a frontend heuristic — true only when this
        # turn's rows actually made it into row_titles (see synthesis.py's
        # turn-finish capture), which is exactly what resolve_selected_rows
        # needs to resolve a checkbox selection against. get_answer_stats
        # tables (never captured) correctly come back False.
        _last_turn = sess_data["session"].last_turn
        _row_selectable = bool(_last_turn and _last_turn.row_titles)

        return ChatResponse(
            answer=display_answer,
            sql=sql,
            reasoning=reasoning,
            domain=orch.domain,
            intent_recap=intent_recap,
            prompt_tokens=result.stats.total_prompt_tokens,
            completion_tokens=result.stats.total_completion_tokens,
            total_tokens=result.stats.total_prompt_tokens + result.stats.total_completion_tokens,
            cache_read_tokens=result.stats.total_cache_read_tokens,
            cache_creation_tokens=result.stats.total_cache_creation_tokens,
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
                row_selectable=_row_selectable,
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

        # This path re-executes the cached SQL directly (line 611) — which is
        # the already identity-injected SQL (see identity.py), so __id
        # columns are fresh here and bypass the cache-time strip in
        # /api/chat. Must strip before strip_uuid_columns' own value-based
        # scrub, same as there — by name, not by sniffing the value.
        columns, rows = strip_row_identity_columns(columns, rows)
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
            # Always False here — this path re-executes the cached SQL
            # directly against the DB and never touches session.turns, so
            # there's no row_titles/row_ids capture to resolve a checkbox
            # selection against (see ResultData.row_selectable's docstring).
            "row_selectable": False,
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
