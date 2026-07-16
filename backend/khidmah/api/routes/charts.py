"""
Chart routes — /api/chart.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from khidmah.charts.advisor import advise, force_config, get_alternatives
from khidmah.core.auth import require_auth
from khidmah.api.deps import get_orch_optional, refetch_for_calendar, _sessions
from khidmah.api.schemas import ChartRequest, ChartConfig

router = APIRouter()


@router.post("/api/chart")
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
        _full = refetch_for_calendar(sess_data["last_sql"], _chart_orch, params=sess_data.get("last_sql_params", {}))
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
