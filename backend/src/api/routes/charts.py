"""
Chart routes — /api/chart.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.charts.advisor import advise, advise_full, force_config_full, get_alternatives
from src.security.auth import require_auth
from src.api.deps import get_orch_optional, _sessions
from src.api.schemas import ChartRequest, ChartConfig

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
    sample_rows = sess_data["last_raw_rows"]
    _chart_orch = get_orch_optional(request)
    _sql = sess_data.get("last_sql")
    _sql_params = sess_data.get("last_sql_params", {})
    _aggregate = not req.view_all

    # Column picker overrides — user explicitly selected which cols to chart
    lc_override = req.x_col
    num_cols_override = req.y_cols

    if req.chart_type:
        cfg, rows_total, rows_shown = force_config_full(
            cols, sample_rows, req.chart_type,
            sql=_sql, orch=_chart_orch, params=_sql_params,
            lc=lc_override, num_cols=num_cols_override, aggregate=_aggregate,
        )
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
        rec = advise(cols, sample_rows)
        rec_type = rec.get("chart_type")
        if req.chart_type == rec_type:
            forced_confidence = "high"
        elif any(req.chart_type in g and rec_type in g for g in _COMPAT if rec_type):
            forced_confidence = "medium"
        else:
            forced_confidence = "low"

        alts = get_alternatives(cols, sample_rows, n=3)
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
            rows_shown=rows_shown, rows_total=rows_total,
            is_aggregated=bool(rows_total and rows_shown and rows_shown < rows_total),
            can_view_all=bool(rows_total and rows_shown and rows_shown < rows_total),
        )
    else:
        rec = advise_full(
            cols, sample_rows,
            sql=_sql, orch=_chart_orch, params=_sql_params, aggregate=_aggregate,
        )
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
            rows_shown=rec.get("rows_shown"), rows_total=rec.get("rows_total"),
            is_aggregated=rec.get("is_aggregated", False),
            can_view_all=rec.get("can_view_all", False),
        )
