"""
Chart Advisor
=============
Analyses SQL result shape and recommends the best Apache ECharts option object.

Two modes controlled by env var USE_LLM_CHART_ADVISOR:
  - "true"  → ask the local Ollama model (smarter, ~1-2s extra latency)
  - "false" → pure rule engine (instant, recommended default)

If LLM mode fails for any reason, silently falls back to rules.

Public API  (unchanged from Chart.js version — fastapi_app.py needs no edits):
  advise(columns, rows)              → dict  (config is an ECharts option object)
  force_config(columns, rows, type)  → dict | None

Config format:
  The "config" key in every returned dict is a plain ECharts option dict.
  Pass it directly to  chart.setOption(config).
  A "_lib": "echarts" sentinel is included so the frontend can assert the library.
"""

from __future__ import annotations
import json
import os
import re
from typing import Any


# ── Colour palette (hex — ECharts prefers hex over rgba) ─────────────────────
_PALETTE = [
    "#6366f1",   # indigo
    "#10b981",   # emerald
    "#f59e0b",   # amber
    "#ef4444",   # red
    "#3b82f6",   # blue
    "#a855f7",   # purple
    "#14b8a6",   # teal
    "#f97316",   # orange
    "#ec4899",   # pink
    "#84cc16",   # lime
]

# Theme colours
_GRID_COLOR  = "rgba(255,255,255,0.06)"
_TICK_COLOR  = "#888"
_LABEL_COLOR = "#bbb"
_AXIS_LINE   = "#444"

_VALID_TYPES = {"bar", "line", "pie", "doughnut", "scatter"}


# ── Column classifier (unchanged) ─────────────────────────────────────────────

_DATE_NAME_RE = re.compile(
    r'\b(date|month|year|week|time|period|quarter|day|created|modified|submitted|reported)\b',
    re.IGNORECASE,
)

def _is_date_col(col: str, values: list) -> bool:
    if _DATE_NAME_RE.search(col):
        return True
    sample = [str(v) for v in values[:5] if v is not None]
    return any(re.match(r'\d{4}[-/]\d{2}', s) for s in sample)

def _coerce_numeric(v) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(',', '').strip())
    except Exception:
        return None

def _is_numeric(values: list) -> bool:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return False
    return all(_coerce_numeric(v) is not None for v in non_null[:10])

def _classify_cols(columns: list[str], rows: list[dict]) -> dict:
    result: dict[str, list] = {"date": [], "numeric": [], "category": []}
    for col in columns:
        vals = [row.get(col) for row in rows]
        if _is_date_col(col, vals):
            result["date"].append(col)
        elif _is_numeric(vals):
            result["numeric"].append(col)
        else:
            result["category"].append(col)
    return result

def _label(col: str) -> str:
    return col.replace("_", " ").title()


# ── Shared ECharts building blocks ────────────────────────────────────────────

def _cat_axis(data: list, rotate: int = 0) -> dict:
    """Category axis (string labels)."""
    ax: dict = {
        "type": "category",
        "data": data,
        "axisLabel": {"color": _TICK_COLOR, "fontSize": 12},
        "axisLine":  {"lineStyle": {"color": _AXIS_LINE}},
        "axisTick":  {"lineStyle": {"color": _AXIS_LINE}},
    }
    if rotate:
        ax["axisLabel"]["rotate"] = rotate
    return ax

def _val_axis() -> dict:
    """Value axis (numeric)."""
    return {
        "type": "value",
        "axisLabel":  {"color": _TICK_COLOR, "fontSize": 12},
        "axisLine":   {"lineStyle": {"color": _AXIS_LINE}},
        "splitLine":  {"lineStyle": {"color": _GRID_COLOR}},
    }

def _grid(left: str = "2%", right: str = "2%", top: str = "60px", bottom: str = "70px") -> dict:
    return {"left": left, "right": right, "top": top, "bottom": bottom, "containLabel": True}

def _tooltip_axis() -> dict:
    return {"trigger": "axis", "axisPointer": {"type": "shadow"}}

def _tooltip_item() -> dict:
    return {"trigger": "item", "formatter": "{b}: {c} ({d}%)"}

def _legend(names: list[str], show: bool = True) -> dict:
    return {
        "show": show,
        "data": names,
        "top": "8px",
        "left": "center",
        "orient": "horizontal",
        "itemGap": 20,
        "itemWidth": 14,
        "itemHeight": 8,
        "textStyle": {"color": _LABEL_COLOR, "fontSize": 12},
        "pageTextStyle": {"color": _LABEL_COLOR},
    }

def _base(extra: dict | None = None) -> dict:
    """Sentinel + palette, merged with any extra top-level keys."""
    out: dict = {"_lib": "echarts", "color": _PALETTE}
    if extra:
        out.update(extra)
    return out


# ── ECharts builders ──────────────────────────────────────────────────────────

def _bar(labels: list, num_cols: list[str], rows: list[dict],
         horizontal: bool = False) -> dict:
    """Vertical or horizontal grouped bar."""
    series = [
        {
            "name": _label(col),
            "type": "bar",
            "data": [_coerce_numeric(r.get(col)) for r in rows],
            "barMaxWidth": 60,
            "emphasis": {"focus": "series"},
        }
        for col in num_cols
    ]
    rotate = 30 if (not horizontal and len(labels) > 12) else 0

    if horizontal:
        # Swap axes for horizontal layout
        opt: dict = {
            "tooltip":  _tooltip_axis(),
            "legend":   _legend([_label(c) for c in num_cols], show=len(num_cols) > 1),
            "grid":     _grid(left="22%", top="16px", bottom="24px"),
            "xAxis":    _val_axis(),
            "yAxis":    _cat_axis(labels),
            "series":   series,
        }
    else:
        opt = {
            "tooltip":  _tooltip_axis(),
            "legend":   _legend([_label(c) for c in num_cols], show=len(num_cols) > 1),
            "grid":     _grid(),
            "xAxis":    _cat_axis(labels, rotate=rotate),
            "yAxis":    _val_axis(),
            "series":   series,
        }
    return _base(opt)


def _line(labels: list, num_cols: list[str], rows: list[dict]) -> dict:
    """Smooth multi-series line chart."""
    series = [
        {
            "name":       _label(col),
            "type":       "line",
            "smooth":     True,
            "data":       [_coerce_numeric(r.get(col)) for r in rows],
            "symbolSize": 6,
            "emphasis":   {"focus": "series"},
        }
        for col in num_cols
    ]
    return _base({
        "tooltip": _tooltip_axis(),
        "legend":  _legend([_label(c) for c in num_cols], show=len(num_cols) > 1),
        "grid":    _grid(),
        "xAxis":   _cat_axis(labels),
        "yAxis":   _val_axis(),
        "series":  series,
    })


def _pie(labels: list, values: list, doughnut: bool = False) -> dict:
    """Pie or doughnut chart."""
    data = [
        {"name": str(lbl), "value": _coerce_numeric(val)}
        for lbl, val in zip(labels, values)
        if _coerce_numeric(val) is not None
    ]
    radius = ["38%", "68%"] if doughnut else "65%"
    series_extra: dict = {}
    if doughnut:
        series_extra["label"] = {"show": True, "formatter": "{b}\n{d}%", "color": _LABEL_COLOR}
    else:
        series_extra["label"] = {"show": True, "formatter": "{b}: {d}%", "color": _LABEL_COLOR}

    return _base({
        "tooltip": _tooltip_item(),
        "legend":  {
            "orient":    "vertical",
            "right":     "3%",
            "top":       "middle",
            "itemGap":   12,
            "itemWidth":  12,
            "itemHeight": 12,
            "textStyle": {"color": _LABEL_COLOR, "fontSize": 12},
        },
        "series": [{
            "type":      "pie",
            "radius":    radius,
            "center":    ["40%", "50%"],
            "data":      data,
            "emphasis":  {
                "itemStyle": {
                    "shadowBlur":    10,
                    "shadowOffsetX": 0,
                    "shadowColor":   "rgba(0,0,0,0.5)",
                }
            },
            **series_extra,
        }],
    })


def _scatter(x_col: str, y_col: str, rows: list[dict]) -> dict:
    """Scatter plot."""
    data = [
        [_coerce_numeric(r.get(x_col)), _coerce_numeric(r.get(y_col))]
        for r in rows
        if _coerce_numeric(r.get(x_col)) is not None and _coerce_numeric(r.get(y_col)) is not None
    ]
    return _base({
        "tooltip": {
            "trigger":   "item",
            "formatter": (
                f"{_label(x_col)}: {{c[0]}}<br/>{_label(y_col)}: {{c[1]}}"
            ),
        },
        "grid":   _grid(),
        "xAxis":  {**_val_axis(), "name": _label(x_col),
                   "nameTextStyle": {"color": _LABEL_COLOR}},
        "yAxis":  {**_val_axis(), "name": _label(y_col),
                   "nameTextStyle": {"color": _LABEL_COLOR}},
        "series": [{
            "type":       "scatter",
            "data":       data,
            "symbolSize": 9,
            "emphasis":   {"focus": "self"},
        }],
    })


def _kpi_bar(num_cols: list[str], rows: list[dict]) -> dict:
    """Single-row KPI summary as a horizontal bar with per-bar colours."""
    data = [
        {"value": _coerce_numeric(rows[0].get(col)),
         "name":  _label(col),
         "itemStyle": {"color": _PALETTE[i % len(_PALETTE)]}}
        for i, col in enumerate(num_cols)
    ]
    return _base({
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid":    _grid(left="20%"),
        "xAxis":   _val_axis(),
        "yAxis":   _cat_axis([_label(c) for c in num_cols]),
        "series":  [{"type": "bar", "data": data, "barMaxWidth": 50}],
    })


# ── Rule-based advisor ────────────────────────────────────────────────────────

def _rules_advise(columns: list[str], rows: list[dict]) -> dict[str, Any]:
    if not rows or not columns:
        return _no_chart("No data")

    t = _classify_cols(columns, rows)
    dates, nums, cats = t["date"], t["numeric"], t["category"]
    n = len(rows)

    # Single-row KPI summary
    if n == 1 and len(nums) >= 2:
        return _result("bar", f"Single-row summary with {len(nums)} metrics", "high",
                       None, nums, _kpi_bar(nums, rows))

    # Time-series
    if dates and len(nums) == 1:
        lc = dates[0]
        return _result("line", f"Time-series on '{lc}'", "high",
                       lc, nums, _line([str(r.get(lc)) for r in rows], nums, rows))
    if dates and len(nums) >= 2:
        lc = dates[0]
        return _result("line", f"Multi-series time-series on '{lc}'", "high",
                       lc, nums, _line([str(r.get(lc)) for r in rows], nums[:6], rows))

    # Low-cardinality category → doughnut
    if cats and len(nums) == 1 and n <= 8:
        lc, vc = cats[0], nums[0]
        return _result("doughnut", f"Low-cardinality category ('{lc}', {n} slices)", "medium",
                       lc, nums, _pie([str(r.get(lc)) for r in rows],
                                      [r.get(vc) for r in rows], doughnut=True))

    # Single numeric → bar (horizontal when many rows)
    if cats and len(nums) == 1:
        lc = cats[0]
        horiz = n > 12
        return _result("bar", f"Category '{lc}' vs numeric — {'horizontal ' if horiz else ''}bar",
                       "high", lc, nums,
                       _bar([str(r.get(lc)) for r in rows], nums, rows, horizontal=horiz))

    # Multiple numerics → grouped bar
    if cats and 2 <= len(nums) <= 5:
        lc = cats[0]
        return _result("bar", f"Multiple metrics per '{lc}' — grouped bar", "high",
                       lc, nums, _bar([str(r.get(lc)) for r in rows], nums, rows))

    # Two numerics, no category/date → scatter
    if len(nums) == 2 and not cats and not dates:
        return _result("scatter", "Two numeric columns — scatter plot", "medium",
                       None, nums, _scatter(nums[0], nums[1], rows))

    # Generic fallback
    if nums:
        lc = cats[0] if cats else (dates[0] if dates else columns[0])
        return _result("bar", "Generic bar chart", "low",
                       lc, nums[:4], _bar([str(r.get(lc)) for r in rows], nums[:4], rows))

    return _no_chart("No numeric columns found")


def _result(chart_type, reason, confidence, lc, vcs, config):
    return {"chart_type": chart_type, "reason": reason, "confidence": confidence,
            "labels_column": lc, "values_columns": vcs, "config": config}

def _no_chart(reason):
    return {"chart_type": None, "reason": reason, "confidence": "low",
            "labels_column": None, "values_columns": [], "config": None}


# ── LLM-based advisor ─────────────────────────────────────────────────────────

_llm_client = None

def _get_llm():
    global _llm_client
    if _llm_client is None:
        from llama_index.llms.ollama import Ollama
        url   = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
        _llm_client = Ollama(model=model, base_url=url, request_timeout=15.0)
    return _llm_client

def _llm_advise(columns: list[str], rows: list[dict]) -> dict[str, Any]:
    """Ask the LLM for the best chart type. Falls back to rules on any failure."""
    try:
        sample = rows[:5]
        prompt = f"""You are a data visualization expert. Given this dataset, choose the BEST chart type.

Columns: {json.dumps(columns)}
Sample rows (first 5 of {len(rows)}): {json.dumps(sample, default=str)}

Available chart types: bar, line, pie, doughnut, scatter

Rules:
- line: for time-series data (dates, months, years, periods)
- bar: for comparing categories, rankings, or multiple metrics
- pie/doughnut: only for part-of-whole with ≤8 categories
- scatter: only for correlation between two numeric variables
- If in doubt, prefer bar

Respond ONLY with valid JSON, no explanation, no markdown:
{{"chart_type": "line", "reason": "monthly time series data", "confidence": "high"}}

confidence must be one of: high, medium, low"""

        raw = str(_get_llm().complete(prompt)).strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON found in: {raw[:200]}")

        parsed     = json.loads(m.group(0))
        chart_type = parsed.get("chart_type", "").lower().strip()
        reason     = parsed.get("reason", "LLM recommendation")
        confidence = parsed.get("confidence", "medium")

        if chart_type not in _VALID_TYPES:
            raise ValueError(f"Invalid chart type from LLM: {chart_type!r}")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        # Build the ECharts option via rules
        rule_result = _rules_advise(columns, rows)
        t = _classify_cols(columns, rows)
        dates, nums, cats = t["date"], t["numeric"], t["category"]
        lc      = cats[0] if cats else (dates[0] if dates else columns[0])
        vc      = nums if nums else [columns[-1]]
        labels  = [str(r.get(lc)) for r in rows]

        config = _build_config_for_type(chart_type, lc, vc, labels, rows)
        if config is None:
            config = rule_result.get("config")

        return {
            "chart_type":    chart_type,
            "reason":        reason,
            "confidence":    confidence,
            "labels_column": lc,
            "values_columns": vc,
            "config":        config,
        }

    except Exception:
        return _rules_advise(columns, rows)


def _build_config_for_type(chart_type: str, lc, num_cols: list,
                            labels: list, rows: list) -> dict | None:
    """Build an ECharts option for a specific chart type."""
    if chart_type == "bar":
        return _bar(labels, num_cols, rows, horizontal=len(rows) > 12)
    if chart_type == "line":
        return _line(labels, num_cols, rows)
    if chart_type in ("pie", "doughnut"):
        vc = num_cols[0] if num_cols else None
        if not vc:
            return None
        vals = [row.get(vc) for row in rows]
        return _pie(labels, vals, doughnut=(chart_type == "doughnut"))
    if chart_type == "scatter" and len(num_cols) >= 2:
        return _scatter(num_cols[0], num_cols[1], rows)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def advise(columns: list[str], rows: list[dict]) -> dict[str, Any]:
    """
    Main entry point. Returns an ECharts option under the "config" key.
    Uses LLM or rules based on USE_LLM_CHART_ADVISOR env var (default: rules).
    """
    use_llm = os.getenv("USE_LLM_CHART_ADVISOR", "false").lower().strip() in ("1", "true", "yes")
    if use_llm:
        return _llm_advise(columns, rows)
    return _rules_advise(columns, rows)


def force_config(columns: list[str], rows: list[dict], chart_type: str) -> dict | None:
    """Generate an ECharts option for a specific type regardless of recommendation."""
    if not rows or not columns or chart_type not in _VALID_TYPES:
        return None
    t = _classify_cols(columns, rows)
    dates, nums, cats = t["date"], t["numeric"], t["category"]
    lc       = cats[0] if cats else (dates[0] if dates else columns[0])
    labels   = [str(r.get(lc)) for r in rows]
    num_cols = nums if nums else [columns[-1]]
    cfg = _build_config_for_type(chart_type, lc, num_cols, labels, rows)
    return cfg or _bar(labels, num_cols, rows)