"""
Chart Advisor
=============
Analyses SQL result shape and recommends the best Chart.js chart type.

Two modes controlled by env var USE_LLM_CHART_ADVISOR:
  - "true"  → ask the local Ollama model (smarter, ~1-2s extra latency)
  - "false" → pure rule engine (instant, but brittle)

If LLM mode fails for any reason, silently falls back to rules.

Public API:
  advise(columns, rows)                 → dict
  force_config(columns, rows, type)     → dict | None
"""

from __future__ import annotations
import json
import os
import re
from typing import Any

# ── Colour palette ────────────────────────────────────────────────────────────
_PALETTE = [
    "rgba(99,102,241,0.85)",   # indigo
    "rgba(16,185,129,0.85)",   # emerald
    "rgba(245,158,11,0.85)",   # amber
    "rgba(239,68,68,0.85)",    # red
    "rgba(59,130,246,0.85)",   # blue
    "rgba(168,85,247,0.85)",   # purple
    "rgba(20,184,166,0.85)",   # teal
    "rgba(249,115,22,0.85)",   # orange
    "rgba(236,72,153,0.85)",   # pink
    "rgba(132,204,22,0.85)",   # lime
]
_BORDER = [c.replace("0.85", "1") for c in _PALETTE]
_GRID_COLOR  = "rgba(255,255,255,0.06)"
_TICK_COLOR  = "rgba(255,255,255,0.55)"
_LABEL_COLOR = "rgba(255,255,255,0.75)"

_VALID_TYPES = {"bar", "line", "pie", "doughnut", "scatter"}


# ── Column classifier ─────────────────────────────────────────────────────────

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


# ── Shared Chart.js options ───────────────────────────────────────────────────

def _scales() -> dict:
    axis = lambda: {
        "grid": {"color": _GRID_COLOR},
        "ticks": {"color": _TICK_COLOR},
    }
    return {"x": axis(), "y": axis()}

def _base_opts(legend: bool = True) -> dict:
    return {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "legend": {"display": legend, "labels": {"color": _LABEL_COLOR}},
            "tooltip": {"mode": "index", "intersect": False},
        },
    }

def _coerce_row_values(rows: list[dict], cols: list[str]) -> list:
    """Return numeric values for a column, coercing strings."""
    return [_coerce_numeric(row.get(cols[0])) for row in rows] if cols else []

def _datasets(rows, num_cols, line=False):
    out = []
    for i, col in enumerate(num_cols):
        c, b = _PALETTE[i % len(_PALETTE)], _BORDER[i % len(_BORDER)]
        vals = [_coerce_numeric(row.get(col)) for row in rows]
        ds: dict = {"label": _label(col), "data": vals,
                    "backgroundColor": c, "borderColor": b, "borderWidth": 1}
        if line:
            ds.update({"fill": False, "tension": 0.35, "pointRadius": 3, "pointHoverRadius": 5})
        out.append(ds)
    return out

def _bar(labels, num_cols, rows, horizontal=False):
    opts = {**_base_opts(len(num_cols) > 1), "scales": _scales()}
    if horizontal:
        opts["indexAxis"] = "y"
    return {"type": "bar", "data": {"labels": labels, "datasets": _datasets(rows, num_cols)}, "options": opts}

def _line(labels, num_cols, rows):
    opts = {**_base_opts(len(num_cols) > 1), "scales": _scales()}
    return {"type": "line", "data": {"labels": labels, "datasets": _datasets(rows, num_cols, line=True)}, "options": opts}

def _pie(labels, values, doughnut=False):
    n = len(labels)
    return {
        "type": "doughnut" if doughnut else "pie",
        "data": {"labels": labels, "datasets": [{
            "data": [_coerce_numeric(v) for v in values],
            "backgroundColor": _PALETTE[:n], "borderColor": _BORDER[:n], "borderWidth": 1,
        }]},
        "options": {**_base_opts(), "plugins": {
            "legend": {"position": "right", "labels": {"color": _LABEL_COLOR}},
            "tooltip": {},
        }},
    }

def _scatter(x_col, y_col, rows):
    return {
        "type": "scatter",
        "data": {"datasets": [{"label": f"{_label(x_col)} vs {_label(y_col)}",
            "data": [{"x": _coerce_numeric(row.get(x_col)),
                      "y": _coerce_numeric(row.get(y_col))} for row in rows],
            "backgroundColor": _PALETTE[0], "borderColor": _BORDER[0]}]},
        "options": {**_base_opts(False), "scales": _scales()},
    }

def _kpi_bar(num_cols, rows):
    labels = [_label(c) for c in num_cols]
    values = [_coerce_numeric(rows[0].get(c)) for c in num_cols]
    return {"type": "bar",
        "data": {"labels": labels, "datasets": [{"label": "Value", "data": values,
            "backgroundColor": _PALETTE[:len(labels)], "borderColor": _BORDER[:len(labels)], "borderWidth": 1}]},
        "options": {**_base_opts(False), "indexAxis": "y", "scales": _scales()}}


# ── Rule-based advisor ────────────────────────────────────────────────────────

def _rules_advise(columns: list[str], rows: list[dict]) -> dict[str, Any]:
    if not rows or not columns:
        return _no_chart("No data")

    t = _classify_cols(columns, rows)
    dates, nums, cats = t["date"], t["numeric"], t["category"]
    n = len(rows)

    if n == 1 and len(nums) >= 2:
        return _result("bar", f"Single-row summary with {len(nums)} metrics", "high",
                       None, nums, _kpi_bar(nums, rows))
    if dates and len(nums) == 1:
        lc = dates[0]
        return _result("line", f"Time-series on '{lc}'", "high",
                       lc, nums, _line([str(r.get(lc)) for r in rows], nums, rows))
    if dates and len(nums) >= 2:
        lc = dates[0]
        return _result("line", f"Multi-series time-series on '{lc}'", "high",
                       lc, nums, _line([str(r.get(lc)) for r in rows], nums[:6], rows))
    if cats and len(nums) == 1 and n <= 8:
        lc, vc = cats[0], nums[0]
        return _result("doughnut", f"Low-cardinality category ('{lc}', {n} slices)", "medium",
                       lc, nums, _pie([str(r.get(lc)) for r in rows], [r.get(vc) for r in rows], doughnut=True))
    if cats and len(nums) == 1:
        lc = cats[0]
        horiz = n > 12
        return _result("bar", f"Category '{lc}' vs numeric — {'horizontal ' if horiz else ''}bar", "high",
                       lc, nums, _bar([str(r.get(lc)) for r in rows], nums, rows, horizontal=horiz))
    if cats and 2 <= len(nums) <= 5:
        lc = cats[0]
        return _result("bar", f"Multiple metrics per '{lc}' — grouped bar", "high",
                       lc, nums, _bar([str(r.get(lc)) for r in rows], nums, rows))
    if len(nums) == 2 and not cats and not dates:
        return _result("scatter", "Two numeric columns — scatter plot", "medium",
                       None, nums, _scatter(nums[0], nums[1], rows))
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
    """Lazy-init the Ollama client, reusing env config."""
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

        # Strip markdown fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)

        # Extract first JSON object
        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON found in: {raw[:200]}")

        parsed = json.loads(m.group(0))
        chart_type = parsed.get("chart_type", "").lower().strip()
        reason     = parsed.get("reason", "LLM recommendation")
        confidence = parsed.get("confidence", "medium")

        if chart_type not in _VALID_TYPES:
            raise ValueError(f"Invalid chart type from LLM: {chart_type!r}")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        # Build the actual Chart.js config using rules engine
        rule_result = _rules_advise(columns, rows)
        t = _classify_cols(columns, rows)
        dates, nums, cats = t["date"], t["numeric"], t["category"]
        lc = cats[0] if cats else (dates[0] if dates else columns[0])
        vc = nums if nums else [columns[-1]]
        labels = [str(r.get(lc)) for r in rows]

        config = _build_config_for_type(chart_type, lc, vc, labels, rows)
        if config is None:
            config = rule_result.get("config")

        return {
            "chart_type": chart_type,
            "reason": reason,
            "confidence": confidence,
            "labels_column": lc,
            "values_columns": vc,
            "config": config,
        }

    except Exception as e:
        # Silent fallback to rules
        return _rules_advise(columns, rows)


def _build_config_for_type(chart_type, lc, num_cols, labels, rows):
    """Build Chart.js config for a specific type."""
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
    Main entry point. Uses LLM or rules based on USE_LLM_CHART_ADVISOR env var.
    NOTE: LLM mode adds ~15s latency per call. Only enable for offline/batch use.
    Default and recommended: rules mode (instant).
    """
    use_llm = os.getenv("USE_LLM_CHART_ADVISOR", "false").lower().strip() in ("1", "true", "yes")
    if use_llm:
        return _llm_advise(columns, rows)
    return _rules_advise(columns, rows)


def force_config(columns: list[str], rows: list[dict], chart_type: str) -> dict | None:
    """Generate a Chart.js config for a specific type regardless of recommendation."""
    if not rows or not columns or chart_type not in _VALID_TYPES:
        return None
    t = _classify_cols(columns, rows)
    dates, nums, cats = t["date"], t["numeric"], t["category"]
    lc = cats[0] if cats else (dates[0] if dates else columns[0])
    labels = [str(r.get(lc)) for r in rows]
    num_cols = nums if nums else [columns[-1]]
    cfg = _build_config_for_type(chart_type, lc, num_cols, labels, rows)
    return cfg or _bar(labels, num_cols, rows)