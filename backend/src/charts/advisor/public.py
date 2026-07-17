"""
Public chart-advisor API: advise(), force_config(), get_alternatives(),
plus the LLM-based advisor path. Extracted verbatim from the original
charts/advisor.py.
"""

from __future__ import annotations
import json
import re

from src.charts.advisor.classify import _classify, _coerce, _VALID_TYPES, _TYPE_LABELS, _ALT_MAP
from src.charts.advisor.common import _result, _no_chart, _lbl
from src.charts.advisor.renderers import (
    _bar, _stacked_bar, _line, _pie, _scatter, _radar, _heatmap, _waterfall,
    _radar_pivot, _nightingale, _funnel, _gauge, _calendar_heatmap, _kpi_bar,
)
from src.charts.advisor.rules import _rules_advise




# ── Alternative generator ─────────────────────────────────────────────────────

def _build_config(chart_type, lc, num_cols, labels, rows):
    """Build ECharts config for a given type. Returns None if not applicable."""
    if chart_type == "bar":
        return _bar(labels, num_cols, rows, horizontal=False)
    if chart_type == "horizontal_bar":
        return _bar(labels, num_cols, rows, horizontal=True)
    if chart_type == "stacked_bar":
        return _stacked_bar(labels, num_cols, rows)
    if chart_type == "line":
        return _line(labels, num_cols, rows)
    if chart_type == "area":
        return _line(labels, num_cols, rows, area=True)
    if chart_type in ("pie", "doughnut"):
        vc = num_cols[0] if num_cols else None
        if not vc: return None
        return _pie(labels, [r.get(vc) for r in rows], doughnut=(chart_type=="doughnut"))
    if chart_type == "nightingale":
        vc = num_cols[0] if num_cols else None
        if not vc: return None
        return _nightingale(labels, [r.get(vc) for r in rows])
    if chart_type == "funnel":
        vc = num_cols[0] if num_cols else None
        if not vc: return None
        return _funnel(labels, [r.get(vc) for r in rows], vc, sort_order="descending")
    if chart_type == "scatter" and len(num_cols) >= 2:
        return _scatter(num_cols[0], num_cols[1], rows)
    if chart_type == "radar":
        if len(num_cols) == 1:
            return _radar_pivot(lc, num_cols[0], rows)
        if len(num_cols) >= 2:
            return _radar(lc, num_cols[:6], rows)
    if chart_type == "heatmap":
        t = _classify(list({lc} | set(num_cols)) if lc else num_cols, rows)
        all_dims = t["category"] + t["date"]
        if len(all_dims) >= 2 and t["numeric"]:
            return _heatmap(all_dims[0], all_dims[1], t["numeric"][0], rows)
    if chart_type == "waterfall" and num_cols:
        # Single-row P&L: column names become the X-axis steps, values come from the one row.
        if len(rows) == 1 and len(num_cols) >= 2:
            metric_labels = [_lbl(c) for c in num_cols]
            metric_vals   = [_coerce(rows[0].get(c)) or 0 for c in num_cols]
            return _waterfall(metric_labels, metric_vals, num_cols[0])
        # Multi-row: one bar per row using the label column and first numeric column.
        vals = [_coerce(r.get(num_cols[0])) or 0 for r in rows]
        return _waterfall(labels, vals, num_cols[0])
    if chart_type == "calendar_heatmap":
        vc = num_cols[0] if num_cols else None
        if lc:
            return _calendar_heatmap(lc, vc, rows)
        return None
    if chart_type == "gauge" and num_cols and rows:
        v = _coerce(rows[0].get(num_cols[0])) or 0
        return _gauge(v, _lbl(num_cols[0]))
    return None



def get_alternatives(columns, rows, n=3):
    """
    Return up to n alternative chart configs for a dataset.
    Used by the frontend to show "Also try" options.
    """
    primary = _rules_advise(columns, rows)
    if not primary["chart_type"]:
        return []

    t = _classify(columns, rows)
    dates, nums, cats = t["date"], t["numeric"], t["category"]
    lc     = primary["labels_column"] or (cats[0] if cats else (dates[0] if dates else columns[0]))
    labels = [str(r.get(lc)) for r in rows] if lc else []
    nc     = primary["values_columns"] or nums

    alts = []
    for alt_type in _ALT_MAP.get(primary["chart_type"], [])[:n]:
        cfg = _build_config(alt_type, lc, nc, labels, rows)
        if cfg:
            alts.append({
                "chart_type": alt_type,
                "label": _TYPE_LABELS.get(alt_type, alt_type.title()),
                "config": cfg,
            })
    return alts



# ── LLM advisor ───────────────────────────────────────────────────────────────

_llm_client = None


def _get_llm():
    global _llm_client
    if _llm_client is None:
        from src.llm.factory import get_chart_llm
        from src.config.settings import get_settings
        settings = get_settings()
        _llm_client = get_chart_llm(settings.ollama_chat_model, settings.ollama_url)
    return _llm_client


def _llm_advise(columns, rows):
    try:
        sample = rows[:5]
        prompt = f"""You are a data visualization expert. Given this dataset, choose the BEST chart type.

Columns: {json.dumps(columns)}
Sample rows (first 5 of {len(rows)}): {json.dumps(sample, default=str)}

Available chart types: bar, horizontal_bar, stacked_bar, line, area, pie, doughnut, scatter, radar, heatmap, waterfall, gauge

Rules:
- line/area: time-series. area for cumulative/volume metrics.
- bar: comparisons with 1-2 metrics. horizontal_bar when >10 rows or long labels.
- stacked_bar: multiple metrics composing a total (capex+opex, pass+fail).
- pie/doughnut: part-of-whole, ≤8 categories. Prefer doughnut.
- scatter: correlation between exactly two numerics.
- radar: compare 3+ metrics across ≤8 entities.
- heatmap: two categorical dimensions × one numeric.
- waterfall: sequential P&L, or mixed positive/negative values.
- gauge: for a single scalar KPI (score, rate, percentage) with a value in 0-100 range.
- Default to bar or horizontal_bar when unsure.

Respond ONLY with JSON: {{"chart_type":"line","reason":"monthly trend","confidence":"high"}}"""

        raw = str(_get_llm().complete(prompt)).strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not m: raise ValueError("No JSON")

        parsed = json.loads(m.group(0))
        ct     = parsed.get("chart_type","").lower().strip()
        reason = parsed.get("reason","LLM recommendation")
        conf   = parsed.get("confidence","medium")

        if ct not in _VALID_TYPES: raise ValueError(f"Invalid: {ct}")
        if conf not in ("high","medium","low"): conf = "medium"

        rule = _rules_advise(columns, rows)
        t    = _classify(columns, rows)
        lc   = t["category"][0] if t["category"] else (t["date"][0] if t["date"] else columns[0])
        nc   = t["numeric"] or [columns[-1]]
        lbls = [str(r.get(lc)) for r in rows]

        cfg = _build_config(ct, lc, nc, lbls, rows)
        if cfg is None: cfg = rule.get("config")

        return {"chart_type":ct,"reason":reason,"confidence":conf,
                "labels_column":lc,"values_columns":nc,"config":cfg}
    except Exception:
        return _rules_advise(columns, rows)



def advise(columns, rows):
    """
    Primary entry point. Returns recommendation + alternatives.

    Response includes:
      chart_type, reason, confidence, config,
      labels_column, values_columns,
      alternatives: list of {chart_type, label, config}
    """
    from src.config.settings import get_settings
    use_llm = get_settings().use_llm_chart_advisor
    result  = _llm_advise(columns, rows) if use_llm else _rules_advise(columns, rows)

    # Always attach alternatives (built from rules regardless of llm/rules mode)
    result["alternatives"] = get_alternatives(columns, rows, n=3)
    return result



def force_config(columns, rows, chart_type, lc=None, num_cols=None):
    """Force a specific chart type regardless of recommendation."""
    if not rows or not columns or chart_type not in _VALID_TYPES:
        return None
    t  = _classify(columns, rows)
    if not lc:
        if chart_type == "calendar_heatmap":
            lc = t["date"][0] if t["date"] else (t["category"][0] if t["category"] else columns[0])
        elif chart_type == "waterfall" and len(rows) == 1 and not t["category"] and not t["date"]:
            # Single-row all-numeric P&L: _build_config will use column names as steps.
            # Don't consume a numeric column as the label — just use a placeholder.
            lc = columns[0]
        else:
            lc = t["category"][0] if t["category"] else (t["date"][0] if t["date"] else columns[0])
    labels   = [str(r.get(lc)) for r in rows]
    if not num_cols:
        if chart_type == "calendar_heatmap":
            num_cols = t["numeric"]  # may be [] — count mode handled inside builder
        elif chart_type == "waterfall" and len(rows) == 1 and not t["category"] and not t["date"]:
            num_cols = t["numeric"] or columns  # all numeric columns become waterfall steps
        else:
            num_cols = t["numeric"] or [columns[-1]]
    cfg = _build_config(chart_type, lc, num_cols, labels, rows)
    if cfg is not None:
        return cfg
    if chart_type == "calendar_heatmap":
        return None  # let API surface a 400 — don't silently fall back to bar
    return _bar(labels, num_cols, rows)
