"""
Chart Advisor
=============
Analyses SQL result shape and recommends the best Apache ECharts option object.

Public API:
  advise(columns, rows)                → dict   — primary recommendation + alternatives
  force_config(columns, rows, type)    → dict | None
  get_alternatives(columns, rows, n)   → list[dict]  — top N ranked alternatives

Return format of advise():
  {
    "chart_type":     str,        # primary recommended type
    "reason":         str,        # human-readable justification
    "confidence":     str,        # "high" | "medium" | "low"
    "config":         dict,       # ECharts option for primary
    "labels_column":  str | None,
    "values_columns": list[str],
    "alternatives": [             # up to 3 ranked alternatives, ready to render
      {
        "chart_type": str,
        "label":      str,        # display label e.g. "Bar Chart"
        "config":     dict,
      },
      ...
    ]
  }
"""

from __future__ import annotations
import json
import os
import re
from typing import Any


# ── Palette ───────────────────────────────────────────────────────────────────
# Professional dark-background palette — sufficient contrast, not garish
_PALETTE = [
    "#6366f1",   # indigo    (primary)
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

# Semantic colours
_C_POSITIVE = "#10b981"   # green  — income, good
_C_NEGATIVE = "#ef4444"   # red    — cost, bad
_C_NEUTRAL  = "#6366f1"   # indigo — neutral

# Theme tokens (applied to axis/grid/label text)
_GRID    = "rgba(255,255,255,0.05)"
_TICK    = "#6b7280"
_LABEL   = "#9ca3af"
_AXLINE  = "#374151"

# ── Valid types ───────────────────────────────────────────────────────────────
_VALID_TYPES = {
    "bar", "horizontal_bar", "stacked_bar",
    "line", "area",
    "pie", "doughnut", "nightingale",
    "scatter",
    "radar",
    "heatmap",
    "calendar_heatmap",
    "waterfall",
    "funnel",
    "gauge",
}

_TYPE_LABELS = {
    "bar":              "Bar Chart",
    "horizontal_bar":   "Horizontal Bar",
    "stacked_bar":      "Stacked Bar",
    "line":             "Line Chart",
    "area":             "Area Chart",
    "pie":              "Pie Chart",
    "doughnut":         "Doughnut Chart",
    "nightingale":      "Nightingale Chart",
    "scatter":          "Scatter Plot",
    "radar":            "Radar Chart",
    "heatmap":          "Heatmap",
    "calendar_heatmap": "Calendar Heatmap",
    "waterfall":        "Waterfall Chart",
    "funnel":           "Funnel Chart",
    "gauge":            "Gauge",
}

# Alternatives map — what makes sense as a second/third option for each type
_ALT_MAP: dict[str, list[str]] = {
    "bar":              ["horizontal_bar", "line", "funnel"],
    "horizontal_bar":   ["bar", "stacked_bar"],
    "stacked_bar":      ["bar", "area"],
    "line":             ["area", "bar", "calendar_heatmap"],
    "area":             ["line", "stacked_bar"],
    "pie":              ["doughnut", "nightingale", "bar"],
    "doughnut":         ["pie", "nightingale", "bar"],
    "nightingale":      ["doughnut", "pie", "bar"],
    "scatter":          ["bar", "line"],
    "radar":            ["bar", "line"],
    "heatmap":          ["bar", "scatter"],
    "calendar_heatmap": ["line", "bar", "heatmap"],
    "waterfall":        ["bar", "line"],
    "funnel":           ["horizontal_bar", "bar"],
    "gauge":            ["bar"],
}


# ── Column classifier ─────────────────────────────────────────────────────────
_DATE_RE   = re.compile(r'(date|month|year|week|\btime\b|period|quarter|\bday\b|created|modified|submitted|reported|month_idx|month_label)', re.I)
_PCT_RE    = re.compile(r'(pct|percent|rate|ratio|share|proportion|conversion)', re.I)
_SCORE_RE  = re.compile(r'(score|rating|grade|index)', re.I)
_INCOME_RE = re.compile(r'(income|revenue|po_amount|po_total|total_po|converted|earned|receipt|billing|invoic)', re.I)
_COST_RE   = re.compile(r'(cost|opex|pr_amount|pr_total|total_pr|expenditure|expense|spent|loss|outflow|paid)', re.I)
_PROFIT_RE = re.compile(r'(profit|net|margin|gain|balance|surplus)', re.I)
_CAPEX_RE  = re.compile(r'(capex|opex|budget|amount)', re.I)
_WORKFLOW_RE = re.compile(r'(status|state|phase|stage|step|pipeline|funnel|tier)', re.I)
_RADAR_DIM_KW = {
    'safety', 'hygiene', 'health', 'quality', 'compliance', 'structure',
    'document', 'fire', 'electric', 'mechanic', 'plumbing', 'hvac',
    'cleanliness', 'maintenance', 'security', 'environment', 'waste',
    'inspection', 'audit', 'check', 'survey', 'assessment',
}

def _is_date_col(col, vals):
    if _DATE_RE.search(col): return True
    # Value must look like a real date — no alpha chars in the first 10 chars
    # (filters out IDs like "2026/05/ST078/INS001" that start with YYYY/MM/)
    non_null = [v for v in vals[:10] if v is not None]
    if not non_null: return False
    date_like = [
        v for v in non_null
        if re.match(r'\d{4}[-/]\d{2}', str(v)) and not re.search(r'[A-Za-z]', str(v)[:10])
    ]
    return len(date_like) >= max(1, len(non_null) // 2)

def _coerce(v) -> float | None:
    if isinstance(v, (int, float)): return float(v)
    try: return float(str(v).replace(',','').strip())
    except: return None

def _is_numeric(vals):
    nz = [v for v in vals if v is not None]
    return bool(nz) and all(_coerce(v) is not None for v in nz[:10])

def _cardinality(vals):
    return len(set(str(v) for v in vals if v is not None))

def _classify(columns, rows):
    out = {"date": [], "numeric": [], "category": []}
    for col in columns:
        vals = [r.get(col) for r in rows]
        if _is_date_col(col, vals): out["date"].append(col)
        elif _is_numeric(vals):     out["numeric"].append(col)
        else:                       out["category"].append(col)
    
    # Priority fix: if we have month_idx and month_label, prefer month_label for display
    # but ensure it's treated as a date (it's already in date list if it matched RE)
    if "month_idx" in columns and "month_label" in columns:
        if "month_label" in out["category"]:
            out["category"].remove("month_label")
            out["date"].append("month_label")
        # Ensure month_label is the primary date for X-axis
        if "month_label" in out["date"]:
            out["date"].remove("month_label")
            out["date"].insert(0, "month_label")
        # Hide month_idx from numeric so it doesn't break rule #6 (Heatmap)
        if "month_idx" in out["numeric"]:
            out["numeric"].remove("month_idx")
        if "month_idx" in out["date"]:
            out["date"].remove("month_idx")
    
    return out

def _lbl(col): return col.replace("_", " ").title()


# ── ECharts building blocks ────────────────────────────────────────────────────
# Every builder targets a dark-panel background.
# All axis/label colours use theme tokens so they're legible on both
# dark cards and light themes (frontend can override via ECharts theme).

def _cat_axis(data, rotate=0):
    ax = {
        "type": "category", "data": data,
        "axisLabel": {"color": _LABEL, "fontSize": 11, "hideOverlap": True},
        "axisLine":  {"lineStyle": {"color": _AXLINE}},
        "axisTick":  {"show": False},
    }
    if rotate: ax["axisLabel"]["rotate"] = rotate
    return ax

def _val_axis(name=""):
    ax = {
        "type": "value",
        "axisLabel":  {"color": _LABEL, "fontSize": 11},
        "axisLine":   {"show": False},
        "splitLine":  {"lineStyle": {"color": _GRID, "type": "dashed"}},
    }
    if name:
        ax["name"] = name
        ax["nameTextStyle"] = {"color": _LABEL, "fontSize": 10, "padding": [0,0,4,0]}
    return ax

def _grid(left="4%", right="4%", top="48px", bottom="52px"):
    return {"left":left,"right":right,"top":top,"bottom":bottom,"containLabel":True}

def _tt_axis():
    return {
        "trigger": "axis",
        "axisPointer": {"type": "shadow", "shadowStyle": {"color": "rgba(255,255,255,0.03)"}},
        "backgroundColor": "rgba(17,24,39,0.92)",
        "borderColor": "rgba(255,255,255,0.08)",
        "borderWidth": 1,
        "textStyle": {"color": "#e5e7eb", "fontSize": 12},
        "extraCssText": "border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,0.4);",
    }

def _tt_item():
    return {
        "trigger": "item",
        "backgroundColor": "rgba(17,24,39,0.92)",
        "borderColor": "rgba(255,255,255,0.08)",
        "borderWidth": 1,
        "textStyle": {"color": "#e5e7eb", "fontSize": 12},
        "extraCssText": "border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,0.4);",
        "formatter": "{b}<br/>{d}%  ({c})",
    }

def _legend(names, show=True):
    if not show or not names: return {"show": False}
    return {
        "show": True, "data": names,
        "top": "6px", "left": "center",
        "orient": "horizontal",
        "itemGap": 18, "itemWidth": 10, "itemHeight": 10,
        "icon": "circle",
        "textStyle": {"color": _LABEL, "fontSize": 11},
        "pageTextStyle": {"color": _LABEL},
    }

def _base(extra=None):
    out = {
        "_lib": "echarts",
        "color": _PALETTE,
        "animation": True,
        "animationDuration": 600,
        "animationEasing": "cubicOut",
    }
    if extra: out.update(extra)
    return out


# ── Chart builders ────────────────────────────────────────────────────────────

def _bar(labels, num_cols, rows, horizontal=False):
    """Grouped bar — vertical or horizontal."""
    series = []
    for i, col in enumerate(num_cols):
        vals = [_coerce(r.get(col)) for r in rows]
        s = {
            "name": _lbl(col), "type": "bar",
            "data": vals,
            "barMaxWidth": 48,
            "barCategoryGap": "35%",
            "emphasis": {"focus": "series"},
            "itemStyle": {"borderRadius": [3,3,0,0] if not horizontal else [0,3,3,0]},
        }
        series.append(s)

    rotate = 35 if (not horizontal and len(labels) > 10) else 0
    show_legend = len(num_cols) > 1

    if horizontal:
        return _base({
            "tooltip": _tt_axis(),
            "legend":  _legend([_lbl(c) for c in num_cols], show=show_legend),
            "grid":    _grid(left="26%", right="6%", top="16px" if not show_legend else "40px", bottom="16px"),
            "xAxis":   _val_axis(),
            "yAxis":   {**_cat_axis(labels), "axisLabel": {"color":_LABEL,"fontSize":11,"width":160,"overflow":"truncate"}},
            "series":  series,
        })
    return _base({
        "tooltip": _tt_axis(),
        "legend":  _legend([_lbl(c) for c in num_cols], show=show_legend),
        "grid":    _grid(bottom="60px" if rotate else "52px"),
        "xAxis":   _cat_axis(labels, rotate=rotate),
        "yAxis":   _val_axis(),
        "series":  series,
    })


def _stacked_bar(labels, num_cols, rows):
    """Stacked bar with rounded top of the stack."""
    series = []
    for i, col in enumerate(num_cols):
        s = {
            "name": _lbl(col), "type": "bar",
            "stack": "total",
            "data": [_coerce(r.get(col)) for r in rows],
            "emphasis": {"focus": "series"},
        }
        # Only round the top of the last (topmost) series
        if i == len(num_cols) - 1:
            s["itemStyle"] = {"borderRadius": [3,3,0,0]}
        series.append(s)

    rotate = 35 if len(labels) > 10 else 0
    return _base({
        "tooltip": _tt_axis(),
        "legend":  _legend([_lbl(c) for c in num_cols]),
        "grid":    _grid(bottom="60px" if rotate else "52px"),
        "xAxis":   _cat_axis(labels, rotate=rotate),
        "yAxis":   _val_axis(),
        "series":  series,
    })


def _line(labels, num_cols, rows, area=False):
    """Smooth line or area chart."""
    series = []
    for i, col in enumerate(num_cols):
        color = _PALETTE[i % len(_PALETTE)]
        s = {
            "name": _lbl(col), "type": "line",
            "smooth": True,
            "data": [_coerce(r.get(col)) for r in rows],
            "symbolSize": 5,
            "symbol": "circle",
            "emphasis": {"focus": "series"},
            "lineStyle": {"width": 2.5},
        }
        if area:
            s["areaStyle"] = {
                "opacity": 0.12,
                "color": {
                    "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                    "colorStops": [
                        {"offset": 0, "color": color},
                        {"offset": 1, "color": "rgba(0,0,0,0)"},
                    ],
                },
            }
        series.append(s)

    return _base({
        "tooltip": {**_tt_axis(), "axisPointer": {"type": "cross", "label": {"backgroundColor": "#374151"}}},
        "legend":  _legend([_lbl(c) for c in num_cols], show=len(num_cols) > 1),
        "grid":    _grid(),
        "xAxis":   _cat_axis(labels),
        "yAxis":   _val_axis(),
        "series":  series,
    })


def _pie(labels, values, doughnut=False, rose=False):
    """Pie, doughnut, or nightingale (rose) chart."""
    data = [
        {"name": str(lbl), "value": _coerce(val)}
        for lbl, val in zip(labels, values)
        if _coerce(val) is not None
    ]
    n = len(data)
    radius = ["42%", "70%"] if doughnut else "68%"
    # Shift pie left so the vertical right-side legend has clear space.
    center = ["38%", "50%"]

    # With many slices the outward label lines fan across the full chart width and
    # collide with the right-side legend.  Above the threshold, suppress all slice
    # labels — the scroll legend and hover tooltip are sufficient for identification.
    LABEL_THRESHOLD = 8
    show_labels = n <= LABEL_THRESHOLD

    extra_series = {}
    if doughnut:
        extra_series["label"] = {
            "show": show_labels,
            "formatter": "{b}\n{d}%",
            "color": _LABEL, "fontSize": 11,
        }
        extra_series["labelLine"] = {"show": show_labels, "length": 12, "length2": 8}
        total = sum(d["value"] for d in data if d["value"])
    else:
        extra_series["label"] = {
            "show": show_labels,
            "formatter": "{b}: {d}%",
            "color": _LABEL, "fontSize": 11,
        }
        extra_series["labelLine"] = {"show": show_labels}

    if rose:
        extra_series["roseType"] = "radius"
        radius = ["15%", "65%"] if doughnut else "68%"

    return _base({
        "tooltip": _tt_item(),
        "legend": {
            # type:scroll paginates the list — no unbounded growth that overlaps the chart.
            "type": "scroll",
            "orient": "vertical", "right": "2%", "top": "10%", "height": "80%",
            "itemGap": 8, "itemWidth": 10, "itemHeight": 10, "icon": "circle",
            "pageIconColor": _LABEL,
            "pageTextStyle": {"color": _LABEL, "fontSize": 10},
            "textStyle": {
                "color": _LABEL, "fontSize": 11,
                "width": 110, "overflow": "truncate",
            },
        },
        "series": [{
            "type": "pie", "radius": radius, "center": center,
            "data": data,
            "emphasis": {
                "itemStyle": {"shadowBlur": 12, "shadowColor": "rgba(0,0,0,0.35)"},
                "label": {"show": show_labels, "fontSize": 13},
            },
            "itemStyle": {"borderRadius": 4, "borderWidth": 2, "borderColor": "rgba(0,0,0,0.15)"},
            **extra_series,
        }],
    })


def _scatter(x_col, y_col, rows, label_col=None):
    """Scatter plot."""
    data = []
    for r in rows:
        x = _coerce(r.get(x_col))
        y = _coerce(r.get(y_col))
        if x is None or y is None: continue
        if label_col:
            data.append({"value": [x, y], "name": str(r.get(label_col, ""))})
        else:
            data.append([x, y])

    tt_fmt = f"{_lbl(x_col)}: {{c[0]}}<br/>{_lbl(y_col)}: {{c[1]}}" + ("<br/>{b}" if label_col else "")
    return _base({
        "tooltip": {**_tt_item(), "formatter": tt_fmt},
        "grid": _grid(),
        "xAxis": {**_val_axis(_lbl(x_col))},
        "yAxis": {**_val_axis(_lbl(y_col))},
        "series": [{
            "type": "scatter", "data": data,
            "symbolSize": 10,
            "emphasis": {"focus": "self", "itemStyle": {"shadowBlur": 8, "shadowColor": "rgba(0,0,0,0.4)"}},
        }],
    })


def _radar(labels_col, metric_cols, rows):
    """Radar chart."""
    maxes = {}
    for col in metric_cols:
        vals = [_coerce(r.get(col)) for r in rows if _coerce(r.get(col)) is not None]
        maxes[col] = max(vals) * 1.15 if vals else 1

    indicators = [{"name": _lbl(c), "max": maxes[c]} for c in metric_cols]
    series_data = [
        {
            "value": [_coerce(row.get(c)) or 0 for c in metric_cols],
            "name": str(row.get(labels_col, f"Row {i+1}")),
        }
        for i, row in enumerate(rows[:8])
    ]

    return _base({
        "tooltip": {**_tt_item(), "formatter": None},
        "legend": _legend([str(r.get(labels_col, f"Row {i+1}")) for i, r in enumerate(rows[:8])],
                          show=len(rows) > 1),
        "radar": {
            "indicator": indicators,
            "shape": "polygon",
            "axisName": {"color": _LABEL, "fontSize": 11},
            "splitLine": {"lineStyle": {"color": _GRID}},
            "splitArea": {"show": True, "areaStyle": {"color": ["rgba(255,255,255,0.01)", "rgba(255,255,255,0.03)"]}},
            "axisLine":  {"lineStyle": {"color": _AXLINE}},
            "center": ["50%", "54%"], "radius": "68%",
        },
        "series": [{
            "type": "radar", "data": series_data,
            "emphasis": {"focus": "series"},
            "symbol": "circle", "symbolSize": 4,
            "lineStyle": {"width": 2},
            "areaStyle": {"opacity": 0.1},
        }],
    })


def _heatmap(row_col, col_col, val_col, rows):
    """Heatmap grid."""
    y_cats = list(dict.fromkeys(str(r.get(row_col,"")) for r in rows))
    x_cats = list(dict.fromkeys(str(r.get(col_col,"")) for r in rows))

    data = []
    for r in rows:
        xi = x_cats.index(str(r.get(col_col,""))) if str(r.get(col_col,"")) in x_cats else -1
        yi = y_cats.index(str(r.get(row_col,""))) if str(r.get(row_col,"")) in y_cats else -1
        v  = _coerce(r.get(val_col))
        if xi >= 0 and yi >= 0 and v is not None:
            data.append([xi, yi, v])

    all_vals = [d[2] for d in data]
    vmin = min(all_vals) if all_vals else 0
    vmax = max(all_vals) if all_vals else 1
    rotate = 35 if len(x_cats) > 7 else 0

    return _base({
        "tooltip": {
            **_tt_item(),
            "formatter": "{b0}: {c[2]}",
        },
        "grid": _grid(left="20%", bottom="80px", right="70px"),
        "xAxis": {
            **_cat_axis(x_cats, rotate=rotate),
            "splitArea": {"show": True, "areaStyle": {"color": ["rgba(255,255,255,0.01)","rgba(255,255,255,0.03)"]}},
        },
        "yAxis": {
            **_cat_axis(y_cats),
            "splitArea": {"show": True, "areaStyle": {"color": ["rgba(255,255,255,0.01)","rgba(255,255,255,0.03)"]}},
        },
        "visualMap": {
            "min": vmin, "max": vmax,
            "calculable": True,
            "orient": "horizontal", "left": "center", "bottom": "8px",
            "textStyle": {"color": _LABEL},
            "inRange": {"color": ["#1e2a42", "#6366f1", "#10b981"]},
        },
        "series": [{
            "type": "heatmap", "data": data,
            "label": {"show": len(data) <= 60, "fontSize": 10, "color": "#e5e7eb"},
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}},
        }],
    })


def _waterfall(labels, values, col_name):
    """Waterfall chart."""
    running = 0.0
    bases, bars, colors = [], [], []
    for v in values:
        if v >= 0:
            bases.append(running)
            bars.append(v)
            colors.append(_C_POSITIVE)
        else:
            bases.append(running + v)
            bars.append(abs(v))
            colors.append(_C_NEGATIVE)
        running += v

    base_s = {
        "type": "bar", "stack": "wf",
        "itemStyle": {"borderColor":"transparent","color":"transparent"},
        "emphasis": {"itemStyle": {"borderColor":"transparent","color":"transparent"}},
        "data": bases, "silent": True,
    }
    bar_s = {
        "type": "bar", "stack": "wf",
        "name": _lbl(col_name),
        "data": [{"value": b, "itemStyle": {"color": c, "borderRadius": [3,3,0,0] if c==_C_POSITIVE else [0,0,3,3]}} for b,c in zip(bars,colors)],
        "barMaxWidth": 56,
        "label": {
            "show": True, "position": "top",
            "formatter": "{c}", "fontSize": 10, "color": _LABEL,
        },
        "emphasis": {"focus": "self"},
    }
    rotate = 30 if len(labels) > 8 else 0
    return _base({
        "tooltip": _tt_axis(),
        "grid": _grid(bottom="70px" if rotate else "52px"),
        "xAxis": _cat_axis(labels, rotate=rotate),
        "yAxis": _val_axis(),
        "series": [base_s, bar_s],
    })


def _radar_pivot(cat_col, num_col, rows):
    """Radar from aggregated data: each row is a spoke (category-value → score)."""
    max_val = max((_coerce(r.get(num_col)) or 0 for r in rows), default=1)
    indicators = [{"name": _lbl(str(r.get(cat_col, ''))), "max": round(max_val * 1.2, 1)} for r in rows]
    values = [_coerce(r.get(num_col)) or 0 for r in rows]
    return _base({
        "tooltip": {**_tt_item(), "formatter": None},
        "legend": {"show": False},
        "radar": {
            "indicator": indicators,
            "shape": "polygon",
            "axisName": {"color": _LABEL, "fontSize": 11},
            "splitLine": {"lineStyle": {"color": _GRID}},
            "splitArea": {"show": True, "areaStyle": {"color": ["rgba(255,255,255,0.01)", "rgba(255,255,255,0.03)"]}},
            "axisLine":  {"lineStyle": {"color": _AXLINE}},
            "center": ["50%", "54%"], "radius": "68%",
        },
        "series": [{
            "type": "radar",
            "data": [{"value": values, "name": "Score"}],
            "emphasis": {"focus": "series"},
            "symbol": "circle", "symbolSize": 4,
            "lineStyle": {"width": 2},
            "areaStyle": {"opacity": 0.15},
        }],
    })


def _nightingale(labels, values):
    """Nightingale (rose) chart. Wraps _pie with rose=True."""
    return _pie(labels, values, doughnut=False, rose=True)

def _funnel(labels, values, col_name, sort_order="descending"):
    """Funnel chart for workflow or sequential drop-off data."""
    data = []
    for lbl, val in zip(labels, values):
        v = _coerce(val)
        if v is not None:
            data.append({"name": str(lbl), "value": v})
            
    # Calculate max for visual scaling
    max_val = max((d["value"] for d in data), default=100)

    return _base({
        "tooltip": {
            **_tt_item(),
            "formatter": "{a} <br/>{b} : {c}",
        },
        "legend": _legend(labels, show=True),
        "series": [{
            "name": _lbl(col_name),
            "type": "funnel",
            "left": "10%",
            "top": 60,
            "bottom": 60,
            "width": "80%",
            "min": 0,
            "max": max_val,
            "minSize": "0%",
            "maxSize": "100%",
            "sort": sort_order, # 'descending', 'ascending', or 'none'
            "gap": 2,
            "label": {
                "show": True,
                "position": "inside",
                "color": "#fff",
                "fontSize": 12,
            },
            "labelLine": {
                "length": 10,
                "lineStyle": {
                    "width": 1,
                    "type": "solid"
                }
            },
            "itemStyle": {
                "borderColor": "#fff",
                "borderWidth": 1
            },
            "emphasis": {
                "label": {
                    "fontSize": 14
                }
            },
            "data": data
        }]
    })

def _gauge(value, label, max_val=None):
    if max_val is None:
        if value <= 100:
            max_val = 100.0
        else:
            magnitude = 10 ** (len(str(int(value))) - 1)
            max_val = float((int(value) // magnitude + 1) * magnitude)
    return _base({
        "series": [{
            "type": "gauge",
            "center": ["50%", "58%"], "radius": "82%",
            "startAngle": 205, "endAngle": -25,
            "min": 0, "max": max_val, "splitNumber": 5,
            "axisLine": {
                "lineStyle": {
                    "width": 18,
                    "color": [[0.4,"#ef4444"],[0.7,"#f59e0b"],[1.0,"#10b981"]],
                }
            },
            "pointer": {
                "length": "58%", "width": 6,
                "itemStyle": {"color": "auto"},
            },
            "axisTick": {"distance": -24, "length": 6, "lineStyle": {"color":"#fff","width":1}},
            "splitLine": {"distance": -30, "length": 14, "lineStyle": {"color":"#fff","width":2}},
            "axisLabel": {"color": "inherit", "distance": 38, "fontSize": 11},
            "title": {
                "fontSize": 13, "fontWeight": 500,
                "color": "inherit",
                "offsetCenter": [0, "45%"],
            },
            "detail": {
                "valueAnimation": True,
                "formatter": "{value}",
                "color": "inherit", "fontSize": 24, "fontWeight": 700,
                "offsetCenter": [0, "18%"],
            },
            "data": [{"value": round(value, 1), "name": label}],
        }],
    })


def _calendar_heatmap(date_col, val_col, rows):
    """Calendar heatmap (GitHub-style) for day-level activity data.
    When val_col is None or yields no numeric data, falls back to counting rows per day."""
    from collections import defaultdict
    date_agg: dict = defaultdict(float)
    found_numeric = False

    for r in rows:
        dv = r.get(date_col)
        if dv is None:
            continue
        ds = str(dv)[:10]
        if not re.match(r'\d{4}-\d{2}-\d{2}', ds):
            continue
        if val_col is not None:
            v = _coerce(r.get(val_col))
            if v is not None:
                date_agg[ds] += v
                found_numeric = True
            else:
                date_agg[ds]  # ensure key exists for fallback
        else:
            date_agg[ds] += 1

    # val_col given but all non-numeric → fall back to row-count per day
    if val_col is not None and not found_numeric:
        date_agg = defaultdict(float)
        for r in rows:
            dv = r.get(date_col)
            if dv is None:
                continue
            ds = str(dv)[:10]
            if re.match(r'\d{4}-\d{2}-\d{2}', ds):
                date_agg[ds] += 1
        val_col = None  # label as "Count" below

    data = [[ds, v] for ds, v in sorted(date_agg.items()) if v > 0]
    if not data:
        return None

    date_strs = sorted(d[0] for d in data)
    years = sorted(set(d[:4] for d in date_strs))

    if len(years) == 1:
        cal_range = years[0]
    else:
        cal_range = [date_strs[0], date_strs[-1]]

    all_vals = [d[1] for d in data]
    vmin, vmax = min(all_vals), max(all_vals)

    return _base({
        "calendar": {
            "range": cal_range,
            "cellSize": ["auto", 18],
            "top": 60,
            "left": 44,
            "right": 20,
            "dayLabel": {
                "firstDay": 1,
                "color": _LABEL, "fontSize": 10,
                "nameMap": ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"],
            },
            "monthLabel": {"color": _LABEL, "fontSize": 10},
            "yearLabel":  {"color": _LABEL, "fontSize": 12, "position": "left"},
            "itemStyle": {
                "color": "rgba(255,255,255,0.03)",
                "borderColor": "rgba(255,255,255,0.08)",
                "borderWidth": 1,
            },
            "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.1)", "width": 1}},
        },
        "tooltip": {
            **_tt_item(),
            "formatter": "{b}: {c}",
        },
        "visualMap": {
            "min": vmin, "max": vmax,
            "calculable": True,
            "orient": "horizontal", "left": "center", "bottom": "8px",
            "textStyle": {"color": _LABEL},
            "inRange": {"color": ["#1e2a42", "#6366f1", "#10b981"]},
        },
        "series": [{
            "type": "heatmap",
            "coordinateSystem": "calendar",
            "name": _lbl(val_col) if val_col else "Count",
            "data": data,
            "label": {"show": False},
            "emphasis": {
                "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(99,102,241,0.5)"},
            },
        }],
    })


def _kpi_bar(num_cols, rows):
    """Single-row multi-KPI horizontal bar, one colour per bar."""
    data = [
        {"value": _coerce(rows[0].get(col)), "name": _lbl(col),
         "itemStyle": {"color": _PALETTE[i % len(_PALETTE)], "borderRadius": [0,3,3,0]}}
        for i, col in enumerate(num_cols)
    ]
    return _base({
        "tooltip": _tt_axis(),
        "grid": _grid(left="24%", right="8%", top="12px", bottom="12px"),
        "xAxis": _val_axis(),
        "yAxis": {
            **_cat_axis([_lbl(c) for c in num_cols]),
            "axisLabel": {"color":_LABEL,"fontSize":11,"width":170,"overflow":"truncate"},
        },
        "series": [{
            "type": "bar", "data": data, "barMaxWidth": 48,
            "label": {"show": True, "position": "right", "color": _LABEL, "fontSize": 11},
        }],
    })


# ── Rule engine ────────────────────────────────────────────────────────────────

def _rules_advise(columns, rows):
    if not rows or not columns:
        return _no_chart("No data")

    t = _classify(columns, rows)
    dates, nums, cats = t["date"], t["numeric"], t["category"]
    n = len(rows)

    has_income = any(_INCOME_RE.search(c) for c in nums)
    has_cost   = any(_COST_RE.search(c)   for c in nums)
    has_profit = any(_PROFIT_RE.search(c) for c in nums)
    has_capex  = any(_CAPEX_RE.search(c)  for c in nums)

    # ── 1. Single-row KPI ─────────────────────────────────────────────────────
    if n == 1:
        # Single score/rate in 0-100 → gauge (column name must signal a bounded metric)
        if len(nums) == 1 and not cats:
            v = _coerce(rows[0].get(nums[0]))
            if v is not None and 0 <= v <= 100 and (_SCORE_RE.search(nums[0]) or _PCT_RE.search(nums[0])):
                return _result("gauge", f"{_lbl(nums[0])} KPI — gauge", "high",
                               None, nums, _gauge(v, _lbl(nums[0])))
        # P&L single row → waterfall across metric names
        if len(nums) >= 2 and (has_income or has_cost or has_profit):
            metric_labels = [_lbl(c) for c in nums]
            metric_vals   = [_coerce(rows[0].get(c)) or 0 for c in nums]
            return _result("waterfall", "P&L breakdown — waterfall", "high",
                           None, nums, _waterfall(metric_labels, metric_vals, nums[0]))
        # Multi-metric → KPI bar
        if len(nums) >= 2:
            return _result("bar", f"Single-row summary — {len(nums)} KPIs", "high",
                           None, nums, _kpi_bar(nums, rows))

    # ── 2. Waterfall — mixed positive/negative ─────────────────────────────────
    if len(nums) == 1 and cats and n <= 20:
        vals = [_coerce(r.get(nums[0])) for r in rows if _coerce(r.get(nums[0])) is not None]
        if any(v < 0 for v in vals):
            lc = cats[0]
            return _result("waterfall", "Mixed positive/negative values — waterfall", "high",
                           lc, nums, _waterfall([str(r.get(lc)) for r in rows],
                                                [_coerce(r.get(nums[0])) or 0 for r in rows], nums[0]))

    # ── 3. Stacked bar — capex/financial multi-metric ─────────────────────────
    if cats and 2 <= len(nums) <= 5 and (has_capex or (has_income and has_cost)):
        lc = cats[0]
        return _result("stacked_bar", f"Compositional breakdown per '{lc}' — stacked bar", "high",
                       lc, nums, _stacked_bar([str(r.get(lc)) for r in rows], nums, rows))

    # P&L multi-row → waterfall
    if (has_income or has_cost or has_profit) and len(nums) >= 2 and cats and n <= 15:
        lc = cats[0]
        metric_labels = [_lbl(c) for c in nums[:3]]
        metric_vals   = [_coerce(rows[0].get(c)) or 0 for c in nums[:3]]
        return _result("waterfall", "P&L decomposition — waterfall", "high",
                       lc, nums[:3], _waterfall(metric_labels, metric_vals, nums[0]))

    # ── 6. Heatmap — two categorical dimensions × one numeric. ─────────────────
    all_cats = cats + dates
    if len(all_cats) >= 2 and len(nums) == 1 and n >= 4:
        rc, cc, nc = all_cats[0], all_cats[1], nums[0]
        if _cardinality([r.get(rc) for r in rows]) >= 2 and _cardinality([r.get(cc) for r in rows]) >= 2:
            return _result("heatmap", f"Grid of '{rc}' × '{cc}' — heatmap", "high",
                           rc, nums, _heatmap(rc, cc, nc, rows))

    # ── 3.5 Calendar heatmap — day-level dates (YYYY-MM-DD), ≥7 unique days ───
    if dates and len(nums) <= 1 and n >= 7:
        dc = dates[0]
        day_like = [
            str(r.get(dc, ''))[:10] for r in rows
            if r.get(dc) and re.match(r'\d{4}-\d{2}-\d{2}', str(r.get(dc, ''))[:10])
        ]
        if len(set(day_like)) >= 7:
            vc = nums[0] if nums else None
            cfg = _calendar_heatmap(dc, vc, rows)
            if cfg:
                reason = (
                    f"Daily {_lbl(vc)} across {len(set(day_like))} days — calendar heatmap"
                    if vc else
                    f"Inspections per day across {len(set(day_like))} days — calendar heatmap"
                )
                return _result("calendar_heatmap", reason, "high", dc, nums, cfg)

    # ── 4. Time-series ─────────────────────────────────────────────────────────
    if dates and _cardinality([r.get(dates[0]) for r in rows]) >= 3:
        lc = dates[0]
        labels = [str(r.get(lc)) for r in rows]
        
        # 4a. Breakdown over time: Date + Category + Numeric
        if cats and len(nums) == 1 and n > 5:
            # Pivot data for stacked bar: rows=dates, columns=categories
            cat_col = cats[0]
            val_col = nums[0]
            unique_cats = list(dict.fromkeys(str(r.get(cat_col)) for r in rows))
            unique_dates = list(dict.fromkeys(str(r.get(lc)) for r in rows))
            
            # If we have a reasonable number of categories and dates, stack them
            if 2 <= len(unique_cats) <= 12 and len(unique_dates) >= 2:
                pivot_rows = []
                for d in unique_dates:
                    row_data = {lc: d}
                    for c in unique_cats:
                        # Find value for this date and category
                        val = next((_coerce(r.get(val_col)) for r in rows if str(r.get(lc)) == d and str(r.get(cat_col)) == c), 0)
                        row_data[c] = val
                    pivot_rows.append(row_data)
                
                return _result("stacked_bar", f"Breakdown of '{val_col}' by '{cat_col}' over time", "high",
                               lc, unique_cats, _stacked_bar(unique_dates, unique_cats, pivot_rows))

        if len(nums) == 1:
            use_area = any(kw in nums[0].lower() for kw in ("total","cumul","volume","sum"))
            ct = "area" if use_area else "line"
            return _result(ct, f"Time-series on '{lc}'", "high",
                           lc, nums, _line(labels, nums, rows, area=use_area))
        if len(nums) >= 2:
            parts = has_income and has_cost
            if parts and len(nums) <= 4:
                return _result("area", "Multi-metric time-series — stacked area", "high",
                               lc, nums[:4], _line(labels, nums[:4], rows, area=True))
            return _result("line", f"Multi-series time-series on '{lc}'", "high",
                           lc, nums, _line(labels, nums[:6], rows))

    # ── 5. Radar — multi-metric profile across few entities ───────────────────
    if cats and len(nums) >= 2 and n <= 12:
        lc = cats[0]
        all_scores = all(bool(_SCORE_RE.search(nc) or _PCT_RE.search(nc)) for nc in nums[:6])
        ranges = []
        for nc in nums[:6]:
            vals = [_coerce(r.get(nc)) for r in rows if _coerce(r.get(nc)) is not None]
            if vals: ranges.append(max(vals) - min(vals))
        similar_scale = not ranges or (max(ranges) / max(min(ranges), 0.001)) < 1000
        max_entities = 12 if all_scores else 8
        if similar_scale and n <= max_entities:
            return _result("radar", f"Multi-metric profile per '{lc}' — radar",
                           "high" if all_scores else "medium",
                           lc, nums[:6], _radar(lc, nums[:6], rows))

    # ── 6.5 Funnel — workflow progression ──────────────────────────────────────
    if cats and len(nums) == 1 and n <= 12:
        lc, nc = cats[0], nums[0]
        # Check if the column name implies a workflow or status
        is_workflow = bool(_WORKFLOW_RE.search(lc))
        # Or if the values themselves look like workflow stages
        if not is_workflow:
            sample_vals = [str(r.get(lc, "")).lower() for r in rows[:5]]
            workflow_keywords = {"open", "pending", "progress", "review", "approved", "closed", "resolved", "rejected", "lead", "opportunity", "qualified", "won", "lost"}
            if any(any(kw in val for kw in workflow_keywords) for val in sample_vals):
                is_workflow = True

        if is_workflow:
            return _result("funnel", f"Workflow/Stage progression — funnel", "high",
                           lc, nums, _funnel([str(r.get(lc)) for r in rows], [_coerce(r.get(nc)) for r in rows], nc, sort_order="descending"))

    # ── 5a. Radar pivot — aggregated score dimensions (1 cat, 1 score col) ───────
    # Fires when: query returns one row per inspection dimension with a score value,
    # e.g. (category='Safety', avg_score=78), (category='Hygiene', avg_score=90)
    if cats and len(nums) == 1 and 3 <= n <= 20:
        lc, nc = cats[0], nums[0]
        is_score_col = bool(_SCORE_RE.search(nc) or _PCT_RE.search(nc))
        cat_vals_lower = [str(r.get(lc, '')).lower() for r in rows]
        cat_looks_like_dims = any(any(kw in v for kw in _RADAR_DIM_KW) for v in cat_vals_lower)
        cat_col_is_type = bool(re.search(r'(type|categor|section|domain|area|dimension)', lc, re.I))
        if is_score_col and (cat_looks_like_dims or cat_col_is_type):
            return _result("radar", f"Score profile across {_lbl(lc)} dimensions — radar", "high",
                           lc, [nc], _radar_pivot(lc, nc, rows))

    # ── 7. Low-cardinality single metric → doughnut or nightingale ─────────────
    if cats and len(nums) == 1 and 3 <= n <= 8:
        lc, vc = cats[0], nums[0]
        vals = [_coerce(r.get(vc)) for r in rows if _coerce(r.get(vc)) is not None]
        
        # Nightingale for high variance, doughnut for more even distributions
        if vals and min(vals) > 0 and (max(vals) / min(vals)) >= 3.0:
            return _result("nightingale", f"Variable magnitude across {n} categories", "high",
                           lc, nums, _nightingale([str(r.get(lc)) for r in rows], [_coerce(r.get(vc)) for r in rows]))
            
        return _result("doughnut", f"Part-of-whole across {n} categories", "high",
                       lc, nums, _pie([str(r.get(lc)) for r in rows], [_coerce(r.get(vc)) for r in rows], doughnut=True))

    # ── 8. Single numeric — bar or horizontal bar ─────────────────────────────
    if cats and len(nums) == 1:
        lc, nc = cats[0], nums[0]
        horiz = n > 10 or any(len(str(r.get(lc,""))) > 18 for r in rows[:5])
        ct = "horizontal_bar" if horiz else "bar"
        return _result(ct, f"{'Horizontal bar' if horiz else 'Bar'} — '{lc}' vs '{nc}'", "high",
                       lc, nums, _bar([str(r.get(lc)) for r in rows], nums, rows, horizontal=horiz))

    # ── 9a. Labeled scatter: 1 category + exactly 2 numerics + many rows ────────
    # When there are ≥ 15 data points, scatter shows correlation better than grouped bar.
    if len(cats) == 1 and len(nums) == 2 and n >= 15:
        lc = cats[0]
        conf = "high" if n >= 5 else "medium"
        return _result("scatter", f"Correlation — scatter plot labeled by '{lc}'", conf,
                       None, nums, _scatter(nums[0], nums[1], rows, label_col=lc))

    # ── 9. Multiple numerics → grouped bar ────────────────────────────────────
    if cats and 2 <= len(nums) <= 6:
        lc = cats[0]
        return _result("bar", f"Grouped bar — {len(nums)} metrics per '{lc}'", "high",
                       lc, nums, _bar([str(r.get(lc)) for r in rows], nums[:6], rows))

    # ── 10. Two numerics no category → scatter ─────────────────────────────────
    if len(nums) >= 2 and not cats and not dates:
        label_col = next((c for c in columns if c not in nums), None)
        conf = "high" if n >= 5 else "medium"
        return _result("scatter", "Correlation — scatter plot", conf,
                       None, nums[:2], _scatter(nums[0], nums[1], rows, label_col=label_col))

    # ── 11. Fallback ───────────────────────────────────────────────────────────
    if nums:
        lc = cats[0] if cats else (dates[0] if dates else columns[0])
        horiz = n > 10
        return _result("horizontal_bar" if horiz else "bar", "Bar chart", "low",
                       lc, nums[:4], _bar([str(r.get(lc)) for r in rows], nums[:4], rows, horizontal=horiz))

    return _no_chart("No numeric columns found")


def _result(chart_type, reason, confidence, lc, vcs, config):
    return {"chart_type": chart_type, "reason": reason, "confidence": confidence,
            "labels_column": lc, "values_columns": vcs, "config": config}

def _no_chart(reason):
    return {"chart_type": None, "reason": reason, "confidence": "low",
            "labels_column": None, "values_columns": [], "config": None}


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
        from llama_index.llms.ollama import Ollama
        url   = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:14b-instruct-q4_K_M")
        _llm_client = Ollama(model=model, base_url=url, request_timeout=15.0)
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


# ── Public API ────────────────────────────────────────────────────────────────

# Public alias for column classification — used by /api/columns endpoint
classify_columns = _classify


def advise(columns, rows):
    """
    Primary entry point. Returns recommendation + alternatives.

    Response includes:
      chart_type, reason, confidence, config,
      labels_column, values_columns,
      alternatives: list of {chart_type, label, config}
    """
    use_llm = os.getenv("USE_LLM_CHART_ADVISOR","false").lower() in ("1","true","yes")
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


# ── Dashboard helpers ──────────────────────────────────────────────────────────

def _prioritize_cards(cards: list) -> list:
    """Sort: financial (capex/income/cost/profit) → score/pct → everything else."""
    def _rank(card):
        col = card.get("_col", "")
        if _INCOME_RE.search(col) or _PROFIT_RE.search(col) or _CAPEX_RE.search(col) or _COST_RE.search(col):
            return 0
        if _SCORE_RE.search(col) or _PCT_RE.search(col):
            return 1
        return 2
    return sorted(cards, key=_rank)


def build_kpi_cards(columns: list, rows: list, max_cards: int = 4) -> list:
    """
    Derive summary KPI cards from a result set.
    Score/pct columns  → avg + range (min/max).
    Other numeric      → total sum.
    First cat column   → count distinct.
    Returns at most max_cards cards, prioritised: financial > score/pct > count.
    """
    t = _classify(columns, rows)
    cards = []
    range_added = False  # only one range card

    for col in t["numeric"]:
        vals = [_coerce(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        is_score = bool(_SCORE_RE.search(col) or _PCT_RE.search(col))
        if is_score:
            cards.append({
                "label": f"Avg {_lbl(col)}",
                "value": round(sum(vals) / len(vals), 2),
                "kind":  "avg",
                "icon":  "star",
                "_col":  col,
            })
            if not range_added:
                cards.append({
                    "label":  f"{_lbl(col)} Range",
                    "value":  round(min(vals), 2),   # min
                    "value2": round(max(vals), 2),   # max
                    "kind":   "range",
                    "icon":   "range",
                    "_col":   col,
                })
                range_added = True
        else:
            cards.append({
                "label": f"Total {_lbl(col)}",
                "value": round(sum(vals), 2),
                "kind":  "sum",
                "icon":  "list",
                "_col":  col,
            })

    # Count-distinct card for the first categorical column
    for col in t["category"]:
        raw_vals = [r.get(col) for r in rows if r.get(col) is not None]
        distinct = len(set(str(v) for v in raw_vals))
        if distinct >= 2:
            cards.append({
                "label": f"Unique {_lbl(col)}",
                "value": distinct,
                "kind":  "distinct",
                "icon":  "hash",
                "_col":  col,
            })
            break  # one distinct card is enough

    prioritized = _prioritize_cards(cards)[:max_cards]
    return [{k: v for k, v in c.items() if k != "_col"} for c in prioritized]


def should_show_dashboard(columns: list, rows: list, chart_config: dict | None) -> bool:
    """
    Gate: show the full animated dashboard only when all four conditions hold:
      1. The chart advisor found a valid chart type (not None).
      2. The result has ≥3 rows (fewer rows are better served by plain answer).
      3. There is at least one date or category column (a sensible axis exists).
      4. There is at least one numeric column (something to summarise on cards).
    Single-value queries, filter/detail lists, and no-chart shapes stay as plain answers.
    """
    if not chart_config or not chart_config.get("chart_type"):
        return False
    if len(rows) < 3:
        return False
    t = _classify(columns, rows)
    return bool(t["date"] or t["category"]) and len(t["numeric"]) >= 1