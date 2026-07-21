"""
One ECharts option-object builder function per chart type. Extracted
verbatim from the original charts/advisor.py.
"""

from __future__ import annotations
import re

from src.charts.advisor.classify import _coerce
from src.charts.advisor.common import (
    _PALETTE, _C_POSITIVE, _C_NEGATIVE, _C_NEUTRAL, _GRID, _TICK, _LABEL, _AXLINE,
    _lbl, _cat_axis, _val_axis, _grid, _tt_axis, _tt_item, _legend, _base,
)




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
