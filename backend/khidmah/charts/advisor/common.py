"""
Palette/color constants and generic ECharts-config helpers shared by
every chart-type renderer. Extracted verbatim from the original
charts/advisor.py.
"""

from __future__ import annotations




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



def _result(chart_type, reason, confidence, lc, vcs, config):
    return {"chart_type": chart_type, "reason": reason, "confidence": confidence,
            "labels_column": lc, "values_columns": vcs, "config": config}


def _no_chart(reason):
    return {"chart_type": None, "reason": reason, "confidence": "low",
            "labels_column": None, "values_columns": [], "config": None}
