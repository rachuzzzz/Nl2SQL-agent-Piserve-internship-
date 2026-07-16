"""
Column classification: numeric/date/categorical detection, chart-type
validity/label/alternative-type tables. Extracted verbatim from the
original charts/advisor.py.
"""

from __future__ import annotations
import re



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


# Public alias for column classification -- used by /api/columns endpoint
classify_columns = _classify
