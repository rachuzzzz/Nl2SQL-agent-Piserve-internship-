"""
Heuristic (non-LLM) chart-type selection. Extracted verbatim from the
original charts/advisor.py.
"""

from __future__ import annotations
import re

from src.charts.advisor.classify import (
    _classify, _is_date_col, _is_numeric, _cardinality, _coerce,
    _DATE_RE, _PCT_RE, _SCORE_RE, _INCOME_RE, _COST_RE, _PROFIT_RE, _CAPEX_RE, _WORKFLOW_RE, _RADAR_DIM_KW,
)
from src.charts.advisor.common import _result, _no_chart, _lbl
from src.charts.advisor.renderers import (
    _bar, _stacked_bar, _line, _pie, _scatter, _radar, _heatmap, _waterfall,
    _radar_pivot, _nightingale, _funnel, _gauge, _calendar_heatmap, _kpi_bar,
)




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
