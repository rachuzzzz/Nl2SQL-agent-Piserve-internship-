"""
KPI card selection/prioritization and the dashboard-vs-chart-suggestion
decision. Extracted verbatim from the original charts/advisor.py.
"""

from __future__ import annotations

from src.charts.advisor.classify import (
    _classify, _coerce, _PCT_RE, _SCORE_RE, _INCOME_RE, _COST_RE, _PROFIT_RE, _CAPEX_RE,
)
from src.charts.advisor.common import _lbl




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