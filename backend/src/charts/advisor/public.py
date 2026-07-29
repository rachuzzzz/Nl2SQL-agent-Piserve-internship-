"""
Public chart-advisor API: advise(), force_config(), get_alternatives(),
plus the LLM-based advisor path. Extracted verbatim from the original
charts/advisor.py.
"""

from __future__ import annotations

from src.charts.advisor.classify import (
    _classify, _coerce, _VALID_TYPES, _TYPE_LABELS, _ALT_MAP, _SCORE_RE, _PCT_RE,
)
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



# Categorical/part-of-whole chart types where hundreds of categories/slices
# are unreadable regardless of screen space — capped to N-1 entries plus one
# synthetic "Other" row summarizing the remainder. N matches (does not
# contradict) the thresholds rules.py's cascade already uses to pick these
# types in the first place (LABEL_THRESHOLD=8 in renderers.py for pie-family,
# rules.py:154's max_entities for radar, rules.py:161's funnel n<=12 gate).
# Deliberately excludes: calendar_heatmap (needs the full date range, not
# top-N), line/area (truncating by value would scramble chronological order —
# downsampling is a separate concern), scatter (more points aid reading
# correlation — rely on pan instead), heatmap/gauge/single-row waterfall
# (already inherently bounded by their own selection rules).
_TOPN_BY_TYPE = {
    "pie": 8, "doughnut": 8, "nightingale": 8,
    "bar": 20, "horizontal_bar": 20,
    "stacked_bar": 15,
    "radar": 8,   # bumped to 12 below when every value column is a score/pct
    "funnel": 12,
}


def _apply_topn_aggregation(chart_type, lc, num_cols, rows):
    """Truncate the row set feeding a categorical chart to a readable N,
    folding the remainder into one synthetic 'Other (k)' row — instead of
    silently rendering (or cutting off) hundreds of bars/slices with no
    indication. Returns (rows_out, total_categories, shown_categories);
    a no-op (unchanged rows, total == shown) when the type isn't in
    _TOPN_BY_TYPE, there's no label column, or there's nothing to truncate."""
    if chart_type not in _TOPN_BY_TYPE or not lc or not rows:
        return rows, len(rows), len(rows)

    total = len(rows)
    n_cap = _TOPN_BY_TYPE[chart_type]
    if chart_type == "radar":
        all_scores = all(bool(_SCORE_RE.search(nc) or _PCT_RE.search(nc)) for nc in num_cols[:6])
        n_cap = 12 if all_scores else 8

    if total <= n_cap:
        return rows, total, total

    def _row_value(r):
        vals = [_coerce(r.get(c)) for c in num_cols]
        return sum(v for v in vals if v is not None)

    ranked = sorted(rows, key=_row_value, reverse=True)
    kept, rest = ranked[:n_cap - 1], ranked[n_cap - 1:]

    other_row = {lc: f"Other ({len(rest)})"}
    for nc in num_cols:
        vals = [_coerce(r.get(nc)) for r in rest if _coerce(r.get(nc)) is not None]
        if chart_type == "radar":
            # A single-entity radar arm has to be one number, not a sum across
            # many entities — average the excluded profiles per axis instead.
            other_row[nc] = (sum(vals) / len(vals)) if vals else 0
        else:
            other_row[nc] = sum(vals)

    return kept + [other_row], total, n_cap


def advise(columns, rows, aggregate: bool = True):
    """
    Primary entry point. Returns recommendation + alternatives.

    Response includes:
      chart_type, reason, confidence, config,
      labels_column, values_columns,
      rows_total, rows_shown, is_aggregated — top-N truncation metadata,
      always present (rows_shown == rows_total, is_aggregated == False when
      nothing was truncated), consumed by ChartConfig (see schemas.py) to
      surface a "Showing top N of M — View all" affordance.
      alternatives: list of {chart_type, label, config}

    aggregate=False skips the top-N + "Other" truncation entirely (used by
    the /api/chart "view all" drill-down — see ChartRequest.view_all) and
    builds the config from every row instead.
    """
    result = _rules_advise(columns, rows)

    lc   = result.get("labels_column")
    nums = result.get("values_columns") or []
    if aggregate:
        agg_rows, total, shown = _apply_topn_aggregation(result["chart_type"], lc, nums, rows)
    else:
        agg_rows, total, shown = rows, len(rows), len(rows)
    result["rows_total"]    = total
    result["rows_shown"]    = shown
    result["is_aggregated"] = shown < total
    if result["is_aggregated"]:
        labels = [str(r.get(lc)) for r in agg_rows] if lc else []
        cfg = _build_config(result["chart_type"], lc, nums, labels, agg_rows)
        if cfg is not None:
            result["config"] = cfg
        else:
            # Couldn't rebuild for the aggregated shape (rare) — fall back to
            # the un-aggregated config rather than showing a broken chart.
            result["rows_shown"]    = result["rows_total"]
            result["is_aggregated"] = False
    result["can_view_all"] = result["is_aggregated"]

    # Always attach alternatives (built from rules)
    result["alternatives"] = get_alternatives(columns, rows, n=3)
    return result


# Chart types where the 50-row sample is already the complete/representative
# answer — skip the extra DB round-trip a fuller re-fetch would cost.
_SKIP_FULL_FETCH_TYPES = {"gauge"}


def advise_full(columns, sample_rows, sql=None, orch=None, params=None, aggregate: bool = True):
    """
    Two-phase chart recommendation — the single entry point all three
    chart-producing routes (chat.py's chart-followup replay, chat.py's
    auto-embed dashboard, routes/charts.py's /api/chart) should call instead
    of bare advise(), so chart data volume is consistent across all of them
    (this replaces the old calendar-only refetch_for_calendar special case).

    Phase 1: advise() on whatever's already cached (sample_rows, capped at
    50 by execute_sql). Phase 2: if the picked chart_type would benefit from
    more data than the sample, fetch up to settings.chart_fetch_cap rows via
    charts.data_fetch.fetch_chart_data and rebuild the SAME chart_type
    against the fuller dataset via _build_config (a straight per-type
    builder, never a re-classification) — the chart type is locked after
    phase 1 and never flips out from under the user between provisional and
    final render.

    sql/orch are optional — pass None (e.g. no cached SQL yet) to skip phase
    2 and return the phase-1 result as-is. Any phase-2 fetch failure or
    unusable result silently falls back to phase 1 — this must never raise.
    aggregate=False skips top-N + "Other" truncation on the fuller dataset
    too (see advise()'s docstring — used by the "view all" drill-down).
    """
    result = advise(columns, sample_rows, aggregate=aggregate)
    chart_type = result.get("chart_type")
    if not chart_type or chart_type in _SKIP_FULL_FETCH_TYPES or not sql or orch is None:
        return result
    # Single-row P&L waterfall — phase 1's one row IS the complete answer.
    if chart_type == "waterfall" and len(sample_rows) == 1:
        return result

    from src.charts.data_fetch import fetch_chart_data
    full_rows = fetch_chart_data(sql, orch, params=params)
    if not full_rows or len(full_rows) <= len(sample_rows):
        # Fetch failed, or the sample already was the whole result set.
        return result

    lc   = result.get("labels_column")
    nums = result.get("values_columns") or []
    if aggregate:
        agg_rows, total, shown = _apply_topn_aggregation(chart_type, lc, nums, full_rows)
    else:
        agg_rows, total, shown = full_rows, len(full_rows), len(full_rows)
    labels = [str(r.get(lc)) for r in agg_rows] if lc else []
    cfg = _build_config(chart_type, lc, nums, labels, agg_rows)
    if cfg is None:
        # Locked type doesn't rebuild against the fuller data shape (rare) —
        # keep the phase-1 config rather than showing a broken chart.
        return result

    result["config"]        = cfg
    result["rows_total"]    = total
    result["rows_shown"]    = shown
    result["is_aggregated"] = shown < total
    result["can_view_all"]  = result["is_aggregated"]
    # Full (non-truncated) fetched rows — callers computing summary stats
    # (e.g. KPI cards' SUM/AVG) should use these instead of the 50-row sample,
    # so a card doesn't silently report a partial total as if it were the
    # whole answer. agg_rows (with the synthetic "Other" row) is NOT this —
    # that's for the chart display only, "Other" would corrupt a SUM.
    result["_full_rows"] = full_rows
    # Alternatives were built off the small sample in phase 1 — left as-is;
    # they're illustrative "try this instead" previews, not the main config.
    return result



def force_config(columns, rows, chart_type, lc=None, num_cols=None, aggregate: bool = True):
    """Force a specific chart type regardless of recommendation.

    Returns (config, rows_total, rows_shown) — config is None if the type/
    data combination isn't buildable. rows_total/rows_shown reflect the same
    top-N + 'Other' aggregation advise() applies (see _apply_topn_aggregation)
    so a user-forced chart type is exactly as readable as an auto-picked one,
    rather than bypassing the readability cap just because it was explicitly
    requested. aggregate=False skips that truncation (see advise()'s
    docstring — used by the "view all" drill-down)."""
    if not rows or not columns or chart_type not in _VALID_TYPES:
        return None, 0, 0
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
    if not num_cols:
        if chart_type == "calendar_heatmap":
            num_cols = t["numeric"]  # may be [] — count mode handled inside builder
        elif chart_type == "waterfall" and len(rows) == 1 and not t["category"] and not t["date"]:
            num_cols = t["numeric"] or columns  # all numeric columns become waterfall steps
        else:
            num_cols = t["numeric"] or [columns[-1]]

    if aggregate:
        agg_rows, total, shown = _apply_topn_aggregation(chart_type, lc, num_cols, rows)
    else:
        agg_rows, total, shown = rows, len(rows), len(rows)
    labels = [str(r.get(lc)) for r in agg_rows]
    cfg = _build_config(chart_type, lc, num_cols, labels, agg_rows)
    if cfg is not None:
        return cfg, total, shown
    if chart_type == "calendar_heatmap":
        return None, total, shown  # let API surface a 400 — don't silently fall back to bar
    return _bar(labels, num_cols, agg_rows), total, shown


def force_config_full(columns, sample_rows, chart_type, sql=None, orch=None,
                       params=None, lc=None, num_cols=None, aggregate: bool = True):
    """Two-phase counterpart to force_config() — fetches a fuller dataset
    before building/aggregating (same flow as advise_full()) for a
    user-forced chart type (explicit /chart <type>, or an axis-picker
    selection) rather than an auto-picked one. sql/orch optional — pass None
    to skip phase 2. Returns (config, rows_total, rows_shown)."""
    rows = sample_rows
    skip = chart_type in _SKIP_FULL_FETCH_TYPES or (chart_type == "waterfall" and len(sample_rows) == 1)
    if not skip and sql and orch is not None:
        from src.charts.data_fetch import fetch_chart_data
        full_rows = fetch_chart_data(sql, orch, params=params)
        if full_rows and len(full_rows) > len(sample_rows):
            rows = full_rows
    return force_config(columns, rows, chart_type, lc=lc, num_cols=num_cols, aggregate=aggregate)
