"""
eval_truth.py — ground-truth layer for the NL2SQL eval.

PROBLEM THIS SOLVES
===================
`real_pass` measures plausibility (query ran, answer non-empty, keywords
present), not correctness. Run 20260610_231945 counted these as passes:
  JOI-07  returned Kingfield 140      — known truth 72
  JOI-06  returned 47                 — known truth 45
  AGG-10  returned 2.08               — known truth ~4.2
  COMP-03 returned a 600% MA rate     — numerically nonsensical

This module adds RESULT-LEVEL verification: each covered case has canonical
SQL (written against today's confirmed semantics), both queries execute, and
the result sets are compared value-by-value (column-name agnostic, numeric
tolerance, order-insensitive where order doesn't matter).

USAGE
=====
From eval.py (see patch in run_eval):

    from eval_truth import GROUND_TRUTH, run_truth_check
    tm = run_truth_check(orch.db_engine, qid, agent_rows)
    # tm = {"covered": bool, "truth_match": bool|None, "detail": str}

Standalone self-test of the comparison logic (no DB needed):

    python eval_truth.py

COMPARISON MODES
================
  scalar — both results are 1 row × 1 value; numeric within tolerance.
  row    — both results are exactly 1 row, N columns; values compared as a
           multiset (column order/name agnostic), numerics within tolerance.
  top1   — "which X has the most Y" cases: only the FIRST row must match
           (label case-insensitive, count within tolerance). The agent may
           return extra rows below the winner.
  set    — full result sets equal as multisets of value-tuples (row order
           and column names ignored; column ORDER within a row matters).

Tolerance: numeric values match when |a-b| <= max(abs_tol, rel_tol*|truth|).
Defaults: abs_tol=0.02, rel_tol=0.01 (1%). Override per case with "tol".
"""

from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════
# Canonical SQL per eval case — written against CONFIRMED semantics:
#   completed = CLOSED | conducted = SUBMITTED/UNDER_REVIEW/RETURN_FOR_MOD
#   DRAFT never counted | good practice = COUNT(DISTINCT ir.id)
#   capex_lost = SUM(ica.capex) - COALESCE(SUM(po.po_amount),0)
#   observation count = per-submission_id, label = 'Observation'
# ═══════════════════════════════════════════════════════════════════════════

GROUND_TRUTH: dict[str, dict] = {

    # ── Aggregates ───────────────────────────────────────────────────────────
    "AGG-01": {  # average inspection score
        "mode": "scalar",
        "sql": """
            SELECT ROUND(AVG(inspection_score)::numeric, 2)
            FROM inspection_report
            WHERE active = true AND status != 'DRAFT'
              AND inspection_score IS NOT NULL
        """,
    },
    "AGG-02": {  # inspections completed this year — completed = CLOSED only
        "mode": "scalar",
        "sql": """
            SELECT COUNT(*)
            FROM inspection_report
            WHERE active = true AND status = 'CLOSED'
              AND EXTRACT(YEAR FROM submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
        """,
    },
    "AGG-04": {  # open mitigative actions
        "mode": "scalar",
        "sql": """
            SELECT COUNT(*)
            FROM inspection_corrective_action
            WHERE active = true AND status = 'OPEN'
        """,
    },
    "AGG-09": {  # % of inspections with >=1 MA
        "mode": "scalar",
        "tol": {"abs": 1.5},  # definitional wiggle (DRAFT handling) allowed
        "sql": """
            SELECT ROUND(
                100.0 * COUNT(DISTINCT ica.inspection_id)
                      / NULLIF(COUNT(DISTINCT ir.id), 0), 1)
            FROM inspection_report ir
            LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id
            WHERE ir.active = true AND ir.status != 'DRAFT'
        """,
    },
    "AGG-10": {  # avg observations per inspection — per-submission, label Observation
        "mode": "scalar",
        "tol": {"abs": 0.3},
        "sql": """
            SELECT ROUND(AVG(c)::numeric, 2) FROM (
                SELECT COUNT(DISTINCT aa.submission_id) AS c
                FROM ai_answers aa
                JOIN ai_questions aq ON aa.element_id = aq.element_id
                WHERE aq.label = 'Observation'
                  AND aa.module_name = 'Inspection Form'
                GROUP BY aa.inspection_report_id
            ) s
        """,
    },
    "AGG-03": {  # total capex and opex across all mitigative actions
        "mode": "row",
        "sql": """
            SELECT COALESCE(SUM(capex), 0) AS total_capex,
                   COALESCE(SUM(opex), 0)  AS total_opex
            FROM inspection_corrective_action
            WHERE active = true
        """,
    },
    "AGG-05": {  # inspections completed this month vs last month — completed = CLOSED only
        "mode": "row",
        "sql": """
            SELECT
                COUNT(*) FILTER (WHERE date_trunc('month', submitted_on) = date_trunc('month', CURRENT_DATE)) AS this_month,
                COUNT(*) FILTER (WHERE date_trunc('month', submitted_on) = date_trunc('month', CURRENT_DATE - INTERVAL '1 month')) AS last_month
            FROM inspection_report
            WHERE active = true AND status = 'CLOSED'
        """,
    },
    "AGG-06": {  # inspections overdue this quarter — inspection_schedule.status='OVERDUE'
                 # is the authoritative field (463 rows total), not date arithmetic on
                 # inspection_report (which has no overdue concept of its own).
        "mode": "scalar",
        "sql": """
            SELECT COUNT(*)
            FROM inspection_schedule
            WHERE active = true AND status = 'OVERDUE'
              AND date_trunc('quarter', due_date) = date_trunc('quarter', CURRENT_DATE)
        """,
    },
    "AGG-07": {  # mitigative actions raised vs closed this quarter
        "mode": "row",
        "sql": """
            SELECT
                COUNT(*) FILTER (WHERE date_trunc('quarter', created_on) = date_trunc('quarter', CURRENT_DATE)) AS raised_this_quarter,
                COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED')
                                  AND date_trunc('quarter', completed_on) = date_trunc('quarter', CURRENT_DATE)) AS closed_this_quarter
            FROM inspection_corrective_action
            WHERE active = true
        """,
    },
    "AGG-08": {  # average inspection duration in hours — total_inspection_hours is
                 # entirely NULL in this DB, must compute from start/end timestamps.
        "mode": "scalar",
        "sql": """
            SELECT ROUND(AVG(EXTRACT(EPOCH FROM (end_date_time - start_date_time)) / 3600.0)::numeric, 2)
            FROM inspection_report
            WHERE active = true AND start_date_time IS NOT NULL AND end_date_time IS NOT NULL
        """,
    },
    "AGG-EXT-01": {  # average mitigative actions per inspection — averaged over
                     # inspections that have >=1 MA (matches AGG-10's established
                     # per-submission averaging convention, not all inspections).
        "mode": "scalar",
        "sql": """
            SELECT ROUND(AVG(c)::numeric, 2) FROM (
                SELECT COUNT(*) AS c
                FROM inspection_corrective_action ica
                JOIN inspection_report ir ON ica.inspection_id = ir.id
                WHERE ica.active = true AND ir.active = true
                GROUP BY ir.id
            ) s
        """,
    },
    "AGG-EXT-02": {  # mitigative actions closed on time vs late
        "mode": "row",
        "sql": """
            SELECT
                COUNT(*) FILTER (WHERE completed_on <= target_close_out_date) AS closed_on_time,
                COUNT(*) FILTER (WHERE completed_on > target_close_out_date)  AS closed_late
            FROM inspection_corrective_action
            WHERE active = true AND status IN ('CLOSED','CLOSE_WITH_DEFERRED')
              AND completed_on IS NOT NULL AND target_close_out_date IS NOT NULL
        """,
    },
    "AGG-EXT-03": {  # percentage of mitigative actions currently overdue
        "mode": "scalar",
        "tol": {"abs": 0.5},
        "sql": """
            SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'OVERDUE') / NULLIF(COUNT(*), 0), 1)
            FROM inspection_corrective_action
            WHERE active = true
        """,
    },
    "AGG-EXT-04": {  # facilities with zero inspections this year
        "mode": "scalar",
        "sql": """
            SELECT COUNT(*)
            FROM facility fac
            WHERE fac.active = true AND NOT EXISTS (
                SELECT 1 FROM inspection_report ir
                WHERE ir.facility_id = fac.id AND ir.active = true
                  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
            )
        """,
    },
    "AGG-EXT-05": {  # total observations recorded across all inspections — same
                     # per-submission counting convention as AGG-10.
        "mode": "scalar",
        "sql": """
            SELECT COUNT(DISTINCT aa.submission_id)
            FROM ai_answers aa
            JOIN ai_questions aq ON aa.element_id = aq.element_id
            WHERE aq.label = 'Observation' AND aa.module_name = 'Inspection Form'
        """,
    },

    # ── Joins / rankings ─────────────────────────────────────────────────────
    "JOI-06": {  # facility with most overdue MAs — winner + count
        "mode": "top1",
        "sql": """
            SELECT fac.name, COUNT(*) AS overdue_count
            FROM inspection_corrective_action ica
            JOIN inspection_report ir ON ica.inspection_id = ir.id
            JOIN facility fac ON ir.facility_id = fac.id
            WHERE ica.status = 'OVERDUE'
              AND ica.active = true AND ir.active = true AND fac.active = true
            GROUP BY fac.id, fac.name
            ORDER BY overdue_count DESC
            LIMIT 1
        """,
    },
    "JOI-07": {  # client with most overdue MAs — JOI-07 returned 140 vs truth 72
        "mode": "top1",
        "sql": """
            SELECT cl.name, COUNT(*) AS overdue_count
            FROM inspection_corrective_action ica
            JOIN inspection_report ir ON ica.inspection_id = ir.id
            JOIN client cl ON ir.client_id = cl.id
            WHERE ica.status = 'OVERDUE'
              AND ica.active = true AND ir.active = true
            GROUP BY cl.id, cl.name
            ORDER BY overdue_count DESC
            LIMIT 1
        """,
    },

    # ── ICA risk ─────────────────────────────────────────────────────────────
    "ICA-03": {  # how many MAs are high risk
        "mode": "scalar",
        "sql": """
            SELECT COUNT(*)
            FROM inspection_corrective_action ica
            JOIN risk_level rl ON ica.risk_level_id = rl.id
            WHERE rl.name = 'High' AND ica.active = true
        """,
    },
    "ICA-01": {  # show all high risk MAs — same shape as ICA-03, "show all" ground
                 # truth is a count since matching a free-form row listing is fragile.
        "mode": "scalar",
        "sql": """
            SELECT COUNT(*)
            FROM inspection_corrective_action ica
            JOIN risk_level rl ON ica.risk_level_id = rl.id
            WHERE rl.name = 'High' AND ica.active = true
        """,
    },
    "ICA-02": {  # risk level breakdown of all MAs — GROUP BY rl.name (human-readable)
        "mode": "set",
        "sql": """
            SELECT rl.name, COUNT(*) AS ma_count
            FROM inspection_corrective_action ica
            JOIN risk_level rl ON ica.risk_level_id = rl.id
            WHERE ica.active = true
            GROUP BY rl.name
        """,
    },
    "ICA-04": {  # open + high-risk MAs grouped by facility (3-table join)
        "mode": "set",
        "sql": """
            SELECT fac.name, COUNT(*) AS open_high_risk_count
            FROM inspection_corrective_action ica
            JOIN risk_level rl ON ica.risk_level_id = rl.id
            JOIN inspection_report ir ON ica.inspection_id = ir.id
            JOIN facility fac ON ir.facility_id = fac.id
            WHERE rl.name = 'High' AND ica.status = 'OPEN'
              AND ica.active = true AND ir.active = true AND fac.active = true
            GROUP BY fac.id, fac.name
        """,
    },

    # ── Financial (P&L ground truth confirmed in pgAdmin) ────────────────────
    "FIN-01": {  # capex generated / converted / lost — one row, three numbers
        "mode": "row",
        "sql": """
            SELECT
                (SELECT COALESCE(SUM(capex), 0)
                 FROM inspection_corrective_action WHERE active = true) AS capex_generated,
                (SELECT COALESCE(SUM(po_amount), 0) FROM po_details)    AS capex_converted,
                (SELECT COALESCE(SUM(capex), 0)
                 FROM inspection_corrective_action WHERE active = true)
                - (SELECT COALESCE(SUM(po_amount), 0) FROM po_details)  AS capex_lost
        """,
    },
    "FIN-06": {  # total PO amount this year + count
        "mode": "row",
        "sql": """
            SELECT COALESCE(SUM(po_amount), 0) AS total_po_amount,
                   COUNT(*)                    AS po_count
            FROM po_details
            WHERE EXTRACT(YEAR FROM po_date) = EXTRACT(YEAR FROM CURRENT_DATE)
        """,
    },
    "FIN-02": {  # total opex incurred from purchase requisitions — SUM(pr.pr_amount), NOT ica.opex
        "mode": "scalar",
        "sql": "SELECT COALESCE(SUM(pr_amount), 0) FROM pr_details",
    },
    "FIN-03": {  # which FM manager generates the most capex — FM managers =
                 # inspectee_user_id (NOT inspector_user_id)
        "mode": "top1",
        "sql": """
            SELECT (u.first_name || ' ' || u.last_name) AS fm_manager,
                   SUM(ica.capex) AS total_capex
            FROM inspection_corrective_action ica
            JOIN inspection_report ir ON ica.inspection_id = ir.id
            JOIN users u ON ir.inspectee_user_id = u.id
            WHERE ica.active = true AND ica.capex IS NOT NULL
            GROUP BY u.id, u.first_name, u.last_name
            ORDER BY total_capex DESC
            LIMIT 1
        """,
    },
    "FIN-04": {  # capex lost from rejected mitigative actions — capex_status = 'REJECTED'
        "mode": "scalar",
        "sql": """
            SELECT COALESCE(SUM(capex), 0)
            FROM inspection_corrective_action
            WHERE active = true AND capex_status = 'REJECTED'
        """,
    },
    "FIN-05": {  # total capex converted (po_details) and opex incurred (pr_details) by
                 # facility — po/pr joined via ica -> ir -> facility. Only facilities
                 # with at least one PO or PR are included (matching the question's
                 # "from po and pr details" framing, not every facility).
        "mode": "set",
        "sql": """
            SELECT fac.name,
                   COALESCE(po_sum.total_po, 0) AS total_capex_converted,
                   COALESCE(pr_sum.total_pr, 0) AS total_opex_incurred
            FROM facility fac
            JOIN inspection_report ir ON ir.facility_id = fac.id
            JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id
            LEFT JOIN (
                SELECT mitigative_action_uuid, SUM(po_amount) AS total_po
                FROM po_details GROUP BY mitigative_action_uuid
            ) po_sum ON po_sum.mitigative_action_uuid = ica.id
            LEFT JOIN (
                SELECT mitigative_action_uuid, SUM(pr_amount) AS total_pr
                FROM pr_details GROUP BY mitigative_action_uuid
            ) pr_sum ON pr_sum.mitigative_action_uuid = ica.id
            WHERE fac.active = true AND ica.active = true
              AND (po_sum.total_po IS NOT NULL OR pr_sum.total_pr IS NOT NULL)
            GROUP BY fac.id, fac.name, po_sum.total_po, pr_sum.total_pr
        """,
    },
    "FIN-07": {  # capex generated vs capex converted by FM manager — FM managers
                 # (inspectee_user_id) generate capex via MAs, NOT inspector_user_id.
                 # generated = SUM(ica.capex), converted = SUM(po.po_amount).
        "mode": "set",
        "sql": """
            SELECT (u.first_name || ' ' || u.last_name) AS fm_manager,
                   SUM(ica.capex) AS capex_generated,
                   COALESCE(SUM(po_sum.total_po), 0) AS capex_converted
            FROM inspection_corrective_action ica
            JOIN inspection_report ir ON ica.inspection_id = ir.id
            JOIN users u ON ir.inspectee_user_id = u.id
            LEFT JOIN (
                SELECT mitigative_action_uuid, SUM(po_amount) AS total_po
                FROM po_details GROUP BY mitigative_action_uuid
            ) po_sum ON po_sum.mitigative_action_uuid = ica.id
            WHERE ica.active = true
            GROUP BY u.id, u.first_name, u.last_name
        """,
    },

    # ── Edge / portfolio ─────────────────────────────────────────────────────
    "EDG-02": {  # questions in the inspection form
        "mode": "scalar",
        "sql": "SELECT COUNT(*) FROM ai_questions",
    },
    "PORT-01": {  # inspectors with portfolios
        "mode": "scalar",
        "sql": """
            SELECT COUNT(DISTINCT ip.inspector_id)
            FROM inspector_portfolio ip
            WHERE ip.active = true
        """,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Value normalization + comparison
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_ABS_TOL = 0.02
DEFAULT_REL_TOL = 0.01


def _norm(v: Any) -> Any:
    """Normalize a cell value for comparison."""
    if v is None:
        return None
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, str):
        s = v.strip()
        # numeric strings → numbers ("86.38", "1,387,935.80")
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return s.lower()
    return str(v)


def _values(row: Any) -> list:
    """Row (dict or sequence) → ordered list of normalized values."""
    if isinstance(row, dict):
        return [_norm(v) for v in row.values()]
    return [_norm(v) for v in row]


def _num_match(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) or \
           abs(a - b) <= max(abs_tol, rel_tol * abs(b))


def _cell_match(a: Any, b: Any, abs_tol: float, rel_tol: float) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        return _num_match(a, b, abs_tol, rel_tol)
    return a == b


def _row_multiset_match(truth_vals: list, agent_vals: list,
                        abs_tol: float, rel_tol: float) -> bool:
    """Match values as a multiset (column order/name agnostic) — used for
    'row' mode where the agent may order/alias columns differently."""
    remaining = list(agent_vals)
    for tv in truth_vals:
        hit = next((i for i, av in enumerate(remaining)
                    if _cell_match(tv, av, abs_tol, rel_tol)), None)
        if hit is None:
            return False
        remaining.pop(hit)
    return len(remaining) == 0 or all(
        # agent may include derived extras (e.g. conversion_pct); allow leftovers
        True for _ in remaining
    )


def compare_results(mode: str, truth_rows: list, agent_rows: list,
                    abs_tol: float = DEFAULT_ABS_TOL,
                    rel_tol: float = DEFAULT_REL_TOL) -> tuple[bool, str]:
    """
    Returns (match, detail). Column-name agnostic throughout.
    """
    if not truth_rows:
        return (not agent_rows, "truth empty" if not agent_rows
                else f"truth empty but agent returned {len(agent_rows)} rows")
    if not agent_rows:
        return False, "agent returned no rows"

    if mode == "scalar":
        tv = _values(truth_rows[0])
        av = _values(agent_rows[0])
        if len(truth_rows) != 1 or len(tv) != 1:
            return False, "truth is not scalar — fix GROUND_TRUTH entry"
        if len(agent_rows) != 1 or len(av) != 1:
            return False, (f"expected scalar, agent returned "
                           f"{len(agent_rows)} row(s) × {len(av)} col(s)")
        ok = _cell_match(tv[0], av[0], abs_tol, rel_tol)
        return ok, f"truth={tv[0]} agent={av[0]}"

    if mode == "row":
        if len(agent_rows) != 1:
            return False, f"expected 1 row, agent returned {len(agent_rows)}"
        tv, av = _values(truth_rows[0]), _values(agent_rows[0])
        ok = _row_multiset_match(tv, av, abs_tol, rel_tol)
        return ok, f"truth={tv} agent={av}"

    if mode == "top1":
        tv = _values(truth_rows[0])
        av = _values(agent_rows[0])
        # every truth value (label + count) must appear in the agent's first row
        ok = _row_multiset_match(tv, av, abs_tol, rel_tol)
        return ok, f"truth_top={tv} agent_top={av}"

    if mode == "set":
        if len(truth_rows) != len(agent_rows):
            return False, f"row count: truth={len(truth_rows)} agent={len(agent_rows)}"
        t_sorted = sorted((tuple(_values(r)) for r in truth_rows), key=repr)
        a_sorted = sorted((tuple(_values(r)) for r in agent_rows), key=repr)
        for tr, ar in zip(t_sorted, a_sorted):
            if len(tr) != len(ar) or not all(
                    _cell_match(x, y, abs_tol, rel_tol) for x, y in zip(tr, ar)):
                return False, f"row mismatch: truth={tr} agent={ar}"
        return True, f"{len(truth_rows)} rows matched"

    return False, f"unknown mode {mode!r}"


# ═══════════════════════════════════════════════════════════════════════════
# Eval integration
# ═══════════════════════════════════════════════════════════════════════════

def extract_agent_rows(result) -> list:
    """Last successful execute_sql step's rows from an orchestrator result."""
    rows = []
    for s in getattr(result, "steps", []):
        if s.tool == "execute_sql" and s.result.get("success"):
            rows = s.result.get("rows") or []
    return rows


def run_truth_check(db_engine, case_id: str, agent_rows: list) -> dict:
    """
    Execute the canonical SQL for case_id and compare with agent_rows.
    Returns {"covered": bool, "truth_match": bool|None, "detail": str}.
    Never raises — truth-layer failures must not crash the eval.
    """
    spec = GROUND_TRUTH.get(case_id)
    if not spec:
        return {"covered": False, "truth_match": None, "detail": ""}
    try:
        from sqlalchemy import text
        with db_engine.connect() as conn:
            res = conn.execute(text(spec["sql"]))
            cols = list(res.keys())
            truth_rows = [dict(zip(cols, r)) for r in res.fetchall()]
        tol = spec.get("tol", {})
        ok, detail = compare_results(
            spec["mode"], truth_rows, agent_rows,
            abs_tol=tol.get("abs", DEFAULT_ABS_TOL),
            rel_tol=tol.get("rel", DEFAULT_REL_TOL),
        )
        return {"covered": True, "truth_match": ok, "detail": detail}
    except Exception as exc:  # noqa: BLE001 — eval must survive truth errors
        return {"covered": True, "truth_match": None,
                "detail": f"truth-layer error: {exc}"}


# ═══════════════════════════════════════════════════════════════════════════
# Self-test — comparison logic only, no DB
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    T = [
        # (name, mode, truth, agent, expect)
        ("scalar exact",       "scalar", [{"c": 264}], [{"high_risk_ca_count": 264}], True),
        ("scalar wrong",       "scalar", [{"c": 264}], [{"x": 230}], False),
        ("scalar Decimal/str", "scalar", [{"c": Decimal("86.38")}], [{"avg": "86.38"}], True),
        ("scalar tolerance",   "scalar", [{"c": 66.8}], [{"p": 67.7}], False),
        ("scalar non-scalar",  "scalar", [{"c": 1}], [{"a": 1, "b": 2}], False),
        ("row multiset",       "row",
         [{"g": 1387935.80, "c": 684243.49, "l": 703692.31}],
         [{"capex_lost": "703692.31", "capex_generated": 1387935.8,
           "capex_converted": Decimal("684243.49")}], True),
        ("row wrong value",    "row",
         [{"g": 1387935.80, "c": 684243.49}],
         [{"a": 1387935.80, "b": 999.0}], False),
        ("top1 winner match",  "top1",
         [{"name": "Kingfield", "n": 72}],
         [{"client_name": "kingfield", "overdue_count": 72}], True),
        ("top1 wrong count",   "top1",
         [{"name": "Kingfield", "n": 72}],
         [{"client_name": "Kingfield", "overdue_count": 140}], False),
        ("top1 wrong winner",  "top1",
         [{"name": "ADJD AA -Al Roya", "n": 45}],
         [{"f": "Awqaf Tower", "n": 45}], False),
        ("set match shuffled", "set",
         [{"a": "x", "n": 1}, {"a": "y", "n": 2}],
         [{"k": "Y", "v": 2}, {"k": "X", "v": 1}], True),
        ("set count mismatch", "set",
         [{"a": "x"}], [{"a": "x"}, {"a": "y"}], False),
        ("agent empty",        "scalar", [{"c": 5}], [], False),
    ]
    failed = 0
    for name, mode, truth, agent, expect in T:
        ok, detail = compare_results(mode, truth, agent)
        mark = "✓" if ok == expect else "✗ FAIL"
        if ok != expect:
            failed += 1
        print(f"{mark} {name:22s} got={ok} want={expect}  ({detail})")
    print(f"\n{len(T) - failed}/{len(T)} self-tests passed")
    raise SystemExit(1 if failed else 0)