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
    "AGG-02": {  # this year's inspections split completed vs conducted. The agent
                 # answers "how many completed this year" with the documented
                 # conducted/completed category summary (core_context status
                 # block), so the ground truth is that 2-row shape, not a bare
                 # scalar. completed = CLOSED; conducted = every other non-DRAFT.
        "mode": "set",
        "sql": """
            SELECT CASE WHEN status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category,
                   COUNT(*) AS n
            FROM inspection_report
            WHERE active = true AND status <> 'DRAFT'
              AND EXTRACT(YEAR FROM submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
            GROUP BY 1
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
    "AGG-05": {  # inspections this month vs last month, split by category — the
                 # agent returns the conducted/completed summary crossed with the
                 # month FILTER (2 rows × {category, this_month, last_month}).
        "mode": "set",
        "sql": """
            SELECT CASE WHEN status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category,
                   COUNT(*) FILTER (WHERE date_trunc('month', submitted_on) = date_trunc('month', CURRENT_DATE)) AS this_month,
                   COUNT(*) FILTER (WHERE date_trunc('month', submitted_on) = date_trunc('month', CURRENT_DATE - INTERVAL '1 month')) AS last_month
            FROM inspection_report
            WHERE active = true AND status <> 'DRAFT'
            GROUP BY 1
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
    "AGG-07": {  # mitigative actions raised vs closed this quarter — FLOW metric:
                 # raised = created_on in quarter; closed = completed_on in
                 # quarter (regardless of when raised). KNOWN DEFECT: the agent
                 # computes a COHORT metric instead (of the MAs raised this
                 # quarter, how many are now closed = 23 vs flow 13) and adds a
                 # still_open column. row mode tolerates the extra column; the
                 # 23-vs-13 value gap is a real agent divergence, left failing
                 # on purpose so the eval surfaces it.
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
    "AGG-08": {  # average inspection duration in hours = end_date_time -
                 # start_date_time (total_inspection_hours is 100% NULL here).
                 # KNOWN DEFECT: the agent averages submitted_on - start_date_time
                 # (= 3.59h, includes report write-up time) instead of the
                 # inspection window itself (= 8.36h). Left failing on purpose.
        "mode": "scalar",
        "sql": """
            SELECT ROUND(AVG(EXTRACT(EPOCH FROM (end_date_time - start_date_time)) / 3600.0)::numeric, 2)
            FROM inspection_report
            WHERE active = true AND start_date_time IS NOT NULL AND end_date_time IS NOT NULL
              AND end_date_time > start_date_time
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
    # ICA-01 ("show all high risk MAs") intentionally has NO ground truth: it is
    # a free-form listing (74 rows, over the agent's 50-row display cap) so a
    # count approximation can't match either. ICA-03 covers the count.
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
    "FIN-05": {  # total capex converted (po_details) and opex incurred (pr_details)
                 # per facility — po/pr pre-aggregated per MA, then summed per
                 # facility via ica -> ir -> facility. HAVING keeps facilities
                 # with at least one PO or PR. subset=True: the agent may also
                 # list facilities that have an MA but no PO/PR (all-zero rows).
        "mode": "set",
        "subset": True,
        "sql": """
            SELECT fac.name,
                   COALESCE(SUM(po_sum.total_po), 0) AS total_capex_converted,
                   COALESCE(SUM(pr_sum.total_pr), 0) AS total_opex_incurred
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
            GROUP BY fac.id, fac.name
            HAVING SUM(po_sum.total_po) IS NOT NULL OR SUM(pr_sum.total_pr) IS NOT NULL
        """,
    },
    "FIN-07": {  # capex generated vs capex converted per FM manager who has
                 # generated capex — FM managers = inspectee_user_id (NOT
                 # inspector_user_id). generated = SUM(ica.capex), converted =
                 # SUM(po.po_amount). HAVING SUM(capex) > 0 drops managers whose
                 # MAs all carry zero capex (the agent excludes them too).
        "mode": "set",
        "subset": True,
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
            HAVING SUM(ica.capex) > 0
        """,
    },

    # ── Edge / portfolio ─────────────────────────────────────────────────────
    "EDG-02": {  # questions in THE INSPECTION FORM specifically. ai_questions has
                 # a module_name column; 'Inspection Form' is one of ~15 modules
                 # (24 rows live). The old canonical COUNT(*) over the whole table
                 # (3350) also counted every audit form's questions (QHSE Audit
                 # 980, Quality Audit 700, ...). Confirmed live 2026-08-27: the
                 # agent's answer of 24 was correct; the canonical was wrong.
        "mode": "scalar",
        "sql": "SELECT COUNT(*) FROM ai_questions WHERE module_name = 'Inspection Form'",
    },
    "PORT-01": {  # inspectors with portfolios — the inspector FK on
                  # inspector_portfolio is `user_id` (uuid); there is no
                  # `inspector_id` column on this table (that lives on
                  # inspection_schedule). Confirmed via live information_schema
                  # 2026-08-27.
        "mode": "scalar",
        "sql": """
            SELECT COUNT(DISTINCT ip.user_id)
            FROM inspector_portfolio ip
            WHERE ip.active = true
        """,
    },

    # ── New coverage: window fns / percentile / stddev / NTILE / FILTER / junction ──
    "WIN-01": {  # inspections where score fell >10 pts vs the previous inspection
                 # at the same facility (LAG window)
        "mode": "count",
        "sql": """
            WITH s AS (
                SELECT ir.id, ir.inspection_score AS sc,
                       LAG(ir.inspection_score) OVER (
                           PARTITION BY ir.facility_id ORDER BY ir.submitted_on) AS prev
                FROM inspection_report ir
                WHERE ir.active = true AND ir.status <> 'DRAFT'
                  AND ir.inspection_score IS NOT NULL
            )
            SELECT COUNT(*) FROM s WHERE prev IS NOT NULL AND sc < prev - 10
        """,
    },
    "WIN-02": {  # number of inspectors that have at least one scored inspection
                 # (ROW_NUMBER per inspector, most-recent row)
        "mode": "count",
        "sql": """
            WITH r AS (
                SELECT ir.inspector_user_id AS u,
                       ROW_NUMBER() OVER (PARTITION BY ir.inspector_user_id
                                          ORDER BY ir.submitted_on DESC) AS rn
                FROM inspection_report ir
                WHERE ir.active = true AND ir.status <> 'DRAFT'
                  AND ir.inspection_score IS NOT NULL
            )
            SELECT COUNT(*) FROM r WHERE rn = 1
        """,
    },
    "PCT-01": {  # median inspection score (PERCENTILE_CONT)
        "mode": "scalar",
        "tol": {"abs": 0.5},
        "sql": """
            SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY inspection_score)
            FROM inspection_report
            WHERE active = true AND status <> 'DRAFT' AND inspection_score IS NOT NULL
        """,
    },
    "PCT-02": {  # 90th percentile of mitigative-action capex
        "mode": "scalar",
        "tol": {"rel": 0.02},
        "sql": """
            SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY capex)
            FROM inspection_corrective_action
            WHERE active = true AND capex IS NOT NULL
        """,
    },
    "STDV-01": {  # standard deviation of inspection scores
        "mode": "scalar",
        "tol": {"abs": 0.5},
        "sql": """
            SELECT ROUND(STDDEV_SAMP(inspection_score)::numeric, 2)
            FROM inspection_report
            WHERE active = true AND status <> 'DRAFT' AND inspection_score IS NOT NULL
        """,
    },
    "STDV-02": {  # inspection type with the most variable scores (min 3 samples)
        "mode": "top1",
        "sql": """
            SELECT it.name, ROUND(STDDEV_SAMP(ir.inspection_score)::numeric, 2) AS sd
            FROM inspection_report ir
            JOIN inspection_type it ON ir.inspection_type_id = it.id
            WHERE ir.active = true AND ir.status <> 'DRAFT' AND ir.inspection_score IS NOT NULL
            GROUP BY it.name
            HAVING COUNT(*) >= 3
            ORDER BY sd DESC
            LIMIT 1
        """,
    },
    # NTL-01 has no ground truth: the agent labels NTILE buckets as free-text
    # ("q1 (lowest 25%)") so an exact row-set match is unreliable; the bucket
    # counts are right, the shape isn't comparable.
    "FIL-01": {  # per inspection type: closed vs still-in-progress counts
        "mode": "set",
        "subset": True,
        "sql": """
            SELECT it.name AS inspection_type,
                   COUNT(*) FILTER (WHERE ir.status = 'CLOSED') AS closed,
                   COUNT(*) FILTER (WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')) AS in_progress
            FROM inspection_report ir
            JOIN inspection_type it ON ir.inspection_type_id = it.id
            WHERE ir.active = true AND ir.status <> 'DRAFT'
            GROUP BY it.name
        """,
    },
    "FIL-02": {  # open / closed / overdue mitigative actions in one row
        "mode": "row",
        "sql": """
            SELECT
                COUNT(*) FILTER (WHERE status = 'OPEN')    AS open,
                COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS closed,
                COUNT(*) FILTER (WHERE status = 'OVERDUE') AS overdue
            FROM inspection_corrective_action
            WHERE active = true
        """,
    },
    "JUN-01": {  # organisations operating in each country (organisation_countries junction)
        "mode": "set",
        "sql": """
            SELECT co.name AS country, COUNT(DISTINCT oc.organisation_id) AS n
            FROM organisation_countries oc
            JOIN country co ON oc.country_id = co.id
            GROUP BY co.name
        """,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # AUDIT DOMAIN  (keys prefixed AUDIT_ — eval_audit.py looks them up that
    # way). Written against AUDIT_SEMANTICS / AUDIT_BUSINESS_RULES:
    #   audit_corrective_action.audit_id -> audit_report.id (NOT .audit_id)
    #   audit_report -> facility is 2 hops via schedule_id -> audit_schedule
    #   audit_report.audit_cycle_id / audit_type_id are DEAD (0/19) -> go via
    #     schedule_id -> audit_schedule.audit_plan_id -> audit_plan
    #   adequacy_status / implementation_status = 'APPROVE' (exact, never 'APPROVED')
    #   review.status='COMPLETED' != approved -> also action='APPROVE'
    #   approval level counts = MAX(review.level), NEVER the stale review_level table
    #   unplanned (audit_schedule.is_unplanned) -> LEFT JOIN audit_plan
    #   audit_cycle has no name column
    # ═══════════════════════════════════════════════════════════════════════

    # ── Plans ────────────────────────────────────────────────────────────────
    "AUDIT_PLAN-01": {  # list all audit plans (non-draft)
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_plan WHERE active = true AND status <> 'DRAFT'",
    },
    "AUDIT_PLAN-02": {  # audit plans still in draft
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_plan WHERE active = true AND status = 'DRAFT'",
    },
    "AUDIT_PLAN-03": {  # audit plans pending approval
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_plan WHERE active = true AND status = 'SUBMITTED'",
    },
    "AUDIT_PLAN-04": {  # approved plans for the current (ONGOING) cycle
        "mode": "scalar",
        "sql": """
            SELECT COUNT(*)
            FROM audit_plan ap
            JOIN audit_cycle acyc ON ap.cycle_id = acyc.id
            WHERE ap.active = true AND ap.status = 'APPROVED' AND acyc.status = 'ONGOING'
        """,
    },
    "AUDIT_PLAN-05": {  # who proposed the most audit plans
        "mode": "top1",
        "sql": """
            SELECT (u.first_name || ' ' || u.last_name) AS proposer, COUNT(*) AS n
            FROM audit_plan ap
            JOIN users u ON ap.user_id = u.id
            WHERE ap.active = true
            GROUP BY u.id, u.first_name, u.last_name
            ORDER BY n DESC
            LIMIT 1
        """,
    },
    "AUDIT_PLAN-06": {  # unplanned audit plans
        "mode": "scalar",
        "sql": "SELECT COUNT(*) FROM audit_plan WHERE active = true AND is_unplanned = true",
    },
    # AUDIT_PLAN-07 ("plans by category and type") has no ground truth: the
    # category x type grid is sensitive to INNER vs LEFT joins and NULL
    # category/type, making an exact row-set match unreliable.
    "AUDIT_PLAN-09": {  # audit plan with the most schedules under it
        "mode": "top1",
        "sql": """
            SELECT ap.title, COUNT(asch.id) AS n
            FROM audit_plan ap
            LEFT JOIN audit_schedule asch ON asch.audit_plan_id = ap.id AND asch.active = true
            WHERE ap.active = true
            GROUP BY ap.id, ap.title
            ORDER BY n DESC
            LIMIT 1
        """,
    },

    # ── Schedules ──────────────────────────────────────────────────────────
    "AUDIT_SCHED-01": {  # list all audit schedules (non-draft)
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_schedule WHERE active = true AND schedule_status <> 'DRAFT'",
    },
    "AUDIT_SCHED-02": {  # ongoing audit schedules
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_schedule WHERE active = true AND schedule_status = 'ONGOING'",
    },
    "AUDIT_SCHED-03": {  # awaiting auditee acceptance
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_schedule WHERE active = true AND schedule_status = 'PENDING_REVIEW'",
    },
    "AUDIT_SCHED-04": {  # schedules completed this year (follow_up_date is the
                         # schedule's own completion date proxy)
        "mode": "scalar",
        "sql": """
            SELECT COUNT(*)
            FROM audit_schedule
            WHERE active = true AND schedule_status = 'COMPLETED'
              AND EXTRACT(YEAR FROM follow_up_date) = EXTRACT(YEAR FROM CURRENT_DATE)
        """,
    },
    "AUDIT_SCHED-05": {  # auditor with the most scheduled visits
        "mode": "top1",
        "sql": """
            SELECT (u.first_name || ' ' || u.last_name) AS auditor, COUNT(*) AS n
            FROM audit_schedule asch
            JOIN users u ON asch.auditor_id = u.id
            WHERE asch.active = true
            GROUP BY u.id, u.first_name, u.last_name
            ORDER BY n DESC
            LIMIT 1
        """,
    },
    "AUDIT_SCHED-06": {  # schedules for facility Al Ghadeer
        "mode": "count",
        "sql": """
            SELECT COUNT(*)
            FROM audit_schedule asch
            JOIN facility f ON asch.facility_id = f.id
            WHERE asch.active = true AND f.name ILIKE '%Al Ghadeer%'
        """,
    },
    "AUDIT_SCHED-07": {  # rescheduled more than once
        "mode": "scalar",
        "sql": "SELECT COUNT(*) FROM audit_schedule WHERE active = true AND reschedule_count > 1",
    },
    "AUDIT_SCHED-09": {  # auditee for the most audit schedules
        "mode": "top1",
        "sql": """
            SELECT (u.first_name || ' ' || u.last_name) AS auditee, COUNT(*) AS n
            FROM audit_schedule asch
            JOIN users u ON asch.auditee_id = u.id
            WHERE asch.active = true
            GROUP BY u.id, u.first_name, u.last_name
            ORDER BY n DESC
            LIMIT 1
        """,
    },
    "AUDIT_SCHED-10": {  # schedules still in draft
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_schedule WHERE active = true AND schedule_status = 'DRAFT'",
    },

    # ── Statuses ──────────────────────────────────────────────────────────
    "AUDIT_STAT-01": {  # audit reports closed with exception
        "mode": "scalar",
        "sql": "SELECT COUNT(*) FROM audit_report WHERE active = true AND status = 'CLOSED_WITH_EXCEPTION'",
    },
    # AUDIT_STAT-02 ("what statuses can a CA have") has no ground truth: a meta
    # question the agent may answer in prose rather than as rows.
    "AUDIT_STAT-03": {  # CA counts by status (agent groups by status)
        "mode": "set",
        "subset": True,
        "sql": """
            SELECT status, COUNT(*) AS n
            FROM audit_corrective_action
            WHERE active = true AND status IN ('OVERDUE','OPEN','CLOSED','CLOSE_WITH_DEFERRED')
            GROUP BY status
        """,
    },
    "AUDIT_STAT-04": {  # audit reports still in draft
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_report WHERE active = true AND status = 'DRAFT'",
    },
    "AUDIT_STAT-05": {  # schedules per schedule_status — DRAFT excluded, matching
                        # the enforced house rule (AUDIT_SCHEDULE_DRAFT_NOT_EXCLUDED)
        "mode": "set",
        "subset": True,
        "sql": """
            SELECT schedule_status, COUNT(*) AS n
            FROM audit_schedule
            WHERE active = true AND schedule_status <> 'DRAFT'
            GROUP BY schedule_status
        """,
    },
    "AUDIT_STAT-06": {  # CAs closed vs closed-with-deferred (agent groups by status)
        "mode": "set",
        "subset": True,
        "sql": """
            SELECT status, COUNT(*) AS n
            FROM audit_corrective_action
            WHERE active = true AND status IN ('CLOSED','CLOSE_WITH_DEFERRED')
            GROUP BY status
        """,
    },

    # ── Cycles ────────────────────────────────────────────────────────────
    "AUDIT_CYCLE-01": {  # all audit cycles
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_cycle WHERE active = true",
    },
    "AUDIT_CYCLE-02": {  # currently active cycle
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_cycle WHERE active = true AND status = 'ONGOING'",
    },
    # AUDIT_CYCLE-03 ("plans proposed in each cycle") intentionally has no
    # ground truth: whether zero-plan cycles belong in the result is genuinely
    # ambiguous (LEFT vs INNER JOIN), so it stays real_pass-only.
    "AUDIT_CYCLE-04": {  # cycles overdue past due date (excluding COMPLETED) — 0
        "mode": "count",
        "sql": """
            SELECT COUNT(*)
            FROM audit_cycle
            WHERE active = true AND due_date < CURRENT_DATE AND status <> 'COMPLETED'
        """,
    },
    "AUDIT_CYCLE-05": {  # plans / schedules / reports in the ONGOING cycle (one row)
        "mode": "row",
        "sql": """
            SELECT
                (SELECT COUNT(*) FROM audit_plan ap
                   JOIN audit_cycle c ON ap.cycle_id = c.id
                   WHERE ap.active AND c.status = 'ONGOING')                    AS plans,
                (SELECT COUNT(*) FROM audit_schedule asch
                   JOIN audit_plan ap ON asch.audit_plan_id = ap.id
                   JOIN audit_cycle c ON ap.cycle_id = c.id
                   WHERE asch.active AND c.status = 'ONGOING')                  AS schedules,
                (SELECT COUNT(*) FROM audit_report arep
                   JOIN audit_schedule asch ON arep.schedule_id = asch.id
                   JOIN audit_plan ap ON asch.audit_plan_id = ap.id
                   JOIN audit_cycle c ON ap.cycle_id = c.id
                   WHERE arep.active AND c.status = 'ONGOING')                  AS reports
        """,
    },

    # ── Forms ─────────────────────────────────────────────────────────────
    "AUDIT_FORMS-01": {  # all audit form templates
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_forms WHERE active = true",
    },
    "AUDIT_FORMS-02": {  # forms in draft status AND still active (the one DRAFT
                         # row is soft-deleted -> expect 0)
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_forms WHERE active = true AND status = 'DRAFT'",
    },
    "AUDIT_FORMS-03": {  # form templates per audit type
        "mode": "set",
        "sql": """
            SELECT aty.name AS audit_type, COUNT(*) AS n
            FROM audit_forms af
            JOIN audit_type aty ON af.audit_type_id = aty.id
            WHERE af.active = true
            GROUP BY aty.name
        """,
    },

    # ── Admin ─────────────────────────────────────────────────────────────
    "AUDIT_ADMIN-01": {  # audits conducted — per AUDIT_REPORT_SEMANTICS,
                         # conducted = status 'SUBMITTED' only (CLOSED /
                         # CLOSED_WITH_EXCEPTION are 'completed', DRAFT is never
                         # counted). "status != 'DRAFT'" is a listed forbidden
                         # filter for this concept.
        "mode": "scalar",
        "sql": "SELECT COUNT(*) FROM audit_report WHERE active = true AND status = 'SUBMITTED'",
    },
    "AUDIT_ADMIN-02": {  # audit universe with the most audit plans (via schedule)
        "mode": "top1",
        "sql": """
            SELECT au.name AS universe, COUNT(DISTINCT ap.id) AS n
            FROM audit_universe au
            JOIN audit_schedule asch ON asch.audit_universe_id = au.id
            JOIN audit_plan ap ON asch.audit_plan_id = ap.id
            WHERE au.active = true AND ap.active = true
            GROUP BY au.id, au.name
            ORDER BY n DESC
            LIMIT 1
        """,
    },
    "AUDIT_ADMIN-03": {  # audits with no corrective action raised
        "mode": "count",
        "sql": """
            SELECT COUNT(*)
            FROM audit_report arep
            WHERE arep.active = true
              AND NOT EXISTS (SELECT 1 FROM audit_corrective_action aca
                              WHERE aca.audit_id = arep.id AND aca.active = true)
        """,
    },
    "AUDIT_ADMIN-04": {  # facility audited the most (2-hop via schedule)
        "mode": "top1",
        "sql": """
            SELECT f.name, COUNT(*) AS n
            FROM audit_report arep
            JOIN audit_schedule asch ON arep.schedule_id = asch.id
            JOIN facility f ON asch.facility_id = f.id
            WHERE arep.active = true
            GROUP BY f.id, f.name
            ORDER BY n DESC
            LIMIT 1
        """,
    },
    "AUDIT_ADMIN-05": {  # distinct active auditors (from schedule, not role table)
        "mode": "count",
        "sql": "SELECT COUNT(DISTINCT auditor_id) FROM audit_schedule WHERE active = true AND auditor_id IS NOT NULL",
    },
    "AUDIT_ADMIN-06": {  # categories and sub-categories counts (one row)
        "mode": "row",
        "sql": """
            SELECT
                (SELECT COUNT(*) FROM audit_category WHERE active = true)     AS categories,
                (SELECT COUNT(*) FROM audit_sub_category WHERE active = true) AS sub_categories
        """,
    },
    "AUDIT_ADMIN-07": {  # average audit score — DRAFT reports are never_counted
                         # per AUDIT_REPORT_SEMANTICS
        "mode": "scalar",
        "tol": {"abs": 1.0},
        "sql": """
            SELECT ROUND(AVG(audit_score)::numeric, 2)
            FROM audit_report
            WHERE active = true AND status <> 'DRAFT' AND audit_score IS NOT NULL
        """,
    },
    "AUDIT_ADMIN-08": {  # audit schedules per country (facility.country_id)
        "mode": "set",
        "sql": """
            SELECT co.name AS country, COUNT(*) AS n
            FROM audit_schedule asch
            JOIN facility f ON asch.facility_id = f.id
            JOIN country co ON f.country_id = co.id
            WHERE asch.active = true
            GROUP BY co.name
        """,
    },

    # ── Portfolios ────────────────────────────────────────────────────────
    # AUDIT_PORT-04 ("portfolios submitted this cycle") has no ground truth:
    # auditor_portfolio has no SUBMITTED status, so "submitted" is ambiguous.
    "AUDIT_PORT-05": {  # auditor covering the most facilities in their portfolio
        "mode": "top1",
        "sql": """
            SELECT (u.first_name || ' ' || u.last_name) AS auditor,
                   COUNT(DISTINCT pod.facility_id) AS n
            FROM auditor_portfolio po
            JOIN users u ON po.user_id = u.id
            JOIN auditor_portfolio_detail pod ON pod.portfolio_id = po.id
            WHERE po.active = true AND pod.active = true AND pod.facility_id IS NOT NULL
            GROUP BY u.id, u.first_name, u.last_name
            ORDER BY n DESC
            LIMIT 1
        """,
    },
    # AUDIT_PORT-06 ("projects per auditor portfolio") has no ground truth: the
    # agent tends to SELECT the portfolio UUID as an extra column, which set
    # mode (column-count strict) can't reconcile.
    "AUDIT_PORT-07": {  # portfolios still pending submission
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM auditor_portfolio WHERE active = true AND status IN ('DRAFT','PENDING')",
    },

    # ── Approval ──────────────────────────────────────────────────────────
    "AUDIT_APPR-02": {  # levels of approval for an audit corrective action
        "mode": "scalar",
        "sql": """
            SELECT MAX(level) FROM review
            WHERE entity_type = 'AUDIT_CORRECTIVE_ACTION' AND review_type = 'IMPLEMENTATION'
        """,
    },
    "AUDIT_APPR-03": {  # levels of approval for an audit plan
        "mode": "scalar",
        "sql": "SELECT MAX(level) FROM review WHERE entity_type = 'AUDIT_PLAN'",
    },
    "AUDIT_APPR-04": {  # audit plans returned for modification
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_plan WHERE active = true AND status = 'RETURNED_FOR_MODIFICATION'",
    },
    "AUDIT_APPR-06": {  # CAs that passed level-2 implementation sign-off
        "mode": "count",
        "sql": """
            SELECT COUNT(DISTINCT entity_id)
            FROM review
            WHERE entity_type = 'AUDIT_CORRECTIVE_ACTION' AND review_type = 'IMPLEMENTATION'
              AND level = 2 AND status = 'COMPLETED' AND action = 'APPROVE'
        """,
    },
    "AUDIT_APPR-07": {  # audit plans pending at each review level
        "mode": "set",
        "sql": """
            SELECT level, COUNT(*) AS n
            FROM review
            WHERE entity_type = 'AUDIT_PLAN' AND status = 'PENDING'
            GROUP BY level
        """,
    },
    "AUDIT_APPR-08": {  # % of completed reviews that were rejections
        "mode": "scalar",
        "tol": {"abs": 1.0},
        "sql": """
            SELECT ROUND(
                100.0 * COUNT(*) FILTER (WHERE action = 'RETURN_FOR_MODIFICATION')
                      / NULLIF(COUNT(*) FILTER (WHERE status = 'COMPLETED'), 0), 1)
            FROM review
        """,
    },
    "AUDIT_APPR-10": {  # approved auditor portfolios via review (entity_type PORTFOLIO)
        "mode": "count",
        "sql": """
            SELECT COUNT(DISTINCT entity_id)
            FROM review
            WHERE entity_type = 'PORTFOLIO' AND status = 'COMPLETED' AND action = 'APPROVE'
        """,
    },
    # APPR-12 / APPR-14 / APPR-15 have NO ground truth: they are listings at
    # CA x review-level grain ("show all CAs AND who approved each level" /
    # "...and by whom"), so the agent's row count is legitimately > the number
    # of distinct CAs and a COUNT(DISTINCT ca) truth can never match. The
    # distinct-CA assertion they were meant to carry is already covered by
    # APPR-06 (level-2 sign-off count).
    "AUDIT_APPR-13": {  # CAs with implementation_status APPROVE (exact enum) —
                        # listing whose row count should equal this filtered
                        # count; catches the implementation-vs-adequacy trap.
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_corrective_action WHERE active = true AND implementation_status = 'APPROVE'",
    },

    # ── Workflow ──────────────────────────────────────────────────────────
    "AUDIT_WORKFLOW-01": {  # unplanned audit schedules
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM audit_schedule WHERE active = true AND is_unplanned = true",
    },
    # AUDIT_WORKFLOW-02 ("unplanned vs tied to an approved plan") has no ground
    # truth: "tied to an approved plan" is sensitive to LEFT-vs-INNER join and
    # to whether DRAFT schedules are in scope, so the exact split isn't stable.
    "AUDIT_WORKFLOW-04": {  # audit plans whose schedules span >1 facility
        "mode": "count",
        "sql": """
            SELECT COUNT(*) FROM (
                SELECT audit_plan_id
                FROM audit_schedule
                WHERE active = true AND audit_plan_id IS NOT NULL
                GROUP BY audit_plan_id
                HAVING COUNT(DISTINCT facility_id) > 1
            ) s
        """,
    },
    "AUDIT_WORKFLOW-05": {  # auditor portfolios under modification post-approval
        "mode": "count",
        "sql": "SELECT COUNT(*) FROM auditor_portfolio WHERE active = true AND status = 'MODIFICATION'",
    },
    "AUDIT_WORKFLOW-06": {  # CAs on track / behind / at risk with adequacy approved
                            # (confirmed live: none — set mode handles truth == [])
        "mode": "set",
        "sql": """
            SELECT progress_stage, COUNT(*) AS n
            FROM audit_corrective_action
            WHERE active = true AND adequacy_status = 'APPROVE'
              AND progress_stage IN ('ON_TRACK', 'BEHIND', 'AT_RISK')
            GROUP BY progress_stage
        """,
    },
    "AUDIT_WORKFLOW-07": {  # CAs that needed a 2nd level of adequacy approval
        "mode": "count",
        "sql": """
            SELECT COUNT(DISTINCT entity_id)
            FROM review
            WHERE entity_type = 'AUDIT_CORRECTIVE_ACTION' AND review_type = 'ADEQUACY' AND level = 2
        """,
    },
    "AUDIT_WORKFLOW-08": {  # both tracks approved but not yet closed
        "mode": "count",
        "sql": """
            SELECT COUNT(*)
            FROM audit_corrective_action
            WHERE active = true
              AND implementation_status = 'APPROVE' AND adequacy_status = 'APPROVE'
              AND status NOT IN ('CLOSED', 'CLOSE_WITH_DEFERRED')
        """,
    },

    # ── Multi-turn (only the fully-specified turns are truth-checkable) ────
    "AUDIT_MULTI-01": {  # turn 1: CAs with both level-1 & level-2 implementation reviewers
        "mode": "count",
        "sql": """
            SELECT COUNT(DISTINCT r1.entity_id)
            FROM review r1
            JOIN review r2 ON r2.entity_id = r1.entity_id
            WHERE r1.entity_type = 'AUDIT_CORRECTIVE_ACTION' AND r1.review_type = 'IMPLEMENTATION'
              AND r1.level = 1
              AND r2.entity_type = 'AUDIT_CORRECTIVE_ACTION' AND r2.review_type = 'IMPLEMENTATION'
              AND r2.level = 2
        """,
    },
    "AUDIT_MULTI-02": {  # turn 2: of that set, how many are CLOSED
        "mode": "scalar",
        "sql": """
            SELECT COUNT(*)
            FROM audit_corrective_action aca
            WHERE aca.active = true AND aca.status = 'CLOSED'
              AND aca.id IN (
                SELECT r1.entity_id
                FROM review r1
                JOIN review r2 ON r2.entity_id = r1.entity_id
                WHERE r1.entity_type = 'AUDIT_CORRECTIVE_ACTION' AND r1.review_type = 'IMPLEMENTATION' AND r1.level = 1
                  AND r2.entity_type = 'AUDIT_CORRECTIVE_ACTION' AND r2.review_type = 'IMPLEMENTATION' AND r2.level = 2
              )
        """,
    },

    # ── New audit coverage: window / percentile / stddev / NTILE / FILTER /
    #    junction / observation-reconstruction / unplanned-chain ─────────────
    "AUDIT_WIN-01": {  # audits whose score fell vs the previous audit at the
                       # same facility (LAG; facility via schedule)
        "mode": "count",
        "sql": """
            WITH s AS (
                SELECT arep.id, arep.audit_score AS sc,
                       LAG(arep.audit_score) OVER (
                           PARTITION BY asch.facility_id ORDER BY arep.submitted_on) AS prev
                FROM audit_report arep
                JOIN audit_schedule asch ON arep.schedule_id = asch.id
                WHERE arep.active = true AND arep.audit_score IS NOT NULL
            )
            SELECT COUNT(*) FROM s WHERE prev IS NOT NULL AND sc < prev
        """,
    },
    "AUDIT_PCT-01": {  # median audit score
        "mode": "scalar",
        "tol": {"abs": 0.5},
        "sql": """
            SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY audit_score)
            FROM audit_report
            WHERE active = true AND audit_score IS NOT NULL
        """,
    },
    "AUDIT_STDV-01": {  # standard deviation of audit scores
        "mode": "scalar",
        "tol": {"abs": 0.5},
        "sql": """
            SELECT ROUND(STDDEV_SAMP(audit_score)::numeric, 2)
            FROM audit_report
            WHERE active = true AND audit_score IS NOT NULL
        """,
    },
    # AUDIT_NTL-01 has no ground truth: the agent's NTILE output shape (quartile
    # labels vs numbers, pivoted vs long) varies too much for a row-set match.
    "AUDIT_FIL-01": {  # per schedule_status: unplanned vs planned counts —
                       # DRAFT excluded per the enforced house rule
        "mode": "set",
        "subset": True,
        "sql": """
            SELECT schedule_status,
                   COUNT(*) FILTER (WHERE is_unplanned = true)  AS unplanned,
                   COUNT(*) FILTER (WHERE is_unplanned = false) AS planned
            FROM audit_schedule
            WHERE active = true AND schedule_status <> 'DRAFT'
            GROUP BY schedule_status
        """,
    },
    "AUDIT_FIL-02": {  # audit CA counts by status — agent groups by status
        "mode": "set",
        "subset": True,
        "sql": """
            SELECT status, COUNT(*) AS n
            FROM audit_corrective_action
            WHERE active = true
              AND status IN ('OPEN','OVERDUE','CLOSED','CLOSE_WITH_DEFERRED')
            GROUP BY status
        """,
    },
    "AUDIT_JUN-01": {  # auditor portfolios covering each audit category
                       # (portfolio_detail_audit_category junction, NOT the dead
                       # audit_category_portfolio table)
        "mode": "set",
        "sql": """
            SELECT ac.name AS category, COUNT(DISTINCT pod.portfolio_id) AS n
            FROM portfolio_detail_audit_category pdac
            JOIN auditor_portfolio_detail pod ON pdac.detail_id = pod.id
            JOIN audit_category ac ON pdac.audit_category_id = ac.id
            GROUP BY ac.name
        """,
    },
    "AUDIT_JUN-03": {  # audit form templates that have at least one facility mapping
        "mode": "count",
        "sql": """
            SELECT COUNT(DISTINCT af.id)
            FROM audit_forms af
            JOIN audit_form_facilities aff ON aff.audit_form_id = af.id
            WHERE af.active = true
        """,
    },
    "AUDIT_APPR-EXT-01": {  # audit CAs with adequacy_status = 'APPROVE' (exact enum)
        "mode": "scalar",
        "sql": "SELECT COUNT(*) FROM audit_corrective_action WHERE active = true AND adequacy_status = 'APPROVE'",
    },
    # APPR-EXT-02 ("which CA reviews were marked completed but returned for
    # modification") has NO ground truth: it is a listing of review rows, and
    # the agent's grain (CA rows vs review rows) is not pinned by the wording.
    "AUDIT_APPR-EXT-03": {  # % of completed audit-plan reviews that were approvals
        "mode": "scalar",
        "tol": {"abs": 1.0},
        "sql": """
            SELECT ROUND(
                100.0 * COUNT(*) FILTER (WHERE action = 'APPROVE')
                      / NULLIF(COUNT(*) FILTER (WHERE status = 'COMPLETED'), 0), 1)
            FROM review
            WHERE entity_type = 'AUDIT_PLAN'
        """,
    },
    "AUDIT_OBS-01": {  # highest number of corrective actions raised against a
                       # single audit observation
        "mode": "scalar",
        "sql": """
            SELECT MAX(n) FROM (
                SELECT COUNT(*) AS n
                FROM audit_corrective_action
                WHERE active = true AND observation_id IS NOT NULL
                GROUP BY observation_id
            ) s
        """,
    },
    "AUDIT_OBS-02": {  # audit CAs linked to a form observation (cast join to
                       # ai_audit_answers.submission_id)
        "mode": "scalar",
        "sql": """
            SELECT COUNT(*)
            FROM audit_corrective_action aca
            WHERE aca.active = true AND aca.observation_id IS NOT NULL
              AND EXISTS (SELECT 1 FROM ai_audit_answers aaa
                          WHERE aaa.submission_id = aca.observation_id::varchar)
        """,
    },
    "AUDIT_UNP-02": {  # average audit score for unplanned audits
        "mode": "scalar",
        "tol": {"abs": 0.5},
        "sql": """
            SELECT ROUND(AVG(arep.audit_score)::numeric, 2)
            FROM audit_report arep
            JOIN audit_schedule asch ON arep.schedule_id = asch.id
            WHERE arep.active = true AND asch.is_unplanned = true
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
                    rel_tol: float = DEFAULT_REL_TOL,
                    subset: bool = False) -> tuple[bool, str]:
    """
    Returns (match, detail). Column-name agnostic throughout.

    `subset` (set mode only): pass when every truth row appears in the agent's
    result, tolerating extra agent rows. Use for "list the X" cases where the
    canonical pins the rows that MUST be present but the agent may legitimately
    include more (e.g. all-zero rows the canonical filtered out).
    """
    # ── count mode ── truth is a single COUNT(*) scalar N; the agent passes if
    # it returned N rows (a listing) OR a single scalar row equal to N. Handled
    # before the empty guards so N==0 with an empty agent result can match.
    if mode == "count":
        tv = _values(truth_rows[0]) if truth_rows else []
        if len(truth_rows) != 1 or len(tv) != 1 or not isinstance(tv[0], float):
            return False, "truth is not a scalar count — fix GROUND_TRUTH entry"
        n = int(round(tv[0]))
        if len(agent_rows) == 1:
            av = _values(agent_rows[0])
            if any(isinstance(x, float) and _num_match(x, float(n), 0.0, 0.0)
                   for x in av):
                return True, f"truth_count={n} agent=scalar {av}"
        ok = len(agent_rows) == n
        return ok, f"truth_count={n} agent_rows={len(agent_rows)}"

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
        if len(agent_rows) != 1:
            return False, (f"expected scalar, agent returned "
                           f"{len(agent_rows)} row(s) × {len(av)} col(s)")
        if len(av) != 1:
            # Agent returned exactly one row but added extra columns (e.g. a
            # count alongside the requested sum). Pass if the truth value is
            # present among them — same leniency 'row'/'top1' already grant
            # for leftover agent columns.
            hit = any(_cell_match(tv[0], x, abs_tol, rel_tol) for x in av)
            return hit, (f"truth={tv[0]} agent_row={av}"
                         + ("" if hit else " (no column matched)"))
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
        t_tuples = [tuple(_values(r)) for r in truth_rows]
        a_tuples = [tuple(_values(r)) for r in agent_rows]
        if subset:
            # every truth row must appear somewhere in the agent result;
            # extra agent rows are allowed.
            remaining = list(a_tuples)
            for tr in t_tuples:
                hit = next((i for i, ar in enumerate(remaining)
                            if len(tr) == len(ar) and all(
                                _cell_match(x, y, abs_tol, rel_tol)
                                for x, y in zip(tr, ar))), None)
                if hit is None:
                    return False, f"missing truth row {tr} (agent has {len(a_tuples)} rows)"
                remaining.pop(hit)
            return True, f"{len(t_tuples)}/{len(t_tuples)} truth rows present (agent +{len(remaining)})"
        if len(truth_rows) != len(agent_rows):
            return False, f"row count: truth={len(truth_rows)} agent={len(agent_rows)}"
        t_sorted = sorted(t_tuples, key=repr)
        a_sorted = sorted(a_tuples, key=repr)
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
            # execute_sql returns the standard tool envelope
            # {"success", "result": {"rows", "columns", ...}, "error"} — the
            # rows live one level down under "result", not at the top level.
            rows = (s.result.get("result") or {}).get("rows") or []
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
            subset=bool(spec.get("subset")),
        )
        return {"covered": True, "truth_match": ok, "detail": detail}
    except Exception as exc:  # noqa: BLE001 — eval must survive truth errors
        return {"covered": True, "truth_match": None,
                "detail": f"truth-layer error: {exc}"}


def smoke_check_canonical(db_engine, prefix: str | None = None) -> list[dict]:
    """
    Execute every canonical SQL in GROUND_TRUTH (optionally only keys starting
    with `prefix`, e.g. "AUDIT_") and report whether it parses + runs. This is
    the guard against canonical drift vs the live schema (renamed columns, dead
    columns, typo enum values) — run it before any full eval.
    Returns [{"case_id", "ok", "rows", "sample", "error"}].
    """
    from sqlalchemy import text
    out = []
    items = sorted(GROUND_TRUTH.items())
    if prefix:
        items = [(k, v) for k, v in items if k.startswith(prefix)]
    for case_id, spec in items:
        rec = {"case_id": case_id, "mode": spec.get("mode"), "ok": False,
               "rows": None, "sample": None, "error": None}
        try:
            with db_engine.connect() as conn:
                res = conn.execute(
                    text(f"SELECT * FROM (\n{spec['sql']}\n) _q LIMIT 5"))
                cols = list(res.keys())
                rows = [dict(zip(cols, r)) for r in res.fetchall()]
            rec["ok"] = True
            rec["rows"] = len(rows)
            rec["sample"] = rows[0] if rows else None
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc).splitlines()[0][:200]
        out.append(rec)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Self-test — comparison logic only, no DB
# ═══════════════════════════════════════════════════════════════════════════

def _run_smoke() -> int:
    """--smoke entry: execute every canonical SQL against the live DB."""
    import argparse
    import os
    import sys as _sys
    ap = argparse.ArgumentParser(prog="eval_truth.py --smoke")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prefix", default=None,
                    help='only keys starting with this, e.g. "AUDIT_"')
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for p in (repo_root, os.path.join(repo_root, "backend")):
        if p not in _sys.path:
            _sys.path.insert(0, p)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(repo_root, ".env"))
    from sqlalchemy import create_engine
    url = os.environ.get("DATABASE_URL") or (
        f"postgresql://{os.environ.get('DB_USER', 'postgres')}:"
        f"{os.environ.get('DB_PASSWORD', '')}@{os.environ.get('DB_HOST', 'localhost')}:"
        f"{os.environ.get('DB_PORT', '5432')}/{os.environ.get('DB_NAME', 'postgres')}"
    )
    engine = create_engine(url)
    rows = smoke_check_canonical(engine, prefix=args.prefix)
    n_fail = sum(1 for r in rows if not r["ok"])
    print(f"\nCANONICAL SQL SMOKE — {len(rows)} cases"
          f"{f' (prefix {args.prefix!r})' if args.prefix else ''}\n" + "─" * 70)
    for r in rows:
        if r["ok"]:
            print(f"  ✓ {r['case_id']:<22} {str(r['mode']):<7} rows={r['rows']}"
                  f"  {r['sample'] if r['sample'] is not None else ''}")
        else:
            print(f"  ✗ {r['case_id']:<22} {str(r['mode']):<7} ERROR: {r['error']}")
    print("─" * 70)
    print(f"{len(rows) - n_fail}/{len(rows)} canonical queries executed cleanly")
    return 1 if n_fail else 0


if __name__ == "__main__":
    import sys as _sys
    if "--smoke" in _sys.argv:
        raise SystemExit(_run_smoke())

    T = [
        # (name, mode, truth, agent, expect)
        ("scalar exact",       "scalar", [{"c": 264}], [{"high_risk_ca_count": 264}], True),
        ("scalar wrong",       "scalar", [{"c": 264}], [{"x": 230}], False),
        ("scalar Decimal/str", "scalar", [{"c": Decimal("86.38")}], [{"avg": "86.38"}], True),
        ("scalar tolerance",   "scalar", [{"c": 66.8}], [{"p": 67.7}], False),
        ("scalar extra col hit",  "scalar", [{"c": 33102}],
         [{"total": 33102.0, "cnt": 18}], True),   # FIN-02 shape: value present
        ("scalar extra col miss", "scalar", [{"c": 5}], [{"a": 1, "b": 2}], False),
        ("scalar multi-row",      "scalar", [{"c": 1}], [{"a": 1}, {"a": 2}], False),
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
        # ── count mode ──
        ("count exact",        "count", [{"c": 3}],
         [{"x": 1}, {"x": 2}, {"x": 3}], True),
        ("count mismatch",     "count", [{"c": 3}], [{"x": 1}, {"x": 2}], False),
        ("count scalar-row",   "count", [{"c": 29}], [{"n": 29}], True),
        ("count zero empty",   "count", [{"c": 0}], [], True),
        # ── set subset ──
        ("set subset pass",   "set-subset",
         [{"a": "x", "n": 1}, {"a": "y", "n": 2}],
         [{"k": "y", "v": 2}, {"k": "x", "v": 1}, {"k": "z", "v": 0}], True),
        ("set subset fail",   "set-subset",
         [{"a": "x", "n": 1}, {"a": "y", "n": 2}],
         [{"k": "x", "v": 1}, {"k": "z", "v": 0}], False),
    ]
    failed = 0
    for name, mode, truth, agent, expect in T:
        sub = mode == "set-subset"
        ok, detail = compare_results("set" if sub else mode, truth, agent, subset=sub)
        mark = "✓" if ok == expect else "✗ FAIL"
        if ok != expect:
            failed += 1
        print(f"{mark} {name:22s} got={ok} want={expect}  ({detail})")
    print(f"\n{len(T) - failed}/{len(T)} self-tests passed")
    raise SystemExit(1 if failed else 0)
    raise SystemExit(1 if failed else 0)