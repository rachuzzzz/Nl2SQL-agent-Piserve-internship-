"""
eval_cases.py
=============
Khidmah Admin NL2SQL Eval Suite — v2

Changes from v1 (104 questions):
  REMOVED — ETL-01, ANS-02, MT-01, MT-02
    These test the ai_answers ETL boundary / most-recently-answered-form path.
    Bugs are fixed in seed_examples.py — re-add once confirmed passing in dev.

  ADDED — ADM, FAC, FIN, CA, RISK, SCHED, TREND categories (admin user queries)

Format per case:
  id        — unique test identifier
  question  — exact string sent to the agent
  keywords  — ALL of these must appear in the generated SQL (case-insensitive)
              Keep tight: 2-5 meaningful tokens only, no stop-words
  category  — grouping label for reporting
  note      — optional explanation for a tricky case
"""

EVAL_CASES = [

    # ══════════════════════════════════════════════════════════════════════
    # ADM — Dashboard / Overview  (admin's first-look queries)
    # ══════════════════════════════════════════════════════════════════════

    {
        "id": "ADM-01",
        "question": "How many inspections were completed this quarter?",
        "keywords": ["inspection_report", "CLOSED", "SUBMITTED", "quarter"],
        "category": "overview",
    },
    {
        "id": "ADM-02",
        "question": "What is the average inspection score across all facilities?",
        "keywords": ["AVG", "inspection_score", "inspection_report"],
        "category": "overview",
    },
    {
        "id": "ADM-03",
        "question": "How many inspections are still under review?",
        "keywords": ["inspection_report", "UNDER_REVIEW", "COUNT"],
        "category": "overview",
    },
    {
        "id": "ADM-04",
        "question": "Show inspection count by inspection type this year",
        "keywords": ["inspection_type", "COUNT", "inspection_report", "CURRENT_DATE"],
        "category": "overview",
    },
    {
        "id": "ADM-05",
        "question": "Which inspection types have the lowest average scores?",
        "keywords": ["inspection_type", "AVG", "inspection_score", "ORDER BY"],
        "category": "overview",
    },
    {
        "id": "ADM-06",
        "question": "How many inspections were returned for modification this month?",
        "keywords": ["inspection_report", "RETURN_FOR_MODIFICATION", "COUNT"],
        "category": "overview",
    },
    {
        "id": "ADM-07",
        "question": "How are we doing overall this quarter?",
        "keywords": ["inspection_report", "inspection_score", "quarter", "AVG"],
        "category": "overview",
    },

    # ══════════════════════════════════════════════════════════════════════
    # FAC — Facility Analysis
    # ══════════════════════════════════════════════════════════════════════

    {
        "id": "FAC-01",
        "question": "Which facility has the most high-risk corrective actions?",
        "keywords": ["facility", "risk_level", "inspection_corrective_action", "COUNT"],
        "category": "facility",
    },
    {
        "id": "FAC-02",
        "question": "Which facility costs us the most in CAPEX and OPEX?",
        "keywords": ["facility", "po_details", "pr_details", "SUM"],
        "category": "facility",
        "note": "Must use po_details for CAPEX and pr_details for OPEX — not ica.capex/ica.opex alone",
    },
    {
        "id": "FAC-03",
        "question": "Show all facilities ranked by average inspection score",
        "keywords": ["facility", "AVG", "inspection_score", "GROUP BY", "ORDER BY"],
        "category": "facility",
    },
    {
        "id": "FAC-04",
        "question": "Which facility has the most overdue corrective actions?",
        "keywords": ["facility", "inspection_corrective_action", "COUNT"],
        "category": "facility",
        "note": "Agent may use status=OVERDUE or target_close_out_date < CURRENT_DATE — both valid",
    },
    {
        "id": "FAC-05",
        "question": "How many inspections were completed at each facility this year?",
        "keywords": ["facility", "inspection_report", "COUNT", "CURRENT_DATE", "GROUP BY"],
        "category": "facility",
    },
    {
        "id": "FAC-06",
        "question": "Which facilities have not been inspected in the last 90 days?",
        "keywords": ["facility", "inspection_report", "submitted_on", "INTERVAL"],
        "category": "facility",
    },
    {
        "id": "FAC-07",
        "question": "Show me facilities where OPEX is higher than CAPEX converted",
        "keywords": ["pr_details", "po_details", "facility", "HAVING"],
        "category": "facility",
        "note": "Requires comparing SUM(pr.pr_amount) > SUM(po.po_amount) per facility",
    },

    # ══════════════════════════════════════════════════════════════════════
    # FIN — Financial (CAPEX / OPEX)
    # ══════════════════════════════════════════════════════════════════════

    {
        "id": "FIN-01",
        "question": "What is the total CAPEX generated, converted, and lost?",
        "keywords": ["SUM", "capex", "po_details", "po_amount"],
        "category": "financial",
        "note": "CAPEX Lost must be arithmetic (generated - converted), not capex_status=REJECTED",
    },
    {
        "id": "FIN-02",
        "question": "How much OPEX have we incurred this year?",
        "keywords": ["pr_details", "pr_amount", "SUM", "CURRENT_DATE"],
        "category": "financial",
    },
    {
        "id": "FIN-03",
        "question": "Show me CAPEX conversion rate by facility",
        "keywords": ["facility", "po_details", "po_amount", "capex", "SUM"],
        "category": "financial",
        "note": "Conversion rate = SUM(po_amount) / NULLIF(SUM(capex), 0) * 100",
    },
    {
        "id": "FIN-04",
        "question": "Which corrective actions have the highest CAPEX still awaiting approval?",
        "keywords": ["inspection_corrective_action", "capex", "capex_status", "IS NULL"],
        "category": "financial",
        "note": "capex_status IS NULL = awaiting approval",
    },
    {
        "id": "FIN-05",
        "question": "How much CAPEX has been converted this quarter vs last quarter?",
        "keywords": ["po_details", "po_amount", "SUM", "quarter"],
        "category": "financial",
    },
    {
        "id": "FIN-06",
        "question": "What percentage of our CAPEX is being converted by the client?",
        "keywords": ["po_details", "po_amount", "capex", "NULLIF", "100"],
        "category": "financial",
    },
    {
        "id": "FIN-07",
        "question": "Show me total CAPEX generated and converted by facility this year",
        "keywords": ["facility", "capex", "po_details", "po_amount", "GROUP BY"],
        "category": "financial",
    },
    {
        "id": "FIN-08",
        "question": "Which corrective actions have POs raised against them?",
        "keywords": ["po_details", "mitigative_action_uuid", "inspection_corrective_action"],
        "category": "financial",
    },
    {
        "id": "FIN-09",
        "question": "Top 5 capex converters",
        "keywords": ["po_details", "po_amount", "SUM", "inspectee_user_id"],
        "category": "financial",
        "note": "Must use po_details.po_amount NOT ica.capex WHERE capex_status=ACCEPTED",
    },
    {
        "id": "FIN-10",
        "question": "Show me the total OPEX incurred from purchase requisitions",
        "keywords": ["pr_details", "pr_amount", "SUM"],
        "category": "financial",
    },

    # ══════════════════════════════════════════════════════════════════════
    # FM — FM Manager Performance
    # ══════════════════════════════════════════════════════════════════════

    {
        "id": "FM-01",
        "question": "Who are the top 5 FM managers by capex conversion rate?",
        "keywords": ["inspectee_user_id", "po_details", "po_amount", "SUM", "conversion"],
        "category": "fm_manager",
        "note": "Specific metric avoids ambiguity clarification. Must use inspectee_user_id. Conversion = SUM(po_amount)/NULLIF(SUM(capex),0)*100",
    },
    {
        "id": "FM-02",
        "question": "Which FM manager has the highest CAPEX conversion rate?",
        "keywords": ["inspectee_user_id", "po_details", "po_amount", "capex", "NULLIF"],
        "category": "fm_manager",
    },
    {
        "id": "FM-03",
        "question": "Which FM manager generates the most CAPEX?",
        "keywords": ["inspectee_user_id", "capex", "SUM", "GROUP BY"],
        "category": "fm_manager",
    },
    {
        "id": "FM-04",
        "question": "Show FM managers ranked by OPEX incurred",
        "keywords": ["inspectee_user_id", "pr_details", "pr_amount", "SUM", "GROUP BY"],
        "category": "fm_manager",
        "note": "High OPEX = bad (Khidmah bears the cost) — must use pr_details not ica.opex",
    },
    {
        "id": "FM-05",
        "question": "Which FM manager has the most overdue corrective actions?",
        "keywords": ["inspectee_user_id", "inspection_corrective_action", "target_close_out_date"],
        "category": "fm_manager",
    },
    {
        "id": "FM-06",
        "question": "Show CAPEX generated and converted by FM manager",
        "keywords": ["inspectee_user_id", "capex", "po_details", "po_amount", "SUM"],
        "category": "fm_manager",
    },

    # ══════════════════════════════════════════════════════════════════════
    # CA — Corrective Actions
    # ══════════════════════════════════════════════════════════════════════

    {
        "id": "CA-01",
        "question": "How many corrective actions are currently open?",
        "keywords": ["inspection_corrective_action", "status", "COUNT"],
        "category": "corrective_action",
    },
    {
        "id": "CA-02",
        "question": "Show me all overdue corrective actions",
        "keywords": ["inspection_corrective_action", "target_close_out_date"],
        "category": "corrective_action",
    },
    {
        "id": "CA-03",
        "question": "Which corrective actions have been deferred more than once?",
        "keywords": ["inspection_corrective_action", "CLOSE_WITH_DEFERRED", "status"],
        "category": "corrective_action",
    },
    {
        "id": "CA-04",
        "question": "What is the average time to close a corrective action?",
        "keywords": ["inspection_corrective_action", "AVG"],
        "category": "corrective_action",
    },
    {
        "id": "CA-05",
        "question": "How many corrective actions were raised this month?",
        "keywords": ["inspection_corrective_action", "COUNT", "created_on", "CURRENT_DATE"],
        "category": "corrective_action",
    },
    {
        "id": "CA-06",
        "question": "Show corrective actions by responsible party",
        "keywords": ["inspection_corrective_action", "responsible", "COUNT", "GROUP BY"],
        "category": "corrective_action",
    },
    {
        "id": "CA-07",
        "question": "Which corrective actions are awaiting CAPEX approval?",
        "keywords": ["inspection_corrective_action", "capex_status", "IS NULL", "capex"],
        "category": "corrective_action",
    },
    {
        "id": "CA-08",
        "question": "What is the corrective action closure rate this quarter?",
        "keywords": ["inspection_corrective_action", "CLOSED", "COUNT", "quarter"],
        "category": "corrective_action",
    },
    {
        "id": "CA-09",
        "question": "Show me corrective actions raised this month that are still pending",
        "keywords": ["inspection_corrective_action", "created_on", "status", "CURRENT_DATE"],
        "category": "corrective_action",
    },
    {
        "id": "CA-10",
        "question": "How many corrective actions does each inspector have on average per inspection?",
        "keywords": ["inspection_corrective_action", "inspection_report", "AVG", "COUNT"],
        "category": "corrective_action",
    },

    # ══════════════════════════════════════════════════════════════════════
    # RISK — Risk / Observations
    # ══════════════════════════════════════════════════════════════════════

    {
        "id": "RISK-01",
        "question": "How many high-risk observations were raised this month?",
        "keywords": ["risk_level", "inspection_corrective_action", "COUNT", "CURRENT_DATE"],
        "category": "risk",
    },
    {
        "id": "RISK-02",
        "question": "Which facility has the most critical risk deviations?",
        "keywords": ["facility", "risk_level", "inspection_corrective_action", "COUNT"],
        "category": "risk",
    },
    {
        "id": "RISK-03",
        "question": "Show all open corrective actions with high risk level",
        "keywords": ["risk_level", "inspection_corrective_action", "status"],
        "category": "risk",
    },
    {
        "id": "RISK-04",
        "question": "What is the breakdown of corrective actions by risk level?",
        "keywords": ["risk_level", "inspection_corrective_action", "COUNT", "GROUP BY"],
        "category": "risk",
    },
    {
        "id": "RISK-05",
        "question": "Which inspection type raises the most high-risk deviations?",
        "keywords": ["inspection_type", "risk_level", "inspection_corrective_action", "COUNT"],
        "category": "risk",
    },
    {
        "id": "RISK-06",
        "question": "Show me the percentage of inspections that have at least one corrective action",
        "keywords": ["inspection_report", "inspection_corrective_action", "COUNT", "NULLIF"],
        "category": "risk",
    },

    # ══════════════════════════════════════════════════════════════════════
    # SCHED — Schedule / Compliance
    # ══════════════════════════════════════════════════════════════════════

    {
        "id": "SCHED-01",
        "question": "Which inspections are overdue this quarter?",
        "keywords": ["inspection_schedule", "quarter", "CURRENT_DATE"],
        "category": "schedule",
    },
    {
        "id": "SCHED-02",
        "question": "How many inspections were scheduled vs completed this month?",
        "keywords": ["inspection_schedule", "COUNT", "CURRENT_DATE"],
        "category": "schedule",
    },
    {
        "id": "SCHED-03",
        "question": "Which inspectors have the most pending inspections?",
        "keywords": ["inspector_user_id", "inspection_report", "COUNT"],
        "category": "schedule",
    },
    {
        "id": "SCHED-04",
        "question": "What is the inspection frequency for each facility?",
        "keywords": ["frequency_definition", "inspector_portfolio_details", "facility"],
        "category": "schedule",
    },
    {
        "id": "SCHED-05",
        "question": "Which inspectors have completed all their scheduled inspections this cycle?",
        "keywords": ["inspection_report", "COMPLETED"],
        "category": "schedule",
    },

    # ══════════════════════════════════════════════════════════════════════
    # TREND — Trend / Time-series
    # ══════════════════════════════════════════════════════════════════════

    {
        "id": "TREND-01",
        "question": "Show me monthly inspection scores for the last 6 months",
        "keywords": ["inspection_score", "date_trunc", "month", "AVG", "GROUP BY"],
        "category": "trend",
    },
    {
        "id": "TREND-02",
        "question": "How has CAPEX conversion rate trended over the last 4 quarters?",
        "keywords": ["po_details", "po_amount", "capex", "quarter", "GROUP BY"],
        "category": "trend",
    },
    {
        "id": "TREND-03",
        "question": "Show inspection count by month this year",
        "keywords": ["inspection_report", "COUNT", "date_trunc", "month", "CURRENT_DATE"],
        "category": "trend",
    },
    {
        "id": "TREND-04",
        "question": "Which facility's score has improved the most since last quarter?",
        "keywords": ["facility", "inspection_score", "AVG", "quarter", "date_trunc"],
        "category": "trend",
    },
    {
        "id": "TREND-05",
        "question": "Are corrective action closure rates improving or declining over the last 3 months?",
        "keywords": ["inspection_corrective_action", "CLOSED", "date_trunc", "month", "COUNT"],
        "category": "trend",
    },
    {
        "id": "TREND-06",
        "question": "Show me OPEX incurred by month this year",
        "keywords": ["pr_details", "pr_amount", "SUM"],
        "category": "trend",
    },

    # ══════════════════════════════════════════════════════════════════════
    # ETL / ANS / MT — Most-recent form answer queries
    # These are kept in eval because ai_answers is a complete flattened
    # snapshot — every inspection_report row has corresponding ai_answers.
    # Fix the aa subquery pattern (no module_name filter in inner subquery)
    # before pushing to prod.
    # ══════════════════════════════════════════════════════════════════════

    {
        "id": "ETL-01",
        "question": "Show me the most recently answered inspection",
        "keywords": ["ai_answers", "inspection_report_id", "submitted_on", "ORDER BY"],
        "category": "recent_form",
        "note": "Must use aa subquery (ORDER BY aa.submitted_on DESC) not ORDER BY ir.submitted_on",
    },
    {
        "id": "ANS-02",
        "question": "List all questions and answers from the most recent inspection form",
        "keywords": ["ai_answers", "ai_questions", "element_id", "submitted_on"],
        "category": "recent_form",
        "note": "Inner subquery must NOT filter by module_name — causes empty result if name differs",
    },
    {
        "id": "MT-01",
        "question": "What was the score of the most recently answered inspection?",
        "keywords": ["inspection_report", "inspection_score", "submitted_on"],
        "category": "recent_form",
    },
    {
        "id": "MT-02",
        "question": "Show me the questions and answers for the most recently answered inspection",
        "keywords": ["ai_answers", "ai_questions", "inspection_report_id", "submitted_on"],
        "category": "recent_form",
    },


]


# ── Convenience helpers ───────────────────────────────────────────────────────

CATEGORIES = sorted({c["category"] for c in EVAL_CASES})

def cases_by_category(category: str) -> list[dict]:
    return [c for c in EVAL_CASES if c["category"] == category]

def case_by_id(eval_id: str) -> dict | None:
    return next((c for c in EVAL_CASES if c["id"] == eval_id), None)

def summary() -> str:
    lines = [f"Total eval cases: {len(EVAL_CASES)}\n"]
    for cat in CATEGORIES:
        cc = cases_by_category(cat)
        lines.append(f"  {cat:20s} {len(cc):3d} cases")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    print()
    for case in EVAL_CASES:
        print(f"  [{case['id']}] {case['question'][:70]}")