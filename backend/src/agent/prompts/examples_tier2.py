"""
Tier-2 worked examples -- extracted verbatim from the original
agent/prompts.py. See src.agent.prompts.builder for how these are
matched against a question and budget-capped.
"""

from src.core.question_patterns import (
    MA_WORKFLOW_TOPIC,
    MA_APPROVAL_WORKFLOW_TOPIC,
    FINANCIAL_ROUTING_TOPIC,
)

EXAMPLES_TIER2 = [
    {
        "title": "MAs awaiting approval",
        # Reuses inject_ma_approval_workflow's question_pattern (business_rules.py)
        "question_pattern": MA_APPROVAL_WORKFLOW_TOPIC,
        "text": """\
─── MAs awaiting approval ───
User: "how many mitigative actions are awaiting approval?" / "show MAs pending review"
Turn 1: {"thought": "Approval questions = implementation_status track, NEVER capex_status. Pending approval = implementation_status = 'PENDING_FOR_REVIEW' (exact string, not 'PENDING').", "tool": "generate_sql", "args": {"question": "mitigative actions pending approval awaiting review", "schema_hint": "SELECT COUNT(*) AS pending_approval_count FROM inspection_corrective_action ica WHERE ica.implementation_status = 'PENDING_FOR_REVIEW' AND ica.active = true;"}}

─── MAs at risk (schedule health, NOT severity) ───
User: "show mitigative actions that are at risk"
Turn 1: {"thought": "'at risk' = progress_stage = 'AT_RISK' (inspectee's schedule health). NOT risk_level = 'High' — that is severity, a different concept.", "tool": "generate_sql", "args": {"question": "mitigative actions at risk behind schedule progress", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.progress_stage, fac.name AS facility_name FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE ica.progress_stage = 'AT_RISK' AND ica.active = true AND ir.active = true LIMIT 100;"}}

─── MAs "at <company-like-name>" — facility, NOT organisation ───
User: "mitigative actions at adnoc"
Turn 1: {"thought": "'adnoc' looks like a company/brand name, but 'at X' always means fac.name here, not organisation — facilities are routinely named after their parent brand (e.g. 'Adnoc Schools- Ghayathi'). The organisation table is a separate, narrower concept and is almost never what 'at X' means. Filter by fac.name ILIKE, not org.name.", "tool": "generate_sql", "args": {"question": "mitigative actions at facilities matching adnoc", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.responsible, ica.status, fac.name AS facility_name FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE fac.name ILIKE '%adnoc%' AND ica.active = true AND ir.active = true AND ir.status != 'DRAFT' ORDER BY ica.created_on DESC LIMIT 100;"}}
NEVER: WHERE org.name ILIKE '%adnoc%' — silently returns 0 rows since the
  match is encoded in the facility name, not an organisation record.
""",
    },
    {
        "title": "list all cycles — frequency_definition, NOT inspection_cycle",
        # New pattern — no prior INJECT rule covered "cycle" vocabulary directly
        # (block_cycle_frequency_missing_subquery / block_inspection_cycle_name
        # are SQL-shape BLOCK rules, not question-level INJECT rules).
        "question_pattern": (
            r"(?i)\bcycles?\b|\binspection\s+frequenc\w*"
        ),
        "text": """\
─── "list all cycles" — frequency_definition, NOT inspection_cycle ───
User: "list all cycles" / "show inspection cycles" / "what cycles are available"
Turn 1: {"thought": "'cycles' here means inspection frequency/cadence types (Daily, Weekly, Quarterly Twice, etc.) from frequency_definition — NOT inspection_cycle instances. frequency_definition columns are id, name, repeat_count, repeat_interval, repeat_unit, active — there is NO description column, do not invent one.", "tool": "generate_sql", "args": {"question": "list all inspection frequency cycles", "schema_hint": "SELECT name AS frequency_name, repeat_count, repeat_interval, repeat_unit FROM frequency_definition WHERE active = true ORDER BY name;"}}
""",
    },
    {
        "title": "inspection_schedule joined to inspection_cycle — NO name column on ic",
        # Schedule + date-range phrasing — different trigger than the bare
        # "cycle" pattern above (this question doesn't contain "cycle" at all,
        # but its schema_hint joins inspection_cycle and must avoid ic.name).
        "question_pattern": (
            r"(?i)\bschedules?\b.{0,30}\b(next|coming|upcoming|this\s+week|today|tomorrow)\b"
            r"|\b(next|coming|upcoming)\s+\d+\s+days?\b.{0,30}\bschedul"
            r"|\bschedul\w*.{0,40}\bcycle\b"
        ),
        "text": """\
─── inspection_schedule joined to inspection_cycle — NO name column on ic ───
User: "inspection schedules for the next 5 days"
Turn 1: {"thought": "Filter by due_date within the next 5 days. If joining inspection_cycle (ic), it has NO 'name' column — its columns are id, status, start_date, end_date, due_date, send_notification_to_inspectors, created_on, modified_on. Use ic.id/ic.status, not ic.name. NEVER select isched.schedule_id — almost always NULL.", "tool": "generate_sql", "args": {"question": "inspection schedules for the next 5 days", "schema_hint": "SELECT isched.id, isched.status, isched.due_date, fac.name AS facility_name, u_insp.first_name || ' ' || u_insp.last_name AS inspector_name, proj.name AS project_name, ic.id AS cycle_id, ic.status AS cycle_status FROM inspection_schedule isched LEFT JOIN facility fac ON isched.facility_id = fac.id LEFT JOIN users u_insp ON isched.inspector_id = u_insp.id LEFT JOIN project proj ON isched.project_id = proj.id LEFT JOIN inspection_cycle ic ON isched.inspection_cycle_id = ic.id WHERE isched.due_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '5 days' AND fac.active = true AND u_insp.active = true ORDER BY isched.due_date ASC LIMIT 100;"}}
""",
    },
    {
        "title": "Capex lost (arithmetic gap) vs capex from rejected MAs — disambiguation",
        # Fires on "rejected" near "capex" AND on "wasted/unconverted/didn't get converted" capex
        # so the model always sees the disambiguation rule before deciding which query to run.
        "question_pattern": (
            r"(?i)capex.{0,40}reject|reject.{0,40}capex"
            r"|capex.{0,40}(wasted|waste|not\s+convert|didn.t\s+convert|unconvert|shortfall|gap)"
            r"|(wasted|waste|not\s+convert|didn.t\s+convert|unconvert|shortfall|gap).{0,40}capex"
        ),
        "text": """\
─── CAPEX Lost (arithmetic gap) vs CAPEX from rejected MAs — CRITICAL DISAMBIGUATION ───
CAPEX Lost  = arithmetic gap = SUM(ica.capex) - COALESCE(SUM(po.po_amount),0)
              — always includes ALL MAs regardless of capex_status
              — user phrases: "capex lost", "didn't get converted", "not converted",
                "unconverted capex", "capex gap", "capex shortfall", "wasted capex"

CAPEX from rejected MAs = WHERE capex_status = 'REJECTED'
              — a SUBSET of the above (only rows the user marked 'REJECTED')
              — user phrase: explicitly "rejected mitigative actions" or "capex_status rejected"

NEVER use capex_status='REJECTED' when the user asks about "wasted" or "lost" capex generally.
NEVER use the arithmetic gap when the user explicitly says "rejected actions".

─── User asking about total CAPEX Lost (arithmetic gap) ───
User: "how much capex didn't get converted?" / "what capex was wasted?" / "show capex that wasn't paid"
Turn 1: {"thought": "User wants the arithmetic CAPEX gap = Generated minus Converted. This is NOT capex_status='REJECTED' — that is only the subset where the user marked the ICA as rejected. Use scalar subqueries for a single-row total.", "tool": "generate_sql", "args": {"question": "total capex lost arithmetic gap generated minus converted", "schema_hint": "SELECT COALESCE((SELECT SUM(capex) FROM inspection_corrective_action WHERE active=true AND capex>0),0) AS capex_generated, COALESCE((SELECT SUM(po_amount) FROM po_details WHERE active=true),0) AS capex_converted, COALESCE((SELECT SUM(capex) FROM inspection_corrective_action WHERE active=true AND capex>0),0) - COALESCE((SELECT SUM(po_amount) FROM po_details WHERE active=true),0) AS capex_lost;"}}

─── User asking specifically about rejected mitigative actions ───
User: "show capex lost from rejected mitigative actions" / "list MAs where capex was rejected"
Turn 1: {"thought": "User explicitly says 'rejected' — they want MA rows where capex_status = 'REJECTED'. This is the subset view, NOT the arithmetic gap. capex_status values: ACCEPTED | REJECTED | DEFERRED | NULL.", "tool": "generate_sql", "args": {"question": "capex from rejected mitigative actions", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.capex, ica.capex_status, fac.name AS facility_name FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE ica.capex_status = 'REJECTED' AND ica.active = true AND ir.active = true ORDER BY ica.capex DESC LIMIT 100;"}}
""",
    },
    {
        "title": "Total capex generated converted and lost",
        # Reuses inject_financial_routing's question_pattern (business_rules.py)
        "question_pattern": FINANCIAL_ROUTING_TOPIC,
        "text": """\
─── Total capex generated converted and lost ───
User: "what is the total capex generated converted and lost"
ROUTING RULE: Use scalar subqueries NOT a JOIN between ica and po_details.
NEVER: LEFT JOIN po_details p ON ica.mitigative_action_uuid = p.mitigative_action_uuid (wrong direction)
NEVER: LEFT JOIN po_details p ON p.mitigative_action_uuid = ica.mitigative_action_uuid (causes row fan-out)
ALWAYS: Use scalar subqueries — one SELECT per metric, no JOIN needed for totals.
Turn 1: {"thought": "Total capex summary — use scalar subqueries, no JOIN. mitigative_action_uuid on po_details points TO ica.id so a direct JOIN causes fan-out. Scalar subqueries give ONE clean row.", "tool": "generate_sql", "args": {"question": "total capex generated converted and lost", "schema_hint": "SELECT (SELECT COALESCE(SUM(capex),0) FROM inspection_corrective_action WHERE active=true AND capex>0) AS capex_generated, (SELECT COALESCE(SUM(po_amount),0) FROM po_details WHERE active=true) AS capex_converted, COALESCE((SELECT SUM(capex) FROM inspection_corrective_action WHERE active=true AND capex>0),0) - COALESCE((SELECT SUM(po_amount) FROM po_details WHERE active=true),0) AS capex_lost, (SELECT COALESCE(SUM(pr_amount),0) FROM pr_details WHERE active=true) AS opex_incurred;"}}
""",
    },
    {
        "title": "Financial data PO PR by facility",
        "question_pattern": (
            r"(?i)\b(po|pr|purchase\s+order|purchase\s+requisition|financial|income|cost|profit|p.?l|revenue)\b"
            r".{0,50}\b(facility|site|location)\b"
            r"|\b(facility|site|location)\b.{0,50}\b(po|pr|purchase|financial|income|cost|profit)\b"
        ),
        "text": """\\u2500\u2500\u2500 Financial data (PO/PR) for a specific facility \u2500\u2500\u2500
User: "show financial data for Al Ghadeer" / "PO and PR amounts at [facility]" / "income and cost for [facility]"
CRITICAL JOIN PATH: po_details.mitigative_action_uuid = ica.id  (ica.id is the UUID primary key)
NEVER: po.mitigative_action_uuid = ica.corrective_action_id  \u2014 corrective_action_id is VARCHAR, causes type error
NEVER: po.mitigative_action_uuid = ir.id  \u2014 matches zero rows
ALWAYS pre-aggregate po_details and pr_details in CTEs to avoid row fan-out from LEFT JOIN.

\u2500\u2500\u2500 Capex/Opex by ALL facilities (no specific facility named) \u2500\u2500\u2500
User: "show total capex and opex by facility" / "capex and opex breakdown by site" / "total po and pr by facility from po and pr details"
Turn 1: {"thought": "User wants capex/opex totals grouped by ALL facilities \u2014 no specific facility filter. Pre-aggregate po_details and pr_details in CTEs first to avoid fan-out, then JOIN and GROUP BY fac.id.", "tool": "generate_sql", "args": {"question": "total capex and opex by all facilities CTE", "schema_hint": "WITH po_agg AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS capex_converted FROM po_details WHERE active = true GROUP BY 1), pr_agg AS (SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS opex_incurred FROM pr_details WHERE active = true GROUP BY 1) SELECT fac.name AS facility_name, COALESCE(SUM(ica.capex),0) AS capex_generated, COALESCE(SUM(po.capex_converted),0) AS capex_converted, COALESCE(SUM(pr.opex_incurred),0) AS opex_incurred FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id LEFT JOIN po_agg po ON po.ica_id = ica.id LEFT JOIN pr_agg pr ON pr.ica_id = ica.id WHERE ica.active = true AND ir.active = true AND fac.active = true AND ir.status != 'DRAFT' GROUP BY fac.id, fac.name ORDER BY capex_generated DESC LIMIT 100;"}}

\u2500\u2500\u2500 Financial data for a specific named facility \u2500\u2500\u2500
User: "show financial data for Al Ghadeer" / "PO and PR amounts at [facility]" / "income and cost for [facility]"
Turn 1: {"thought": "Financial data for a facility \u2014 must join po_details and pr_details via ica.id (UUID PK), NOT ica.corrective_action_id (varchar). Pre-aggregate in CTEs to avoid fan-out.", "tool": "generate_sql", "args": {"question": "financial data PO PR income cost for facility", "schema_hint": "WITH po_agg AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_income FROM po_details WHERE active = true GROUP BY 1), pr_agg AS (SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_cost FROM pr_details WHERE active = true GROUP BY 1) SELECT fac.name AS facility_name, COALESCE(SUM(po.total_income),0) AS income, COALESCE(SUM(pr.total_cost),0) AS cost, COALESCE(SUM(po.total_income),0) - COALESCE(SUM(pr.total_cost),0) AS profit FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id LEFT JOIN po_agg po ON po.ica_id = ica.id LEFT JOIN pr_agg pr ON pr.ica_id = ica.id WHERE fac.name ILIKE '%[FACILITY]%' AND ica.active = true AND ir.active = true AND fac.active = true GROUP BY fac.id, fac.name;"}}
""",
    },
    {
        "title": "FM manager CAPEX conversion rate",
        # NOT reusing inject_fm_manager_routing's pattern here — that pattern
        # requires "convert"/"converter" but this phrasing says "conversion",
        # so inject_fm_manager_routing/inject_top_capex_converters do NOT
        # actually match "which FM manager has the highest capex conversion
        # rate?" (verified 2026-06-12 — word-form gap in those regexes,
        # out of scope to fix here). This pattern is intentionally narrower
        # and specific to "FM/facility manager" + "conversion" proximity.
        "question_pattern": (
            r"(?i)\b(fm|facility)\s+manager\b.{0,40}\bconversion\b"
            r"|\bconversion\s+rate\b.{0,40}\b(fm|facility)\s+manager"
            r"|\b(best|highest|top)\b.{0,20}\b(fm|facility)\s+manager\b.{0,30}\bcapex"
        ),
        "text": """\
─── FM manager CAPEX conversion rate ───
User: "which FM manager has the highest capex conversion rate?"
ROUTING RULE: CAPEX conversion rate = SUM(po_details.po_amount) / SUM(ica.capex).
NEVER use ica.expenditure for this — it is a category flag, not a financial amount.
NEVER use capex_status='ACCEPTED' — use po_details for actual converted amounts.
Turn 1: {"thought": "Need FM manager ranked by capex conversion rate = SUM(po.po_amount)/SUM(ica.capex). Must use po_details CTE, not ica.expenditure.", "tool": "generate_sql", "args": {"question": "FM manager with highest capex conversion rate", "schema_hint": "WITH po_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po FROM po_details WHERE active = true GROUP BY 1) SELECT u.first_name || ' ' || u.last_name AS fm_manager, COALESCE(SUM(ica.capex),0) AS capex_generated, COALESCE(SUM(p.total_po),0) AS capex_converted, ROUND(COALESCE(SUM(p.total_po),0)*100.0/NULLIF(SUM(ica.capex),0),1) AS conversion_rate FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id=ir.id JOIN users u ON ir.inspectee_user_id=u.id LEFT JOIN po_by_ica p ON p.ica_id=ica.id WHERE ir.active=true AND ica.active=true AND u.active=true AND ica.capex>0 GROUP BY u.id,u.first_name,u.last_name HAVING SUM(ica.capex)>0 ORDER BY conversion_rate DESC NULLS LAST LIMIT 1;"}}
""",
    },
    {
        "title": "MA-related worked examples (timeliness, MA-by-inspection, MA-by-month, observations on MAs)",
        # Reuses the SAME pattern as CONDITIONAL_BLOCKS["MA workflow columns"] —
        # both fire together for mitigative-action questions, single source
        # of truth for "is this a MA question" at the question level.
        "question_pattern": MA_WORKFLOW_TOPIC,
        "text": """\
─── MAs not closed on time / closed late ───
User: "how many mitigative actions were not closed on time?" / "show closed-late mitigative actions"
Turn 1: {"thought": "'closed late' is DIFFERENT from 'overdue' — overdue means still OPEN and past due (status='OVERDUE', the authoritative field); closed late means it DID close (status='CLOSED') but completed_on was after target_close_out_date. These are mutually exclusive — NEVER OR them into one count. close_on is a dead column (all NULL); use completed_on.", "tool": "generate_sql", "args": {"question": "mitigative actions not closed on time closed late vs overdue", "schema_hint": "SELECT COUNT(*) FILTER (WHERE status = 'OVERDUE') AS overdue_count, COUNT(*) FILTER (WHERE status = 'CLOSED' AND completed_on > target_close_out_date) AS closed_late_count FROM inspection_corrective_action WHERE active = true AND target_close_out_date IS NOT NULL;"}}

─── MAs timeliness breakdown (closed on time / closed late / overdue) ───
User: "show mitigative actions closed on time vs closed late vs overdue" / "give me a timeliness breakdown of mitigative actions"
Turn 1: {"thought": "Three-category breakdown using only ica columns (status, completed_on, target_close_out_date) — SINGLE TABLE, no joins needed. NEVER add JOIN inspection_report or reference ir.active here — there's nothing for ir to join on and it causes a missing-FROM-clause error. 'overdue' = status = 'OVERDUE', the authoritative field, not date arithmetic.", "tool": "generate_sql", "args": {"question": "mitigative actions timeliness breakdown closed on time closed late overdue", "schema_hint": "SELECT CASE WHEN status = 'CLOSED' AND completed_on <= target_close_out_date THEN 'closed on time' WHEN status = 'CLOSED' AND completed_on > target_close_out_date THEN 'closed late' WHEN status = 'OVERDUE' THEN 'overdue' ELSE 'other' END AS timeliness_category, COUNT(*) AS ca_count FROM inspection_corrective_action WHERE active = true AND target_close_out_date IS NOT NULL GROUP BY timeliness_category ORDER BY timeliness_category;"}}

─── Most common causes of mitigative actions — ica.cause, NOT form answers ───
User: "what are the most common causes of mitigative actions" / "top MA causes" / "most frequent causes"
ROUTING: ica.cause is a plain VARCHAR column on inspection_corrective_action — NOT a form field. NEVER use get_answer_stats. Always generate_sql.
Turn 1: {"thought": "ica.cause is a plain column, not a form answer — generate_sql with GROUP BY ica.cause, never get_answer_stats.", "tool": "generate_sql", "args": {"question": "most common causes of mitigative actions ranked by frequency", "schema_hint": "SELECT ica.cause, COUNT(*) AS ca_count, ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),1) AS percentage FROM inspection_corrective_action ica WHERE ica.active = true AND ica.cause IS NOT NULL GROUP BY ica.cause ORDER BY ca_count DESC LIMIT 20;"}}

─── Mitigative actions ───
User: "show open mitigative actions with causes"
Turn 1: {"thought": "Plain SQL on inspection_corrective_action.", "tool": "generate_sql", "args": {"question": "open mitigative actions with causes", "schema_hint": "inspection_corrective_action: corrective_action_id, cause, corrective_action, responsible, status. WHERE status = 'OPEN'"}}

─── Observation for a specific mitigative action ───
User: "show the observation for the first mitigative action"
Turn 1: {"thought": "User wants the free-text observation the inspector wrote for this MA. ica.observation_id (UUID) links to ai_answers.submission_id (varchar) — a group of answers for that observation. Must filter aq.label = 'Observation' to get the free-text field only, not Observation Type/Risk Level/Impact. Cast: aa.submission_id = ica.observation_id::varchar. NEVER use get_answers(inspection_id=...) — returns whole form.", "tool": "generate_sql", "args": {"question": "observation text linked to mitigative action", "schema_hint": "SELECT aa.answer_text AS observation, ir.inspection_id, fac.name AS facility_name FROM inspection_corrective_action ica JOIN ai_answers aa ON aa.submission_id = ica.observation_id::varchar JOIN ai_questions aq ON aa.element_id = aq.element_id JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE ica.corrective_action_id = '<CA_ID>' AND aq.label = 'Observation' AND ica.active = true AND aa.answer_text IS NOT NULL LIMIT 1;"}}
─── High risk mitigative actions for a specific month ───
User: "show high risk mitigative actions from February 2026"
Turn 1: {"thought": "Need high risk MAs from Feb 2026. Risk level is on ica.risk_level_id — JOIN risk_level table. Date filter on ir.submitted_on using EXTRACT. NEVER join ai_answers to get risk level — ica.risk_level_id is a direct UUID FK to risk_level table.", "tool": "generate_sql", "args": {"question": "high risk mitigative actions from February 2026", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.responsible, ica.status, rl.name AS risk_level, ica.target_close_out_date, fac.name AS facility_name FROM inspection_corrective_action ica JOIN risk_level rl ON ica.risk_level_id = rl.id JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE rl.name = 'High' AND EXTRACT(MONTH FROM ir.submitted_on) = 2 AND EXTRACT(YEAR FROM ir.submitted_on) = 2026 AND ica.active = true AND ir.active = true AND ir.status != 'DRAFT' ORDER BY ica.target_close_out_date ASC LIMIT 100;"}}
─── Mitigative actions for a specific inspection ───
User: "show mitigative actions for inspection 2026/04/ST158/INS005"
Turn 1: {"thought": "Need mitigative actions for a specific inspection. ica.inspection_id is UUID FK to inspection_report.id — must join and filter by ir.inspection_id (varchar). NEVER use ica.due_date (use target_close_out_date), NEVER use ica.actual_date (use close_on), NEVER use ica.mitigative_action_uuid (that is on po_details/pr_details pointing TO ica.id).", "tool": "generate_sql", "args": {"question": "mitigative actions for inspection 2026/04/ST158/INS005", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.responsible, ica.status, ica.target_close_out_date, ica.close_on, ica.capex, ica.opex, ica.progress_stage FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id WHERE ir.inspection_id = '2026/04/ST158/INS005' AND ica.active = true AND ir.active = true ORDER BY ica.created_on ASC LIMIT 100;"}}
─── Responsible party filter — NEVER GROUP BY all, filter to what was asked ───
User: "compare overdue mitigative actions between CLIENT and INTERNAL_OPERATIONS"
COLUMN: ica.responsible — enum, always uppercase: 'CLIENT' | 'INTERNAL_OPERATIONS' | 'SUB_CONTRACTOR'
CRITICAL: when user names specific parties, add WHERE ica.responsible IN (...) — do NOT return all groups.
Turn 1: {"thought": "ica.responsible is the enum column for who owns the MA. User asked about CLIENT and INTERNAL_OPERATIONS only — must filter with IN, not a naked GROUP BY which would also return SUB_CONTRACTOR.", "tool": "generate_sql", "args": {"question": "overdue mitigative action count by responsible party CLIENT and INTERNAL_OPERATIONS", "schema_hint": "SELECT ica.responsible, COUNT(*) AS overdue_count FROM inspection_corrective_action ica WHERE ica.status = 'OVERDUE' AND ica.responsible IN ('CLIENT', 'INTERNAL_OPERATIONS') AND ica.active = true GROUP BY ica.responsible ORDER BY overdue_count DESC;"}}""",
    },
    {
        "title": "Schedule/frequency/portfolio worked examples",
        "question_pattern": (
            r"(?i)\bschedul\w*|\bfrequenc\w*|\bportfolios?\b"
            r"|\boverdue\b.{0,30}\binspections?\b|\binspections?\b.{0,30}\boverdue\b"
        ),
        "text": """\
─── inspection_schedule queries — DOES have an active column ───
COLUMNS: isched.id (PK), isched.active, isched.status, isched.due_date, isched.schedule_date (often NULL — prefer due_date for display)
NEVER SELECT isched.schedule_id — it is a nullable internal reference UUID, almost always NULL and useless for display.
User: "list all inspection schedules with facility, inspector, project and status"
Turn 1: {"thought": "inspection_schedule DOES have an active column — add isched.active = true. facility/users/project also have active columns, filter those too. NEVER select isched.schedule_id — it is almost always NULL. Use isched.id (the PK) for row identification and isched.due_date as the primary date.", "tool": "generate_sql", "args": {"question": "list all inspection schedules with facility, inspector, project and status", "schema_hint": "SELECT isched.id, isched.status, isched.due_date, fac.name AS facility_name, u_insp.first_name || ' ' || u_insp.last_name AS inspector_name, proj.name AS project_name FROM inspection_schedule isched LEFT JOIN facility fac ON isched.facility_id = fac.id LEFT JOIN users u_insp ON isched.inspector_id = u_insp.id LEFT JOIN project proj ON isched.project_id = proj.id WHERE isched.active = true AND fac.active = true AND u_insp.active = true ORDER BY isched.due_date ASC LIMIT 100;"}}

─── Facilities scheduled for a specific frequency ───
User: "which facilities are scheduled for daily inspections"
Turn 1: {"thought": "Daily inspections come from frequency_definition → inspector_portfolio_details → facility. Do NOT use inspection_schedule for this — frequency is set on the portfolio, not the schedule. Filter frequency_definition.name = 'Daily'. inspector_portfolio_details isn't in my default schema context, so I'll list it in relevant_tables to get its real columns.", "tool": "generate_sql", "args": {"question": "facilities scheduled for daily inspections", "schema_hint": "SELECT DISTINCT fac.name AS facility_name, fd.name AS frequency FROM frequency_definition fd JOIN inspector_portfolio_details ipd ON ipd.frequency_definition_id = fd.id JOIN facility fac ON fac.id = ipd.facility_id WHERE fd.name = 'Daily' AND fd.active = true AND ipd.active = true AND fac.active = true ORDER BY fac.name;", "relevant_tables": ["inspector_portfolio_details", "frequency_definition"]}}
─── Inspection schedule: overdue inspections ───
User: "show all overdue inspections"
Turn 1: {"thought": "Overdue inspections are in inspection_schedule WHERE status = 'OVERDUE'. inspection_schedule has NO inspection_id or inspection_report_id column — get facility, inspector, inspectee from FKs on inspection_schedule itself. To reach inspection_report instead, the FK is on the report: inspection_report.schedule_id = isched.id (not the other way round). Do NOT select inspection_id from inspection_schedule.", "tool": "generate_sql", "args": {"question": "overdue inspections with facility inspector and due date", "schema_hint": "SELECT fac.name AS facility_name, proj.name AS project_name, u_insp.first_name || ' ' || u_insp.last_name AS inspector_name, u_ins.first_name || ' ' || u_ins.last_name AS inspectee_name, isched.due_date, (CURRENT_DATE - isched.due_date::date) AS days_overdue FROM inspection_schedule isched JOIN facility fac ON isched.facility_id = fac.id LEFT JOIN project proj ON isched.project_id = proj.id JOIN users u_insp ON isched.inspector_id = u_insp.id LEFT JOIN users u_ins ON isched.inspectee_id = u_ins.id WHERE isched.status = 'OVERDUE' AND isched.active = true AND fac.active = true AND u_insp.active = true ORDER BY isched.due_date ASC LIMIT 100;"}}
─── Inspection schedule: schedule status counts ───
User: "how many inspections are overdue vs pending vs completed"
Turn 1: {"thought": "Count by status from inspection_schedule. inspection_schedule DOES have an active column — filter WHERE active = true to exclude cancelled/removed schedules.", "tool": "generate_sql", "args": {"question": "inspection schedule status counts", "schema_hint": "SELECT status, COUNT(*) AS count FROM inspection_schedule WHERE active = true GROUP BY status ORDER BY count DESC;"}}""",
    },
    # Form answers / observations / risk level / form-scores worked examples —
    # split into one entry per scenario (was a single ~12.9K-char block shared
    # by one broad question_pattern matching any of "forms/answer/questions/
    # observations/risk level/deviations/good practice/total score/ai_answers"
    # — meaning EVERY one of these 19 scenarios fired together on nearly any
    # form-related question, e.g. "which facilities have high risk
    # observations but no mitigative actions" pulled in the full block even
    # though only 1-2 of the 19 examples are actually relevant. Combined with
    # the baseline SYSTEM_PROMPT this pushed the reasoning-LLM prompt close
    # enough to its 20K-token num_ctx to produce empty completions (see
    # _select_within_budget above — it can only ration BETWEEN blocks that
    # match, it can't shrink a single block that matches alone). Splitting
    # into per-scenario patterns lets the budget/scoring mechanism actually
    # do its job: a question now pulls in only the 1-3 truly relevant
    # examples (a few hundred chars each) instead of the whole ~12.9K.
    {
        "title": "Form answers: specific question",
        "question_pattern": (
            r"(?i)\bwhat\s+did\b.{0,25}\banswer\w*\s+for\b"
            r"|\bwhat\s+(is|was)\s+the\s+answer\s+(for|to)\b"
        ),
        "text": """\
─── Form answers: specific question ───
User: "what did inspectors answer for 'risk level'?"
Turn 1: {"thought": "Form answers are in ai_answers — use get_answers.", "tool": "get_answers", "args": {"label": "risk level", "limit": 50}}
""",
    },
    {
        "title": "Form answers: distribution",
        "question_pattern": (
            r"(?i)\bmost\s+common\s+answer\b|\banswer\s+distribution\b"
            r"|\bbreakdown\s+of\s+answers?\b"
        ),
        "text": """\
─── Form answers: distribution ───
User: "most common answer to 'condition of PPE'?"
Turn 1: {"thought": "Need distribution — get_answer_stats.", "tool": "get_answer_stats", "args": {"label": "condition of PPE"}}
""",
    },
    {
        "title": "Form answers: average score for a scored question",
        "question_pattern": (
            r"(?i)\baverage\s+score\s+for\s+the\b|\bavg\s+score\s+for\s+the\b"
        ),
        "text": """\
─── Form answers: average score for a scored question ───
User: "what is the average score for the risk level question?"
Turn 1: {"thought": "Risk Level is a scored dropdown question. get_answer_stats returns score stats from ai_answers.score — not answer_numeric which is NULL for dropdowns.", "tool": "get_answer_stats", "args": {"label": "risk level"}}
""",
    },
    {
        "title": "Scores: which inspection had highest total score",
        "question_pattern": (
            r"(?i)\b(highest|lowest|top)\s+total\s+score\b"
            r"|\btotal\s+score\b.{0,30}\binspection\b"
        ),
        "text": """\
─── Scores: which inspection had highest total score ───
User: "which inspection had the highest total score?"
Turn 1: {"thought": "Total score = SUM of ai_answers.score per inspection. NOT inspection_report.inspection_score which is the overall report score. Use ir.inspection_id (varchar) not ir.id (UUID).", "tool": "generate_sql", "args": {"question": "inspection with highest total score from ai_answers", "schema_hint": "SELECT ir.inspection_id, SUM(aa.score) AS total_score FROM ai_answers aa JOIN inspection_report ir ON aa.inspection_report_id = ir.id WHERE aa.score IS NOT NULL AND ir.status != 'DRAFT' GROUP BY ir.inspection_id ORDER BY total_score DESC LIMIT 1;"}}
""",
    },
    {
        "title": "Form answers: observation text at a specific facility",
        "question_pattern": (
            r"(?i)\bwrote\s+in\s+the\s+observation\b"
            r"|\bobservation\s+(text|field)\s+(at|for)\b"
        ),
        "text": """\
─── Form answers: observation text at a specific facility ───
User: "what did inspectors write in the observation field for Al Ghadeer?"
Turn 1: {"thought": "Need ai_answers filtered by question label='Observation' AND facility='Al Ghadeer'. Must join: ai_answers → ai_questions (label) → inspection_report → facility. Exclude 'Observation Type' and 'Unique Identifier' — only the free-text Observation field.", "tool": "generate_sql", "args": {"question": "observation text at Al Ghadeer facility", "schema_hint": "SELECT aa.answer_text AS observation, ir.inspection_id, ir.submitted_on FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id JOIN inspection_report ir ON aa.inspection_report_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE aq.label = 'Observation' AND fac.name ILIKE '%Al Ghadeer%' AND aa.answer_text IS NOT NULL AND ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 100;"}}
""",
    },
    {
        "title": "Form answers: filter by answer value (high risk observations)",
        "question_pattern": (
            r"(?i)\bhigh\s*[- ]?risk\s+observations?\b"
            r"|\bobservations?\b.{0,20}\brisk\s+(level\s+)?is\s+high\b"
        ),
        "text": """\
─── Form answers: filter by answer value ───
User: "show all high risk observations"
Turn 1: {"thought": "User wants observation TEXT for inspections where Risk Level answer = High. Must use TWO arms: subquery gets inspection_report_ids where Risk Level=High, outer query gets Observation answer_text for those inspections. NEVER return risk level rows themselves — return the observation text.", "tool": "generate_sql", "args": {"question": "observation text for inspections where risk level is high", "schema_hint": "SELECT obs.answer_text AS observation, ir.inspection_id, ir.submitted_on FROM ai_answers obs JOIN ai_questions obs_q ON obs.element_id = obs_q.element_id JOIN inspection_report ir ON obs.inspection_report_id = ir.id WHERE obs_q.label = 'Observation' AND obs.inspection_report_id IN (SELECT aa2.inspection_report_id FROM ai_answers aa2 JOIN ai_questions aq2 ON aa2.element_id = aq2.element_id WHERE aq2.label = 'Risk Level' AND aa2.answer_text ILIKE '%High%') AND ir.status != 'DRAFT' AND ir.active = true ORDER BY ir.submitted_on DESC LIMIT 100;"}}
""",
    },
    {
        "title": "Form answers: by inspection",
        "question_pattern": (
            r"(?i)\banswers?\s+in\s+inspection\b|\ball\s+answers?\s+for\s+inspection\b"
        ),
        "text": """\
─── Form answers: by inspection ───
User: "all answers in inspection INS-2024-001"
Turn 1: {"thought": "Filter by inspection_id.", "tool": "get_answers", "args": {"inspection_id": "INS-2024-001", "limit": 100}}
""",
    },
    {
        "title": "Form answers: most recently filled form",
        "question_pattern": (
            # Covers both "most recently filled/answered" phrasing (MT-01:
            # "show me the most recently answered inspection") and "questions
            # and answers {in,from} the most recent(ly) [inspection] form"
            # (ANS-02: "...from the most recent inspection form" — note the
            # "inspection" infix and "from" instead of "in", both of which a
            # too-literal first-draft pattern missed, silently dropping the
            # "use an aa subquery, NOT inspection_report" guidance these two
            # eval cases depend on).
            r"(?i)\bmost\s+recently\s+(filled|answered)\b"
            r"|\bquestions?\s+and\s+answers?\s+(in|from)\s+the\s+most\s+recent\w*"
            r"\s+(inspection\s+)?form\b"
        ),
        "text": """\
─── Form answers: most recently filled form ───
User: "show all questions and answers in the most recently filled form"
Turn 1: {"thought": "Need most recent inspection that has Inspection Form answers — using aa subquery, NOT inspection_report subquery (newest ir row may have no answers yet). CRITICAL: module_name filter goes on OUTER query only — NEVER inside the subquery or it returns NULL.", "tool": "generate_sql", "args": {"question": "questions and answers from most recent inspection form", "schema_hint": "SELECT aq.label AS question, aa.answer_text AS answer FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id WHERE aa.inspection_report_id = (SELECT aa2.inspection_report_id FROM ai_answers aa2                    ORDER BY aa2.submitted_on DESC LIMIT 1) AND aa.module_name = 'Inspection Form' ORDER BY aq.label LIMIT 100;"}}
""",
    },
    {
        "title": "Scores for a named inspector's last inspection",
        "question_pattern": (
            r"(?i)\bscores?\s+for\s+the\s+last\s+inspection\b"
            r"|\bquestions\s+and\s+scores\s+for\b"
        ),
        "text": """\
─── Scores for a named inspector's last inspection ───
User: "show questions and scores for the last inspection by George"
ROUTING RULE: ALWAYS use generate_sql here. NEVER use get_answers — it cannot filter by inspector name.
Turn 1: {"thought": "Need ai_answers.score filtered to George's last inspection. Must use generate_sql with subquery to find the inspection_report_id first.", "tool": "generate_sql", "args": {"question": "questions and scores for last inspection by George", "schema_hint": "SELECT aq.label AS question, aa.answer_text AS answer, aa.score FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id WHERE aa.inspection_report_id = (SELECT ir.id FROM inspection_report ir JOIN users u ON ir.inspector_user_id = u.id WHERE (u.first_name ILIKE '%George%' OR u.last_name ILIKE '%George%') AND ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 1) AND aa.module_name = 'Inspection Form' AND aa.score IS NOT NULL ORDER BY aa.score DESC LIMIT 100;"}}
""",
    },
    {
        "title": "Form discovery",
        "question_pattern": (
            r"(?i)\bwhich\s+forms?\s+ask\b|\bforms?\s+that\s+ask\s+about\b"
        ),
        "text": """\
─── Form discovery ───
User: "which forms ask about emergency contact?"
Turn 1: {"thought": "Semantic search over question labels.", "tool": "semantic_search", "args": {"query": "emergency contact"}}
""",
    },
    {
        "title": "What questions are in a form",
        "question_pattern": (
            r"(?i)\bwhat\s+questions?\s+are\s+in\b|\bquestions?\s+in\s+the\s+\w+\s+form\b"
        ),
        "text": """\
─── What questions are in a form ───
User: "what questions are in the vehicle inspection form?"
Turn 1: {"thought": "Search ai_questions by form name.", "tool": "search_questions", "args": {"query": "", "form_name": "vehicle inspection", "limit": 50}}
""",
    },
    {
        "title": "Which observation appears most as high risk",
        "question_pattern": (
            r"(?i)\bobservation\b.{0,20}\bappears?\s+most\b"
            r"|\bmost\s+frequent\w*\s+high\s*[- ]?risk\s+observation"
        ),
        "text": """\
─── Which observation appears most as high risk ───
User: "which observation appears most frequently as high risk"
ROUTING RULE: This asks for OBSERVATION TEXT where risk level = High.
Do NOT call get_answer_stats(label="Risk Level") — that returns the risk level distribution.
Do NOT call get_answer_stats(label="Observation") — that ignores the high-risk filter.
MUST use generate_sql with a subquery joining risk level answers to observation answers.
Turn 1: {"thought": "Need observations filtered to inspections where Risk Level = High. Must use SQL subquery — get_answer_stats cannot cross-filter between two questions.", "tool": "generate_sql", "args": {"question": "observation text most frequent in high risk inspections", "schema_hint": "SELECT obs.answer_text AS observation, COUNT(*) AS frequency FROM ai_answers risk_aa JOIN ai_questions risk_aq ON risk_aa.element_id = risk_aq.element_id JOIN inspection_report ir ON risk_aa.inspection_report_id = ir.id JOIN ai_answers obs ON obs.inspection_report_id = ir.id JOIN ai_questions obs_q ON obs.element_id = obs_q.element_id WHERE risk_aq.label = 'Risk Level' AND risk_aa.answer_text ILIKE '%High%' AND obs_q.label = 'Observation' AND obs.answer_text IS NOT NULL AND ir.status != 'DRAFT' GROUP BY obs.answer_text ORDER BY frequency DESC LIMIT 1;"}}
""",
    },
    {
        "title": "Q&A for most recently answered inspection",
        "question_pattern": (
            r"(?i)\bquestions?\s+and\s+answers?\s+for\s+the\s+most\s+recently\s+answered\b"
        ),
        "text": """\
─── Q&A for most recently answered inspection ───
User: "show me the questions and answers for the most recently answered inspection"
ROUTING RULE: ONLY use this when the question explicitly asks for both QUESTIONS AND ANSWERS
(Q&A content from the form). Do NOT use this for "show me the most recently answered
inspection" — that only needs inspection metadata (inspection_id, score, facility) and
should use generate_sql on inspection_report directly.
Turn 1: {"thought": "User wants Q&A content — questions + answers from the form. Must use single SQL with ai_answers + ai_questions JOIN and aa.submitted_on subquery. Do NOT first look up inspection_id separately.", "tool": "generate_sql", "args": {"question": "list all questions and answers from the most recently answered inspection form", "schema_hint": "SELECT aq.label AS question, aa.answer_text AS answer, aa.module_name AS form_section\nFROM ai_answers aa\nJOIN ai_questions aq ON aa.element_id = aq.element_id\nWHERE aa.inspection_report_id = (\n    SELECT aa2.inspection_report_id FROM ai_answers aa2\n    ORDER BY aa2.submitted_on DESC LIMIT 1\n)\nORDER BY aa.module_name, aq.label LIMIT 200;"}}
""",
    },
    {
        "title": "Inspector with most high-impact observations",
        "question_pattern": (
            r"(?i)\bhigh\s*[- ]?impact\s+observations?\b|\bmost\s+impact\w*\s+observations?\b"
        ),
        "text": """\
─── Inspector with most high-impact observations ───
User: "which inspector raises the most high impact observations?"
ROUTING RULE: Impact is a MULTI-SELECT CATEGORY field. There is NO answer_text value called 'High' for Impact.
Categories are: 'Client Relationship', 'Health & Safety (HS)', 'Financial', 'Environmental' etc.
'High impact' = high aa.score (more/heavier categories selected). Count = most Impact answers.
NEVER generate: aq.label ILIKE '%Impact%' OR aa.answer_text ILIKE '%High%'
ALWAYS use: aq.label = 'Impact' (exact match) and ORDER BY aa.score DESC or COUNT(*) DESC
Turn 1: {"thought": "Impact is multi-select category, not severity. 'High impact' = high aa.score. Count Impact answers per inspector ordered by avg score.", "tool": "generate_sql", "args": {"question": "inspector with most high impact score observations", "schema_hint": "SELECT u.first_name || ' ' || u.last_name AS inspector_name, COUNT(*) AS impact_count, ROUND(AVG(aa.score)::numeric, 2) AS avg_impact_score FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id JOIN inspection_report ir ON aa.inspection_report_id = ir.id JOIN users u ON ir.inspector_user_id = u.id WHERE aq.label = 'Impact' AND aa.score IS NOT NULL AND aa.module_name = 'Inspection Form' AND ir.active = true AND ir.status != 'DRAFT' AND u.active = true GROUP BY u.id, u.first_name, u.last_name ORDER BY avg_impact_score DESC LIMIT 10;"}}
""",
    },
    {
        "title": "Count deviations at a facility",
        "question_pattern": r"(?i)\bdeviations?\b",
        "text": """\
─── Count deviations at a facility ───
User: "how many deviations were raised at Al Ghadeer this year?"
ROUTING RULE: 'Deviation' is a KNOWN answer value for 'Observation Type'. Route directly to generate_sql — do NOT call search_questions or semantic_search first.
Turn 1: {"thought": "'Deviation' is the known answer value for aq.label='Observation Type'. Route directly to generate_sql — no question lookup needed.", "tool": "generate_sql", "args": {"question": "count deviations at Al Ghadeer this year", "schema_hint": "SELECT COUNT(*) AS deviation_count FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id JOIN inspection_report ir ON aa.inspection_report_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE aq.label = 'Observation Type' AND aa.answer_text ILIKE '%Deviation%' AND aa.module_name = 'Inspection Form' AND fac.name ILIKE '%Al Ghadeer%' AND ir.active = true AND ir.status != 'DRAFT' AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE);"}}
""",
    },
    {
        "title": "Facilities that never had a good practice",
        "question_pattern": (
            r"(?i)\bnever\s+had\s+a?\s*good\s+practice\b|\bgood\s+practice\b.{0,20}\bnever\b"
        ),
        "text": """\
─── Facilities that never had a good practice ───
User: "which facilities have never had a good practice observation?"
ROUTING RULE: This is a facility-level negation query. Route directly to generate_sql — do NOT call search_questions, semantic_search, or get_answers first.
Turn 1: {"thought": "Facility-level negation — needs NOT IN subquery. 'Good Practice' is a known value of 'Observation Type'. Route directly to generate_sql.", "tool": "generate_sql", "args": {"question": "facilities that have never had a good practice observation", "schema_hint": "SELECT fac.name FROM facility fac WHERE fac.active = true AND fac.id IN (SELECT DISTINCT ir.facility_id FROM inspection_report ir WHERE ir.active = true AND ir.status != 'DRAFT') AND fac.id NOT IN (SELECT DISTINCT ir.facility_id FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id JOIN inspection_report ir ON aa.inspection_report_id = ir.id WHERE aq.label = 'Observation Type' AND aa.answer_text ILIKE '%Good Practice%' AND aa.module_name = 'Inspection Form' AND ir.active = true AND ir.status != 'DRAFT') ORDER BY fac.name;"}}
""",
    },
    {
        "title": "Count of questions in a form",
        "question_pattern": (
            r"(?i)\bhow\s+many\s+questions?\s+(are\s+)?in\b|\bcount\s+of\s+questions?\s+in\b"
        ),
        "text": """\
User: "how many questions are in the inspection form?"
ROUTING RULE: Do NOT use search_questions or list_forms — those return rows not a count. Use generate_sql.
Turn 1: {"thought": "Count distinct questions from ai_questions filtered by module_name.", "tool": "generate_sql", "args": {"question": "count questions in inspection form", "schema_hint": "SELECT COUNT(DISTINCT element_id) AS question_count FROM ai_questions WHERE module_name ILIKE '%Inspection Form%';"}}
""",
    },
    {
        "title": "List forms",
        "question_pattern": r"(?i)\bhow\s+many\s+forms?\s+are\s+there\b|\blist\s+(all\s+)?forms?\b",
        "text": """\
─── List forms ───
User: "how many forms are there?"
Turn 1: {"thought": "list_forms gives distinct form names.", "tool": "list_forms", "args": {}}
""",
    },
    {
        "title": "Average form answer score",
        "question_pattern": r"(?i)\baverage\s+score\s+in\s+the\b|\bavg\s+score\s+in\s+the\b",
        "text": """\
─── Average form answer score ───
User: "average score in the fire safety form"
Turn 1: {"thought": "Aggregate over ai_answer.answer_text for fire safety.", "tool": "get_answer_stats", "args": {"form_name": "fire safety", "label": "score"}}
""",
    },
    {
        "title": "Inspector performance ranking — DO NOT use AVG(inspection_score)",
        "question_pattern": (
            r"(?i)\b(best|top|worst|poorest|highest|lowest)\b.{0,30}\b(inspector|inspectors)\b"
            r"|\b(inspector|inspectors)\b.{0,30}\b(perform\w*|rank\w*|effective\w*|produc\w*)\b"
            r"|\bwhich\s+inspector\b.{0,30}\b(best|worst|most|highest|top)\b"
            r"|\binspector\b.{0,30}\b(leaderboard|ranking|comparison)\b"
        ),
        "text": """\
─── Inspector performance ranking ───
CRITICAL: DO NOT use AVG(ir.inspection_score) to rank inspectors.
inspection_score measures SITE compliance quality, not inspector skill.
An inspector assigned only to well-run facilities always ranks first — facility bias.

"Best inspector" is AMBIGUOUS. The correct behaviour is:
  • If question says "best by volume / most inspections" → rank by COUNT(ir.id) conducted
  • If question says "most thorough / most observations / most findings" → rank by observation count
  • If question says "average site score" or explicitly mentions inspection_score → ok to use AVG(ir.inspection_score) but label it avg_site_score and add inspection_count alongside
  • If "best" is vague → use the CTE pattern (volume + observations + high-risk) OR ask for clarification

NEVER alias the same aggregate twice (COUNT(*) AS x AND COUNT(*) AS y in same SELECT).

User: "who are the top 10 best performing inspectors?"
Turn 1: {"thought": "'Best performing' is ambiguous — volume? findings? site score? I must NOT use AVG(inspection_score). I should ask for clarification.", "tool": "final_answer", "args": {"answer": "Could you clarify what you mean by 'best performing'?\n• Most inspections conducted\n• Most observations/deviations found\n• Most high-risk observations found\n• Highest average site compliance score (note: this reflects facility quality, not inspector skill)\n\nLet me know and I'll pull the right data."}}

User follow-up: "most inspections conducted" (user answered the clarification — EXECUTE NOW, do NOT ask again)
Turn 1: {"thought": "User just clarified: 'most inspections conducted' = volume. Previous turn was my clarification question. I must NOT ask again — proceed immediately with CTE pattern.", "tool": "generate_sql", "args": {"question": "top inspectors by inspection volume and observations found", "schema_hint": "WITH obs AS (SELECT aa.inspection_report_id, COUNT(DISTINCT aa.submission_id) AS total_obs, COUNT(DISTINCT CASE WHEN aq.label = 'Risk Level' AND aa.answer_text ILIKE '%High%' THEN aa.submission_id END) AS high_risk_obs FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id WHERE aa.module_name = 'Inspection Form' GROUP BY aa.inspection_report_id) SELECT u.first_name || ' ' || u.last_name AS inspector_name, COUNT(ir.id) AS assigned, COUNT(ir.id) FILTER (WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION','CLOSED')) AS conducted, COALESCE(SUM(obs.total_obs),0) AS total_observations, COALESCE(SUM(obs.high_risk_obs),0) AS high_risk_observations FROM users u JOIN inspection_report ir ON ir.inspector_user_id = u.id AND ir.active = true LEFT JOIN obs ON obs.inspection_report_id = ir.id WHERE u.active = true AND ir.status != 'DRAFT' GROUP BY u.id, u.first_name, u.last_name ORDER BY conducted DESC LIMIT 10;"}}

User: "which inspector has the lowest average inspection score?" (explicitly about inspection_score)
Turn 1: {"thought": "User explicitly asked for average inspection_score per inspector — this is the avg SITE score for their assigned inspections, not inspector quality. I'll label it clearly.", "tool": "generate_sql", "args": {"question": "inspector with lowest average site inspection score", "schema_hint": "SELECT u.first_name || ' ' || u.last_name AS inspector_name, ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_site_score, COUNT(*) AS inspection_count FROM inspection_report ir JOIN users u ON ir.inspector_user_id = u.id WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL GROUP BY u.id, u.first_name, u.last_name ORDER BY avg_site_score ASC LIMIT 1;"}}
""",
    },
    {
        "title": "FM manager performance ranking — domain-correct metrics, NOT inspector metrics",
        "question_pattern": (
            r"(?i)\b(best|top|worst|highest|lowest|top\s+\d+)\b.{0,40}\b(manager|managers|fm\s+manager|facility\s+manager)\b"
            r"|\b(manager|fm\s+manager|facility\s+manager)s?\b.{0,30}\b(perform\w*|rank\w*|best|top|worst|leaderboard)\b"
            r"|\bwhich\s+(manager|fm\s+manager|facility\s+manager)\b.{0,30}\b(best|most|highest|top)\b"
        ),
        "text": """\
─── FM manager performance ranking ───
CRITICAL: FM managers are INSPECTEES (inspectee_user_id). They do NOT conduct inspections.
NEVER offer "most inspections conducted" or "most observations found" for FM managers.

FM manager metrics (domain-correct):
  • Capex converted: SUM(po_details.po_amount) via inspectee_user_id
  • Mitigative actions closed: COUNT(ica WHERE status='CLOSED')
  • Open / overdue MAs: COUNT(ica WHERE status != 'CLOSED')
  • Average facility compliance: AVG(ir.inspection_score) on their sites

"Best manager" is AMBIGUOUS on two dimensions: entity type AND metric.
When both are unclear → ask BOTH in ONE message (never two separate turns).

User: "who are the top 10 best managers?"
Turn 1: {"thought": "'Manager' is ambiguous (inspector vs FM manager) and 'best' metric is also unknown. I must ask BOTH dimensions together in ONE message — not ask entity first then metric next turn.", "tool": "final_answer", "args": {"answer": "Could you clarify who and what 'best' means?\n\n**If you mean Inspectors** (people who conduct inspections):\n• Most inspections conducted\n• Most observations/findings raised\n• Most high-risk observations found\n\n**If you mean FM Managers** (facility managers being inspected):\n• Most capex converted (highest PO amounts paid by client)\n• Most mitigative actions closed\n• Highest average facility compliance score\n\nJust say which group and metric and I'll pull the data."}}

User follow-up: "i mean FM managers by most capex converted" (user answered both — DO NOT ask again, execute immediately)
Turn 1: {"thought": "User clarified: FM managers (inspectee_user_id) ranked by capex converted (po_details). Previous turn was my clarification. I must NOT ask again. Proceed.", "tool": "generate_sql", "args": {"question": "top FM managers by capex converted using po_details", "schema_hint": "WITH po_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po FROM po_details WHERE active = true GROUP BY 1) SELECT u.first_name || ' ' || u.last_name AS fm_manager, COALESCE(SUM(ica.capex), 0) AS capex_generated, COALESCE(SUM(p.total_po), 0) AS capex_converted, ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_rate_pct FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id AND ir.active = true JOIN users u ON ir.inspectee_user_id = u.id AND u.active = true LEFT JOIN po_by_ica p ON p.ica_id = ica.id WHERE ica.active = true AND ica.capex > 0 GROUP BY u.id, u.first_name, u.last_name ORDER BY capex_converted DESC LIMIT 10;"}}

User follow-up: "i mean FM managers" alone (user answered entity only — ask metric ONCE more with FM-correct options, then stop asking)
Turn 1: {"thought": "User clarified entity = FM managers. I know to use inspectee_user_id. Still need metric — I will ask ONCE more about metric only, with FM-manager-appropriate options, NEVER inspector options.", "tool": "final_answer", "args": {"answer": "Got it — FM managers. What metric should I use to rank them?\n• Most capex converted (highest PO amounts paid by client)\n• Most mitigative actions closed\n• Highest average facility compliance score\n\nLet me know and I'll run it."}}

User follow-up after "i mean FM managers": "most capex converted" (DO NOT ask again — execute immediately)
Turn 1: {"thought": "User answered metric. Entity already confirmed (FM managers = inspectee_user_id). Metric = capex converted = SUM(po_details.po_amount). Must NOT ask again.", "tool": "generate_sql", "args": {"question": "top FM managers by capex converted using po_details", "schema_hint": "WITH po_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po FROM po_details WHERE active = true GROUP BY 1) SELECT u.first_name || ' ' || u.last_name AS fm_manager, COALESCE(SUM(ica.capex), 0) AS capex_generated, COALESCE(SUM(p.total_po), 0) AS capex_converted FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id AND ir.active = true JOIN users u ON ir.inspectee_user_id = u.id AND u.active = true LEFT JOIN po_by_ica p ON p.ica_id = ica.id WHERE ica.active = true AND ica.capex > 0 GROUP BY u.id, u.first_name, u.last_name ORDER BY capex_converted DESC LIMIT 10;"}}
""",
    },
]
