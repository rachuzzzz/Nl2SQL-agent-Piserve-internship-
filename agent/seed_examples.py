"""
Seed examples for NL→SQL retrieval.

Each entry is a dict with:
  nl:     natural language question (used for embedding + CrossEncoder)
  sql:    canonical correct SQL
  domain: financial | corrective_action | inspection | schedule | form_answers | general
  intent: single_value | ranked_list | aggregate | filter | trend
  limit:  int or None — the LIMIT value in the SQL

STATUS GROUPINGS + CAST RULES come from core.semantics_registry — the single
source of truth. Edit there, not here.
"""

import re

# ── Registry-generated blocks (single source of truth) ──────────────────────
try:
    from core.semantics_registry import (
        core_context_status_block as _status_block,
        core_context_cast_block as _cast_block,
        core_context_counting_block as _counting_block,
        core_context_columns_block as _columns_block,
        core_context_ca_workflow_block as _ca_workflow_block,
    )
    _STATUS_BLOCK = _status_block()
    _CAST_BLOCK = _cast_block()
    _COUNTING_BLOCK = _counting_block()
    _COLUMNS_BLOCK = _columns_block()
    _CA_WORKFLOW_BLOCK = _ca_workflow_block()
except ImportError:
    # Fallback — keep in sync with core/semantics_registry.py
    _STATUS_BLOCK = """\
-- DRAFT RECORDS ARE NEVER COUNTED — GLOBAL RULE:
--   DRAFT inspections are incomplete — they must NEVER appear in any count, list, average, or aggregate.
--
-- STATUS GROUPINGS — always bucket into two categories (never expose raw status names to user):
--   conducted  = status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION') — in-flight
--   completed  = status = 'CLOSED'                                                — fully approved
--
-- DEFAULT for "total inspections / how many inspections / inspections in <month>":
--   Show as TWO-ROW summary using CASE grouping:
--     CASE WHEN ir.status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category
--   NEVER show raw status breakdown (SUBMITTED/UNDER_REVIEW/etc) unless user explicitly asks
--   NEVER use status IN ('CLOSED','SUBMITTED') — that mixes conducted and completed
--   NEVER use status != 'DRAFT' alone for conducted — CLOSED is not conducted"""
    _CAST_BLOCK = """\
-- OBSERVATION → CA JOIN — exact form:
--   aa.submission_id = ica.observation_id::varchar
--   NEVER: aa.submission_id::uuid = ica.observation_id
--   NEVER: ica.observation_id::text = aa.submission_id"""
    _COUNTING_BLOCK = """\
-- GOOD PRACTICE / DEVIATION COUNTING — always COUNT(DISTINCT ir.id), NEVER COUNT(*):
--   each inspection is Good Practice OR Deviation (mutually exclusive); COUNT(*) on ai_answers rows overcounts.
--   Same rule applies to deviation counts.
--   WRONG:  SELECT COUNT(*) ... WHERE aq.label='Observation Type' AND aa.answer_text ILIKE '%Good Practice%'
--   CORRECT: SELECT COUNT(DISTINCT ir.id) ... same filters
-- OBSERVATIONS PER INSPECTION — use COUNT(DISTINCT aa.submission_id):
--   one submission_id per observation group; COUNT(*) counts every answer row (Risk Level, Impact, ...)"""
    _COLUMNS_BLOCK = """\
-- SOFT-DELETE: these tables have an 'active' boolean column — ALWAYS filter WHERE active = true:
--   inspection_report, inspection_corrective_action, facility, users,
--   inspection_type, inspector_portfolio, inspector_portfolio_details,
--   frequency_definition, po_details, pr_details, risk_level
-- NO active column: ai_answers, ai_questions, role, user_roles, inspection_schedule
--   NEVER use aa.active or aq.active — these columns do not exist"""
    _CA_WORKFLOW_BLOCK = """\
-- CORRECTIVE ACTION WORKFLOW — four independent tracks on ica:
--   status (system):            OPEN | CLOSED | OVERDUE | CLOSE_WITH_DEFERRED
--   progress_stage (inspectee): NOT_STARTED, ON_TRACK, BEHIND, AT_RISK, COMPLETED, CLOSE_WITH_DEFERRED
--   implementation_status (reviewers): NULL | PENDING_FOR_REVIEW | APPROVE | RETURN_FOR_MODIFICATION
--     EXACT STRINGS: 'APPROVE' not 'APPROVED' — 'PENDING_FOR_REVIEW' not 'PENDING'
--   capex_status (USER-ENTERED, declarative only): NULL | ACCEPTED | REJECTED | DEFERRED
--     po_details is the authoritative money record — POs legitimately exist on REJECTED rows
-- INVARIANTS:
--   status = 'CLOSED'  ⟺  implementation_status = 'APPROVE' (closure IS the approval)
--   RETURN_FOR_MODIFICATION resets progress_stage to NULL — progress_stage='COMPLETED' MISSES returned CAs
--   inspection_report closes only when ALL its CAs are closed
-- TRAP: 'at risk' = progress_stage='AT_RISK' (schedule health), NOT risk_level='High' (severity)"""

CORE_CONTEXT = """\

### SQL RULES FOR DEEPSEEK (not for end user):
-- ir.inspection_id (VARCHAR) = display id | ir.id (UUID) = JOIN/subquery only, NEVER SELECT
-- ICA join: ON ica.inspection_id = ir.id  (NOT ir.inspection_id = ica.inspection_id)
-- responsible: 'CLIENT','INTERNAL_OPERATIONS','SUB_CONTRACTOR'
""" + _STATUS_BLOCK + """
--
""" + _CAST_BLOCK + """
--
-- inspection_schedule columns: schedule_date (NOT scheduled_date), due_date, schedule_id, status
--   NEVER: sched.scheduled_date — column is schedule_date
--   NEVER: sched.active — inspection_schedule has no active column
--
-- UNION / UNION ALL TYPE SAFETY:
--   Every column position must have the SAME type in all branches.
--   NEVER 'N/A' placeholder against a date/numeric/enum column — runtime type-mismatch error.
--   Placeholders must be explicitly typed: NULL::text, NULL::date, NULL::numeric.
--   When mixing tables, cast real date/enum columns ::text in BOTH branches.
--   PREFER scalar subqueries in one row over UNION of heterogeneous tables.
-- NEVER: ir.id in SELECT | NEVER: ica.risk_level_id vs string | NEVER: GROUP BY ica.risk_level_id
-- completed_cycle = status='COMPLETED' (NOT 'CLOSED', NOT status!='DRAFT')
-- NEVER: ipd.cycle_id (no such column on inspector_portfolio_details)
-- Cycle->Freq: inspection_cycle.id -> inspector_portfolio.cycle_id
--   -> inspector_portfolio_details.portfolio_id -> frequency_definition.id
-- FM Manager / inspectee = ir.inspectee_user_id (NOT ir.inspector_user_id)
-- capex_status: NULL=awaiting approval | ACCEPTED | REJECTED | DEFERRED
-- expenditure: NO_EXPENDITURE | CAPEX | OPEX | CAPEX_AND_OPEX
--
-- BUSINESS WORKFLOW (Khidmah inspection system):
--   Admin sets inspection schedules → Inspectors submit portfolios → Inspectors visit sites
--   Inspector creates Inspection Report (ir) with Observations (Good Practice or Deviation)
--   Deviation observations have a risk score (risk_level) and observation text
--   FM Manager (inspectee) raises Corrective Actions (ICA) against those observations
--   Each ICA has a proposed cost (ica.capex) and/or internal cost (ica.opex)
--   Client pays via Purchase Orders → po_details (CAPEX Converted — CLIENT pays — GOOD)
--   Khidmah pays via Purchase Requisitions → pr_details (OPEX Incurred — KHIDMAH pays — BAD)
--   ai_answers is a complete flattened snapshot — every inspection_report has ai_answers rows
--
-- FINANCIAL METRICS:
-- CAPEX Generated  = SUM(ica.capex)        — FM manager's proposed cost on the ICA
-- CAPEX Converted  = SUM(po.po_amount)     — actual PO raised by client (good: client paying)
-- CAPEX Lost       = SUM(ica.capex) - SUM(po.po_amount) — arithmetic gap (NOT capex_status='REJECTED')
-- OPEX Incurred    = SUM(pr.pr_amount)     — Khidmah's own spend via PR (bad: org paying itself)
-- po_details JOIN: ON po.mitigative_action_uuid = ica.id  (one ICA can have many POs)
-- pr_details JOIN: ON pr.mitigative_action_uuid = ica.id  (one ICA can have many PRs)
--
-- FM MANAGER PERFORMANCE = high CAPEX conversion rate + low OPEX
--   Conversion rate = SUM(po.po_amount) / NULLIF(SUM(ica.capex), 0) * 100
--   High OPEX means Khidmah is absorbing costs the client should bear — a negative indicator
--
""" + _COLUMNS_BLOCK + """
--
-- DATE FILTERING — month/year specific:
--   CORRECT:  EXTRACT(MONTH FROM ir.submitted_on) = 1 AND EXTRACT(YEAR ...) = EXTRACT(YEAR FROM CURRENT_DATE)
--   CORRECT:  ir.submitted_on >= '2026-01-01' AND ir.submitted_on < '2026-02-01'  (only when year explicit)
--   WRONG:    TO_CHAR(ir.submitted_on, 'Month YYYY') = 'January 2026'
--             ↑ NEVER use for WHERE filters — TO_CHAR pads month with spaces → 0 rows
--   WRONG:    date_trunc('month', ir.submitted_on) = date_trunc('month', DATE '2023-01-01')
--             ↑ NEVER hardcode a year like 2023 or 2024 in date literals — use CURRENT_DATE
--   TO_CHAR is ONLY valid for GROUP BY labels / SELECT display, never WHERE filters
"""

SEED_EXAMPLES: list[dict] = [
    {
        # Lookup inspector or details for a specific inspection by its full ID.
        # inspection_id format is always 'YYYY/MM/STXXX/INSXXX' — NEVER truncate it.
        # ir.inspection_id is VARCHAR — always match the FULL string as given by the user.
        'nl':     'who inspected inspection 2026/01/ST074/INS002 inspector for specific inspection details about inspection by id inspector of first inspection in the list who did this inspection',
        'sql':    """\
-- inspection_id is the FULL VARCHAR: '2026/01/ST074/INS002' — NEVER truncate to 'ST074/INS002'
SELECT ir.inspection_id,
       u.first_name || ' ' || u.last_name AS inspector_name,
       fac.name AS facility_name,
       it.name  AS inspection_type_name,
       ir.inspection_score,
       ir.status,
       ir.submitted_on
FROM inspection_report ir
JOIN users u   ON ir.inspector_user_id = u.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE ir.inspection_id = '2026/01/ST074/INS002'
  AND ir.active = true
  AND ir.status != 'DRAFT';
""",
        'domain': 'inspection',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'how many inspections were completed this year total inspections this year inspections in 2026 how many inspections in a month',
        'sql':    """\
-- DEFAULT for "total/how many inspections" — show as conducted vs completed two-row summary
-- conducted = SUBMITTED + UNDER_REVIEW + RETURN_FOR_MODIFICATION (in-flight, non-draft, non-closed)
-- completed = CLOSED only
-- NEVER show raw status breakdown unless user explicitly asks to drill in
SELECT
    CASE WHEN ir.status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category,
    COUNT(*) AS inspection_count
FROM inspection_report ir
WHERE ir.active = true
  AND ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY category
ORDER BY category DESC;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'total inspections in a specific month inspections in March 2026 inspections in February how many inspections in March',
        'sql':    """\
-- For a specific month/year: conducted vs completed two-row summary
-- conducted = IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')
-- completed = CLOSED
SELECT
    CASE WHEN ir.status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category,
    COUNT(*) AS inspection_count
FROM inspection_report ir
WHERE ir.active = true
  AND ir.status != 'DRAFT'
  AND EXTRACT(MONTH FROM ir.submitted_on) = 3
  AND EXTRACT(YEAR FROM ir.submitted_on) = 2026
GROUP BY category
ORDER BY category DESC;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show only conducted inspections drill into conducted how many conducted inspections in March',
        'sql':    """\
-- Drill-down into conducted only
-- conducted = IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')
-- NEVER: status != 'DRAFT' AND status != 'CLOSED' — use explicit IN list, Deepseek drops double negation
SELECT COUNT(*) AS conducted_count
FROM inspection_report ir
WHERE ir.active = true
  AND ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')
  AND EXTRACT(MONTH FROM ir.submitted_on) = 3
  AND EXTRACT(YEAR FROM ir.submitted_on) = 2026;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show only completed inspections drill into completed how many completed inspections in March',
        'sql':    """\
-- Drill-down into completed only
-- completed = status = 'CLOSED' only
SELECT COUNT(*) AS completed_count
FROM inspection_report ir
WHERE ir.active = true
  AND ir.status = 'CLOSED'
  AND EXTRACT(MONTH FROM ir.submitted_on) = 3
  AND EXTRACT(YEAR FROM ir.submitted_on) = 2026;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'what needs attention right now critical issues urgent items what should I look at dashboard summary action items',
        'sql':    """\
-- "Needs attention" dashboard — ONE ROW of scalar subqueries.
-- NEVER UNION ALL heterogeneous tables (CA list + schedule list) — column types
-- clash ('N/A' text vs date/enum) and the query fails at runtime.
SELECT
    (SELECT COUNT(*) FROM inspection_corrective_action
     WHERE status = 'OVERDUE' AND active = true)                    AS overdue_corrective_actions,
    (SELECT COUNT(*) FROM inspection_corrective_action
     WHERE status = 'OPEN' AND active = true)                       AS open_corrective_actions,
    (SELECT COUNT(*) FROM inspection_schedule
     WHERE status = 'OVERDUE')                                      AS overdue_scheduled_inspections,
    (SELECT COUNT(*) FROM inspection_corrective_action ica
     JOIN risk_level rl ON ica.risk_level_id = rl.id
     WHERE rl.name = 'High' AND ica.status IN ('OPEN','OVERDUE')
       AND ica.active = true)                                       AS high_risk_open_cas;
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'how many open corrective actions are there',
        'sql':    """\
SELECT COUNT(*) AS open_count
FROM inspection_corrective_action ica
WHERE ica.active = true AND ica.status = 'OPEN';
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'corrective actions pending approval awaiting review CAs waiting for FM manager approval pending review',
        'sql':    """\
-- Pending approval = implementation_status = 'PENDING_FOR_REVIEW' (EXACT string — not 'PENDING')
-- These are CAs the inspectee marked COMPLETED, awaiting FM manager + inspector approval
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.progress_stage, ica.implementation_status,
       fac.name AS facility_name, ir.inspection_id
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.implementation_status = 'PENDING_FOR_REVIEW'
  AND ica.active = true AND ir.active = true
ORDER BY ica.target_close_out_date ASC NULLS LAST
LIMIT 100;
""",
        'domain': 'corrective_action',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'corrective actions sent back returned for modification rejected by reviewer CAs returned to inspectee',
        'sql':    """\
-- Sent back = implementation_status = 'RETURN_FOR_MODIFICATION'
-- NOTE: this resets progress_stage to NULL — do not also filter progress_stage
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.implementation_status,
       fac.name AS facility_name, ir.inspection_id
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.implementation_status = 'RETURN_FOR_MODIFICATION'
  AND ica.active = true AND ir.active = true
ORDER BY ica.target_close_out_date ASC NULLS LAST
LIMIT 100;
""",
        'domain': 'corrective_action',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'corrective actions at risk behind schedule on track not started progress breakdown CA progress status',
        'sql':    """\
-- progress_stage = inspectee's self-reported SCHEDULE HEALTH (not severity!)
-- 'at risk' here = progress_stage='AT_RISK' — NOT risk_level='High'
-- Values: NOT_STARTED, ON_TRACK, BEHIND, AT_RISK, COMPLETED, CLOSE_WITH_DEFERRED
-- NULL progress_stage = not started yet (fresh CA)
SELECT COALESCE(ica.progress_stage, 'NOT_STARTED') AS progress_stage,
       COUNT(*) AS ca_count
FROM inspection_corrective_action ica
WHERE ica.active = true
  AND ica.status != 'CLOSED'
GROUP BY ica.progress_stage
ORDER BY ca_count DESC;
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'income cost and profit by facility P&L per facility revenue spend by facility capex converted vs opex incurred per facility financial data PO PR purchase order purchase requisition by facility show POs and PRs at facility income and cost at specific facility',
        'sql':    """\
-- P&L by facility: Income = SUM(po.po_amount), Cost = SUM(pr.pr_amount)
-- CRITICAL JOIN PATH: mitigative_action_uuid → ica.id (the CA) — NEVER ir.id!
-- po/pr_details → ica → inspection_report → facility (the ICA hop is mandatory)
-- WRONG: mitigative_action_uuid = ir.id — matches ZERO rows → silent all-zero table
-- FACILITY FILTER: NEVER put fac.name ILIKE inside a CTE — fac is not in scope there.
-- Filter by fac.name ILIKE in the outer WHERE clause only, after joining facility.
WITH po_agg AS (
    SELECT ir.facility_id, SUM(pd.po_amount) AS total_income
    FROM po_details pd
    JOIN inspection_corrective_action ica ON pd.mitigative_action_uuid = ica.id
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    WHERE ir.active = true AND ir.status != 'DRAFT' AND ica.active = true
    GROUP BY ir.facility_id
),
pr_agg AS (
    SELECT ir.facility_id, SUM(pr.pr_amount) AS total_cost
    FROM pr_details pr
    JOIN inspection_corrective_action ica ON pr.mitigative_action_uuid = ica.id
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    WHERE ir.active = true AND ir.status != 'DRAFT' AND ica.active = true
    GROUP BY ir.facility_id
)
SELECT fac.name AS facility_name,
       COALESCE(po.total_income, 0)                              AS income,
       COALESCE(pr.total_cost, 0)                                AS cost,
       COALESCE(po.total_income, 0) - COALESCE(pr.total_cost, 0) AS profit
FROM facility fac
LEFT JOIN po_agg po ON po.facility_id = fac.id
LEFT JOIN pr_agg pr ON pr.facility_id = fac.id
WHERE fac.active = true
  AND (po.total_income IS NOT NULL OR pr.total_cost IS NOT NULL)
ORDER BY income DESC
LIMIT 100;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  100,
    },
    {
        'nl':     'financial data for specific facility PO income PR cost profit at one facility show POs and PRs for Al Ghadeer income and cost for facility name',
        'sql':    """\
-- Financial summary for a SPECIFIC facility by name
-- CRITICAL: fac.name ILIKE filter goes in OUTER WHERE only — NEVER inside a CTE
-- CTEs aggregate by facility_id; outer query joins facility and filters by name
WITH po_agg AS (
    SELECT ir.facility_id, SUM(pd.po_amount) AS total_income
    FROM po_details pd
    JOIN inspection_corrective_action ica ON pd.mitigative_action_uuid = ica.id
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    WHERE ir.active = true AND ir.status != 'DRAFT' AND ica.active = true
    GROUP BY ir.facility_id
),
pr_agg AS (
    SELECT ir.facility_id, SUM(pr.pr_amount) AS total_cost
    FROM pr_details pr
    JOIN inspection_corrective_action ica ON pr.mitigative_action_uuid = ica.id
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    WHERE ir.active = true AND ir.status != 'DRAFT' AND ica.active = true
    GROUP BY ir.facility_id
)
SELECT fac.name AS facility_name,
       COALESCE(po.total_income, 0)                               AS income,
       COALESCE(pr.total_cost, 0)                                 AS cost,
       COALESCE(po.total_income, 0) - COALESCE(pr.total_cost, 0)  AS profit
FROM facility fac
LEFT JOIN po_agg po ON po.facility_id = fac.id
LEFT JOIN pr_agg pr ON pr.facility_id = fac.id
WHERE fac.name ILIKE '%Al Ghadeer%'
  AND fac.active = true;
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'corrective actions not closed on time closed late past deadline late closure overdue when closed',
        'sql':    """\
-- 'closed late' / 'not closed on time' — MUTUALLY EXCLUSIVE from 'overdue':
--   overdue     = still OPEN, past due, has NOT closed yet
--   closed_late = DID close (status='CLOSED'), but completed_on was after target_close_out_date
-- close_on is a DEAD COLUMN (all NULL) — use completed_on for closure timing
-- NEVER OR these two concepts together into one count
SELECT
    COUNT(*) FILTER (
        WHERE status NOT IN ('CLOSED','CLOSE_WITH_DEFERRED')
          AND target_close_out_date < CURRENT_DATE
    ) AS overdue_count,
    COUNT(*) FILTER (
        WHERE status = 'CLOSED' AND completed_on > target_close_out_date
    ) AS closed_late_count
FROM inspection_corrective_action
WHERE active = true AND target_close_out_date IS NOT NULL;
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'list all cycles show inspection cycles what cycles are available inspection frequency cycles cycle types',
        'sql':    """\
-- "cycles" = inspection frequency/cadence types from frequency_definition
-- (Daily, Weekly, Quarterly Twice, etc.) — NOT inspection_cycle (cycle INSTANCES).
-- frequency_definition columns: id, name, repeat_count, repeat_interval, repeat_unit, active
-- NO 'description' column — do not select fd.description (hallucinated column)
SELECT name AS frequency_name, repeat_count, repeat_interval, repeat_unit
FROM frequency_definition
WHERE active = true
ORDER BY name;
""",
        'domain': 'general',
        'intent': 'filter',
        'limit':  None,
    },
    {
        'nl':     'what is the average inspection score',
        'sql':    """\
SELECT ROUND(AVG(ir.inspection_score)::numeric, 2) AS average_score
FROM inspection_report ir
WHERE ir.active = true
  AND ir.status != 'DRAFT'
  AND ir.inspection_score IS NOT NULL;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'total capex and opex across all corrective actions',
        'sql':    """\
SELECT SUM(ica.capex) AS total_capex, SUM(ica.opex) AS total_opex
FROM inspection_corrective_action ica
WHERE ica.active = true;
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'what is the total income cost and net profit from corrective actions P&L profit loss',
        'sql':    """\
-- P&L MODEL: income = po_details.po_amount (client paid Khidmah)
--            cost   = pr_details.pr_amount (Khidmah paid out)
--            profit = income - cost
-- ica.capex is a QUOTE only — NOT income. NEVER use ica.capex as revenue.
-- Pre-aggregate in CTEs to avoid row fan-out from multiple POs/PRs per ICA
WITH po_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_income
    FROM po_details WHERE active = true GROUP BY 1
),
pr_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_cost
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    COALESCE(SUM(po.total_income), 0)                                          AS total_income,
    COALESCE(SUM(pr.total_cost),  0)                                           AS total_cost,
    COALESCE(SUM(po.total_income), 0) - COALESCE(SUM(pr.total_cost), 0)        AS net_profit
FROM inspection_corrective_action ica
LEFT JOIN po_agg po ON po.ica_id = ica.id
LEFT JOIN pr_agg pr ON pr.ica_id = ica.id
WHERE ica.active = true;
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show profit and loss by facility most profitable facility income cost profit',
        'sql':    """\
-- P&L by facility: income = SUM(po.po_amount), cost = SUM(pr.pr_amount), profit = income - cost
WITH po_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_income
    FROM po_details WHERE active = true GROUP BY 1
),
pr_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_cost
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    fac.name AS facility_name,
    COALESCE(SUM(po.total_income), 0)                                    AS income,
    COALESCE(SUM(pr.total_cost),  0)                                     AS cost,
    COALESCE(SUM(po.total_income), 0) - COALESCE(SUM(pr.total_cost), 0) AS profit
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN po_agg po ON po.ica_id = ica.id
LEFT JOIN pr_agg pr ON pr.ica_id = ica.id
WHERE ica.active = true AND ir.active = true AND fac.active = true
GROUP BY fac.id, fac.name
ORDER BY profit DESC;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  None,
    },
    {
        'nl':     'show profit and loss by FM manager which FM manager is most profitable',
        'sql':    """\
-- P&L by FM manager: inspectee_user_id is the FM manager on inspection_report
WITH po_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_income
    FROM po_details WHERE active = true GROUP BY 1
),
pr_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_cost
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    u.first_name || ' ' || u.last_name AS fm_manager,
    COALESCE(SUM(po.total_income), 0)                                    AS income,
    COALESCE(SUM(pr.total_cost),  0)                                     AS cost,
    COALESCE(SUM(po.total_income), 0) - COALESCE(SUM(pr.total_cost), 0) AS profit
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspectee_user_id = u.id
LEFT JOIN po_agg po ON po.ica_id = ica.id
LEFT JOIN pr_agg pr ON pr.ica_id = ica.id
WHERE ica.active = true AND ir.active = true AND u.active = true
GROUP BY u.id, u.first_name, u.last_name
ORDER BY profit DESC;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  None,
    },
    {
        'nl':     'show profit and loss by region income cost profit by region',
        'sql':    """\
WITH po_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_income
    FROM po_details WHERE active = true GROUP BY 1
),
pr_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_cost
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    reg.name AS region_name,
    COALESCE(SUM(po.total_income), 0)                                    AS income,
    COALESCE(SUM(pr.total_cost),  0)                                     AS cost,
    COALESCE(SUM(po.total_income), 0) - COALESCE(SUM(pr.total_cost), 0) AS profit
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN region reg ON fac.region_id = reg.id
LEFT JOIN po_agg po ON po.ica_id = ica.id
LEFT JOIN pr_agg pr ON pr.ica_id = ica.id
WHERE ica.active = true AND ir.active = true AND fac.active = true AND reg.active = true
GROUP BY reg.id, reg.name
ORDER BY profit DESC;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  None,
    },
    {
        'nl':     'show profit and loss by entity income cost profit by entity',
        'sql':    """\
WITH po_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_income
    FROM po_details WHERE active = true GROUP BY 1
),
pr_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_cost
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    ent.name AS entity_name,
    COALESCE(SUM(po.total_income), 0)                                    AS income,
    COALESCE(SUM(pr.total_cost),  0)                                     AS cost,
    COALESCE(SUM(po.total_income), 0) - COALESCE(SUM(pr.total_cost), 0) AS profit
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN entity ent ON ir.entity_id = ent.id
LEFT JOIN po_agg po ON po.ica_id = ica.id
LEFT JOIN pr_agg pr ON pr.ica_id = ica.id
WHERE ica.active = true AND ir.active = true AND ent.active = true
GROUP BY ent.id, ent.name
ORDER BY profit DESC;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  None,
    },
    {
        'nl':     'what percentage of inspections resulted in at least one corrective action',
        'sql':    """\
-- USE THIS EXACT SQL — scalar percentage, no GROUP BY:
SELECT ROUND(
    100.0 * COUNT(DISTINCT ica.inspection_id) / NULLIF(COUNT(DISTINCT ir.id), 0), 1
) AS pct_with_ca
FROM inspection_report ir
LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id
WHERE ir.active = true AND ir.status != 'DRAFT';
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'average inspection duration in hours',
        'sql':    """\
-- USE THIS EXACT PATTERN — total_inspection_hours does NOT exist; use timestamp math:
SELECT ROUND(
    AVG(EXTRACT(EPOCH FROM (ir.submitted_on - ir.start_date_time)) / 3600)::numeric, 2
) AS avg_duration_hours
FROM inspection_report ir
WHERE ir.active = true
  AND ir.status != 'DRAFT'
  AND ir.start_date_time IS NOT NULL;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'average number of observations per inspection',
        'sql':    """\
-- Subquery pattern: count per inspection FIRST, then AVG
SELECT ROUND(AVG(obs_count)::numeric, 2) AS avg_obs_per_inspection
FROM (
    SELECT aa.inspection_report_id, COUNT(*) AS obs_count
    FROM ai_answers aa
    JOIN ai_questions aq ON aa.element_id = aq.element_id
    WHERE aq.label = 'Observation' AND aa.answer_text IS NOT NULL
    GROUP BY aa.inspection_report_id
) sub;
""",
        'domain': 'form_answers',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'how many inspections were completed this month vs last month conducted completed comparison month over month',
        'sql':    """\
-- Month-over-month: conducted vs completed two-row summary for this month and last month
-- conducted = SUBMITTED + UNDER_REVIEW + RETURN_FOR_MODIFICATION
-- completed = CLOSED
SELECT
    CASE WHEN ir.status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category,
    COUNT(*) FILTER (WHERE date_trunc('month', ir.submitted_on) = date_trunc('month', NOW()))            AS this_month,
    COUNT(*) FILTER (WHERE date_trunc('month', ir.submitted_on) = date_trunc('month', NOW() - INTERVAL '1 month')) AS last_month
FROM inspection_report ir
WHERE ir.active = true
  AND ir.status != 'DRAFT'
  AND ir.submitted_on >= date_trunc('month', NOW() - INTERVAL '1 month')
GROUP BY category
ORDER BY category DESC;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'how many corrective actions were raised vs closed this quarter',
        'sql':    """\
-- USE THIS EXACT SQL — single-row FILTER aggregate, NOT GROUP BY:
-- Single-row FILTER — NOT GROUP BY status
SELECT COUNT(*) AS total_raised,
       COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS total_closed,
       COUNT(*) FILTER (WHERE status IN ('OPEN','OVERDUE')) AS still_open
FROM inspection_corrective_action
WHERE active = true
  AND created_on >= date_trunc('quarter', NOW());
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'how has the average inspection score changed month by month this year',
        'sql':    """\
SELECT TO_CHAR(date_trunc('month', ir.submitted_on), 'Month YYYY') AS month,
       COUNT(*) AS inspection_count,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score
FROM inspection_report ir
WHERE ir.active = true
  AND ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY date_trunc('month', ir.submitted_on)
ORDER BY date_trunc('month', ir.submitted_on);
""",
        'domain': 'inspection',
        'intent': 'trend',
        'limit':  None,
    },
    {
        # CRITICAL: date range filter for a specific month.
        # TO_CHAR(submitted_on, 'Month YYYY') = 'January 2026' ALWAYS RETURNS 0 ROWS
        # because TO_CHAR pads month names with spaces ('January   2026' ≠ 'January 2026').
        # Use EXTRACT(MONTH/YEAR) for dynamic month filters — NEVER hardcode a year like DATE '2023-01-01'
        'nl':     'list inspections submitted in January list inspections from January 2026 filter inspections by month January',
        'sql':    """\
-- MONTH FILTER — use EXTRACT for dynamic month+year, NEVER TO_CHAR in WHERE, NEVER hardcode year
-- For "January this year": EXTRACT(MONTH FROM ir.submitted_on) = 1 AND EXTRACT(YEAR ...) = EXTRACT(YEAR FROM CURRENT_DATE)
-- For "January 2026" specifically: EXTRACT(MONTH ...) = 1 AND EXTRACT(YEAR ...) = 2026
-- status != 'DRAFT' used here since listing inspections (not filtering by completed/conducted)
SELECT ir.inspection_id,
       fac.name AS facility_name,
       it.name  AS inspection_type_name,
       ir.submitted_on,
       ir.status
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE ir.active = true
  AND ir.status != 'DRAFT'
  AND EXTRACT(MONTH FROM ir.submitted_on) = 1
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
ORDER BY ir.submitted_on DESC
LIMIT 100;
""",
        'domain': 'inspection',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'which month had the most high risk observations this year',
        'sql':    """\
SELECT TO_CHAR(date_trunc('month', ir.submitted_on), 'Month YYYY') AS month,
       COUNT(*) AS high_risk_count
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN ai_answers risk_aa ON risk_aa.inspection_report_id = aa.inspection_report_id
JOIN ai_questions risk_aq ON risk_aa.element_id = risk_aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
WHERE aq.label = 'Observation'
  AND risk_aq.label = 'Risk Level' AND risk_aa.answer_text ILIKE '%High%'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY date_trunc('month', ir.submitted_on)
ORDER BY high_risk_count DESC LIMIT 1;
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'are corrective actions being closed faster or slower compared to last quarter',
        'sql':    """\
-- USE THIS EXACT SQL for quarter-over-quarter closure rate:
-- Returns TWO ROWS: one per quarter — use GROUP BY date_trunc quarter
-- VALIDATOR NOTE: GROUP BY here is VALID (trend comparison, not wrong shape)
SELECT
    CASE
        WHEN date_trunc('quarter', ica.target_close_out_date) = date_trunc('quarter', CURRENT_DATE - INTERVAL '3 months')
        THEN 'Q1 ' || EXTRACT(YEAR FROM CURRENT_DATE)::text
        ELSE 'Q2 ' || EXTRACT(YEAR FROM CURRENT_DATE)::text
    END AS quarter,
    COUNT(*) AS total_raised,
    COUNT(*) FILTER (WHERE ica.status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS closed_count,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE ica.status IN ('CLOSED','CLOSE_WITH_DEFERRED'))
        / NULLIF(COUNT(*), 0), 2
    ) AS closure_rate_pct
FROM inspection_corrective_action ica
WHERE ica.target_close_out_date >= date_trunc('year', CURRENT_DATE)
  AND ica.target_close_out_date < date_trunc('year', CURRENT_DATE) + INTERVAL '1 year'
GROUP BY 1
ORDER BY 1;
""",
        'domain': 'corrective_action',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'show the last 5 inspections by submission date',
        'sql':    """\
SELECT ir.inspection_id, fac.name AS facility_name,
       it.name AS inspection_type_name, ir.submitted_on, ir.status, ir.inspection_score
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 5;
""",
        'domain': 'inspection',
        'intent': 'filter',
        'limit':  5,
    },
    {
        'nl':     'show all overdue corrective actions',
        'sql':    """\
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.target_close_out_date,
       fac.name AS facility_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.status = 'OVERDUE'
ORDER BY ica.target_close_out_date ASC LIMIT 100;
""",
        'domain': 'corrective_action',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'which inspections were returned for modification',
        'sql':    """\
SELECT ir.inspection_id, fac.name AS facility_name,
       it.name AS inspection_type_name, ir.submitted_on
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE ir.status = 'RETURN_FOR_MODIFICATION'
ORDER BY ir.submitted_on DESC LIMIT 50;
""",
        'domain': 'inspection',
        'intent': 'filter',
        'limit':  50,
    },
    {
        'nl':     'show me inspections with a score below 70',
        'sql':    """\
SELECT ir.inspection_id, ir.inspection_score, ir.status,
       fac.name AS facility_name, ir.submitted_on
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.inspection_score < 70 AND ir.status != 'DRAFT'
ORDER BY ir.inspection_score ASC LIMIT 50;
""",
        'domain': 'inspection',
        'intent': 'filter',
        'limit':  50,
    },
    {
        'nl':     "which facilities haven't been inspected in the last 30 days",
        'sql':    """\
SELECT fac.name AS facility_name
FROM facility fac
WHERE fac.id NOT IN (
    SELECT DISTINCT ir.facility_id FROM inspection_report ir
    WHERE ir.submitted_on >= CURRENT_DATE - INTERVAL '30 days'
      AND ir.status != 'DRAFT'
)
ORDER BY fac.name LIMIT 100;
""",
        'domain': 'inspection',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'how many inspections were done on weekends',
        'sql':    """\
SELECT COUNT(*) AS weekend_inspection_count
FROM inspection_report ir
WHERE EXTRACT(ISODOW FROM ir.submitted_on) IN (6, 7)
  AND ir.status != 'DRAFT';
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show inspection scores grouped by facility',
        'sql':    """\
SELECT fac.name AS facility_name,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score,
       COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
GROUP BY fac.name ORDER BY avg_score DESC LIMIT 50;
""",
        'domain': 'inspection',
        'intent': 'filter',
        'limit':  50,
    },
    {
        'nl':     'which inspector has conducted the most inspections this year',
        'sql':    """\
-- conducted = IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION') — in-flight, not closed
-- completed = CLOSED only
-- assigned  = all non-draft reports
SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       COUNT(*) AS assigned,
       COUNT(*) FILTER (WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')) AS conducted,
       COUNT(*) FILTER (WHERE ir.status = 'CLOSED') AS completed
FROM inspection_report ir
JOIN users u ON ir.inspector_user_id = u.id
WHERE ir.active = true
  AND ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY u.id, u.first_name, u.last_name
ORDER BY conducted DESC LIMIT 1;
""",
        'domain': 'inspection',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        # Inspector performance summary — assigned, conducted, observations, high-risk.
        # CRITICAL DEFINITIONS (match the application):
        #   assigned   = COUNT(ir.id)                              — all reports linked to inspector
        #   conducted  = COUNT FILTER (WHERE status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION'))
        #   total_observations = SUM of distinct observation submissions per inspection
        #                        (one per ai_answers.submission_id group, NOT COUNT(*) on answer rows)
        #   high_risk_observations = observations where risk level answer = 'High'
        # NEVER join ai_answers twice in the main query (fan-out/cross-join).
        # Pre-aggregate observations in a CTE to avoid row multiplication.
        # NEVER join inspection_schedule — it is irrelevant to inspector performance counts.
        # NEVER use AVG(ir.inspection_score) here — that's avg SITE compliance, not inspector quality.
        'nl':     'inspector performance summary assigned conducted observations high risk inspector report breakdown inspector stats best performing inspector top inspectors by performance which inspector performs best inspector ranking inspector effectiveness who is best inspector',
        'sql':    """\
-- Inspector performance: assigned, conducted, total observations, high-risk observations
-- NEVER rank by AVG(ir.inspection_score) — that measures avg site compliance of assigned
-- facilities, not inspector quality. Tough-site inspectors will always rank last unfairly.
-- conducted = SUBMITTED only (NOT status != 'DRAFT' — that overcounts)
-- Observations pre-aggregated in CTE to prevent double-join fan-out
WITH obs AS (
    SELECT aa.inspection_report_id,
           COUNT(DISTINCT aa.submission_id)   AS total_obs,
           COUNT(DISTINCT CASE
               WHEN aq.label = 'Risk Level' AND aa.answer_text ILIKE '%High%'
               THEN aa.submission_id END)     AS high_risk_obs
    FROM ai_answers aa
    JOIN ai_questions aq ON aa.element_id = aq.element_id
    WHERE aa.module_name = 'Inspection Form'
    GROUP BY aa.inspection_report_id
)
SELECT u.first_name || ' ' || u.last_name          AS inspector_name,
       COUNT(ir.id)                                AS assigned,
       COUNT(ir.id) FILTER (WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')) AS conducted,
       COUNT(ir.id) FILTER (WHERE ir.status = 'CLOSED') AS completed,
       COALESCE(SUM(obs.total_obs), 0)             AS total_observations,
       COALESCE(SUM(obs.high_risk_obs), 0)         AS high_risk_observations
FROM users u
JOIN inspection_report ir ON ir.inspector_user_id = u.id
    AND ir.active = true
LEFT JOIN obs ON obs.inspection_report_id = ir.id
WHERE u.active = true AND ir.status != 'DRAFT'
GROUP BY u.id, u.first_name, u.last_name
ORDER BY conducted DESC NULLS LAST
LIMIT 100;
""",
        'domain': 'inspection',
        'intent': 'ranked_list',
        'limit':  100,
    },
    {
        'nl':     'show average inspection score by inspection type',
        'sql':    """\
SELECT it.name AS inspection_type_name,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score,
       COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
GROUP BY it.name ORDER BY avg_score DESC;
""",
        'domain': 'inspection',
        'intent': 'filter',
        'limit':  None,
    },
    {
        'nl':     'show corrective actions with their inspection facility and inspector',
        'sql':    """\
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.status,
       fac.name AS facility_name,
       u.first_name || ' ' || u.last_name AS inspector_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN users u ON ir.inspector_user_id = u.id
LIMIT 100;
""",
        'domain': 'corrective_action',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'which client has the most overdue corrective actions',
        'sql':    """\
SELECT cl.name AS client_name, COUNT(*) AS overdue_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN client cl ON ir.client_id = cl.id
WHERE ica.status = 'OVERDUE'
GROUP BY cl.name ORDER BY overdue_count DESC LIMIT 10;
""",
        'domain': 'corrective_action',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'show inspection scores grouped by project',
        'sql':    """\
-- project JOIN: inspection_report.project_id → project.id
-- ALWAYS GROUP BY proj.name, not ir.project_id (uuid)
SELECT proj.name AS project_name,
       COUNT(*) AS inspection_count,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score
FROM inspection_report ir
JOIN project proj ON ir.project_id = proj.id
WHERE ir.active = true AND ir.status != 'DRAFT'
  AND proj.active = true
GROUP BY proj.id, proj.name
ORDER BY avg_score DESC;
""",
        'domain': 'inspection',
        'intent': 'ranked_list',
        'limit':  None,
    },
    {
        'nl':     'show inspection count grouped by entity',
        'sql':    """-- entity JOIN: inspection_report.entity_id → entity.id (DIRECT FK)
-- entity is a standalone business/contract entity — not linked to organisation
SELECT ent.name AS entity_name,
       COUNT(*) AS inspection_count,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score
FROM inspection_report ir
JOIN entity ent ON ir.entity_id = ent.id
WHERE ir.active = true AND ir.status != 'DRAFT'
  AND ent.active = true
GROUP BY ent.id, ent.name
ORDER BY inspection_count DESC;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'which entity has the most overdue corrective actions',
        'sql':    """SELECT ent.name AS entity_name, COUNT(*) AS overdue_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN entity ent ON ir.entity_id = ent.id
WHERE ica.status = 'OVERDUE'
  AND ica.active = true AND ir.active = true AND ent.active = true
GROUP BY ent.id, ent.name
ORDER BY overdue_count DESC LIMIT 1;
""",
        'domain': 'corrective_action',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'show inspection count grouped by country',
        'sql':    """\
-- country JOIN: inspection_report.country_id → country.id
-- ALWAYS GROUP BY c.name, not ir.country_id (uuid)
SELECT c.name AS country_name,
       COUNT(*) AS inspection_count,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score
FROM inspection_report ir
JOIN country c ON ir.country_id = c.id
WHERE ir.active = true AND ir.status != 'DRAFT'
  AND c.active = true
GROUP BY c.id, c.name
ORDER BY inspection_count DESC;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show inspection count grouped by region',
        'sql':    """-- region JOIN: facility.region_id → region.id (region has country_id for hierarchy)
-- No direct region_id on inspection_report — must go via facility
SELECT reg.name AS region_name,
       COUNT(*) AS inspection_count,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
JOIN region reg ON fac.region_id = reg.id
WHERE ir.active = true AND ir.status != 'DRAFT'
  AND fac.active = true AND reg.active = true
GROUP BY reg.id, reg.name
ORDER BY inspection_count DESC;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show facilities grouped by region which region has most facilities',
        'sql':    """SELECT reg.name AS region_name, COUNT(*) AS facility_count
FROM facility fac
JOIN region reg ON fac.region_id = reg.id
WHERE fac.active = true AND reg.active = true
GROUP BY reg.id, reg.name
ORDER BY facility_count DESC;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show corrective action count by region',
        'sql':    """SELECT reg.name AS region_name, COUNT(*) AS ca_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN region reg ON fac.region_id = reg.id
WHERE ica.active = true AND ir.active = true AND fac.active = true AND reg.active = true
GROUP BY reg.id, reg.name
ORDER BY ca_count DESC;
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show inspections grouped by organisation',
        'sql':    """\
-- organisation has NO direct FK on inspection_report
-- ONLY join path: inspection_report → facility (facility_id) → organisation (organization_id)
-- facility.organization_id uses American spelling; table name 'organisation' uses British spelling
-- ALWAYS GROUP BY org.name, not the UUID column
SELECT org.name AS organisation_name,
       COUNT(*) AS inspection_count,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
JOIN organisation org ON fac.organization_id = org.id
WHERE ir.active = true AND ir.status != 'DRAFT'
  AND fac.active = true AND org.active = true
GROUP BY org.id, org.name
ORDER BY inspection_count DESC;
""",
        'domain': 'inspection',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'which project has the highest average inspection score',
        'sql':    """\
SELECT proj.name AS project_name,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score,
       COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN project proj ON ir.project_id = proj.id
WHERE ir.active = true AND ir.status != 'DRAFT'
  AND proj.active = true AND ir.inspection_score IS NOT NULL
GROUP BY proj.id, proj.name
ORDER BY avg_score DESC LIMIT 1;
""",
        'domain': 'inspection',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'show corrective action count by project',
        'sql':    """\
SELECT proj.name AS project_name, COUNT(*) AS ca_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN project proj ON ir.project_id = proj.id
WHERE ica.active = true AND ir.active = true AND proj.active = true
GROUP BY proj.id, proj.name
ORDER BY ca_count DESC;
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show capex and opex by country',
        'sql':    """\
SELECT c.name AS country_name,
       COALESCE(SUM(ica.capex), 0) AS total_capex,
       COALESCE(SUM(ica.opex),  0) AS total_opex
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN country c ON ir.country_id = c.id
WHERE ica.active = true AND ir.active = true AND c.active = true
GROUP BY c.id, c.name
ORDER BY total_capex DESC;
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'what is the total capex and opex by facility',
        'sql':    """\
SELECT fac.name AS facility_name,
       SUM(ica.capex) AS total_capex, SUM(ica.opex) AS total_opex
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
GROUP BY fac.name ORDER BY total_capex DESC NULLS LAST LIMIT 50;
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  50,
    },
    {
        'nl':     'show facilities that have been inspected more than 5 times this year',
        'sql':    """\
SELECT fac.name AS facility_name, COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY fac.name HAVING COUNT(*) > 5
ORDER BY inspection_count DESC;
""",
        'domain': 'inspection',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'which projects have not had any inspections this year',
        'sql':    """\
SELECT proj.name AS project_name
FROM project proj
WHERE proj.id NOT IN (
    SELECT DISTINCT ir.project_id FROM inspection_report ir
    WHERE ir.status != 'DRAFT'
      AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
      AND ir.project_id IS NOT NULL
)
ORDER BY proj.name;
""",
        'domain': 'inspection',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'what are the most common observation types across all inspections',
        'sql':    """\
-- Use get_answer_stats tool for this — it aggregates answer_text for a question label
-- SQL fallback:
SELECT aa.answer_text AS observation_type, COUNT(*) AS frequency
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aq.label = 'Observation Type' AND aa.answer_text IS NOT NULL
GROUP BY aa.answer_text ORDER BY frequency DESC LIMIT 20;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  20,
    },
    {
        'nl':     'count deviations raised at a facility this year observation type deviation',
        'sql':    """\
-- CORRECT label for deviation type: aq.label = 'Observation Type' (NOT 'Observation')
-- 'Observation' label = free-text field, 'Observation Type' = Deviation | Good Practice dropdown
-- NEVER: aq.label ILIKE '%Observation%' — that matches BOTH and overcounts
-- aa.answer_text ILIKE '%Deviation%' matches the enum value 'Deviation'
-- BUSINESS RULE: each inspection is Good Practice OR Deviation — count DISTINCT reports not answer rows
SELECT COUNT(DISTINCT ir.id) AS deviation_count
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE aq.label = 'Observation Type'
  AND aa.answer_text ILIKE '%Deviation%'
  AND aa.module_name = 'Inspection Form'
  AND fac.name ILIKE '%facility_name%'
  AND ir.active = true AND ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE);
""",
        'domain': 'form_answers',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'count good practice observations raised at a facility this year',
        'sql':    """\
-- CORRECT: aq.label = 'Observation Type' for the dropdown, answer_text = 'Good Practice'
-- BUSINESS RULE: an inspection is classified as EITHER good practice OR deviation (not both).
-- Count DISTINCT inspection reports — NOT COUNT(*) answer rows — to avoid double-counting.
-- COUNT(*) would overcount if a report has multiple 'Observation Type' answer rows.
SELECT COUNT(DISTINCT ir.id) AS good_practice_count
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE aq.label = 'Observation Type'
  AND aa.answer_text ILIKE '%Good Practice%'
  AND aa.module_name = 'Inspection Form'
  AND fac.name ILIKE '%facility_name%'
  AND ir.active = true AND ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE);
""",
        'domain': 'form_answers',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'good practice count by facility which facility has most good practice inspections top facilities good practice ranked',
        'sql':    """\
-- Good practice count PER FACILITY — count DISTINCT inspection reports, NOT answer rows.
-- BUSINESS RULE: each inspection is classified as Good Practice OR Deviation (mutually exclusive).
-- COUNT(*) overcounts; COUNT(DISTINCT ir.id) correctly counts one per report.
-- NEVER use COUNT(*) for good practice / deviation per-facility counts.
SELECT fac.name AS facility_name,
       COUNT(DISTINCT ir.id) AS good_practice_count
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE aq.label = 'Observation Type'
  AND aa.answer_text ILIKE '%Good Practice%'
  AND aa.module_name = 'Inspection Form'
  AND fac.active = true
  AND ir.active = true
  AND ir.status != 'DRAFT'
GROUP BY fac.id, fac.name
ORDER BY good_practice_count DESC
LIMIT 10;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'which facilities have never had a good practice observation facilities without good practice',
        'sql':    """\
-- Facility-level negation: facilities that have NEVER had a Good Practice answer across all history
-- Pattern: facility IN (has inspections) AND facility NOT IN (has ever had Good Practice)
-- NEVER use GROUP BY + HAVING COUNT = 0 — that only checks per-inspection, not facility-wide
SELECT fac.name AS facility_name
FROM facility fac
WHERE fac.active = true
  AND fac.id IN (
      -- only include facilities that have at least one non-draft inspection
      SELECT DISTINCT ir.facility_id
      FROM inspection_report ir
      WHERE ir.active = true AND ir.status != 'DRAFT'
  )
  AND fac.id NOT IN (
      -- exclude facilities that have EVER had a Good Practice observation
      SELECT DISTINCT ir.facility_id
      FROM ai_answers aa
      JOIN ai_questions aq ON aa.element_id = aq.element_id
      JOIN inspection_report ir ON aa.inspection_report_id = ir.id
      WHERE aq.label = 'Observation Type'
        AND aa.answer_text ILIKE '%Good Practice%'
        AND aa.module_name = 'Inspection Form'
        AND ir.active = true AND ir.status != 'DRAFT'
  )
ORDER BY fac.name;
""",
        'domain': 'form_answers',
        'intent': 'filter',
        'limit':  None,
    },
    {
        'nl':     'list all questions and answers from the most recent inspection form',
        'sql':    """\
-- CRITICAL: use aa subquery — ai_answers.submitted_on may differ from ir.submitted_on
-- NEVER filter module_name in the inner subquery — it may differ and return NULL
SELECT aq.label AS question, aa.answer_text AS answer, aa.module_name
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aa.inspection_report_id = (
    SELECT aa2.inspection_report_id FROM ai_answers aa2
    WHERE aa2.submitted_on IS NOT NULL
    ORDER BY aa2.submitted_on DESC LIMIT 1
)
ORDER BY aa.module_name, aq.label LIMIT 200;
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'show me the questions and answers for the most recently answered inspection',
        'sql':    """\
-- Q&A for most recently answered inspection — single SQL, no two-step lookup
-- Use aa.submitted_on subquery — NOT ir.submitted_on
SELECT aq.label AS question, aa.answer_text AS answer, aa.module_name AS form_section
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aa.inspection_report_id = (
    SELECT aa2.inspection_report_id FROM ai_answers aa2
    WHERE aa2.submitted_on IS NOT NULL
    ORDER BY aa2.submitted_on DESC LIMIT 1
)
ORDER BY aa.module_name, aq.label LIMIT 200;
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'what questions and answers are in the latest inspection',
        'sql':    """\
SELECT aq.label AS question, aa.answer_text AS answer, aa.module_name AS section
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aa.inspection_report_id = (
    SELECT aa2.inspection_report_id FROM ai_answers aa2
    WHERE aa2.submitted_on IS NOT NULL
    ORDER BY aa2.submitted_on DESC LIMIT 1
)
ORDER BY aa.module_name, aq.label LIMIT 200;
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'show high risk observations',
        'sql':    """\
-- Form answer risk — use ai_answers WHERE aq.label = 'Risk Level', NOT risk_level table
SELECT aa.inspection_id, obs.answer_text AS observation, ir.submitted_on
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN ai_answers obs ON obs.inspection_report_id = ir.id
JOIN ai_questions obs_q ON obs.element_id = obs_q.element_id
WHERE aq.label = 'Risk Level' AND aa.answer_text ILIKE '%High%'
  AND obs_q.label = 'Observation'
  AND ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 100;
""",
        'domain': 'form_answers',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'what did inspectors write in the observation field for a specific facility',
        'sql':    """\
SELECT aa.answer_text AS observation, ir.inspection_id, ir.submitted_on
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE aq.label = 'Observation'
  AND fac.name ILIKE '%facility_name%'
  AND aa.answer_text IS NOT NULL
  AND ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 100;
""",
        'domain': 'form_answers',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'what are the most common causes of corrective actions',
        'sql':    """\
SELECT ica.cause, COUNT(*) AS frequency
FROM inspection_corrective_action ica
WHERE ica.cause IS NOT NULL
GROUP BY ica.cause ORDER BY frequency DESC LIMIT 50;
""",
        'domain': 'corrective_action',
        'intent': 'ranked_list',
        'limit':  50,
    },
    {
        'nl':     'what are the repetitive observations in the last 6 months',
        'sql':    """\
SELECT aa.answer_text AS observation, COUNT(*) AS occurrences
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
WHERE aq.label = 'Observation'
  AND aa.answer_text IS NOT NULL
  AND ir.submitted_on >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY aa.answer_text HAVING COUNT(*) > 1
ORDER BY occurrences DESC LIMIT 20;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  20,
    },
    {
        'nl':     'show me questions with their scores for the last inspection by George',
        'sql':    """\
SELECT aq.label AS question, aa.answer_text AS answer, aa.score
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aa.inspection_report_id = (
    -- ir.id in subquery WHERE is CORRECT (UUID PK, not displayed)
    SELECT ir.id FROM inspection_report ir
    JOIN users u ON ir.inspector_user_id = u.id
    WHERE u.first_name || ' ' || u.last_name ILIKE '%George%'
      AND ir.status != 'DRAFT'
    ORDER BY ir.submitted_on DESC LIMIT 1
)
  AND aa.module_name = 'Inspection Form'
ORDER BY aq.label;
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'which inspector gets the highest form answer scores best scoring inspector',
        'sql':    """\
-- aa.score is meaningful ONLY for 'Impact' (range 1-27, severity signal)
-- Risk Level and Observation Type scores are form config artifacts (always 4 or 1) — ignore for scoring
-- 'Observation' free-text has no score — never filter aq.label = 'Observation' for scores
SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       ROUND(AVG(aa.score)::numeric, 2) AS avg_impact_score,
       SUM(aa.score)                    AS total_impact_score,
       COUNT(aa.id)                     AS observation_count
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN users u ON ir.inspector_user_id = u.id
WHERE aa.score IS NOT NULL
  AND aq.label = 'Impact'
  AND aa.module_name = 'Inspection Form'
  AND ir.active = true AND ir.status != 'DRAFT'
  AND u.active = true
GROUP BY u.id, u.first_name, u.last_name
ORDER BY avg_impact_score DESC LIMIT 10;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'which facility has the best form answer scores highest scored observations facility',
        'sql':    """\
-- aa.score meaningful ONLY for 'Impact' (range 1-27, severity signal)
-- Risk Level and Observation Type scores are form config artifacts — never use for ranking
SELECT fac.name AS facility_name,
       ROUND(AVG(aa.score)::numeric, 2) AS avg_impact_score,
       COUNT(aa.id)                     AS observation_count
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE aa.score IS NOT NULL
  AND aq.label = 'Impact'
  AND aa.module_name = 'Inspection Form'
  AND ir.active = true AND ir.status != 'DRAFT'
  AND fac.active = true
GROUP BY fac.id, fac.name
ORDER BY avg_impact_score DESC LIMIT 20;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  20,
    },
    {
        'nl':     'which inspector raises the most health and safety impact observations impact category HS',
        'sql':    """-- Impact is a MULTI-SELECT category field, NOT severity. answer_text contains category names:
-- 'Client Relationship', 'Health & Safety (HS)', 'Financial', 'Environmental',
-- 'Legal / Regulatory', 'Reputational', 'Operational - Asset Downtime',
-- 'Operational - Manpower/Resource Impact', 'Operational - Service Failure', 'Quality'
-- Multi-select answers are stored as comma-separated strings in a single answer_text row
-- To count by category: aa.answer_text ILIKE '%category name%'
-- aa.score = weighted points (higher = more/higher-weight categories); NOT a severity level
SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       COUNT(*) AS hs_impact_count
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN users u ON ir.inspector_user_id = u.id
WHERE aq.label = 'Impact'
  AND aa.answer_text ILIKE '%Health & Safety%'
  AND aa.module_name = 'Inspection Form'
  AND ir.active = true AND ir.status != 'DRAFT'
  AND u.active = true
GROUP BY u.id, u.first_name, u.last_name
ORDER BY hs_impact_count DESC LIMIT 10;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'what are the most common impact categories across all inspections impact breakdown',
        'sql':    """-- Impact answer_text is comma-separated multi-select — individual categories appear as entries
SELECT aa.answer_text AS impact_categories, COUNT(*) AS frequency
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aq.label = 'Impact'
  AND aa.answer_text IS NOT NULL
  AND aa.module_name = 'Inspection Form'
GROUP BY aa.answer_text
ORDER BY frequency DESC LIMIT 20;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  20,
    },
    {
        'nl':     'show observation with highest impact score impact scored answers',
        'sql':    """-- 'Impact' is the label for scored impact answers (scores 1-27)
-- 'Observation' is free-text with NO score — never use aq.label = 'Observation' for scores
SELECT aa.answer_text AS impact_answer,
       aa.score       AS impact_score,
       ir.inspection_id,
       ir.submitted_on,
       fac.name       AS facility_name,
       u.first_name || ' ' || u.last_name AS inspector_name
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN users u ON ir.inspector_user_id = u.id
WHERE aq.label = 'Impact'
  AND aa.score IS NOT NULL
  AND aa.module_name = 'Inspection Form'
  AND ir.active = true AND ir.status != 'DRAFT'
ORDER BY aa.score DESC LIMIT 10;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'show corrective actions linked to a specific observation CAs for observation corrective actions for this observation corrective actions for first observation corrective actions raised from this finding',
        'sql':    """-- OBSERVATION → CA PATH: aa.submission_id = ica.observation_id::varchar
-- Cast UUID side to varchar — NOT aa.submission_id::uuid (submission_id is varchar, observation_id is UUID)
-- This is the DIRECT link — NOT ica.inspection_id = ir.id (that returns ALL CAs in the inspection)
-- CORRECT: aa.submission_id = ica.observation_id::varchar  (398 CAs have non-NULL observation_id)
-- NOT: JOIN inspection_report ir ON ica.inspection_id = ir.id (returns ALL CAs for the whole inspection)
SELECT ica.corrective_action_id,
       ica.cause,
       ica.corrective_action,
       ica.correction,
       ica.responsible,
       ica.status,
       ica.capex,
       aa.answer_text AS linked_observation
FROM inspection_corrective_action ica
JOIN ai_answers aa ON aa.submission_id = ica.observation_id::varchar
WHERE ica.observation_id IS NOT NULL
  AND ica.active = true
  AND aa.answer_text ILIKE '%observation_text%'
ORDER BY ica.corrective_action_id;
""",
        'domain': 'corrective_action',
        'intent': 'filter',
        'limit':  None,
    },
    {
        'nl':     'which observation triggered the most corrective actions observation with most CAs',
        'sql':    """-- CAST direction: aa.submission_id = ica.observation_id::varchar
-- submission_id is varchar, observation_id is UUID — cast UUID to varchar, NOT varchar to uuid
SELECT aa.answer_text AS observation,
       COUNT(ica.id)  AS ca_count,
       ir.inspection_id
FROM inspection_corrective_action ica
JOIN ai_answers aa ON aa.submission_id = ica.observation_id::varchar
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
WHERE ica.active = true AND ica.observation_id IS NOT NULL
GROUP BY aa.id, aa.answer_text, ir.inspection_id
ORDER BY ca_count DESC LIMIT 10;
""",
        'domain': 'corrective_action',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'show observation with highest score for a specific inspector highest impact score',
        'sql':    """\
-- aa.score is meaningful ONLY for 'Impact' (range 1-27, severity signal)
-- Risk Level and Observation Type scores are form config artifacts (always 4 or 1) — never rank by them
-- 'Observation' (free-text field) has NO score — never filter aq.label = 'Observation' for scores
-- Always wrap OR name conditions in parentheses to avoid precedence bug
SELECT aq.label AS question_type,
       aa.answer_text AS answer,
       aa.score,
       ir.inspection_id,
       ir.submitted_on,
       u.first_name || ' ' || u.last_name AS inspector_name
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN users u ON ir.inspector_user_id = u.id
WHERE (u.first_name ILIKE '%inspector_name%' OR u.last_name ILIKE '%inspector_name%')
  AND aq.label = 'Impact'
  AND aa.score IS NOT NULL
  AND aa.module_name = 'Inspection Form'
  AND ir.active = true AND ir.status != 'DRAFT'
  AND u.active = true
ORDER BY aa.score DESC LIMIT 10;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'what is the total form answer score for an inspection',
        'sql':    """\
-- SUM(aa.score) = total weighted points for all answers in one inspection
-- Use ir.inspection_id (varchar) for display, ir.id (UUID) for JOIN only
SELECT ir.inspection_id,
       SUM(aa.score)  AS total_score,
       MAX(aa.max_score) AS max_possible,
       COUNT(aa.id)   AS questions_answered
FROM ai_answers aa
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
WHERE ir.inspection_id = 'INS-XXXX'
  AND aa.score IS NOT NULL
  AND aa.module_name = 'Inspection Form'
GROUP BY ir.inspection_id;
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'what is the risk level breakdown of all corrective actions',
        'sql':    """\
-- ICA risk_level_id is UUID FK — NEVER compare to string, NEVER GROUP BY UUID
-- Filter by ica.active only — do NOT reference ir.active without joining inspection_report
SELECT rl.name AS risk_level_name, COUNT(*) AS action_count
FROM inspection_corrective_action ica
JOIN risk_level rl ON ica.risk_level_id = rl.id
WHERE ica.active = true
GROUP BY rl.name ORDER BY action_count DESC;
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'how many corrective actions are high risk',
        'sql':    """\
SELECT COUNT(*) AS high_risk_count
FROM inspection_corrective_action ica
JOIN risk_level rl ON ica.risk_level_id = rl.id
WHERE rl.name = 'High';
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show all high risk corrective actions',
        'sql':    """\
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.status, rl.name AS risk_level_name,
       fac.name AS facility_name, ica.target_close_out_date
FROM inspection_corrective_action ica
JOIN risk_level rl ON ica.risk_level_id = rl.id
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE rl.name = 'High'
ORDER BY ica.target_close_out_date DESC LIMIT 100;
""",
        'domain': 'corrective_action',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'show open corrective actions with high risk grouped by facility',
        'sql':    """\
SELECT fac.name AS facility_name, COUNT(*) AS high_risk_open_count
FROM inspection_corrective_action ica
JOIN risk_level rl ON ica.risk_level_id = rl.id
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE rl.name = 'High' AND ica.status = 'OPEN'
GROUP BY fac.name ORDER BY high_risk_open_count DESC;
""",
        'domain': 'corrective_action',
        'intent': 'filter',
        'limit':  None,
    },
    {
        'nl':     'show all deferred corrective actions',
        'sql':    """\
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.status, ica.deferred_on, ica.target_close_out_date,
       fac.name AS facility_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.status = 'CLOSE_WITH_DEFERRED'
ORDER BY fac.name, ica.deferred_on DESC LIMIT 100;
""",
        'domain': 'corrective_action',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'which facility has the most recurring deferred issues',
        'sql':    """\
SELECT fac.name AS facility_name, COUNT(*) AS deferred_issue_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.status = 'CLOSE_WITH_DEFERRED'
GROUP BY fac.name ORDER BY deferred_issue_count DESC LIMIT 20;
""",
        'domain': 'corrective_action',
        'intent': 'ranked_list',
        'limit':  20,
    },
    {
        'nl':     'how many corrective actions have been carried forward more than once',
        'sql':    """\
SELECT COUNT(*) AS multi_deferred_count FROM (
    SELECT ica.cause, fac.name
    FROM inspection_corrective_action ica
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    JOIN facility fac ON ir.facility_id = fac.id
    WHERE ica.status = 'CLOSE_WITH_DEFERRED'
    GROUP BY ica.cause, fac.name HAVING COUNT(*) > 1
) sub;
""",
        'domain': 'corrective_action',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show inspections where the score dropped compared to the previous inspection at the same facility',
        'sql':    """\
WITH scored AS (
  SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on,
         ir.inspection_score,
         LAG(ir.inspection_score) OVER (PARTITION BY ir.facility_id ORDER BY ir.submitted_on) AS previous_score
  FROM inspection_report ir
  JOIN facility fac ON ir.facility_id = fac.id
  WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
)
SELECT inspection_id, facility_name, submitted_on,
       inspection_score AS current_score, previous_score,
       ROUND((previous_score - inspection_score)::numeric, 2) AS score_drop
FROM scored
WHERE previous_score IS NOT NULL AND inspection_score < previous_score
ORDER BY score_drop DESC LIMIT 50;
""",
        'domain': 'inspection',
        'intent': 'filter',
        'limit':  50,
    },
    {
        'nl':     'which facility improved the most in score compared to last quarter',
        'sql':    """\
-- NEVER use ir.current_qtr_score or ir.last_qtr_score — columns don't exist
WITH qtr AS (
  SELECT ir.facility_id, fac.name AS facility_name,
    ROUND(AVG(CASE WHEN ir.submitted_on >= date_trunc('quarter', NOW()) THEN ir.inspection_score END)::numeric, 1) AS this_q,
    ROUND(AVG(CASE WHEN ir.submitted_on >= date_trunc('quarter', NOW()-INTERVAL '3 months')
                    AND ir.submitted_on < date_trunc('quarter', NOW()) THEN ir.inspection_score END)::numeric, 1) AS last_q
  FROM inspection_report ir
  JOIN facility fac ON ir.facility_id = fac.id
  WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
  GROUP BY ir.facility_id, fac.name
  HAVING AVG(CASE WHEN ir.submitted_on >= date_trunc('quarter', NOW()) THEN ir.inspection_score END) IS NOT NULL
     AND AVG(CASE WHEN ir.submitted_on >= date_trunc('quarter', NOW()-INTERVAL '3 months')
                   AND ir.submitted_on < date_trunc('quarter', NOW()) THEN ir.inspection_score END) IS NOT NULL
)
SELECT facility_name, this_q AS this_quarter_avg, last_q AS last_quarter_avg,
       ROUND((this_q - last_q)::numeric, 1) AS improvement
FROM qtr WHERE this_q > last_q
ORDER BY improvement DESC LIMIT 1;
""",
        'domain': 'inspection',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        # NOTE: inspection_score measures SITE compliance, not inspector quality.
        # This query answers "which inspector's assigned sites score lowest on average" —
        # it does NOT identify a poor-performing inspector (they may just work tough sites).
        # Use ONLY when the user explicitly asks about average site scores per inspector.
        # For actual inspector performance → use the CTE pattern with observations/volume.
        'nl':     'which inspector has the lowest average inspection score average site score by inspector',
        'sql':    """\
-- avg_site_score = avg site compliance score of inspections assigned to this inspector
-- NOT a measure of inspector quality — tough-site inspectors will always rank last
SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_site_score,
       COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN users u ON ir.inspector_user_id = u.id
WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
GROUP BY u.id, u.first_name, u.last_name
ORDER BY avg_site_score ASC LIMIT 1;
""",
        'domain': 'inspection',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'show me the most recently answered inspection',
        'sql':    """\
-- CRITICAL: use ai_answers subquery NOT inspection_report ORDER BY submitted_on
-- ai_answers is a complete snapshot — anchor on aa.submitted_on for accurate form order.
-- NEVER filter module_name in the inner subquery — if the name differs it returns NULL.
-- Always use the aa subquery to get the inspection with the latest form submission:
SELECT ir.inspection_id, fac.name AS facility_name,
       it.name AS inspection_type_name,
       u.first_name || ' ' || u.last_name AS inspector_name,
       ir.submitted_on, ir.status, ir.inspection_score
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
JOIN users u ON ir.inspector_user_id = u.id
WHERE ir.id = (
    SELECT aa.inspection_report_id FROM ai_answers aa
        ORDER BY aa.submitted_on DESC LIMIT 1
);
-- NEVER: ORDER BY ir.submitted_on DESC LIMIT 1 — newest ir may have no form data
-- NEVER: WHERE aa.module_name = 'Inspection Form' in inner subquery — causes empty result if name differs
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'latest inspection with answers',
        'sql':    """\
-- Use ai_answers to find most recent inspection WITH form data (not inspection_report):
-- NEVER filter module_name in the inner subquery — causes empty result if name differs
SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on, ir.status
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.id = (
    SELECT aa.inspection_report_id FROM ai_answers aa
        ORDER BY aa.submitted_on DESC LIMIT 1
);
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'most recent inspection that has form answers',
        'sql':    """\
-- NEVER add module_name filter here — it restricts results and may return NULL
SELECT aa.inspection_report_id
FROM ai_answers aa
ORDER BY aa.submitted_on DESC LIMIT 1;
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'show breakdown of corrective action status by responsible party',
        'sql':    """\
SELECT ica.responsible AS responsible_party,
       COUNT(*) FILTER (WHERE ica.status = 'OPEN') AS open_count,
       COUNT(*) FILTER (WHERE ica.status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS closed_count,
       COUNT(*) FILTER (WHERE ica.status = 'OVERDUE') AS overdue_count
FROM inspection_corrective_action ica
GROUP BY ica.responsible ORDER BY ica.responsible;
""",
        'domain': 'corrective_action',
        'intent': 'filter',
        'limit':  None,
    },
    {
        'nl':     "which inspectors haven't submitted any inspections in the last 30 days inspectors with no recent activity inactive inspectors new inspectors who haven't inspected yet inspectors never submitted",
        'sql':    """\
-- CORRECT: start from users JOIN role, LEFT JOIN inspection_report
-- This includes new inspectors who have NEVER submitted (last_inspection_date IS NULL)
-- WRONG pattern: NOT IN (SELECT inspector_user_id FROM inspection_report WHERE ...)
--   → silently excludes brand-new inspectors with zero rows in inspection_report
SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       MAX(ir.submitted_on)               AS last_inspection_date
FROM users u
JOIN user_roles ur ON ur.user_id = u.id
JOIN role r        ON r.id = ur.role_id
LEFT JOIN inspection_report ir
       ON ir.inspector_user_id = u.id
      AND ir.status != 'DRAFT'
      AND ir.active = true
WHERE r.name ILIKE '%inspector%'
  AND u.active = true
GROUP BY u.id, u.first_name, u.last_name
HAVING MAX(ir.submitted_on) < CURRENT_DATE - INTERVAL '30 days'
    OR MAX(ir.submitted_on) IS NULL
ORDER BY last_inspection_date ASC NULLS FIRST
LIMIT 100;
""",
        'domain': 'inspection',
        'intent': 'filter',
        'limit':  None,
    },
    {
        'nl':     'how many inspections are overdue this quarter',
        'sql':    """\
SELECT COUNT(*) AS overdue_count
FROM inspection_schedule
WHERE status = 'OVERDUE'
  AND due_date >= date_trunc('quarter', NOW())
  AND due_date < date_trunc('quarter', NOW()) + INTERVAL '3 months';
""",
        'domain': 'schedule',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show me the inspection schedule for the next 30 days',
        'sql':    """\
-- inspection_schedule columns: id, status, due_date, scheduled_date,
--   facility_id, inspector_id, inspection_cycle_id, portfolio_details_id, inspection_report_id
-- NEVER: inspection_schedule.inspection_id (does not exist — use inspection_report_id FK)
SELECT isched.due_date, isched.status AS schedule_status,
       fac.name AS facility_name,
       u.first_name || ' ' || u.last_name AS inspector_name
FROM inspection_schedule isched
JOIN facility fac ON isched.facility_id = fac.id
LEFT JOIN users u ON isched.inspector_id = u.id
WHERE isched.due_date >= CURRENT_DATE
  AND isched.due_date < CURRENT_DATE + INTERVAL '30 days'

ORDER BY isched.due_date ASC LIMIT 100;
""",
        'domain': 'schedule',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'how many pending or scheduled inspections this month',
        'sql':    """\
SELECT status, COUNT(*) AS count
FROM inspection_schedule
WHERE status IN ('PENDING', 'ONGOING')
  AND due_date >= date_trunc('month', NOW())
  AND due_date < date_trunc('month', NOW()) + INTERVAL '1 month'
GROUP BY status;
""",
        'domain': 'schedule',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'what is the frequency breakdown of inspections in a cycle',
        'sql':    """\
-- EXACT column names (wrong names caused errors):
--   inspection_schedule.portfolio_details_id → inspector_portfolio_details.id  (NOT portfolio_id)
--   inspector_portfolio_details.frequency_definition_id → frequency_definition.id
--   NEVER: ipd.cycle_id (does not exist)
SELECT fd.name AS frequency_name, fd.repeat_count, fd.repeat_interval, fd.repeat_unit,
       COUNT(*) AS schedule_count
FROM inspection_schedule isched
JOIN inspector_portfolio_details ipd ON isched.portfolio_details_id = ipd.id
JOIN frequency_definition fd ON ipd.frequency_definition_id = fd.id
GROUP BY fd.name, fd.repeat_count, fd.repeat_interval, fd.repeat_unit
ORDER BY schedule_count DESC;
""",
        'domain': 'schedule',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'show the inspection schedule for a specific facility',
        'sql':    """\
SELECT isched.due_date, isched.status AS schedule_status,
       fd.name AS frequency_name,
       ir.inspection_id, ir.submitted_on
FROM inspection_schedule isched
JOIN inspector_portfolio_details ipd ON isched.portfolio_details_id = ipd.id
JOIN frequency_definition fd ON ipd.frequency_definition_id = fd.id
JOIN facility fac ON isched.facility_id = fac.id
LEFT JOIN inspection_report ir ON isched.inspection_report_id = ir.id
WHERE fac.name ILIKE '%facility_name%'
ORDER BY isched.due_date DESC LIMIT 50;
""",
        'domain': 'schedule',
        'intent': 'filter',
        'limit':  50,
    },
    {
        'nl':     'what facilities are assigned to each inspection frequency type',
        'sql':    """\
-- CRITICAL join chain: frequency_definition → inspector_portfolio_details → facility
-- inspector_portfolio_details has: id, portfolio_id, facility_id, frequency_definition_id
-- DO NOT join via inspection_schedule — go directly through ipd
-- NEVER use ipd.cycle_id (does not exist)
SELECT fd.name AS frequency_name,
       fac.name AS facility_name,
       COUNT(*) AS assignment_count
FROM frequency_definition fd
JOIN inspector_portfolio_details ipd ON ipd.frequency_definition_id = fd.id
JOIN facility fac ON ipd.facility_id = fac.id
WHERE ipd.active = true AND fac.active = true AND fd.active = true
GROUP BY fd.name, fac.name
ORDER BY fd.name, fac.name;
""",
        'domain': 'schedule',
        'intent': 'filter',
        'limit':  None,
    },
    {
        'nl':     'what is the most common inspection frequency in the portfolio',
        'sql':    """\
-- inspector_portfolio_details.frequency_definition_id → frequency_definition.id
-- Count assignments per frequency — most common = highest COUNT
SELECT fd.name AS frequency_name, COUNT(*) AS assignment_count
FROM inspector_portfolio_details ipd
JOIN frequency_definition fd ON ipd.frequency_definition_id = fd.id
WHERE ipd.active = true AND fd.active = true
GROUP BY fd.id, fd.name
ORDER BY assignment_count DESC LIMIT 1;
""",
        'domain': 'schedule',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'which communication group does a facility belong to show communication group for facility',
        'sql':    """\
SELECT icg.name AS communication_group, fac.name AS facility_name
FROM inspection_communication_group icg
JOIN inspection_communication_group_facilities icgf ON icgf.group_id = icg.id
JOIN facility fac ON fac.id = icgf.facility_id
WHERE icg.active = true AND fac.active = true
ORDER BY icg.name, fac.name;
""",
        'domain': 'organisation',
        'intent': 'list',
        'limit':  None,
    },
    {
        'nl':     'show all facilities in a communication group list facilities for communication group',
        'sql':    """\
SELECT fac.name AS facility_name, icg.name AS communication_group
FROM inspection_communication_group icg
JOIN inspection_communication_group_facilities icgf ON icgf.group_id = icg.id
JOIN facility fac ON fac.id = icgf.facility_id
WHERE icg.active = true AND fac.active = true
  AND icg.name ILIKE '%group_name%'
ORDER BY fac.name;
""",
        'domain': 'organisation',
        'intent': 'list',
        'limit':  None,
    },
    {
        'nl':     'show emails for a communication group who gets notified for a group',
        'sql':    """\
SELECT icg.name AS communication_group, ige.email
FROM inspection_communication_group icg
JOIN inspection_group_emails ige ON ige.group_id = icg.id
WHERE icg.active = true
  AND icg.name ILIKE '%group_name%'
ORDER BY ige.email;
""",
        'domain': 'organisation',
        'intent': 'list',
        'limit':  None,
    },
    {
        'nl':     'which organisations are in a communication group show organisations for communication group',
        'sql':    """\
SELECT icg.name AS communication_group, org.name AS organisation_name
FROM inspection_communication_group icg
JOIN inspection_communication_group_organisations icgo ON icgo.group_id = icg.id
JOIN organisation org ON org.id = icgo.organisation_id
WHERE icg.active = true AND org.active = true
ORDER BY icg.name, org.name;
""",
        'domain': 'organisation',
        'intent': 'list',
        'limit':  None,
    },
    {
        'nl':     'which countries does an organisation operate in organisation countries',
        'sql':    """\
SELECT org.name AS organisation_name, c.name AS country_name
FROM organisation org
JOIN organisation_countries oc ON oc.organisation_id = org.id
JOIN country c ON c.id = oc.country_id
WHERE org.active = true AND c.active = true
ORDER BY org.name, c.name;
""",
        'domain': 'organisation',
        'intent': 'list',
        'limit':  None,
    },
    {
        'nl':     'list all communication groups show all groups',
        'sql':    """\
SELECT icg.name AS communication_group,
       COUNT(DISTINCT icgf.facility_id)    AS facility_count,
       COUNT(DISTINCT icgo.organisation_id) AS organisation_count,
       COUNT(DISTINCT ige.email)           AS email_count
FROM inspection_communication_group icg
LEFT JOIN inspection_communication_group_facilities icgf
       ON icgf.group_id = icg.id
LEFT JOIN inspection_communication_group_organisations icgo
       ON icgo.group_id = icg.id
LEFT JOIN inspection_group_emails ige
       ON ige.group_id = icg.id
WHERE icg.active = true
GROUP BY icg.id, icg.name
ORDER BY icg.name;
""",
        'domain': 'organisation',
        'intent': 'list',
        'limit':  None,
    },
    {
        'nl':     'which facilities have high risk observations but no corrective actions raised',
        'sql':    """\
-- Cross-domain: ai_answers (risk=High) LEFT JOIN inspection_corrective_action IS NULL
-- Step 1: find inspection_report_ids with high risk observations via ai_answers
-- Step 2: LEFT JOIN ica, keep only those with no ica rows
SELECT DISTINCT fac.name AS facility_name,
       ir.inspection_id,
       ir.submitted_on
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id AND ica.active = true
WHERE aq.label = 'Risk Level'
  AND aa.answer_text ILIKE '%High%'
  AND aa.module_name = 'Inspection Form'
  AND ir.active = true AND ir.status != 'DRAFT'
  AND ica.id IS NULL
ORDER BY ir.submitted_on DESC LIMIT 50;
""",
        'domain': 'form_answers',
        'intent': 'filter',
        'limit':  50,
    },
    {
        'nl':     'how many inspectors are assigned portfolios',
        'sql':    """\
-- USE THIS EXACT SQL for counting inspectors — simple COUNT(DISTINCT), NO GROUP BY:
-- inspector_portfolio: id, user_id (FK→users.id), status, cycle_id
-- inspector_portfolio_details: id, portfolio_id (FK→inspector_portfolio.id), facility_id, frequency_definition_id
-- NEVER add GROUP BY for a simple count query
SELECT COUNT(DISTINCT ip.user_id) AS inspector_count
FROM inspector_portfolio ip
WHERE ip.status != 'DRAFT';
""",
        'domain': 'schedule',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'which inspector has the most facilities in their portfolio',
        'sql':    """\
-- USE THIS EXACT SQL for inspector facility count — GROUP BY user id (not inspector_id):
SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       COUNT(DISTINCT ipd.facility_id) AS facility_count
FROM inspector_portfolio ip
JOIN inspector_portfolio_details ipd ON ipd.portfolio_id = ip.id
JOIN users u ON ip.user_id = u.id
WHERE ip.status != 'DRAFT'
GROUP BY u.id, u.first_name, u.last_name
ORDER BY facility_count DESC LIMIT 10;
""",
        'domain': 'schedule',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'list inspector portfolio assignments with facility and frequency',
        'sql':    """\
SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       fac.name AS facility_name,
       fd.name AS frequency_name,
       it.name AS inspection_type_name
FROM inspector_portfolio ip
JOIN inspector_portfolio_details ipd ON ipd.portfolio_id = ip.id
JOIN users u ON ip.user_id = u.id
JOIN facility fac ON ipd.facility_id = fac.id
JOIN frequency_definition fd ON ipd.frequency_definition_id = fd.id
LEFT JOIN inspection_type it ON ipd.inspection_type_id = it.id
WHERE ip.status != 'DRAFT'
ORDER BY u.last_name, fac.name LIMIT 100;
""",
        'domain': 'schedule',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'most recent inspection with form data available',
        'sql':    """\
-- Must use ai_answers subquery — inspection_report.submitted_on may point to
-- ai_answers.submitted_on timestamps may differ from ir.submitted_on.
SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on, ir.status
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.id = (
    SELECT aa.inspection_report_id FROM ai_answers aa
    WHERE aa.module_name = 'Inspection Form'
    ORDER BY aa.submitted_on DESC LIMIT 1
);
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'which inspections are not yet indexed in form answers',
        'sql':    """\
-- Inspections anchored via ai_answers subquery for accurate form submission order:
SELECT ir.inspection_id, ir.status, ir.submitted_on,
       fac.name AS facility_name
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.id NOT IN (
    SELECT DISTINCT aa.inspection_report_id FROM ai_answers aa
    WHERE aa.inspection_report_id IS NOT NULL
)
  AND ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 50;
""",
        'domain': 'form_answers',
        'intent': 'filter',
        'limit':  50,
    },
    {
        'nl':     'what is the frequency of a specific inspection cycle',
        'sql':    """\
-- USE THIS EXACT SQL for cycle frequency lookup — ONLY for inspection_cycle queries, not portfolio counts:
-- CYCLE FREQUENCY PATH: ic.id -> ip(ip.cycle_id=ic.id) -> ipd(ipd.portfolio_id=ip.id) -> fd
-- NEVER: ipd.cycle_id  |  completed_cycle = status='COMPLETED'
-- NOTE: use DISTINCT subquery shape, not GROUP BY, for this lookup
-- SCOPE: this is for ONE SPECIFIC cycle (by UUID). For "list all cycles" /
-- "what cycles are available" use the frequency_definition query instead
-- (no inspection_cycle join needed at all for that question).
SELECT DISTINCT fd.name AS frequency_name,
       fd.repeat_count, fd.repeat_interval, fd.repeat_unit
FROM inspection_cycle ic
JOIN inspector_portfolio ip          ON ip.cycle_id = ic.id
JOIN inspector_portfolio_details ipd ON ipd.portfolio_id = ip.id
JOIN frequency_definition fd         ON fd.id = ipd.frequency_definition_id
WHERE ic.id = '<CYCLE_UUID>'
  AND ic.status = 'COMPLETED';

""",
        'domain': 'schedule',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'frequency of the most recently completed inspection cycle',
        'sql':    """\
-- USE THIS EXACT SQL for most-recent-cycle frequency — ONLY for cycle queries:
-- completed_cycle = status = 'COMPLETED' (not CLOSED, not status != 'DRAFT')
-- Join: ic -> ip(ip.cycle_id=ic.id) -> ipd(ipd.portfolio_id=ip.id) -> fd
-- SCOPE: this is for the SINGLE MOST RECENT cycle — the subquery
-- "WHERE ic.id = (SELECT id FROM inspection_cycle WHERE status='COMPLETED'
-- ORDER BY end_date DESC LIMIT 1)" is REQUIRED and selects exactly ONE cycle.
-- DROPPING this subquery turns the query into "frequencies for ALL completed
-- cycles" — a DIFFERENT question. If the user asked "list all cycles" /
-- "what cycles are available", do NOT use this shape at all — use the
-- frequency_definition query (no inspection_cycle join) instead.
SELECT DISTINCT fd.name AS frequency_name,
       fd.repeat_count, fd.repeat_interval, fd.repeat_unit
FROM inspection_cycle ic
JOIN inspector_portfolio ip          ON ip.cycle_id = ic.id
JOIN inspector_portfolio_details ipd ON ipd.portfolio_id = ip.id
JOIN frequency_definition fd         ON fd.id = ipd.frequency_definition_id
WHERE ic.id = (
    SELECT id FROM inspection_cycle WHERE status = 'COMPLETED'
    ORDER BY end_date DESC LIMIT 1
);

""",
        'domain': 'schedule',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'most recently completed inspection cycle',
        'sql':    """\
-- completed_cycle: status = 'COMPLETED' only
SELECT ic.id AS inspection_cycle_id,
       TO_CHAR(ic.start_date, 'Month YYYY') AS month,
       ic.status, ic.start_date, ic.end_date, ic.modified_on
FROM inspection_cycle ic
WHERE ic.status = 'COMPLETED'
ORDER BY ic.end_date DESC LIMIT 1;

""",
        'domain': 'schedule',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'top 5 FM managers by capex generated leaderboard',
        'sql':    """\
-- USE THIS EXACT SQL for CAPEX GENERATOR leaderboard (top 5):
-- CAPEX Generator = FM managers ranked by total capex raised (any status)
-- FM Manager = inspectee_user_id (NOT inspector_user_id)
-- CAPEX Generated       = SUM(capex) WHERE capex > 0
-- CAPEX Awaiting Approval = capex > 0 AND capex_status IS NULL (null = not yet actioned)
-- capex_status values: NULL=awaiting | ACCEPTED=approved | REJECTED | DEFERRED
SELECT u.first_name || ' ' || u.last_name AS fm_manager,
       SUM(ica.capex) AS capex_generated,
       SUM(CASE WHEN ica.capex_status IS NULL THEN ica.capex ELSE 0 END) AS capex_awaiting_approval,
       COUNT(DISTINCT ica.corrective_action_id) AS action_count
FROM inspection_report ir
JOIN users u ON ir.inspectee_user_id = u.id
JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id
WHERE ir.active = true AND ica.active = true AND u.active = true
  AND ica.capex > 0
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_generated DESC
LIMIT 5;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  5,
    },
    {
        'nl':     'top 5 capex converters fm managers by capex converted leaderboard',
        'sql':    """\
-- CAPEX Converted = SUM(po.po_amount) from po_details — actual PO raised by client
-- NEVER use SUM(ica.capex) WHERE capex_status='ACCEPTED' — that is WRONG
-- NEVER direct LEFT JOIN po_details (inflates SUM(ica.capex) via row multiplication)
-- ALWAYS pre-aggregate po_details in a CTE first
-- FM Manager = inspectee_user_id (NOT inspector_user_id)
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id,
           SUM(po_amount)         AS total_po
    FROM po_details
    WHERE active = true
    GROUP BY mitigative_action_uuid
)
SELECT
    u.first_name || ' ' || u.last_name  AS fm_manager,
    COUNT(DISTINCT ica.id)              AS ca_count,
    COALESCE(SUM(ica.capex), 0)         AS capex_generated,
    COALESCE(SUM(p.total_po), 0)        AS capex_converted,
    ROUND(
        COALESCE(SUM(p.total_po), 0) * 100.0
        / NULLIF(SUM(ica.capex), 0), 1
    )                                   AS conversion_pct
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u              ON ir.inspectee_user_id = u.id
LEFT JOIN po_by_ica p     ON p.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND u.active = true
  AND ica.capex > 0
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_converted DESC
LIMIT 5;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  5,
    },
    {
        'nl':     'which corrective actions are awaiting capex approval',
        'sql':    """\
-- capex_status IS NULL = awaiting approval (not yet actioned by approver)
-- capex_status = 'ACCEPTED' = approved/converted
SELECT ica.corrective_action_id,
       ica.capex,
       ica.capex_status,
       fac.name AS facility_name,
       u.first_name || ' ' || u.last_name AS fm_manager
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN users u ON ir.inspectee_user_id = u.id
WHERE ir.active = true AND ica.active = true AND fac.active = true AND u.active = true
  AND ica.capex > 0
  AND ica.capex_status IS NULL
ORDER BY ica.capex DESC
LIMIT 50;
""",
        'domain': 'financial',
        'intent': 'filter',
        'limit':  50,
    },
    {
        'nl':     'which corrective action has the highest cost capex opex',
        'sql':    """\
-- USE THIS EXACT SQL for highest cost corrective action:
-- capex and opex are NUMERIC columns (decimal amounts)
-- expenditure is a STRING ENUM: NO_EXPENDITURE | CAPEX | OPEX | CAPEX_AND_OPEX
-- NEVER cast expenditure to numeric — it is not an amount
-- Total cost = COALESCE(capex,0) + COALESCE(opex,0)
SELECT ica.corrective_action_id,
       ica.cause,
       ica.corrective_action,
       ica.capex,
       ica.opex,
       ica.expenditure,
       (COALESCE(ica.capex, 0) + COALESCE(ica.opex, 0)) AS total_cost,
       fac.name AS facility_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.active = true AND ica.active = true AND fac.active = true
  AND (ica.capex > 0 OR ica.opex > 0)
ORDER BY total_cost DESC
LIMIT 1;
""",
        'domain': 'financial',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'capex breakdown for each corrective action at facility PO PR per corrective action conversion percentage per CA generated converted by corrective action',
        'sql':    """\
-- Per-corrective-action capex/PO/PR breakdown for a facility
-- CRITICAL: pre-aggregate po_details and pr_details in SEPARATE CTEs before joining to ica.
-- NEVER LEFT JOIN po_details AND pr_details directly into a GROUP BY query —
-- if a CA has >1 PO row or >1 PR row, the cross-join multiplies rows and
-- corrupts every SUM() (confirmed bug: 2 POs + 1 PR returned half the real capex).
WITH po_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po, COUNT(*) AS po_count
    FROM po_details WHERE active = true GROUP BY 1
),
pr_agg AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr, COUNT(*) AS pr_count
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    ica.corrective_action_id,
    ica.cause,
    ica.corrective_action,
    ica.responsible,
    ica.status,
    COALESCE(po.total_po, 0)                                              AS capex_generated,
    COALESCE(pr.total_pr, 0)                                              AS capex_converted,
    ROUND(COALESCE(pr.total_pr, 0) * 100.0 / NULLIF(po.total_po, 0), 2)   AS converted_percentage,
    COALESCE(po.po_count, 0)                                              AS po_count,
    COALESCE(pr.pr_count, 0)                                              AS pr_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN po_agg po ON po.ica_id = ica.id
LEFT JOIN pr_agg pr ON pr.ica_id = ica.id
WHERE fac.name ILIKE '%[FACILITY]%'
  AND ica.active = true AND ir.active = true AND ir.status != 'DRAFT'
ORDER BY ica.corrective_action_id;
""",
        'domain': 'financial',
        'intent': 'per_row_breakdown',
        'limit':  None,
    },
    {
        'nl':     'total capex generated converted lost and opex incurred',
        'sql':    """\
-- USE THIS EXACT SQL — ONE ROW scalar summary, NO GROUP BY, NO LEFT JOIN:
-- CAPEX Generated = SUM(ica.capex)        — amount quoted on each ICA
-- CAPEX Converted = SUM(po.po_amount)     — actual PO raised by client (po_details)
-- CAPEX Lost      = Generated - Converted — arithmetic gap, NOT capex_status='REJECTED'
-- OPEX Incurred   = SUM(pr.pr_amount)     — actual PR raised internally (pr_details)
SELECT
    (SELECT COALESCE(SUM(capex), 0)    FROM inspection_corrective_action
     WHERE active = true AND capex > 0)                                    AS capex_generated,
    (SELECT COALESCE(SUM(po_amount), 0) FROM po_details
     WHERE active = true)                                                   AS capex_converted,
    COALESCE(
        (SELECT SUM(capex) FROM inspection_corrective_action WHERE active = true AND capex > 0), 0
    ) - COALESCE(
        (SELECT SUM(po_amount) FROM po_details WHERE active = true), 0
    )                                                                       AS capex_lost,
    (SELECT COALESCE(SUM(pr_amount), 0) FROM pr_details
     WHERE active = true)                                                   AS opex_incurred;
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'top inspectors by capex generated converted',
        'sql':    """\
-- CAPEX Generated = SUM(ica.capex) | CAPEX Converted = SUM(po.po_amount) from po_details
-- Inspector = inspector_user_id (NOT inspectee_user_id — inspectee is the FM Manager)
-- CAPEX Lost = Generated - Converted (arithmetic gap, NOT capex_status='REJECTED')
-- Pre-aggregate po_details in CTE to avoid row multiplication (one ICA → many POs)
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true
    GROUP BY mitigative_action_uuid
)
SELECT
    u.first_name || ' ' || u.last_name      AS inspector_name,
    COUNT(DISTINCT ica.id)                  AS ca_count,
    COALESCE(SUM(ica.capex), 0)             AS capex_generated,
    COALESCE(SUM(p.total_po), 0)            AS capex_converted,
    COALESCE(SUM(ica.capex), 0)
        - COALESCE(SUM(p.total_po), 0)      AS capex_lost,
    ROUND(
        COALESCE(SUM(p.total_po), 0) * 100.0
        / NULLIF(SUM(ica.capex), 0), 1
    )                                        AS conversion_pct
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspector_user_id = u.id
LEFT JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND u.active = true
  AND ica.capex > 0
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_generated DESC
LIMIT 10;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'opex incurred by facility from pr details',
        'sql':    """\
-- OPEX Incurred = SUM(pr.pr_amount) from pr_details — actual internal spend
-- pr_details JOIN: ON pr.mitigative_action_uuid = ica.id
SELECT 
    fac.name AS facility_name,
    COUNT(DISTINCT ica.id) AS ca_count,
    SUM(ica.opex) AS opex_estimated,
    COALESCE(SUM(pr.pr_amount), 0) AS opex_incurred
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN pr_details pr ON pr.mitigative_action_uuid = ica.id AND pr.active = true
WHERE ir.active = true AND ica.active = true AND fac.active = true
  AND (ica.opex > 0 OR pr.id IS NOT NULL)
GROUP BY fac.id, fac.name
ORDER BY opex_incurred DESC
LIMIT 20;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  20,
    },
    {
        'nl':     'list corrective actions with rejected capex status workflow',
        'sql':    """\
-- capex_status = 'REJECTED' means the client rejected the ICA for approval (workflow status)
-- This lists ICAs by their workflow rejection — it is NOT the same as CAPEX Lost
-- CAPEX Lost (financial metric) = SUM(ica.capex) - SUM(po.po_amount)
-- Use this query only when user asks 'which actions were rejected' or 'rejected status'
SELECT
    ica.corrective_action_id,
    ica.capex                               AS capex_quoted,
    COALESCE(SUM(po.po_amount), 0)          AS po_amount_raised,
    ica.capex - COALESCE(SUM(po.po_amount), 0) AS amount_lost,
    fac.name                                AS facility_name,
    u.first_name || ' ' || u.last_name      AS fm_manager,
    ica.modified_on::date                   AS rejected_on
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN users u ON ir.inspectee_user_id = u.id
LEFT JOIN po_details po ON po.mitigative_action_uuid = ica.id AND po.active = true
WHERE ir.active = true AND ica.active = true AND fac.active = true AND u.active = true
  AND ica.capex_status = 'REJECTED'
GROUP BY ica.id, ica.corrective_action_id, ica.capex,
         fac.name, u.first_name, u.last_name, ica.modified_on
ORDER BY ica.capex DESC
LIMIT 50;
""",
        'domain': 'financial',
        'intent': 'filter',
        'limit':  50,
    },
    {
        'nl':     'total opex incurred from purchase requisitions pr details',
        'sql':    """\
-- USE THIS EXACT SQL — total OPEX incurred via actual PR spend:
-- IMPORTANT: ica.opex is the ESTIMATE, pr.pr_amount is the ACTUAL spend
-- Always use pr_details.pr_amount for 'opex incurred' questions
SELECT 
    COALESCE(SUM(pr.pr_amount), 0) AS total_opex_incurred,
    COUNT(DISTINCT pr.id)           AS pr_count
FROM pr_details pr
JOIN inspection_corrective_action ica ON ica.id = pr.mitigative_action_uuid;
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'total purchase order amount raised this year from po details',
        'sql':    """\
-- USE THIS EXACT SQL — returns ONE ROW with total PO amount:
-- DO NOT list individual rows — return SUM as a scalar result
-- po_details.po_date is the date the PO was raised
SELECT 
    SUM(po.po_amount)    AS total_po_amount,
    COUNT(DISTINCT po.id) AS po_count
FROM po_details po
WHERE EXTRACT(YEAR FROM po.po_date) = EXTRACT(YEAR FROM CURRENT_DATE);
""",
        'domain': 'financial',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'total purchase order amount po amount raised this year',
        'sql':    """\
-- USE THIS EXACT SQL — returns ONE ROW total PO amount for this year:
-- po_details table contains purchase orders (CAPEX payments from client)
-- DO NOT list individual rows — SUM into a single scalar result
SELECT 
    SUM(po.po_amount)     AS total_po_amount,
    COUNT(DISTINCT po.id) AS po_count
FROM po_details po
WHERE EXTRACT(YEAR FROM po.po_date) = EXTRACT(YEAR FROM CURRENT_DATE);
""",
        'domain': 'financial',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'show high risk corrective actions from a specific cycle scoped to cycle',
        'sql':    """\
-- USE THIS EXACT SQL for high risk CAs scoped to a cycle:
-- When active context has a cycle_id, ALWAYS add WHERE ir.cycle_id = '<UUID>'
-- Replace <CYCLE_UUID> with the actual cycle UUID from conversation context
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.status, rl.name AS risk_level_name,
       fac.name AS facility_name, ica.target_close_out_date
FROM inspection_corrective_action ica
JOIN risk_level rl ON ica.risk_level_id = rl.id
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE rl.name = 'High'
  AND ir.cycle_id = '<CYCLE_UUID>'
ORDER BY ica.target_close_out_date DESC LIMIT 100;
""",
        'domain': 'corrective_action',
        'intent': 'filter',
        'limit':  100,
    },
    {
        'nl':     'show all inspectors in a cycle facilities in a specific cycle',
        'sql':    """\
-- USE THIS EXACT SQL for inspectors/facilities within a cycle:
-- NEVER use ir.facility_name — it does not exist on inspection_report
-- Always JOIN facility fac ON ir.facility_id = fac.id to get fac.name
SELECT DISTINCT u.first_name || ' ' || u.last_name AS inspector_name,
       fac.name AS facility_name,
       ir.inspection_id, ir.submitted_on, ir.status
FROM inspection_report ir
JOIN users u ON ir.inspector_user_id = u.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.cycle_id = '<CYCLE_UUID>'
  AND ir.status != 'DRAFT'
ORDER BY fac.name, ir.submitted_on DESC
LIMIT 50;
""",
        'domain': 'inspection',
        'intent': 'filter',
        'limit':  50,
    },
    {
        'nl':     'give me a breakdown of this quarter quarterly summary',
        'sql':    """\
-- USE THIS EXACT SQL for quarterly summary — single aggregated row:
SELECT
    EXTRACT(YEAR FROM CURRENT_DATE)::int          AS year,
    EXTRACT(QUARTER FROM CURRENT_DATE)::int        AS quarter,
    COUNT(*)                                       AS inspection_count,
    ROUND(AVG(ir.inspection_score)::numeric, 2)    AS avg_score,
    COUNT(*) FILTER (WHERE ir.status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS closed_count,
    COUNT(*) FILTER (WHERE ir.status = 'SUBMITTED')                       AS submitted_count,
    COUNT(*) FILTER (WHERE ir.status = 'UNDER_REVIEW')                    AS under_review_count,
    COUNT(*) FILTER (WHERE ir.status = 'RETURNED')                        AS returned_count
FROM inspection_report ir
WHERE ir.status != 'DRAFT'
  AND date_trunc('quarter', ir.submitted_on) = date_trunc('quarter', CURRENT_DATE);
""",
        'domain': 'inspection',
        'intent': 'trend',
        'limit':  None,
    },
    {
        # CAPEX Lost by grouping (region, facility, country, entity).
        # Formula: SUM(ica.capex) - SUM(po.po_amount) — NOT po minus pr (that is profit).
        # NEVER: SUM(po.total_po) - SUM(pr.total_pr) — that computes income minus cost (P&L).
        'nl':     'capex lost by region capex lost by facility capex lost by country capex lost breakdown',
        'sql':    """\
-- CAPEX Lost = SUM(ica.capex) - SUM(po.po_amount) — arithmetic gap
-- NEVER: SUM(po) - SUM(pr) — that is profit/loss, not capex lost
-- Pre-aggregate po_details to avoid row multiplication
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
)
SELECT
    reg.name                                        AS region_name,
    COUNT(DISTINCT ica.id)                          AS ca_count,
    COALESCE(SUM(ica.capex), 0)                     AS capex_generated,
    COALESCE(SUM(p.total_po), 0)                    AS capex_converted,
    COALESCE(SUM(ica.capex), 0)
        - COALESCE(SUM(p.total_po), 0)              AS capex_lost
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN region reg ON fac.region_id = reg.id
LEFT JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ica.active = true AND ir.active = true AND fac.active = true AND reg.active = true
  AND ica.capex > 0
GROUP BY reg.id, reg.name
ORDER BY capex_lost DESC;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  None,
    },
    {
        'nl':     'capex generated converted lost by fm manager facility manager',
        'sql':    """\
-- CAPEX Generated = SUM(ica.capex)    — amount quoted on ICA
-- CAPEX Converted = SUM(po.po_amount) — actual PO raised via po_details
-- CAPEX Lost      = Generated - Converted (arithmetic gap, NOT capex_status='REJECTED')
-- Pre-aggregate po_details to avoid row multiplication (multiple POs per ICA)
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details
    WHERE active = true
    GROUP BY mitigative_action_uuid
)
SELECT
    u.first_name || ' ' || u.last_name      AS fm_manager_name,
    COUNT(DISTINCT ica.id)                  AS ca_count,
    COALESCE(SUM(ica.capex), 0)             AS capex_generated,
    COALESCE(SUM(p.total_po), 0)            AS capex_converted,
    COALESCE(SUM(ica.capex), 0)
        - COALESCE(SUM(p.total_po), 0)      AS capex_lost,
    ROUND(
        COALESCE(SUM(p.total_po), 0) * 100.0
        / NULLIF(SUM(ica.capex), 0), 1
    )                                        AS conversion_pct
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspectee_user_id = u.id
LEFT JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND u.active = true
  AND ica.capex > 0
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_generated DESC
LIMIT 20;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  20,
    },
    {
        'nl':     'which fm manager performs best top performing fm manager',
        'sql':    """\
-- FM Manager performance = high CAPEX conversion rate + low OPEX
-- CAPEX Converted (client pays via PO) = GOOD — Khidmah not bearing the cost
-- OPEX Incurred (Khidmah pays via PR) = BAD  — org absorbing the cost internally
-- Rank by conversion_pct DESC, then opex_incurred ASC
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
),
pr_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    u.first_name || ' ' || u.last_name          AS fm_manager,
    COUNT(DISTINCT ir.id)                        AS inspection_count,
    COUNT(DISTINCT ica.id)                       AS ca_count,
    COALESCE(SUM(ica.capex), 0)                  AS capex_generated,
    COALESCE(SUM(p.total_po), 0)                 AS capex_converted,
    COALESCE(SUM(ica.capex), 0)
        - COALESCE(SUM(p.total_po), 0)           AS capex_lost,
    ROUND(
        COALESCE(SUM(p.total_po), 0) * 100.0
        / NULLIF(SUM(ica.capex), 0), 1
    )                                             AS conversion_pct,
    COALESCE(SUM(pr.total_pr), 0)                AS opex_incurred
FROM inspection_report ir
JOIN users u ON ir.inspectee_user_id = u.id
LEFT JOIN inspection_corrective_action ica
       ON ica.inspection_id = ir.id AND ica.active = true AND ica.capex > 0
LEFT JOIN po_by_ica p  ON p.ica_id  = ica.id
LEFT JOIN pr_by_ica pr ON pr.ica_id = ica.id
WHERE ir.active = true AND u.active = true
GROUP BY u.id, u.first_name, u.last_name
HAVING SUM(ica.capex) > 0
ORDER BY conversion_pct DESC NULLS LAST, opex_incurred ASC
LIMIT 20;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  20,
    },
    {
        'nl':     'which facility costs us the most in capex and opex',
        'sql':    """\
-- CAPEX = SUM(po_details.po_amount) — client pays via PO
-- OPEX = SUM(pr_details.pr_amount) — Khidmah pays internally
-- NEVER use ica.capex or ica.opex as the financial measure here
-- Pre-aggregate to avoid row multiplication
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
),
pr_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    fac.name                                     AS facility_name,
    COALESCE(SUM(ica.capex), 0)                  AS capex_generated,
    COALESCE(SUM(p.total_po), 0)                 AS capex_converted,
    COALESCE(SUM(pr.total_pr), 0)                AS opex_incurred,
    COALESCE(SUM(ica.capex), 0)
        + COALESCE(SUM(pr.total_pr), 0)          AS total_cost
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN po_by_ica p  ON p.ica_id  = ica.id
LEFT JOIN pr_by_ica pr ON pr.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND fac.active = true
GROUP BY fac.id, fac.name
ORDER BY total_cost DESC LIMIT 1;
""",
        'domain': 'financial',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'which facility costs the most capex opex total expenditure by facility',
        'sql':    """\
-- Total facility cost = CAPEX Generated + OPEX Incurred
-- CAPEX = client should cover via PO | OPEX = Khidmah pays internally (bad)
-- Pre-aggregate both po_details and pr_details to avoid row multiplication
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
),
pr_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    fac.name                                     AS facility_name,
    COUNT(DISTINCT ir.id)                        AS inspection_count,
    COUNT(DISTINCT ica.id)                       AS ca_count,
    COALESCE(SUM(ica.capex), 0)                  AS capex_generated,
    COALESCE(SUM(p.total_po), 0)                 AS capex_converted,
    COALESCE(SUM(pr.total_pr), 0)                AS opex_incurred,
    COALESCE(SUM(ica.capex), 0)
        + COALESCE(SUM(pr.total_pr), 0)          AS total_cost
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN inspection_corrective_action ica
       ON ica.inspection_id = ir.id AND ica.active = true
LEFT JOIN po_by_ica p  ON p.ica_id  = ica.id
LEFT JOIN pr_by_ica pr ON pr.ica_id = ica.id
WHERE ir.active = true AND fac.active = true
GROUP BY fac.id, fac.name
ORDER BY total_cost DESC
LIMIT 20;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  20,
    },
    {
        # CAPEX Lost filtered by month — always use date_trunc for month arithmetic.
        # NEVER: EXTRACT(MONTH FROM col) - 1  (fails in January, returns month 0)
        # ALWAYS: date_trunc('month', col) = date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
        'nl':     'capex lost last month capex lost this month capex lost by month filter capex by month',
        'sql':    """\
-- CAPEX Lost = SUM(ica.capex) - SUM(po.po_amount) — arithmetic gap, NOT capex_status
-- Month filter: ALWAYS use date_trunc arithmetic — NEVER EXTRACT(MONTH)-1 (fails in January)
-- Last month: date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
-- This month: date_trunc('month', CURRENT_DATE)
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
)
SELECT
    COALESCE(SUM(ica.capex), 0)          AS capex_generated,
    COALESCE(SUM(p.total_po), 0)         AS capex_converted,
    COALESCE(SUM(ica.capex), 0)
        - COALESCE(SUM(p.total_po), 0)   AS capex_lost
FROM inspection_corrective_action ica
LEFT JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ica.active = true
  AND ica.capex > 0
  AND date_trunc('month', ica.created_on) = date_trunc('month', CURRENT_DATE - INTERVAL '1 month');
""",
        'domain': 'financial',
        'intent': 'single_value',
        'limit':  None,
    },
    {
        'nl':     'what is the total capex generated converted and lost',
        'sql':    """\
-- USE po_details for capex_converted, arithmetic for capex_lost
-- NEVER use capex_status='ACCEPTED' for converted or 'REJECTED' for lost
SELECT
    (SELECT COALESCE(SUM(capex), 0)    FROM inspection_corrective_action
     WHERE active = true AND capex > 0)                                    AS capex_generated,
    (SELECT COALESCE(SUM(po_amount), 0) FROM po_details
     WHERE active = true)                                                   AS capex_converted,
    COALESCE(
        (SELECT SUM(capex) FROM inspection_corrective_action
         WHERE active = true AND capex > 0), 0
    ) - COALESCE(
        (SELECT SUM(po_amount) FROM po_details WHERE active = true), 0
    )                                                                       AS capex_lost;
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'what percentage of our capex is being converted by the client',
        'sql':    """\
-- CAPEX converted by client = SUM(po_details.po_amount) — NOT ica.capex WHERE responsible='CLIENT'
-- The po_details table records actual client PO payments
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
)
SELECT
    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 2) AS client_percentage
FROM inspection_corrective_action ica
JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ica.active = true AND ica.capex > 0;
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'total capex generated converted lost financial summary',
        'sql':    """\
-- USE THIS EXACT SQL — ONE ROW scalar summary, NO GROUP BY:
-- CAPEX Lost = Generated - Converted (arithmetic), NOT WHERE capex_status='REJECTED'
SELECT
    (SELECT COALESCE(SUM(capex), 0)    FROM inspection_corrective_action
     WHERE active = true AND capex > 0)                                    AS capex_generated,
    (SELECT COALESCE(SUM(po_amount), 0) FROM po_details
     WHERE active = true)                                                   AS capex_converted,
    COALESCE(
        (SELECT SUM(capex) FROM inspection_corrective_action
         WHERE active = true AND capex > 0), 0
    ) - COALESCE(
        (SELECT SUM(po_amount) FROM po_details WHERE active = true), 0
    )                                                                       AS capex_lost,
    (SELECT COALESCE(SUM(pr_amount), 0) FROM pr_details
     WHERE active = true)                                                   AS opex_incurred;
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'how much opex have we incurred this year total opex incurred',
        'sql':    """\
-- OPEX Incurred = SUM(pr.pr_amount) from pr_details — Khidmah's internal spend via PR
-- NEVER use SUM(ica.opex) — ica.opex is just the estimated amount on the ICA record
SELECT
    COALESCE(SUM(pr.pr_amount), 0)  AS total_opex_incurred,
    COUNT(DISTINCT pr.id)           AS pr_count
FROM pr_details pr
WHERE pr.active = true
  AND EXTRACT(YEAR FROM pr.pr_date) = EXTRACT(YEAR FROM CURRENT_DATE);
""",
        'domain': 'financial',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'opex incurred by month this year monthly opex breakdown',
        'sql':    """\
-- OPEX Incurred = SUM(pr.pr_amount) from pr_details — group by pr_date not ica.created_on
SELECT
    date_trunc('month', pr.pr_date)         AS month,
    TO_CHAR(pr.pr_date, 'Month YYYY')       AS month_label,
    COALESCE(SUM(pr.pr_amount), 0)          AS total_opex_incurred
FROM pr_details pr
WHERE pr.active = true
  AND EXTRACT(YEAR FROM pr.pr_date) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY date_trunc('month', pr.pr_date), TO_CHAR(pr.pr_date, 'Month YYYY')
ORDER BY month;
""",
        'domain': 'financial',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'facility costs most capex opex which facility highest spend total cost',
        'sql':    """\
-- Total facility cost = CAPEX Generated + OPEX Incurred (not ica.opex/capex alone)
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
),
pr_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    fac.name                                AS facility_name,
    COALESCE(SUM(ica.capex), 0)             AS capex_generated,
    COALESCE(SUM(p.total_po), 0)            AS capex_converted,
    COALESCE(SUM(pr.total_pr), 0)           AS opex_incurred,
    COALESCE(SUM(ica.capex), 0)
        + COALESCE(SUM(pr.total_pr), 0)     AS total_cost
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN po_by_ica p  ON p.ica_id  = ica.id
LEFT JOIN pr_by_ica pr ON pr.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND fac.active = true
GROUP BY fac.id, fac.name
ORDER BY total_cost DESC
LIMIT 20;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  20,
    },
    {
        'nl':     'facilities where opex higher than capex converted opex exceeds capex po',
        'sql':    """\
-- OPEX > CAPEX Converted: use pr_details and po_details — NOT ica.opex/ica.capex columns
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
),
pr_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    fac.name                            AS facility_name,
    COALESCE(SUM(pr.total_pr), 0)       AS opex_incurred,
    COALESCE(SUM(p.total_po), 0)        AS capex_converted,
    COALESCE(SUM(pr.total_tr), 0)
        - COALESCE(SUM(p.total_po), 0)  AS opex_excess
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN po_by_ica p  ON p.ica_id  = ica.id
LEFT JOIN pr_by_ica pr ON pr.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND fac.active = true
GROUP BY fac.id, fac.name
HAVING COALESCE(SUM(pr.total_tr), 0) > COALESCE(SUM(p.total_po), 0)
ORDER BY opex_excess DESC;
""",
        'domain': 'financial',
        'intent': 'filter',
        'limit':  None,
    },
    {
        'nl':     'show me total capex generated and converted by facility this year',
        'sql':    """\
-- FIN-07: CAPEX generated = SUM(ica.capex), converted = SUM(po.po_amount) via po_details
-- NEVER use capex_status='ACCEPTED' as a filter — use po_details for converted amount
-- Pre-aggregate po_details to avoid row multiplication
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
)
SELECT
    fac.name                                        AS facility_name,
    COALESCE(SUM(ica.capex), 0)                     AS capex_generated,
    COALESCE(SUM(p.total_po), 0)                    AS capex_converted,
    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND fac.active = true AND ica.capex > 0
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY fac.id, fac.name
ORDER BY capex_generated DESC NULLS LAST;
""",
        'domain': 'financial',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'top 5 capex converters who converted the most capex po amount',
        'sql':    """\
-- FIN-09: CAPEX converters = FM managers ranked by SUM(po.po_amount) from po_details
-- NEVER rank by ica.capex or ORDER BY capex DESC — that's GENERATED not CONVERTED
-- NEVER filter capex_status='ACCEPTED' — use po_details for actual converted amounts
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
)
SELECT
    u.first_name || ' ' || u.last_name  AS fm_manager,
    COUNT(DISTINCT ica.id)              AS ca_count,
    COALESCE(SUM(ica.capex), 0)         AS capex_generated,
    COALESCE(SUM(p.total_po), 0)        AS capex_converted
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspectee_user_id = u.id
LEFT JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND u.active = true AND ica.capex > 0
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_converted DESC NULLS LAST
LIMIT 5;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  5,
    },
    {
        'nl':     'capex conversion rate by facility show conversion percentage facility',
        'sql':    """\
-- CAPEX Conversion Rate = SUM(po.po_amount) / NULLIF(SUM(ica.capex),0) * 100
-- NEVER use ica.expenditure enum or capex_status='ACCEPTED'
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
)
SELECT
    fac.name                                        AS facility_name,
    COALESCE(SUM(ica.capex), 0)                     AS capex_generated,
    COALESCE(SUM(p.total_po), 0)                    AS capex_converted,
    ROUND(
        COALESCE(SUM(p.total_po), 0) * 100.0
        / NULLIF(SUM(ica.capex), 0), 1
    )                                               AS conversion_pct
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND fac.active = true AND ica.capex > 0
GROUP BY fac.id, fac.name
ORDER BY conversion_pct DESC NULLS LAST;
""",
        'domain': 'financial',
        'intent': 'filter',
        'limit':  None,
    },
    {
        'nl':     'capex converted this quarter vs last quarter capex trend quarterly',
        'sql':    """\
-- CAPEX Converted = SUM(po.po_amount) from po_details — NOT capex_status='ACCEPTED'
-- Use po.po_date to determine the quarter — NOT ica.close_on or ica.modified_on
SELECT
    date_trunc('quarter', po.po_date)               AS quarter_start,
    TO_CHAR(po.po_date, '"Q"Q YYYY')                AS quarter_label,
    COALESCE(SUM(po.po_amount), 0)                  AS capex_converted,
    COUNT(DISTINCT po.mitigative_action_uuid)        AS po_count
FROM po_details po
WHERE po.active = true
  AND po.po_date >= date_trunc('quarter', CURRENT_DATE) - INTERVAL '3 months'
GROUP BY date_trunc('quarter', po.po_date), TO_CHAR(po.po_date, '"Q"Q YYYY')
ORDER BY quarter_start;
""",
        'domain': 'financial',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'capex conversion rate trend quarterly how has capex trend 4 quarters',
        'sql':    """\
-- CAPEX quarterly conversion trend — use po_date from po_details NOT capex_status
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id,
           SUM(po_amount) AS total_po, MIN(po_date) AS earliest_po_date
    FROM po_details WHERE active = true GROUP BY 1
)
SELECT
    date_trunc('quarter', p.earliest_po_date)       AS quarter_start,
    TO_CHAR(p.earliest_po_date, '"Q"Q YYYY')        AS quarter_label,
    COALESCE(SUM(ica.capex), 0)                     AS capex_generated,
    COALESCE(SUM(p.total_po), 0)                    AS capex_converted,
    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct
FROM inspection_corrective_action ica
JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ica.active = true
  AND p.earliest_po_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY date_trunc('quarter', p.earliest_po_date), TO_CHAR(p.earliest_po_date, '"Q"Q YYYY')
ORDER BY quarter_start;
""",
        'domain': 'financial',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'high risk observations raised this month count risk level observations month',
        'sql':    """\
-- High-risk OBSERVATIONS = corrective actions with risk_level JOIN — NOT ai_answers
-- NEVER: ai_answers WHERE label='Observation' AND answer_text ILIKE '%High%'
SELECT COUNT(*) AS high_risk_observation_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN risk_level rl ON ica.risk_level_id = rl.id
WHERE rl.name = 'High'
  AND ir.active = true AND ica.active = true
  AND date_trunc('month', ir.submitted_on) = date_trunc('month', CURRENT_DATE);
""",
        'domain': 'form_answers',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'which facility most critical high risk deviations risk by facility critical',
        'sql':    """\
-- NEVER use ai_answers for risk counts — use risk_level table
SELECT fac.name AS facility_name, COUNT(ica.id) AS high_risk_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN risk_level rl ON ica.risk_level_id = rl.id
WHERE rl.name = 'High'
  AND ir.active = true AND ica.active = true AND fac.active = true
GROUP BY fac.id, fac.name
ORDER BY high_risk_count DESC LIMIT 10;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'top 5 performing fm managers performance best fm manager rank',
        'sql':    """\
-- FM Manager performance = high CAPEX conversion rate + low OPEX
-- NEVER rank by avg_score (that's inspector quality, not FM performance)
-- NEVER GROUP BY ica.responsible (gives CLIENT/INTERNAL/SUB_CONTRACTOR enum, not people)
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
),
pr_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    u.first_name || ' ' || u.last_name          AS fm_manager,
    COALESCE(SUM(ica.capex), 0)                  AS capex_generated,
    COALESCE(SUM(p.total_po), 0)                 AS capex_converted,
    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct,
    COALESCE(SUM(pr.total_pr), 0)                AS opex_incurred
FROM inspection_report ir
JOIN users u ON ir.inspectee_user_id = u.id
LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id AND ica.active = true AND ica.capex > 0
LEFT JOIN po_by_ica p  ON p.ica_id  = ica.id
LEFT JOIN pr_by_ica pr ON pr.ica_id = ica.id
WHERE ir.active = true AND u.active = true
GROUP BY u.id, u.first_name, u.last_name
HAVING SUM(ica.capex) > 0
ORDER BY conversion_pct DESC NULLS LAST, opex_incurred ASC
LIMIT 5;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  5,
    },
    {
        'nl':     'which fm manager generates most capex capex generator fm manager',
        'sql':    """\
-- FM Manager = inspectee_user_id (NOT inspector_user_id, NOT ica.responsible)
-- NEVER GROUP BY ica.responsible — that enum gives CLIENT/INTERNAL/SUB_CONTRACTOR not people
SELECT
    u.first_name || ' ' || u.last_name  AS fm_manager,
    COUNT(DISTINCT ica.id)              AS ca_count,
    COALESCE(SUM(ica.capex), 0)         AS capex_generated
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspectee_user_id = u.id
WHERE ir.active = true AND ica.active = true AND u.active = true AND ica.capex > 0
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_generated DESC LIMIT 10;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'fm manager most overdue corrective actions which fm manager overdue',
        'sql':    """\
-- FM Manager = inspectee_user_id (NOT inspector_user_id)
-- Overdue = target_close_out_date < CURRENT_DATE AND status not closed
SELECT
    u.first_name || ' ' || u.last_name  AS fm_manager,
    COUNT(ica.id)                       AS overdue_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspectee_user_id = u.id
WHERE ir.active = true AND ica.active = true AND u.active = true
  AND ica.target_close_out_date < CURRENT_DATE
  AND ica.status NOT IN ('CLOSED', 'CLOSE_WITH_DEFERRED')
GROUP BY u.id, u.first_name, u.last_name
ORDER BY overdue_count DESC LIMIT 10;
""",
        'domain': 'corrective_action',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'top 5 capex converters who are the top capex converters',
        'sql':    """\
-- CAPEX Converter = FM manager ranked by SUM(po.po_amount) from po_details
-- NOT: ORDER BY ica.capex (that's CAPEX Generated, not Converted)
-- NOT: capex_status='ACCEPTED' (that's a workflow flag, not conversion)
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
)
SELECT
    u.first_name || ' ' || u.last_name      AS fm_manager,
    COUNT(DISTINCT ica.id)                  AS ca_count,
    COALESCE(SUM(ica.capex), 0)             AS capex_generated,
    COALESCE(SUM(p.total_po), 0)            AS capex_converted,
    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspectee_user_id = u.id
LEFT JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND u.active = true AND ica.capex > 0
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_converted DESC
LIMIT 5;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  5,
    },
    {
        'nl':     'who are the top 5 performing fm managers best fm manager',
        'sql':    """\
-- FM Manager performance = high CAPEX conversion rate + low OPEX
-- NEVER rank by avg_inspection_score — that's INSPECTOR quality, not FM performance
-- NEVER GROUP BY ica.responsible — that gives CLIENT/INTERNAL/SUB_CONTRACTOR enum not people
-- Filter to FM-role users via EXISTS to avoid row fan-out from user_roles join
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
),
pr_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    u.first_name || ' ' || u.last_name          AS fm_manager,
    COALESCE(SUM(ica.capex), 0)                  AS capex_generated,
    COALESCE(SUM(p.total_po), 0)                 AS capex_converted,
    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct,
    COALESCE(SUM(pr.total_tr), 0)                AS opex_incurred
FROM inspection_report ir
JOIN users u ON ir.inspectee_user_id = u.id
LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id AND ica.active = true AND ica.capex > 0
LEFT JOIN po_by_ica p  ON p.ica_id  = ica.id
LEFT JOIN pr_by_ica pr ON pr.ica_id = ica.id
WHERE ir.active = true AND u.active = true
  AND EXISTS (
      SELECT 1 FROM user_roles ur JOIN role r ON ur.role_id = r.id
      WHERE ur.user_id = u.id
        AND r.name IN ('FM Manager','FM Engineer','AD FM',
                       'Asst FM Manager','FM Admin','FM Supervisor')
  )
GROUP BY u.id, u.first_name, u.last_name
HAVING SUM(ica.capex) > 0
ORDER BY conversion_pct DESC NULLS LAST, opex_incurred ASC
LIMIT 5;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  5,
    },
    {
        'nl':     'which fm manager has the highest capex conversion rate single top fm manager',
        'sql':    """\
-- FM-02: single FM manager with THE HIGHEST conversion rate — LIMIT 1
-- conversion_rate = SUM(po.po_amount) / NULLIF(SUM(ica.capex), 0) * 100
-- LIMIT 1 returns the single top person. LIMIT 5+ returns a ranked list.
-- FM Manager = inspectee_user_id — no role filter needed via this join path
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
    FROM po_details WHERE active = true GROUP BY 1
)
SELECT
    u.first_name || ' ' || u.last_name  AS fm_manager,
    COUNT(DISTINCT ica.id)              AS ca_count,
    COALESCE(SUM(ica.capex), 0)         AS capex_generated,
    COALESCE(SUM(p.total_po), 0)        AS capex_converted,
    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_rate
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspectee_user_id = u.id
LEFT JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ir.active = true AND ica.active = true AND u.active = true AND ica.capex > 0
GROUP BY u.id, u.first_name, u.last_name
HAVING SUM(ica.capex) > 0
ORDER BY conversion_rate DESC NULLS LAST
LIMIT 1;
""",
        'domain': 'financial',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'which fm manager generates most capex fm manager capex generator',
        'sql':    """\
-- FM Manager = inspectee_user_id (NOT inspector_user_id, NOT ica.responsible)
-- CAPEX Generated = SUM(ica.capex) — the proposed amount on each ICA
-- GROUP BY u.id not by ica.responsible (ica.responsible is CLIENT/INTERNAL/SUB_CONTRACTOR enum)
-- Filter to FM-role users via EXISTS subquery
SELECT
    u.first_name || ' ' || u.last_name  AS fm_manager,
    COUNT(DISTINCT ica.id)              AS ca_count,
    COALESCE(SUM(ica.capex), 0)         AS capex_generated
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspectee_user_id = u.id
WHERE ir.active = true AND ica.active = true AND u.active = true AND ica.capex > 0
  AND EXISTS (
      SELECT 1 FROM user_roles ur JOIN role r ON ur.role_id = r.id
      WHERE ur.user_id = u.id
        AND r.name IN ('FM Manager','FM Engineer','AD FM',
                       'Asst FM Manager','FM Admin','FM Supervisor')
  )
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_generated DESC LIMIT 10;
""",
        'domain': 'financial',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'fm managers ranked by opex incurred show fm manager opex',
        'sql':    """\
-- OPEX Incurred = SUM(pr.pr_amount) from pr_details — NOT SUM(ica.opex)
-- FM Manager = inspectee_user_id
-- pr_details links via: pr.mitigative_action_uuid = ica.id
WITH pr_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr
    FROM pr_details WHERE active = true GROUP BY 1
)
SELECT
    u.first_name || ' ' || u.last_name  AS fm_manager,
    COUNT(DISTINCT ica.id)              AS ca_count,
    COALESCE(SUM(pr.total_tr), 0)       AS opex_incurred
FROM inspection_report ir
JOIN users u ON ir.inspectee_user_id = u.id
LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id AND ica.active = true
LEFT JOIN pr_by_ica pr ON pr.ica_id = ica.id
WHERE ir.active = true AND u.active = true
GROUP BY u.id, u.first_name, u.last_name
ORDER BY opex_incurred DESC NULLS LAST
LIMIT 20;
""",
        'domain': 'financial',
        'intent': 'filter',
        'limit':  20,
    },
    {
        'nl':     'which fm manager has most overdue corrective actions fm manager overdue',
        'sql':    """\
-- FM Manager = inspectee_user_id (NOT inspector_user_id, NOT ica.responsible)
-- NEVER GROUP BY ica.responsible — gives CLIENT/INTERNAL_OPERATIONS/SUB_CONTRACTOR not people
-- Overdue: target_close_out_date < CURRENT_DATE AND status NOT IN closed statuses
SELECT
    u.first_name || ' ' || u.last_name  AS fm_manager,
    COUNT(ica.id)                       AS overdue_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspectee_user_id = u.id
WHERE ir.active = true AND ica.active = true AND u.active = true
  AND ica.target_close_out_date < CURRENT_DATE
  AND ica.status NOT IN ('CLOSED', 'CLOSE_WITH_DEFERRED')
GROUP BY u.id, u.first_name, u.last_name
ORDER BY overdue_count DESC LIMIT 10;
""",
        'domain': 'corrective_action',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'which inspection type raises most high risk deviations inspection type risk',
        'sql':    """\
-- High-risk deviations by inspection type — use risk_level table JOIN, NOT ai_answers
-- NEVER: ai_answers WHERE label='Risk Level' AND answer_text ILIKE '%High%'
SELECT
    it.name AS inspection_type_name,
    COUNT(*) AS high_risk_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
JOIN risk_level rl ON ica.risk_level_id = rl.id
WHERE rl.name = 'High'
  AND ir.active = true AND ica.active = true AND it.active = true
GROUP BY it.id, it.name
ORDER BY high_risk_count DESC LIMIT 10;
""",
        'domain': 'form_answers',
        'intent': 'ranked_list',
        'limit':  10,
    },
    {
        'nl':     'how many inspections were scheduled vs completed this month scheduled completed comparison',
        'sql':    """\
-- Scheduled = rows in inspection_schedule for this month
-- Conducted = SUBMITTED + UNDER_REVIEW + RETURN_FOR_MODIFICATION
-- Completed = CLOSED only
SELECT
    (SELECT COUNT(*)
     FROM inspection_schedule
     WHERE date_trunc('month', due_date) = date_trunc('month', CURRENT_DATE)) AS scheduled_count,
    COUNT(*) FILTER (WHERE ir.status != 'CLOSED') AS conducted_count,
    COUNT(*) FILTER (WHERE ir.status = 'CLOSED')  AS completed_count
FROM inspection_report ir
WHERE ir.active = true
  AND ir.status != 'DRAFT'
  AND date_trunc('month', ir.submitted_on) = date_trunc('month', CURRENT_DATE);
""",
        'domain': 'schedule',
        'intent': 'aggregate',
        'limit':  None,
    },
    {
        'nl':     'capex conversion rate quarterly trend 4 quarters capex converted trend quarterly',
        'sql':    """\
-- CAPEX Conversion Rate quarterly trend = SUM(po.po_amount) / SUM(ica.capex) per quarter
-- NEVER use capex_status='ACCEPTED' for conversion — that's a workflow approval flag
-- capex_status is a valid column but it does NOT measure conversion (use po_details)
-- Use po.po_date to determine which quarter the PO was raised
WITH po_by_ica AS (
    SELECT mitigative_action_uuid AS ica_id,
           SUM(po_amount) AS total_po,
           MIN(po_date)   AS earliest_po_date
    FROM po_details WHERE active = true GROUP BY 1
)
SELECT
    date_trunc('quarter', p.earliest_po_date)       AS quarter_start,
    TO_CHAR(p.earliest_po_date, '"Q"Q YYYY')        AS quarter_label,
    COALESCE(SUM(ica.capex), 0)                     AS capex_generated,
    COALESCE(SUM(p.total_po), 0)                    AS capex_converted,
    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct
FROM inspection_corrective_action ica
JOIN po_by_ica p ON p.ica_id = ica.id
WHERE ica.active = true
  AND p.earliest_po_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY date_trunc('quarter', p.earliest_po_date), TO_CHAR(p.earliest_po_date, '"Q"Q YYYY')
ORDER BY quarter_start;
""",
        'domain': 'financial',
        'intent': 'trend',
        'limit':  None,
    },
    {
        'nl':     'list questions and answers most recent inspection form show qa most recent form',
        'sql':    """\
-- List all Q&A from the most recent inspection WITH form answers
-- Use ai_answers subquery anchored on aa.submitted_on — NOT ir.submitted_on
-- NEVER filter module_name in the inner subquery (causes empty result)
SELECT
    aq.label        AS question,
    aa.answer_text  AS answer,
    aa.module_name  AS form_section
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aa.inspection_report_id = (
    SELECT aa2.inspection_report_id FROM ai_answers aa2
    WHERE aa2.submitted_on IS NOT NULL
    ORDER BY aa2.submitted_on DESC LIMIT 1
)
  ORDER BY aa.module_name, aq.label
LIMIT 200;
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
    {
        'nl':     'show questions and answers most recently answered inspection qa for most recent',
        'sql':    """\
-- Q&A for most recently answered inspection — use ai_answers + ai_questions
-- Anchor on aa.submitted_on DESC not ir.submitted_on
SELECT
    aq.label        AS question,
    aa.answer_text  AS answer,
    aa.module_name  AS form_section
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aa.inspection_report_id = (
    SELECT aa2.inspection_report_id FROM ai_answers aa2
    WHERE aa2.submitted_on IS NOT NULL
    ORDER BY aa2.submitted_on DESC LIMIT 1
)
  ORDER BY aa.module_name, aq.label
LIMIT 200;
""",
        'domain': 'form_answers',
        'intent': 'single_value',
        'limit':  1,
    },
]

# ── Conditional CORE_CONTEXT additions — included only when relevant ────────
# Deepseek-side counterpart to prompts.py's CONDITIONAL_BLOCKS / EXAMPLES_TIER2.
# _COUNTING_BLOCK (good practice/deviation/observation counting, ~148 tok) and
# _CA_WORKFLOW_BLOCK (CA workflow columns, ~805 tok) were always-on parts of
# CORE_CONTEXT before 2026-06-13's truncation fix — both grew this session and
# are narrowly topical, so they're now conditional like their Llama-side
# counterparts (CONDITIONAL_BLOCKS["CA workflow columns"] and the FORM-pattern
# Tier-2 examples in prompts.py).
#
# Patterns are intentionally KEPT IN SYNC with prompts.py's CA/FORM patterns —
# both sides should treat "is this question about corrective actions / form
# answers" the same way. If you edit one, edit the other.
CORE_CONTEXT_CONDITIONAL = [
    {
        "title": "CA workflow columns (Deepseek)",
        "question_pattern": (
            r"(?i)\bcorrective\s+actions?\b|\bCAs?\b|\bicas?\b"
            r"|\bcapex\b|\bopex\b|\bdeferred\b"
            r"|\bprogress[_\s]stage\b|\bimplementation[_\s]status\b"
            r"|\bneeds?\s+attention\b|\bcritical\s+issues?\b"
        ),
        "get_text": lambda: _CA_WORKFLOW_BLOCK,
    },
    {
        "title": "Good practice / deviation / observation counting (Deepseek)",
        "question_pattern": (
            r"(?i)\b(forms?|answer\w*|questions?|observations?|risk\s*levels?"
            r"|deviations?|good\s+practice|total\s+score|ai_answers)\b"
        ),
        "get_text": lambda: _COUNTING_BLOCK,
    },
]


def build_core_context_additions(question: str) -> str:
    """
    Return additional CORE_CONTEXT blocks whose question_pattern matches the
    current question. Call from tools.py's generate_sql: append the result to
    CORE_CONTEXT before building dynamic_examples.

    Returns "" if nothing matches — no size penalty for unrelated questions.
    """
    if not question:
        return ""
    matched = []
    for blk in CORE_CONTEXT_CONDITIONAL:
        try:
            if re.search(blk["question_pattern"], question):
                matched.append(blk["get_text"]())
        except re.error:
            continue
    if not matched:
        return ""
    return "\n--\n" + "\n--\n".join(matched)