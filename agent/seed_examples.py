"""
Seed example corpus for SeedExampleIndex.
Each entry is a (nl_question, sql_snippet) pair.

The NL question is embedded at startup via BAAI/bge-small-en-v1.5.
At generate_sql call time, top-k most similar examples are retrieved
by cosine similarity and injected into the SQL prompt.

ALWAYS-INJECTED section (first 3 entries, tagged "core"):
  These are included regardless of similarity score because they encode
  schema invariants that apply to nearly every query (ir.id trap, ICA join).

DOMAIN EXAMPLES (remaining entries):
  Embedded and retrieved by semantic similarity to the incoming question.
  ~150-600 tokens injected per query vs 6081 tokens static.

Adding new examples: append (nl_question, sql_snippet) tuples.
The system self-improves as new query patterns are identified.
"""

# ── Always-injected schema invariants ────────────────────────────────────────
# Not embedded — prepended unconditionally to every SQL call.
CORE_CONTEXT = """\
### SQL RULES FOR DEEPSEEK (not for end user):
-- ir.inspection_id (VARCHAR) = display id | ir.id (UUID) = JOIN/subquery only, NEVER SELECT
-- ICA join: ON ica.inspection_id = ir.id  (NOT ir.inspection_id = ica.inspection_id)
-- completed = status IN ('CLOSED','SUBMITTED') | responsible: 'CLIENT','INTERNAL_OPERATIONS','SUB_CONTRACTOR'
-- NEVER: ir.id in SELECT | NEVER: ica.risk_level_id vs string | NEVER: GROUP BY ica.risk_level_id
-- completed_cycle = status='COMPLETED' (NOT 'CLOSED', NOT status!='DRAFT')
-- NEVER: ipd.cycle_id (no such column on inspector_portfolio_details)
-- Cycle->Freq: inspection_cycle.id -> inspector_portfolio.cycle_id
--   -> inspector_portfolio_details.portfolio_id -> frequency_definition.id
-- FM Manager / inspectee = ir.inspectee_user_id (NOT ir.inspector_user_id)
-- capex_status: NULL=awaiting approval | ACCEPTED | REJECTED | DEFERRED
-- expenditure: NO_EXPENDITURE | CAPEX | OPEX | CAPEX_AND_OPEX
-- CAPEX Generated  = SUM(ica.capex)               — original quoted amount
-- CAPEX Converted  = SUM(po.po_amount)             — actual PO raised by client
-- CAPEX Lost       = SUM(ica.capex) WHERE capex_status='REJECTED'
-- OPEX Incurred    = SUM(pr.pr_amount)             — actual PR raised internally
-- po_details JOIN: ON po.mitigative_action_uuid = ica.id  (one ICA can have many POs)
-- pr_details JOIN: ON pr.mitigative_action_uuid = ica.id  (one ICA can have many PRs)
"""

# ── Seed corpus ───────────────────────────────────────────────────────────────
# Format: (nl_question, sql_snippet)
# nl_question is what gets embedded. Keep it concise and representative.

SEED_EXAMPLES: list[tuple[str, str]] = [

    # ── Aggregate / Count ────────────────────────────────────────────────────
    (
        "how many inspections were completed this year",
        """\
SELECT COUNT(*) AS completed_count
FROM inspection_report ir
WHERE ir.status IN ('CLOSED','SUBMITTED')
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE);""",
    ),
    (
        "how many open corrective actions are there",
        """\
SELECT COUNT(*) AS open_count
FROM inspection_corrective_action WHERE status = 'OPEN';""",
    ),
    (
        "what is the average inspection score",
        """\
SELECT ROUND(AVG(ir.inspection_score)::numeric, 2) AS average_score
FROM inspection_report ir
WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL;""",
    ),
    (
        "total capex and opex across all corrective actions",
        """\
SELECT SUM(ica.capex) AS total_capex, SUM(ica.opex) AS total_opex
FROM inspection_corrective_action ica;""",
    ),
    (
        "what percentage of inspections resulted in at least one corrective action",
        """\
-- USE THIS EXACT SQL — scalar percentage, no GROUP BY:
SELECT ROUND(
    100.0 * COUNT(DISTINCT ica.inspection_id) / NULLIF(COUNT(DISTINCT ir.id), 0), 1
) AS pct_with_ca
FROM inspection_report ir
LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id
WHERE ir.status != 'DRAFT';""",
    ),
    (
        "average inspection duration in hours",
        """\
-- USE THIS EXACT PATTERN — total_inspection_hours does NOT exist; use timestamp math:
SELECT ROUND(
    AVG(EXTRACT(EPOCH FROM (ir.submitted_on - ir.start_date_time)) / 3600)::numeric, 2
) AS avg_duration_hours
FROM inspection_report ir
WHERE ir.status != 'DRAFT' AND ir.start_date_time IS NOT NULL;""",
    ),
    (
        "average number of observations per inspection",
        """\
-- Subquery pattern: count per inspection FIRST, then AVG
SELECT ROUND(AVG(obs_count)::numeric, 2) AS avg_obs_per_inspection
FROM (
    SELECT aa.inspection_report_id, COUNT(*) AS obs_count
    FROM ai_answers aa
    JOIN ai_questions aq ON aa.element_id = aq.element_id
    WHERE aq.label = 'Observation' AND aa.answer_text IS NOT NULL
    GROUP BY aa.inspection_report_id
) sub;""",
    ),

    # ── Temporal / Comparison ────────────────────────────────────────────────
    (
        "how many inspections were completed this month vs last month",
        """\
-- USE THIS EXACT SQL — single-row FILTER comparison, NOT GROUP BY:
-- Single-row FILTER form — NOT GROUP BY
SELECT
  COUNT(*) FILTER (WHERE date_trunc('month', ir.submitted_on) = date_trunc('month', NOW())) AS this_month_count,
  COUNT(*) FILTER (WHERE date_trunc('month', ir.submitted_on) = date_trunc('month', NOW()-INTERVAL '1 month')) AS last_month_count
FROM inspection_report ir WHERE ir.status IN ('CLOSED','SUBMITTED');""",
    ),
    (
        "how many corrective actions were raised vs closed this quarter",
        """\
-- USE THIS EXACT SQL — single-row FILTER aggregate, NOT GROUP BY:
-- Single-row FILTER — NOT GROUP BY status
SELECT COUNT(*) AS total_raised,
       COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS total_closed,
       COUNT(*) FILTER (WHERE status IN ('OPEN','OVERDUE')) AS still_open
FROM inspection_corrective_action
WHERE created_on >= date_trunc('quarter', NOW());""",
    ),
    (
        "how has the average inspection score changed month by month this year",
        """\
SELECT TO_CHAR(date_trunc('month', ir.submitted_on), 'Month YYYY') AS month,
       COUNT(*) AS inspection_count,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score
FROM inspection_report ir
WHERE ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY date_trunc('month', ir.submitted_on)
ORDER BY date_trunc('month', ir.submitted_on);""",
    ),
    (
        "which month had the most high risk observations this year",
        """\
SELECT TO_CHAR(date_trunc('month', ir.submitted_on), 'Month YYYY') AS month,
       COUNT(*) AS high_risk_count
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN ai_answers risk_aa ON risk_aa.inspection_report_id = aa.inspection_report_id
JOIN ai_questions risk_aq ON risk_aa.element_id = risk_aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
WHERE aq.label = 'Observation'
  AND risk_aq.label = 'Risk Level' AND risk_aa.answer_text = 'High'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY date_trunc('month', ir.submitted_on)
ORDER BY high_risk_count DESC LIMIT 1;""",
    ),
    (
        "are corrective actions being closed faster or slower compared to last quarter",
        """\
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
ORDER BY 1;""",
    ),

    # ── Filter / List ────────────────────────────────────────────────────────
    (
        "show the last 5 inspections by submission date",
        """\
SELECT ir.inspection_id, fac.name AS facility_name,
       it.name AS inspection_type_name, ir.submitted_on, ir.status, ir.inspection_score
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 5;""",
    ),
    (
        "show all overdue corrective actions",
        """\
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.target_close_out_date,
       fac.name AS facility_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.status = 'OVERDUE'
ORDER BY ica.target_close_out_date ASC LIMIT 100;""",
    ),
    (
        "which inspections were returned for modification",
        """\
SELECT ir.inspection_id, fac.name AS facility_name,
       it.name AS inspection_type_name, ir.submitted_on
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE ir.status = 'RETURN_FOR_MODIFICATION'
ORDER BY ir.submitted_on DESC LIMIT 50;""",
    ),
    (
        "show me inspections with a score below 70",
        """\
SELECT ir.inspection_id, ir.inspection_score, ir.status,
       fac.name AS facility_name, ir.submitted_on
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.inspection_score < 70 AND ir.status != 'DRAFT'
ORDER BY ir.inspection_score ASC LIMIT 50;""",
    ),
    (
        "which facilities haven't been inspected in the last 30 days",
        """\
SELECT fac.name AS facility_name
FROM facility fac
WHERE fac.id NOT IN (
    SELECT DISTINCT ir.facility_id FROM inspection_report ir
    WHERE ir.submitted_on >= CURRENT_DATE - INTERVAL '30 days'
      AND ir.status != 'DRAFT'
)
ORDER BY fac.name LIMIT 100;""",
    ),
    (
        "how many inspections were done on weekends",
        """\
SELECT COUNT(*) AS weekend_inspection_count
FROM inspection_report ir
WHERE EXTRACT(ISODOW FROM ir.submitted_on) IN (6, 7)
  AND ir.status != 'DRAFT';""",
    ),

    # ── Join / Grouped ───────────────────────────────────────────────────────
    (
        "show inspection scores grouped by facility",
        """\
SELECT fac.name AS facility_name,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score,
       COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
GROUP BY fac.name ORDER BY avg_score DESC LIMIT 50;""",
    ),
    (
        "which inspector has conducted the most inspections this year",
        """\
SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN users u ON ir.inspector_user_id = u.id
WHERE ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY u.id, u.first_name, u.last_name
ORDER BY inspection_count DESC LIMIT 1;""",
    ),
    (
        "show average inspection score by inspection type",
        """\
SELECT it.name AS inspection_type_name,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score,
       COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
GROUP BY it.name ORDER BY avg_score DESC;""",
    ),
    (
        "show corrective actions with their inspection facility and inspector",
        """\
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.status,
       fac.name AS facility_name,
       u.first_name || ' ' || u.last_name AS inspector_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN users u ON ir.inspector_user_id = u.id
LIMIT 100;""",
    ),
    (
        "which client has the most overdue corrective actions",
        """\
SELECT cl.name AS client_name, COUNT(*) AS overdue_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN client cl ON ir.client_id = cl.id
WHERE ica.status = 'OVERDUE'
GROUP BY cl.name ORDER BY overdue_count DESC LIMIT 10;""",
    ),
    (
        "what is the total capex and opex by facility",
        """\
SELECT fac.name AS facility_name,
       SUM(ica.capex) AS total_capex, SUM(ica.opex) AS total_opex
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
GROUP BY fac.name ORDER BY total_capex DESC NULLS LAST LIMIT 50;""",
    ),
    (
        "show facilities that have been inspected more than 5 times this year",
        """\
SELECT fac.name AS facility_name, COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY fac.name HAVING COUNT(*) > 5
ORDER BY inspection_count DESC;""",
    ),
    (
        "which projects have not had any inspections this year",
        """\
SELECT proj.name AS project_name
FROM project proj
WHERE proj.id NOT IN (
    SELECT DISTINCT ir.project_id FROM inspection_report ir
    WHERE ir.status != 'DRAFT'
      AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
      AND ir.project_id IS NOT NULL
)
ORDER BY proj.name;""",
    ),

    # ── Form Answers ─────────────────────────────────────────────────────────
    (
        "what are the most common observation types across all inspections",
        """\
-- Use get_answer_stats tool for this — it aggregates answer_text for a question label
-- SQL fallback:
SELECT aa.answer_text AS observation_type, COUNT(*) AS frequency
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aq.label = 'Observation Type' AND aa.answer_text IS NOT NULL
GROUP BY aa.answer_text ORDER BY frequency DESC LIMIT 20;""",
    ),
    (
        "list all questions and answers from the most recent inspection form",
        """\
-- CRITICAL: use aa subquery — newest inspection_report may have no ai_answers (ETL gap)
SELECT aq.label AS question, aa.answer_text AS answer
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aa.inspection_report_id = (
    SELECT aa2.inspection_report_id FROM ai_answers aa2
    WHERE aa2.module_name = 'Inspection Form'
    ORDER BY aa2.submitted_on DESC LIMIT 1
)
  AND aa.module_name = 'Inspection Form'
ORDER BY aq.label LIMIT 100;""",
    ),
    (
        "show high risk observations",
        """\
-- Form answer risk — use ai_answers WHERE aq.label = 'Risk Level', NOT risk_level table
SELECT aa.inspection_id, obs.answer_text AS observation, ir.submitted_on
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN ai_answers obs ON obs.inspection_report_id = ir.id
JOIN ai_questions obs_q ON obs.element_id = obs_q.element_id
WHERE aq.label = 'Risk Level' AND aa.answer_text = 'High'
  AND obs_q.label = 'Observation'
  AND ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 100;""",
    ),
    (
        "what did inspectors write in the observation field for a specific facility",
        """\
SELECT aa.answer_text AS observation, ir.inspection_id, ir.submitted_on
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE aq.label = 'Observation'
  AND fac.name ILIKE '%facility_name%'
  AND aa.answer_text IS NOT NULL
  AND ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 100;""",
    ),
    (
        "what are the most common causes of corrective actions",
        """\
SELECT ica.cause, COUNT(*) AS frequency
FROM inspection_corrective_action ica
WHERE ica.cause IS NOT NULL
GROUP BY ica.cause ORDER BY frequency DESC LIMIT 50;""",
    ),
    (
        "what are the repetitive observations in the last 6 months",
        """\
SELECT aa.answer_text AS observation, COUNT(*) AS occurrences
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
WHERE aq.label = 'Observation'
  AND aa.answer_text IS NOT NULL
  AND ir.submitted_on >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY aa.answer_text HAVING COUNT(*) > 1
ORDER BY occurrences DESC LIMIT 20;""",
    ),
    (
        "show me questions with their scores for the last inspection by George",
        """\
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
ORDER BY aq.label;""",
    ),

    # ── ICA Risk Level ────────────────────────────────────────────────────────
    (
        "what is the risk level breakdown of all corrective actions",
        """\
-- ICA risk_level_id is UUID FK — NEVER compare to string, NEVER GROUP BY UUID
SELECT rl.name AS risk_level_name, COUNT(*) AS action_count
FROM inspection_corrective_action ica
JOIN risk_level rl ON ica.risk_level_id = rl.id
GROUP BY rl.name ORDER BY action_count DESC;""",
    ),
    (
        "how many corrective actions are high risk",
        """\
SELECT COUNT(*) AS high_risk_count
FROM inspection_corrective_action ica
JOIN risk_level rl ON ica.risk_level_id = rl.id
WHERE rl.name = 'High';""",
    ),
    (
        "show all high risk corrective actions",
        """\
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.status, rl.name AS risk_level_name,
       fac.name AS facility_name, ica.target_close_out_date
FROM inspection_corrective_action ica
JOIN risk_level rl ON ica.risk_level_id = rl.id
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE rl.name = 'High'
ORDER BY ica.target_close_out_date DESC LIMIT 100;""",
    ),
    (
        "show open corrective actions with high risk grouped by facility",
        """\
SELECT fac.name AS facility_name, COUNT(*) AS high_risk_open_count
FROM inspection_corrective_action ica
JOIN risk_level rl ON ica.risk_level_id = rl.id
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE rl.name = 'High' AND ica.status = 'OPEN'
GROUP BY fac.name ORDER BY high_risk_open_count DESC;""",
    ),

    # ── Deferred Chain ────────────────────────────────────────────────────────
    (
        "show all deferred corrective actions",
        """\
SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.status, ica.deferred_on, ica.target_close_out_date,
       fac.name AS facility_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.status = 'CLOSE_WITH_DEFERRED'
ORDER BY fac.name, ica.deferred_on DESC LIMIT 100;""",
    ),
    (
        "which facility has the most recurring deferred issues",
        """\
SELECT fac.name AS facility_name, COUNT(*) AS deferred_issue_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.status = 'CLOSE_WITH_DEFERRED'
GROUP BY fac.name ORDER BY deferred_issue_count DESC LIMIT 20;""",
    ),
    (
        "how many corrective actions have been carried forward more than once",
        """\
SELECT COUNT(*) AS multi_deferred_count FROM (
    SELECT ica.cause, fac.name
    FROM inspection_corrective_action ica
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    JOIN facility fac ON ir.facility_id = fac.id
    WHERE ica.status = 'CLOSE_WITH_DEFERRED'
    GROUP BY ica.cause, fac.name HAVING COUNT(*) > 1
) sub;""",
    ),

    # ── Scores / Trends ───────────────────────────────────────────────────────
    (
        "show inspections where the score dropped compared to the previous inspection at the same facility",
        """\
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
ORDER BY score_drop DESC LIMIT 50;""",
    ),
    (
        "which facility improved the most in score compared to last quarter",
        """\
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
ORDER BY improvement DESC LIMIT 1;""",
    ),
    (
        "which inspector has the lowest average inspection score",
        """\
SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       ROUND(AVG(ir.inspection_score)::numeric, 2) AS avg_score
FROM inspection_report ir
JOIN users u ON ir.inspector_user_id = u.id
WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
GROUP BY u.id, u.first_name, u.last_name
ORDER BY avg_score ASC LIMIT 1;""",
    ),

    # ── Multi-turn ────────────────────────────────────────────────────────────
    (
        "show me the most recently answered inspection",
        """\
-- CRITICAL: use ai_answers subquery NOT inspection_report ORDER BY submitted_on
-- The newest inspection_report row may have NO ai_answers yet (ETL gap).
-- Always find the most recent inspection WITH form data via ai_answers:
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
    WHERE aa.module_name = 'Inspection Form'
    ORDER BY aa.submitted_on DESC LIMIT 1
);
-- NEVER: ORDER BY ir.submitted_on DESC LIMIT 1 — newest ir may have no form data""",
    ),
    (
        "latest inspection with answers",
        """\
-- Use ai_answers to find most recent inspection WITH form data (not inspection_report):
SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on, ir.status
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.id = (
    SELECT aa.inspection_report_id FROM ai_answers aa
    WHERE aa.module_name = 'Inspection Form'
    ORDER BY aa.submitted_on DESC LIMIT 1
);""",
    ),
    (
        "most recent inspection that has form answers",
        """\
SELECT aa.inspection_report_id FROM ai_answers aa
WHERE aa.module_name = 'Inspection Form'
ORDER BY aa.submitted_on DESC LIMIT 1;""",
    ),

    # ── People ───────────────────────────────────────────────────────────────
    (
        "show breakdown of corrective action status by responsible party",
        """\
SELECT ica.responsible AS responsible_party,
       COUNT(*) FILTER (WHERE ica.status = 'OPEN') AS open_count,
       COUNT(*) FILTER (WHERE ica.status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS closed_count,
       COUNT(*) FILTER (WHERE ica.status = 'OVERDUE') AS overdue_count
FROM inspection_corrective_action ica
GROUP BY ica.responsible ORDER BY ica.responsible;""",
    ),
    (
        "which inspectors haven't submitted any inspections in the last 30 days",
        """\
SELECT u.first_name || ' ' || u.last_name AS inspector_name
FROM users u
WHERE u.id NOT IN (
    SELECT DISTINCT ir.inspector_user_id FROM inspection_report ir
    WHERE ir.submitted_on >= CURRENT_DATE - INTERVAL '30 days'
      AND ir.status != 'DRAFT'
      AND ir.inspector_user_id IS NOT NULL
)
ORDER BY u.first_name, u.last_name;""",
    ),

    # ── Schedule / Frequency ─────────────────────────────────────────────────
    (
        "how many inspections are overdue this quarter",
        """\
SELECT COUNT(*) AS overdue_count
FROM inspection_schedule
WHERE status = 'OVERDUE'
  AND due_date >= date_trunc('quarter', NOW())
  AND due_date < date_trunc('quarter', NOW()) + INTERVAL '3 months';""",
    ),
    (
        "show me the inspection schedule for the next 30 days",
        """\
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

ORDER BY isched.due_date ASC LIMIT 100;""",
    ),
    (
        "how many pending or scheduled inspections this month",
        """\
SELECT status, COUNT(*) AS count
FROM inspection_schedule
WHERE status IN ('PENDING', 'ONGOING')
  AND due_date >= date_trunc('month', NOW())
  AND due_date < date_trunc('month', NOW()) + INTERVAL '1 month'
GROUP BY status;""",
    ),
    (
        "what is the frequency breakdown of inspections in a cycle",
        """\
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
ORDER BY schedule_count DESC;""",
    ),
    (
        "show the inspection schedule for a specific facility",
        """\
SELECT isched.due_date, isched.status AS schedule_status,
       fd.name AS frequency_name,
       ir.inspection_id, ir.submitted_on
FROM inspection_schedule isched
JOIN inspector_portfolio_details ipd ON isched.portfolio_details_id = ipd.id
JOIN frequency_definition fd ON ipd.frequency_definition_id = fd.id
JOIN facility fac ON isched.facility_id = fac.id
LEFT JOIN inspection_report ir ON isched.inspection_report_id = ir.id
WHERE fac.name ILIKE '%facility_name%'
ORDER BY isched.due_date DESC LIMIT 50;""",
    ),

    # ── Portfolio / Inspector assignment ─────────────────────────────────────
    (
        "how many inspectors are assigned portfolios",
        """\
-- USE THIS EXACT SQL for counting inspectors — simple COUNT(DISTINCT), NO GROUP BY:
-- inspector_portfolio: id, user_id (FK→users.id), status, cycle_id
-- inspector_portfolio_details: id, portfolio_id (FK→inspector_portfolio.id), facility_id, frequency_definition_id
-- NEVER add GROUP BY for a simple count query
SELECT COUNT(DISTINCT ip.user_id) AS inspector_count
FROM inspector_portfolio ip
WHERE ip.status != 'DRAFT';""",
    ),
    (
        "which inspector has the most facilities in their portfolio",
        """\
-- USE THIS EXACT SQL for inspector facility count — GROUP BY user id (not inspector_id):
SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       COUNT(DISTINCT ipd.facility_id) AS facility_count
FROM inspector_portfolio ip
JOIN inspector_portfolio_details ipd ON ipd.portfolio_id = ip.id
JOIN users u ON ip.user_id = u.id
WHERE ip.status != 'DRAFT'
GROUP BY u.id, u.first_name, u.last_name
ORDER BY facility_count DESC LIMIT 10;""",
    ),
    (
        "list inspector portfolio assignments with facility and frequency",
        """\
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
ORDER BY u.last_name, fac.name LIMIT 100;""",
    ),

    # ── ETL boundary (phrase variants) ───────────────────────────────────────
    (
        "most recent inspection with form data available",
        """\
-- Must use ai_answers subquery — inspection_report.submitted_on may point to
-- an inspection with NO ai_answers yet (ETL gap).
SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on, ir.status
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.id = (
    SELECT aa.inspection_report_id FROM ai_answers aa
    WHERE aa.module_name = 'Inspection Form'
    ORDER BY aa.submitted_on DESC LIMIT 1
);""",
    ),
    (
        "which inspections are not yet indexed in form answers",
        """\
-- Inspections in inspection_report with no matching row in ai_answers (ETL gap):
SELECT ir.inspection_id, ir.status, ir.submitted_on,
       fac.name AS facility_name
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.id NOT IN (
    SELECT DISTINCT aa.inspection_report_id FROM ai_answers aa
    WHERE aa.inspection_report_id IS NOT NULL
)
  AND ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 50;""",
    ),

    # -- Cycle / Frequency multi-turn (ONLY for cycle queries, not portfolio counts) --
    (
        "what is the frequency of a specific inspection cycle",
        """\
-- USE THIS EXACT SQL for cycle frequency lookup — ONLY for inspection_cycle queries, not portfolio counts:
-- CYCLE FREQUENCY PATH: ic.id -> ip(ip.cycle_id=ic.id) -> ipd(ipd.portfolio_id=ip.id) -> fd
-- NEVER: ipd.cycle_id  |  completed_cycle = status='COMPLETED'
-- NOTE: use DISTINCT subquery shape, not GROUP BY, for this lookup
SELECT DISTINCT fd.name AS frequency_name,
       fd.repeat_count, fd.repeat_interval, fd.repeat_unit
FROM inspection_cycle ic
JOIN inspector_portfolio ip          ON ip.cycle_id = ic.id
JOIN inspector_portfolio_details ipd ON ipd.portfolio_id = ip.id
JOIN frequency_definition fd         ON fd.id = ipd.frequency_definition_id
WHERE ic.id = '<CYCLE_UUID>'
  AND ic.status = 'COMPLETED';
""",
    ),
    (
        "frequency of the most recently completed inspection cycle",
        """\
-- USE THIS EXACT SQL for most-recent-cycle frequency — ONLY for cycle queries:
-- completed_cycle = status = 'COMPLETED' (not CLOSED, not status != 'DRAFT')
-- Join: ic -> ip(ip.cycle_id=ic.id) -> ipd(ipd.portfolio_id=ip.id) -> fd
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
    ),
    (
        "most recently completed inspection cycle",
        """\
-- completed_cycle: status = 'COMPLETED' only
SELECT ic.id AS inspection_cycle_id,
       TO_CHAR(ic.start_date, 'Month YYYY') AS month,
       ic.status, ic.start_date, ic.end_date, ic.modified_on
FROM inspection_cycle ic
WHERE ic.status = 'COMPLETED'
ORDER BY ic.end_date DESC LIMIT 1;
""",
    ),

    # ── FM Manager / CAPEX Leaderboard ───────────────────────────────────────
    (
        "top 5 FM managers by capex generated leaderboard",
        """-- USE THIS EXACT SQL for CAPEX GENERATOR leaderboard (top 5):
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
WHERE ica.capex > 0
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_generated DESC
LIMIT 5;""",
    ),
    (
        "top 5 FM managers by capex converted approved leaderboard",
        """-- USE THIS EXACT SQL for CAPEX CONVERTER leaderboard (top 5):
-- CAPEX Converter = FM managers ranked by capex that reached ACCEPTED status
-- Generator vs Converter: Generator=total raised, Converter=total approved (ACCEPTED)
-- FM Manager = inspectee_user_id (NOT inspector_user_id)
SELECT u.first_name || ' ' || u.last_name AS fm_manager,
       SUM(ica.capex) AS capex_converted,
       COUNT(DISTINCT ica.corrective_action_id) AS actions_converted
FROM inspection_report ir
JOIN users u ON ir.inspectee_user_id = u.id
JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id
WHERE ica.capex > 0
  AND ica.capex_status = 'ACCEPTED'
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_converted DESC
LIMIT 5;""",
    ),
    (
        "which corrective actions are awaiting capex approval",
        """-- capex_status IS NULL = awaiting approval (not yet actioned by approver)
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
WHERE ica.capex > 0
  AND ica.capex_status IS NULL
ORDER BY ica.capex DESC
LIMIT 50;""",
    ),

    # ── Cost / expenditure queries ────────────────────────────────────────────
    (
        "which corrective action has the highest cost capex opex",
        """-- USE THIS EXACT SQL for highest cost corrective action:
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
WHERE (ica.capex > 0 OR ica.opex > 0)
ORDER BY total_cost DESC
LIMIT 1;""",
    ),

    # ── Financial / CAPEX / OPEX ─────────────────────────────────────────────
    (
        "total capex generated converted lost and opex incurred",
        """-- USE THIS EXACT SQL — ONE ROW scalar summary, NO GROUP BY, NO LEFT JOIN:
-- Scalar subqueries avoid row multiplication and GROUP BY problems
SELECT
    (SELECT COALESCE(SUM(capex), 0) FROM inspection_corrective_action
     WHERE capex > 0)                                                    AS capex_generated,
    (SELECT COALESCE(SUM(po_amount), 0) FROM po_details)                  AS capex_converted,
    (SELECT COALESCE(SUM(capex), 0) FROM inspection_corrective_action
     WHERE capex_status = 'REJECTED')                                     AS capex_lost,
    (SELECT COALESCE(SUM(pr_amount), 0) FROM pr_details)                  AS opex_incurred;""",
    ),
    (
        "top inspectors by capex generated converted",
        """-- USE THIS EXACT SQL for inspector CAPEX leaderboard:
-- Inspector = inspector_user_id on inspection_report (NOT inspectee_user_id)
-- CAPEX Generated = SUM(ica.capex) | CAPEX Converted = SUM(po.po_amount)
-- Use subquery to avoid row multiplication from multiple POs per ICA
SELECT 
    u.first_name || ' ' || u.last_name AS inspector_name,
    COUNT(DISTINCT ica.id) AS ca_count,
    SUM(ica.capex) AS capex_generated,
    COALESCE(SUM(po.po_amount), 0) AS capex_converted,
    SUM(CASE WHEN ica.capex_status = 'REJECTED' THEN ica.capex ELSE 0 END) AS capex_lost
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN users u ON ir.inspector_user_id = u.id
LEFT JOIN po_details po ON po.mitigative_action_uuid = ica.id
WHERE ica.capex > 0
GROUP BY u.id, u.first_name, u.last_name
ORDER BY capex_generated DESC
LIMIT 10;""",
    ),
    (
        "opex incurred by facility from pr details",
        """-- OPEX Incurred = SUM(pr.pr_amount) from pr_details — actual internal spend
-- pr_details JOIN: ON pr.mitigative_action_uuid = ica.id
SELECT 
    fac.name AS facility_name,
    COUNT(DISTINCT ica.id) AS ca_count,
    SUM(ica.opex) AS opex_estimated,
    COALESCE(SUM(pr.pr_amount), 0) AS opex_incurred
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
LEFT JOIN pr_details pr ON pr.mitigative_action_uuid = ica.id
WHERE ica.opex > 0 OR pr.id IS NOT NULL
GROUP BY fac.id, fac.name
ORDER BY opex_incurred DESC
LIMIT 20;""",
    ),
    (
        "capex lost rejected corrective actions",
        """-- CAPEX Lost = corrective actions where capex_status = 'REJECTED'
-- These had POs raised but client ultimately rejected
SELECT 
    ica.corrective_action_id,
    ica.capex AS capex_quoted,
    COALESCE(SUM(po.po_amount), 0) AS po_amount_raised,
    fac.name AS facility_name,
    u.first_name || ' ' || u.last_name AS inspector_name,
    ica.modified_on::date AS rejected_on
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN users u ON ir.inspector_user_id = u.id
LEFT JOIN po_details po ON po.mitigative_action_uuid = ica.id
WHERE ica.capex_status = 'REJECTED'
GROUP BY ica.id, ica.corrective_action_id, ica.capex, fac.name,
         u.first_name, u.last_name, ica.modified_on
ORDER BY ica.capex DESC
LIMIT 50;""",
    ),

    (
        "total opex incurred from purchase requisitions pr details",
        """-- USE THIS EXACT SQL — total OPEX incurred via actual PR spend:
-- IMPORTANT: ica.opex is the ESTIMATE, pr.pr_amount is the ACTUAL spend
-- Always use pr_details.pr_amount for 'opex incurred' questions
SELECT 
    COALESCE(SUM(pr.pr_amount), 0) AS total_opex_incurred,
    COUNT(DISTINCT pr.id)           AS pr_count
FROM pr_details pr
JOIN inspection_corrective_action ica ON ica.id = pr.mitigative_action_uuid;""",
    ),
    (
        "total purchase order amount raised this year from po details",
        """-- USE THIS EXACT SQL — returns ONE ROW with total PO amount:
-- DO NOT list individual rows — return SUM as a scalar result
-- po_details.po_date is the date the PO was raised
SELECT 
    SUM(po.po_amount)    AS total_po_amount,
    COUNT(DISTINCT po.id) AS po_count
FROM po_details po
WHERE EXTRACT(YEAR FROM po.po_date) = EXTRACT(YEAR FROM CURRENT_DATE);""",
    ),

    (
        "total purchase order amount po amount raised this year",
        """-- USE THIS EXACT SQL — returns ONE ROW total PO amount for this year:
-- po_details table contains purchase orders (CAPEX payments from client)
-- DO NOT list individual rows — SUM into a single scalar result
SELECT 
    SUM(po.po_amount)     AS total_po_amount,
    COUNT(DISTINCT po.id) AS po_count
FROM po_details po
WHERE EXTRACT(YEAR FROM po.po_date) = EXTRACT(YEAR FROM CURRENT_DATE);""",
    ),
    (
        "show high risk corrective actions from a specific cycle scoped to cycle",
        """-- USE THIS EXACT SQL for high risk CAs scoped to a cycle:
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
ORDER BY ica.target_close_out_date DESC LIMIT 100;""",
    ),
    (
        "show all inspectors in a cycle facilities in a specific cycle",
        """-- USE THIS EXACT SQL for inspectors/facilities within a cycle:
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
LIMIT 50;""",
    ),

    (
        "give me a breakdown of this quarter quarterly summary",
        """-- USE THIS EXACT SQL for quarterly summary — single aggregated row:
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
  AND date_trunc('quarter', ir.submitted_on) = date_trunc('quarter', CURRENT_DATE);""",
    ),

    (
        "total capex generated converted lost financial summary",
        """-- USE THIS EXACT SQL — ONE ROW scalar summary, NO GROUP BY:
SELECT
    (SELECT COALESCE(SUM(capex), 0) FROM inspection_corrective_action
     WHERE capex > 0)                                                    AS capex_generated,
    (SELECT COALESCE(SUM(po_amount), 0) FROM po_details)                  AS capex_converted,
    (SELECT COALESCE(SUM(capex), 0) FROM inspection_corrective_action
     WHERE capex_status = 'REJECTED')                                     AS capex_lost,
    (SELECT COALESCE(SUM(pr_amount), 0) FROM pr_details)                  AS opex_incurred;""",
    ),

]