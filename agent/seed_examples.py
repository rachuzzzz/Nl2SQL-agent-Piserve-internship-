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
-- inspection_report: use ir.inspection_id (VARCHAR) for display, ir.id (UUID) only in JOINs/subqueries
-- ICA join: JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id  (NOT ir.inspection_id = ica.inspection_id)
-- completed inspection = status IN ('CLOSED','SUBMITTED') | responsible: 'CLIENT','INTERNAL_OPERATIONS','SUB_CONTRACTOR'
-- NEVER: ir.id in SELECT list | NEVER: ica.risk_level_id compared to string | NEVER: GROUP BY ica.risk_level_id
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
-- Single-row FILTER form — NOT GROUP BY
SELECT
  COUNT(*) FILTER (WHERE date_trunc('month', ir.submitted_on) = date_trunc('month', NOW())) AS this_month_count,
  COUNT(*) FILTER (WHERE date_trunc('month', ir.submitted_on) = date_trunc('month', NOW()-INTERVAL '1 month')) AS last_month_count
FROM inspection_report ir WHERE ir.status IN ('CLOSED','SUBMITTED');""",
    ),
    (
        "how many corrective actions were raised vs closed this quarter",
        """\
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
SELECT TO_CHAR(date_trunc('quarter', ica.created_on), '"Q"Q YYYY') AS quarter,
       COUNT(*) FILTER (WHERE ica.status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS closed_count,
       ROUND(
           100.0 * COUNT(*) FILTER (WHERE ica.status IN ('CLOSED','CLOSE_WITH_DEFERRED'))
           / NULLIF(COUNT(*), 0), 2
       ) AS closure_rate_pct
FROM inspection_corrective_action ica
WHERE ica.created_on >= date_trunc('quarter', NOW() - INTERVAL '3 months')
GROUP BY date_trunc('quarter', ica.created_on)
ORDER BY quarter;""",
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
);""",
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
]