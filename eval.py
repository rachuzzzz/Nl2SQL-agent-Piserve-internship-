"""
NL2SQL Agent — Evaluation Test Set (59 queries)
================================================
Usage:
  python3 eval.py
  python3 eval.py --model-pair "llama3.1:8b + deepseek:6.7b"
  python3 eval.py --skip-multiturn
  python3 eval.py --category "Aggregate"
  python3 eval.py --output my_results.json

Scoring notes
─────────────
keyword_pct : percentage of expected_contains keywords found in the answer.
              None when expected_contains is empty (vacuous pass removed).
empty_result: True when the answer matches known "no data" phrases.
              Questions marked expected_nonempty=True are force-failed when
              the result is empty, even if success=True.
real_pass   : True only when success=True AND not a forced-empty-fail.
              Use this column for the headline pass rate.
"""

TEST_SET = [
    # ── Aggregates ───────────────────────────────────────────────────────────
    {
        "id": "AGG-01", "category": "Aggregate",
        "question": "what is the average inspection score",
        "expected_contains": ["86", "avg", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Baseline",
    },
    {
        "id": "AGG-02", "category": "Aggregate",
        "question": "how many inspections were completed this year",
        "expected_contains": ["inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Date filter + count",
    },
    {
        "id": "AGG-03", "category": "Aggregate",
        "question": "what is the total capex and opex across all corrective actions",
        "expected_contains": ["capex", "opex"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Two aggregates",
    },
    {
        "id": "AGG-04", "category": "Aggregate",
        "question": "how many open corrective actions are there",
        "expected_contains": ["open"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Status filter count",
    },
    {
        "id": "AGG-05", "category": "Aggregate",
        "question": "how many inspections were completed this month vs last month",
        "expected_contains": ["month"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "date_trunc comparison",
    },
    {
        "id": "AGG-06", "category": "Aggregate",
        "question": "how many inspections are overdue this quarter",
        "expected_contains": ["overdue"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Quarter + date filter",
    },
    {
        "id": "AGG-07", "category": "Aggregate",
        "question": "how many corrective actions were raised vs closed this quarter",
        "expected_contains": ["open", "closed"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "FILTER clause",
    },
    {
        "id": "AGG-08", "category": "Aggregate",
        "question": "what is the average inspection duration in hours",
        "expected_contains": ["hour"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "start_date_time / end_date_time",
    },
    {
        "id": "AGG-09", "category": "Aggregate",
        "question": "what percentage of inspections resulted in at least one corrective action",
        "expected_contains": ["%"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "LEFT JOIN + COUNT DISTINCT ~66.8%",
    },
    {
        "id": "AGG-10", "category": "Aggregate",
        "question": "what is the average number of observations per inspection",
        "expected_contains": ["4", "observation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Subquery AVG ~4.2",
    },
    # ── Filters ──────────────────────────────────────────────────────────────
    {
        "id": "FLT-01", "category": "Filter",
        "question": "show the last 5 inspections by submission date",
        "expected_contains": ["inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ORDER BY + LIMIT",
    },
    {
        "id": "FLT-02", "category": "Filter",
        "question": "show all overdue corrective actions",
        "expected_contains": ["action"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Date comparison filter",
    },
    {
        "id": "FLT-03", "category": "Filter",
        "question": "show high risk observations",
        "expected_contains": ["high", "observation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "answer_text filter",
    },
    {
        "id": "FLT-04", "category": "Filter",
        "question": "which inspections were returned for modification",
        "expected_contains": ["inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Enum status filter",
    },
    {
        "id": "FLT-05", "category": "Filter",
        "question": "show me inspections with a score below 70",
        "expected_contains": ["score", "inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Numeric threshold ~16",
    },
    {
        "id": "FLT-06", "category": "Filter",
        "question": "how many inspections were done on weekends",
        "expected_contains": ["2", "inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "EXTRACT(ISODOW)",
    },
    {
        "id": "FLT-07", "category": "Filter",
        "question": "which facilities haven't been inspected in the last 30 days",
        "expected_contains": ["facility"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "NOT IN subquery",
    },
    # ── Joins ─────────────────────────────────────────────────────────────────
    {
        "id": "JOI-01", "category": "Join",
        "question": "show inspection scores grouped by facility",
        "expected_contains": ["facility", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "GROUP BY + JOIN",
    },
    {
        "id": "JOI-02", "category": "Join",
        "question": "which inspector has conducted the most inspections this year",
        "expected_contains": ["inspector", "neenu"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "JOIN users + COUNT",
    },
    {
        "id": "JOI-03", "category": "Join",
        "question": "show average inspection score by inspection type",
        "expected_contains": ["type", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "JOIN inspection_type",
    },
    {
        "id": "JOI-04", "category": "Join",
        "question": "show corrective actions with their inspection facility and inspector",
        "expected_contains": ["facility", "inspector"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "4-table join",
    },
    {
        "id": "JOI-05", "category": "Join",
        "question": "what is the current open corrective action count by facility",
        "expected_contains": ["facility", "open"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Correct JOIN direction",
    },
    {
        "id": "JOI-06", "category": "Join",
        "question": "which facility has the highest number of overdue corrective actions",
        "expected_contains": ["facility", "overdue"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ADJD AA -Al Roya: 45",
    },
    {
        "id": "JOI-07", "category": "Join",
        "question": "which client has the most overdue corrective actions",
        "expected_contains": ["client"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Kingfield: 72",
    },
    {
        "id": "JOI-08", "category": "Join",
        "question": "what is the total capex and opex by facility",
        "expected_contains": ["facility", "capex", "opex"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ica->ir->facility chain",
    },
    {
        "id": "JOI-09", "category": "Join",
        "question": "show me the top 5 facilities by average inspection score",
        "expected_contains": ["facility", "score", "bridges"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ORDER BY avg DESC LIMIT 5",
    },
    {
        "id": "JOI-10", "category": "Join",
        "question": "show facilities where every inspection this year scored above 90",
        "expected_contains": ["facility", "score"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "HAVING MIN(score) > 90",
    },
    {
        "id": "JOI-11", "category": "Join",
        "question": "show facilities that have been inspected more than 5 times this year",
        "expected_contains": ["facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "HAVING COUNT > 5",
    },
    {
        "id": "JOI-12", "category": "Join",
        "question": "which projects have not had any inspections this year",
        "expected_contains": ["project"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "NOT IN subquery ~42 projects",
    },
    # ── Form Answers ──────────────────────────────────────────────────────────
    {
        "id": "ANS-01", "category": "Form Answers",
        "question": "what are the most common observation types across all inspections",
        "expected_contains": ["deviation", "observation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "get_answer_stats Deviation:359",
    },
    {
        "id": "ANS-02", "category": "Form Answers",
        "question": "list all questions and answers from the most recent inspection form",
        "expected_contains": ["question", "answer"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "module_name filter regression",
    },
    {
        "id": "ANS-03", "category": "Form Answers",
        "question": "show all risk levels from the last 5 inspections",
        "expected_contains": ["risk"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "May be empty if DRAFT",
    },
    {
        "id": "ANS-04", "category": "Form Answers",
        "question": "what did inspectors write in the observation field for Al Ghadeer",
        "expected_contains": ["observation", "ghadeer"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Cross-domain 4-table JOIN",
    },
    {
        "id": "ANS-05", "category": "Form Answers",
        "question": "what are the most common causes of corrective actions",
        "expected_contains": ["cause"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ica.cause NOT form answers",
    },
    {
        "id": "ANS-06", "category": "Form Answers",
        "question": "what are the most common risk levels across all inspections",
        "expected_contains": ["high", "medium", "low"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "get_answer_stats High:230",
    },
    {
        "id": "ANS-07", "category": "Form Answers",
        "question": "what are the repetitive observations in the last 6 months",
        "expected_contains": ["observation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "GROUP BY HAVING COUNT > 1",
    },
    {
        "id": "ANS-08", "category": "Form Answers",
        "question": "which observation appears most frequently as high risk",
        "expected_contains": ["observation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Subquery: risk High -> obs text",
    },
    # ── Scores ────────────────────────────────────────────────────────────────
    {
        "id": "SCR-01", "category": "Scores",
        "question": "show me questions with their scores for the last inspection by George",
        "expected_contains": ["score", "george"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ai_answers.score + users JOIN",
    },
    {
        "id": "SCR-02", "category": "Scores",
        "question": "what is the average score for the risk level question",
        "expected_contains": ["score", "risk"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "get_answer_stats avg 3.24",
    },
    {
        "id": "SCR-03", "category": "Scores",
        "question": "which inspection had the highest total score",
        "expected_contains": ["score", "inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "SUM ai_answers.score = 102",
    },
    {
        "id": "SCR-04", "category": "Scores",
        "question": "which inspection type has the highest average score",
        "expected_contains": ["type", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Hygiene and grooming: 94",
    },
    {
        "id": "SCR-05", "category": "Scores",
        "question": "which inspector has the lowest average inspection score",
        "expected_contains": ["inspector", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "George Inspector: 69.5",
    },
    # ── Trends ────────────────────────────────────────────────────────────────
    {
        "id": "TRD-01", "category": "Trends",
        "question": "how has the average inspection score changed month by month this year",
        "expected_contains": ["january", "february"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "TO_CHAR + date_trunc",
    },
    {
        "id": "TRD-02", "category": "Trends",
        "question": "are corrective actions being closed faster or slower compared to last quarter",
        "expected_contains": ["quarter", "closed"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "close_on arithmetic blocked by validator — answer uses status-based closure rate",
    },
    {
        "id": "TRD-03", "category": "Trends",
        "question": "which month had the most high risk observations this year",
        "expected_contains": ["january", "month"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "GROUP BY month Jan:73",
    },
    {
        "id": "TRD-04", "category": "Trends",
        "question": "show inspections where the score dropped compared to the previous inspection at the same facility",
        "expected_contains": ["score", "facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "LAG() PARTITION BY facility_id",
    },
    {
        "id": "TRD-05", "category": "Trends",
        "question": "which facility improved the most in score compared to last quarter",
        "expected_contains": ["facility"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "LAG() quarter filter",
    },
    {
        "id": "TRD-06", "category": "Trends",
        "question": "which inspection type has the most corrective actions raised",
        "expected_contains": ["type", "action"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Regulatory/Insurance:103",
    },
    # ── People ────────────────────────────────────────────────────────────────
    {
        "id": "PPL-01", "category": "People",
        "question": "which inspector has conducted the most inspections this year",
        "expected_contains": ["inspector", "neenu"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "neenu extinsp1:49",
    },
    {
        "id": "PPL-02", "category": "People",
        "question": "show breakdown of corrective action status by responsible party",
        "expected_contains": ["client", "internal"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "responsible is ENUM not FK",
    },
    {
        "id": "PPL-03", "category": "People",
        "question": "which inspectors haven't submitted any inspections in the last 30 days",
        "expected_contains": ["inspector"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "NOT IN subquery",
    },
    # ── Multi-turn ────────────────────────────────────────────────────────────
    {
        "id": "MT-01", "category": "Multi-turn",
        "question": "show me the most recently answered inspection",
        "expected_contains": ["facility", "inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Turn 1 — uses ai_answers.submitted_on so the result always has form data",
    },
    {
        "id": "MT-02", "category": "Multi-turn",
        "question": "who inspected it",
        "expected_contains": ["inspector"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Turn 2 uses facility context",
        "depends_on": "MT-01",
    },
    {
        "id": "MT-03", "category": "Multi-turn",
        "question": "fetch all questions and answers she filled for that site",
        "expected_contains": ["question", "answer"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Turn 3 inspection_id context (12584 regression)",
        "depends_on": "MT-02",
    },
    {
        "id": "MT-04", "category": "Multi-turn",
        "question": "show more",
        "expected_contains": [],
        "expected_nonempty": True,
        "expected_llm_calls": 0, "notes": "Pagination — must return Q&A rows not inspector names",
        "depends_on": "MT-03",
    },
    # ── ICA Risk Level — Problem 2 fix ───────────────────────────────────────
    {
        "id": "ICA-01", "category": "ICA Risk Level",
        "question": "show all high risk corrective actions",
        "expected_contains": ["high", "risk"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "P2 fix: JOIN risk_level lookup, not WHERE risk_level_id='High'",
    },
    {
        "id": "ICA-02", "category": "ICA Risk Level",
        "question": "what is the risk level breakdown of all corrective actions",
        "expected_contains": ["high", "medium"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "P2 fix: GROUP BY rl.name — must show human-readable names not UUIDs",
    },
    {
        "id": "ICA-03", "category": "ICA Risk Level",
        "question": "how many corrective actions are high risk",
        "expected_contains": ["high"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "P2 fix: COUNT with JOIN risk_level — before fix returned 0",
    },
    {
        "id": "ICA-04", "category": "ICA Risk Level",
        "question": "show open corrective actions with high risk grouped by facility",
        "expected_contains": ["facility", "high"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "P2 fix: 3-table join ica + risk_level + ir + facility",
    },
    # ── Deferred Chain — Problem 5 fix ───────────────────────────────────────
    {
        "id": "DEF-01", "category": "Deferred Chain",
        "question": "show all deferred corrective actions",
        "expected_contains": ["deferred", "action"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "P5 fix: WHERE status = 'CLOSE_WITH_DEFERRED'",
    },
    {
        "id": "DEF-02", "category": "Deferred Chain",
        "question": "which issues recur most at Al Ghadeer",
        "expected_contains": ["cause"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "P5 fix: CLOSE_WITH_DEFERRED at facility — cause values may be short codes",
    },
    {
        "id": "DEF-03", "category": "Deferred Chain",
        "question": "which facility has the most recurring deferred issues",
        "expected_contains": ["facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "P5 fix: GROUP BY fac.name WHERE CLOSE_WITH_DEFERRED",
    },
    {
        "id": "DEF-04", "category": "Deferred Chain",
        "question": "how many corrective actions have been carried forward more than once",
        "expected_contains": ["deferred"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "P5 fix: HAVING COUNT > 1 on CLOSE_WITH_DEFERRED per cause",
    },
    {
        "id": "DEF-05", "category": "Deferred Chain",
        "question": "show the history of deferred actions at Al Ghadeer facility",
        "expected_contains": ["action"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "P5 fix: ordered deferred chain with dates at a specific facility",
    },
    # ── Edge Cases ────────────────────────────────────────────────────────────
    {
        "id": "EDG-01", "category": "Edge Case",
        "question": "show all high risk observations this month",
        "expected_contains": ["high", "observation"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "Answer + date filter",
    },
    {
        "id": "EDG-02", "category": "Edge Case",
        "question": "how many questions are in the inspection form",
        "expected_contains": ["240", "question"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "COUNT ai_questions = 240",
    },
    {
        "id": "EDG-03", "category": "Edge Case",
        "question": "show corrective actions where responsible is client",
        "expected_contains": ["client", "action"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Enum value filter",
    },
    {
        "id": "EDG-04", "category": "Edge Case",
        "question": "xyz abc nonsense query that makes no sense",
        "expected_contains": ["not", "don"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "Garbage input graceful decline",
    },

    # ── Frequency / Schedule ──────────────────────────────────────────────────
    {
        "id": "FREQ-01", "category": "Schedule",
        "question": "how many inspections are pending this quarter",
        "expected_contains": ["pending"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "inspection_schedule status=PENDING",
    },
    {
        "id": "FREQ-02", "category": "Schedule",
        "question": "what is the most common inspection frequency in our portfolio",
        "expected_contains": ["daily", "monthly", "weekly", "frequency", "quarterly", "annual"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "frequency_definition → inspector_portfolio_details join",
    },
    {
        "id": "FREQ-03", "category": "Schedule",
        "question": "which facilities are scheduled for daily inspections",
        "expected_contains": ["daily", "facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "frequency_definition.name = 'Daily' → inspector_portfolio_details → facility",
    },
    {
        "id": "FREQ-04", "category": "Schedule",
        "question": "show me the inspection schedule for the next 30 days",
        "expected_contains": ["facility", "due"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "inspection_schedule due_date BETWEEN NOW AND NOW+30d",
    },
    {
        "id": "FREQ-05", "category": "Schedule",
        "question": "how many completed vs overdue inspections in the schedule this year",
        "expected_contains": ["completed", "overdue"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "inspection_schedule GROUP BY status filter this year",
    },

    # ── Cross-domain Complexity ───────────────────────────────────────────────
    {
        "id": "CROSS-01", "category": "Cross-domain",
        "question": "which facilities have the most overdue corrective actions and the lowest inspection scores",
        "expected_contains": ["facility", "overdue", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 2,
        "notes": "ICA OVERDUE count + avg inspection_score, both joined to facility",
    },
    {
        "id": "CROSS-02", "category": "Cross-domain",
        "question": "which inspection types generate the most high risk corrective actions",
        "expected_contains": ["inspection", "high", "risk"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "inspection_type → ir → ica → risk_level JOIN chain",
    },
    {
        "id": "CROSS-03", "category": "Cross-domain",
        "question": "list facilities that were inspected this year but have zero corrective actions",
        "expected_contains": ["facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "LEFT JOIN ica WHERE ica.id IS NULL",
    },
    {
        "id": "CROSS-04", "category": "Cross-domain",
        "question": "which facilities have high risk observations but no corrective actions raised",
        "expected_contains": ["facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 2,
        "notes": "ai_answers risk_level=High LEFT JOIN ica IS NULL",
    },

    # ── ETL Boundary Awareness ────────────────────────────────────────────────
    {
        "id": "ETL-01", "category": "ETL Boundary",
        "question": "show me the most recent inspection that has form data available",
        "expected_contains": ["inspection", "facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Must use aa subquery NOT ir.submitted_on — ETL gap awareness",
    },
    {
        "id": "ETL-02", "category": "ETL Boundary",
        "question": "list the last 10 inspections that have form answers indexed",
        "expected_contains": ["inspection", "facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "SELECT DISTINCT inspection_report_id FROM ai_answers ORDER BY submitted_on DESC LIMIT 10",
    },
    {
        "id": "ETL-03", "category": "ETL Boundary",
        "question": "how many inspections have been submitted but not yet indexed with form answers",
        "expected_contains": [],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "ir NOT IN (SELECT DISTINCT inspection_report_id FROM ai_answers) — may be 0 or more",
    },

    # ── Portfolio / Assignment ────────────────────────────────────────────────
    {
        "id": "PORT-01", "category": "Portfolio",
        "question": "how many inspectors are currently assigned portfolios",
        "expected_contains": ["inspector"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "COUNT DISTINCT inspector_portfolio.inspector_id",
    },
    {
        "id": "PORT-02", "category": "Portfolio",
        "question": "which inspector has the most facilities in their portfolio",
        "expected_contains": ["inspector", "facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "inspector_portfolio_details GROUP BY portfolio → inspector",
    },
    {
        "id": "PORT-03", "category": "Portfolio",
        "question": "what facilities are assigned to each inspection frequency type",
        "expected_contains": ["facility", "frequency", "daily", "monthly", "weekly", "quarterly", "annual"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "frequency_definition JOIN inspector_portfolio_details JOIN facility",
    },

    # ── Conversational / Natural Phrasing ────────────────────────────────────
    {
        "id": "CONV-01", "category": "Conversational",
        "question": "show me the worst performing facilities",
        "expected_contains": ["facility", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ORDER BY avg_score ASC — natural phrasing",
    },
    {
        "id": "CONV-02", "category": "Conversational",
        "question": "what needs attention right now",
        "expected_contains": ["overdue", "risk", "corrective", "facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 2,
        "notes": "Ambiguous — expect overdue CA or high risk observations",
    },
    {
        "id": "CONV-03", "category": "Conversational",
        "question": "how are we doing overall",
        "expected_contains": ["score", "inspection", "corrective", "average"],
        "expected_nonempty": True,
        "expected_llm_calls": 2, "notes": "High-level summary — avg score or completion rate",
    },
    {
        "id": "CONV-04", "category": "Conversational",
        "question": "any critical issues I should know about",
        "expected_contains": ["high", "risk", "overdue", "corrective"],
        "expected_nonempty": True,
        "expected_llm_calls": 2, "notes": "Should surface high risk + overdue items",
    },
    {
        "id": "CONV-05", "category": "Conversational",
        "question": "give me a breakdown of this quarter",
        "expected_contains": ["quarter", "inspection", "corrective"],
        "expected_nonempty": True,
        "expected_llm_calls": 2, "notes": "Quarter summary — inspections done, CAs raised",
    },

    # ── Comparison / Ranking ─────────────────────────────────────────────────
    {
        "id": "COMP-01", "category": "Comparison",
        "question": "which month had the best average inspection score this year",
        "expected_contains": ["month", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ORDER BY avg_score DESC LIMIT 1 grouped by month",
    },
    {
        "id": "COMP-02", "category": "Comparison",
        "question": "rank inspection types by their average corrective actions per inspection",
        "expected_contains": ["inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "GROUP BY inspection_type ORDER BY avg_ca DESC",
    },
    {
        "id": "COMP-03", "category": "Comparison",
        "question": "which inspection type has the highest corrective action rate",
        "expected_contains": ["inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "CA count / inspection count by type",
    },
    {
        "id": "COMP-04", "category": "Comparison",
        "question": "compare overdue corrective action counts between CLIENT and INTERNAL_OPERATIONS",
        "expected_contains": ["client", "internal"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "FILTER aggregate by responsible enum — two-column comparison",
    },

    # ── Extended Aggregates ───────────────────────────────────────────────────
    {
        "id": "AGG-EXT-01", "category": "Extended Aggregate",
        "question": "what is the average number of corrective actions per inspection",
        "expected_contains": ["average", "avg", "corrective"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Subquery: COUNT per inspection, then AVG",
    },
    {
        "id": "AGG-EXT-02", "category": "Extended Aggregate",
        "question": "how many corrective actions were closed on time vs late",
        "expected_contains": ["closed", "time", "late", "on_time", "overdue"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "FILTER: status=CLOSED AND close_on <= target vs > target",
    },
    {
        "id": "AGG-EXT-03", "category": "Extended Aggregate",
        "question": "what percentage of corrective actions are currently overdue",
        "expected_contains": ["overdue", "%", "percent"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "OVERDUE / total * 100",
    },
    {
        "id": "AGG-EXT-04", "category": "Extended Aggregate",
        "question": "how many facilities have had zero inspections this year",
        "expected_contains": ["facility", "zero", "no inspection", "0"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "facility LEFT JOIN ir WHERE EXTRACT(YEAR)... IS NULL — zero-inspection facilities",
    },
    {
        "id": "AGG-EXT-05", "category": "Extended Aggregate",
        "question": "what is the total number of observations recorded across all inspections",
        "expected_contains": ["observation", "total", "count"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "COUNT ai_answers WHERE label='Observation'",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

# Phrases that indicate the agent returned no usable data.
# Used to detect false-positive successes.
_EMPTY_PHRASES = (
    "the query returned no results",
    "no answers found",
    "no results found",
    "there were no",
    "there is no data",
    "no data to calculate",
    "no inspections found",
    "no matching",
    "could not find",
    "found 0 result",   # exact "found 0 result" — avoids matching "found 10 result(s)"
    "returned 0",
)


def _is_empty_result(answer: str) -> bool:
    al = answer.lower()
    return any(p in al for p in _EMPTY_PHRASES)


# ── Runner ───────────────────────────────────────────────────────────────────

import sys, os, time, json, argparse
from datetime import datetime
from collections import defaultdict


def run_eval(model_pair_label="llama3.1:8b + deepseek:6.7b",
             skip_multiturn=False, category_filter=None, output_file=None):

    from agent.orchestrator import AgentOrchestrator, ConversationSession
    from dotenv import load_dotenv
    load_dotenv()

    tests = TEST_SET
    if category_filter:
        tests = [t for t in tests if t["category"].lower() == category_filter.lower()]
    if skip_multiturn:
        tests = [t for t in tests if t["category"] != "Multi-turn"]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  InspectAI NL2SQL Evaluation — {model_pair_label}")
    print(f"  {len(tests)} questions  |  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{sep}\n")

    orch    = AgentOrchestrator.from_env()
    results = []
    session = ConversationSession()

    for i, test in enumerate(tests):
        cat  = test["category"]
        qid  = test["id"]
        q    = test["question"]
        print(f"[{i+1:02d}/{len(tests)}] {qid} ({cat})  Q: {q[:65]}")
        t0 = time.time()
        try:
            result = orch.query(q, session=session if cat == "Multi-turn" else None)
            wall   = time.time() - t0
            al     = result.answer.lower()

            # ── keyword scoring ──────────────────────────────────────────────
            keywords = test["expected_contains"]
            kt = len(keywords)
            kh = sum(1 for kw in keywords if kw.lower() in al)
            # None when no keywords defined — avoids vacuous 100%
            kp = round(kh / kt * 100) if kt else None

            # ── sql / parse error counting ───────────────────────────────────
            se = sum(1 for s in result.steps
                     if s.tool == "execute_sql" and not s.result.get("success"))
            pe = sum(1 for s in result.steps if s.tool == "[parse_error]")

            # ── empty-result detection ───────────────────────────────────────
            empty = _is_empty_result(result.answer)

            # Force fail when we know the result must not be empty
            forced_fail = bool(test.get("expected_nonempty") and empty)
            real_pass   = result.success and not forced_fail

            row = dict(
                id=qid,
                category=cat,
                question=q,
                model_pair=model_pair_label,
                wall_s=round(wall, 2),
                llm_calls=result.stats.total_llm_calls,
                prompt_tokens=result.stats.total_prompt_tokens,
                completion_tokens=result.stats.total_completion_tokens,
                sql_errors=se,
                parse_errors=pe,
                keyword_hits=f"{kh}/{kt}" if kt else "n/a",
                keyword_pct=kp,
                empty_result=empty,
                forced_fail=forced_fail,
                real_pass=real_pass,
                answer_preview=result.answer[:250],
                success=result.success,
                manual_score=None,
                notes=test["notes"],
            )
            results.append(row)

            # ── per-question console line ────────────────────────────────────
            perf = ("🟢" if result.stats.total_llm_calls <= 1
                    else "🟡" if result.stats.total_llm_calls <= 3 else "🔴")
            kw_str = f"{kh}/{kt}" if kt else "n/a"
            status = "✅" if real_pass else ("❌ EMPTY" if forced_fail else "❌")
            print(f"         {perf} {wall:.1f}s | {result.stats.total_llm_calls} calls | "
                  f"kw {kw_str} | SQL {'OK' if se == 0 else str(se) + ' err'} | {status}")
            print(f"         -> {result.answer[:100]}\n")

        except Exception as e:
            wall = time.time() - t0
            print(f"         CRASH {wall:.1f}s: {e}\n")
            results.append(dict(
                id=qid, category=cat, question=q, model_pair=model_pair_label,
                wall_s=round(wall, 2), llm_calls=-1, prompt_tokens=0,
                completion_tokens=0, sql_errors=-1, parse_errors=-1,
                keyword_hits="0/0", keyword_pct=0,
                empty_result=False, forced_fail=True, real_pass=False,
                answer_preview=f"CRASH: {e}",
                success=False, manual_score=0, notes=test["notes"],
            ))

    # ── summary ──────────────────────────────────────────────────────────────
    total      = len(results)
    sql_ok     = sum(1 for r in results if r["sql_errors"] == 0)
    single     = sum(1 for r in results if r["llm_calls"] == 1)
    real_pass  = sum(1 for r in results if r["real_pass"])
    forced_fails = sum(1 for r in results if r["forced_fail"])
    avg_t      = sum(r["wall_s"] for r in results) / total if total else 0
    tot_se     = sum(r["sql_errors"]   for r in results if r["sql_errors"]   >= 0)
    tot_pe     = sum(r["parse_errors"] for r in results if r["parse_errors"] >= 0)

    # keyword avg only over questions that have keywords defined
    kp_vals = [r["keyword_pct"] for r in results
               if r["keyword_pct"] is not None and r["keyword_pct"] >= 0]
    avg_kp  = round(sum(kp_vals) / len(kp_vals)) if kp_vals else 0

    cs = defaultdict(lambda: {
        "count": 0, "sql_ok": 0, "single": 0,
        "real_pass": 0, "empty": 0, "time": 0.0,
    })
    for r in results:
        c = r["category"]
        cs[c]["count"]     += 1
        cs[c]["sql_ok"]    += (r["sql_errors"] == 0)
        cs[c]["single"]    += (r["llm_calls"] == 1)
        cs[c]["real_pass"] += int(r["real_pass"])
        cs[c]["empty"]     += int(r["empty_result"])
        cs[c]["time"]      += r["wall_s"]

    print(f"\n{sep}")
    print(f"  SUMMARY — {model_pair_label}")
    print(f"{sep}")
    print(f"  {'Category':<18} {'Tests':>5} {'Pass':>7} {'SQL OK':>7} "
          f"{'1-call':>7} {'Empty':>6} {'AvgTime':>9}")
    print(f"  {'-'*62}")
    for cat, s in sorted(cs.items()):
        avg = s["time"] / s["count"] if s["count"] else 0
        print(f"  {cat:<18} {s['count']:>5} "
              f"{s['real_pass']:>4}/{s['count']:<2} "
              f"{s['sql_ok']:>4}/{s['count']:<2} "
              f"{s['single']:>4}/{s['count']:<2} "
              f"{s['empty']:>5}  "
              f"{avg:>7.1f}s")

    print(f"\n  Real pass:    {real_pass}/{total} "
          f"({100 * real_pass // total if total else 0}%)"
          f"  [{forced_fails} forced-fail on empty result]")
    print(f"  SQL success:  {sql_ok}/{total} "
          f"({100 * sql_ok // total if total else 0}%)")
    print(f"  Single-call:  {single}/{total} "
          f"({100 * single // total if total else 0}%)")
    print(f"  Avg time:     {avg_t:.1f}s")
    print(f"  Avg kw hit:   {avg_kp}%  (over {len(kp_vals)} keyword-tested questions)")
    print(f"  SQL errors:   {tot_se}  |  Parse errors: {tot_pe}")

    # ── forced-fail detail ───────────────────────────────────────────────────
    ff_rows = [r for r in results if r["forced_fail"]]
    if ff_rows:
        print(f"\n  FORCED FAILS (empty result where data expected):")
        for r in ff_rows:
            print(f"    {r['id']:<8} {r['question'][:60]}")

    # ── save ─────────────────────────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = output_file or f"eval_results_{ts}.json"
    with open(fname, "w") as f:
        json.dump(dict(
            model_pair=model_pair_label,
            timestamp=ts,
            total=total,
            real_pass=real_pass,
            sql_ok=sql_ok,
            single_call=single,
            avg_time_s=round(avg_t, 2),
            avg_keyword_pct=avg_kp,
            sql_errors_total=tot_se,
            parse_errors_total=tot_pe,
            forced_fails=forced_fails,
            results=results,
        ), f, indent=2)
    print(f"\n  Results saved -> {fname}\n")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="InspectAI NL2SQL Evaluation")
    p.add_argument("--model-pair", default="llama3.1:8b + deepseek:6.7b")
    p.add_argument("--skip-multiturn", action="store_true")
    p.add_argument("--category", default=None,
                   help="Aggregate | Filter | Join | Form Answers | Scores | "
                        "Trends | People | Multi-turn | ICA Risk Level | "
                        "Deferred Chain | Edge Case")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    run_eval(a.model_pair, a.skip_multiturn, a.category, a.output)