"""
NL2SQL Agent — Evaluation Test Set (105 queries)
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
        "question": "give me this year's inspection counts split by completed vs conducted",
        "expected_contains": ["complet", "conduct"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Year filter + conducted/completed category split (2 rows). "
                 "completed = CLOSED; conducted = other non-DRAFT.",
    },
    {
        "id": "AGG-03", "category": "Aggregate",
        "question": "what is the total capex and opex across all mitigative actions",
        "expected_contains": ["capex", "opex"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Two aggregates",
    },
    {
        "id": "AGG-04", "category": "Aggregate",
        "question": "how many open mitigative actions are there",
        "expected_contains": ["open"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Status filter count",
    },
    {
        "id": "AGG-05", "category": "Aggregate",
        "question": "break down inspections this month vs last month by completed and conducted",
        "expected_contains": ["month"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "date_trunc month FILTER crossed with the conducted/completed "
                 "category split (2 rows x this_month/last_month)",
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
        "question": "how many mitigative actions were raised this quarter, and how many were closed this quarter",
        "expected_contains": ["raised", "closed"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "FLOW metric via two FILTER clauses: raised = created_on in "
                 "quarter (132); closed = completed_on in quarter (13). KNOWN "
                 "DEFECT: agent returns a COHORT count instead (raised-this-"
                 "quarter that are now closed = 23) — left failing to surface it.",
    },
    {
        "id": "AGG-08", "category": "Aggregate",
        "question": "what is the average inspection duration in hours",
        "expected_contains": ["hour"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "AVG(end_date_time - start_date_time) = 8.36h. KNOWN DEFECT: "
                 "agent uses submitted_on - start_date_time (3.59h, includes "
                 "write-up time) — left failing to surface it.",
    },
    {
        "id": "AGG-09", "category": "Aggregate",
        "question": "what percentage of inspections resulted in at least one mitigative action",
        "expected_contains": ["%"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "LEFT JOIN + COUNT DISTINCT ~66.8%",
    },
    {
        "id": "AGG-10", "category": "Aggregate",
        "question": "what is the average number of observations per inspection",
        "expected_contains": ["observation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Subquery AVG per submission_id, label='Observation'",
        "data_dependency": "ai_answers",
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
        "question": "show all overdue mitigative actions",
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
    {
        "id": "FLT-08", "category": "Filter",
        "question": "how many facilities are in each country",
        "expected_contains": ["country"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Org hierarchy coverage — facility.country_id -> country (live path; NOT via organisation, "
                 "facility.organization_id is confirmed always NULL)",
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
        "question": "show mitigative actions with their inspection facility and inspector",
        "expected_contains": ["facility", "inspector"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "4-table join",
    },
    {
        "id": "JOI-05", "category": "Join",
        "question": "what is the current open mitigative action count by facility",
        "expected_contains": ["facility", "open"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Correct JOIN direction",
    },
    {
        "id": "JOI-06", "category": "Join",
        "question": "which facility has the highest number of overdue mitigative actions",
        "expected_contains": ["facility", "overdue"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "top1 — facility with most OVERDUE MAs via ica->ir->facility",
    },
    {
        "id": "JOI-07", "category": "Join",
        "question": "which client has the highest number of overdue mitigative actions",
        "expected_contains": ["client", "overdue"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "top1 — client with most OVERDUE MAs via ica->ir->client",
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
    {
        "id": "JOI-13", "category": "Join",
        "question": "show inspection reports grouped by entity name",
        "expected_contains": ["entity"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "entity coverage — inspection_report.entity_id -> entity.id is DIRECT (not via facility)",
    },
    {
        "id": "JOI-14", "category": "Join",
        "question": "which communication group does Al Ghadeer facility belong to",
        "expected_contains": ["group"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "COMMUNICATION GROUPS coverage — facility -> inspection_communication_group_facilities "
                 "-> inspection_communication_group junction",
    },
    {
        "id": "JOI-15", "category": "Join",
        "question": "which countries does each organisation operate in",
        "expected_contains": ["country", "organisation"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "organisation_countries junction coverage — facility.organization_id is dead (always "
                 "NULL live), must reach organisation via organisation_countries or the communication-"
                 "group junction tables instead",
    },
    # ── Form Answers ──────────────────────────────────────────────────────────
    {
        "id": "ANS-01", "category": "Form Answers",
        "question": "what are the most common observation types across all inspections",
        "expected_contains": ["deviation", "observation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "get_answer_stats Deviation:359",
        "data_dependency": "ai_answers",
    },
    {
        "id": "ANS-02", "category": "Form Answers",
        "question": "list all questions and answers from the most recent inspection form",
        "expected_contains": ["question", "answer"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "module_name filter regression",
        "data_dependency": "ai_answers",
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
        "data_dependency": "ai_answers",
    },
    {
        "id": "ANS-05", "category": "Form Answers",
        "question": "what are the most common causes of mitigative actions",
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
        "data_dependency": "ai_answers",
    },
    {
        "id": "ANS-07", "category": "Form Answers",
        "question": "what are the repetitive observations in the last 6 months",
        "expected_contains": ["observation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "GROUP BY HAVING COUNT > 1",
        "data_dependency": "ai_answers",
    },
    {
        "id": "ANS-08", "category": "Form Answers",
        "question": "which observation appears most frequently as high risk",
        "expected_contains": ["observation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Subquery: risk High -> obs text",
        "data_dependency": "ai_answers",
    },
    # ── Scores ────────────────────────────────────────────────────────────────
    {
        "id": "SCR-01", "category": "Scores",
        "question": "show me questions with their scores for the last inspection by George",
        "expected_contains": ["score", "george"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ai_answers.score + users JOIN",
        "data_dependency": "ai_answers",
    },
    {
        "id": "SCR-02", "category": "Scores",
        "question": "what is the average score for the risk level question",
        "expected_contains": ["score", "risk"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "get_answer_stats avg 3.24",
        "data_dependency": "ai_answers",
    },
    {
        "id": "SCR-03", "category": "Scores",
        "question": "which inspection had the highest total score",
        "expected_contains": ["score", "inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "SUM ai_answers.score = 102",
        "data_dependency": "ai_answers",
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
        "question": "are mitigative actions being closed faster or slower compared to last quarter",
        "expected_contains": ["quarter", "closed"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "close_on is a dead column (always NULL) — answer must use completed_on for status-based closure rate",
    },
    {
        "id": "TRD-03", "category": "Trends",
        "question": "which month had the most high risk observations this year",
        "expected_contains": ["month"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "GROUP BY month over ai_answers observations",
        "data_dependency": "ai_answers",
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
        "question": "which inspection type has the most mitigative actions raised",
        "expected_contains": ["type", "action"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Regulatory/Insurance:103",
    },
    # ── People ────────────────────────────────────────────────────────────────
    {
        "id": "PPL-02", "category": "People",
        "question": "show breakdown of mitigative action status by responsible party",
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
    {
        "id": "PPL-04", "category": "People",
        "question": "how many inspections have an FM Manager as the facility contact",
        "expected_contains": ["fm manager", "manager"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "USER ROLES coverage — role filter must use EXISTS(user_roles JOIN role) not a plain "
                 "JOIN, since one user can hold multiple roles and a JOIN fans out rows",
    },
    {
        "id": "PPL-05", "category": "People",
        "question": "who approved the most recently completed inspection report",
        "expected_contains": ["approve"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "APPROVAL WORKFLOW coverage — review.status='COMPLETED' is NOT the same as approved; "
                 "must also filter action='APPROVE' (a COMPLETED review can be a rejection)",
    },
    # ── Multi-turn ────────────────────────────────────────────────────────────
    {
        "id": "MT-01", "category": "Multi-turn",
        "question": "show me the most recently answered inspection",
        "expected_contains": ["facility", "inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Turn 1 — uses ai_answers.submitted_on so the result always has form data",
        "data_dependency": "ai_answers",
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
        "data_dependency": "ai_answers",
    },
    {
        "id": "MT-04", "category": "Multi-turn",
        "question": "show more",
        "expected_contains": [],
        "expected_nonempty": True,
        "expected_llm_calls": 0, "notes": "Pagination — must return Q&A rows not inspector names",
        "depends_on": "MT-03",
        "data_dependency": "ai_answers",
    },
    # ── ICA Risk Level — Problem 2 fix ───────────────────────────────────────
    {
        "id": "ICA-01", "category": "ICA Risk Level",
        "question": "show all high risk mitigative actions",
        "expected_contains": ["high", "risk"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "P2 fix: JOIN risk_level lookup, not WHERE risk_level_id='High'",
    },
    {
        "id": "ICA-02", "category": "ICA Risk Level",
        "question": "what is the risk level breakdown of all mitigative actions",
        "expected_contains": ["high", "medium"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "P2 fix: GROUP BY rl.name — must show human-readable names not UUIDs",
    },
    {
        "id": "ICA-03", "category": "ICA Risk Level",
        "question": "how many mitigative actions are high risk",
        "expected_contains": ["high"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "P2 fix: COUNT with JOIN risk_level — before fix returned 0",
    },
    {
        "id": "ICA-04", "category": "ICA Risk Level",
        "question": "show open mitigative actions with high risk grouped by facility",
        "expected_contains": ["facility", "high"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "P2 fix: 3-table join ica + risk_level + ir + facility",
    },
    # ── Deferred Chain — Problem 5 fix ───────────────────────────────────────
    {
        "id": "DEF-01", "category": "Deferred Chain",
        "question": "show all deferred mitigative actions",
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
        "data_dependency": "ai_answers",
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
        "question": "how many mitigative actions have been carried forward more than once",
        "expected_contains": ["deferred"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "P5 fix: HAVING COUNT > 1 on CLOSE_WITH_DEFERRED per cause",
    },
    # ── Edge Cases ────────────────────────────────────────────────────────────
    {
        "id": "EDG-01", "category": "Edge Case",
        "question": "show all high risk observations this month",
        "expected_contains": ["high", "observation"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "Answer + date filter",
        "data_dependency": "ai_answers",
    },
    {
        "id": "EDG-02", "category": "Edge Case",
        "question": "how many questions are in the inspection form",
        "expected_contains": ["24", "question"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "COUNT ai_questions WHERE module_name='Inspection Form' = 24 "
                 "(the whole table also counts every audit form's questions)",
    },
    {
        "id": "EDG-03", "category": "Edge Case",
        "question": "show mitigative actions where responsible is client",
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
        "question": "which facilities have the most overdue mitigative actions and the lowest inspection scores",
        "expected_contains": ["facility", "overdue", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 2,
        "notes": "ICA OVERDUE count + avg inspection_score, both joined to facility",
    },
    {
        "id": "CROSS-02", "category": "Cross-domain",
        "question": "which inspection types generate the most high risk mitigative actions",
        "expected_contains": ["inspection", "high", "risk"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "inspection_type → ir → ica → risk_level JOIN chain",
    },
    {
        "id": "CROSS-03", "category": "Cross-domain",
        "question": "list facilities that were inspected this year but have zero mitigative actions",
        "expected_contains": ["facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "LEFT JOIN ica WHERE ica.id IS NULL",
    },
    {
        "id": "CROSS-04", "category": "Cross-domain",
        "question": "which facilities have high risk observations but no mitigative actions raised",
        "expected_contains": ["facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 2,
        "notes": "ai_answers risk_level=High LEFT JOIN ica IS NULL",
        "data_dependency": "ai_answers",
    },

    # ── ETL Boundary Awareness ────────────────────────────────────────────────
    {
        "id": "ETL-01", "category": "ETL Boundary",
        "question": "show me the most recent inspection that has form data available",
        "expected_contains": ["inspection", "facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Must use aa subquery NOT ir.submitted_on — ETL gap awareness",
        "data_dependency": "ai_answers",
    },
    {
        "id": "ETL-02", "category": "ETL Boundary",
        "question": "list the last 10 inspections that have form answers indexed",
        "expected_contains": ["inspection", "facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "SELECT DISTINCT inspection_report_id FROM ai_answers ORDER BY submitted_on DESC LIMIT 10",
        "data_dependency": "ai_answers",
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
        "id": "CONV-02", "category": "Conversational",
        "question": "what needs attention right now",
        "expected_contains": ["overdue", "risk", "mitigative", "facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 2,
        "notes": "Ambiguous — expect overdue MA or high risk observations",
    },
    {
        "id": "CONV-03", "category": "Conversational",
        "question": "how are we doing overall",
        "expected_contains": ["score", "inspection", "mitigative", "average"],
        "expected_nonempty": True,
        "expected_llm_calls": 2, "notes": "High-level summary — avg score or completion rate",
    },
    {
        "id": "CONV-04", "category": "Conversational",
        "question": "any critical issues I should know about",
        "expected_contains": ["high", "risk", "overdue", "mitigative"],
        "expected_nonempty": True,
        "expected_llm_calls": 2, "notes": "Should surface high risk + overdue items",
    },
    {
        "id": "CONV-05", "category": "Conversational",
        "question": "give me a breakdown of this quarter",
        "expected_contains": ["quarter", "inspection", "mitigative"],
        "expected_nonempty": True,
        "expected_llm_calls": 2, "notes": "Quarter summary — inspections done, MAs raised",
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
        "question": "rank inspection types by their average mitigative actions per inspection",
        "expected_contains": ["inspection"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "GROUP BY inspection_type ORDER BY avg_ca DESC",
    },
    {
        "id": "COMP-04", "category": "Comparison",
        "question": "compare overdue mitigative action counts between CLIENT and INTERNAL_OPERATIONS",
        "expected_contains": ["client", "internal"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "FILTER aggregate by responsible enum — two-column comparison",
    },

    # ── Extended Aggregates ───────────────────────────────────────────────────
    {
        "id": "AGG-EXT-01", "category": "Extended Aggregate",
        "question": "what is the average number of mitigative actions per inspection",
        "expected_contains": ["average", "avg", "mitigative"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Subquery: COUNT per inspection, then AVG",
    },
    {
        "id": "AGG-EXT-02", "category": "Extended Aggregate",
        "question": "how many mitigative actions were closed on time vs late",
        "expected_contains": ["closed", "time", "late", "on_time", "overdue"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "FILTER: status=CLOSED AND completed_on <= target_close_out_date vs > target_close_out_date",
    },
    {
        "id": "AGG-EXT-03", "category": "Extended Aggregate",
        "question": "what percentage of mitigative actions are currently overdue",
        "expected_contains": ["overdue", "%", "percent"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "OVERDUE / ALL active MAs * 100 = 32.7%. KNOWN DEFECT: agent "
                 "scopes the denominator to OVERDUE too -> always 100%. Left "
                 "failing to surface it.",
    },
    {
        "id": "AGG-EXT-04", "category": "Extended Aggregate",
        "question": "how many facilities have had zero inspections this year — return a single number",
        "expected_contains": ["facilit"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "scalar COUNT — facility WHERE NOT EXISTS(ir this year). ~146.",
    },
    {
        "id": "AGG-EXT-05", "category": "Extended Aggregate",
        "question": "what is the total number of observations recorded across all inspections",
        "expected_contains": ["observation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "COUNT DISTINCT submission_id WHERE label='Observation'",
        "data_dependency": "ai_answers",
    },

    # ── Financial / CAPEX / OPEX ─────────────────────────────────────────────
    {
        "id": "FIN-01", "category": "Financial",
        "question": "what is the total capex generated converted and lost",
        "expected_contains": ["capex_generated", "capex_converted", "capex_lost"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Scalar summary ONE ROW — SUM(ica.capex) vs SUM(po.po_amount) vs REJECTED",
    },
    {
        "id": "FIN-02", "category": "Financial",
        "question": "what is the total opex incurred from purchase requisitions",
        "expected_contains": ["opex_incurred", "pr"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "SUM(pr.pr_amount) from pr_details — NOT ica.opex",
    },
    {
        "id": "FIN-03", "category": "Financial",
        "question": "which fm managers generate the most capex",
        "expected_contains": ["inspectee_user_id", "capex"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "FM managers = inspectee_user_id (NOT inspector_user_id) → users JOIN ica GROUP BY inspectee",
    },
    {
        "id": "FIN-04", "category": "Financial",
        "question": "what is the total capex on mitigative actions whose capex_status is REJECTED",
        "expected_contains": ["capex"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "scalar SUM(capex) WHERE capex_status='REJECTED' (aggregate, "
                 "not a listing; avoids the 'capex lost' glossary rule)",
    },
    {
        "id": "FIN-05", "category": "Financial",
        "question": "for each facility that has at least one purchase order or requisition, show total capex converted and opex incurred",
        "expected_contains": ["facility", "capex", "opex"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "po_details + pr_details pre-aggregated per MA, summed per "
                 "facility via ica -> ir -> facility; only facilities with a PO/PR",
    },
    {
        "id": "FIN-06", "category": "Financial",
        "question": "what is the total purchase order amount raised this year",
        "expected_contains": ["total_po_amount", "po_count"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Scalar: SUM(po.po_amount) WHERE po_date this year — ONE ROW not list",
    },
    {
        "id": "FIN-07", "category": "Financial",
        "question": "compare capex generated vs capex converted by FM manager",
        "expected_contains": ["inspectee_user_id", "generated", "converted"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "FM managers (inspectee_user_id) generate capex via MAs — NOT inspector_user_id. SUM(ica.capex) vs SUM(po.po_amount) per FM manager",
    },

    # ── Window functions / percentile / stddev / NTILE / FILTER / junctions ──
    {
        "id": "WIN-01", "category": "Trends",
        "question": "show inspections where the score dropped more than 10 points from the previous inspection at the same facility",
        "expected_contains": ["facility", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "LAG(inspection_score) OVER (PARTITION BY facility_id ORDER BY submitted_on); count = 10",
    },
    {
        "id": "WIN-02", "category": "Scores",
        "question": "for each inspector, what was the score of their single most recent inspection",
        "expected_contains": ["inspector", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "ROW_NUMBER() PARTITION BY inspector_user_id ORDER BY submitted_on DESC = 1; 15 inspectors",
    },
    {
        "id": "PCT-01", "category": "Aggregate",
        "question": "what is the median inspection score",
        "expected_contains": ["median", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY inspection_score) = 93",
    },
    {
        "id": "PCT-02", "category": "Financial",
        "question": "what is the 90th percentile of mitigative action capex",
        "expected_contains": ["percentile", "capex"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY capex) = 5200",
    },
    {
        "id": "STDV-01", "category": "Aggregate",
        "question": "what is the standard deviation of inspection scores",
        "expected_contains": ["deviation", "score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "STDDEV_SAMP(inspection_score) = 14.88",
    },
    {
        "id": "STDV-02", "category": "Comparison",
        "question": "which inspection type has the most variable inspection scores",
        "expected_contains": ["type"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "STDDEV_SAMP GROUP BY type ORDER BY DESC LIMIT 1",
    },
    {
        "id": "NTL-01", "category": "Comparison",
        "question": "split facilities into four quartiles by average inspection score and show how many facilities are in each quartile",
        "expected_contains": ["quartile"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "NTILE(4) OVER (ORDER BY avg_score); 4 rows ~9/8/8/8",
    },
    {
        "id": "FIL-01", "category": "Join",
        "question": "for each inspection type, how many inspections are closed versus still in progress",
        "expected_contains": ["type", "closed"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "COUNT(*) FILTER (...) x2 GROUP BY inspection_type",
    },
    {
        "id": "FIL-02", "category": "Aggregate",
        "question": "give me open, closed, and overdue mitigative action counts in one row",
        "expected_contains": ["open", "closed", "overdue"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "3 FILTER columns over inspection_corrective_action",
    },
    {
        "id": "JUN-01", "category": "Join",
        "question": "how many organisations operate in each country",
        "expected_contains": ["country", "organisation"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "organisation_countries junction JOIN country GROUP BY country",
    },
    {
        "id": "DIS-01", "category": "Conversational",
        "question": "show me the scores",
        "expected_contains": [],
        "expected_nonempty": True,
        "expected_llm_calls": 2,
        "notes": "Under-specified — a good answer asks which scores (inspection / "
                 "form-answer / by whom). real_pass-only, no ground truth.",
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

import sys, os, time, json, argparse, re
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_table_mention_re_cache: dict[str, "re.Pattern"] = {}


def _table_mentioned(text: str, tables: set) -> bool:
    """Phase 3.12: word-boundary, case-insensitive check for whether any of
    `tables` is named in `text` — used to test relevance of a fixed-overhead
    relationship-catalogue section against a specific query's named/used
    tables. Approximate (a section could mention a table name in passing
    without the content actually being useful, or use a synonym/alias that
    doesn't match at all) — a text-search proxy for relevance, not a
    ground-truth judgment; see profile_fixed_overhead_relevance's docstring."""
    for t in tables:
        pat = _table_mention_re_cache.get(t)
        if pat is None:
            pat = re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)
            _table_mention_re_cache[t] = pat
        if pat.search(text):
            return True
    return False


def run_eval(model_pair_label="llama3.1:8b + deepseek:6.7b",
             skip_multiturn=False, category_filter=None, output_file=None,
             profile_tokens=False, orch=None):

    from src.agent.orchestrator import AgentOrchestrator, ConversationSession
    from src.llm.tokenize import count_tokens_llamacpp
    from src.knowledge.fuzzy_vocab import fuzzy_augment
    from src.retrieval import text_similarity
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

    # Phase 3.14: accept a pre-built orchestrator (e.g. from ablation_3_14.py,
    # which reuses ONE orchestrator across 10 full-suite runs — baseline +
    # 9 section-ablated variants — to avoid paying embedding-model/semantic-
    # index/reranker startup cost 10 times over). orch=None (every existing
    # caller) preserves the original always-construct-fresh behavior exactly.
    if orch is None:
        orch = AgentOrchestrator.from_env(profile=profile_tokens)
    results = []
    session = ConversationSession()

    # ── data-health preflight — detect a stale/incomplete answer-layer load
    # (the one-time ai_answers insert) so questions that depend on it are
    # bucketed as `snapshot_blocked` instead of counting against the headline
    # pass rate. They still run, so a repaired snapshot auto-recovers them. ──
    from data_health import run_preflight, blocked_dependencies, classify_row, format_report
    data_health = run_preflight(orch.db_engine, "inspection")
    blocked_deps = blocked_dependencies(data_health)
    print(format_report(data_health) + "\n")

    # ── Phase 3.10: static schema-block composition, computed ONCE (see
    # DatabaseSchema.profile_slim_schema_tokens docstring — the slim block's
    # content doesn't depend on the question, so per-query re-tokenizing
    # would just re-measure the same ~29 lines 103 times). _schema_tables_cost
    # merges the key_columns/lookup_tables sub-dicts (disjoint table-name
    # keys, confirmed via USEFUL_COLUMNS/LOOKUP_TABLES) for the per-table
    # utilization diff below — fk_joins is reported separately (Part A) but
    # NOT included in the diff (its lines aren't attributable to one table
    # the same way a key-columns/lookup line is). _table_used_count/
    # _table_unused_count accumulate per-table frequency across the whole
    # run for the "which tables are dead weight" ranking. ──────────────────
    _schema_profile = None
    _schema_tables_cost = {}
    _table_used_count = defaultdict(int)
    _table_unused_count = defaultdict(int)
    _n_utilization_queries = 0
    # Confirmed live in this run's smoke test (contradicts generation.py's
    # 2026-07-30/08-04 "essentially never populated" comment on
    # relevant_tables — that observation has since drifted): the reasoning
    # model DOES sometimes supply tables/relevant_tables, which switches
    # _select_schema_block onto for_schema_hint()'s per-question full-column
    # dump instead of the static slim block Part A measures. Those queries
    # are excluded from the slim-table diff below (their auto_schema isn't
    # the content Part A profiled) and counted here instead.
    _n_targeted_variant_queries = 0

    # ── Phase 3.11: targeted-branch (for_schema_hint) decomposition. Same
    # one-time-per-table rationale as 3.10's _schema_profile (see
    # DatabaseSchema.profile_targeted_schema_tables docstring) — lets every
    # per-query reconstruction below be a dict lookup, not a tokenize call.
    # _targeted_call_samples collects one row per "targeted"-variant
    # generate_sql call across the whole run — used for the composition
    # breakdown, the planner-vs-SQL table-count comparison (hypothesis A),
    # and the real_total distribution (hypothesis: is the ~4x slim-vs-
    # targeted gap one consistent cost, or a few outlier calls?). ─────────
    _targeted_schema_profile = None
    _targeted_call_samples = []

    # ── Phase 3.12: relevance of the two FIXED blocks 3.11 identified as
    # ~78% of every targeted call (_fmt_lookup ~979 tok, table_relationships
    # ~1922 tok) — same one-time-per-item computation as 3.11's per-table
    # costs (see DatabaseSchema.profile_fixed_overhead_relevance docstring).
    # _fixed_item_relevant_named/_used count, per fixed-overhead sub-part,
    # how many targeted calls actually needed it (mentions a table the
    # reasoning model named / a table the generated SQL used); combined
    # with each sub-part's constant token cost this gives an exact
    # per-call "how much of this fixed tax was dead weight" estimate. ────
    _fixed_overhead_profile = None
    _fixed_item_relevant_named = defaultdict(int)
    _fixed_item_relevant_used = defaultdict(int)
    _fixed_overhead_calls = []

    # ── Phase 3.17 Part A: reasoning-LLM prompt composition — the first
    # instrumentation of this prompt at all (see agent/prompts/builder.py's
    # profile_prompt_composition docstring). _reasoning_profile is the
    # one-time static-base + per-entry breakdown (invariant per process);
    # _reasoning_call_samples collects one real /tokenize measurement of
    # build_dynamic_prompt_additions()'s ACTUAL per-call output per question
    # (not a reconstruction from the one-time per-entry numbers — see that
    # docstring for why) plus, for Multi-turn questions only, a real
    # tokenize of session.build_context_block()'s actual output.
    _reasoning_profile = None
    _reasoning_call_samples = []
    _reasoning_title_hits = defaultdict(int)

    # ── Phase 3.17 Part B: SQL-gen side's remaining un-audited pieces —
    # CORE_CONTEXT's static base + SQL_GENERATION_PROMPT's skeleton (one-
    # time, see sql/prompts.py's profile_core_context_composition
    # docstring) plus real per-call build_core_context_additions() output,
    # same "tokenize the actual output, don't reconstruct" approach as
    # Part A2.
    _core_context_profile = None
    _core_context_call_samples = []
    _core_context_title_hits = defaultdict(int)

    if profile_tokens:
        _sql_base_url = getattr(orch.sql_llm, "api_base", "") or getattr(orch.sql_llm, "base_url", "")
        _schema_profile = orch.db_schema.profile_slim_schema_tokens(_sql_base_url)
        _schema_tables_cost = {**_schema_profile["key_columns"], **_schema_profile["lookup_tables"]}
        _targeted_schema_profile = orch.db_schema.profile_targeted_schema_tables(_sql_base_url)
        _fixed_overhead_profile = orch.db_schema.profile_fixed_overhead_relevance(_sql_base_url)
        _reasoning_base_url = getattr(orch.reasoning_llm, "api_base", "") or getattr(orch.reasoning_llm, "base_url", "")
        _reasoning_profile = orch._builder_module.profile_prompt_composition(orch._system_prompt, _reasoning_base_url)
        from src.sql.prompts import profile_core_context_composition, build_core_context_additions, CORE_CONTEXT_CONDITIONAL
        _core_context_profile = profile_core_context_composition(_sql_base_url)

    for i, test in enumerate(tests):
        cat  = test["category"]
        qid  = test["id"]
        q    = test["question"]
        print(f"[{i+1:02d}/{len(tests)}] {qid} ({cat})  Q: {q[:65]}")
        t0 = time.time()
        try:
            result = orch.query(q, session=session if cat == "Multi-turn" else None,
                                visibility_scope=test.get("visibility_scope", {}))
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

            # ── ground-truth comparison (result-level correctness) ──────────
            try:
                from eval_truth import run_truth_check, extract_agent_rows
                tm = run_truth_check(orch.db_engine, qid, extract_agent_rows(result))
            except Exception as _te:  # truth layer must never crash the eval
                tm = {"covered": False, "truth_match": None, "detail": f"err: {_te}"}

            # Force fail when we know the result must not be empty
            forced_fail = bool(test.get("expected_nonempty") and empty)
            real_pass   = result.success and not forced_fail

            # snapshot_blocked: this question depends on an answer layer the
            # preflight flagged as broken AND it went wrong (empty / truth miss)
            dep = test.get("data_dependency")
            snap_blocked = classify_row(dep, blocked_deps, real_pass, tm["truth_match"])

            # ── Phase 3.17 Part A2: reasoning-side per-call dynamic additions
            # — tokenize the ACTUAL build_dynamic_prompt_additions() output
            # for this question (same fuzzy_augment + text_similarity
            # score_fn loop.py itself uses — see orchestrator/loop.py:427-437
            # — so budget-capping when 2+ blocks match behaves identically to
            # production) rather than reconstructing from the one-time
            # per-entry costs in _reasoning_profile — avoids the exact
            # "reconstruction drifts from production logic" trap 3.13 hit
            # once already for the SQL-gen side. Pure function of the
            # question text + embed_model, no session-state side effects —
            # safe to call for every question regardless of category.
            # NOTE: session.build_context_block() (conversation-history
            # cost) is deliberately NOT measured here — it mutates session
            # state (pending_resolved_disclosure) and calling it out of the
            # exact order loop.py itself calls it in risks corrupting the
            # MT-01..04 depends_on chain for a number that's only "thin
            # coverage" anyway (see the 3.17 plan's Part A2 scope note).
            if profile_tokens and _reasoning_profile:
                _q_for_matching = fuzzy_augment(q)
                _tier2_score_fn = lambda qq, title: text_similarity(orch.embed_model, qq, title)
                _dyn_additions = orch._builder_module.build_dynamic_prompt_additions(
                    _q_for_matching, score_fn=_tier2_score_fn
                )
                _dyn_tok = count_tokens_llamacpp(_dyn_additions, _reasoning_base_url) if _dyn_additions else 0
                _reasoning_call_samples.append({"id": qid, "dynamic_additions_tokens": _dyn_tok})
                for blk in orch._builder_module.CONDITIONAL_BLOCKS:
                    if re.search(blk["question_pattern"], _q_for_matching):
                        _reasoning_title_hits[blk["title"]] += 1
                for ex in orch._builder_module.EXAMPLES_TIER2:
                    if re.search(ex["question_pattern"], _q_for_matching):
                        _reasoning_title_hits[ex["title"]] += 1

            # ── Phase 3.17 Part B: SQL-gen side's build_core_context_additions()
            # real per-call output — same rationale as Part A2, raw `q` (not
            # fuzzy_augment'd — generate_sql() calls this with the raw
            # question, see generation.py's core_context assembly).
            if profile_tokens and _core_context_profile:
                _cc_additions = build_core_context_additions(q)
                _cc_tok = count_tokens_llamacpp(_cc_additions, _sql_base_url) if _cc_additions else 0
                _core_context_call_samples.append({"id": qid, "additions_tokens": _cc_tok})
                for blk in CORE_CONTEXT_CONDITIONAL:
                    if re.search(blk["question_pattern"], q):
                        _core_context_title_hits[blk["title"]] += 1

            # ── Phase 3.10: schema-block utilization — diff the static
            # table set (_schema_tables_cost, Part A) against the tables the
            # generated SQL actually referenced this turn. generated_tables
            # is already computed by generate_sql() for every call (see
            # generation.py's _diagnostics, NOT gated by _profile) — no new
            # instrumentation needed, just reading it back out of the steps
            # this turn already took. Unioned across every generate_sql step
            # (a multi-call turn's retries may reference different tables).
            #
            # Skipped entirely (and counted separately) when any generate_sql
            # step this turn used the "targeted" schema_block_variant (see
            # generation.py's schema_hint_breakdown comment) — that branch's
            # auto_schema is a per-question for_schema_hint() dump, not the
            # static for_sql_prompt_slim() content _schema_tables_cost prices,
            # so diffing against it would silently mislabel real content as
            # "unused."
            _sh_included = _sh_used = 0
            _sh_unused_tokens = 0
            if profile_tokens and _schema_tables_cost:
                _gen_tables = set()
                _is_targeted = False
                for s in result.steps:
                    if s.tool == "generate_sql":
                        _sql_result = s.result.get("result") or {}
                        _gt = (_sql_result.get("_diagnostics") or {}).get("generated_tables") or []
                        _gen_tables.update(t.lower() for t in _gt)
                        _sh_bd = (_sql_result.get("_prompt_composition") or {}).get("schema_hint_breakdown") or {}
                        _variant = _sh_bd.get("schema_block_variant")
                        if _variant == "targeted":
                            _is_targeted = True
                            # ── Phase 3.11: reconstruct this call's targeted
                            # schema-block cost via per-table lookup (Part A,
                            # computed once above) and compare against the
                            # real per-call auto_schema_tokens this same step
                            # already measured (3.9/3.10's _profile block) —
                            # exact reconciliation, not the run-level
                            # subtraction estimate from the first 3.10 pass.
                            if _targeted_schema_profile:
                                _tt = _sh_bd.get("targeted_tables") or []
                                _pt = _targeted_schema_profile["per_table"]
                                _cols_tok = sum(_pt.get(t, {}).get("columns_tokens", 0) for t in _tt)
                                _fk_tok = sum(_pt.get(t, {}).get("fk_tokens", 0) for t in _tt)
                                _enum_tok = sum(_pt.get(t, {}).get("enum_tokens", 0) for t in _tt)
                                _useful_tok = sum(_pt.get(t, {}).get("useful_output_tokens", 0) for t in _tt)
                                _header_tok = sum(_pt.get(t, {}).get("header_tokens", 0) for t in _tt)
                                # Phase 3.13: for_schema_hint() now scopes
                                # _fmt_lookup() to each call's own FK
                                # neighborhood (see DatabaseSchema.
                                # targeted_lookup_scope) instead of always
                                # rendering the full ~979-tok block —
                                # reconstruct THIS call's actual scoped
                                # lookup cost from 3.12's per-item costs
                                # (_fixed_overhead_profile), not the old
                                # flat per-run constant, so the real-vs-
                                # reconstructed reconciliation below stays
                                # meaningful post-3.13.
                                if _fixed_overhead_profile:
                                    _scope = orch.db_schema.targeted_lookup_scope(_tt)
                                    _lookup_tok = (
                                        sum(item["tokens"] for k, item in _fixed_overhead_profile["lookup_tables"].items()
                                            if k in _scope["lookup_tables"])
                                        + sum(item["tokens"] for k, item in _fixed_overhead_profile["status_enums"].items()
                                              if k.split(".")[0] in _scope["status_enums"])
                                        + sum(item["tokens"] for k, item in _fixed_overhead_profile["join_patterns"].items()
                                              if k in _scope["join_patterns"])
                                    )
                                else:
                                    _lookup_tok = _targeted_schema_profile["lookup_section_tokens"]
                                _rel_tok = _targeted_schema_profile["table_relationships_tokens"]
                                _diag = _sql_result.get("_diagnostics") or {}
                                _targeted_call_samples.append(dict(
                                    id=qid,
                                    table_count=len(_tt),
                                    planner_table_count=len(_diag.get("planner_tables") or []),
                                    sql_table_count=len(set(t.lower() for t in (_diag.get("generated_tables") or []))),
                                    columns_tokens=_cols_tok, fk_tokens=_fk_tok, enum_tokens=_enum_tok,
                                    useful_output_tokens=_useful_tok, header_tokens=_header_tok,
                                    lookup_section_tokens=_lookup_tok, table_relationships_tokens=_rel_tok,
                                    reconstructed_total=(_cols_tok + _fk_tok + _enum_tok + _useful_tok
                                                         + _header_tok + _lookup_tok + _rel_tok),
                                    real_total=_sh_bd.get("auto_schema_tokens", 0),
                                ))

                            # ── Phase 3.12: relevance of the two fixed
                            # blocks against THIS call's named tables and
                            # THIS call's actually-generated-SQL tables —
                            # see eval.py's _table_mentioned and
                            # profile_fixed_overhead_relevance's docstring.
                            if _fixed_overhead_profile:
                                _diag = _sql_result.get("_diagnostics") or {}
                                _named = set(t.lower() for t in _tt)
                                _used = set(t.lower() for t in (_diag.get("generated_tables") or []))
                                _wasted_named = _wasted_used = 0
                                for _cat in ("lookup_tables", "status_enums", "join_patterns"):
                                    for _key, _item in _fixed_overhead_profile[_cat].items():
                                        _ck = (_cat, _key)
                                        _rel_named = _item.get("table") in _named
                                        _rel_used = _item.get("table") in _used
                                        if _rel_named:
                                            _fixed_item_relevant_named[_ck] += 1
                                        else:
                                            _wasted_named += _item["tokens"]
                                        if _rel_used:
                                            _fixed_item_relevant_used[_ck] += 1
                                        else:
                                            _wasted_used += _item["tokens"]
                                for _title, _item in _fixed_overhead_profile["relationship_sections"].items():
                                    _ck = ("relationship_sections", _title)
                                    _rel_named = _table_mentioned(_item["text"], _named)
                                    _rel_used = _table_mentioned(_item["text"], _used)
                                    if _rel_named:
                                        _fixed_item_relevant_named[_ck] += 1
                                    else:
                                        _wasted_named += _item["tokens"]
                                    if _rel_used:
                                        _fixed_item_relevant_used[_ck] += 1
                                    else:
                                        _wasted_used += _item["tokens"]
                                _fixed_overhead_calls.append(dict(
                                    id=qid, wasted_named_tokens=_wasted_named, wasted_used_tokens=_wasted_used,
                                ))
                if _is_targeted:
                    _n_targeted_variant_queries += 1
                elif _gen_tables or any(s.tool == "generate_sql" for s in result.steps):
                    _included = set(_schema_tables_cost.keys())
                    _used = _included & _gen_tables
                    _unused = _included - _used
                    _sh_included = len(_included)
                    _sh_used = len(_used)
                    _sh_unused_tokens = sum(_schema_tables_cost[t] for t in _unused)
                    _n_utilization_queries += 1
                    for t in _used:
                        _table_used_count[t] += 1
                    for t in _unused:
                        _table_unused_count[t] += 1

            row = dict(
                id=qid,
                category=cat,
                question=q,
                model_pair=model_pair_label,
                wall_s=round(wall, 2),
                llm_calls=result.stats.total_llm_calls,
                reasoning_ms=result.stats.total_llm_latency_ms,
                sql_gen_ms=result.stats.total_sql_gen_ms,
                sql_exec_ms=result.stats.total_sql_exec_ms,
                reasoning_prefill_ms=result.stats.reasoning_prefill_ms,
                reasoning_decode_ms=result.stats.reasoning_decode_ms,
                sql_gen_prefill_ms=result.stats.sql_gen_prefill_ms,
                sql_gen_decode_ms=result.stats.sql_gen_decode_ms,
                prompt_tokens=result.stats.total_prompt_tokens,
                completion_tokens=result.stats.total_completion_tokens,
                # Phase 3.8 profiling — all zero unless run with
                # --profile-tokens (see QueryStats.sql_gen_question_tokens).
                sql_gen_question_tokens=result.stats.sql_gen_question_tokens,
                sql_gen_schema_hint_tokens=result.stats.sql_gen_schema_hint_tokens,
                sql_gen_core_context_tokens=result.stats.sql_gen_core_context_tokens,
                sql_gen_retrieved_example_tokens=result.stats.sql_gen_retrieved_example_tokens,
                sql_gen_total_prompt_tokens=result.stats.sql_gen_total_prompt_tokens,
                sql_gen_profiled_calls=result.stats.sql_gen_profiled_calls,
                # Phase 3.9 profiling — schema_hint's own sub-contributors,
                # all zero unless run with --profile-tokens.
                sql_gen_auto_schema_tokens=result.stats.sql_gen_auto_schema_tokens,
                sql_gen_mandatory_hint_tokens=result.stats.sql_gen_mandatory_hint_tokens,
                sql_gen_planner_notes_tokens=result.stats.sql_gen_planner_notes_tokens,
                sql_gen_join_path_tokens=result.stats.sql_gen_join_path_tokens,
                sql_gen_business_rules_tokens=result.stats.sql_gen_business_rules_tokens,
                sql_gen_draft_sql_tokens=result.stats.sql_gen_draft_sql_tokens,
                # Phase 3.10 profiling — schema-block utilization, all zero
                # unless run with --profile-tokens.
                schema_tables_included=_sh_included,
                schema_tables_used=_sh_used,
                schema_tables_unused_tokens=_sh_unused_tokens,
                sql_errors=se,
                parse_errors=pe,
                keyword_hits=f"{kh}/{kt}" if kt else "n/a",
                keyword_pct=kp,
                empty_result=empty,
                forced_fail=forced_fail,
                real_pass=real_pass,
                data_dependency=dep,
                snapshot_blocked=snap_blocked,
                truth_covered=tm["covered"],
                truth_match=tm["truth_match"],
                truth_detail=tm["detail"],
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
            truth_str = ""
            if tm["covered"]:
                truth_str = (" | truth ✓" if tm["truth_match"]
                             else f" | truth ✗ ({tm['detail'][:60]})"
                             if tm["truth_match"] is False
                             else " | truth ?")
            print(f"         {perf} {wall:.1f}s | {result.stats.total_llm_calls} calls | "
                  f"kw {kw_str} | SQL {'OK' if se == 0 else str(se) + ' err'} | {status}{truth_str}")
            print(f"         -> {result.answer[:100]}\n")

        except Exception as e:
            wall = time.time() - t0
            print(f"         CRASH {wall:.1f}s: {e}\n")
            results.append(dict(
                id=qid, category=cat, question=q, model_pair=model_pair_label,
                wall_s=round(wall, 2), llm_calls=-1,
                reasoning_ms=0, sql_gen_ms=0, sql_exec_ms=0,
                reasoning_prefill_ms=0, reasoning_decode_ms=0,
                sql_gen_prefill_ms=0, sql_gen_decode_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                sql_gen_question_tokens=0, sql_gen_schema_hint_tokens=0,
                sql_gen_core_context_tokens=0, sql_gen_retrieved_example_tokens=0,
                sql_gen_total_prompt_tokens=0, sql_gen_profiled_calls=0,
                sql_gen_auto_schema_tokens=0, sql_gen_mandatory_hint_tokens=0,
                sql_gen_planner_notes_tokens=0, sql_gen_join_path_tokens=0,
                sql_gen_business_rules_tokens=0, sql_gen_draft_sql_tokens=0,
                schema_tables_included=0, schema_tables_used=0, schema_tables_unused_tokens=0,
                sql_errors=-1, parse_errors=-1,
                keyword_hits="0/0", keyword_pct=0,
                empty_result=False, forced_fail=True, real_pass=False,
                data_dependency=test.get("data_dependency"),
                snapshot_blocked=bool(test.get("data_dependency") in blocked_deps),
                truth_covered=False, truth_match=None, truth_detail="",
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

    # ── snapshot-blocked accounting — questions excluded from the headline
    # rate because a broken answer-layer load (not the agent) sank them ──────
    snap_ids   = [r["id"] for r in results if r.get("snapshot_blocked")]
    n_snap     = len(snap_ids)
    eff_total  = total - n_snap
    real_pass_adjusted = sum(1 for r in results
                             if r["real_pass"] and not r.get("snapshot_blocked"))
    tot_se     = sum(r["sql_errors"]   for r in results if r["sql_errors"]   >= 0)
    tot_pe     = sum(r["parse_errors"] for r in results if r["parse_errors"] >= 0)

    # ── latency breakdown — Phase 3.1/3.2 profiling, mirrored from
    # eval_audit.py — see QueryStats.total_sql_gen_ms/reasoning_prefill_ms
    # docstrings for what these fields mean and why sql_gen is tracked
    # separately from raw Postgres exec time. ──────────────────────────────
    _ok = [r for r in results if r["llm_calls"] >= 0]
    _tot_wall_ms   = sum(r["wall_s"] * 1000 for r in _ok)
    _tot_reason_ms = sum(r["reasoning_ms"] for r in _ok)
    _tot_sqlgen_ms = sum(r["sql_gen_ms"]   for r in _ok)
    _tot_sqlexec_ms= sum(r["sql_exec_ms"]  for r in _ok)
    _tot_other_ms  = max(0, _tot_wall_ms - _tot_reason_ms - _tot_sqlgen_ms - _tot_sqlexec_ms)
    _pct = lambda part: round(100 * part / _tot_wall_ms) if _tot_wall_ms else 0
    _tot_reason_pf = sum(r["reasoning_prefill_ms"] for r in _ok)
    _tot_reason_dc = sum(r["reasoning_decode_ms"]  for r in _ok)
    _tot_sqlgen_pf = sum(r["sql_gen_prefill_ms"]   for r in _ok)
    _tot_sqlgen_dc = sum(r["sql_gen_decode_ms"]    for r in _ok)
    _sub_pct = lambda part, whole: round(100 * part / whole) if whole else 0

    # ── prompt-composition breakdown — Phase 3.8 profiling, only non-zero
    # when run with --profile-tokens (see QueryStats.sql_gen_question_tokens
    # docstring). _profiled_calls is the divisor for per-call averages, not
    # len(_ok) — a turn can dispatch generate_sql 0+ times. ─────────────────
    _tot_q_tok    = sum(r["sql_gen_question_tokens"]           for r in _ok)
    _tot_sh_tok   = sum(r["sql_gen_schema_hint_tokens"]         for r in _ok)
    _tot_cc_tok   = sum(r["sql_gen_core_context_tokens"]        for r in _ok)
    _tot_re_tok   = sum(r["sql_gen_retrieved_example_tokens"]   for r in _ok)
    _tot_total_tok= sum(r["sql_gen_total_prompt_tokens"]        for r in _ok)
    _profiled_calls = sum(r["sql_gen_profiled_calls"]           for r in _ok)
    _comp_pct = lambda part: round(100 * part / _tot_total_tok) if _tot_total_tok else 0

    # ── schema_hint sub-composition — Phase 3.9 profiling, only non-zero
    # when run with --profile-tokens. Percentages below are relative to
    # _tot_sh_tok (schema_hint's own total, from the 3.8 block above), not
    # the full prompt — this section explains schema_hint's internals, not
    # the whole prompt's. ────────────────────────────────────────────────
    _tot_auto_tok = sum(r["sql_gen_auto_schema_tokens"]     for r in _ok)
    _tot_mh_tok   = sum(r["sql_gen_mandatory_hint_tokens"]  for r in _ok)
    _tot_notes_tok= sum(r["sql_gen_planner_notes_tokens"]   for r in _ok)
    _tot_jp_tok   = sum(r["sql_gen_join_path_tokens"]       for r in _ok)
    _tot_br_tok   = sum(r["sql_gen_business_rules_tokens"]  for r in _ok)
    _tot_draft_tok= sum(r["sql_gen_draft_sql_tokens"]       for r in _ok)
    _sh_pct = lambda part: round(100 * part / _tot_sh_tok, 1) if _tot_sh_tok else 0.0

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
    if n_snap:
        print(f"  Real pass*:   {real_pass_adjusted}/{eff_total} "
              f"({100 * real_pass_adjusted // eff_total if eff_total else 0}%)"
              f"  [* excludes {n_snap} snapshot-blocked: {', '.join(snap_ids)}]")
    print(f"  SQL success:  {sql_ok}/{total} "
          f"({100 * sql_ok // total if total else 0}%)")
    print(f"  Single-call:  {single}/{total} "
          f"({100 * single // total if total else 0}%)")
    print(f"  Avg time:     {avg_t:.1f}s")
    print(f"  Avg kw hit:   {avg_kp}%  (over {len(kp_vals)} keyword-tested questions)")
    print(f"  SQL errors:   {tot_se}  |  Parse errors: {tot_pe}")
    print(f"\n  LATENCY BREAKDOWN (of {_tot_wall_ms/1000:.1f}s total wall time, {len(_ok)} queries)")
    print(f"    reasoning LLM (chat/classify/synthesize): {_tot_reason_ms/1000:>7.1f}s  ({_pct(_tot_reason_ms):>3d}%)")
    print(f"    SQL-gen (retrieval+prompt+coder LLM+val): {_tot_sqlgen_ms/1000:>7.1f}s  ({_pct(_tot_sqlgen_ms):>3d}%)")
    print(f"    SQL execution (Postgres):                 {_tot_sqlexec_ms/1000:>7.1f}s  ({_pct(_tot_sqlexec_ms):>3d}%)")
    print(f"    other/unaccounted overhead:                {_tot_other_ms/1000:>7.1f}s  ({_pct(_tot_other_ms):>3d}%)")
    if _tot_reason_pf + _tot_reason_dc + _tot_sqlgen_pf + _tot_sqlgen_dc:
        print(f"\n  PREFILL vs DECODE SPLIT (llama.cpp `timings` field)")
        print(f"    reasoning: prefill {_tot_reason_pf/1000:>6.1f}s ({_sub_pct(_tot_reason_pf, _tot_reason_ms):>3d}%)"
              f"  |  decode {_tot_reason_dc/1000:>6.1f}s ({_sub_pct(_tot_reason_dc, _tot_reason_ms):>3d}%)")
        print(f"    sql-gen  : prefill {_tot_sqlgen_pf/1000:>6.1f}s ({_sub_pct(_tot_sqlgen_pf, _tot_sqlgen_ms):>3d}%)"
              f"  |  decode {_tot_sqlgen_dc/1000:>6.1f}s ({_sub_pct(_tot_sqlgen_dc, _tot_sqlgen_ms):>3d}%)")

    if _profiled_calls:
        # reconciliation: the four dynamic components vs the model's own
        # reported total for the same prompt (see generation.py's "First
        # call only" comment) — a large unreconciled delta means either the
        # static template scaffolding (expected, and fine) or an
        # undocumented fifth prompt component (worth investigating).
        _sum_components = _tot_q_tok + _tot_sh_tok + _tot_cc_tok + _tot_re_tok
        _unreconciled = _tot_total_tok - _sum_components
        print(f"\n  PROMPT COMPOSITION (Phase 3.8, {_profiled_calls} profiled sql-gen calls, real /tokenize counts)")
        print(f"    question          : {_tot_q_tok:>7d} tok  avg {_tot_q_tok/_profiled_calls:>6.1f}  ({_comp_pct(_tot_q_tok):>3d}% of total)")
        print(f"    schema_hint       : {_tot_sh_tok:>7d} tok  avg {_tot_sh_tok/_profiled_calls:>6.1f}  ({_comp_pct(_tot_sh_tok):>3d}% of total)  [dynamic, uncached]")
        print(f"    core_context      : {_tot_cc_tok:>7d} tok  avg {_tot_cc_tok/_profiled_calls:>6.1f}  ({_comp_pct(_tot_cc_tok):>3d}% of total)  [static, cacheable]")
        print(f"    retrieved_examples: {_tot_re_tok:>7d} tok  avg {_tot_re_tok/_profiled_calls:>6.1f}  ({_comp_pct(_tot_re_tok):>3d}% of total)  [dynamic, uncached]")
        print(f"    template/other    : {_unreconciled:>7d} tok  avg {_unreconciled/_profiled_calls:>6.1f}  ({_comp_pct(_unreconciled):>3d}% of total)  [static scaffolding + reconciliation slack]")
        print(f"    total (model-reported): {_tot_total_tok:>7d} tok  avg {_tot_total_tok/_profiled_calls:>6.1f}")

        # Phase 3.9 — schema_hint's own sub-composition. boilerplate is
        # derived by subtraction (the "#### ..." headers/labels
        # _build_additional_context adds around the six atoms below) — if
        # this goes negative or the printed total doesn't equal _tot_sh_tok
        # exactly, an atom was missed; stop and debug before trusting the
        # distribution (see the 3.9 plan's reconciliation gate).
        _sh_sum_atoms = (_tot_auto_tok + _tot_mh_tok + _tot_notes_tok
                         + _tot_jp_tok + _tot_br_tok + _tot_draft_tok)
        _sh_boilerplate = _tot_sh_tok - _sh_sum_atoms
        print(f"\n  SCHEMA_HINT BREAKDOWN (Phase 3.9, {_profiled_calls} profiled sql-gen calls, % of schema_hint total)")
        print(f"    auto_schema       : {_tot_auto_tok:>7d} tok  avg {_tot_auto_tok/_profiled_calls:>6.1f}  ({_sh_pct(_tot_auto_tok):>5.1f}%)")
        print(f"    mandatory_hint    : {_tot_mh_tok:>7d} tok  avg {_tot_mh_tok/_profiled_calls:>6.1f}  ({_sh_pct(_tot_mh_tok):>5.1f}%)")
        print(f"    planner_notes     : {_tot_notes_tok:>7d} tok  avg {_tot_notes_tok/_profiled_calls:>6.1f}  ({_sh_pct(_tot_notes_tok):>5.1f}%)")
        print(f"    join_path         : {_tot_jp_tok:>7d} tok  avg {_tot_jp_tok/_profiled_calls:>6.1f}  ({_sh_pct(_tot_jp_tok):>5.1f}%)")
        print(f"    business_rules    : {_tot_br_tok:>7d} tok  avg {_tot_br_tok/_profiled_calls:>6.1f}  ({_sh_pct(_tot_br_tok):>5.1f}%)")
        print(f"    draft_sql         : {_tot_draft_tok:>7d} tok  avg {_tot_draft_tok/_profiled_calls:>6.1f}  ({_sh_pct(_tot_draft_tok):>5.1f}%)")
        print(f"    boilerplate       : {_sh_boilerplate:>7d} tok  avg {_sh_boilerplate/_profiled_calls:>6.1f}  ({_sh_pct(_sh_boilerplate):>5.1f}%)  [headers/labels, by subtraction]")
        print(f"    {'-'*58}")
        print(f"    schema_hint total : {_tot_sh_tok:>7d} tok  avg {_tot_sh_tok/_profiled_calls:>6.1f}  (100.0%)")

    if _reasoning_profile and _reasoning_call_samples:
        # Phase 3.17 Part A — reasoning-LLM prompt composition. First-ever
        # token instrumentation of this prompt (see agent/prompts/builder.py's
        # profile_prompt_composition docstring) — static base is invariant
        # per process (computed once above), dynamic additions are the real
        # mean of every profiled question's actual build_dynamic_prompt_
        # additions() output (Part A2, per-question loop above).
        _static_tot = _reasoning_profile["system_prompt_total_tokens"]
        _n_rc = len(_reasoning_call_samples)
        _avg_dyn = sum(s["dynamic_additions_tokens"] for s in _reasoning_call_samples) / _n_rc
        print(f"\n  REASONING PROMPT COMPOSITION (Phase 3.17, {_n_rc} profiled reasoning calls, real /tokenize counts)")
        print(f"    static base (always-on, computed once): {_static_tot:>7d} tok")
        for _label, _info in sorted(_reasoning_profile["system_prompt_sections"].items(),
                                     key=lambda kv: -kv[1]["tokens"]):
            print(f"      {_label:32s}: {_info['tokens']:>6d} tok")
        print(f"    dynamic additions (per call, avg)      : {_avg_dyn:>7.1f} tok"
              f"  ({100*_avg_dyn/(_static_tot+_avg_dyn):.1f}% of static+dynamic total)")
        print(f"    static + avg dynamic (per-call estimate): {_static_tot + _avg_dyn:>7.1f} tok")
        print(f"\n    CONDITIONAL_BLOCKS / EXAMPLES_TIER2 firing rate (of {_n_rc} profiled questions):")
        _all_titles = (
            [(t, "CONDITIONAL") for t in _reasoning_profile["conditional_blocks"]]
            + [(t, "TIER2") for t in _reasoning_profile["examples_tier2"]]
        )
        for _title, _kind in _all_titles:
            _hits = _reasoning_title_hits.get(_title, 0)
            if _hits:
                _src = _reasoning_profile["conditional_blocks"] if _kind == "CONDITIONAL" else _reasoning_profile["examples_tier2"]
                print(f"      [{_kind:11s}] {_hits:>3d}/{_n_rc:<3d} ({_src[_title]['tokens']:>5d} tok)  {_title}")
        _n_zero_match = sum(
            1 for s in _reasoning_call_samples if s["dynamic_additions_tokens"] == 0
        )
        print(f"    ({_n_zero_match}/{_n_rc} questions matched nothing — 0 dynamic-addition tokens)")
        print(f"    NOTE: conversation-history cost (session.build_context_block) is NOT")
        print(f"    measured here — see Part A2's scope note. A real logged production case")
        print(f"    hit 83,456 total system_content chars by turn 18 (session.py:168-173).")

    if _core_context_profile and _core_context_call_samples:
        # Phase 3.17 Part B — CORE_CONTEXT / SQL_GENERATION_PROMPT skeleton
        # composition, extending Phase 3.8's opaque core_context_tokens /
        # template_or_other top-level totals with a real breakdown.
        _cc_static = _core_context_profile["core_context_static_total_tokens"]
        _n_cc = len(_core_context_call_samples)
        _avg_cc_dyn = sum(s["additions_tokens"] for s in _core_context_call_samples) / _n_cc
        print(f"\n  CORE_CONTEXT / SKELETON COMPOSITION (Phase 3.17 Part B, {_n_cc} profiled sql-gen calls)")
        print(f"    CORE_CONTEXT static base (always-on)   : {_cc_static:>7d} tok")
        for _name, _info in _core_context_profile["named_blocks"].items():
            print(f"      {_name:24s}: {_info['tokens']:>6d} tok")
        print(f"      {'prose (by subtraction)':24s}: {_core_context_profile['prose_only_tokens']:>6d} tok")
        print(f"    SQL_GENERATION_PROMPT skeleton (always-on): {_core_context_profile['sql_generation_prompt_skeleton_tokens']:>7d} tok")
        print(f"    CORE_CONTEXT_CONDITIONAL additions (per call, avg): {_avg_cc_dyn:>7.1f} tok")
        for _blk in CORE_CONTEXT_CONDITIONAL:
            _hits = _core_context_title_hits.get(_blk["title"], 0)
            if _hits:
                _tok = _core_context_profile["conditional_blocks"][_blk["title"]]["tokens"]
                print(f"      [{_hits:>3d}/{_n_cc:<3d} ({_tok:>5d} tok)]  {_blk['title']}")
        if _profiled_calls and _tot_re_tok:
            # top_k=4 fixed (generation.py:490-492) — dividing the measured
            # per-call retrieved_examples total by it gives an "avg tokens
            # per example slot" figure (a slight underestimate on any call
            # that retrieved fewer than 4, e.g. very early retries), useful
            # for judging whether top_k=4 looks oversized without new
            # per-example instrumentation.
            _avg_re_per_call = _tot_re_tok / _profiled_calls
            print(f"    retrieved_examples avg/call: {_avg_re_per_call:>7.1f} tok  "
                  f"(~{_avg_re_per_call/4:.1f} tok/example at fixed top_k=4)")

    if _schema_profile and _n_utilization_queries:
        # Phase 3.10 — Part A: static composition of the slim schema block
        # (see DatabaseSchema.profile_slim_schema_tokens — computed once,
        # not per query, since the content is question-independent).
        _kc_total = _schema_profile["key_columns_total"]
        _lt_total = _schema_profile["lookup_tables_total"]
        _fk_total = _schema_profile["fk_joins_total"]
        _static_total = _schema_profile["static_block_total"]
        print(f"\n  SCHEMA BLOCK COMPOSITION (Phase 3.10, static — computed once, real /tokenize counts)")
        print(f"    key_columns   ({len(_schema_profile['key_columns']):>2} tables): {_kc_total:>5d} tok")
        print(f"    lookup_tables ({len(_schema_profile['lookup_tables']):>2} tables): {_lt_total:>5d} tok")
        print(f"    fk_joins      ({len(_schema_profile['fk_joins']):>2} FKs)   : {_fk_total:>5d} tok")
        print(f"    static block total          : {_static_total:>5d} tok")

        # Phase 3.10 — Part B: does the generated SQL actually reference the
        # tables the block spent tokens describing? fk_joins excluded from
        # this diff (its lines aren't attributable to one table the way a
        # key_columns/lookup_tables line is — see the accumulation comment
        # above in the main loop).
        _tot_sh_incl = sum(r["schema_tables_included"] for r in _ok)
        _tot_sh_used = sum(r["schema_tables_used"] for r in _ok)
        _tot_sh_wasted_tok = sum(r["schema_tables_unused_tokens"] for r in _ok)
        _diffable_total = _kc_total + _lt_total
        _used_pct = round(100 * _tot_sh_used / _tot_sh_incl) if _tot_sh_incl else 0
        _wasted_pct = (round(100 * (_tot_sh_wasted_tok / _n_utilization_queries) / _diffable_total)
                       if _diffable_total else 0)
        print(f"\n  SCHEMA UTILIZATION (Phase 3.10, {_n_utilization_queries} queries — key_columns + lookup_tables only, {_diffable_total} tok diffable)")
        if _n_targeted_variant_queries:
            print(f"    ({_n_targeted_variant_queries} additional queries used the \"targeted\" for_schema_hint() "
                  f"branch instead of the static slim block — excluded from this diff, see schema_block_variant)")
        print(f"    avg tables included    : {_tot_sh_incl/_n_utilization_queries:.1f}")
        print(f"    avg tables used         : {_tot_sh_used/_n_utilization_queries:.1f}  ({_used_pct}%)")
        print(f"    avg unused tokens/query : {_tot_sh_wasted_tok/_n_utilization_queries:.1f}  ({_wasted_pct}% of diffable block)")

        print(f"\n    Table usage across {_n_utilization_queries} queries (ranked by cost x unused-frequency, biggest opportunity first):")
        _ranked = sorted(
            _schema_tables_cost.items(),
            key=lambda kv: kv[1] * _table_unused_count.get(kv[0], 0), reverse=True,
        )
        for tname, cost in _ranked:
            _unused_n = _table_unused_count.get(tname, 0)
            _pct = round(100 * _unused_n / _n_utilization_queries)
            print(f"      {tname:<28} {cost:>4} tok  unused in {_unused_n:>3}/{_n_utilization_queries} ({_pct:>3d}%)")

    if _targeted_call_samples:
        # Phase 3.11 — targeted-branch (for_schema_hint) decomposition.
        # Exact per-call reconstruction (dict lookup against Part A's
        # once-per-run per-table costs), not the run-level subtraction
        # estimate the first 3.10 pass used.
        _n_tgt = len(_targeted_call_samples)
        _pctile = lambda vals, p: (lambda s: s[min(len(s) - 1, int(round(p * (len(s) - 1))))])(sorted(vals))
        _real_totals = [s["real_total"] for s in _targeted_call_samples]

        _tot_cols = sum(s["columns_tokens"] for s in _targeted_call_samples)
        _tot_fk = sum(s["fk_tokens"] for s in _targeted_call_samples)
        _tot_enum = sum(s["enum_tokens"] for s in _targeted_call_samples)
        _tot_useful = sum(s["useful_output_tokens"] for s in _targeted_call_samples)
        _tot_header = sum(s["header_tokens"] for s in _targeted_call_samples)
        _tot_lookup = sum(s["lookup_section_tokens"] for s in _targeted_call_samples)
        _tot_rel = sum(s["table_relationships_tokens"] for s in _targeted_call_samples)
        _tot_reconstructed = sum(s["reconstructed_total"] for s in _targeted_call_samples)
        _tot_real = sum(_real_totals)
        _tot_slack = _tot_real - _tot_reconstructed  # join separators + BPE boundary effects
        _tgt_pct = lambda part: round(100 * part / _tot_real) if _tot_real else 0

        print(f"\n  TARGETED-BRANCH COMPOSITION (Phase 3.11, {_n_tgt} calls, exact per-call reconstruction)")
        print(f"    per-table columns/types : {_tot_cols:>6d} tok  avg {_tot_cols/_n_tgt:>7.1f}  ({_tgt_pct(_tot_cols):>3d}%)")
        print(f"    per-table FK details     : {_tot_fk:>6d} tok  avg {_tot_fk/_n_tgt:>7.1f}  ({_tgt_pct(_tot_fk):>3d}%)")
        print(f"    per-table enum values    : {_tot_enum:>6d} tok  avg {_tot_enum/_n_tgt:>7.1f}  ({_tgt_pct(_tot_enum):>3d}%)")
        print(f"    useful output columns    : {_tot_useful:>6d} tok  avg {_tot_useful/_n_tgt:>7.1f}  ({_tgt_pct(_tot_useful):>3d}%)")
        print(f"    table headers/labels     : {_tot_header:>6d} tok  avg {_tot_header/_n_tgt:>7.1f}  ({_tgt_pct(_tot_header):>3d}%)")
        print(f"    lookup section (fixed)   : {_tot_lookup:>6d} tok  avg {_tot_lookup/_n_tgt:>7.1f}  ({_tgt_pct(_tot_lookup):>3d}%)  [same every call, regardless of table_count]")
        print(f"    table_relationships (fixed): {_tot_rel:>4d} tok  avg {_tot_rel/_n_tgt:>7.1f}  ({_tgt_pct(_tot_rel):>3d}%)  [same every call, regardless of table_count]")
        print(f"    join separators/slack    : {_tot_slack:>6d} tok  avg {_tot_slack/_n_tgt:>7.1f}  ({_tgt_pct(_tot_slack):>3d}%)  [reconstruction vs real, by subtraction]")
        print(f"    {'-'*68}")
        print(f"    real total (measured)    : {_tot_real:>6d} tok  avg {_tot_real/_n_tgt:>7.1f}")

        _fixed_total = _tot_lookup + _tot_rel
        print(f"\n    Fixed overhead (lookup + table_relationships) is {_tgt_pct(_fixed_total)}% of every "
              f"targeted call, independent of table_count — this is baseline cost the \"targeted\" branch "
              f"pays even for a single named table.")

        # Hypothesis A: is the planner naming more tables than the SQL uses?
        _avg_selected = sum(s["table_count"] for s in _targeted_call_samples) / _n_tgt
        _avg_planner = sum(s["planner_table_count"] for s in _targeted_call_samples) / _n_tgt
        _avg_sql = sum(s["sql_table_count"] for s in _targeted_call_samples) / _n_tgt
        print(f"\n  TARGETED-BRANCH TABLE SELECTION (Phase 3.11, hypothesis A)")
        print(f"    avg tables named (rendered): {_avg_selected:.1f}")
        print(f"    avg tables in generated SQL: {_avg_sql:.1f}")
        if _avg_selected:
            print(f"    over-selection ratio       : {_avg_selected/_avg_sql:.2f}x" if _avg_sql else "    over-selection ratio       : n/a (no SQL tables extracted)")

        # Distribution: is the ~4x targeted-vs-slim gap one consistent cost,
        # or driven by a handful of outlier calls naming many tables?
        print(f"\n  TARGETED-BRANCH real_total DISTRIBUTION (Phase 3.11, {_n_tgt} calls)")
        print(f"    min={min(_real_totals):>5d}  p50={_pctile(_real_totals,0.50):>5d}  p75={_pctile(_real_totals,0.75):>5d}  "
              f"p90={_pctile(_real_totals,0.90):>5d}  p95={_pctile(_real_totals,0.95):>5d}  max={max(_real_totals):>5d}")

    if _fixed_overhead_calls:
        # Phase 3.12 — relevance of the two fixed blocks (lookup + table
        # relationships) 3.11 identified as ~78% of every targeted call.
        _n_fov = len(_fixed_overhead_calls)
        _all_items = [(cat, key, item) for cat in ("lookup_tables", "status_enums", "join_patterns")
                      for key, item in _fixed_overhead_profile[cat].items()]
        _all_items += [("relationship_sections", title, item)
                       for title, item in _fixed_overhead_profile["relationship_sections"].items()]
        _fixed_total_tokens = sum(item["tokens"] for _, _, item in _all_items)
        _avg_wasted_named = sum(c["wasted_named_tokens"] for c in _fixed_overhead_calls) / _n_fov
        _avg_wasted_used = sum(c["wasted_used_tokens"] for c in _fixed_overhead_calls) / _n_fov

        print(f"\n  FIXED-OVERHEAD RELEVANCE (Phase 3.12, {_n_fov} targeted calls, {_fixed_total_tokens} tok fixed content)")
        print(f"    avg tokens irrelevant to NAMED tables  : {_avg_wasted_named:.1f}  ({round(100*_avg_wasted_named/_fixed_total_tokens) if _fixed_total_tokens else 0}% of fixed content)")
        print(f"    avg tokens irrelevant to GENERATED SQL : {_avg_wasted_used:.1f}  ({round(100*_avg_wasted_used/_fixed_total_tokens) if _fixed_total_tokens else 0}% of fixed content)")

        print(f"\n    Per-item relevance (ranked by cost x irrelevant-to-generated-SQL frequency, biggest opportunity first):")
        _ranked_items = sorted(
            _all_items,
            key=lambda t: t[2]["tokens"] * (_n_fov - _fixed_item_relevant_used.get((t[0], t[1]), 0)),
            reverse=True,
        )
        for cat, key, item in _ranked_items:
            _used_n = _fixed_item_relevant_used.get((cat, key), 0)
            _named_n = _fixed_item_relevant_named.get((cat, key), 0)
            _label = key if len(key) <= 42 else key[:39] + "..."
            print(f"      [{cat:<20}] {_label:<42} {item['tokens']:>4} tok  "
                  f"relevant to SQL {_used_n:>3}/{_n_fov} ({round(100*_used_n/_n_fov):>3d}%)  "
                  f"relevant to named {_named_n:>3}/{_n_fov} ({round(100*_named_n/_n_fov):>3d}%)")

    # ── ground-truth summary ─────────────────────────────────────────────────
    truth_rows = [r for r in results if r.get("truth_covered")]
    truth_pass = sum(1 for r in truth_rows if r.get("truth_match") is True)
    truth_fail = [r for r in truth_rows if r.get("truth_match") is False]
    truth_err  = [r for r in truth_rows if r.get("truth_match") is None]
    truth_rows_eff = [r for r in truth_rows if not r.get("snapshot_blocked")]
    truth_pass_adjusted = sum(1 for r in truth_rows_eff if r.get("truth_match") is True)
    if truth_rows:
        print(f"  TRUTH MATCH:  {truth_pass}/{len(truth_rows)} "
              f"({100 * truth_pass // len(truth_rows)}%)  "
              f"— result-level correctness on covered cases")
        if len(truth_rows_eff) != len(truth_rows):
            print(f"  TRUTH MATCH*: {truth_pass_adjusted}/{len(truth_rows_eff)} "
                  f"({100 * truth_pass_adjusted // len(truth_rows_eff) if truth_rows_eff else 0}%)"
                  f"  [* excludes snapshot-blocked]")
        for r in truth_fail:
            tag = " (snapshot-blocked)" if r.get("snapshot_blocked") else ""
            print(f"    ✗ {r['id']:<10} {r['truth_detail'][:80]}{tag}")
        for r in truth_err:
            print(f"    ? {r['id']:<10} {r['truth_detail'][:80]}")

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
            truth_covered=len(truth_rows),
            truth_pass=truth_pass,
            data_health=data_health,
            snapshot_blocked=n_snap,
            snapshot_blocked_ids=snap_ids,
            real_pass_adjusted=real_pass_adjusted,
            truth_covered_adjusted=len(truth_rows_eff),
            truth_pass_adjusted=truth_pass_adjusted,
            latency_breakdown_ms=dict(
                total_wall=round(_tot_wall_ms), reasoning=_tot_reason_ms,
                sql_gen=_tot_sqlgen_ms, sql_exec=_tot_sqlexec_ms, other=_tot_other_ms,
                reasoning_prefill=_tot_reason_pf, reasoning_decode=_tot_reason_dc,
                sql_gen_prefill=_tot_sqlgen_pf, sql_gen_decode=_tot_sqlgen_dc,
            ),
            # Phase 3.8 profiling — all zero/absent unless run with
            # --profile-tokens. See PROMPT COMPOSITION console section above.
            prompt_composition_tokens=dict(
                profiled_calls=_profiled_calls,
                question=_tot_q_tok, schema_hint=_tot_sh_tok,
                core_context=_tot_cc_tok, retrieved_examples=_tot_re_tok,
                template_or_other=(_tot_total_tok - _tot_q_tok - _tot_sh_tok - _tot_cc_tok - _tot_re_tok),
                total_model_reported=_tot_total_tok,
            ) if _profiled_calls else None,
            # Phase 3.9 profiling — all zero/absent unless run with
            # --profile-tokens. See SCHEMA_HINT BREAKDOWN console section
            # above; boilerplate = schema_hint - sum(the 6 atoms).
            schema_hint_breakdown_tokens=dict(
                profiled_calls=_profiled_calls,
                auto_schema=_tot_auto_tok, mandatory_hint=_tot_mh_tok,
                planner_notes=_tot_notes_tok, join_path=_tot_jp_tok,
                business_rules=_tot_br_tok, draft_sql=_tot_draft_tok,
                boilerplate=(_tot_sh_tok - _tot_auto_tok - _tot_mh_tok - _tot_notes_tok
                             - _tot_jp_tok - _tot_br_tok - _tot_draft_tok),
                schema_hint_total=_tot_sh_tok,
            ) if _profiled_calls else None,
            # Phase 3.17 Part A profiling — absent unless run with
            # --profile-tokens. See REASONING PROMPT COMPOSITION console
            # section above. static_base is invariant per process (one-time
            # profile_prompt_composition() call); per_call_dynamic_additions
            # is one real /tokenize sample per profiled question (Part A2);
            # conversation-history cost is NOT included (see Part A2's
            # scope note in the per-question loop above).
            reasoning_prompt_composition_tokens=dict(
                profiled_calls=len(_reasoning_call_samples),
                static_base=_reasoning_profile,
                per_call_dynamic_additions=_reasoning_call_samples,
                conditional_block_examples_tier2_hits=dict(_reasoning_title_hits),
            ) if _reasoning_profile and _reasoning_call_samples else None,
            # Phase 3.17 Part B profiling — absent unless run with
            # --profile-tokens. See CORE_CONTEXT / SKELETON COMPOSITION
            # console section above.
            core_context_composition_tokens=dict(
                profiled_calls=len(_core_context_call_samples),
                static_base=_core_context_profile,
                per_call_additions=_core_context_call_samples,
                conditional_block_hits=dict(_core_context_title_hits),
            ) if _core_context_profile and _core_context_call_samples else None,
            # Phase 3.10 profiling — absent unless run with --profile-tokens.
            # schema_block_static_composition is Part A (computed once, see
            # DatabaseSchema.profile_slim_schema_tokens); schema_utilization
            # is Part B (per-query diff against _diagnostics.generated_tables,
            # aggregated — see SCHEMA UTILIZATION console section above).
            schema_block_static_composition=_schema_profile,
            schema_utilization=dict(
                queries=_n_utilization_queries,
                targeted_variant_queries=_n_targeted_variant_queries,
                avg_tables_included=(_tot_sh_incl / _n_utilization_queries),
                avg_tables_used=(_tot_sh_used / _n_utilization_queries),
                avg_unused_tokens=(_tot_sh_wasted_tok / _n_utilization_queries),
                table_unused_frequency=dict(_table_unused_count),
                table_used_frequency=dict(_table_used_count),
            ) if _n_utilization_queries else None,
            # Phase 3.11 profiling — absent unless run with --profile-tokens
            # and at least one "targeted"-variant call occurred. See
            # TARGETED-BRANCH console sections above. targeted_calls is the
            # raw per-call sample list (one row per targeted generate_sql
            # call) — kept for offline re-aggregation (e.g. re-deriving
            # percentiles at a different cut) without re-running the suite.
            targeted_branch_decomposition=dict(
                calls=len(_targeted_call_samples),
                avg_tables_named=(sum(s["table_count"] for s in _targeted_call_samples) / len(_targeted_call_samples)),
                avg_planner_tables=(sum(s["planner_table_count"] for s in _targeted_call_samples) / len(_targeted_call_samples)),
                avg_sql_tables=(sum(s["sql_table_count"] for s in _targeted_call_samples) / len(_targeted_call_samples)),
                composition_tokens=dict(
                    columns=sum(s["columns_tokens"] for s in _targeted_call_samples),
                    fk=sum(s["fk_tokens"] for s in _targeted_call_samples),
                    enum=sum(s["enum_tokens"] for s in _targeted_call_samples),
                    useful_output=sum(s["useful_output_tokens"] for s in _targeted_call_samples),
                    headers=sum(s["header_tokens"] for s in _targeted_call_samples),
                    lookup_section_fixed=sum(s["lookup_section_tokens"] for s in _targeted_call_samples),
                    table_relationships_fixed=sum(s["table_relationships_tokens"] for s in _targeted_call_samples),
                    reconstruction_slack=(sum(s["real_total"] for s in _targeted_call_samples)
                                          - sum(s["reconstructed_total"] for s in _targeted_call_samples)),
                    real_total=sum(s["real_total"] for s in _targeted_call_samples),
                ),
                real_total_distribution=dict(
                    min=min(s["real_total"] for s in _targeted_call_samples),
                    max=max(s["real_total"] for s in _targeted_call_samples),
                    values_sorted=sorted(s["real_total"] for s in _targeted_call_samples),
                ),
                targeted_calls=_targeted_call_samples,
            ) if _targeted_call_samples else None,
            # Phase 3.12 profiling — absent unless run with --profile-tokens
            # and at least one "targeted"-variant call occurred. See
            # FIXED-OVERHEAD RELEVANCE console section above.
            fixed_overhead_relevance=dict(
                calls=len(_fixed_overhead_calls),
                fixed_total_tokens=sum(item["tokens"] for cat in ("lookup_tables", "status_enums", "join_patterns")
                                       for item in _fixed_overhead_profile[cat].values())
                                   + sum(item["tokens"] for item in _fixed_overhead_profile["relationship_sections"].values()),
                avg_wasted_named_tokens=(sum(c["wasted_named_tokens"] for c in _fixed_overhead_calls) / len(_fixed_overhead_calls)),
                avg_wasted_used_tokens=(sum(c["wasted_used_tokens"] for c in _fixed_overhead_calls) / len(_fixed_overhead_calls)),
                items=[
                    dict(category=cat, key=key, tokens=item["tokens"],
                         relevant_to_named=_fixed_item_relevant_named.get((cat, key), 0),
                         relevant_to_generated_sql=_fixed_item_relevant_used.get((cat, key), 0))
                    for cat in ("lookup_tables", "status_enums", "join_patterns", "relationship_sections")
                    for key, item in _fixed_overhead_profile[cat].items()
                ],
                calls_detail=_fixed_overhead_calls,
            ) if _fixed_overhead_calls else None,
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
                        "Deferred Chain | Edge Case | Financial")
    p.add_argument("--output", default=None)
    p.add_argument("--profile-tokens", action="store_true",
                   help="Phase 3.8: tokenize dynamic prompt components (question/"
                        "schema_hint/core_context/retrieved examples) via llama.cpp's "
                        "/tokenize endpoint and report the breakdown. Adds a few extra "
                        "HTTP round-trips per generate_sql call; off by default.")
    a = p.parse_args()
    run_eval(a.model_pair, a.skip_multiturn, a.category, a.output, a.profile_tokens)