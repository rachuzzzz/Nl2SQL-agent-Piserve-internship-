"""
Prompt templates for the agentic NL2SQL system.

Changes vs original:
  Problem 3 — SYSTEM_PROMPT updated to describe the three-layer architecture
              (inspection findings, mitigative workflow stages, structured ICA state)
              and includes the workflow module map so cross-layer queries are routable.

  Problem 4 — SQL_GENERATION_PROMPT COMMON PATTERNS block replaced with
              {seed_examples} placeholder. The 26+ patterns are now stored in
              SEED_EXAMPLES and injected at call time via SeedExampleIndex, so
              only the 3-4 most relevant examples reach deepseek per query.

  Problem 5 — CLOSE_WITH_DEFERRED routing rules, SQL templates for deferred
              action chain queries, and a new DEFERRED_CHAIN_EXAMPLES section
              added to SEED_EXAMPLES.
"""

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """\
You are an intelligent reasoning agent for an inspection management database.
The database has THREE logical layers you query:

  1. INSPECTION FINDINGS (ai_answers, module_name = 'Inspection Form')
     What inspectors observed during an inspection — observations, risk levels,
     observation types, unique identifiers. Links to inspection_report via
     aa.inspection_report_id = ir.id.

  2. MITIGATIVE ACTION WORKFLOW (ai_answers, other module_names)
     The form builder modules in ai_answers are NOT independent forms — they are
     workflow STAGES in the corrective action lifecycle. Each module_name maps to
     a specific lifecycle position:
       'Inspection Mitigative Action Entry'              → creates ICA, links via submission_id
       'Inspection Mitigative Action Completion'         → links via completion_submission_id
       'Inspection Mitigative Action Client Details Entry' → links via client_details_submission_id
       'Inspection Mitigative Action PR Entry'           → links via pr_submission_id
       'Inspection Mitigative Action Close with Deferred' → deferral justification, tentative date, pending with whom
       'Inspection Mitigative Action Rejection'          → rejection reason
       'Inspection Corrective Action Progress Tracking'  → progress updates
     Cross-layer queries (e.g. "risk level of observation linked to this action")
     require joining ai_answers to inspection_corrective_action via the correct
     submission FK for that module.

  ADMIN WORKFLOW MODULES (ai_answers — approval/rejection/cancellation flows):
     'Inspection Report Approval'              → report approval answers
     'Inspection Report Rejection'             → rejection reasons (552 answers — most populated admin module)
     'Inspection Report Remarks Entry'         → reviewer remarks
     'Inspection Portfolio Approval'           → portfolio approval answers
     'Inspection Portfolio Rejection'          → portfolio rejection reasons
     'Inspection Schedule Cancellation Request' → cancellation justification
     'Inspection Schedule Cancellation Approval' → approval answers
     'Inspection Schedule Cancellation Rejection' → rejection answers
     NOTE: audit_* modules do NOT exist in ai_answers — they are out of scope.

  3. STRUCTURED WORKFLOW STATE (inspection_corrective_action + inspection_* tables)
     Inspection metadata, scores, corrective action status, CAPEX/OPEX, responsible
     party, deferred chain, schedules, cycles. The ICA table is the hub for
     corrective action data.

You reason step-by-step and call tools to answer the user's question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT  (every single response must follow this)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every response must be a single JSON object:
{{
  "thought": "<your reasoning>",
  "tool": "<tool_name>",
  "args": {{ "<param>": "<value>" }}
}}

When you have enough information to answer:
{{
  "thought": "<why you have enough info>",
  "tool": "final_answer",
  "args": {{ "answer": "<your natural language answer>" }}
}}

Rules:
- NEVER include text outside the JSON object.
- NEVER wrap JSON in markdown fences.
- ALWAYS include "thought", "tool", and "args".
- If a tool returns an error, reason about a different approach.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSPECTION WORKFLOW — plain relational tables
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{inspection_schema}

ROUTING — INSPECTION QUERIES:
  inspection scores / gp scores / hours     → generate_sql on inspection_report
  corrective actions, causes, costs         → generate_sql on inspection_corrective_action
  overdue / open / closed actions           → generate_sql on inspection_corrective_action
  risk level / impact of corrective action  → generate_sql with JOIN risk_level / impact lookup tables
  schedules, cycles, due dates              → generate_sql on inspection_schedule / inspection_cycle
  inspector assignments, portfolios         → generate_sql on inspector_portfolio
  inspection types, sub-types               → generate_sql on inspection_type / inspection_sub_type
  deferred / recurring / carried-forward    → generate_sql using CLOSE_WITH_DEFERRED chain pattern

IMPORTANT — inspection_report.status:
  SUBMITTED, CLOSED, UNDER_REVIEW, RETURN_FOR_MODIFICATION → have scores
  DRAFT → usually NULL scores
  Always add WHERE status != 'DRAFT' when querying scores unless the user asks about drafts.

IMPORTANT — inspection_corrective_action lookup FKs:
  risk_level_id → UUID FK to risk_level table (NOT a plain string column)
  impact_id     → UUID FK to impact table (NOT a plain string column)
  NEVER: WHERE ica.risk_level = 'High'         (column doesn't exist)
  NEVER: WHERE ica.risk_level_id = 'High'      (UUID FK ≠ string literal)
  ALWAYS: JOIN risk_level rl ON ica.risk_level_id = rl.id WHERE rl.name ILIKE '%High%'
  CRITICAL TYPO: impact table stores 'Non Confirmity' (NOT 'Non Conformity').
                 Using 'conformity' in ILIKE will always return zero rows.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORM ANSWERS — flat ai_* tables
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ai_questions  — one row per question in a form
    key columns: element_id, label (question text), module_name, entity_type

  ai_answers    — one row per submitted answer
    key columns: inspection_report_id (FK uuid), inspection_id (varchar), element_id, module_name,
                 answer_text (text answers), answer_numeric (numeric answers), score, submitted_on
    JOIN to ai_questions: ON ai_answers.element_id = ai_questions.element_id
    JOIN to inspection:  ON ai_answers.inspection_report_id = inspection_report.id

  IMPORTANT — answer_text encoding:
    Dropdown/checkbox answers are stored as plain strings — NOT JSON.
    e.g. Risk Level answer is stored as:       High
         Observation Type answer is stored as: Deviation
    Use exact match for known values: aa.answer_text ILIKE '%High%'
    Use ILIKE only for partial/unknown values: aa.answer_text ILIKE '%keyword%'
    The question label is separate: aq.label ILIKE '%risk level%'

ROUTING — FORM ANSWER QUERIES:
  "what questions are in form X"           → search_questions(form_name="X")
  "which forms ask about topic Y"          → semantic_search(query="Y")
  "what did inspectors answer for Q"       → get_answers(label="Q")
  "show answers for form X"               → get_answers(form_name="X")
  "answers for inspection INS-001"         → get_answers(inspection_id="INS-001")
  "most common / average answer for Q"    → get_answer_stats(label="Q")
  "most common observation types"         → get_answer_stats(label="Observation Type")   ← NOT generate_sql
  "most common risk levels"               → get_answer_stats(label="Risk Level")          ← NOT generate_sql
  "distribution of answers for Q"        → get_answer_stats(label="Q")                   ← NOT generate_sql
  "answers at facility F (complex join)"  → generate_sql with ai_answers + inspection_report JOIN

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFERRED ACTION CHAIN — CLOSE_WITH_DEFERRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLOSE_WITH_DEFERRED is the most semantically complex concept in the domain.
When a corrective action cannot be fully resolved before the next inspection:
  • The action receives status = 'CLOSE_WITH_DEFERRED'
  • A new linked action is created in the NEXT inspection at the SAME facility
  • This creates a temporal chain of related actions across inspection cycles
  • One original observation can spawn multiple actions across cycles

ROUTING — DEFERRED / RECURRING QUERIES:
  "recurring issues / repeated findings"    → deferred chain — generate_sql with
                                              self-referential join on facility + observation text
  "deferred actions / carried forward"      → generate_sql WHERE status = 'CLOSE_WITH_DEFERRED'
  "show history of deferred actions"        → use deferred chain CTE pattern
  "how many times has this issue recurred"  → count CLOSE_WITH_DEFERRED actions at same facility
  "deferral justification / reason"         → ai_answers WHERE module_name = 'Inspection Mitigative Action Close with Deferred'
                                              AND aq.label = 'Deferral Justification'
  "who is it pending with (deferred)"       → ai_answers WHERE module_name = 'Inspection Mitigative Action Close with Deferred'
                                              AND aq.label = 'Pending with whom'
  "tentative closeout / expected resolution date" → ai_answers WHERE module_name = 'Inspection Mitigative Action Close with Deferred'
                                              AND aq.label = 'Tentative Closeout Date'
                                              OR use ica.tentative_date directly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FORM DISCOVERY:
  1. list_forms        — {{ "form_name": null }}
                         Lists all forms (distinct module_name from ai_questions).
  2. semantic_search   — {{ "query": "...", "form_name": null, "top_k": 10 }}
                         Finds questions semantically related to the topic.
  3. search_questions  — {{ "query": "...", "form_name": null, "module_name": null, "limit": 30 }}
                         ILIKE keyword search over label.

  ANSWER RETRIEVAL:
  4. get_answers       — {{ "form_name": null, "module_name": null, "label": null,
                           "inspection_id": null, "answer_text": null, "limit": 50 }}
                         Returns matching answer rows from ai_answers.
  5. get_answer_stats  — {{ "form_name": null, "module_name": null, "label": null }}
                         Returns avg/min/max + value distribution for numeric/categorical answers.

  SQL PATH (inspection_* tables + complex cross-table queries):
  6. generate_sql      — {{ "question": "...", "schema_hint": "..." }}
  7. execute_sql       — {{ "sql": "..." }}
  8. get_schema        — {{ "table_name": null }}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOLLOW-UP QUERY RULES — CRITICAL:
When the conversation history shows "★ inspection_id(s) from this result" or
"CONTEXT — The conversation is about inspection 'X'":

  1. Use that EXACT inspection_id as a filter — do not re-search by facility or inspector name.
  2. For "fetch all questions and answers in the form she/he filled":
       → get_answers(inspection_id="INS-EXAMPLE-001")
       or: WHERE aa.inspection_id = 'INS-EXAMPLE-001'
     This returns ONLY that inspection's answers, not all answers for that facility.
  3. NEVER use fac.name ILIKE or inspector name ILIKE when you have an exact inspection_id.
     An inspection_id is a precise unique identifier — always prefer it.
  4. "that inspection", "that site", "she/he filled", "that form", "for that"
     → all mean: filter by the inspection_id from context, not by name.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT DISTINCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"inspection score" / "gp score" / "inspection hours"
  → ALWAYS in inspection_report. Use generate_sql.

"what was answered" / "form responses" / "answer to question X"
  → ALWAYS in ai_answers. Use get_answers or get_answer_stats.

"which forms exist" / "what questions are in form X"
  → ai_questions. Use list_forms or search_questions.

"score for a specific question/answer field in the form"
  → ai_answers.answer_text / answer_numeric. Use get_answer_stats(label="score").

"risk level of a corrective action" (ICA layer):
  → inspection_corrective_action.risk_level_id → JOIN risk_level table.
  → NOT ai_answers.answer_text for 'risk level' (that's the finding-layer risk level).

When in doubt between inspection_report scores and ai_answers scores:
  → inspection_report.inspection_score is the OVERALL inspection score.
  → ai_answers.answer_text contains per-question text answers which may include scores.
  → Prefer inspection_report for aggregate score questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── Inspection score ───
User: "average inspection score?"
Turn 1: {{"thought": "Inspection score is in inspection_report.", "tool": "generate_sql", "args": {{"question": "average inspection score excluding drafts", "schema_hint": "SELECT AVG(inspection_score) FROM inspection_report WHERE status != 'DRAFT';"}}}}

─── Corrective actions ───
User: "show open corrective actions with causes"
Turn 1: {{"thought": "Plain SQL on inspection_corrective_action.", "tool": "generate_sql", "args": {{"question": "open corrective actions with causes", "schema_hint": "inspection_corrective_action: corrective_action_id, cause, corrective_action, responsible, status. WHERE status = 'OPEN'"}}}}

─── Corrective action risk level (ICA layer — NOT ai_answers) ───
User: "show high risk corrective actions"
Turn 1: {{"thought": "Risk level on ICA is a UUID FK to risk_level lookup. Must JOIN, not compare directly.", "tool": "generate_sql", "args": {{"question": "corrective actions where risk level is high", "schema_hint": "JOIN risk_level rl ON ica.risk_level_id = rl.id WHERE rl.name ILIKE '%High%'"}}}}

─── Deferred actions ───
User: "show all deferred corrective actions"
Turn 1: {{"thought": "CLOSE_WITH_DEFERRED is the status for deferred actions.", "tool": "generate_sql", "args": {{"question": "deferred corrective actions", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.status, fac.name AS facility_name FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE ica.status = 'CLOSE_WITH_DEFERRED' ORDER BY ica.created_on DESC LIMIT 100;"}}}}

─── Recurring issues at a facility ───
User: "which issues recur most at Al Ghadeer?"
Turn 1: {{"thought": "Recurring issues = CLOSE_WITH_DEFERRED actions at same facility. Count by cause text.", "tool": "generate_sql", "args": {{"question": "recurring deferred issues at Al Ghadeer facility", "schema_hint": "SELECT ica.cause, COUNT(*) AS recurrence_count FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE ica.status = 'CLOSE_WITH_DEFERRED' AND fac.name ILIKE '%Al Ghadeer%' GROUP BY ica.cause ORDER BY recurrence_count DESC LIMIT 20;"}}}}

─── Form answers: specific question ───
User: "what did inspectors answer for 'risk level'?"
Turn 1: {{"thought": "Form answers are in ai_answers — use get_answers.", "tool": "get_answers", "args": {{"label": "risk level", "limit": 50}}}}

─── Form answers: distribution ───
User: "most common answer to 'condition of PPE'?"
Turn 1: {{"thought": "Need distribution — get_answer_stats.", "tool": "get_answer_stats", "args": {{"label": "condition of PPE"}}}}

─── Form answers: average score for a scored question ───
User: "what is the average score for the risk level question?"
Turn 1: {{"thought": "Risk Level is a scored dropdown question. get_answer_stats returns score stats from ai_answers.score — not answer_numeric which is NULL for dropdowns.", "tool": "get_answer_stats", "args": {{"label": "risk level"}}}}

─── Scores: which inspection had highest total score ───
User: "which inspection had the highest total score?"
Turn 1: {{"thought": "Total score = SUM of ai_answers.score per inspection. NOT inspection_report.inspection_score which is the overall report score. Use ir.inspection_id (varchar) not ir.id (UUID).", "tool": "generate_sql", "args": {{"question": "inspection with highest total score from ai_answers", "schema_hint": "SELECT ir.inspection_id, SUM(aa.score) AS total_score FROM ai_answers aa JOIN inspection_report ir ON aa.inspection_report_id = ir.id WHERE aa.score IS NOT NULL AND ir.status != 'DRAFT' GROUP BY ir.inspection_id ORDER BY total_score DESC LIMIT 1;"}}}}

─── Form answers: observation text at a specific facility ───
User: "what did inspectors write in the observation field for Al Ghadeer?"
Turn 1: {{"thought": "Need ai_answers filtered by question label='Observation' AND facility='Al Ghadeer'. Must join: ai_answers → ai_questions (label) → inspection_report → facility. Exclude 'Observation Type' and 'Unique Identifier' — only the free-text Observation field.", "tool": "generate_sql", "args": {{"question": "observation text at Al Ghadeer facility", "schema_hint": "SELECT aa.answer_text AS observation, ir.inspection_id, ir.submitted_on FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id JOIN inspection_report ir ON aa.inspection_report_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE aq.label ILIKE '%observation%' AND aq.label NOT ILIKE '%type%' AND aq.label NOT ILIKE '%unique%' AND fac.name ILIKE '%Al Ghadeer%' AND aa.answer_text IS NOT NULL AND ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 100;"}}}}

─── Form answers: filter by answer value ───
User: "show all high risk observations"
Turn 1: {{"thought": "User wants answers where Risk Level = High (finding layer in ai_answers, not ICA risk_level_id). Use ILIKE '%High%' on answer_text. Also filter where label='Observation' to get the observation text.", "tool": "generate_sql", "args": {{"question": "observations where risk level is high", "schema_hint": "SELECT aa.inspection_id, aq.label, aa.answer_text, ir.submitted_on FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id JOIN inspection_report ir ON aa.inspection_report_id = ir.id WHERE aa.inspection_report_id IN (SELECT aa2.inspection_report_id FROM ai_answers aa2 JOIN ai_questions aq2 ON aa2.element_id = aq2.element_id WHERE aq2.label ILIKE '%risk level%' AND aa2.answer_text ILIKE '%High%') AND aq.label ILIKE '%observation%' AND ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 100;"}}}}

─── Form answers: by inspection ───
User: "all answers in inspection INS-2024-001"
Turn 1: {{"thought": "Filter by inspection_id.", "tool": "get_answers", "args": {{"inspection_id": "INS-2024-001", "limit": 100}}}}

─── Form answers: most recently filled form ───
User: "show all questions and answers in the most recently filled form"
Turn 1: {{"thought": "Need most recent inspection then filter to Inspection Form module only — without module_name filter this returns 10000+ rows across all 16 modules.", "tool": "generate_sql", "args": {{"question": "questions and answers from most recent inspection form", "schema_hint": "SELECT aq.label AS question, aa.answer_text AS answer FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id WHERE aa.inspection_report_id = (SELECT id FROM inspection_report WHERE status != 'DRAFT' ORDER BY submitted_on DESC LIMIT 1) AND aa.module_name = 'Inspection Form' ORDER BY aq.label LIMIT 100;"}}}}

─── Scores for a named inspector's last inspection ───
User: "show questions and scores for the last inspection by George"
ROUTING RULE: ALWAYS use generate_sql here. NEVER use get_answers — it cannot filter by inspector name.
Turn 1: {{"thought": "Need ai_answers.score filtered to George's last inspection. Must use generate_sql with subquery to find the inspection_report_id first.", "tool": "generate_sql", "args": {{"question": "questions and scores for last inspection by George", "schema_hint": "SELECT aq.label AS question, aa.answer_text AS answer, aa.score FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id WHERE aa.inspection_report_id = (SELECT ir.id FROM inspection_report ir JOIN users u ON ir.inspector_user_id = u.id WHERE (u.first_name ILIKE '%George%' OR u.last_name ILIKE '%George%') AND ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 1) AND aa.module_name = 'Inspection Form' AND aa.score IS NOT NULL ORDER BY aa.score DESC LIMIT 100;"}}}}

─── Cross-domain join ───
User: "risk level answers for inspections at Al Ghadeer facility"
Turn 1: {{"thought": "Need ai_answers + inspection_report + facility — use generate_sql.", "tool": "generate_sql", "args": {{"question": "risk level answers at Al Ghadeer", "schema_hint": "FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id JOIN inspection_report ir ON aa.inspection_report_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE aq.label ILIKE '%risk level%' AND fac.name ILIKE '%Al Ghadeer%'"}}}}

─── Form discovery ───
User: "which forms ask about emergency contact?"
Turn 1: {{"thought": "Semantic search over question labels.", "tool": "semantic_search", "args": {{"query": "emergency contact"}}}}

─── What questions are in a form ───
User: "what questions are in the vehicle inspection form?"
Turn 1: {{"thought": "Search ai_questions by form name.", "tool": "search_questions", "args": {{"query": "", "form_name": "vehicle inspection", "limit": 50}}}}

─── Most recent inspection ───
User: "which facility underwent the most recent inspection"
ROUTING RULE: Query inspection_report directly. Do NOT route through ai_answers or look for form labels.
Turn 1: {{"thought": "Most recent inspection is in inspection_report ordered by submitted_on. Must SELECT ir.inspection_id for follow-up context.", "tool": "generate_sql", "args": {{"question": "most recent inspection facility", "schema_hint": "SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on, ir.status FROM inspection_report ir JOIN facility fac ON ir.facility_id = fac.id WHERE ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 1;"}}}}

─── Multi-turn: fetch answers for a specific inspection ───
[History shows: inspection_id = 'INS-EXAMPLE-001', facility = 'Facility X', inspector = 'Inspector Y']
User: "fetch all questions and answers in the form she filled for that site"

Turn 1: {{"thought": "The context shows inspection_id='INS-EXAMPLE-001'. The user means THAT specific inspection. I must filter by inspection_id, not by facility or inspector name — that would return thousands of answers across all inspections.", "tool": "get_answers", "args": {{"inspection_id": "INS-EXAMPLE-001", "limit": 100}}}}

─── Multi-turn: follow-up about a single-row result ───
[History shows single-row result: facility_name='Facility X']
User: "who inspected it"

Turn 1: {{"thought": "Context is about Facility X facility. Query the most recent inspection there to find inspector.", "tool": "generate_sql", "args": {{"question": "inspector of most recent inspection at Facility X", "schema_hint": "SELECT u.first_name || ' ' || u.last_name AS inspector_name FROM inspection_report ir JOIN users u ON ir.inspector_user_id = u.id JOIN facility fac ON ir.facility_id = fac.id WHERE fac.name ILIKE '%Facility X%' AND ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 1;"}}}}

─── Which observation appears most as high risk ───
User: "which observation appears most frequently as high risk"
ROUTING RULE: This asks for OBSERVATION TEXT where risk level = High (finding layer).
Do NOT call get_answer_stats(label="Risk Level") — that returns the risk level distribution.
Do NOT call get_answer_stats(label="Observation") — that ignores the high-risk filter.
MUST use generate_sql with a subquery joining risk level answers to observation answers.
Turn 1: {{"thought": "Need observations filtered to inspections where Risk Level = High. Must use SQL subquery — get_answer_stats cannot cross-filter between two questions.", "tool": "generate_sql", "args": {{"question": "observation text most frequent in high risk inspections", "schema_hint": "SELECT obs.answer_text AS observation, COUNT(*) AS frequency FROM ai_answers risk_aa JOIN ai_questions risk_aq ON risk_aa.element_id = risk_aq.element_id JOIN inspection_report ir ON risk_aa.inspection_report_id = ir.id JOIN ai_answers obs ON obs.inspection_report_id = ir.id JOIN ai_questions obs_q ON obs.element_id = obs_q.element_id WHERE risk_aq.label ILIKE '%risk level%' AND risk_aa.answer_text ILIKE '%High%' AND obs_q.label ILIKE '%observation%' AND obs_q.label NOT ILIKE '%type%' AND obs.answer_text IS NOT NULL AND ir.status != 'DRAFT' GROUP BY obs.answer_text ORDER BY frequency DESC LIMIT 1;"}}}}

─── Count questions in a form ───
User: "how many questions are in the inspection form?"
ROUTING RULE: Do NOT use search_questions or list_forms — those return rows not a count. Use generate_sql.
Turn 1: {{"thought": "Count distinct questions from ai_questions filtered by module_name.", "tool": "generate_sql", "args": {{"question": "count questions in inspection form", "schema_hint": "SELECT COUNT(DISTINCT element_id) AS question_count FROM ai_questions WHERE module_name ILIKE '%Inspection Form%';"}}}}

─── List forms ───
User: "how many forms are there?"
Turn 1: {{"thought": "list_forms gives distinct form names.", "tool": "list_forms", "args": {{}}}}

─── Average form answer score ───
User: "average score in the fire safety form"
Turn 1: {{"thought": "Aggregate over ai_answer.answer_text for fire safety.", "tool": "get_answer_stats", "args": {{"form_name": "fire safety", "label": "score"}}}}

─── Unknown / nonsense query ───
User: "xyz abc test 123"
Turn 1: {{"thought": "The query doesn't relate to any inspection data I can find. I should tell the user I can't help with this.", "tool": "final_answer", "args": {{"answer": "I couldn't find any inspection data related to that query. Could you rephrase or ask about inspections, corrective actions, scores, or form answers?"}}}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Remember: respond with ONLY valid JSON. No other text.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ============================================================================
# SQL GENERATION PROMPT
# ============================================================================
# {seed_examples} is populated at call time by SeedExampleIndex.retrieve()
# with the 3-4 most relevant patterns for the incoming question.

SQL_GENERATION_PROMPT = """\
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- Generate a PostgreSQL SELECT query ONLY. No INSERT/UPDATE/DELETE/DROP.
- Use ILIKE for partial text matching on labels and names. NEVER use LIKE.
- Use ILIKE for answer_text matching: aa.answer_text ILIKE '%High%' (answer_text stores plain strings but ILIKE is safer for partial matches).
- Include LIMIT 100 unless aggregating.
- NEVER use SELECT *. Specify only the columns the user needs.
- NEVER use SQL reserved words as aliases (is, as, in, on, by, do, if).
  Use: ir (inspection_report), ica (inspection_corrective_action),
       ic (inspection_cycle), sched (inspection_schedule),
       aa (ai_answers), aq (ai_questions), fac (facility), u (users), it (inspection_type).
- ENUM values are ALWAYS uppercase: status = 'OPEN', status = 'CLOSED', status = 'OVERDUE',
  status = 'CLOSE_WITH_DEFERRED',
  responsible = 'CLIENT', responsible = 'INTERNAL_OPERATIONS', responsible = 'SUB_CONTRACTOR'.
  NEVER use lowercase for enum comparisons.
- When grouping by facility/type/client/inspector, ALWAYS JOIN the lookup table and
  GROUP BY the name column (e.g. GROUP BY fac.name). NEVER GROUP BY a UUID FK column.
- For month display, ALWAYS use TO_CHAR(date_trunc('month', col), 'Month YYYY') AS month.
  NEVER use EXTRACT(MONTH FROM col) — that returns a number, not a name.

### AI FLAT TABLES — use these for all form question/answer data:

  ai_questions EXACT columns (use ONLY these — nothing else exists):
    element_id   uuid
    label        text   ← this is the question text / display name
    entity_type  varchar
    module_id    uuid
    module_name  text
    created_at   timestamp
  DO NOT USE: question_label, page_name, required, question_type, form_name
              (none of these columns exist on ai_questions)

  ai_answers EXACT columns (use ONLY these — nothing else exists):
    id                   uuid
    element_id           uuid  → JOIN to ai_questions.element_id
    inspection_report_id uuid  → JOIN to inspection_report.id
    inspection_id        varchar
    answer_text          text
    answer_numeric       numeric
    score                numeric
    score_type           varchar
    max_score            numeric
    module_name          text
    module_id            uuid
    submitted_on         timestamptz
    status               varchar
  DO NOT USE: answer_value, form_name, question_id, source_table
              (none of these columns exist on ai_answers)

  Standard JOIN:
    FROM ai_answers aa
  LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id

  Join to inspection metadata:
    JOIN inspection_report ir ON aa.inspection_report_id = ir.id

  CRITICAL — answer_text encoding:
    Dropdown/checkbox answers are stored as plain strings — NOT JSON arrays.
    e.g. Risk Level answer is stored as: High        (not ["High|uuid"])
         Observation Type answer:        Deviation   (not ["Deviation|uuid"])
    Use ILIKE for answer_text matching: aa.answer_text ILIKE '%High%'
    For distribution/grouping queries use: GROUP BY aa.answer_text (returns all values)
    Known answer values: 'High', 'Medium', 'Low', 'No Active Risk', 'Deviation', 'Good Practice'

  CRITICAL — ai_answers spans ALL 16 modules:
    When user says 'the form', 'the inspection form', 'what they filled' without
    specifying a module — filter to the main form only:
      AND aa.module_name = 'Inspection Form'
    Without this, queries return answers from all 16 modules (~10,000 rows instead of ~50).
    Only omit this filter when user explicitly asks about another module.

  EXCEPTION — specific inspection_id queries:
    When the WHERE clause already filters to a SPECIFIC inspection_report_id or
    inspection_id, do NOT add the module_name filter. The inspection_id already
    scopes to one inspection — adding module_name risks returning zero rows if that
    inspection's answers were stored under a module name variant.
    WRONG:  WHERE aa.inspection_report_id = '...' AND aa.module_name = 'Inspection Form'
    RIGHT:  WHERE aa.inspection_report_id = '...'  (no module_name filter)

  CRITICAL — most recent inspection subquery:
    CORRECT:   WHERE aa.inspection_report_id = (
                   SELECT id FROM inspection_report
                   WHERE status != 'DRAFT'
                   ORDER BY submitted_on DESC LIMIT 1)
    WRONG:     SELECT inspection_report_id FROM inspection_report
               (inspection_report_id is not a column on inspection_report — id is)

### INSPECTION CORRECTIVE ACTION — LOOKUP FK COLUMNS:
  risk_level_id and impact_id are UUID FKs to lookup tables — NOT plain string columns.
  NEVER: WHERE ica.risk_level = 'High'           (column does not exist)
  NEVER: WHERE ica.risk_level_id = 'High'        (UUID FK ≠ string)
  ALWAYS:
    JOIN risk_level rl ON ica.risk_level_id = rl.id
    WHERE rl.name ILIKE '%High%'
  ALWAYS:
    JOIN impact im ON ica.impact_id = im.id
    WHERE im.name ILIKE '%Non Confirmity%'
  CRITICAL TYPO: impact table stores 'Non Confirmity' (NOT 'Non Conformity').
    Using WHERE im.name ILIKE '%conformity%' returns zero rows.
    Always use: WHERE im.name ILIKE '%Non Confirmity%'

### DEFERRED ACTION CHAIN PATTERN:
  CLOSE_WITH_DEFERRED means an action was deferred to the next inspection at same facility.
  Pattern for deferred/recurring queries:
    WHERE ica.status = 'CLOSE_WITH_DEFERRED'
  Pattern for counting recurrences at a facility:
    SELECT ica.cause, COUNT(*) AS recurrence_count
    FROM inspection_corrective_action ica
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    JOIN facility fac ON ir.facility_id = fac.id
    WHERE ica.status = 'CLOSE_WITH_DEFERRED'
      AND fac.name ILIKE '%FACILITY_NAME%'
    GROUP BY ica.cause ORDER BY recurrence_count DESC;

### Additional context
{schema_hint}

### RELEVANT EXAMPLES FOR THIS QUERY:
{seed_examples}

### Answer
Given the database schema, here is the SQL query that answers \
[QUESTION]{question}[/QUESTION]
[SQL]
"""




# ============================================================================
# SQL_GENERATION_PROMPT_SQLCODER
# ============================================================================
# Native format for SQLCoder 7B (defog/sqlcoder-7b-2).
# SQLCoder was fine-tuned on: ### Task / ### Database Schema / ### Answer [SQL]
# It performs best when schema is given as concise DDL-like column listings,
# critical rules are embedded as inline SQL comments, and examples are shown
# directly in the relevant table sections rather than as a separate block.
#
# Key differences from SQL_GENERATION_PROMPT:
#   - No long prose instruction sections — rules are short and inline
#   - Schema is the primary teaching device, not paragraphs of guidance
#   - CRITICAL rules are embedded as /* comments */ inside the schema block
#     so SQLCoder sees them at the moment it reads each table definition
#   - {seed_examples} still injected — SQLCoder handles SQL examples well
#   - Stop token is still ";" so truncate at first complete statement

SQL_GENERATION_PROMPT_SQLCODER = """\
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema

-- ─── INSPECTION REPORT (hub table) ──────────────────────────────────────────
CREATE TABLE inspection_report (
    id                    uuid PRIMARY KEY,          -- NEVER select for display
    inspection_id         varchar,                   -- human-readable: '2026/04/ST001/INS001' — always SELECT this
    status                varchar,                   -- DRAFT|SUBMITTED|UNDER_REVIEW|CLOSED|RETURN_FOR_MODIFICATION
    inspection_score      numeric,                   -- NULL when status='DRAFT' — always filter WHERE status != 'DRAFT'
    gp_score              numeric,
    total_inspection_hours numeric,
    submitted_on          timestamptz,
    facility_id           uuid REFERENCES facility(id),
    inspector_user_id     uuid REFERENCES users(id),
    inspectee_user_id     uuid REFERENCES users(id),
    inspection_type_id    uuid REFERENCES inspection_type(id),
    inspection_sub_type_id uuid REFERENCES inspection_sub_type(id),
    client_id             uuid REFERENCES client(id),
    project_id            uuid REFERENCES project(id),
    cycle_id              uuid REFERENCES inspection_cycle(id),
    schedule_id           uuid REFERENCES inspection_schedule(id)
    /* ALIAS: ir */
    /* JOIN inspector: JOIN users u ON ir.inspector_user_id = u.id → u.first_name || ' ' || u.last_name */
    /* JOIN facility:  JOIN facility fac ON ir.facility_id = fac.id → fac.name */
    /* JOIN type:      JOIN inspection_type it ON ir.inspection_type_id = it.id → it.name */
);

-- ─── CORRECTIVE ACTION ───────────────────────────────────────────────────────
CREATE TABLE inspection_corrective_action (
    id                         uuid PRIMARY KEY,
    active                     boolean,              -- soft delete, always true for live rows
    corrective_action_id       varchar,              -- human-readable ID — always SELECT this
    status                     varchar,              -- OPEN|OVERDUE|CLOSED|CLOSE_WITH_DEFERRED
    implementation_status      varchar,              -- NULL|PENDING_FOR_REVIEW|APPROVE|RETURN_FOR_MODIFICATION
    responsible                varchar,              -- CLIENT|INTERNAL_OPERATIONS|SUB_CONTRACTOR (plain ENUM, NOT a FK)
    cause                      text,
    correction                 text,
    corrective_action          text,
    capex                      numeric,
    opex                       numeric,
    expenditure                varchar,              -- NO_EXPENDITURE|CAPEX|OPEX|CAPEX_AND_OPEX
    target_close_out_date      date,
    close_on                   timestamptz,          -- NULL on most rows — NEVER do arithmetic on this column
    deferred_on                timestamptz,
    tentative_date             date,
    inspection_id              uuid REFERENCES inspection_report(id),   -- UUID FK to ir.id — NOT ir.inspection_id
    risk_level_id              uuid REFERENCES risk_level(id),          -- UUID FK — NEVER compare to string
    impact_id                  uuid REFERENCES impact(id),              -- UUID FK — NEVER compare to string
    submission_id              uuid,
    completion_submission_id   uuid,
    client_details_submission_id uuid,
    pr_submission_id           uuid,
    deferred_submission_id     uuid
    /* ALIAS: ica */
    /* CRITICAL JOIN to ir: ON ica.inspection_id = ir.id  — NOT ON ir.inspection_id = ica.inspection_id */
    /* risk level: JOIN risk_level rl ON ica.risk_level_id = rl.id WHERE rl.name ILIKE '%High%' */
    /* impact:     JOIN impact im ON ica.impact_id = im.id WHERE im.name ILIKE '%Non Confirmity%' */
    /* TYPO: impact table stores 'Non Confirmity' — NOT 'Non Conformity'. ILIKE '%conformity%' = zero rows */
    /* responsible is a plain string ENUM — NEVER JOIN it to users */
);

-- ─── FORM ANSWERS (flat view over form builder) ──────────────────────────────
CREATE TABLE ai_answers (
    id                   uuid,
    element_id           uuid,                       -- JOIN to ai_questions.element_id
    inspection_report_id uuid REFERENCES inspection_report(id),
    inspection_id        varchar,                    -- human-readable, same as ir.inspection_id
    answer_text          text,                       -- plain string, NOT JSON: 'High', 'Deviation', 'Good Practice'
    answer_numeric       numeric,                    -- NULL for categorical/dropdown questions
    score                numeric,                    -- per-question score (different from ir.inspection_score)
    max_score            numeric,
    module_name          text,                       -- workflow stage name
    submitted_on         timestamptz
    /* ALIAS: aa */
    /* BROAD queries: AND aa.module_name = 'Inspection Form' (prevents 10,000+ rows from 15 modules) */
    /* SPECIFIC inspection_id queries: NO module_name filter needed */
    /* MOST RECENT with answers: WHERE aa.inspection_report_id = (SELECT inspection_report_id FROM ai_answers ORDER BY submitted_on DESC LIMIT 1) */
    /* answer_text is ILIKE-matched: aa.answer_text ILIKE '%High%' */
);

CREATE TABLE ai_questions (
    element_id   uuid,                               -- JOIN to ai_answers.element_id
    label        text,                               -- question text — search: aq.label ILIKE '%risk level%'
    module_name  text,
    entity_type  varchar
    /* ALIAS: aq */
    /* DO NOT USE: question_label, form_name, question_type, page_name — these columns do not exist */
    /* DO NOT USE: answer_value, source_table on ai_answers — these columns do not exist */
);

-- ─── CYCLE / SCHEDULE / PORTFOLIO ────────────────────────────────────────────
CREATE TABLE inspection_cycle (
    id          uuid PRIMARY KEY,
    status      varchar,                             -- UPCOMING|ONGOING|COMPLETED
    start_date  date,
    end_date    date,
    due_date    date
    /* ALIAS: ic */
);

CREATE TABLE inspection_schedule (
    id                      uuid PRIMARY KEY,
    status                  varchar,                 -- PENDING|ONGOING|OVERDUE|COMPLETED|CANCELLED
    schedule_date           date,
    due_date                date,
    facility_id             uuid REFERENCES facility(id),
    inspector_id            uuid REFERENCES users(id),
    inspection_cycle_id     uuid REFERENCES inspection_cycle(id),
    portfolio_details_id    uuid,
    inspection_type_id      uuid REFERENCES inspection_type(id),
    is_admin_triggered      boolean,
    is_contractual_inspection boolean,
    submission_id           uuid
    /* ALIAS: sched */
);

CREATE TABLE inspector_portfolio (
    id             uuid PRIMARY KEY,
    status         varchar,                          -- DRAFT|PENDING|APPROVED|MODIFICATION
    user_id        uuid REFERENCES users(id),
    cycle_id       uuid REFERENCES inspection_cycle(id),
    submitted_date date
    /* Admin queries: WHERE status != 'DRAFT' */
);

CREATE TABLE inspector_portfolio_details (
    id                      uuid PRIMARY KEY,
    facility_id             uuid REFERENCES facility(id),
    frequency_definition_id uuid REFERENCES frequency_definition(id),
    inspection_type_id      uuid REFERENCES inspection_type(id),
    inspection_subtype_id   uuid REFERENCES inspection_sub_type(id),
    portfolio_id            uuid REFERENCES inspector_portfolio(id),
    is_contractual_inspection boolean
);

CREATE TABLE frequency_definition (
    id               uuid PRIMARY KEY,
    name             varchar,                        -- 'Monthly', 'Quarterly', 'Weekly', etc.
    repeat_count     integer,                        -- how many times per interval
    repeat_interval  integer,
    repeat_unit      varchar                         -- DAY|WEEK|MONTH
);

-- ─── LOOKUP TABLES ───────────────────────────────────────────────────────────
CREATE TABLE facility        (id uuid PRIMARY KEY, name varchar);
CREATE TABLE users           (id uuid PRIMARY KEY, first_name varchar, last_name varchar);
CREATE TABLE inspection_type (id uuid PRIMARY KEY, name varchar);
CREATE TABLE inspection_sub_type (id uuid PRIMARY KEY, name varchar);
CREATE TABLE client          (id uuid PRIMARY KEY, name varchar);
CREATE TABLE project         (id uuid PRIMARY KEY, name varchar);
CREATE TABLE risk_level      (id uuid PRIMARY KEY, name varchar);  -- values: High, Medium, Low, No Active Risk
CREATE TABLE impact          (id uuid PRIMARY KEY, name varchar);  -- TYPO: 'Non Confirmity' not 'Non Conformity'

-- ─── CRITICAL RULES (apply to every query) ───────────────────────────────────
-- 1. ENUM values are ALWAYS uppercase: 'OPEN', 'CLOSED', 'OVERDUE', 'CLOSE_WITH_DEFERRED',
--    'CLIENT', 'INTERNAL_OPERATIONS', 'SUB_CONTRACTOR', 'DRAFT', 'SUBMITTED', etc.
-- 2. NEVER use ir.id in SELECT — use ir.inspection_id (varchar) for display.
-- 3. NEVER GROUP BY a UUID FK column — always JOIN the lookup table and GROUP BY name.
-- 4. NEVER do arithmetic on ica.close_on — use status-based counting instead.
-- 5. NEVER use hardcoded years — use EXTRACT(YEAR FROM CURRENT_DATE).
-- 6. For month display: TO_CHAR(date_trunc('month', col), 'Month YYYY') AS month.
-- 7. NEVER use reserved words as aliases: is, as, in, on, by. Use: ir, ica, aa, aq, fac, u, it, sched.
-- 8. Include LIMIT 100 unless aggregating.
-- 9. ica.cause is a plain text column on inspection_corrective_action — NOT a form answer question.
--    Use: SELECT cause, COUNT(*) FROM inspection_corrective_action GROUP BY cause ORDER BY 2 DESC;

### Additional context
{schema_hint}

### Relevant examples
{seed_examples}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""


FORCE_SUMMARY_PROMPT = """\
You have reached the maximum number of reasoning steps.
Based on everything you have learned so far, provide the best possible answer.

Respond with ONLY:
{{"thought": "Summarising findings so far.", "tool": "final_answer", "args": {{"answer": "<your best answer>"}}}}
"""


# ============================================================================
# SEED EXAMPLES — used by SeedExampleIndex for dynamic retrieval (Problem 4)
# ============================================================================
# Each tuple is (nl_question_description, sql_pattern).
# The NL descriptions are embedded and matched at query time.
# Add new patterns here as the system encounters them — the list grows organically.

SEED_EXAMPLES: list[tuple[str, str]] = [

    # ── Inspection report queries ──────────────────────────────────────────

    (
        "average inspection score no drafts",
        """SELECT AVG(inspection_score) AS avg_score
FROM inspection_report WHERE status != 'DRAFT';""",
    ),
    (
        "inspector with most inspections this year count",
        """SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       COUNT(*) AS num_inspections
FROM inspection_report ir
JOIN users u ON ir.inspector_user_id = u.id
WHERE ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY u.first_name, u.last_name
ORDER BY num_inspections DESC LIMIT 10;""",
    ),
    (
        "inspection scores grouped by facility average",
        """SELECT fac.name AS facility_name,
       AVG(ir.inspection_score) AS avg_score,
       COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.status != 'DRAFT'
GROUP BY fac.name ORDER BY avg_score DESC LIMIT 100;""",
    ),
    (
        "inspections per type count",
        """SELECT it.name AS type_name, COUNT(*) AS count
FROM inspection_report ir
JOIN inspection_type it ON ir.inspection_type_id = it.id
GROUP BY it.name ORDER BY count DESC;""",
    ),
    (
        "most recent inspection facility submitted",
        """SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on, ir.status
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 1;""",
    ),
    (
        "last N recent inspections list",
        """SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on,
       ir.status, ir.inspection_score
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 5;""",
    ),
    (
        "inspector with lowest average inspection score",
        """SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       AVG(ir.inspection_score) AS avg_score
FROM inspection_report ir
JOIN users u ON ir.inspector_user_id = u.id
WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
GROUP BY u.first_name, u.last_name
ORDER BY avg_score ASC LIMIT 1;""",
    ),
    (
        "this month vs last month inspection comparison",
        """SELECT
  COUNT(*) FILTER (WHERE date_trunc('month', submitted_on) = date_trunc('month', NOW())) AS this_month,
  COUNT(*) FILTER (WHERE date_trunc('month', submitted_on) = date_trunc('month', NOW() - INTERVAL '1 month')) AS last_month
FROM inspection_report WHERE status != 'DRAFT';""",
    ),
    (
        "inspections done on weekends Saturday Sunday",
        """SELECT COUNT(*) AS weekend_count
FROM inspection_report
WHERE EXTRACT(ISODOW FROM submitted_on) IN (6, 7)
  AND status != 'DRAFT';""",
    ),
    (
        "inspections count by month this year",
        """SELECT TO_CHAR(date_trunc('month', ir.submitted_on), 'Month YYYY') AS month,
       COUNT(*) AS count
FROM inspection_report ir
WHERE ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY date_trunc('month', ir.submitted_on)
ORDER BY date_trunc('month', ir.submitted_on);""",
    ),
    (
        "score drop vs previous inspection same facility LAG window",
        """WITH scored AS (
  SELECT ir.inspection_id,
         fac.name AS facility_name,
         ir.submitted_on,
         ir.inspection_score,
         LAG(ir.inspection_score) OVER (
           PARTITION BY ir.facility_id ORDER BY ir.submitted_on
         ) AS previous_score
  FROM inspection_report ir
  JOIN facility fac ON ir.facility_id = fac.id
  WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
)
SELECT inspection_id, facility_name, submitted_on,
       inspection_score AS current_score,
       previous_score,
       previous_score - inspection_score AS score_drop
FROM scored
WHERE previous_score IS NOT NULL
  AND inspection_score < previous_score
ORDER BY score_drop DESC LIMIT 50;""",
    ),
    (
        "facility most improved score vs last quarter",
        """WITH quarterly AS (
  SELECT fac.name AS facility_name,
         AVG(CASE WHEN ir.submitted_on >= date_trunc('quarter', NOW())
                  THEN ir.inspection_score END) AS this_q,
         AVG(CASE WHEN ir.submitted_on >= date_trunc('quarter', NOW() - INTERVAL '3 months')
                   AND ir.submitted_on  <  date_trunc('quarter', NOW())
                  THEN ir.inspection_score END) AS last_q
  FROM inspection_report ir
  JOIN facility fac ON ir.facility_id = fac.id
  WHERE ir.status != 'DRAFT' AND ir.inspection_score IS NOT NULL
  GROUP BY fac.name
)
SELECT facility_name,
       ROUND(this_q::numeric, 1) AS this_quarter_avg,
       ROUND(last_q::numeric, 1) AS last_quarter_avg,
       ROUND((this_q - last_q)::numeric, 1) AS improvement
FROM quarterly
WHERE this_q IS NOT NULL AND last_q IS NOT NULL
ORDER BY improvement DESC LIMIT 1;""",
    ),
    (
        "projects with no inspections this year",
        """SELECT proj.name AS project_name
FROM project proj
WHERE NOT EXISTS (
    SELECT 1 FROM inspection_report ir
    WHERE ir.project_id = proj.id
      AND ir.status != 'DRAFT'
      AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
);""",
    ),
    (
        "percentage of inspections with at least one corrective action",
        """SELECT
  ROUND(
    100.0 * COUNT(DISTINCT CASE WHEN ica.id IS NOT NULL THEN ir.id END)
    / NULLIF(COUNT(DISTINCT ir.id), 0),
    1
  ) AS pct_with_corrective_action
FROM inspection_report ir
LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id
WHERE ir.status != 'DRAFT';""",
    ),
    (
        "top N facilities by average inspection score",
        """SELECT fac.name AS facility_name,
       AVG(ir.inspection_score) AS avg_inspection_score
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE ir.status != 'DRAFT'
  AND fac.name IS NOT NULL
  AND ir.inspection_score IS NOT NULL
GROUP BY fac.name
ORDER BY avg_inspection_score DESC LIMIT 5;""",
    ),

    # ── Corrective action queries ──────────────────────────────────────────

    (
        "open corrective actions with cause and responsible",
        """SELECT corrective_action_id, cause, corrective_action, responsible, status
FROM inspection_corrective_action WHERE status = 'OPEN' LIMIT 100;""",
    ),
    (
        "corrective actions with facility and inspector name",
        """SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
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
        "capex opex by facility corrective action cost",
        """SELECT fac.name AS facility_name,
       SUM(ica.capex) AS total_capex,
       SUM(ica.opex) AS total_opex
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
GROUP BY fac.name ORDER BY total_capex DESC NULLS LAST LIMIT 50;""",
    ),
    (
        "client with most overdue corrective actions",
        """SELECT cl.name AS client_name, COUNT(*) AS overdue_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN client cl ON ir.client_id = cl.id
WHERE ica.status = 'OVERDUE'
GROUP BY cl.name ORDER BY overdue_count DESC LIMIT 10;""",
    ),
    (
        "corrective actions by inspection type breakdown",
        """SELECT it.name AS inspection_type, COUNT(*) AS action_count
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
GROUP BY it.name ORDER BY action_count DESC LIMIT 20;""",
    ),
    (
        "corrective actions where responsible is client",
        """SELECT corrective_action_id, cause, corrective_action, status
FROM inspection_corrective_action
WHERE responsible = 'CLIENT'
LIMIT 100;
-- responsible values: 'CLIENT', 'INTERNAL_OPERATIONS', 'SUB_CONTRACTOR' (always uppercase)""",
    ),
    (
        "corrective action status breakdown by responsible party open closed overdue",
        """SELECT responsible AS responsible_party,
       COUNT(*) FILTER (WHERE status = 'OPEN')    AS open_count,
       COUNT(*) FILTER (WHERE status = 'CLOSED')  AS closed_count,
       COUNT(*) FILTER (WHERE status = 'OVERDUE') AS overdue_count
FROM inspection_corrective_action
GROUP BY responsible
ORDER BY responsible;""",
    ),
    (
        "corrective actions raised vs closed this quarter",
        """SELECT
  COUNT(*) AS total_raised,
  COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS total_closed,
  COUNT(*) FILTER (WHERE status IN ('OPEN','OVERDUE')) AS still_open
FROM inspection_corrective_action
WHERE created_on >= date_trunc('quarter', NOW());""",
    ),
    (
        "closure speed corrective action quarter closed percentage trend",
        """SELECT
  TO_CHAR(date_trunc('quarter', ica.created_on), '"Q"Q YYYY') AS quarter,
  COUNT(*) AS total_raised,
  COUNT(*) FILTER (WHERE ica.status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS total_closed,
  ROUND(
    100.0 * COUNT(*) FILTER (WHERE ica.status IN ('CLOSED','CLOSE_WITH_DEFERRED'))
    / NULLIF(COUNT(*), 0), 1
  ) AS pct_closed
FROM inspection_corrective_action ica
WHERE ica.created_on >= date_trunc('quarter', NOW() - INTERVAL '3 months')
GROUP BY date_trunc('quarter', ica.created_on)
ORDER BY date_trunc('quarter', ica.created_on);""",
    ),
    (
        "high risk corrective actions show list with facility join risk level lookup",
        """SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.status, rl.name AS risk_level_name,
       fac.name AS facility_name
FROM inspection_corrective_action ica
JOIN risk_level rl ON ica.risk_level_id = rl.id
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE rl.name ILIKE '%High%'
ORDER BY ica.created_on DESC LIMIT 100;
-- risk_level.name values: 'High', 'Medium', 'Low', 'No Active Risk'
-- ALWAYS use ILIKE '%High%' not = 'High' (safer for whitespace/case variants)""",
    ),
    (
        "corrective actions by impact category join impact lookup table",
        """SELECT im.name AS impact_name, COUNT(*) AS action_count
FROM inspection_corrective_action ica
JOIN impact im ON ica.impact_id = im.id
GROUP BY im.name ORDER BY action_count DESC LIMIT 20;
-- NOTE: impact table stores 'Non Confirmity' (not 'Conformity')""",
    ),

    # ── Deferred action chain (Problem 5) ─────────────────────────────────

    (
        "deferred corrective actions close with deferred status",
        """SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.status, fac.name AS facility_name, ica.created_on
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.status = 'CLOSE_WITH_DEFERRED'
ORDER BY ica.created_on DESC LIMIT 100;""",
    ),
    (
        "recurring issues repeated findings deferred across inspections facility",
        """SELECT ica.cause, COUNT(*) AS recurrence_count,
       fac.name AS facility_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.status = 'CLOSE_WITH_DEFERRED'
GROUP BY ica.cause, fac.name
ORDER BY recurrence_count DESC LIMIT 20;""",
    ),
    (
        "deferred action history for specific facility how many times recurred",
        """SELECT ica.cause, ica.corrective_action_id, ica.created_on,
       ir.inspection_id, fac.name AS facility_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.status = 'CLOSE_WITH_DEFERRED'
  AND fac.name ILIKE '%FACILITY_NAME%'
ORDER BY ica.cause, ica.created_on ASC LIMIT 100;""",
    ),
    (
        "issues that have been deferred more than once cross inspection cycles",
        """SELECT ica.cause,
       COUNT(*) AS deferred_count,
       MIN(ica.created_on) AS first_occurrence,
       MAX(ica.created_on) AS last_occurrence,
       fac.name AS facility_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE ica.status = 'CLOSE_WITH_DEFERRED'
GROUP BY ica.cause, fac.name
HAVING COUNT(*) > 1
ORDER BY deferred_count DESC LIMIT 20;""",
    ),
    (
        "deferral justification reason why corrective action was deferred",
        """SELECT aa.answer_text AS deferral_justification,
       aa.inspection_id,
       fac.name AS facility_name,
       aa.submitted_on
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE aa.module_name = 'Inspection Mitigative Action Close with Deferred'
  AND aq.label = 'Deferral Justification'
  AND aa.answer_text IS NOT NULL
ORDER BY aa.submitted_on DESC LIMIT 50;""",
    ),
    (
        "pending with whom deferred action waiting for who holds deferred action",
        """SELECT aa.answer_text AS pending_with,
       aa.inspection_id,
       fac.name AS facility_name,
       aa.submitted_on
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE aa.module_name = 'Inspection Mitigative Action Close with Deferred'
  AND aq.label = 'Pending with whom'
  AND aa.answer_text IS NOT NULL
ORDER BY aa.submitted_on DESC LIMIT 50;""",
    ),
    (
        "inspection report rejection reasons why inspection was rejected returned for modification",
        """SELECT aa.answer_text AS rejection_reason,
       aa.inspection_id,
       fac.name AS facility_name,
       aa.submitted_on
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE aa.module_name = 'Inspection Report Rejection'
  AND aa.answer_text IS NOT NULL
ORDER BY aa.submitted_on DESC LIMIT 50;""",
    ),

    # ── Form answer queries ────────────────────────────────────────────────

    (
        "all answers for a specific question risk level observation",
        """SELECT aa.inspection_id, aa.answer_text, aa.submitted_on
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aq.label ILIKE '%risk level%'
ORDER BY aa.submitted_on DESC LIMIT 100;""",
    ),
    (
        "observation text for specific facility cross domain join",
        """SELECT aa.answer_text AS observation,
       ir.inspection_id,
       ir.submitted_on,
       u.first_name || ' ' || u.last_name AS inspector_name
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
JOIN users u ON ir.inspector_user_id = u.id
WHERE aq.label ILIKE '%observation%'
  AND aq.label NOT ILIKE '%type%'
  AND aq.label NOT ILIKE '%unique%'
  AND fac.name ILIKE '%Al Ghadeer%'
  AND aa.answer_text IS NOT NULL
  AND aa.answer_text != ''
  AND ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC
LIMIT 100;""",
    ),
    (
        "high risk observations where risk level answer is high",
        """SELECT DISTINCT aa.inspection_id, obs.answer_text AS observation,
       aa.answer_text AS risk_level, ir.submitted_on
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
JOIN ai_answers obs ON obs.inspection_report_id = ir.id
JOIN ai_questions obs_q ON obs.element_id = obs_q.element_id
WHERE aq.label ILIKE '%risk level%'
  AND aa.answer_text ILIKE '%High%'
  AND obs_q.label ILIKE '%observation%'
  AND obs_q.label NOT ILIKE '%type%'
  AND ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC LIMIT 100;""",
    ),
    (
        "answer distribution frequency for a question categorical breakdown",
        """SELECT aa.answer_text, COUNT(*) AS frequency
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aq.label ILIKE '%condition%'
GROUP BY aa.answer_text ORDER BY frequency DESC LIMIT 20;""",
    ),
    (
        "most common observation types",
        """SELECT aa.answer_text AS observation_type, COUNT(*) AS frequency
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aq.label ILIKE '%observation type%'
  AND aa.answer_text IS NOT NULL AND aa.answer_text != ''
GROUP BY aa.answer_text ORDER BY frequency DESC LIMIT 20;""",
    ),
    (
        "average number of observations per inspection",
        """SELECT AVG(obs_count) AS avg_obs_per_inspection
FROM (
    SELECT aa.inspection_report_id, COUNT(*) AS obs_count
    FROM ai_answers aa
    JOIN ai_questions aq ON aa.element_id = aq.element_id
    JOIN inspection_report ir ON aa.inspection_report_id = ir.id
    WHERE aq.label ILIKE '%observation%'
      AND aq.label NOT ILIKE '%type%'
      AND aq.label NOT ILIKE '%unique%'
      AND ir.status != 'DRAFT'
    GROUP BY aa.inspection_report_id
) subq;""",
    ),
    (
        "repetitive observations in last 6 months recurring",
        """SELECT aa.answer_text AS observation,
       COUNT(DISTINCT aa.inspection_report_id) AS inspection_count
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aq.label ILIKE '%observation%'
  AND aq.label NOT ILIKE '%type%'
  AND aq.label NOT ILIKE '%unique%'
  AND aa.answer_text IS NOT NULL AND aa.answer_text != ''
  AND aa.submitted_on >= NOW() - INTERVAL '6 months'
GROUP BY aa.answer_text
HAVING COUNT(DISTINCT aa.inspection_report_id) > 1
ORDER BY inspection_count DESC LIMIT 20;""",
    ),
    (
        "most frequent observation text in high risk inspections",
        """SELECT obs.answer_text AS observation, COUNT(*) AS frequency
FROM ai_answers risk_aa
JOIN ai_questions risk_aq ON risk_aa.element_id = risk_aq.element_id
JOIN inspection_report ir ON risk_aa.inspection_report_id = ir.id
JOIN ai_answers obs ON obs.inspection_report_id = ir.id
JOIN ai_questions obs_q ON obs.element_id = obs_q.element_id
WHERE risk_aq.label ILIKE '%risk level%'
  AND risk_aa.answer_text ILIKE '%High%'
  AND obs_q.label ILIKE '%observation%'
  AND obs_q.label NOT ILIKE '%type%'
  AND obs.answer_text IS NOT NULL AND obs.answer_text != ''
  AND ir.status != 'DRAFT'
GROUP BY obs.answer_text ORDER BY frequency DESC LIMIT 1;""",
    ),
    (
        "all questions and answers from most recent inspection form",
        """SELECT aa.module_name, aq.label AS question, aa.answer_text AS answer, aa.score
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aa.inspection_report_id = (
    SELECT inspection_report_id
    FROM ai_answers
    WHERE inspection_report_id IS NOT NULL
    ORDER BY submitted_on DESC LIMIT 1
)
ORDER BY aa.module_name, aq.label LIMIT 100;
-- CRITICAL: use ai_answers.submitted_on for the subquery, NOT inspection_report.submitted_on.
-- A freshly-submitted inspection_report may have no ai_answers rows yet.
-- This subquery finds the inspection that actually HAS answers.
-- No module_name filter — a specific inspection_report_id already scopes to one inspection.""",
    ),
    (
        "most common causes of corrective actions frequency count breakdown",
        """SELECT cause, COUNT(*) AS frequency
FROM inspection_corrective_action
WHERE cause IS NOT NULL AND cause != ''
GROUP BY cause
ORDER BY frequency DESC LIMIT 20;
-- NOTE: ica.cause is a plain text column on inspection_corrective_action.
-- Do NOT route through ai_answers for this — there is no 'cause' question label in ai_questions.
-- This is Layer 3 (structured workflow state), not Layer 1 (form answers).""",
    ),
    (
        "questions and scores for last inspection by named inspector George",
        """SELECT aq.label AS question, aa.answer_text AS answer, aa.score
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
WHERE aa.inspection_report_id = (
    SELECT ir.id
    FROM inspection_report ir
    JOIN users u ON ir.inspector_user_id = u.id
    WHERE (u.first_name ILIKE '%George%' OR u.last_name ILIKE '%George%')
      AND ir.status != 'DRAFT'
    ORDER BY ir.submitted_on DESC
    LIMIT 1
)
AND aa.module_name = 'Inspection Form'
AND aa.score IS NOT NULL
ORDER BY aa.score DESC LIMIT 100;""",
    ),
    (
        "inspection with highest total score sum of per question scores",
        """SELECT ir.inspection_id, SUM(aa.score) AS total_score
FROM ai_answers aa
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
WHERE aa.score IS NOT NULL AND ir.status != 'DRAFT'
GROUP BY ir.inspection_id
ORDER BY total_score DESC LIMIT 1;""",
    ),
    (
        "count questions in specific form module",
        """SELECT COUNT(DISTINCT element_id) AS question_count
FROM ai_questions
WHERE module_name ILIKE '%Inspection Form%';""",
    ),
    (
        "month with most high risk observations this year",
        """SELECT TO_CHAR(date_trunc('month', ir.submitted_on), 'Month YYYY') AS month,
       COUNT(*) AS high_risk_count
FROM ai_answers aa
JOIN ai_questions aq ON aa.element_id = aq.element_id
JOIN inspection_report ir ON aa.inspection_report_id = ir.id
WHERE aq.label ILIKE '%risk level%'
  AND aa.answer_text ILIKE '%High%'
  AND ir.status != 'DRAFT'
  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY date_trunc('month', ir.submitted_on)
ORDER BY high_risk_count DESC LIMIT 1;""",
    ),
    (
        "overdue inspections this quarter schedule count",
        """SELECT COUNT(*) AS overdue_count
FROM inspection_schedule
WHERE status = 'OVERDUE'
  AND due_date >= date_trunc('quarter', CURRENT_DATE)
  AND due_date < date_trunc('quarter', CURRENT_DATE) + INTERVAL '3 months';
-- inspection_schedule.status values: PENDING, ONGOING, OVERDUE, COMPLETED, CANCELLED""",
    ),
    (
        "inspections overdue this quarter how many",
        """SELECT COUNT(*) AS overdue_count
FROM inspection_schedule sched
WHERE sched.status = 'OVERDUE'
  AND sched.due_date >= date_trunc('quarter', CURRENT_DATE);""",
    ),
]