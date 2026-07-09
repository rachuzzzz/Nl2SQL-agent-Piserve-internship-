"""
Prompt templates for the agentic NL2SQL system.

STATUS GROUPINGS come from core.semantics_registry — the single source of
truth. Edit there, not here.
"""

import re

# ── Registry-generated status block (single source of truth) ─────────────────
try:
    from core.semantics_registry import (
        llama_status_block as _llama_status_block,
        llama_ca_workflow_block as _llama_ca_workflow_block,
        llama_columns_block as _llama_columns_block,
    )
    _STATUS_BLOCK = _llama_status_block()
    _CA_WORKFLOW_BLOCK = _llama_ca_workflow_block()
    _COLUMNS_BLOCK = _llama_columns_block()
except ImportError:
    # Fallback — keep in sync with core/semantics_registry.py
    _STATUS_BLOCK = """\
IMPORTANT — inspection_report.status values and groupings:
  DRAFT                    → incomplete, never submitted — NEVER count or show
  SUBMITTED                → inspector filed the report (in-flight)
  UNDER_REVIEW             → under FM/admin review (in-flight)
  RETURN_FOR_MODIFICATION  → sent back to inspector (in-flight)
  CLOSED                   → fully reviewed and approved (final)

  STATUS GROUPINGS — always bucket into two categories for reporting:
    conducted = IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')
    completed = 'CLOSED'

  DEFAULT for "total / how many / count inspections":
    Show as TWO-ROW summary using CASE grouping:
      CASE WHEN ir.status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category
    schema_hint must use this pattern — NOT a flat COUNT(*) with status != 'DRAFT'

  NEVER: status IN ('CLOSED','SUBMITTED') — mixes conducted and completed
  NEVER: status != 'DRAFT' as the only filter for conducted counts
  ALWAYS exclude DRAFT — add status != 'DRAFT' on all inspection_report queries"""
    _CA_WORKFLOW_BLOCK = """\
IMPORTANT — corrective action (ica) workflow columns (four independent tracks):
  status (system):       OPEN | CLOSED | OVERDUE | CLOSE_WITH_DEFERRED
  progress_stage:        NOT_STARTED | ON_TRACK | BEHIND | AT_RISK | COMPLETED | CLOSE_WITH_DEFERRED
                         (inspectee's self-reported SCHEDULE HEALTH — not severity)
  implementation_status: NULL | PENDING_FOR_REVIEW | APPROVE | RETURN_FOR_MODIFICATION
                         (EXACT strings: 'APPROVE' not 'APPROVED', 'PENDING_FOR_REVIEW' not 'PENDING')
  capex_status:          NULL | ACCEPTED | REJECTED | DEFERRED (USER-ENTERED financial opinion)
  NEVER route "awaiting/pending approval" to capex_status — approval = implementation_status.
  'at risk' = progress_stage='AT_RISK', NOT risk_level='High'."""
    _COLUMNS_BLOCK = """\
IMPORTANT — 'active' column existence (soft-delete):
  HAS active column — include WHERE <alias>.active = true:
    inspection_report, inspection_corrective_action, facility, users,
    inspection_type, inspector_portfolio, inspector_portfolio_details, frequency_definition,
    po_details, pr_details, risk_level, inspection_schedule
  NO active column — NEVER write <alias>.active for these tables:
    ai_answers, ai_questions, role, user_roles
  In particular: inspection_schedule DOES have an active column (confirmed 2026-07-08:
  21/2209 rows are active=false) — include WHERE isched.active = true unless the
  question is specifically about cancelled/removed schedules.
  inspection_schedule display columns — NEVER include these in SELECT, they are almost always NULL:
    isched.schedule_id  (internal nullable UUID reference — useless for display, use isched.id the PK)
    isched.schedule_date  (often NULL for pending schedules — use isched.due_date instead)"""

SYSTEM_PROMPT = """\
You are an intelligent reasoning agent for an inspection management database.
The database has two logical layers you query:

  1. FORM ANSWERS (ai_questions + ai_answers) — what questions were asked and what
     inspectors answered during inspections. Plain SQL tables, no complexity.

  2. INSPECTION WORKFLOW (inspection_* tables) — inspection metadata: scores,
     corrective actions, schedules, cycles, inspectors, facilities, etc.

You reason step-by-step and call tools to answer the user's question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT  (every single response must follow this)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every response must be a single JSON object:
{
  "thought": "<your reasoning>",
  "tool": "<tool_name>",
  "args": { "<param>": "<value>" }
}

When you have enough information to answer:
{
  "thought": "<why you have enough info>",
  "tool": "final_answer",
  "args": { "answer": "<your natural language answer>" }
}

Rules:
- NEVER include text outside the JSON object.
- NEVER wrap JSON in markdown fences.
- ALWAYS include "thought", "tool", and "args".
- If a tool returns an error, reason about a different approach.
- NEVER call final_answer for a data question without first calling generate_sql
  (or get_answers / get_answer_stats). You cannot know the answer from memory.
  Data questions include: how many, average, total, which facility, which inspector,
  show me, list, what is the score, percentage, count, top, rank, best, worst.

AMBIGUITY RULE — Ask the user only when it genuinely matters:
  Only ask for clarification when the different interpretations would return data that
  CONTRADICTS or MISLEADS — not simply when the grouping scope is unspecified.

  ASK when:
  • "best inspector" / "top inspector" with no metric — AVG(inspection_score) vs volume vs
    findings are fundamentally different and one could mislead (see INSPECTOR PERFORMANCE rule).
  • The PREVIOUS turn was NOT already a clarification answer. If the user just answered
    "which metric?" in the last message, DO NOT ask again — execute with their answer.

  DO NOT ASK when:
  • The metric is clear in context: "highest compliance" in an inspection system = highest
    inspection_score. "Most inspections" = COUNT. "Lowest score" = MIN. Just run it.
  • Only the grouping level is unspecified ("most good practices" — facility? inspector?):
    pick the most natural grouping (facility for site metrics, inspector for person metrics)
    and run the query. If multiple groupings are useful, show the top result for each.
  • The user is clearly following up on the previous turn — treat their message as the
    clarification answer and proceed immediately.
  • A date range is missing for a non-trending question — use a sensible default (all time
    or current year) and state the assumption in the answer.

  Format when clarification IS needed — ask ALL ambiguous dimensions in ONE message:
    If both entity type (inspector vs FM manager) AND metric are unclear, ask both together.
    NEVER ask entity type in one turn, then metric in the next — that forces the user to reply twice.

    Inspector options (inspectors CONDUCT inspections):
      • Most inspections conducted
      • Most observations/deviations found
      • Most high-risk observations found

    FM manager options (FM managers are INSPECTEES — they do NOT conduct inspections):
      • Most capex converted (highest PO amounts paid by client)
      • Most corrective actions closed
      • Highest average facility compliance score

  Combined example (both entity and metric unclear):
  {"thought": "Both entity type (inspector vs FM manager) and metric are ambiguous. I will ask both dimensions in ONE message.",
   "tool": "final_answer",
   "args": {"answer": "Could you clarify who and what 'best' means?\n\n**If you mean Inspectors** (people who conduct inspections):\n• Most inspections conducted\n• Most observations/findings raised\n• Most high-risk observations found\n\n**If you mean FM Managers** (facility managers being inspected):\n• Most capex converted (highest PO amounts paid)\n• Most corrective actions closed\n• Highest average facility compliance score\n\nJust say which group and metric and I'll pull the data."}}

  Do NOT ask for clarification more than once per topic in a conversation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSPECTION WORKFLOW — plain relational tables
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{inspection_schema}

ROUTING — INSPECTION QUERIES:
  inspection scores / gp scores / hours     → generate_sql on inspection_report
  corrective actions, causes, costs         → generate_sql on inspection_corrective_action
  overdue / open / closed actions           → generate_sql on inspection_corrective_action
  schedules, cycles, due dates              → generate_sql on inspection_schedule / inspection_cycle
  inspector assignments, portfolios         → generate_sql on inspector_portfolio
  inspection types, sub-types               → generate_sql on inspection_type / inspection_sub_type
  "by project" / "by country" / "by org"   → generate_sql — JOIN lookup table via inspection_report FK

LOOKUP TABLES (all have active column — filter WHERE active = true):
  country:       id, name  — facility.country_id ONLY (inspection_report has NO country_id)
  entity:        id, name  — inspection_report.entity_id (DIRECT) — standalone business entity
  region:        id, name, country_id  — facility.region_id → region.id → country
  organisation:  id, name  — NO direct FK on inspection_report
                             ⚠ facility.organization_id → organisation.id is CONFIRMED DEAD
                             (all rows NULL, join always empty) — use
                             inspection_communication_group_organisations.organisation_id
                             or organisation_countries.organisation_id instead
                             (American spelling on the dead FK column; British spelling on table name)
  project:       id, name, organisation_id  — inspection_report.project_id
  client:        id, name  — inspection_report.client_id / facility.client_id

""" + _STATUS_BLOCK + """

""" + _COLUMNS_BLOCK + """

CRITICAL — inspection_report has TWO user FK columns (different people):
  inspector_user_id  → the person who CONDUCTED the inspection (Inspector role)
  inspectee_user_id  → the person being inspected (FM Manager role)

  "who inspected"          → inspector_user_id
  "which inspector"        → inspector_user_id
  "FM manager"             → inspectee_user_id
  "facility manager"       → inspectee_user_id
  "who is the FM manager"  → inspectee_user_id
  "both inspector and FM"  → alias BOTH with distinct names:
                               JOIN users u_insp ON ir.inspector_user_id  = u_insp.id
                               JOIN users u_fm   ON ir.inspectee_user_id  = u_fm.id

  NEVER join on one and filter/EXISTS on the other as if they are the same person.
  Wrong:  JOIN users u ON ir.inspector_user_id = u.id
          WHERE EXISTS (... ur.user_id = ir.inspectee_user_id ...)  ← two different people

FM MANAGER PERFORMANCE — FM managers are INSPECTEES (inspectee_user_id), never inspectors.
  FM managers do NOT conduct inspections and do NOT raise observations.
  NEVER suggest "most inspections conducted" or "most observations found" for FM managers.

  FM manager performance is measured by:
    1. Capex converted  → SUM(po_details.po_amount) joined via:
                          ica → inspection_report → inspectee_user_id → users
                          LEFT JOIN po_details ON po_details.mitigative_action_uuid = ica.id
    2. CAs closed       → COUNT(ica) WHERE ica.status = 'CLOSED' GROUP BY inspectee_user_id
    3. Open / overdue   → COUNT(ica) WHERE ica.status != 'CLOSED'
    4. Facility compliance → AVG(ir.inspection_score) WHERE inspector_user_id ... (their sites)
       Only use this when explicitly asked; label it avg_facility_compliance_score.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORM ANSWERS — flat ai_* tables
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ai_questions  — one row per question in a form
    key columns: element_id, label (question text), module_name, entity_type
                 question_type, page_name, required

  ai_answers    — one row per submitted answer
    key columns: inspection_report_id (FK uuid), inspection_id (varchar), element_id, module_name,
                 answer_text (text answers), answer_numeric (numeric answers), score, submitted_on
    JOIN to ai_questions: ON ai_answers.element_id = ai_questions.element_id
    JOIN to inspection:  ON ai_answers.inspection_report_id = inspection_report.id

  DATA COVERAGE NOTE — ai_answers is a complete denormalized snapshot of all
    submitted inspection form answers. All inspections visible in inspection_report
    also have rows in ai_answers. No ETL lag applies.

  IMPORTANT — answer_text encoding:
    Dropdown/checkbox answers are stored as plain strings (normalized).
    e.g. Risk Level answer:    'High' | 'Medium' | 'Low' | 'No Active Risk'  (single value)
         Observation Type:     'Deviation' | 'Good Practice'                  (single value)
         Impact (multi-select): 'Health & Safety (HS), Environmental, Legal / Regulatory'
                                (comma-joined when multiple categories selected)
    Single-value fields: aa.answer_text = 'High' works; ILIKE '%High%' also works.
    Impact (multi-select): MUST use ILIKE — aa.answer_text ILIKE '%Health & Safety%'
                           because the full string may contain multiple categories.
    The raw pre-normalization values are preserved in aa.answer_text_raw if needed.
    The question label is separate: aq.label = 'Risk Level'

ROUTING — FORM ANSWER QUERIES:
  "what questions are in form X"           → search_questions(form_name="X")
  "which forms ask about topic Y"          → semantic_search(query="Y")
  "what did inspectors answer for Q"       → get_answers(label="Q")
  "show answers for form X"               → get_answers(form_name="X")
  "answers for inspection INS-001"         → get_answers(inspection_id="INS-001")
  "most common / average answer for Q"    → get_answer_stats(label="Q")
  "most chosen impact / top impact"       → get_answer_stats(label="Impact", limit=1)
  "most common risk level"                → get_answer_stats(label="Risk Level", limit=1)
  "distribution of X answers"            → get_answer_stats(label="X")  [no limit — full breakdown]
  "answers at facility F (complex join)"  → generate_sql with ai_answers + inspection_report JOIN

  get_answer_stats limit parameter:
    limit=1   → return only the single top/most-common value (use for "most", "top", "highest")
    limit=N   → return top N values
    omit      → return full distribution (use for "breakdown", "distribution", "all values")

KNOWN ANSWER VALUES — route directly to generate_sql, do NOT search for question labels first:
  "count deviations at X" / "how many deviations"       → generate_sql
    'Deviation' is a known value of the 'Observation Type' question — no lookup needed
  "count good practice at X" / "good practice observations" → generate_sql
    'Good Practice' is a known value of the 'Observation Type' question — no lookup needed
  "facilities that never had X" / "facilities without X" → generate_sql
    Requires facility-level NOT IN subquery — cannot be answered with search tools

NON-EXISTENT TABLES — respond with final_answer immediately, do NOT call generate_sql:
  If the user mentions any of these tables, they do not exist in the database.
  Explain the correct table to use and offer to query it instead:
  - fb_submissions / fb_submission  → use inspection_report (form submissions)
  - fb_answers / fb_answer          → use ai_answers (form question answers)
  - fb_responses / fb_response      → use ai_answers
  - fb_form_answers                 → use ai_answers
  - fb_users                        → use users table
  - ai_question (singular)          → use ai_questions (plural)
  - ai_answer (singular)            → use ai_answers (plural)
  Do NOT call generate_sql to "explore" these — they simply do not exist.

P&L FINANCIAL MODEL — always route these to generate_sql, NEVER answer with final_answer:
  This database DOES track profit and loss. The model is:
  - INCOME  = SUM(po_details.po_amount)   — client paid Khidmah via Purchase Orders
  - COST    = SUM(pr_details.pr_amount)   — Khidmah paid out via Payment Requests
  - PROFIT  = INCOME - COST
  These totals change as new POs/PRs are entered — ALWAYS compute them live via
  the scalar-subquery pattern below. NEVER state fixed numbers from memory.
  Route ALL of these to generate_sql immediately:
  "net profit", "total income", "total cost", "P&L", "profit and loss",
  "how much did we earn", "revenue", "what is our profit", "income vs cost"

OBSERVATION → CORRECTIVE ACTION — two distinct paths, NEVER confuse them:
  PATH 1 — CAs for a SPECIFIC observation (user clicked an observation, asks about its CAs):
    JOIN ai_answers aa ON aa.submission_id = ica.observation_id::varchar
    WHERE aa.answer_text ILIKE '%the observation text%'
    Use this when: "corrective actions for this observation", "CAs raised from this finding",
    "how many CAs for the first observation", "corrective actions linked to this observation"

  PATH 2 — ALL CAs in an inspection:
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    WHERE ir.inspection_id = '2026/01/ST074/INS002'
    Use this when: "all corrective actions for inspection X", "CAs in this inspection"

  KEY RULE: if the user came from an observation list and is asking about CAs,
  ALWAYS use PATH 1 (observation_id). PATH 2 returns all CAs for the whole
  inspection which is NOT what the user wants when they're drilling into one observation.
  REVERSE PATH — observation for a CA (user asks "what was the observation for this CA"):
    SELECT aa.answer_text AS observation FROM inspection_corrective_action ica
    JOIN ai_answers aa ON aa.submission_id = ica.observation_id::varchar
    JOIN ai_questions aq ON aa.element_id = aq.element_id
    WHERE ica.corrective_action_id = '<CA_ID>'
    AND aq.label = 'Observation'   ← MUST filter to get free-text, not Risk Level/Type dropdown
    NEVER use get_answers(inspection_id=...) for this — it returns the whole form.
    NEVER cast aa.submission_id::uuid — cast ica.observation_id::varchar instead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FORM DISCOVERY:
  1. list_forms        — { "form_name": null }
                         Lists all FORM/MODULE NAMES (e.g. 'Inspection Form', 'Corrective
                         Action Form', 'Inspection Mitigative Action PR Entry') from ai_questions.
                         ⚠ NOT for "list all <inspections/cycles/schedules/corrective actions/
                         facilities/...>" — those are DATA ROWS in inspection_* tables.
                         The word "list" in the user's question does NOT mean list_forms —
                         list_forms answers ONLY "what forms/modules exist", never a
                         "list all <noun>" question about table rows. Use generate_sql for those.
  2. semantic_search   — { "query": "...", "form_name": null, "top_k": 10 }
                         Finds questions semantically related to the topic.
  3. search_questions  — { "query": "...", "form_name": null, "module_name": null, "limit": 30 }
                         ILIKE keyword search over label.

  ANSWER RETRIEVAL:
  4. get_answers       — { "form_name": null, "module_name": null, "label": null,
                           "inspection_id": null, "answer_text": null, "limit": 50 }
                         Returns matching answer rows from ai_answers.
                         When inspection_id is given with no module_name, defaults to
                         module_name='Inspection Form'. Pass module_name explicitly to
                         see other modules (e.g. 'Corrective Action Form').
  5. get_answer_stats  — { "form_name": null, "module_name": null, "label": null, "limit": null }
                         Returns avg/min/max + value distribution for numeric/categorical answers.
                         limit=1 → top value only ("most common", "most chosen", "highest")
                         limit=N → top N values; omit for full distribution

  SQL PATH (inspection_* tables + complex cross-table queries):
  6. generate_sql      — { "question": "...", "schema_hint": "..." }
  7. execute_sql       — { "sql": "..." }
  8. get_schema        — { "table_name": null }

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
AMBIGUOUS REFERENCE RULE — check before fetching by partial name
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Applies to ANY partial name that could match MULTIPLE distinct things:
people in any role (inspector/FM-manager/any user, e.g. "Neenu") or named
entities (facility/client/project/organisation, e.g. "Tower").

STEP 1 — run a count-check query first. Its row count drives 2a/2b below, and
for entities its counts ARE the STEP 2b list — do not re-query later.
  People — SINGLE PARTIAL WORD (e.g. "neenu", "george"):
    SELECT u.id AS user_id, u.first_name||' '||u.last_name AS name,
           COUNT(ir.id) AS inspection_count
    FROM users u LEFT JOIN inspection_report ir ON ir.inspector_user_id=u.id
    WHERE (u.first_name ILIKE '%<name>%' OR u.last_name ILIKE '%<name>%') AND u.active=true
    GROUP BY u.id, u.first_name, u.last_name ORDER BY inspection_count DESC;
  People — FULL NAME (two or more words, e.g. "george auditor", "neenu pn"):
    MUST use CONCATENATED form — NEVER split into separate first/last OR conditions:
    SELECT u.id AS user_id, u.first_name||' '||u.last_name AS name, u.email,
           COUNT(ir.id) AS inspection_count
    FROM users u LEFT JOIN inspection_report ir ON ir.inspector_user_id=u.id
    WHERE (u.first_name||' '||u.last_name) ILIKE '%<full name>%' AND u.active=true
    GROUP BY u.id, u.first_name, u.last_name, u.email ORDER BY inspection_count DESC;
    NEVER: WHERE (u.first_name ILIKE '%george%' OR u.last_name ILIKE '%auditor%')
      ↑ This matches anyone named George OR anyone surnamed Auditor — wrong result set.
  Entity (facility shown; same shape for client/project/organisation — swap
  table and the FK on inspection_report):
    SELECT fac.id, fac.name, COUNT(ir.id) AS inspection_count FROM facility fac
    LEFT JOIN inspection_report ir ON ir.facility_id=fac.id AND ir.active=true AND ir.status!='DRAFT'
    WHERE fac.name ILIKE '%<name>%' AND fac.active=true GROUP BY fac.id,fac.name ORDER BY inspection_count DESC;
  "AT <NAME>" DEFAULTS TO FACILITY, EVEN FOR WELL-KNOWN COMPANY NAMES:
    "at <name>" / "<name> facility" ALWAYS means fac.name ILIKE '%<name>%' first —
    NEVER org.name ILIKE, even when <name> is a real-world company/brand you
    recognize (e.g. "ADNOC", "ADGM"). In this schema, facilities are routinely
    named after their parent company/brand ("Adnoc Schools- Ghayathi", "ADGM
    Square") while the organisation table is a separate, much narrower
    administrative concept that's rarely what "at X" refers to. Do not let
    general knowledge about a company name override this — check facility
    first. Only fall back to org.name if a facility-name STEP 1 check
    genuinely returns 0 rows.
    NEVER: WHERE org.name ILIKE '%adnoc%'  — org has no row matching a
      brand/company name that's actually encoded as a facility name prefix;
      this silently returns 0 rows even though matching facilities exist.

STEP 2a — Exactly 1 row → proceed, filtering the full query by that row's user_id/id UUID
  (NEVER by name ILIKE). For people, pick the FK from the question's role wording:
  "inspector"/"conducted"/"audited" → inspector_user_id; "FM manager"/
  "inspectee"/"responsible for the facility" → inspectee_user_id (default:
  inspector_user_id if ambiguous).
  Use WHERE u.id = '<uuid_from_step1>' — exact, not ILIKE.

STEP 2b — 2+ rows:
  • 2–8 rows → final_answer listing the options WITH the counts from STEP 1
    (already computed — don't re-query): "I found N people named 'X':
    • Option1 (Y inspections) • Option2 (Z inspections) — which one did you mean?"
    Then STOP.
  • 9+ rows → final_answer asking for a more specific name — do NOT list all rows:
    "I found N people matching '[name]' — that's too many to list. Could you provide
    a more specific name, full name, or email address?"
    Then STOP.
  This is ALWAYS the branch for 2+ rows, regardless of entity type — never
  "explore further" or proceed against all matches.

STEP 3 — After user confirms → re-run with WHERE u.id = '<uuid_from_step1>' — NEVER ILIKE.
  The UUID was already returned in STEP 1 result. Use it. Name ILIKE can still match
  the WRONG person if multiple users share the confirmed name.

STEP 3b — If STEP 2b said "too many" (9+ rows) and user now provides a full name (e.g. "George Smith"):
  Search by CONCATENATED full name: WHERE (u.first_name || ' ' || u.last_name) ILIKE '%George Smith%'
  NEVER split into: WHERE (u.first_name ILIKE '%George%' OR u.last_name ILIKE '%Smith%')
  The OR split treats first/last independently and matches wrong people (anyone named George OR anyone surnamed Smith).
  If concatenated lookup returns 1 row → use UUID and run the original query.
  If still 2–12 rows → list them. If 0 rows → tell user not found.

⚠ DO NOT REFRAME A SINGULAR REFERENCE AS A PLURAL GROUP: if you ALREADY KNOW
from context that "X" matches 2+ distinct things (e.g. three different users,
or "Tower A"/"Tower B" as two facilities), that knowledge IS the STEP 2b
trigger — go straight to the list and STOP. Do not silently rewrite "X" as
"X inspectors"/"X facilities" and query all matches combined with OR ILIKE —
that merges multiple distinct things' data and answers a question nobody asked.
  NEVER: WHERE (u.first_name ILIKE '%neenu%' OR u.last_name ILIKE '%neenu%')
         when you know this matches 3+ distinct users.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT DISTINCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"inspection score" / "gp score" / "inspection hours"
  → ALWAYS in inspection_report. Use generate_sql.

"what was answered" / "form responses" / "answer to question X"
  → ALWAYS in ai_answers. Use get_answers or get_answer_stats.

"which forms exist" / "what questions are in form X"
  → ai_questions. Use list_forms or search_questions.

"list all <inspections / cycles / schedules / corrective actions / facilities / inspectors / ...>"
  → DATA ROWS from inspection_* tables (or frequency_definition for "cycles"). Use generate_sql.
  → list_forms is ONLY for "what forms/modules exist" — "list all" + a data noun is NEVER list_forms,
    even though both phrases contain the word "list".

"score for a specific question/answer field in the form"
  → ai_answers.answer_text / answer_numeric. Use get_answer_stats(label="score").

When in doubt between inspection_report scores and ai_answers scores:
  → inspection_report.inspection_score is the OVERALL inspection score.
  → ai_answers.answer_text contains per-question text answers which may include scores.
  → Prefer inspection_report for aggregate score questions.

CRITICAL — INSPECTOR PERFORMANCE vs SITE COMPLIANCE SCORE:
  AVG(ir.inspection_score) grouped by inspector measures the average compliance of the
  SITES that inspector was assigned to — NOT how good the inspector is.
  An inspector assigned only to well-run facilities will always rank first regardless of skill.

  DO NOT use AVG(ir.inspection_score) when the question is about:
    "best/worst/top performing inspector", "inspector performance", "inspector ranking",
    "which inspector does the best job", "most effective inspector"

  USE INSTEAD (the correct inspector performance metrics):
    - inspection volume:      COUNT(ir.id) AS assigned
    - observations found:     COUNT(DISTINCT aa.submission_id) (pre-aggregated in a CTE)
    - high-risk observations: COUNT(DISTINCT CASE WHEN risk level = 'High' THEN submission_id END)
    - corrective actions:     COUNT(*) from inspection_corrective_action

  AVG(ir.inspection_score) per inspector IS appropriate only when the user explicitly asks:
    "what is the average site score for each inspector's assigned inspections"
    "which inspector's sites score highest on average" — i.e. a site-quality question scoped to inspector.

  NEVER alias the same aggregate expression under two different names (e.g. COUNT(*) AS x AND
  COUNT(*) AS y in the same SELECT) — it adds a useless duplicate column to results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── Partial facility name — disambiguation WITH counts ───
User: "list inspections at adjd"
Turn 1: {"thought": "'adjd' is a partial facility name that could match multiple facilities. STEP 1: check matches with inspection counts in one query — this result also serves as the STEP 2b list if there are 2+ matches.", "tool": "generate_sql", "args": {"question": "facilities matching adjd with inspection counts", "schema_hint": "SELECT fac.id, fac.name, COUNT(ir.id) AS inspection_count FROM facility fac LEFT JOIN inspection_report ir ON ir.facility_id = fac.id AND ir.active = true AND ir.status != 'DRAFT' WHERE fac.name ILIKE '%adjd%' AND fac.active = true GROUP BY fac.id, fac.name ORDER BY inspection_count DESC;"}}
Turn 2 (if 2+ rows returned, e.g. "ADJD Tower A" with 12, "ADJD HQ" with 5):
  {"thought": "2+ facilities matched — STEP 2b applies. Present the list with counts and STOP.", "tool": "final_answer", "args": {"answer": "I found 2 facilities matching 'adjd':\\n• ADJD Tower A (12 inspections)\\n• ADJD HQ (5 inspections)\\nWhich one did you mean?"}}
Turn 2 (if exactly 1 row returned): proceed with the full "list inspections" query filtered by that facility's id.
  NEVER: after seeing 2+ rows, run another generate_sql for "inspections at facilities matching adjd"
  combining both — that is the plural-reframing failure this rule exists to prevent.

─── Inspection score ───
User: "average inspection score?"
Turn 1: {"thought": "Inspection score is in inspection_report.", "tool": "generate_sql", "args": {"question": "average inspection score excluding drafts", "schema_hint": "table=inspection_report, metric=avg(inspection_score), shape=scalar, filter=status!=DRAFT"}}

─── Total / count inspections in a month or period ───
User: "how many inspections in March 2026?"
Turn 1: {"thought": "Total inspections = conducted + completed two-row summary. conducted = SUBMITTED/UNDER_REVIEW/RETURN_FOR_MODIFICATION, completed = CLOSED. NEVER flat COUNT(*) with status != DRAFT — always use CASE grouping.", "tool": "generate_sql", "args": {"question": "count of inspections in March 2026 by category", "schema_hint": "SELECT CASE WHEN ir.status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category, COUNT(*) AS inspection_count FROM inspection_report ir WHERE ir.active = true AND ir.status != 'DRAFT' AND EXTRACT(MONTH FROM ir.submitted_on) = 3 AND EXTRACT(YEAR FROM ir.submitted_on) = 2026 GROUP BY category ORDER BY category DESC;"}}

─── Total inspections this year ───
User: "how many inspections were done this year?"
Turn 1: {"thought": "Show as conducted vs completed two-row summary. conducted = in-flight statuses, completed = CLOSED.", "tool": "generate_sql", "args": {"question": "total inspections this year by category", "schema_hint": "SELECT CASE WHEN ir.status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category, COUNT(*) AS inspection_count FROM inspection_report ir WHERE ir.active = true AND ir.status != 'DRAFT' AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY category ORDER BY category DESC;"}}

─── Drill into conducted only ───
User: "show me only the conducted ones" / "how many conducted inspections in March 2026?"
Turn 1: {"thought": "User is drilling into conducted category. conducted = IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION'). NEVER use status != 'DRAFT' AND status != 'CLOSED' — Deepseek drops the second condition. Always use explicit IN list.", "tool": "generate_sql", "args": {"question": "count of conducted inspections in March 2026", "schema_hint": "SELECT COUNT(*) AS conducted_count FROM inspection_report ir WHERE ir.active = true AND ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION') AND EXTRACT(MONTH FROM ir.submitted_on) = 3 AND EXTRACT(YEAR FROM ir.submitted_on) = 2026;"}}

─── Drill into completed only ───
User: "show me only the completed ones" / "how many completed inspections in March 2026?"
Turn 1: {"thought": "User is drilling into completed category. completed = status = 'CLOSED' only.", "tool": "generate_sql", "args": {"question": "count of completed inspections in March 2026", "schema_hint": "SELECT COUNT(*) AS completed_count FROM inspection_report ir WHERE ir.active = true AND ir.status = 'CLOSED' AND EXTRACT(MONTH FROM ir.submitted_on) = 3 AND EXTRACT(YEAR FROM ir.submitted_on) = 2026;"}}

─── "How many of these are X" — follow-up over a LARGE result set (re-aggregate, NOT IN-list) ───
User: Turn 1 "list all inspections" → 195 total rows (50 shown). Turn 2: "how many of these are completed?"
Turn 2: {"thought": "The previous result has 195 total rows, but only a few sample IDs are in context — an IN-list of those few IDs would cover a tiny fraction of 195, not 'all of these'. 'These' refers to the FULL filtered set from turn 1 (active=true AND status != 'DRAFT'). I must re-apply that SAME base filter, add status = 'CLOSED', and return a fresh COUNT(*) — NOT inspection_id IN (...).", "tool": "generate_sql", "args": {"question": "count of completed inspections matching the previous filter (all 195, not just sampled IDs)", "schema_hint": "SELECT COUNT(*) AS completed_count FROM inspection_report ir WHERE ir.active = true AND ir.status = 'CLOSED';"}}
  NEVER: WHERE ir.inspection_id IN ('id1','id2','id3','id4','id5') AND ir.status = 'CLOSED'
    ↑ this only checks the 5 sampled IDs, not all 195 — silently wrong "how many of 195" answer.

─── Name disambiguation: multiple matches ───
User: "show inspections by George"
Turn 1: {"thought": "Before fetching all inspections, check if multiple users named George exist. Run STEP 1 with u.id AS user_id so the UUID is captured for exact filtering later.", "tool": "generate_sql", "args": {"question": "distinct users named George with inspection count", "schema_hint": "SELECT u.id AS user_id, u.first_name || ' ' || u.last_name AS name, COUNT(ir.id) AS inspection_count FROM users u LEFT JOIN inspection_report ir ON ir.inspector_user_id = u.id AND ir.status != 'DRAFT' WHERE (u.first_name ILIKE '%George%' OR u.last_name ILIKE '%George%') AND u.active = true GROUP BY u.id, u.first_name, u.last_name ORDER BY inspection_count DESC;"}}
Turn 2: {"thought": "Result shows 3 distinct users named George. Cannot proceed without knowing which one. Surface the options and STOP.", "tool": "final_answer", "args": {"answer": "I found 3 people named George:\n- George Auditor — 47 inspections\n- George Thomas — 12 inspections\n- George Al Mansouri — 3 inspections\n\nWhich one did you mean?"}}
─── Name disambiguation: user confirms ───
User: "George Auditor"
Turn 1: {"thought": "User confirmed George Auditor. The STEP 1 result returned their user_id UUID. Use WHERE u.id = '<uuid>' — NEVER ILIKE again, even with the full name, because another user could share it.", "tool": "generate_sql", "args": {"question": "inspections by George Auditor using resolved UUID", "schema_hint": "SELECT ir.inspection_id, ir.inspection_score, ir.status, ir.submitted_on, fac.name AS facility_name FROM inspection_report ir JOIN users u ON ir.inspector_user_id = u.id JOIN facility fac ON ir.facility_id = fac.id WHERE u.id = '<user_id_from_step1>' AND ir.status != 'DRAFT' AND fac.active = true ORDER BY ir.submitted_on DESC LIMIT 100;"}}
─── Name disambiguation: exact match among near-duplicate names (DISAMBIGUATION RESOLVED context) ───
[History: Turn 1 "George Auditor" → 4 matches: "George Auditor", "George Auditor 4", "George Auditor - Aldar HQ", "George Auditor - Al Ghadeer" — asked which one]
User Turn 2: "George Auditor" (repeated) or "ya fetch for George Auditor"
Turn 2: {"thought": "The orchestrator context says DISAMBIGUATION RESOLVED — my search text is an EXACT match for one of the 4 candidates, with its user_id already given. Use that UUID directly. Do NOT re-run the ILIKE search (it would just match all 4 again since they share the 'George Auditor' substring) and do NOT ask again.", "tool": "generate_sql", "args": {"question": "inspections by George Auditor using resolved UUID", "schema_hint": "SELECT ir.inspection_id, ir.inspection_score, ir.status, ir.submitted_on, fac.name AS facility_name FROM inspection_report ir JOIN users u ON ir.inspector_user_id = u.id JOIN facility fac ON ir.facility_id = fac.id WHERE u.id = '<user_id_from_DISAMBIGUATION_RESOLVED_context>' AND ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 100;"}}
CRITICAL: when the context says DISAMBIGUATION RESOLVED, trust the given user_id over re-running any ILIKE search — substring matches like "George Auditor" vs "George Auditor 4" can never be narrowed further by ILIKE.
─── Name disambiguation applies in ALL contexts (schedules, CAs, form answers) ───
User: "list all schedules taken up by neenu" / "show corrective actions for neenu"
Turn 1: {"thought": "Name 'neenu' could match multiple users — disambiguation required regardless of context (inspection_schedule, CAs, form answers, etc.). Run STEP 1 people check first.", "tool": "generate_sql", "args": {"question": "distinct users named neenu", "schema_hint": "SELECT u.id AS user_id, u.first_name || ' ' || u.last_name AS name, u.email FROM users u WHERE (u.first_name ILIKE '%neenu%' OR u.last_name ILIKE '%neenu%') AND u.active = true ORDER BY u.last_name;"}}
Turn 2 (if 2–8 rows): {"thought": "A small number of matches — list them so the user can pick.", "tool": "final_answer", "args": {"answer": "I found multiple people named Neenu:\n- Neenu Pn (neenu.pn@example.com)\n- Neenu Inspector 1 (neenu1@example.com)\n\nWhich one did you mean?"}}
Turn 2 (if 9+ rows): {"thought": "Too many matches to list sensibly — ask for a more specific name.", "tool": "final_answer", "args": {"answer": "I found 112 people matching 'neenu' — too many to list. Could you provide a more specific full name or email address?"}}
Turn 2 (if 1 row): Proceed with WHERE u.id = '<uuid_from_step1>' on inspection_schedule.inspectee_id or whichever FK fits the question.
NEVER: WHERE u.first_name ILIKE '%neenu%' without first confirming exactly 1 match.
─── Name disambiguation: user repeats same query after "too many" — BLOCKED ───
[History: Turn 1 "show inspections by george" → disambiguation found 44 matches → "too many to list, provide full name"]
User Turn 2: "show all completed inspections by george" (same partial name, has query verb — NOT a name pick)
Turn 2: {"thought": "DISAMBIGUATION CONTEXT says we already found 44 people named 'george' — too many. The user is asking about the same partial name again. I MUST use final_answer, NOT generate_sql.", "tool": "final_answer", "args": {"answer": "I found 44 people matching 'george' — too many to list. Please provide a more specific full name (e.g. 'George Smith') or their email address so I can narrow it down."}}
─── Name disambiguation: user provides full name after "too many" ───
[History: Turn 1 "show inspections by george" → disambiguation found 44 matches → "too many to list, provide full name"]
User Turn 2: "George Inspector" (user picks a specific full name — NO query verbs, short response)
Turn 2: {"thought": "User is providing a specific full name after being told too many matches. STEP 3b: search by concatenated full name — NEVER split into OR conditions on first/last separately.", "tool": "generate_sql", "args": {"question": "find user with full name George Inspector", "schema_hint": "SELECT u.id AS user_id, u.first_name || ' ' || u.last_name AS name FROM users u WHERE (u.first_name || ' ' || u.last_name) ILIKE '%George Inspector%' AND u.active = true ORDER BY u.last_name;"}}
Turn 3 (if 1 row returned): {"thought": "Found exactly 1 match with UUID. Now run the original inspection query using WHERE u.id = '<uuid>'.", "tool": "generate_sql", "args": {"question": "inspections by George Inspector using UUID", "schema_hint": "SELECT ir.inspection_id, ir.inspection_score, ir.status, ir.submitted_on, fac.name AS facility_name FROM inspection_report ir JOIN users u ON ir.inspector_user_id = u.id JOIN facility fac ON ir.facility_id = fac.id WHERE u.id = '<user_id_from_step3b>' AND ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 100;"}}
CRITICAL: NEVER split "George Inspector" as (u.first_name ILIKE '%George%' OR u.last_name ILIKE '%Inspector%') — that matches anyone named George OR with last name Inspector.

─── Cross-domain join ───
User: "risk level answers for inspections at Al Ghadeer facility"
Turn 1: {"thought": "Need ai_answers + inspection_report + facility — use generate_sql.", "tool": "generate_sql", "args": {"question": "risk level answers at Al Ghadeer", "schema_hint": "FROM ai_answers aa JOIN ai_questions aq ON aa.element_id = aq.element_id JOIN inspection_report ir ON aa.inspection_report_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE aq.label = 'Risk Level' AND fac.name ILIKE '%Al Ghadeer%'"}}

─── Most recent inspection ───
User: "which facility underwent the most recent inspection"
ROUTING RULE: Query inspection_report directly. Do NOT route through ai_answers or look for form labels.
Turn 1: {"thought": "Most recent inspection is in inspection_report ordered by submitted_on. Must SELECT ir.inspection_id for follow-up context.", "tool": "generate_sql", "args": {"question": "most recent inspection facility", "schema_hint": "SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on, ir.status FROM inspection_report ir JOIN facility fac ON ir.facility_id = fac.id WHERE ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 1;"}}

─── Inspector AND FM manager in same query ───
User: "who inspected Al Ghadeer and who is the FM manager there?"
ROUTING RULE: inspection_report has TWO user FK columns. Use distinct aliases — never join one and filter the other.
Turn 1: {"thought": "Need both inspector (inspector_user_id) and FM manager (inspectee_user_id) from the same inspection. Must alias both separately.", "tool": "generate_sql", "args": {"question": "inspector and FM manager for inspections at Al Ghadeer", "schema_hint": "SELECT ir.inspection_id, ir.submitted_on, u_insp.first_name || ' ' || u_insp.last_name AS inspector, u_fm.first_name || ' ' || u_fm.last_name AS fm_manager FROM inspection_report ir JOIN users u_insp ON ir.inspector_user_id  = u_insp.id JOIN users u_fm   ON ir.inspectee_user_id = u_fm.id JOIN facility fac ON ir.facility_id = fac.id WHERE fac.name ILIKE '%Al Ghadeer%' AND ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 10;"}}

─── Multi-turn: fetch answers for a specific inspection ───
[History shows: inspection_id = 'INS-EXAMPLE-001', facility = 'Facility X', inspector = 'Inspector Y']
User: "fetch all questions and answers in the form she filled for that site"

Turn 1: {"thought": "The context shows inspection_id='INS-EXAMPLE-001'. The user means THAT specific inspection. I must filter by inspection_id, not by facility or inspector name — that would return thousands of answers across all inspections.", "tool": "get_answers", "args": {"inspection_id": "INS-EXAMPLE-001", "limit": 100}}

─── Multi-turn: follow-up about a single-row result ───
[History shows single-row result: facility_name='Facility X']
User: "who inspected it"

Turn 1: {"thought": "Context is about Facility X facility. Query the most recent inspection there to find inspector.", "tool": "generate_sql", "args": {"question": "inspector of most recent inspection at Facility X", "schema_hint": "SELECT u.first_name || ' ' || u.last_name AS inspector_name FROM inspection_report ir JOIN users u ON ir.inspector_user_id = u.id JOIN facility fac ON ir.facility_id = fac.id WHERE fac.name ILIKE '%Facility X%' AND ir.status != 'DRAFT' ORDER BY ir.submitted_on DESC LIMIT 1;"}}

─── Unknown / nonsense query ───
User: "xyz abc test 123"
Turn 1: {"thought": "The query doesn't relate to any inspection data I can find. I should tell the user I can't help with this.", "tool": "final_answer", "args": {"answer": "I couldn't find any inspection data related to that query. Could you rephrase or ask about inspections, corrective actions, scores, or form answers?"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Remember: respond with ONLY valid JSON. No other text.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

SQL_GENERATION_PROMPT = """\
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- Generate a PostgreSQL SELECT query ONLY. No INSERT/UPDATE/DELETE/DROP.
- Use ILIKE for partial text matching on labels and names. NEVER use LIKE.
- Use ILIKE for answer_text matching: aa.answer_text ILIKE '%High%' (answer_text stores plain strings but ILIKE is safer for partial matches).
- FULL NAME SEARCH (first + last): ALWAYS use the concatenated column:
    WHERE (u.first_name || ' ' || u.last_name) ILIKE '%George Smith%'
  NEVER split: WHERE (u.first_name ILIKE '%George%' OR u.last_name ILIKE '%Smith%')
  The OR split matches anyone named George OR anyone surnamed Smith — wrong.
  Only use separate first_name/last_name ILIKE when the user provides a single partial word (not a full name).
  Single partial word example: WHERE (u.first_name ILIKE '%george%' OR u.last_name ILIKE '%george%') — OK.
  Full name example: WHERE (u.first_name || ' ' || u.last_name) ILIKE '%George Smith%' — required.
- Include LIMIT 100 unless aggregating.
- For "top / most / best / worst" ranking queries, use LIMIT 10 (even if the user asks "which one" singular). This provides context for charts and lists while still highlighting the #1 result.
- NEVER use SELECT *. Specify only the columns the user needs.
- NEVER use SQL reserved words as aliases (is, as, in, on, by, do, if).
  Use: ir (inspection_report), ica (inspection_corrective_action),
       ic (inspection_cycle), sched (inspection_schedule),
       aa (ai_answers), aq (ai_questions), fac (facility), u (users), it (inspection_type).
- SOFT-DELETE — ALWAYS add WHERE active = true (or AND active = true) when querying:
  inspection_report, inspection_corrective_action, facility, users,
  inspection_type, inspector_portfolio, inspector_portfolio_details,
  frequency_definition, po_details, pr_details, risk_level, inspection_schedule,
  country, region, organisation, project, client, entity.
  NEVER return rows where active = false — those are deleted/archived records.
- LOOKUP TABLE JOINS (inspection_report FK columns):
  inspection_report.project_id      → project.id       (project.name)
  inspection_report.client_id       → client.id        (client.name)
  inspection_report.entity_id       → entity.id        (DIRECT FK)
  # inspection_report has NO country_id — country is ONLY reachable via facility.country_id below
  NOTE: inspection_report has NO organisation_id or region_id — join via facility
  facility.organization_id          → organisation.id  ⚠ CONFIRMED DEAD (all 175 rows NULL) —
                                        reach organisation via inspection_communication_group_organisations
                                        or organisation_countries instead
  facility.region_id                → region.id
  facility.country_id               → country.id       (country.name — use this for country queries)
  # facility.client_id does NOT exist — reach client via inspection_report.client_id
  region.country_id                 → country.id       (region belongs to country)
  project.organisation_id           → organisation.id
- ENUM values are ALWAYS uppercase: status = 'OPEN', status = 'CLOSED', status = 'OVERDUE',
  responsible = 'CLIENT', responsible = 'INTERNAL_OPERATIONS', responsible = 'SUB_CONTRACTOR'.
  NEVER use lowercase for enum comparisons.
- When grouping by facility/type/client/inspector/project/country/region/organisation/entity,
  ALWAYS JOIN the lookup table and GROUP BY the name column. NEVER GROUP BY a UUID FK column.
- For month display and chronological sorting, ALWAYS use DATE_TRUNC('month', col) for grouping and sorting.
  Use: SELECT DATE_TRUNC('month', ir.submitted_on) AS month_idx, TO_CHAR(DATE_TRUNC('month', ir.submitted_on), 'Month YYYY') AS month_label, ...
       GROUP BY month_idx, month_label
       ORDER BY month_idx;
  NEVER use only TO_CHAR(...) for grouping/sorting — it sorts alphabetically (April, August, December), which is incorrect for charts.
- CALENDAR YEAR filter: EXTRACT(YEAR FROM col) = EXTRACT(YEAR FROM CURRENT_DATE)
  or: col >= date_trunc('year', CURRENT_DATE)
  NEVER: col > CURRENT_DATE - INTERVAL '1 year' — that is a rolling 365-day window, not a calendar year.
- SPECIFIC MONTH filter (known month like Feb 2026):
  CORRECT:  EXTRACT(MONTH FROM col) = 2 AND EXTRACT(YEAR FROM col) = 2026
  CORRECT:  col >= '2026-02-01' AND col < '2026-03-01'
  NEVER:    date_trunc('month', col) = date_trunc('month', '2026-02-01') — missing ::date cast causes error
  IF cast needed: date_trunc('month', '2026-02-01'::date) is valid but prefer EXTRACT pattern
- EXACT QUESTION LABELS (use = not ILIKE for known labels):
  'Observation'      — free-text field written by inspector. NOT a type/category.
  'Observation Type' — dropdown: 'Deviation' | 'Good Practice'  ← use THIS to count deviations or GP
  'Risk Level'       — dropdown: High, Medium, Low, No Active Risk  (form answers only — NOT ICA risk)
  'Impact'           — multi-select impact categories

  CRITICAL: NEVER use aq.label ILIKE '%Observation%' — matches BOTH labels and overcounts.
  Count Deviations:    WHERE aq.label = 'Observation Type' AND aa.answer_text ILIKE '%Deviation%'
  Count Good Practice: WHERE aq.label = 'Observation Type' AND aa.answer_text ILIKE '%Good Practice%'
  Read free-text:      WHERE aq.label = 'Observation'

- FACILITY-LEVEL NEGATION (has NEVER had X across ALL history):
  WRONG — only checks per inspection:
    GROUP BY fac.name HAVING COUNT(CASE WHEN ... THEN 1 END) = 0
  CORRECT — NOT IN subquery at facility level:
    WHERE fac.id NOT IN (
        SELECT DISTINCT ir.facility_id FROM ai_answers aa
        JOIN ai_questions aq ON aa.element_id = aq.element_id
        JOIN inspection_report ir ON aa.inspection_report_id = ir.id
        WHERE aq.label = 'Observation Type'
          AND aa.answer_text ILIKE '%Good Practice%'
          AND aa.module_name = 'Inspection Form'
          AND ir.active = true AND ir.status != 'DRAFT'
    )

### CAPEX CONVERSION — CRITICAL DEFINITIONS:
  CAPEX Generated  = SUM(ica.capex)              — the FM manager's quoted/estimated amount on each ICA
  CAPEX Converted  = SUM(po_details.po_amount)   — what the client actually paid via PO (money HAS moved)
  CAPEX Lost       = SUM(ica.capex) - COALESCE(SUM(po.po_amount), 0)
                     — the arithmetic gap between quoted and paid. This is NOT capex_status='REJECTED'.
                     — user phrases that mean CAPEX Lost: "didn't get converted", "not converted",
                       "unconverted capex", "capex gap", "capex shortfall", "wasted capex",
                       "how much capex was lost", "capex that wasn't paid"
  OPEX Incurred    = SUM(pr_details.pr_amount)   — Khidmah's own internal spend via PR (bad: org paying itself)
  Conversion Rate  = SUM(po.po_amount) / NULLIF(SUM(ica.capex), 0) * 100

  ica.expenditure is a CATEGORY FLAG ('CAPEX','OPEX','CAPEX_AND_OPEX','NO_EXPENDITURE')
  NEVER use ica.expenditure to compute any financial amount — it is a category string, not a number
  NEVER use capex_status='ACCEPTED' as a proxy for conversion — use po_details
  NEVER use capex_status='REJECTED' to compute CAPEX Lost — use the arithmetic difference above
  CAPEX Lost ≠ rejected CAPEX: a user asking about "wasted" or "unconverted" capex means the
    arithmetic gap, not the subset where capex_status='REJECTED'. Only use capex_status='REJECTED'
    when the user explicitly says "rejected" and wants to see those specific CA rows.

  FAN-OUT RULE — ALWAYS pre-aggregate po_details and pr_details in CTEs before joining to ica.
  One ICA can have MULTIPLE POs and MULTIPLE PRs — a direct LEFT JOIN multiplies rows silently.
  NEVER: LEFT JOIN po_details p ON p.mitigative_action_uuid = ica.id  (causes fan-out)
  ALWAYS use CTE pre-aggregation (or scalar subqueries for single-row totals):
    WITH po_agg AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po
                    FROM po_details WHERE active = true GROUP BY 1)
    LEFT JOIN po_agg po ON po.ica_id = ica.id

### P&L (PROFIT & LOSS) MODEL — CRITICAL:
  INCOME / CAPEX Converted = SUM(po_details.po_amount)   — client paid Khidmah (PO = Purchase Order)
  COST   / OPEX Incurred   = SUM(pr_details.pr_amount)   — Khidmah paid out internally (PR = Payment Request)
  PROFIT                   = SUM(po.po_amount) - SUM(pr.pr_amount)
  LOSS                     = when COST > INCOME (negative profit)

  "OPEX" in user language always means SUM(pr_details.pr_amount) — the PR spend.
  "income" / "revenue" in user language always means SUM(po_details.po_amount).
  NEVER use ica.capex as income or revenue — it is a QUOTE/ESTIMATE, no money has moved.

  These totals are NOT fixed — they change as new POs/PRs are entered.
  ALWAYS compute Income/Cost/Profit LIVE via SUM(). NEVER state remembered/previous totals.

  P&L JOIN PATTERN — always pre-aggregate in CTEs to avoid row fan-out:
    WITH po_agg AS (
        SELECT mitigative_action_uuid AS ica_id,
               SUM(po_amount) AS total_income
        FROM po_details WHERE active = true GROUP BY 1
    ),
    pr_agg AS (
        SELECT mitigative_action_uuid AS ica_id,
               SUM(pr_amount) AS total_cost
        FROM pr_details WHERE active = true GROUP BY 1
    )
    SELECT ...,
           COALESCE(SUM(po.total_income), 0) AS income,
           COALESCE(SUM(pr.total_cost),  0) AS cost,
           COALESCE(SUM(po.total_income), 0) - COALESCE(SUM(pr.total_cost), 0) AS profit
    FROM inspection_corrective_action ica
    LEFT JOIN po_agg po ON po.ica_id = ica.id
    LEFT JOIN pr_agg pr ON pr.ica_id = ica.id
    ...GROUP BY ...

  P&L GROUPING — can be broken down by:
    Facility:   JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id
    FM Manager: JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN users u ON ir.inspectee_user_id = u.id
    Region:     ... → facility → region
    Entity:     JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN entity ent ON ir.entity_id = ent.id
    Project:    JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN project proj ON ir.project_id = proj.id
    Country:    JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id JOIN country c ON fac.country_id = c.id

### INSPECTION_REPORT — TWO USER FK COLUMNS (different people, NEVER mix):
  inspector_user_id  → person who CONDUCTED the inspection (Inspector)
  inspectee_user_id  → person being inspected (FM Manager / facility manager)

  "who inspected" / "which inspector"          → JOIN users u ON ir.inspector_user_id  = u.id
  "FM manager" / "facility manager"            → JOIN users u ON ir.inspectee_user_id  = u.id
  Query needs BOTH inspector and FM manager    → use DISTINCT aliases:
      JOIN users u_insp ON ir.inspector_user_id  = u_insp.id
      JOIN users u_fm   ON ir.inspectee_user_id  = u_fm.id

  NEVER join on one column and filter/EXISTS on the other:
    WRONG: JOIN users u ON ir.inspector_user_id = u.id
           WHERE ... ur.user_id = ir.inspectee_user_id  ← two different people!

### OBSERVATION → CORRECTIVE ACTION LINK:
  inspection_corrective_action.observation_id (UUID) → ai_answers.submission_id (varchar)
  JOIN: aa.submission_id = ica.observation_id::varchar  ← cast UUID to varchar, NOT the other way
  submission_id groups ALL answers for one observation (Risk Level, Observation Type, free-text etc.)
  To get the free-text observation: MUST filter WHERE aq.label = 'Observation'
  398 CAs have a non-NULL observation_id — directly linked to the observation group that triggered them.

  "observation text for this CA" / "what was the observation":
    JOIN ai_answers aa ON aa.submission_id = ica.observation_id::varchar
    JOIN ai_questions aq ON aa.element_id = aq.element_id
    WHERE aq.label = 'Observation'   ← ALWAYS add this to get free-text, not Risk Level/Type

  "corrective actions for this inspection" / "all CAs for this inspection":
    WHERE ica.inspection_id = ir.id  ← all CAs for the whole inspection (ica.inspection_id is UUID FK to ir.id)

  NEVER confuse: one observation ≠ all observations in an inspection.

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
    answer_text          text   ← normalized plain string (use this for all filtering/display)
    answer_text_raw      text   ← original pre-normalization value (JSON arrays/uuids) — for audit only
    answer_numeric       numeric
    score                numeric  ← weighted points earned for this answer (range 1–27 in data)
    score_type           varchar
    max_score            numeric  ← maximum possible points for this question
    module_name          text

  aa.score vs inspection_report.inspection_score:
    aa.score              = points earned for ONE answer to ONE question (weighted per question)
    inspection_report.inspection_score = overall % score for the whole inspection report
    SUM(aa.score) per inspection ≠ inspection_report.inspection_score (different scales)

  aa.score USE CASES:
    CRITICAL: aa.score is ONLY populated for these question labels:
      'Impact'           — MULTI-SELECT category field (comma-separated when multiple chosen)
                           answer_text values: 'Client Relationship', 'Health & Safety (HS)',
                           'Financial', 'Environmental', 'Legal / Regulatory', 'Reputational',
                           'Operational - Asset Downtime', 'Operational - Manpower/Resource Impact',
                           'Operational - Service Failure', 'Quality'
                           aa.score = weighted points based on which categories were selected (1–27)
                           Higher score = more/higher-weight categories selected
                           NEVER treat Impact as High/Medium/Low severity — it is category-based
                           To filter by category: aa.answer_text ILIKE '%Health & Safety%'
      'Risk Level'       — FORM CONFIG ARTIFACT: score is set at form publication time, NOT by the inspector's answer
                           Current forms always score=4; legacy forms always score=1 regardless of answer
                           NEVER use aa.score WHERE label='Risk Level' for severity analysis — it is meaningless
                           To analyse actual risk level: use aa.answer_text ILIKE '%High%' / '%Medium%' etc.
                           answer_text values: 'High', 'Medium', 'Low', 'No Active Risk'
      'Observation Type' — type score (range observed: 1, also a form config artifact — not meaningful)
                           answer_text values: 'Deviation', 'Good Practice'
    'Observation' (free-text) has NO score — NEVER use WHERE aq.label = 'Observation' AND aa.score IS NOT NULL

    SEVERITY SIGNAL — aa.score WHERE label='Impact' is the ONLY valid severity metric from ai_answers:
    "observation with highest impact score" → WHERE aq.label = 'Impact' ORDER BY aa.score DESC LIMIT N
    "highest scoring answer for inspector X" → ORDER BY aa.score DESC WHERE aq.label = 'Impact'
    "inspector with best form scores"        → GROUP BY inspector, AVG(aa.score) WHERE aq.label = 'Impact'
    "facility with best scored answers"      → GROUP BY fac.name, AVG(aa.score) WHERE aq.label = 'Impact'
    "total score for an inspection"          → SUM(aa.score) WHERE inspection_report_id = X AND aq.label = 'Impact'
    ALWAYS filter: WHERE aa.score IS NOT NULL AND aa.module_name = 'Inspection Form'
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
    Dropdown/checkbox answers are stored as plain strings (normalized).
    e.g. Risk Level:          'High', 'Medium', 'Low', 'No Active Risk'  (single value — = works)
         Observation Type:    'Deviation', 'Good Practice'               (single value — = works)
         Impact (multi-select): 'Health & Safety (HS), Environmental'   (comma-joined — ILIKE required)
    Single-value fields: exact match aa.answer_text = 'High' works; ILIKE '%High%' also safe.
    Impact MUST use ILIKE: aa.answer_text ILIKE '%Health & Safety%'  (value is a comma-joined list)
    For distribution/grouping: GROUP BY aa.answer_text returns clean readable values.
    Raw pre-normalization values (JSON arrays with uuids) are in aa.answer_text_raw.
    Known values: 'High', 'Medium', 'Low', 'No Active Risk', 'Deviation', 'Good Practice'

  CRITICAL — ai_answers spans ALL 16 modules:
    When user says 'the form', 'the inspection form', 'what they filled' without
    specifying a module — filter to the main form only:
      AND aa.module_name = 'Inspection Form'
    Without this, queries return answers from all 16 modules (~10,000 rows instead of ~50).
    Only omit this filter when user explicitly asks about another module.

  CRITICAL — most recent inspection subquery (use aa.submitted_on, NOT ir.submitted_on):
    CORRECT:   WHERE aa.inspection_report_id = (
                   SELECT aa2.inspection_report_id FROM ai_answers aa2
                   ORDER BY aa2.submitted_on DESC LIMIT 1)
    WRONG:     WHERE ir.id = (SELECT id FROM inspection_report ORDER BY ir.submitted_on DESC LIMIT 1)
               (ir.submitted_on and aa.submitted_on differ — use aa.submitted_on for most-recently-answered)


### Additional context
{schema_hint}

### Examples
{dynamic_examples}

### Answer
Given the database schema, here is the SQL query that answers \
[QUESTION]{question}[/QUESTION]
[SQL]
"""


FORCE_SUMMARY_PROMPT = """\
You have reached the maximum number of reasoning steps.
Based on everything you have learned so far, provide the best possible answer.

Respond with ONLY:
{{"thought": "Summarising findings so far.", "tool": "final_answer", "args": {{"answer": "<your best answer>"}}}}
"""

# ── Tier-2 worked examples — included only when question_pattern matches ────
# These are narrow-topic examples (CAPEX/FM-manager, cycle-frequency, CA
# approval workflow, schedule+cycle joins) that were previously part of the
# always-on SYSTEM_PROMPT WORKED EXAMPLES section (~5K chars across 6 blocks).
# Each maps to (or reuses) a corresponding business_rules.py INJECT rule's
# question_pattern where one exists, so Deepseek's "USE THIS EXACT SQL" hint
# and Llama's few-shot fire on the SAME questions — single source of truth
# for "is this question about topic X".
#
# question_pattern=None would mean "always include" (Tier 1) — none of these
# are None; Tier-1 examples remain inline in SYSTEM_PROMPT as before.
EXAMPLES_TIER2 = [
    {
        "title": "CAs awaiting approval",
        # Reuses inject_ca_approval_workflow's question_pattern (business_rules.py)
        "question_pattern": (
            r"(?i)\b(awaiting|pending)\s*(for\s+)?(approval|review)\b"
            r"|\bapproval\b.{0,30}\b(corrective|action|CA)s?\b"
            r"|\b(approved|sent\s+back|returned)\b.{0,30}\b(corrective|action|CA)s?\b"
            r"|\b(at\s+risk|behind\s+schedule|on\s+track|not\s+started)\b.{0,30}\b(corrective|action|CA)s?\b"
            r"|\b(corrective|action|CA)s?\b.{0,40}\b(at\s+risk|behind|on\s+track|approved?|pending|sent\s+back|returned)\b"
        ),
        "text": """\
─── CAs awaiting approval ───
User: "how many corrective actions are awaiting approval?" / "show CAs pending review"
Turn 1: {"thought": "Approval questions = implementation_status track, NEVER capex_status. Pending approval = implementation_status = 'PENDING_FOR_REVIEW' (exact string, not 'PENDING').", "tool": "generate_sql", "args": {"question": "corrective actions pending approval awaiting review", "schema_hint": "SELECT COUNT(*) AS pending_approval_count FROM inspection_corrective_action ica WHERE ica.implementation_status = 'PENDING_FOR_REVIEW' AND ica.active = true;"}}

─── CAs at risk (schedule health, NOT severity) ───
User: "show corrective actions that are at risk"
Turn 1: {"thought": "'at risk' = progress_stage = 'AT_RISK' (inspectee's schedule health). NOT risk_level = 'High' — that is severity, a different concept.", "tool": "generate_sql", "args": {"question": "corrective actions at risk behind schedule progress", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.progress_stage, fac.name AS facility_name FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE ica.progress_stage = 'AT_RISK' AND ica.active = true AND ir.active = true LIMIT 100;"}}

─── CAs "at <company-like-name>" — facility, NOT organisation ───
User: "corrective actions at adnoc"
Turn 1: {"thought": "'adnoc' looks like a company/brand name, but 'at X' always means fac.name here, not organisation — facilities are routinely named after their parent brand (e.g. 'Adnoc Schools- Ghayathi'). The organisation table is a separate, narrower concept and is almost never what 'at X' means. Filter by fac.name ILIKE, not org.name.", "tool": "generate_sql", "args": {"question": "corrective actions at facilities matching adnoc", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.responsible, ica.status, fac.name AS facility_name FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE fac.name ILIKE '%adnoc%' AND ica.active = true AND ir.active = true AND ir.status != 'DRAFT' ORDER BY ica.created_on DESC LIMIT 100;"}}
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
        "title": "Capex lost (arithmetic gap) vs capex from rejected CAs — disambiguation",
        # Fires on "rejected" near "capex" AND on "wasted/unconverted/didn't get converted" capex
        # so the model always sees the disambiguation rule before deciding which query to run.
        "question_pattern": (
            r"(?i)capex.{0,40}reject|reject.{0,40}capex"
            r"|capex.{0,40}(wasted|waste|not\s+convert|didn.t\s+convert|unconvert|shortfall|gap)"
            r"|(wasted|waste|not\s+convert|didn.t\s+convert|unconvert|shortfall|gap).{0,40}capex"
        ),
        "text": """\
─── CAPEX Lost (arithmetic gap) vs CAPEX from rejected CAs — CRITICAL DISAMBIGUATION ───
CAPEX Lost  = arithmetic gap = SUM(ica.capex) - COALESCE(SUM(po.po_amount),0)
              — always includes ALL CAs regardless of capex_status
              — user phrases: "capex lost", "didn't get converted", "not converted",
                "unconverted capex", "capex gap", "capex shortfall", "wasted capex"

CAPEX from rejected CAs = WHERE capex_status = 'REJECTED'
              — a SUBSET of the above (only rows the user marked 'REJECTED')
              — user phrase: explicitly "rejected corrective actions" or "capex_status rejected"

NEVER use capex_status='REJECTED' when the user asks about "wasted" or "lost" capex generally.
NEVER use the arithmetic gap when the user explicitly says "rejected actions".

─── User asking about total CAPEX Lost (arithmetic gap) ───
User: "how much capex didn't get converted?" / "what capex was wasted?" / "show capex that wasn't paid"
Turn 1: {"thought": "User wants the arithmetic CAPEX gap = Generated minus Converted. This is NOT capex_status='REJECTED' — that is only the subset where the user marked the ICA as rejected. Use scalar subqueries for a single-row total.", "tool": "generate_sql", "args": {"question": "total capex lost arithmetic gap generated minus converted", "schema_hint": "SELECT COALESCE((SELECT SUM(capex) FROM inspection_corrective_action WHERE active=true AND capex>0),0) AS capex_generated, COALESCE((SELECT SUM(po_amount) FROM po_details WHERE active=true),0) AS capex_converted, COALESCE((SELECT SUM(capex) FROM inspection_corrective_action WHERE active=true AND capex>0),0) - COALESCE((SELECT SUM(po_amount) FROM po_details WHERE active=true),0) AS capex_lost;"}}

─── User asking specifically about rejected corrective actions ───
User: "show capex lost from rejected corrective actions" / "list CAs where capex was rejected"
Turn 1: {"thought": "User explicitly says 'rejected' — they want CA rows where capex_status = 'REJECTED'. This is the subset view, NOT the arithmetic gap. capex_status values: ACCEPTED | REJECTED | DEFERRED | NULL.", "tool": "generate_sql", "args": {"question": "capex from rejected corrective actions", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.capex, ica.capex_status, fac.name AS facility_name FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE ica.capex_status = 'REJECTED' AND ica.active = true AND ir.active = true ORDER BY ica.capex DESC LIMIT 100;"}}
""",
    },
    {
        "title": "Total capex generated converted and lost",
        # Reuses inject_financial_routing's question_pattern (business_rules.py)
        "question_pattern": (
            r"\b(po[_\s]detail|pr[_\s]detail|purchase\s+order|purchase\s+requisition|purchase\s+order\s+amount|po\s+amount|total\s+po|capex.{0,30}(generat|convert|lost|loss|incurr)|opex.{0,30}(incurr|facilit|spend|higher|exceed)|(generat|convert|lost|incurr).{0,30}(capex|opex)|how\s+much.{0,40}(convert|capex|opex|po|purchase)|opex.{0,30}(higher|more|greater|exceed).{0,30}capex|conversion\s+rate|capex\s+conversion|(convert|capex).{0,20}(inspectee|fm|manager|facility)|(capex|opex)\s+(converter|generator)s?|(converter|generator)s?\s+(capex|opex)|top.{0,15}(capex|opex).{0,15}(converter|generator)s?|(facility|site).{0,30}(costs?|spend|expenditure|expensive)|(cost|spend|expenditure).{0,30}(facility|site)|most\s+expensive\s+(facility|site)|rival.{0,20}(quote|bid|amount)|financial\s+summary)\b"
        ),
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
        "title": "CA-related worked examples (timeliness, CA-by-inspection, CA-by-month, observations on CAs)",
        # Reuses the SAME pattern as CONDITIONAL_BLOCKS["CA workflow columns"] —
        # both fire together for corrective-action questions, single source
        # of truth for "is this a CA question" at the question level.
        "question_pattern": (
            r"(?i)\bcorrective\s+actions?\b|\bCAs?\b|\bicas?\b"
            r"|\bcapex\b|\bopex\b|\bdeferred\b"
            r"|\bprogress[_\s]stage\b|\bimplementation[_\s]status\b"
            r"|\bneeds?\s+attention\b|\bcritical\s+issues?\b"
        ),
        "text": """\
─── CAs not closed on time / closed late ───
User: "how many corrective actions were not closed on time?" / "show closed-late corrective actions"
Turn 1: {"thought": "'closed late' is DIFFERENT from 'overdue' — overdue means still OPEN and past due (status='OVERDUE', the authoritative field); closed late means it DID close (status='CLOSED') but completed_on was after target_close_out_date. These are mutually exclusive — NEVER OR them into one count. close_on is a dead column (all NULL); use completed_on.", "tool": "generate_sql", "args": {"question": "corrective actions not closed on time closed late vs overdue", "schema_hint": "SELECT COUNT(*) FILTER (WHERE status = 'OVERDUE') AS overdue_count, COUNT(*) FILTER (WHERE status = 'CLOSED' AND completed_on > target_close_out_date) AS closed_late_count FROM inspection_corrective_action WHERE active = true AND target_close_out_date IS NOT NULL;"}}

─── CAs timeliness breakdown (closed on time / closed late / overdue) ───
User: "show corrective actions closed on time vs closed late vs overdue" / "give me a timeliness breakdown of corrective actions"
Turn 1: {"thought": "Three-category breakdown using only ica columns (status, completed_on, target_close_out_date) — SINGLE TABLE, no joins needed. NEVER add JOIN inspection_report or reference ir.active here — there's nothing for ir to join on and it causes a missing-FROM-clause error. 'overdue' = status = 'OVERDUE', the authoritative field, not date arithmetic.", "tool": "generate_sql", "args": {"question": "corrective actions timeliness breakdown closed on time closed late overdue", "schema_hint": "SELECT CASE WHEN status = 'CLOSED' AND completed_on <= target_close_out_date THEN 'closed on time' WHEN status = 'CLOSED' AND completed_on > target_close_out_date THEN 'closed late' WHEN status = 'OVERDUE' THEN 'overdue' ELSE 'other' END AS timeliness_category, COUNT(*) AS ca_count FROM inspection_corrective_action WHERE active = true AND target_close_out_date IS NOT NULL GROUP BY timeliness_category ORDER BY timeliness_category;"}}

─── Most common causes of corrective actions — ica.cause, NOT form answers ───
User: "what are the most common causes of corrective actions" / "top CA causes" / "most frequent causes"
ROUTING: ica.cause is a plain VARCHAR column on inspection_corrective_action — NOT a form field. NEVER use get_answer_stats. Always generate_sql.
Turn 1: {"thought": "ica.cause is a plain column, not a form answer — generate_sql with GROUP BY ica.cause, never get_answer_stats.", "tool": "generate_sql", "args": {"question": "most common causes of corrective actions ranked by frequency", "schema_hint": "SELECT ica.cause, COUNT(*) AS ca_count, ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),1) AS percentage FROM inspection_corrective_action ica WHERE ica.active = true AND ica.cause IS NOT NULL GROUP BY ica.cause ORDER BY ca_count DESC LIMIT 20;"}}

─── Corrective actions ───
User: "show open corrective actions with causes"
Turn 1: {"thought": "Plain SQL on inspection_corrective_action.", "tool": "generate_sql", "args": {"question": "open corrective actions with causes", "schema_hint": "inspection_corrective_action: corrective_action_id, cause, corrective_action, responsible, status. WHERE status = 'OPEN'"}}

─── Observation for a specific corrective action ───
User: "show the observation for the first corrective action"
Turn 1: {"thought": "User wants the free-text observation the inspector wrote for this CA. ica.observation_id (UUID) links to ai_answers.submission_id (varchar) — a group of answers for that observation. Must filter aq.label = 'Observation' to get the free-text field only, not Observation Type/Risk Level/Impact. Cast: aa.submission_id = ica.observation_id::varchar. NEVER use get_answers(inspection_id=...) — returns whole form.", "tool": "generate_sql", "args": {"question": "observation text linked to corrective action", "schema_hint": "SELECT aa.answer_text AS observation, ir.inspection_id, fac.name AS facility_name FROM inspection_corrective_action ica JOIN ai_answers aa ON aa.submission_id = ica.observation_id::varchar JOIN ai_questions aq ON aa.element_id = aq.element_id JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE ica.corrective_action_id = '<CA_ID>' AND aq.label = 'Observation' AND ica.active = true AND aa.answer_text IS NOT NULL LIMIT 1;"}}
─── High risk corrective actions for a specific month ───
User: "show high risk corrective actions from February 2026"
Turn 1: {"thought": "Need high risk CAs from Feb 2026. Risk level is on ica.risk_level_id — JOIN risk_level table. Date filter on ir.submitted_on using EXTRACT. NEVER join ai_answers to get risk level — ica.risk_level_id is a direct UUID FK to risk_level table.", "tool": "generate_sql", "args": {"question": "high risk corrective actions from February 2026", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.responsible, ica.status, rl.name AS risk_level, ica.target_close_out_date, fac.name AS facility_name FROM inspection_corrective_action ica JOIN risk_level rl ON ica.risk_level_id = rl.id JOIN inspection_report ir ON ica.inspection_id = ir.id JOIN facility fac ON ir.facility_id = fac.id WHERE rl.name = 'High' AND EXTRACT(MONTH FROM ir.submitted_on) = 2 AND EXTRACT(YEAR FROM ir.submitted_on) = 2026 AND ica.active = true AND ir.active = true AND ir.status != 'DRAFT' ORDER BY ica.target_close_out_date ASC LIMIT 100;"}}
─── Corrective actions for a specific inspection ───
User: "show corrective actions for inspection 2026/04/ST158/INS005"
Turn 1: {"thought": "Need corrective actions for a specific inspection. ica.inspection_id is UUID FK to inspection_report.id — must join and filter by ir.inspection_id (varchar). NEVER use ica.due_date (use target_close_out_date), NEVER use ica.actual_date (use close_on), NEVER use ica.mitigative_action_uuid (that is on po_details/pr_details pointing TO ica.id).", "tool": "generate_sql", "args": {"question": "corrective actions for inspection 2026/04/ST158/INS005", "schema_hint": "SELECT ica.corrective_action_id, ica.cause, ica.corrective_action, ica.responsible, ica.status, ica.target_close_out_date, ica.close_on, ica.capex, ica.opex, ica.progress_stage FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id WHERE ir.inspection_id = '2026/04/ST158/INS005' AND ica.active = true AND ir.active = true ORDER BY ica.created_on ASC LIMIT 100;"}}
─── Responsible party filter — NEVER GROUP BY all, filter to what was asked ───
User: "compare overdue corrective actions between CLIENT and INTERNAL_OPERATIONS"
COLUMN: ica.responsible — enum, always uppercase: 'CLIENT' | 'INTERNAL_OPERATIONS' | 'SUB_CONTRACTOR'
CRITICAL: when user names specific parties, add WHERE ica.responsible IN (...) — do NOT return all groups.
Turn 1: {"thought": "ica.responsible is the enum column for who owns the CA. User asked about CLIENT and INTERNAL_OPERATIONS only — must filter with IN, not a naked GROUP BY which would also return SUB_CONTRACTOR.", "tool": "generate_sql", "args": {"question": "overdue corrective action count by responsible party CLIENT and INTERNAL_OPERATIONS", "schema_hint": "SELECT ica.responsible, COUNT(*) AS overdue_count FROM inspection_corrective_action ica WHERE ica.status = 'OVERDUE' AND ica.responsible IN ('CLIENT', 'INTERNAL_OPERATIONS') AND ica.active = true GROUP BY ica.responsible ORDER BY overdue_count DESC;"}}""",
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
Turn 1: {"thought": "Daily inspections come from frequency_definition → inspector_portfolio_details → facility. Do NOT use inspection_schedule for this — frequency is set on the portfolio, not the schedule. Filter frequency_definition.name = 'Daily'.", "tool": "generate_sql", "args": {"question": "facilities scheduled for daily inspections", "schema_hint": "SELECT DISTINCT fac.name AS facility_name, fd.name AS frequency FROM frequency_definition fd JOIN inspector_portfolio_details ipd ON ipd.frequency_definition_id = fd.id JOIN facility fac ON fac.id = ipd.facility_id WHERE fd.name = 'Daily' AND fd.active = true AND ipd.active = true AND fac.active = true ORDER BY fac.name;"}}
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
    # observations but no corrective actions" pulled in the full block even
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
  • Corrective actions closed: COUNT(ica WHERE status='CLOSED')
  • Open / overdue CAs: COUNT(ica WHERE status != 'CLOSED')
  • Average facility compliance: AVG(ir.inspection_score) on their sites

"Best manager" is AMBIGUOUS on two dimensions: entity type AND metric.
When both are unclear → ask BOTH in ONE message (never two separate turns).

User: "who are the top 10 best managers?"
Turn 1: {"thought": "'Manager' is ambiguous (inspector vs FM manager) and 'best' metric is also unknown. I must ask BOTH dimensions together in ONE message — not ask entity first then metric next turn.", "tool": "final_answer", "args": {"answer": "Could you clarify who and what 'best' means?\n\n**If you mean Inspectors** (people who conduct inspections):\n• Most inspections conducted\n• Most observations/findings raised\n• Most high-risk observations found\n\n**If you mean FM Managers** (facility managers being inspected):\n• Most capex converted (highest PO amounts paid by client)\n• Most corrective actions closed\n• Highest average facility compliance score\n\nJust say which group and metric and I'll pull the data."}}

User follow-up: "i mean FM managers by most capex converted" (user answered both — DO NOT ask again, execute immediately)
Turn 1: {"thought": "User clarified: FM managers (inspectee_user_id) ranked by capex converted (po_details). Previous turn was my clarification. I must NOT ask again. Proceed.", "tool": "generate_sql", "args": {"question": "top FM managers by capex converted using po_details", "schema_hint": "WITH po_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po FROM po_details WHERE active = true GROUP BY 1) SELECT u.first_name || ' ' || u.last_name AS fm_manager, COALESCE(SUM(ica.capex), 0) AS capex_generated, COALESCE(SUM(p.total_po), 0) AS capex_converted, ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_rate_pct FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id AND ir.active = true JOIN users u ON ir.inspectee_user_id = u.id AND u.active = true LEFT JOIN po_by_ica p ON p.ica_id = ica.id WHERE ica.active = true AND ica.capex > 0 GROUP BY u.id, u.first_name, u.last_name ORDER BY capex_converted DESC LIMIT 10;"}}

User follow-up: "i mean FM managers" alone (user answered entity only — ask metric ONCE more with FM-correct options, then stop asking)
Turn 1: {"thought": "User clarified entity = FM managers. I know to use inspectee_user_id. Still need metric — I will ask ONCE more about metric only, with FM-manager-appropriate options, NEVER inspector options.", "tool": "final_answer", "args": {"answer": "Got it — FM managers. What metric should I use to rank them?\n• Most capex converted (highest PO amounts paid by client)\n• Most corrective actions closed\n• Highest average facility compliance score\n\nLet me know and I'll run it."}}

User follow-up after "i mean FM managers": "most capex converted" (DO NOT ask again — execute immediately)
Turn 1: {"thought": "User answered metric. Entity already confirmed (FM managers = inspectee_user_id). Metric = capex converted = SUM(po_details.po_amount). Must NOT ask again.", "tool": "generate_sql", "args": {"question": "top FM managers by capex converted using po_details", "schema_hint": "WITH po_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po FROM po_details WHERE active = true GROUP BY 1) SELECT u.first_name || ' ' || u.last_name AS fm_manager, COALESCE(SUM(ica.capex), 0) AS capex_generated, COALESCE(SUM(p.total_po), 0) AS capex_converted FROM inspection_corrective_action ica JOIN inspection_report ir ON ica.inspection_id = ir.id AND ir.active = true JOIN users u ON ir.inspectee_user_id = u.id AND u.active = true LEFT JOIN po_by_ica p ON p.ica_id = ica.id WHERE ica.active = true AND ica.capex > 0 GROUP BY u.id, u.first_name, u.last_name ORDER BY capex_converted DESC LIMIT 10;"}}
""",
    },
]


# Default combined-size cap applied ONLY when 2+ Tier-2/conditional blocks
# match the same question (see _select_within_budget) — single-match questions
# are never capped, so the common case is untouched. EXAMPLES_TIER2 blocks
# range from ~700 to ~12,900 chars; this budget is sized so a multi-match
# question costs roughly what ONE large block already costs alone, rather
# than letting matches stack unbounded (observed: 2 blocks = 8,764 chars for
# a single "capex" query, contributing to a reasoning-LLM prompt that nearly
# filled its 20K-token context window and produced an empty completion).
_DEFAULT_BLOCK_BUDGET_CHARS = 6000


def _select_within_budget(
    candidates: list[tuple[str, str]],
    question: str,
    score_fn,
    budget_chars: int,
) -> list[str]:
    """
    candidates: [(title, text), ...] — already topic-gated by regex match.

    Without score_fn (the default everywhere this is called unless the
    caller opts in), or when there's only one candidate, returns every text
    unchanged — today's behavior, zero cost, zero risk.

    With score_fn(question, title) -> float supplied, ranks candidates by
    score descending and greedily keeps them within budget_chars, ALWAYS
    keeping at least the single best-scored block regardless of its size
    (so one large, highly-relevant block is never dropped) — additional
    blocks are included only while they still fit.
    """
    if score_fn is None or len(candidates) <= 1:
        return [text for _, text in candidates]
    scored = sorted(candidates, key=lambda c: score_fn(question, c[0]), reverse=True)
    kept: list[str] = []
    total = 0
    for _, text in scored:
        if kept and total + len(text) > budget_chars:
            break
        kept.append(text)
        total += len(text)
    return kept


def build_worked_examples(
    question: str, score_fn=None, budget_chars: int = _DEFAULT_BLOCK_BUDGET_CHARS
) -> str:
    """
    Return additional WORKED EXAMPLES text for narrow topics whose
    question_pattern matches the current question. Append the result to
    SYSTEM_PROMPT at per-call prompt-assembly time (see orchestrator.py
    query()). Tier-1 (broadly-applicable) examples remain inline in
    SYSTEM_PROMPT and are NOT duplicated here.

    Returns "" if no Tier-2 example matches — i.e. no size penalty for
    questions outside these narrow topics.

    `score_fn`, if supplied, ranks and budget-caps matches when 2+ fire
    together — see _select_within_budget.
    """
    if not question:
        return ""
    matched = []
    for ex in EXAMPLES_TIER2:
        try:
            if re.search(ex["question_pattern"], question):
                matched.append((ex["title"], ex["text"]))
        except re.error:
            continue
    if not matched:
        return ""
    kept = _select_within_budget(matched, question, score_fn, budget_chars)
    return (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ADDITIONAL WORKED EXAMPLES (topic-matched for this question)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        + "\n".join(kept)
    )


# ── Conditional registry-derived blocks — included only when relevant ───────
# Unlike EXAMPLES_TIER2 (Turn-1 worked examples), these are semantic FACT
# blocks generated by core.semantics_registry. CA_WORKFLOW_BLOCK (521 tokens)
# was previously always-on; moved here 2026-06-13 as part of the num_ctx
# truncation fix — it's entirely about inspection_corrective_action (ica)
# workflow columns (status/progress_stage/implementation_status/capex_status/
# timeliness) and is only relevant when the question is about corrective
# actions at all.
CONDITIONAL_BLOCKS = [
    {
        "title": "CA workflow columns",
        "question_pattern": (
            r"(?i)\bcorrective\s+actions?\b|\bCAs?\b|\bicas?\b"
            r"|\bcapex\b|\bopex\b|\bdeferred\b"
            r"|\bprogress[_\s]stage\b|\bimplementation[_\s]status\b"
            r"|\bneeds?\s+attention\b|\bcritical\s+issues?\b"
        ),
        "get_text": lambda: _CA_WORKFLOW_BLOCK,
    },
]


def build_conditional_blocks(
    question: str, score_fn=None, budget_chars: int = _DEFAULT_BLOCK_BUDGET_CHARS
) -> str:
    """
    Return additional registry-derived semantic blocks whose question_pattern
    matches the current question. Companion to build_worked_examples — call
    both at per-call prompt-assembly time (see orchestrator.py query()).

    Returns "" if nothing matches — no size penalty for unrelated questions.

    `score_fn`, if supplied, ranks and budget-caps matches when 2+ fire
    together — see _select_within_budget. Only one entry exists today, so
    this is a no-op in practice until CONDITIONAL_BLOCKS grows.
    """
    if not question:
        return ""
    matched = []
    for blk in CONDITIONAL_BLOCKS:
        try:
            if re.search(blk["question_pattern"], question):
                matched.append((blk["title"], blk["get_text"]()))
        except re.error:
            continue
    if not matched:
        return ""
    kept = _select_within_budget(matched, question, score_fn, budget_chars)
    return (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ADDITIONAL CONTEXT (topic-matched for this question)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        + "\n\n".join(kept)
    )


def build_dynamic_prompt_additions(question: str, score_fn=None) -> str:
    """
    Single integration point: combines build_worked_examples (Tier-2 few-shots)
    and build_conditional_blocks (registry semantic facts) into one string to
    append to SYSTEM_PROMPT. Call this from orchestrator.py instead of either
    function individually, so future conditional-injection categories only
    need to be added here.

    `score_fn(question, title) -> float`, if supplied, ranks and budget-caps
    same-question multi-block matches (see _select_within_budget) instead of
    concatenating every regex match unconditionally — without it, behavior is
    unchanged from before this parameter existed.
    """
    return (
        build_conditional_blocks(question, score_fn=score_fn)
        + build_worked_examples(question, score_fn=score_fn)
    )

    #123