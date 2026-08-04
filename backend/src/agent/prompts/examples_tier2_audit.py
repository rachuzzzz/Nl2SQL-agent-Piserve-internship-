"""
Tier-2 worked examples for the audit domain — same shape as
agent/prompts/examples_tier2.py ({"title", "question_pattern", "text"}),
matched against a question and budget-capped by builder_audit.py.
"""

EXAMPLES_TIER2_AUDIT = [
    {
        "title": "Facility/project from a report or CA — multi-hop, no direct FK",
        "question_pattern": (
            r"(?i)(?:audit\s+)?(?:report|corrective\s+action)s?\b.{0,40}\b(facility|site|project)\b"
            r"|\b(facility|site|project)\b.{0,40}(?:audit\s+)?(?:report|corrective\s+action)"
        ),
        "text": """\
─── Facility/project from a report or corrective action ───
User: "show audit reports with their facility"
Turn 1: {"thought": "audit_report has NO facility_id/project_id of its own (confirmed live) — two-hop via audit_schedule, unlike inspection_report where facility_id is direct.", "tool": "generate_sql", "args": {"question": "audit reports with facility name", "schema_hint": "SELECT arep.audit_id, arep.status, fac.name AS facility_name FROM audit_report arep JOIN audit_schedule asch ON arep.schedule_id = asch.id LEFT JOIN facility fac ON asch.facility_id = fac.id WHERE arep.active = true AND asch.active = true ORDER BY arep.submitted_on DESC LIMIT 100;"}}

User: "corrective actions with their facility"
Turn 1: {"thought": "From a corrective action, facility is THREE hops: audit_id -> audit_report -> schedule_id -> audit_schedule -> facility_id.", "tool": "generate_sql", "args": {"question": "corrective actions with facility name", "schema_hint": "SELECT aca.corrective_action_id, fac.name AS facility_name FROM audit_corrective_action aca JOIN audit_report arep ON aca.audit_id = arep.id JOIN audit_schedule asch ON arep.schedule_id = asch.id LEFT JOIN facility fac ON asch.facility_id = fac.id WHERE aca.active = true AND arep.active = true AND asch.active = true LIMIT 100;"}}
""",
    },
    {
        "title": "Audit type resolution — three-hop trap via audit_plan",
        "question_pattern": r"(?i)\baudit\s+type\b|\b(internal|external|statutory|third.?party|second\s+party|first\s+party)\s+audit",
        "text": """\
─── Audit type — three-hop trap, audit_report.audit_type_id is DEAD ───
User: "what audit type was this report"
Turn 1: {"thought": "audit_report.audit_type_id is always NULL (confirmed live). Real path: schedule_id -> audit_schedule -> audit_plan_id -> audit_plan -> audit_type_id -> audit_type.", "tool": "generate_sql", "args": {"question": "audit report audit type via audit plan", "schema_hint": "SELECT arep.audit_id, aty.name AS audit_type_name FROM audit_report arep JOIN audit_schedule asch ON arep.schedule_id = asch.id JOIN audit_plan ap ON asch.audit_plan_id = ap.id JOIN audit_type aty ON ap.audit_type_id = aty.id WHERE arep.active = true AND asch.active = true AND ap.active = true LIMIT 100;"}}
NEVER: WHERE arep.audit_type_id = ... — always NULL, silently returns 0 rows.
""",
    },
    {
        # Mirrors inspection's "Inspector performance ranking" worked example
        # (examples_tier2.py) — same facility/site-quality-vs-person-skill bias
        # applies to audit_score. Added 2026-07-29 after an observed inconsistency:
        # "who is the worst inspector" correctly triggered a clarifying question
        # (via the inspection-domain example), but "who is the worst auditor" had
        # no equivalent guard and just silently picked "most overdue corrective
        # actions" as its own interpretation without asking — one of several
        # equally plausible metrics (score, overdue CAs, high-risk findings), not
        # something the audit domain should guess on its own.
        "title": "Auditor performance ranking — ambiguous 'best/worst', don't misuse audit_score",
        "question_pattern": (
            r"(?i)\b(best|top|worst|poorest|highest|lowest)\b.{0,30}\b(auditor|auditors)\b"
            r"|\b(auditor|auditors)\b.{0,30}\b(perform\w*|rank\w*|effective\w*|produc\w*)\b"
            r"|\bwhich\s+auditor\b.{0,30}\b(best|worst|most|highest|top)\b"
            r"|\bauditor\b.{0,30}\b(leaderboard|ranking|comparison)\b"
        ),
        "text": """\
─── Auditor performance ranking ───
CRITICAL: DO NOT use AVG(arep.audit_score) to rank auditors.
audit_score measures the AUDITED FACILITY/ENTITY's compliance, not the auditor's own
skill — an auditor assigned only to well-run facilities always ranks first (facility bias),
same trap as inspection_score for inspectors.

"Best/worst auditor" is AMBIGUOUS. The correct behaviour is:
  • If question says "best/worst by volume / most audits" → rank by COUNT(DISTINCT asch.id) conducted
  • If question says "most overdue corrective actions / most findings" → rank by overdue CA count
  • If question says "average audit score" or explicitly mentions audit_score → ok to use
    AVG(arep.audit_score) but label it avg_audit_score and add audit_count alongside
  • If "best"/"worst" is vague → ask for clarification (do not silently pick one metric)

User: "who is the worst auditor"
Turn 1: {"thought": "'Worst' is ambiguous for an auditor — volume, overdue CAs found, high-risk findings, or audit_score (which is facility quality, not auditor skill)? I must NOT guess a metric silently — ask for clarification, same as the inspector-performance rule.", "tool": "final_answer", "args": {"answer": "Could you clarify what you mean by 'worst auditor'?\\n\\n• Fewest audits conducted\\n• Most overdue corrective actions found\\n• Most high-risk findings\\n• Lowest average audit score (note: this reflects the audited facility's compliance, not the auditor's own skill)\\n\\nLet me know and I'll pull the right data."}}

User follow-up: "most overdue corrective actions" (user answered the clarification — EXECUTE NOW, do NOT ask again)
Turn 1: {"thought": "User clarified: most overdue CAs found. Previous turn was my clarification question — proceed immediately, do not ask again.", "tool": "generate_sql", "args": {"question": "auditor with the most overdue corrective actions found", "schema_hint": "SELECT u.first_name || ' ' || u.last_name AS auditor_name, COUNT(*) AS overdue_ca_count FROM audit_corrective_action aca JOIN audit_report arep ON aca.audit_id = arep.id JOIN audit_schedule asch ON arep.schedule_id = asch.id JOIN users u ON asch.auditor_id = u.id WHERE aca.status = 'OVERDUE' AND aca.active = true AND arep.active = true AND asch.active = true AND aca.adequacy_status = 'APPROVE' GROUP BY u.id, u.first_name, u.last_name ORDER BY overdue_ca_count DESC LIMIT 10;"}}

User: "which auditor has the lowest average audit score?" (explicitly about audit_score)
Turn 1: {"thought": "User explicitly asked for average audit_score per auditor — this is the avg FACILITY/audit score for what they audited, not auditor quality. I'll label it clearly.", "tool": "generate_sql", "args": {"question": "auditor with lowest average audit score", "schema_hint": "SELECT u.first_name || ' ' || u.last_name AS auditor_name, ROUND(AVG(arep.audit_score)::numeric, 2) AS avg_audit_score, COUNT(*) AS audit_count FROM audit_report arep JOIN audit_schedule asch ON arep.schedule_id = asch.id JOIN users u ON asch.auditor_id = u.id WHERE arep.active = true AND asch.active = true AND arep.audit_score IS NOT NULL GROUP BY u.id, u.first_name, u.last_name ORDER BY avg_audit_score ASC LIMIT 1;"}}
""",
    },
    {
        "title": "Auditor/auditee — assignment column, not role table",
        "question_pattern": r"(?i)\bwho\s+(is|are)\s+the\s+audit(?:or|ee)s?\b|\blist\s+(all\s+)?audit(?:or|ee)s?\b",
        "text": """\
─── Who is the auditor/auditee ───
User: "who is the auditor for this schedule"
Turn 1: {"thought": "Never filter role.name ILIKE '%auditor%' to find who does audit work — several assigned auditors hold unrelated roles (HSE Officer, Internal Inspector, Super Admin, FM Manager confirmed live). Query the assignment column directly.", "tool": "generate_sql", "args": {"question": "auditor assigned to this schedule", "schema_hint": "SELECT u.first_name || ' ' || u.last_name AS auditor_name FROM audit_schedule asch JOIN users u ON asch.auditor_id = u.id WHERE asch.active = true LIMIT 100;"}}
""",
    },
    {
        "title": "Pending approval — polymorphic review table",
        "question_pattern": (
            r"(?i)\bpending\s+(approval|review)\b|\bawaiting\s+(approval|review)\b"
            r"|\bwho\s+approved\b|\badequacy\b|\bimplementation\s+(approval|review)\b"
        ),
        "text": """\
─── Pending approval — the polymorphic 'review' table ───
User: "what audit plans are pending approval"
Turn 1: {"thought": "review.entity_id is polymorphic — always filter WHERE entity_type = 'AUDIT_PLAN' AND entity_id = ap.id. Reviewer role is dynamic/permission-driven, never hardcoded as a correctness gate.", "tool": "generate_sql", "args": {"question": "audit plans with pending review", "schema_hint": "SELECT ap.title, rev.level, rev.status FROM review rev JOIN audit_plan ap ON rev.entity_id = ap.id AND rev.entity_type = 'AUDIT_PLAN' WHERE rev.status = 'PENDING' AND ap.active = true ORDER BY ap.created_on ASC LIMIT 100;"}}

User: "which corrective actions are pending implementation sign-off from the auditor"
Turn 1: {"thought": "IMPLEMENTATION review_type is two-level: level 1 = auditee-side, level 2 = auditor-side final sign-off. 'from the auditor' means level 2.", "tool": "generate_sql", "args": {"question": "corrective actions pending implementation review level 2", "schema_hint": "SELECT aca.corrective_action_id FROM audit_corrective_action aca JOIN review rev ON rev.entity_id = aca.id AND rev.entity_type = 'AUDIT_CORRECTIVE_ACTION' AND rev.review_type = 'IMPLEMENTATION' WHERE rev.level = 2 AND rev.status = 'PENDING' AND aca.active = true LIMIT 100;"}}

User: "what is pending adequacy review"
Turn 1: {"thought": "ADEQUACY review_type on AUDIT_CORRECTIVE_ACTION is the auditor-side check on the proposed fix, level 1.", "tool": "generate_sql", "args": {"question": "corrective actions pending adequacy review", "schema_hint": "SELECT aca.corrective_action_id FROM audit_corrective_action aca JOIN review rev ON rev.entity_id = aca.id AND rev.entity_type = 'AUDIT_CORRECTIVE_ACTION' AND rev.review_type = 'ADEQUACY' WHERE rev.status = 'PENDING' AND aca.active = true LIMIT 100;"}}
""",
    },
    {
        "title": "impact vs compliance vs risk_level — three separate axes",
        "question_pattern": r"(?i)\bimpact\b|\bcompliance\b|\brisk\s*level\b",
        "text": """\
─── impact / compliance / risk_level — pick the table from intent, not label text ───
User: "show corrective actions with high impact"
Turn 1: {"thought": "'impact' = business dimension affected, use impact_id. impact and compliance share some identical label strings with DIFFERENT UUIDs per table — never match a word against 'whichever table has it', pick from question intent.", "tool": "generate_sql", "args": {"question": "corrective actions ranked by impact scale", "schema_hint": "SELECT aca.corrective_action_id, imp.name AS impact_name, imp.scale FROM audit_corrective_action aca JOIN impact imp ON aca.impact_id = imp.id WHERE aca.active = true ORDER BY imp.scale DESC LIMIT 100;"}}
""",
    },
    {
        "title": "Who proposed/raised a CA — created_by, not responsible or pending_with",
        "question_pattern": (
            r"(?i)\bwho\s+(proposed|raised|raised\s+it|logged|created|filed|reported)\b.{0,30}"
            r"\bcorrective\s+action"
            r"|\bcorrective\s+action.{0,30}\bwho\s+(proposed|raised|logged|created|filed)\b"
        ),
        "text": """\
─── Who proposed/raised a corrective action — created_by, NOT responsible ───
User: "who proposed the corrective action"
Turn 1: {"thought": "'who proposed/raised/logged' means the person who created the CA record — that's aca.created_by (UUID FK to users, populated on every row), NOT aca.responsible (a party/organization enum: KHIDMAH/KHIDMAH_AND_CLIENT/CLIENT/SERVICE_PROVIDER_CONTRACTOR/CONSULTANT — a category, not a person) and NOT aca.pending_with (free text, mostly NULL, unreliable).", "tool": "generate_sql", "args": {"question": "who created (raised) this corrective action", "schema_hint": "SELECT aca.corrective_action_id, u.first_name || ' ' || u.last_name AS raised_by FROM audit_corrective_action aca JOIN users u ON aca.created_by = u.id WHERE aca.active = true AND u.active = true LIMIT 100;"}}
NEVER: SELECT aca.responsible AS proposer — that returns a party/organization label
(e.g. 'KHIDMAH_AND_CLIENT'), not a person's name.
""",
    },
    {
        "title": "Dead portfolio junction tables — never use the audit_*_portfolio family",
        "question_pattern": r"(?i)portfolio\b.{0,30}\b(categor|type|subcategor)|\b(categor|type|subcategor)\w*\b.{0,30}portfolio",
        "text": """\
─── Portfolio category/type coverage — dead junction trap ───
User: "what categories does this auditor portfolio cover"
Turn 1: {"thought": "audit_category_portfolio is always 0 rows (dead, confirmed live) despite the name reading as the obvious table. Real path is portfolio_detail_audit_category via auditor_portfolio_detail.", "tool": "generate_sql", "args": {"question": "portfolio category coverage via portfolio_detail_audit_category", "schema_hint": "SELECT acat.name AS category_name FROM auditor_portfolio_detail pod JOIN portfolio_detail_audit_category pdac ON pdac.detail_id = pod.id JOIN audit_category acat ON pdac.audit_category_id = acat.id WHERE pod.active = true LIMIT 100;"}}
NEVER: JOIN audit_category_portfolio — always 0 rows.
""",
    },
    {
        # Added 2026-07-31 after a live UI session dead-end (confirmed by user):
        # "how many audit plans are under this cycle" (COUNT(*) only, no row ids)
        # -> "show details" -> "of the above", BOTH turns asked the user to
        # clarify instead of resolving. Root cause: the FOLLOW-UP QUERY RULES
        # mechanism only reuses a resolved ENTITY id from ACTIVE CONTEXT — a
        # COUNT-only turn never stores one (there IS no single row to point
        # at), so a terse "show details" had literally nothing to attach to
        # and the model defaulted to asking. The fix isn't an id lookup, it's
        # re-deriving the listing from the COUNT query's own WHERE/JOIN
        # scope — the filter that made it a valid count is exactly the filter
        # that makes a valid listing.
        "title": "'Show details' / 'of the above' after a COUNT-only turn — expand the same filter, don't ask",
        "question_pattern": (
            r"(?i)\bshow\s+(me\s+)?(the\s+)?details?\b"
            r"|\bof\s+the\s+above\b"
            r"|^\s*details?\s*$"
            r"|^\s*list\s+(them|those|it)\s*$"
        ),
        "text": """\
─── 'Show details' / 'of the above' — resolve, don't reflexively clarify ───
RULE: A terse follow-up like "show details" or "of the above" is NEVER itself
ambiguous about WHAT to show — it always means "expand whatever the last turn
was about into full rows." What's ambiguous (if anything) is only WHICH prior
turn to expand. Resolve in this order, and only ask for clarification if
BOTH of the following fail:
  1. If an earlier turn in this conversation already returned real row-level
     data (a listing, not just a scalar), expand/detail THAT turn's rows —
     prefer it over a more recent scalar COUNT/aggregate turn, since a COUNT
     has no rows to show details OF.
  2. Otherwise, if the most recent turn was a COUNT/aggregate with a clear
     scoping filter (a named cycle, plan, facility, date range, status,
     etc.), re-run the SAME underlying entity + SAME WHERE/JOIN scope as
     that count, but SELECT full columns instead of COUNT(*). The filter
     that made the count meaningful is exactly the filter that makes the
     listing meaningful — do not discard it and do not ask the user to
     restate it.
Only ask for clarification if NEITHER a prior listing NOR an unambiguous
count-scope exists to expand.

User: "how many audit plans are under this cycle" (cycle already resolved earlier this conversation)
Turn 1: {"thought": "Scalar count, scoped to one resolved cycle_id.", "tool": "generate_sql", "args": {"question": "count of audit plans under this cycle", "schema_hint": "SELECT COUNT(*) AS total_plans FROM audit_plan ap WHERE ap.cycle_id = '<resolved_cycle_id>' AND ap.active = true;"}}

User: "show details"
Turn 1: {"thought": "No row-level list exists yet in this conversation to point at — the only prior turn was a COUNT. Per the RULE, re-run the SAME entity + SAME WHERE scope (cycle_id) as that count, this time selecting full rows instead of COUNT(*). This is NOT ambiguous — 'details' of the thing I just counted means the rows behind the count. Do not ask for clarification.", "tool": "generate_sql", "args": {"question": "list audit plans under this cycle with full details", "schema_hint": "SELECT ap.title AS audit_plan_title, ap.status AS audit_plan_status, ap.audit_type_id, ap.category_id, ap.cycle_id FROM audit_plan ap WHERE ap.cycle_id = '<resolved_cycle_id>' AND ap.active = true ORDER BY ap.created_on DESC LIMIT 100;"}}

Contrast — when a REAL listing already happened earlier, prefer it over a later scalar count:
User turn A: "show all corrective actions and who approved their implementation statuses" -> returns 15 row-level CAs.
User turn B: "how many of the above are closed" -> scalar COUNT(*) reusing turn A's filter, no rows.
User turn C: "show details"
Turn 1: {"thought": "Most recent turn (B) was a scalar COUNT with no rows of its own. Per the RULE, prefer the earlier turn (A) that has real row-level data — 'details' means the 15 CAs from turn A, not a re-expansion of turn B's count. Reuse turn A's filters/columns.", "tool": "generate_sql", "args": {"question": "corrective actions and who approved their implementation statuses, in detail", "schema_hint": "<same shape as turn A's query>"}}
""",
    },
]
