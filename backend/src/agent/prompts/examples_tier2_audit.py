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
        "title": "Dead portfolio junction tables — never use the audit_*_portfolio family",
        "question_pattern": r"(?i)portfolio\b.{0,30}\b(categor|type|subcategor)|\b(categor|type|subcategor)\w*\b.{0,30}portfolio",
        "text": """\
─── Portfolio category/type coverage — dead junction trap ───
User: "what categories does this auditor portfolio cover"
Turn 1: {"thought": "audit_category_portfolio is always 0 rows (dead, confirmed live) despite the name reading as the obvious table. Real path is portfolio_detail_audit_category via auditor_portfolio_detail.", "tool": "generate_sql", "args": {"question": "portfolio category coverage via portfolio_detail_audit_category", "schema_hint": "SELECT acat.name AS category_name FROM auditor_portfolio_detail pod JOIN portfolio_detail_audit_category pdac ON pdac.detail_id = pod.id JOIN audit_category acat ON pdac.audit_category_id = acat.id WHERE pod.active = true LIMIT 100;"}}
NEVER: JOIN audit_category_portfolio — always 0 rows.
""",
    },
]
