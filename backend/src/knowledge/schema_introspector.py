"""
Schema Introspector
====================
Introspects the database at startup and builds the schema text that gets
injected into prompts.

Covers two domains:
  - INSPECTION WORKFLOW tables (inspection_report, inspection_corrective_action, etc.)
  - AI FLAT TABLES (ai_question, ai_answer)

The fb_* form-builder tables are intentionally excluded — the ai_* tables
are the public interface for all form question/answer data.

AUDIT domain constants (AUDIT_*) live in this same file — the DatabaseSchema
class and introspect_schema() are parameterized by these constant sets (see
"DatabaseSchema is a dependency-injected dataclass" below) rather than
duplicated, since the introspection MECHANISM is identical between domains;
only the table lists / business rules / relationships text differ. Content
verified live against the DB (not inferred) — see docs/Audit_Domain_Schema_Reference.docx
and the domain-router plan's §1b for the review/review_level approval-workflow
findings this was built from.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine
from src.knowledge.question_patterns import (
    USER_ROLES_TOPIC, APPROVAL_WORKFLOW_TOPIC, COMMUNICATION_GROUPS_TOPIC,
)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

ENUM_COLUMNS = {
    "inspection_corrective_action.responsible": [
        "CLIENT", "INTERNAL_OPERATIONS", "SUB_CONTRACTOR",
    ],
    "inspection_corrective_action.status": [
        "OPEN", "CLOSED", "OVERDUE", "CLOSE_WITH_DEFERRED",
    ],
    "inspection_corrective_action.expenditure": [
        "NO_EXPENDITURE", "CAPEX", "OPEX", "CAPEX_AND_OPEX",
    ],  # STRING ENUM — not a numeric amount; capex/opex columns hold the numeric values
    "inspection_corrective_action.capex_status": [
        "ACCEPTED", "REJECTED", "DEFERRED",
    ],  # NULL = awaiting approval
    "inspection_corrective_action.pending_with": [],
    "inspection_report.status": [
        "DRAFT", "SUBMITTED", "CLOSED", "UNDER_REVIEW", "RETURN_FOR_MODIFICATION",
    ],
    "inspection_schedule.status": [
        "PENDING", "ONGOING", "COMPLETED", "OVERDUE", "CANCELLED",
    ],
    "inspection_cycle.status": [],
    "inspector_portfolio.status": [
        "APPROVED", "PENDING", "DRAFT", "MODIFICATION",
    ],  # DRAFT = not yet submitted; PENDING = awaiting review; MODIFICATION = sent back
}

# users.id (UUID PK) vs users.id_fk (integer Drupal UID) is a recurring source of
# "operator does not exist: uuid = integer" failures — the model repeatedly guesses
# `u.id = <int>` for "my role" / "Drupal UID <n>" style questions. Surfaced directly
# on the users lookup-table line (both _fmt_lookup and for_sql_prompt_slim) since
# that's the block actually reaching the SQL-generation model by default.
_USERS_ID_NOTE = (
    "id_fk is the integer Drupal UID (e.g. 11410) — id is the UUID PK. "
    "Filtering by a raw numeric user ID → WHERE id_fk = <int>, NEVER WHERE id = <int>."
)


def _split_relationship_sections(text: str) -> list[tuple[str, str]]:
    """
    Phase 3.12: split a TABLE_RELATIONSHIPS-shaped constant into
    (title, body) sections on its ALL-CAPS, column-0 heading lines (e.g.
    "WHERE CONCEPTS LIVE:", "HOW TABLES CONNECT (...):") — every other line
    in these constants is indented. Read-only text analysis for
    DatabaseSchema.profile_fixed_overhead_relevance's relevance-by-section
    measurement; never used to build a prompt, so approximate section
    boundaries (a multi-line heading collapses to one space-joined title)
    are fine — confirmed live: covers every line of TABLE_RELATIONSHIPS
    with none dropped or duplicated.
    """
    sections: list[tuple[str, str]] = []
    title_lines: list[str] = []
    body_lines: list[str] = []
    mode = "title"
    for line in text.split("\n"):
        is_heading = bool(line) and not line[0].isspace()
        if is_heading:
            if mode == "body":
                sections.append((" ".join(title_lines).strip(), "\n".join(body_lines)))
                title_lines, body_lines = [], []
                mode = "title"
            title_lines.append(line)
        else:
            body_lines.append(line)
            mode = "body"
    if title_lines or body_lines:
        sections.append((" ".join(title_lines).strip(), "\n".join(body_lines)))
    return sections


BUSINESS_RULES = [
    # NOTE: the observation↔MA cast direction below must match
    # core.semantics_registry.SEMANTICS["joins"]["observation_to_ma"]["sql"]
    # — "aa.submission_id = ica.observation_id::varchar"
    "DRAFT inspections usually have NULL scores. Always add WHERE status != 'DRAFT' "
    "when querying inspection_score, unless the user explicitly asks about drafts.",
    "NEVER use SELECT *. Always specify the columns the user needs.",
    "NEVER use SQL reserved words as table aliases (is, as, in, on, by, do, if).",
    "Use ILIKE for text matching, never LIKE.",
    "corrective_action_id is a human-readable ID like '2026/01/ST064/INS001_MA001', not a UUID.",
    "inspection_corrective_action.inspection_id is the PRIMARY FK to inspection_report.id.",
    # NOTE: agent/prompts/templates/system_prompt.txt (~line 300-330, the
    # "OBSERVATION -> MITIGATIVE ACTION" worked-example section) is an
    # independently hand-written restatement of this same PATH 1/2/3 rule
    # (different example values/phrasing, not generated from this source) —
    # currently in sync but with no shared source or consistency-checker
    # coverage between them. If you edit the rule here, check that file too.
    "OBSERVATION → MITIGATIVE ACTION — two distinct paths, NEVER confuse them: "
    "PATH 1 — MAs for a SPECIFIC observation: JOIN ai_answers aa ON aa.submission_id = ica.observation_id::varchar "
    "(cast the UUID side to varchar — NEVER aa.submission_id::uuid = ica.observation_id, NEVER ::text). "
    "PATH 2 — ALL MAs for an inspection: JOIN ica ON ica.inspection_id = ir.id. "
    "Use PATH 1 only when the user references a SPECIFIC observation; "
    "use PATH 2 for 'mitigative actions for this inspection' generally. "
    "PATH 3 — the REVERSE of PATH 1 (singular): the observation a SPECIFIC MA was raised "
    "against ('what was the observation for this MA', 'observation text for MA <id>'): "
    "JOIN ai_answers aa ON aa.submission_id = ica.observation_id::varchar (same cast rule as "
    "PATH 1), then filter WHERE aq.label = 'Observation' after joining ai_questions aq ON "
    "aa.element_id = aq.element_id — inspection's free-text finding field is literally "
    "labeled 'Observation' (unlike audit, which uses 'Findings / Remarks'). NEVER use "
    "get_answers(inspection_id=...) for this — it returns the whole form, not the one "
    "observation. "
    "PATH 4 — ALL answers for the whole inspection report (reconstructing the full "
    "filled-out form): JOIN ai_answers aa ON aa.inspection_report_id = ir.id, then GROUP BY "
    "aa.submission_id since each distinct submission_id is one repeatable entry (e.g. one "
    "observation) within the form. MUST also filter aa.module_name = 'Inspection Form' — "
    "confirmed live 2026-08-10 that ai_answers rows for one inspection_report_id span "
    "multiple modules (MA-workflow submodules like Client Details Entry/PR Entry/Completion/ "
    "Report Approval-Rejection share the same inspection_report_id) — omitting this pulls in "
    "workflow noise, not the form. Use PATH 4 for 'reconstruct this inspection's form' / "
    "'all answers for this inspection' questions; NEVER get_answers for this (flat/"
    "LIMIT=50/ungrouped — loses the submission_id grouping and can silently truncate a form).",
    "RISK LEVEL / IMPACT — OBSERVATION vs MITIGATIVE ACTION (confirmed live 2026-08-10, "
    "user-supplied business fact, extended same day to Impact): BOTH are FORM ANSWERS — Risk "
    "Level is auto-derived from Impact/etc. entered on the SAME submission as the observation "
    "text, and Impact is itself a directly-entered multi-select form answer in that identical "
    "submission group. Both live in ai_answers, in the SAME submission_id group as the "
    "observation, NOT on inspection_corrective_action (which has NO compliance_id at all — "
    "that axis is audit-only). 'high risk observations', 'risk/impact mappings for these "
    "observations', 'findings that are high risk' always mean aa.answer_text WHERE aq.label "
    "IN ('Risk Level','Impact'), matched to their sibling Observation answer via submission_id "
    "(NEVER via inspection_report_id/ir.id — that pairs answers from ANY entry in the report, "
    "not necessarily the SAME one; two prior seed examples in this codebase made exactly that "
    "mistake for Risk Level before this fix, and audit's mirror confirmed live the same day "
    "that Impact needs the identical treatment — a follow-up asking for 'risk mappings' fell "
    "back to the CA's own risk_level_id/impact_id instead). ica.risk_level_id/impact_id are "
    "SEPARATE, later values that only exist once a MA is actually raised against the "
    "observation — use them ONLY for questions specifically about the mitigative ACTION's own "
    "classification ('high risk mitigative actions', 'impact of this MA'), never as a stand-in "
    "for the observation's own form answers — using them for a question about OBSERVATIONS "
    "silently excludes every observation that never had a MA raised.",
    "Use LEFT JOIN (not INNER JOIN) for optional lookups like inspection_sub_type, "
    "entity, and inspection_report_remark.",
    "VOCABULARY — mitigative action status: "
    "'open' = 'OPEN'. 'overdue' = 'OVERDUE'. "
    "'closed'/'resolved' = status IN ('CLOSED','CLOSE_WITH_DEFERRED').",
    "VOCABULARY — inspection report status: "
    "'completed'/'done' = 'CLOSED'. 'pending review' = 'UNDER_REVIEW'. "
    "'returned'/'rejected' = 'RETURN_FOR_MODIFICATION'.",
    "For form answers use ai_answers (plural). For question labels use ai_questions (plural). "
    "JOIN via: ai_answers.element_id = ai_questions.element_id.",
    "JOIN ai_answers to inspection_report via: ai_answers.inspection_report_id = inspection_report.id.",
    "CRITICAL — inspection_report has TWO id columns: "
    "'id' is the UUID primary key — NEVER select this for display (outputs UUID garbage). "
    "'inspection_id' is the human-readable varchar like '2026/04/ST001/INS001' — ALWAYS use "
    "ir.inspection_id (not ir.id) when showing the inspection identifier to the user.",
    "USER ROLES — users have multiple roles via user_roles(user_id, role_id) → role(id, name). "
    "To filter by role use EXISTS not JOIN (JOIN fans out rows): "
    "WHERE EXISTS (SELECT 1 FROM user_roles ur JOIN role r ON ur.role_id = r.id "
    "WHERE ur.user_id = u.id AND r.name = 'FM Manager'). "
    "FM Manager role names: fm_manager, fm_engineer, ad_fm, asst_fm_manager, fm_admin, fm_supervisor. "
    "Inspector role names (verified 2026-07-08 via SELECT name, COUNT(*) FROM role JOIN user_roles): "
    "'Internal Inspector' (101 users) and 'External Inspector' (2 users) — Title Case with spaces, "
    "matching the FM-side convention. 'Internal Auditor' (122) and 'External Auditor' (14) also exist "
    "but belong to the separate Audit domain, not this app — do NOT match them for inspection-domain "
    "role filters (e.g. avoid ILIKE '%inspector%' or '%audit%'; use an explicit IN list instead).",
    "APPROVAL WORKFLOW — 'review' is a shared, POLYMORPHIC multi-level approval log (also used by "
    "the separate Audit domain for AUDIT_PLAN/AUDIT_CORRECTIVE_ACTION/SCHEDULE/PORTFOLIO — this app "
    "only ever queries the inspection-side entity_type values). review.entity_id has NO real FK — "
    "always filter WHERE entity_type = '<TYPE>' AND entity_id = <target>.id. entity_type values for "
    "inspection: 'INSPECTION_REPORT' (2 levels, both actively used — roughly even split between "
    "level 1 and level 2, confirmed live), 'INSPECTION_PORTFOLIO' (1 level only), "
    "'INSPECTION_SCHEDULE' (1 level only). review_type is NULL for all inspection entity_types "
    "(unlike audit's AUDIT_CORRECTIVE_ACTION, inspection has no ADEQUACY/IMPLEMENTATION split).",
    "'COMPLETED' ON review IS NOT 'APPROVED' — confirmed live (same table, same trap as audit): a "
    "COMPLETED review can have action = 'RETURN_FOR_MODIFICATION' (rejected/sent back) instead of "
    "'APPROVE'. 'who approved this inspection report/portfolio/schedule' MUST filter "
    "WHERE status = 'COMPLETED' AND action = 'APPROVE' — status = 'COMPLETED' alone silently "
    "includes rejected reviews. action is NULL while status = 'PENDING'.",
    "review_level IS NOT RELIABLE for 'how many levels of approval' — it's a tiny 4-row config table "
    "with NO row for INSPECTION_REPORT (despite INSPECTION_REPORT genuinely having 2 levels in real "
    "review data) and only a stale-looking row for INSPECTION_PORTFOLIO. NEVER read review_level to "
    "answer a levels question — compute it from review itself: SELECT entity_type, "
    "COUNT(DISTINCT level) or MAX(level) FROM review WHERE entity_type = '<TYPE>' GROUP BY entity_type.",
]

TABLE_RELATIONSHIPS = """\
WHERE CONCEPTS LIVE:
  - Inspection TYPE names → inspection_type.name
  - Inspector names → users.first_name + users.last_name (JOIN via inspector_user_id)
  - Facility names → facility.name (JOIN via facility_id)
  - Project names → project.name (JOIN via project_id)
  - Client names → client.name (JOIN via client_id)
  - Entity names       → entity.name        (JOIN via inspection_report.entity_id — DIRECT)
  - Organisation names → organisation.name (JOIN via facility.organization_id — ⚠ confirmed always
                          0 rows live; use the communication-group junction tables instead, see below)
  - Country names      → country.name      (JOIN via facility.country_id — inspection_report has
                          NO country_id column, despite what older comments in this file used to say)
  - Region names       → region.name       (JOIN via facility.region_id → region)
  - Scores, status, dates → inspection_report
  - Causes, costs, overdue → inspection_corrective_action
  - Form question labels → ai_questions.label  (NOT question_label)
  - Form text answers → ai_answers.answer_text
  - Form numeric answers → ai_answers.answer_numeric
  - Form scores → ai_answers.score

HOW TABLES CONNECT (inspection_report is the hub):
  inspection_report.inspection_type_id  → inspection_type.id
  inspection_report.facility_id         → facility.id
  inspection_report.project_id          → project.id
  inspection_report.client_id           → client.id
  inspection_report.entity_id   → entity.id    (DIRECT)
  inspection_report → facility (facility_id) → organisation (organization_id)  ⚠ DEAD — see below
  inspection_report → facility (facility_id) → region (region_id) → country (country_id)
  NOTE: inspection_report has NO direct organisation_id, region_id, OR country_id — join via facility
  facility.organization_id → organisation.id   ← American spelling on FK; British on table name.
    ⚠ CONFIRMED DEAD 2026-07-08: facility.organization_id is NULL on all 175 rows in production;
    this join always returns 0 rows. organisation IS reachable, just not from facility — use
    inspection_communication_group_organisations.organisation_id or organisation_countries.organisation_id
    instead (see COMMUNICATION GROUPS below). In current data both of those paths also only ever
    point at a single organisation, so "break down by organisation" queries won't be meaningful yet.
    This is a data snapshot, not a DB-enforced constraint — if it ever needs re-checking, run
    SELECT COUNT(*) FROM facility WHERE organization_id IS NOT NULL again.
  facility.region_id       → region.id
  region.country_id        → country.id        ← region belongs to a country
  inspection_report.inspector_user_id   → users.id
  inspection_report.inspectee_user_id   → users.id  (FM Manager — facility manager being inspected)
  inspection_report.cycle_id            → inspection_cycle.id
  inspection_report.schedule_id         → inspection_schedule.id
  inspection_corrective_action.inspection_id → inspection_report.id
  ai_answers.inspection_report_id       → inspection_report.id
  ai_answers.element_id                 → ai_questions.element_id

  facility.country_id      → country.id
  # facility.client_id does NOT exist — reach client via inspection_report.client_id
  # facility.organization_id → organisation.id — DEAD, see the ⚠ note above (always NULL)

  project.organisation_id  → organisation.id

TABLE COLUMNS (lookup tables):
  country:      id (uuid), active (bool), name (varchar)
  organisation: id (uuid), active (bool), name (varchar), logo (varchar)
  entity:       id (uuid), active (bool), name (varchar), logo (varchar)
               — standalone business/contract entity, inspection_report.entity_id → entity.id
  region:       id (uuid), active (bool), name (varchar), country_id (uuid→country.id)
  project:      id (uuid), active (bool), name (varchar), organisation_id (uuid→organisation.id)
  client:       id (uuid), active (bool), name (varchar)

  SOFT DELETE — include WHERE active = true for: country, region, organisation, project, client, entity

COMMUNICATION GROUPS:
  inspection_communication_group: id (uuid), name (varchar), active (bool)
    — A named group of recipients for inspection notifications

  Junction tables (many-to-many):
    inspection_communication_group_facilities    — group_id (→icg.id), facility_id (→facility.id)
    inspection_communication_group_organisations — group_id (→icg.id), organisation_id (→organisation.id)
    inspection_group_emails                      — group_id (→icg.id), email (varchar)

  organisation_countries: organisation_id (→organisation.id), country_id (→country.id)
    — which countries an organisation operates in

  TYPICAL QUERIES:
  "which communication group does facility X belong to?"
    → JOIN icgf ON icgf.facility_id = fac.id → JOIN icg ON icg.id = icgf.group_id

  "show all facilities in communication group Y"
    → JOIN icgf ON icgf.group_id = icg.id → JOIN facility ON facility.id = icgf.facility_id

  "what emails does communication group X have?"
    → JOIN ige ON ige.group_id = icg.id → SELECT ige.email

  "which organisations are in communication group X?"
    → JOIN icgo ON icgo.group_id = icg.id → JOIN organisation ON organisation.id = icgo.organisation_id

  "which countries does organisation X operate in?"
    → JOIN oc ON oc.organisation_id = org.id → JOIN country ON country.id = oc.country_id

USER ROLES:
  users.id → user_roles.user_id → user_roles.role_id → role.id → role.name
  user_roles(user_id, role_id) — one user can have MULTIPLE roles
  role.name values relevant to inspections:
    FM-side (inspectee):  FM Manager, FM Engineer, AD FM, Asst FM Manager,
                          FM Admin, FM Supervisor, hse_officer
    Inspector-side (verified 2026-07-08, Title Case with spaces like FM-side):
                          Internal Inspector (101 users), External Inspector (2 users).
                          NOTE: 'Internal Auditor' (122) and 'External Auditor' (14) also exist
                          in the role table but belong to the separate, unwired Audit domain —
                          NEVER match them for "inspector" questions in this app.
  CRITICAL — use EXISTS subquery (not JOIN) to avoid row fan-out:
    WHERE EXISTS (
        SELECT 1 FROM user_roles ur JOIN role r ON ur.role_id = r.id
        WHERE ur.user_id = ir.inspectee_user_id
          AND r.name IN ('FM Manager','FM Engineer','AD FM',
                         'Asst FM Manager','FM Admin','FM Supervisor')
    )

CRITICAL ai_* JOIN PATTERN:
  FROM ai_answers aa
  LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id
  JOIN inspection_report ir ON aa.inspection_report_id = ir.id
  WHERE aq.label ILIKE '%keyword%'

MULTI-HOP EXAMPLES:
  "answers at Al Ghadeer":
    FROM ai_answers aa
    LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id
    JOIN inspection_report ir ON aa.inspection_report_id = ir.id
    JOIN facility fac ON ir.facility_id = fac.id
    WHERE fac.name ILIKE '%Al Ghadeer%'

  "mitigative actions for safety inspections":
    FROM inspection_corrective_action ica
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    JOIN inspection_type it ON ir.inspection_type_id = it.id
    WHERE it.name ILIKE '%safety%'

RULE: TYPE names (Safety, PPE, Hygiene, etc.) live ONLY in inspection_type.name.

APPROVAL WORKFLOW (review table — see BUSINESS_RULES for the two traps: review_level
unreliability and status='COMPLETED' != approved):
  review.entity_id     → POLYMORPHIC, no real FK — filter by entity_type + entity_id
  review.assigned_to_id → users.id
  review.reviewed_by    → users.id
  entity_type values used by this app: 'INSPECTION_REPORT', 'INSPECTION_PORTFOLIO',
    'INSPECTION_SCHEDULE' (entity_id = inspection_report.id / inspector_portfolio.id /
    inspection_schedule.id respectively)
  "who approved inspection X":
    SELECT u.first_name || ' ' || u.last_name AS approved_by
    FROM review rev
    JOIN inspection_report ir ON rev.entity_id = ir.id AND rev.entity_type = 'INSPECTION_REPORT'
    JOIN users u ON rev.reviewed_by = u.id
    WHERE rev.status = 'COMPLETED' AND rev.action = 'APPROVE' AND ir.inspection_id = '<id>'
"""

LOOKUP_TABLES = {
    "users":               {"name_expr": "first_name || ' ' || last_name", "name_col": None},
    "facility":            {"name_expr": None, "name_col": "name"},
    "project":             {"name_expr": None, "name_col": "name"},
    "client":              {"name_expr": None, "name_col": "name"},
    "entity":              {"name_expr": None, "name_col": "name"},
    "organisation":        {"name_expr": None, "name_col": "name"},
    "country":             {"name_expr": None, "name_col": "name"},
    "region":              {"name_expr": None, "name_col": "name"},
    "inspection_type":     {"name_expr": None, "name_col": "name"},
    "inspection_sub_type": {"name_expr": None, "name_col": "name"},
    # Frequency/cadence lookup — defines how often inspections happen in a portfolio
    # columns: id, name, repeat_count (int), repeat_interval (int), repeat_unit (DAY/WEEK/MONTH)
    # e.g. 'Quarterly Twice' = repeat_count=2, repeat_interval=3, repeat_unit=MONTH
    # JOIN via: inspector_portfolio_details.frequency_definition_id → frequency_definition.id
    "frequency_definition": {"name_expr": None, "name_col": "name"},
    # Risk/impact lookup tables
    "risk_level":          {"name_expr": None, "name_col": "name"},
    "impact":              {"name_expr": None, "name_col": "name"},
}

FK_RESOLUTION = {
    "inspector_user_id":     ("users", "u",   "u.first_name || ' ' || u.last_name AS inspector_name"),
    "inspectee_user_id":     ("users", "insp","insp.first_name || ' ' || insp.last_name AS inspectee_name"),
    "facility_id":           ("facility", "fac", "fac.name AS facility_name"),
    "project_id":            ("project",  "proj", "proj.name AS project_name"),
    "client_id":             ("client",   "cl",   "cl.name AS client_name"),
    "entity_id":             ("entity",   "ent",  "ent.name AS entity_name"),
    "inspection_type_id":    ("inspection_type",     "it",  "it.name AS inspection_type_name"),
    "inspection_sub_type_id":("inspection_sub_type", "ist", "ist.name AS inspection_subtype_name"),
    # Frequency/cadence FK — inspector_portfolio_details only
    "frequency_definition_id": (
        "frequency_definition", "fd",
        "fd.name AS frequency_name, fd.repeat_count, fd.repeat_interval, fd.repeat_unit"
    ),
    # Risk and impact FKs — inspection_corrective_action only
    "risk_level_id":         ("risk_level", "rl",  "rl.name AS risk_level_name"),
    "impact_id":             ("impact",     "imp", "imp.name AS impact_name"),
    # Schedule → portfolio_details FK (critical: the column is portfolio_details_id, not portfolio_id)
    "portfolio_details_id":  ("inspector_portfolio_details", "ipd", "ipd.id"),
    # inspection_schedule → inspection_cycle FK
    "inspection_cycle_id":   ("inspection_cycle", "ic", "ic.id"),
    # inspector_portfolio → cycle
    "cycle_id":              ("inspection_cycle", "ic", "ic.id"),
    # review table FKs — shared polymorphic approval log, see BUSINESS_RULES
    "assigned_to_id":        ("users", "u_asg", "u_asg.first_name || ' ' || u_asg.last_name AS assigned_to_name"),
    "reviewed_by":           ("users", "u_rev", "u_rev.first_name || ' ' || u_rev.last_name AS reviewed_by_name"),
}

OPTIONAL_FKS = {
    "inspection_sub_type_id", "entity_id", "close_submission_id",
    "closed_by_user_id", "schedule_id",
}

INSPECTION_TABLES = [
    "inspection_report",
    "inspection_corrective_action",
    "inspection_schedule",
    "inspection_cycle",
    "inspection_type",
    "inspection_sub_type",
    "inspector_portfolio",
    "inspector_portfolio_details",
    "inspectioncaprogress_tracking",
    "inspection_report_remark",
    "accompanying_inspectors",
    "additional_inspectees",
    "po_details",    # CAPEX Converted — JOIN via po.mitigative_action_uuid = ica.id
    "pr_details",    # OPEX Incurred   — JOIN via pr.mitigative_action_uuid = ica.id
    "review",        # shared polymorphic approval log — also used by audit, see BUSINESS_RULES
    "review_level",  # NOT reliable for level counts — see BUSINESS_RULES warning
]

AI_TABLES = [
    "ai_questions",
    "ai_answers",
]

# CRITICAL: inspection_report has TWO id-like columns:
#   id             → UUID primary key  — NEVER select this for display, always UUID garbage
#   inspection_id  → varchar like '2026/04/ST001/INS001' — ALWAYS use this for display
# deepseek commonly selects ir.id when it should select ir.inspection_id. Listing
# inspection_id first in USEFUL_COLUMNS reinforces the correct choice.
USEFUL_COLUMNS = {
    "inspection_report": [
        "inspection_id",    # VARCHAR display id — use this NOT ir.id (UUID)
        "inspection_score", "status",
        "submitted_on", "start_date_time",
        "facility_id", "inspection_type_id", "inspector_user_id", "inspectee_user_id",
        "project_id", "client_id",
        # inspectee_user_id = FM Manager (facility manager receiving the inspection)
        # duration = EPOCH(ir.submitted_on - ir.start_date_time)/3600 — NOT total_inspection_hours
    ],
    "inspection_corrective_action": [
        "corrective_action_id", "cause", "correction", "corrective_action",
        "responsible", "status", "progress_stage", "capex", "opex",
        "capex_status", "opex_status", "expenditure",
        "target_close_out_date", "age", "deferred_on",
        "risk_level_id", "impact_id", "inspection_id",
        "external_stakeholder_email", "external_stakeholder_name", "external_stakeholder_phone",
        "internal_stakeholder_id",
        # internal_stakeholder_id → users.id. Both stakeholder groups are real and in active use
        # (confirmed 2026-07-08: external_* populated on ~23% of rows, internal_stakeholder_id on 58%)
        # capex_status: NULL=awaiting approval, ACCEPTED, REJECTED, DEFERRED
        # expenditure: STRING ENUM — NO_EXPENDITURE | CAPEX | OPEX | CAPEX_AND_OPEX
        #   (NOT a numeric amount — it is a category flag)
        # capex and opex are the NUMERIC amounts (decimal)
        # ica.capex = CAPEX Generated (quoted amount by inspectee)
        # po_details.po_amount = CAPEX Converted (actual PO raised)
        # pr_details.pr_amount = OPEX Incurred (actual PR raised internally)
        # CAPEX Lost = SUM(ica.capex) - COALESCE(SUM(po.po_amount),0) — arithmetic diff
        # DO NOT use capex_status='REJECTED' for loss — use arithmetic difference
        # Join: po_details ON po.mitigative_action_uuid = ica.id
        # Join: pr_details ON pr.mitigative_action_uuid = ica.id
        # close_on is NULL for ALL rows — use modified_on as closure date proxy
        # CAPEX loss form answers (module: Inspection Mitigative Action Completion):
        #   'Is CAPEX Lost' — answer_text = 'true'
        #   'Amount Difference for CPAEX Loss' — answer_numeric (NOTE: typo CPAEX)
        #   'CAPEX Lost To(Company/Entity/Subcontractor)' — answer_text
        #   JOIN via: ica.completion_submission_id = ai_answers.inspection_report_id
    ],
    "inspection_schedule": [
        "id", "active", "status", "due_date", "schedule_date",
        "facility_id", "inspector_id", "inspection_cycle_id", "portfolio_details_id",
        # 'active' IS a real soft-delete column (confirmed 2026-07-08: 21/2209 rows are false) —
        #   ALWAYS filter WHERE active = true unless the question is specifically about
        #   cancelled/removed schedules.
        # ⚠ NO 'frequency_definition_id' column — frequency via portfolio_details_id → ipd.frequency_definition_id
        # ⚠ NO 'inspection_report_id' column — reach the report via inspection_report.schedule_id = isched.id
    ],
    "inspector_portfolio_details": [
        "id", "portfolio_id", "facility_id",
        "frequency_definition_id", "inspection_type_id",
    ],
    "ai_questions": [
        "element_id", "label", "module_name", "module_id", "entity_type",
    ],
    "po_details": [
        "id", "po_amount", "po_date", "po_number", "mitigative_action_uuid",
        # po_amount = CAPEX Converted (actual PO amount client agreed to pay)
        # JOIN: ON po.mitigative_action_uuid = ica.id
    ],
    "pr_details": [
        "id", "pr_amount", "pr_date", "pr_number", "mitigative_action_uuid",
        # pr_amount = OPEX Incurred (actual internal spend via PR)
        # JOIN: ON pr.mitigative_action_uuid = ica.id
    ],
    "ai_answers": [
        "id", "inspection_id", "inspection_report_id", "element_id", "module_name",
        "answer_text", "answer_text_raw", "answer_numeric", "score", "score_type", "max_score",
        "submitted_on", "submission_id", "created_on",
        # id/submission_id/created_on confirmed live 2026-08-10 (information_schema.columns) —
        # previously missing from this list despite being used in the OBSERVATION -> MA join
        # rule (submission_id) elsewhere in this file; created_on/id needed for a deterministic
        # ORDER BY when reconstructing a whole form (see AUDIT_USEFUL_COLUMNS' ai_audit_answers
        # entry, which mirrors this table and already lists them).
    ],
    "review": [
        "entity_id", "entity_type", "review_type", "level", "status", "action",
        "assigned_to_id", "reviewed_by", "reviewed_on", "submission_id",
        # entity_type for inspection: 'INSPECTION_REPORT' (2 levels), 'INSPECTION_PORTFOLIO' (1),
        #   'INSPECTION_SCHEDULE' (1) — see BUSINESS_RULES for the review_level warning and the
        #   status='COMPLETED' != approved trap.
    ],
}

# Columns to suppress entirely from the schema block shown to deepseek.
# These exist in the DB but are either unreliable, blocked, or create noise.
_SUPPRESS_COLUMNS: set[str] = {
    # NOTE: "adequacy_status" was previously suppressed here with the comment
    # "adequacy workflow not in use yet" — misleading stray entry, removed
    # 2026-07-28. adequacy_status doesn't exist on inspection_corrective_action
    # at all (this set is inspection-only), and audit's OWN adequacy workflow
    # (audit_corrective_action.adequacy_status) is very much in active use —
    # see _AUDIT_SUPPRESS_COLUMNS below, which correctly does NOT suppress it.
    "total_inspection_hours",  # column exists but is always NULL — use EPOCH(submitted_on-start_date_time)
    "subform_id", "internal_stakeholder_id", "observation_id",
    "client_details_submission_id", "completion_submission_id",
    "deferred_submission_id", "pr_submission_id", "return_submission_id",
    "active", "created_by", "modified_by",  # audit/system columns — no query use
}


# ---------------------------------------------------------------------------
# AUDIT DOMAIN constants — verified live against the DB (row-count-by-row-count
# queries, not just docs). See docs/Audit_Domain_Schema_Reference.docx and the
# domain-router plan's §1b for how this was derived. Data-freshness note: live
# row counts (audit_report=13, audit_corrective_action=12 as of this build) are
# much smaller than the doc's figures (46 / 69) — the environment was reseeded
# since the doc was written — but every STRUCTURAL trap below (dead columns,
# missing FKs, dead junction tables, CA_DRAFT placeholder ratio) was
# independently re-confirmed live and still holds.
# ---------------------------------------------------------------------------

AUDIT_ENUM_COLUMNS = {
    "audit_schedule.schedule_status": [
        "DRAFT", "PENDING_REVIEW", "APPROVED", "ONGOING", "COMPLETED", "RETURN_MODIFICATION",
    ],
    "audit_schedule.audit_status": [],  # DEAD — always NULL, confirmed live. Use schedule_status.
    "audit_plan.status": [
        "DRAFT", "SUBMITTED", "APPROVED", "RETURNED_FOR_MODIFICATION",
    ],
    "auditor_portfolio.status": [
        "APPROVED", "MODIFICATION",
    ],
    "audit_report.status": [
        "DRAFT", "SUBMITTED", "CLOSED", "CLOSED_WITH_EXCEPTION",
    ],  # CLOSED/CLOSED_WITH_EXCEPTION are in the schema enum but not present in current data
    # Confirmed live 2026-07-29 (qid=7f386051): "how many distinct audit forms are
    # currently available" generated WHERE status = 'ACTIVE' — that value does not
    # exist on this column at all (only DRAFT/SUBMITTED do), so the query silently
    # returned 0 instead of the correct 5. "Currently available" means active = true
    # (the boolean column) — audit_forms.status has no 'ACTIVE'/'PUBLISHED' concept;
    # do not filter status for an "available"/"active" question, only the active flag.
    "audit_forms.status": [
        "DRAFT", "SUBMITTED",
    ],
    "audit_corrective_action.status": [
        "DRAFT", "SUBMITTED", "OPEN", "OVERDUE", "CLOSED", "CLOSE_WITH_DEFERRED",
    ],
    "audit_corrective_action.implementation_status": [
        "PENDING_FOR_REVIEW", "APPROVE", "RETURN_FOR_MODIFICATION",
    ],  # NULL = not yet submitted for review
    "audit_corrective_action.adequacy_status": [
        "PENDING_FOR_REVIEW", "APPROVE", "RETURN_FOR_MODIFICATION",
    ],  # NULL = not yet submitted for review
    "audit_corrective_action.progress_stage": [
        "NOT_STARTED", "ON_TRACK", "BEHIND", "AT_RISK", "COMPLETED", "CLOSE_WITH_DEFERRED",
    ],  # NULL progress_stage = not started (mirrors inspection's progress_stage convention)
    # NOT the same values as inspection_corrective_action.responsible — confirmed
    # live 2026-07-29 this entry had been copy-pasted from inspection's enum
    # (CLIENT/INTERNAL_OPERATIONS/SUB_CONTRACTOR) without checking audit's own
    # live data, same root cause as the earlier ai_audit_answers column-name
    # copy-paste. A filter like WHERE aca.responsible = 'INTERNAL_OPERATIONS'
    # would silently match zero rows in the audit domain.
    "audit_corrective_action.responsible": [
        "KHIDMAH", "KHIDMAH_AND_CLIENT", "CLIENT", "SERVICE_PROVIDER_CONTRACTOR",
        "CONSULTANT",
    ],
    "audit_corrective_action.pending_with": [],  # messy free text (names/emails), NOT an enum
    "review.status": ["PENDING", "COMPLETED"],
    "review.action": ["APPROVE", "RETURN_FOR_MODIFICATION"],  # NULL when status=PENDING
    "review.review_type": ["ADEQUACY", "IMPLEMENTATION"],  # NULL for entity types that aren't split (PLAN/PORTFOLIO/SCHEDULE)
}

AUDIT_BUSINESS_RULES = [
    "AUDIT_PLAN vs AUDIT_REPORT — 'conducted'/'completed' are audit_report vocabulary ONLY, "
    "confirmed live as a real bug (not hypothetical): 'show the most recently conducted audit' "
    "queried audit_plan (a still-pending proposal, status IN ('SUBMITTED', "
    "'RETURNED_FOR_MODIFICATION')) instead of audit_report (an actual audit EVENT that was "
    "carried out — has a report, form answers, corrective actions, a facility) — confirmed wrong "
    "by the very next question asking what the auditor filled out for that 'audit', which only "
    "exists on audit_report/ai_audit_answers, never on audit_plan. "
    "RULE: a BARE, unqualified 'audit'/'the audit' (no 'plan' qualifier) means audit_report — "
    "conducted = arep.status = 'SUBMITTED', completed = arep.status IN ('CLOSED', "
    "'CLOSED_WITH_EXCEPTION'). "
    "AUDIT_PLAN DOES NOT USE 'conducted'/'completed' AT ALL — confirmed real usage: users refer "
    "to plan status directly by its own vocabulary ('submitted plans', 'approved plans', 'plans "
    "returned for modification'), never as 'conducted'/'completed' plans. Do not translate "
    "conducted/completed into an ap.status filter — if the question says 'audit plan(s)', map "
    "the status word the user actually said straight onto ap.status "
    "(DRAFT/SUBMITTED/APPROVED/RETURNED_FOR_MODIFICATION), and if it's a vague status word with "
    "no direct audit_plan status match, treat 'audit' as bare and use audit_report instead. When "
    "unsure and the question could plausibly lead to a follow-up about findings/answers/facility, "
    "use audit_report. THIS RULE IS ABOUT "
    "STATUS/LIFECYCLE questions only — for an ATTRIBUTE question (auditee, auditor, facility, "
    "schedule date) about an entity a recent turn already resolved to a specific audit_plan, "
    "do not ask the user to re-specify which audit; join straight through audit_schedule (which "
    "carries auditee/auditor/facility for both plans and reports) using that plan's id.",
    "NEVER use SELECT *. Always specify the columns the user needs.",
    "NEVER use SQL reserved words as table aliases (is, as, in, on, by, do, if).",
    "Use ILIKE for text matching, never LIKE.",
    "SOFT DELETE — every audit_* table has an 'active' boolean; always add WHERE active = true "
    "unless the question is specifically about deleted/archived records. EXCEPTION (confirmed "
    "live): the audit_form_* pure junction tables — audit_form_facilities, audit_form_countries, "
    "audit_form_organizations, audit_form_regions, audit_form_projects — are bare 2-column "
    "mapping tables (audit_form_id + the other FK only) with NO active column at all. A live-"
    "caught hallucination: the model assumed every table follows the soft-delete convention and "
    "added 'aff.active = true' on audit_form_facilities, getting a real Postgres 'column does "
    "not exist' error. Never add an active filter on these five tables specifically.",
    "LAST-MODIFIED TIMESTAMP NAMING: every audit_* table (confirmed live across all 16 of them, "
    "including audit_corrective_action/audit_plan/audit_report/audit_schedule/auditor_portfolio) "
    "uses modified_on/modified_by for its last-update timestamp — there is NO updated_on, "
    "updated_at, or modified_at column anywhere in this domain. A live-caught hallucination: the "
    "model assumed 'updated_on' (a common Rails/Django-style convention) and got a real Postgres "
    "'column does not exist' error at execution time. created_on/created_by is the matching "
    "creation-timestamp pair, also consistent across every audit_* table.",
    "AUDIT FORM-ANSWER LAYER — corrected 2026-07-29, supersedes an earlier stale rule that said "
    "this was empty. ai_audit_answers (audit's own flat answer table, NOT ai_answers — that "
    "belongs to inspection) is live and populated (445 rows). It was built by copy-pasting "
    "inspection's ai_answers schema, which left inspection-flavored column names that have since "
    "been renamed: inspection_report_id -> audit_report_id (uuid FK to audit_report.id), "
    "inspection_id -> audit_id (varchar, matches audit_report.audit_id display code), "
    "inspector_user_id -> auditor_user_id (uuid, the auditor who filed the report). "
    "ai_questions is a SHARED table across both domains (differentiated by module_name, not a "
    "domain column) — audit's real conduct-form modules are 'Quality Audit', "
    "'Health & Safety Audit', 'Logistics form', 'Statutory audit form - Soft services', "
    "'Sustainability -water', plus workflow-remark modules like 'Audit Corrective Action Entry' "
    "and the various Plan/Portfolio/Schedule Approval/Rejection modules. semantic_search / "
    "get_answers / get_answer_stats now work for audit questions the same way they do for "
    "inspection. See OBSERVATION -> CORRECTIVE ACTION rule below for the two-path form-answer "
    "join pattern.",
    "OBSERVATION → CORRECTIVE ACTION — two distinct paths, NEVER confuse them (audit's equivalent "
    "of inspection's OBSERVATION -> MITIGATIVE ACTION two-path rule): "
    "PATH 1 — the SPECIFIC observation a CA was raised against: "
    "JOIN ai_audit_answers aaa ON aaa.submission_id = aca.observation_id::varchar "
    "(cast the UUID side to varchar — NEVER aaa.submission_id::uuid = aca.observation_id, NEVER "
    "aca.observation_id::text = aaa.submission_id). submission_id groups ALL answers for one "
    "observation entry; to get the free-text finding itself, further filter "
    "WHERE aq.label = 'Findings / Remarks' after joining ai_questions aq ON aaa.element_id = "
    "aq.element_id. NOTE: unlike inspection (whose free-text field is literally labeled "
    "'Observation'), audit's conduct-form question set has NO question labeled 'Observation' — "
    "confirmed live; the label 'Findings / Remarks' is the equivalent field. Other labels in the "
    "same observation group include Audit Checkpoint, Compliance, Impact, Requirement, "
    "Requirement Type, Risk Level, Schedule Date and Time, Supporting Documents. "
    "PATH 2 — ALL answers for the whole audit report (reconstructing the full filled-out form): "
    "JOIN ai_audit_answers aaa ON aaa.audit_report_id = arep.id, then GROUP BY aaa.submission_id "
    "since each distinct submission_id is one repeatable entry (e.g. one observation) within "
    "the form. Use PATH 1 only when the user references a SPECIFIC CA/observation; use PATH 2 "
    "for 'reconstruct this audit's form' / 'all answers for this audit' questions. "
    "PATH 3 — the REVERSE of PATH 1 (forward, plural): ALL corrective actions raised against a "
    "SPECIFIC observation ('corrective actions for this observation', 'CAs raised against this "
    "finding'). Same join/cast as PATH 1 (aaa.submission_id = aca.observation_id::varchar, same "
    "aq.label = 'Findings / Remarks' anchor) but NOT filtered to one corrective_action_id — this "
    "returns every CA tied to that one observation_id. NOT aca.audit_id = arep.id (that returns "
    "ALL CAs in the whole report, unrelated to which observation raised them). Added 2026-08-10 — "
    "this had no worked example anywhere despite PATH 1 being seeded twice; it is the unresolved "
    "half of the 2026-07-31 incident that only ever got the whole-report-observations fix.",
    "RISK LEVEL / IMPACT / COMPLIANCE — OBSERVATION vs CORRECTIVE ACTION (confirmed live "
    "2026-08-10, user-supplied business fact, extended same day to Impact/Compliance): all "
    "THREE are FORM ANSWERS — Risk Level is auto-derived from Impact/Compliance/etc. entered "
    "on the SAME submission as the Findings/Remarks text, and Impact/Compliance are themselves "
    "directly-entered form answers in that identical submission group (confirmed in the "
    "observation group's label list: Audit Checkpoint, Compliance, Impact, Requirement, "
    "Requirement Type, Risk Level, ...). All three live in ai_audit_answers, in the SAME "
    "submission_id group as the observation, NOT on audit_corrective_action. 'high risk "
    "observations', 'risk/impact/compliance mappings for these observations', 'findings that "
    "are high risk' always mean aaa.answer_text WHERE aq.label IN ('Risk Level','Impact',"
    "'Compliance'), matched to their sibling Findings/Remarks answer via submission_id (NEVER "
    "via audit_report_id/arep.id — that pairs answers from ANY entry in the report, not "
    "necessarily the SAME one; two prior seed examples in this codebase made exactly that "
    "mistake for Risk Level before this fix, and a live follow-up asking for 'risk mappings' "
    "immediately reproduced the identical mistake via aca.risk_level_id/impact_id/"
    "compliance_id instead). aca.risk_level_id/impact_id/compliance_id are SEPARATE, later "
    "values that only exist once a CA is actually raised against the observation — use them "
    "ONLY for questions specifically about the corrective ACTION's own classification ('high "
    "risk corrective actions', 'risk level of this CA', 'impact of this CA'), never as a "
    "stand-in for the observation's own form answers — using them for a question about "
    "OBSERVATIONS silently excludes every observation that never had a CA raised, and can "
    "show a DIFFERENT value than what the auditor actually entered for that observation.",
    "audit_schedule.audit_status is a DEAD column — always NULL (confirmed live). "
    "The real schedule status is audit_schedule.schedule_status. NEVER filter/select audit_status.",
    "audit_report has NO facility_id or project_id column of its own — TWO-HOP via "
    "audit_report.schedule_id → audit_schedule.id → audit_schedule.facility_id/project_id. "
    "This differs from inspection, where inspection_report.facility_id is direct.",
    "audit_report.audit_type_id is a DEAD column — always NULL (confirmed live). "
    "The real audit-type path is THREE hops: audit_report.schedule_id → audit_schedule.id "
    "→ audit_schedule.audit_plan_id → audit_plan.id → audit_plan.audit_type_id → audit_type.id.",
    "audit_report.audit_cycle_id is ALSO a DEAD column — always NULL (confirmed live "
    "2026-07-31: 0 of 17 active rows populated), same trap as audit_type_id above just on a "
    "different FK on the same table. Live-caught: 'list all audits' INNER JOINed audit_cycle "
    "straight through this column and silently returned 0 rows for every real audit_report row. "
    "The real cycle path is THREE hops: audit_report.schedule_id → audit_schedule.id → "
    "audit_schedule.audit_plan_id → audit_plan.id → audit_plan.cycle_id → audit_cycle.id. NEVER "
    "join audit_cycle directly off audit_report.audit_cycle_id.",
    "DEAD JUNCTION TABLES — audit_category_portfolio, audit_sub_category_portfolio, "
    "audit_type_portfolio, and audit_cycle_auditor_portfolios are ALL always 0 rows (confirmed live) "
    "despite naming that reads as equally valid to the real tables. NEVER join them. "
    "Use portfolio_detail_audit_category / portfolio_detail_audit_subcategory / "
    "portfolio_detail_audit_type instead (joined via auditor_portfolio_detail.id → detail_id), "
    "and auditor_portfolio.audit_cycle_id directly (NOT the dead junction) for cycle coverage.",
    "'auditor' as a ROLE NAME is unreliable for 'who is doing audit work' questions — several users "
    "actually assigned as audit_schedule.auditor_id hold unrelated roles (HSE Officer, Internal "
    "Inspector, Super Admin, FM Manager). Query audit_schedule.auditor_id/.auditee_id directly. "
    "Only use the role table for explicit role/access-management questions, never to enumerate "
    "'who audits'.",
    "audit_corrective_action.corrective_action_id is the literal placeholder string 'CA_DRAFT' for "
    "a large share of rows (draft/unsubmitted CAs) — display-only, NEVER assume uniqueness, and "
    "never let a literal-ID lookup accidentally match every draft row.",
    "audit_corrective_action.pending_with is mostly NULL with a few stray free-text names/emails, "
    "not a clean enum — unreliable for exact-match filtering.",
    "audit_corrective_action has THREE separate TEXT columns, confirmed live via DB "
    "introspection: cause (why it happened), correction (the immediate fix already applied), "
    "and corrective_action (the root-cause/longer-term action proposed to prevent recurrence). "
    "corrective_action is NOT a synonym for correction and must never be answered with "
    "correction's value — and neither is text at all like corrective_action_id, the VARCHAR "
    "display code sharing a near-identical name.",
    "risk_level, impact, and compliance are THREE separate, identically-shaped lookup tables "
    "(id, name, scale, short_code, is_positive_impact) on audit_corrective_action, answering "
    "different questions: risk_level=severity, compliance=compliance-gap classification, "
    "impact=business-dimension affected. They share several identical label strings across "
    "tables with DIFFERENT UUIDs — pick the source table (risk_level_id / impact_id / "
    "compliance_id) from what the question actually asks about, never from the label text alone. "
    "NEVER INNER JOIN more than one of these three axes at once unless the question explicitly "
    "asks about more than one — confirmed live: a 'show details' follow-up that INNER-joined all "
    "three silently returned 0 rows for 6 known corrective actions because impact_id/compliance_id "
    "were NULL on those rows, dropping their correction/cause text along with everything else. "
    "For a generic 'show details'/'more info' follow-up, reuse only the axis already established "
    "by the prior filter; if displaying an extra axis without filtering on it, use LEFT JOIN.",
    "APPROVAL WORKFLOW — the 'review' table is a shared, POLYMORPHIC multi-level approval log "
    "(also used by inspection's INSPECTION_REPORT/INSPECTION_PORTFOLIO/INSPECTION_SCHEDULE — "
    "audit_report itself is NOT reviewed this way, only DRAFT→SUBMITTED directly). "
    "review.entity_id has NO real FK — always filter WHERE entity_type = '<TYPE>' AND "
    "entity_id = <target>.id. entity_type values for audit: 'AUDIT_PLAN' (2 levels), "
    "'PORTFOLIO' = auditor_portfolio (1 level), 'SCHEDULE' = audit_schedule (auditee's "
    "accept/reschedule step on the scheduled visit, level 1, rare level 2), "
    "'AUDIT_CORRECTIVE_ACTION' split by review_type: 'ADEQUACY' (auditor-side check, level 1, "
    "rare level 2) and 'IMPLEMENTATION' (two-level: auditee-side then auditor-side sign-off). "
    "Reviewer ROLE NAMES are dynamic, permission-driven labels, not fixed identities — NEVER "
    "hardcode a specific expected role name as a filter; describe the workflow STAGE "
    "(entity_type + review_type + level) instead, and join review→users→user_roles→role only "
    "when the question asks 'who reviewed/approved', not as a correctness gate.",
    "audit_schedule.reschedule_requested / reschedule_count reflect the auditee's SCHEDULE-level "
    "review step (propose a new date vs. accept) — not a separate workflow from 'review'.",
    "COLUMN-NAMING TRAP: portfolio_detail_audit_subcategory's FK column is audit_subcategory_id "
    "(no underscore before 'category') — NOT audit_sub_category_id, even though the TABLE it "
    "points to is named audit_sub_category (with the underscore). Confirmed live.",
    "DATE FOR 'WHEN AN AUDIT WAS CONDUCTED': use audit_report.submitted_on, NOT "
    "audit_schedule.follow_up_date. CORRECTED 2026-07-31 — live-confirmed the previous version "
    "of this rule was wrong: 'show audits conducted on jul 6 2026' filtered on follow_up_date and "
    "returned 0 rows for a date that genuinely had a submitted audit (submitted_on matched, "
    "follow_up_date did not). Checked the real data: follow_up_date values cluster almost "
    "entirely on month-end dates (2026-06-30, 07-29/30/31, 08-31) across the WHOLE table, not "
    "just a few rows — it's a target/due-by date for the schedule, not the day the audit actually "
    "happened. audit_schedule has no column recording the actual visit date at all; "
    "audit_report.submitted_on is the closest real signal and is treated as 'the audit was "
    "conducted that day' per business convention. Use follow_up_date only for genuinely "
    "forward-looking/scheduling questions ('which audits are due this month', 'overdue "
    "schedules') — never for 'when did/was this audit happen/conducted/occur'. "
    "audit_schedule has NO 'due_date' column (a live model confirmed to hallucinate this) — do "
    "NOT invent one. audit_cycle.due_date is a COMPLETELY DIFFERENT concept — the whole cycle's "
    "own end-of-period deadline, reached via audit_schedule.audit_plan_id → audit_plan.cycle_id "
    "→ audit_cycle.id (three hops, since audit_schedule has no direct cycle_id either). NEVER "
    "substitute audit_cycle.due_date when the question is about when an audit was conducted — "
    "that answers a different question ('when does this cycle wrap up') and silently returns 0 "
    "rows or the wrong rows.",
    "audit_cycle HAS NO name/cycle_name/title/code COLUMN (confirmed live) — identify a cycle by "
    "its acyc.start_date/acyc.end_date/acyc.status only, never invent a label column for it. This "
    "is the single most common mistake seen in practice: any question about audit_plan/schedule/ "
    "report that ALSO joins audit_cycle for context (not just cycle-specific questions) tends to "
    "add a hallucinated acyc.name — burned 7+ wasted LLM calls across eval runs before this rule "
    "was added. Same pattern as inspection_cycle, which also has no name column.",
    "audit_cycle.status is a STRICT 3-value CHECK constraint (confirmed live): UPCOMING, ONGOING, "
    "COMPLETED. There is NO 'CLOSED' value — a live-caught hallucination that silently returns "
    "wrong rows instead of erroring, because status != 'CLOSED' is trivially true for every row "
    "(no row can ever equal a value the column doesn't have), so it acts as a no-op filter and "
    "returns ALL cycles including COMPLETED/UPCOMING ones. 'currently active'/'ongoing' cycle "
    "MUST filter status = 'ONGOING' exactly — never status != 'CLOSED' or != 'COMPLETED' as a "
    "substitute for 'active', since UPCOMING would incorrectly pass either of those too.",
    "auditor_portfolio.status is a STRICT 4-value CHECK constraint (confirmed live): APPROVED, "
    "DRAFT, MODIFICATION, PENDING. There is NO 'SUBMITTED' value — a live-caught hallucination "
    "(structurally impossible, not just unused). 'under modification'/'needs modification'/"
    "'returned for rework' (including the post-approval-edit workflow branch — a super-admin/"
    "audit-in-charge-permission user editing an already-APPROVED portfolio) MUST filter "
    "status = 'MODIFICATION' exactly. Confirmed live: without this rule stated up front, the "
    "reasoning step has defaulted to status = 'APPROVED' instead for a 'which portfolios are "
    "under modification' question, answering a completely different question with no error or "
    "warning of any kind. 'pending submission' = status IN ('DRAFT', 'PENDING') instead — DRAFT/"
    "PENDING are valid schema values with 0 populated rows in the current dataset, not invalid ones.",
    "'COMPLETED' ON review IS NOT 'APPROVED' — confirmed live: of 261 COMPLETED reviews, 73 (28%) "
    "have action = 'RETURN_FOR_MODIFICATION' (rejected/sent back), not 'APPROVE'. This 28% figure is "
    "a data snapshot, not a fixed rule — re-verify via SELECT action, COUNT(*) FROM review WHERE "
    "status='COMPLETED' GROUP BY action if it ever needs re-checking; the underlying RULE (status "
    "COMPLETED != approved, always check action too) holds regardless of the exact percentage. "
    "review.status "
    "only tracks whether the review STEP is finished, not its outcome. 'who approved X' / 'was X "
    "approved' / 'approved corrective actions|plans|schedules' MUST filter "
    "WHERE status = 'COMPLETED' AND action = 'APPROVE' — filtering status = 'COMPLETED' alone "
    "silently includes rejected reviews and overcounts/misattributes approvals. Conversely "
    "'who rejected/returned X for modification' filters action = 'RETURN_FOR_MODIFICATION'. "
    "action is NULL while status = 'PENDING' (not yet actioned either way).",
    "audit_universe_id lives ONLY on audit_schedule — audit_plan has NO audit_universe_id column "
    "(confirmed live), and there is NO direct FK between audit_plan and audit_universe at all. "
    "For 'which universe does this plan belong to' OR 'which universe has the most plans', the "
    "ONLY valid path is the three-table hop audit_universe <- audit_schedule.audit_universe_id, "
    "audit_schedule.audit_plan_id -> audit_plan — confirmed live this gets it wrong by joining "
    "audit_plan.audit_type_id directly to audit_universe.id instead (a live-caught mistake: both "
    "happen to be UUID-typed so it doesn't error, it just silently matches nothing and returns 0 "
    "rows for every universe). NEVER join audit_plan to audit_universe on audit_type_id, category_id, "
    "or any column other than via audit_schedule. For a plan's own subject-matter tag (a DIFFERENT "
    "question), use audit_plan.category_id -> audit_category instead. NOTE: because a plan can have "
    "schedules in more than one universe (confirmed live: 2 plans do), COUNT(DISTINCT ap.id) per "
    "universe can legitimately count the same plan under more than one universe — that is correct, "
    "not a duplicate-row bug.",
    "ALIAS COLLISION — 'ap' is the established alias for audit_plan throughout this codebase's seed "
    "examples; auditor_portfolio uses 'po'. NEVER alias auditor_portfolio as 'ap' — it does not have "
    "a 'title' column (that's audit_plan.title) or a 'portfolio_id' column (its own PK is just 'id'). "
    "Confirmed live: reusing 'ap' for auditor_portfolio caused a hallucinated po.portfolio_id lookup.",
    "review_level IS NOT A RELIABLE SOURCE for 'how many levels of approval does X have' — it is a "
    "small, incomplete, stale-looking config table (4 rows total: AUDIT_CORRECTIVE_ACTION=1, "
    "INSPECTION_PORTFOLIO=1, PORTFOLIO=1, SCHEDULE=2 — a data snapshot, re-verify via "
    "SELECT * FROM review_level if it ever needs re-checking) that has NO row at all for AUDIT_PLAN or "
    "INSPECTION_REPORT, and even where it does have a row it disagrees with real data — "
    "AUDIT_CORRECTIVE_ACTION actually reaches level 2 for BOTH review_type='ADEQUACY' and "
    "'IMPLEMENTATION' (confirmed live), not the single level review_level claims. NEVER join or "
    "read review_level to answer a 'how many levels' question. Instead compute it empirically from "
    "review itself: SELECT entity_type, review_type, COUNT(DISTINCT level) or MAX(level) FROM review "
    "WHERE entity_type = '<TYPE>' [AND review_type = '<TYPE>'] GROUP BY entity_type, review_type — "
    "review_type only meaningfully splits AUDIT_CORRECTIVE_ACTION (ADEQUACY vs IMPLEMENTATION); "
    "it is NULL for every other entity_type. "
    "WHY adequacy sometimes reaches level 2 (confirmed live 2026-07-28): it is NOT random or a "
    "data-quality issue — the auditor explicitly escalates by checking a 'needs further approval' "
    "option in the frontend when submitting the adequacy review. No dedicated schema column tracks "
    "this trigger; a 2-level adequacy review is a normal, auditor-initiated case.",
    "UNPLANNED AUDITS ARE A FULL FAST-TRACK, not just 'a schedule missing its plan' (confirmed live "
    "2026-07-28): when audit_schedule.is_unplanned = true, the audit is created directly by a "
    "super-admin-permission user and skips the ENTIRE audit_plan/portfolio/cycle chain AND the "
    "auditee accept/reschedule review step (entity_type='SCHEDULE') — it goes straight to the "
    "auditor conducting it. audit_schedule.audit_plan_id is therefore NOT guaranteed populated for "
    "these rows. A query that INNER JOINs audit_plan to list/aggregate audit_schedule rows will "
    "silently drop every unplanned audit with no error or row-count signal — use LEFT JOIN "
    "audit_plan when the question shouldn't implicitly exclude unplanned audits. "
    "audit_plan.is_unplanned is a SEPARATE, much rarer flag (the plan itself created ad hoc) — "
    "do not conflate the two is_unplanned columns.",
    "POST-APPROVAL PORTFOLIO EDITS are a real, distinct workflow branch (confirmed live 2026-07-28): "
    "a super-admin/audit-in-charge-permission user can modify an auditor_portfolio row after it has "
    "already reached status='APPROVED' — this is not an error state or a sign the row should have "
    "gone back to a pending/returned status first.",
    "WHO ON audit_corrective_action — FOUR different columns answer FOUR different 'who' "
    "questions, confirmed live 2026-07-29 (qid=f87d531f: 'who proposed the corrective action' "
    "got answered with aca.responsible AS proposer, returning values like "
    "'SERVICE_PROVIDER_CONTRACTOR'/'KHIDMAH_AND_CLIENT' as if they were a person's name): "
    "created_by (UUID FK to users, populated on every row) = who raised/logged this CA — the "
    "right answer for 'who proposed/raised/logged/created' questions. internal_stakeholder_id "
    "(UUID FK to users, populated on every row regardless of responsible's value) = the "
    "Khidmah-side assigned point of contact/owner — the right answer for 'who owns/is assigned "
    "to' questions. responsible (VARCHAR enum: KHIDMAH, KHIDMAH_AND_CLIENT, CLIENT, "
    "SERVICE_PROVIDER_CONTRACTOR, CONSULTANT — NOT a users FK, NEVER join it to users) = which "
    "PARTY/ORGANIZATION is responsible for executing the fix — a category, not a person; only "
    "the right answer when the question asks 'which party/organization'. pending_with (free "
    "text, mostly NULL, unreliable) = whichever name/email a user happened to type in, not a "
    "reliable lookup. Never substitute one for another just because the question says 'who'.",
    "STAKEHOLDER/AUDITOR GRAIN MISMATCH — confirmed live 2026-07-29 (qid=4337464c): "
    "internal_stakeholder_id lives on audit_corrective_action (one value PER CORRECTIVE "
    "ACTION), while auditor_id/auditee_id live on audit_schedule (one value for the WHOLE "
    "audit). 'who are the auditor, auditee, and internal stakeholder for this audit' joined "
    "both together but selected no corrective_action_id — got back 5 rows with the SAME "
    "auditor/auditee repeated and a DIFFERENT stakeholder each time, nothing explaining why. "
    "The fan-out itself was correct (the audit genuinely has 5 CAs, each with its own "
    "stakeholder) — it just needs corrective_action_id in the SELECT so it reads as '5 CAs, "
    "each with its own stakeholder' instead of looking like duplicate/inconsistent data. "
    "ALWAYS include aca.corrective_action_id when internal_stakeholder_id and auditor_id/"
    "auditee_id appear in the same query.",
    "audit_schedule.audit_sub_category_id is NULL on ~28% of active schedules, confirmed live "
    "2026-07-29 (11/39) — and NOT limited to unplanned audits (category_id on audit_plan and "
    "auditor_id on audit_schedule are, by contrast, never NULL). A query that INNER JOINs "
    "audit_sub_category to answer a 'who/what' question about a specific audit will silently "
    "return zero rows whenever sub-category alone happens to be unset, even though the auditor "
    "and category both exist — confirmed live: 'who conducted it, what category and sub "
    "category' returned rows=0 for exactly this reason on an audit with a real auditor and "
    "category. Use LEFT JOIN audit_sub_category unless the question explicitly filters for "
    "schedules that DO have a sub-category assigned.",
    "SCOPE DISCIPLINE for general 'list <entity>' / 'list <entity> with details' questions — "
    "confirmed live 2026-07-29 (qid=d5e2317b): 'list all overdue corrective actions' (no mention "
    "of facility, project, client, entity, or responsible/pending_with) generated a schema_hint "
    "with all six joined in anyway, self-injecting THREE unrelated hallucinated FKs "
    "(aca.responsible = users.id, aca.pending_with = users.id, direct aca.facility_id/"
    "project_id) that the validator then had to block, costing a wasted iteration on a question "
    "that never asked for any of it. DEFAULT to the entity's own columns only (for "
    "audit_corrective_action: corrective_action_id, status, progress_stage, implementation_status, "
    "adequacy_status, cause, correction, corrective_action, target_close_out_date, completed_on) "
    "plus whatever the question's own words name. Only reach for facility/project/client/entity "
    "(multi-hop via audit_report → audit_schedule) or risk_level/impact/compliance/reviewer "
    "identity when the question itself names that concept — 'with details'/'full details' alone "
    "does NOT license adding facility, project, client, entity, or responsible/pending_with; it "
    "means 'show the entity's own text fields', not 'join everything reachable'.",
]

AUDIT_TABLE_RELATIONSHIPS = """\
AUDIT WORKFLOW, END TO END:
  audit_universe ─┐
  audit_category ──┼──▶ audit_plan ──▶ audit_schedule ──▶ audit_report ──▶ audit_corrective_action
  audit_type ──────┘                        │
                                             └── (separately) auditor_portfolio — standing
                                                 "auditor responsible for X this cycle" assignment,
                                                 NOT a step in this linear chain.

HOW TABLES CONNECT:
  audit_plan.cycle_id              → audit_cycle.id
  audit_plan.audit_type_id         → audit_type.id
  audit_plan.category_id           → audit_category.id
  TRAP — audit_plan has NO audit_type_name or category_name column of its
  own (confirmed live: a schema_hint selecting ap.audit_type_name/
  ap.category_name directly got a real Postgres 'column does not exist'
  error, costing a retry). These are FK ids only — JOIN audit_type aty ON
  ap.audit_type_id = aty.id and SELECT aty.name AS audit_type_name (same
  pattern for category via audit_category acat / acat.name AS category_name).
  audit_plan.user_id               → users.id           (the proposer — verified: always 'Internal Auditor')
  audit_schedule.audit_plan_id     → audit_plan.id
  audit_schedule.auditor_id        → users.id            (who conducts the visit)
  audit_schedule.auditee_id        → users.id            (who is being audited)
  audit_schedule.facility_id       → facility.id
  facility.country_id              → country.id           (DIRECT — no need to go via region)
  facility.region_id               → region.id             (region.country_id → country.id also reaches country)
  audit_schedule.project_id        → project.id
  audit_schedule.audit_universe_id → audit_universe.id
  audit_schedule.audit_sub_category_id → audit_sub_category.id
  audit_report.schedule_id         → audit_schedule.id   (⚠ ONLY path to facility/project — see below)
  audit_report.audit_cycle_id      → audit_cycle.id
  audit_report.audit_form_used_id  → audit_forms.id      (always populated — the reliable form link)
  audit_report.client_id           → client.id           (owning client org)
  audit_report.entity_id           → entity.id            (operating/contracting business unit —
                                       genuinely independent of client_id, NOT redundant; both
                                       always populated)
  audit_corrective_action.audit_id → audit_report.id
  TRAP — audit_report ALSO has its own column literally named audit_id, but it's a
  DIFFERENT thing: audit_report.audit_id is a VARCHAR display code (e.g.
  '2026/07/ST034/AUD001'), not the PK. audit_corrective_action.audit_id is a UUID
  and must join to audit_report.id — NEVER audit_report.audit_id, even inside a
  subquery built to "fetch the audit's id" (confirmed live: a correlated subquery
  written as `SELECT arep.audit_id FROM audit_report arep WHERE ...` produced
  'operator does not exist: uuid = character varying' — the same trap as a direct
  JOIN, just one level removed). Only select audit_report.audit_id for DISPLAY.
  audit_corrective_action.risk_level_id  → risk_level.id
  audit_corrective_action.impact_id      → impact.id
  audit_corrective_action.compliance_id  → compliance.id
  audit_corrective_action.internal_stakeholder_id → users.id
  auditor_portfolio.user_id        → users.id
  auditor_portfolio.audit_cycle_id → audit_cycle.id      (DIRECT — NOT via the dead
                                       audit_cycle_auditor_portfolios junction)
  auditor_portfolio_detail.portfolio_id → auditor_portfolio.id
  auditor_portfolio_detail.facility_id  → facility.id
  auditor_portfolio_detail.project_id   → project.id
  portfolio_detail_audit_category.detail_id    → auditor_portfolio_detail.id
  portfolio_detail_audit_subcategory.detail_id → auditor_portfolio_detail.id
  portfolio_detail_audit_type.detail_id        → auditor_portfolio_detail.id
  review.entity_id                 → POLYMORPHIC, no real FK — filter by entity_type (see rules)
  review.assigned_to_id            → users.id
  review.reviewed_by               → users.id

TRAP — facility/project from a report or corrective action, TWO/THREE hops (NOT direct):
  From audit_report:            audit_report.schedule_id → audit_schedule.id → .facility_id / .project_id
  From audit_corrective_action: .audit_id → audit_report.id → .schedule_id → audit_schedule.id → .facility_id

TRAP — audit type of a report/schedule, THREE/TWO hops (NOT direct — audit_report.audit_type_id is dead):
  From audit_report:  .schedule_id → audit_schedule.id → .audit_plan_id → audit_plan.id → .audit_type_id
  From audit_schedule: .audit_plan_id → audit_plan.id → .audit_type_id

TIMELINESS:
  Corrective action: status NOT IN ('CLOSED','CLOSE_WITH_DEFERRED') = open; status = 'OVERDUE' is
    its own explicit state, not derived from date arithmetic.
  Report: status NOT IN ('CLOSED','CLOSED_WITH_EXCEPTION') = open.
  Schedule: use schedule_status, never audit_status (dead).
"""

# ---------------------------------------------------------------------------
# Topic-gated core/topical split (2026-08 token audit)
# ---------------------------------------------------------------------------
# BUSINESS_RULES/TABLE_RELATIONSHIPS/AUDIT_BUSINESS_RULES/AUDIT_TABLE_RELATIONSHIPS
# above are left completely unchanged — for_system_prompt()/for_schema_hint()
# and anything else that wants the full content still gets it verbatim. The
# reasoning-LLM system prompt (agent/orchestrator/loop.py's __init__), which
# previously spliced ALL of BUSINESS_RULES/TABLE_RELATIONSHIPS in
# unconditionally on every question, now uses only the CORE subset below;
# the narrow-topic remainder is surfaced per-question via
# agent/prompts/builder.py's/builder_audit.py's CONDITIONAL_BLOCKS (same
# mechanism already used for MA_WORKFLOW_BLOCK) instead of paying its full
# token cost on every call. Audit's AUDIT_BUSINESS_RULES alone measured
# 29,014 chars (~7,250 tokens), unconditionally, on a domain where the
# reasoning prompt was observed at ~67K chars total.
#
# What stays CORE (never gated): anything that guards the MODEL's own
# hallucination risk regardless of what the user asked — dead columns,
# join-hop traps, alias collisions, naming traps, the hub-and-spoke FK map.
# There's no reliable question-text signal for "the model is about to
# invent this column," so those rules must fire on every question. Only
# rules whose relevance genuinely follows from the QUESTION's own wording
# (risk level, who approved, communication group, portfolio status, ...)
# are moved to topic gating below.
#
# Extraction is done by filtering the EXISTING lists/string (never by
# retyping prose), so the extracted text is byte-identical to the source.

def _partition_rules(rules: list, marker_groups: dict) -> tuple:
    """Split `rules` into (core, {topic: [matched rules]}) by rule-prefix
    match against marker_groups = {topic: [prefix, ...]}."""
    prefix_to_topic = {
        m: topic for topic, markers in marker_groups.items() for m in markers
    }
    core = []
    topical = {topic: [] for topic in marker_groups}
    for rule in rules:
        hit = next((t for m, t in prefix_to_topic.items() if rule.startswith(m)), None)
        if hit:
            topical[hit].append(rule)
        else:
            core.append(rule)
    return core, topical


def _extract_tr_section(text: str, start_marker: str, end_marker) -> tuple:
    """Cut [start_marker, end_marker) out of `text`. end_marker=None cuts to
    the end of the string. Returns (remainder, chunk)."""
    i = text.index(start_marker)
    j = len(text) if end_marker is None else text.index(end_marker, i + len(start_marker))
    return text[:i] + text[j:], text[i:j]


# NOTE: "observation_ma" (the OBSERVATION -> MITIGATIVE ACTION multi-path
# join rule + RISK LEVEL/IMPACT rule) was gated here originally but reverted
# to always-core 2026-08-10 — eval regression (FLT-02/FLT-03 in eval.py,
# reproduced on 2 separate reruns): with this rule gated, the reasoning LLM's
# own draft_sql stopped following the documented submission_id/aq.label join
# pattern (e.g. drafted `JOIN risk_level rl ON obs.answer_text = rl.name`
# instead) even though the rule text was confirmed present in the assembled
# prompt — moving it from its fixed position (embedded right after schema)
# to an appended end-of-prompt CONDITIONAL_BLOCKS section, stacked alongside
# other same-topic content (MA_WORKFLOW_BLOCK + a Tier-2 worked example),
# was enough to make the model stop reliably applying it. This is the
# largest, most structurally complex rule in BUSINESS_RULES (~2.5-3K chars,
# multi-path join logic) — too fragile to gate given the demonstrated risk;
# the smaller/simpler topics below (user_roles, approval_workflow) showed no
# such regression across the same eval run.
_INSPECTION_RULE_TOPICS = {
    "user_roles": [
        "USER ROLES — users have multiple roles",
    ],
    "approval_workflow": [
        "APPROVAL WORKFLOW — 'review' is a shared",
        "'COMPLETED' ON review IS NOT 'APPROVED'",
        "review_level IS NOT RELIABLE for 'how many levels",
    ],
}
BUSINESS_RULES_CORE, _INSPECTION_RULES_TOPICAL = _partition_rules(
    BUSINESS_RULES, _INSPECTION_RULE_TOPICS)

_tr_rest, _TR_COMMUNICATION_GROUPS = _extract_tr_section(
    TABLE_RELATIONSHIPS, "COMMUNICATION GROUPS:\n", "USER ROLES:\n")
_tr_rest, _TR_USER_ROLES = _extract_tr_section(
    _tr_rest, "USER ROLES:\n", "CRITICAL ai_* JOIN PATTERN:\n")
_tr_rest, _TR_APPROVAL_WORKFLOW = _extract_tr_section(
    _tr_rest, "APPROVAL WORKFLOW (review table", None)
TABLE_RELATIONSHIPS_CORE = _tr_rest
del _tr_rest

# Phase 3.15: for_schema_hint()'s SQL-gen "targeted" branch — the complement
# of TABLE_RELATIONSHIPS_CORE above. 3.14's causal ablation (full 105-
# question suite, all 6 CORE sections removed at once) showed zero
# correctness regression; these 3 sections were kept unconditional instead
# of also being cut because a keyword-coverage check found the eval suite
# had zero/weak coverage of their topics at ablation time (communication
# groups, approval-workflow's COMPLETED-vs-approved trap, user-role
# EXISTS-not-JOIN fan-out) — since fixed by eval.py's FLT-08/JOI-13..15/
# PPL-04/PPL-05, but not re-ablated individually, so left in rather than
# assumed safe. Reuses the same extracted chunks as TABLE_RELATIONSHIPS_CORE
# so the two constants are guaranteed to partition TABLE_RELATIONSHIPS with
# no section duplicated or dropped.
TABLE_RELATIONSHIPS_SQL_GEN = _TR_COMMUNICATION_GROUPS + _TR_USER_ROLES + _TR_APPROVAL_WORKFLOW

INSPECTION_RULE_TOPIC_TEXT = {
    "user_roles": (
        "\n".join(f"  - {r}" for r in _INSPECTION_RULES_TOPICAL["user_roles"])
        + "\n\n" + _TR_USER_ROLES
    ),
    "approval_workflow": (
        "\n".join(f"  - {r}" for r in _INSPECTION_RULES_TOPICAL["approval_workflow"])
        + "\n\n" + _TR_APPROVAL_WORKFLOW
    ),
    "communication_groups": _TR_COMMUNICATION_GROUPS,
}

_AUDIT_RULE_TOPICS = {
    "form_answers": [
        "AUDIT FORM-ANSWER LAYER — corrected 2026-07-29",
        "OBSERVATION → CORRECTIVE ACTION — two distinct paths",
        "RISK LEVEL / IMPACT / COMPLIANCE — OBSERVATION vs CORRECTIVE ACTION",
        "risk_level, impact, and compliance are THREE separate",
    ],
    "approval_workflow": [
        "APPROVAL WORKFLOW — the 'review' table is a shared",
        "'COMPLETED' ON review IS NOT 'APPROVED' — confirmed live",
        "review_level IS NOT A RELIABLE SOURCE",
    ],
    "who_questions": [
        "'auditor' as a ROLE NAME is unreliable",
        "WHO ON audit_corrective_action — FOUR different columns",
        "STAKEHOLDER/AUDITOR GRAIN MISMATCH",
    ],
    "cycle": [
        "audit_cycle HAS NO name/cycle_name/title/code COLUMN",
        "audit_cycle.status is a STRICT 3-value CHECK constraint",
    ],
    "portfolio": [
        "auditor_portfolio.status is a STRICT 4-value CHECK constraint",
        "POST-APPROVAL PORTFOLIO EDITS are a real",
    ],
    "universe": [
        "audit_universe_id lives ONLY on audit_schedule",
    ],
    "date_conducted": [
        "DATE FOR 'WHEN AN AUDIT WAS CONDUCTED'",
    ],
}
AUDIT_BUSINESS_RULES_CORE, _AUDIT_RULES_TOPICAL = _partition_rules(
    AUDIT_BUSINESS_RULES, _AUDIT_RULE_TOPICS)

# AUDIT_TABLE_RELATIONSHIPS is left fully core (not split) — unlike
# inspection's, it's almost entirely hub-and-spoke FK map + TRAP warnings
# (dead columns, multi-hop joins), which per the policy above can't be
# safely question-text-gated.
AUDIT_TABLE_RELATIONSHIPS_CORE = AUDIT_TABLE_RELATIONSHIPS

AUDIT_RULE_TOPIC_TEXT = {
    topic: "\n".join(f"  - {r}" for r in rules)
    for topic, rules in _AUDIT_RULES_TOPICAL.items()
}


AUDIT_LOOKUP_TABLES = {
    "audit_universe":      {"name_expr": None, "name_col": "name"},
    "audit_category":      {"name_expr": None, "name_col": "name"},
    "audit_sub_category":  {"name_expr": None, "name_col": "name"},
    "audit_type":          {"name_expr": None, "name_col": "name"},
    "audit_forms":         {"name_expr": None, "name_col": "title"},
    "risk_level":          {"name_expr": None, "name_col": "name"},
    "impact":              {"name_expr": None, "name_col": "name"},
    "compliance":          {"name_expr": None, "name_col": "name"},
    "facility":            {"name_expr": None, "name_col": "name"},
    "project":              {"name_expr": None, "name_col": "name"},
    "client":               {"name_expr": None, "name_col": "name"},
    "entity":               {"name_expr": None, "name_col": "name"},
    "users":                {"name_expr": "first_name || ' ' || last_name", "name_col": None},
}

AUDIT_FK_RESOLUTION = {
    "auditor_id":           ("users", "u_aud",  "u_aud.first_name || ' ' || u_aud.last_name AS auditor_name"),
    "auditee_id":           ("users", "u_ee",   "u_ee.first_name || ' ' || u_ee.last_name AS auditee_name"),
    "facility_id":          ("facility", "fac", "fac.name AS facility_name"),
    "project_id":           ("project",  "proj", "proj.name AS project_name"),
    "client_id":            ("client",   "cl",   "cl.name AS client_name"),
    "entity_id":            ("entity",   "ent",  "ent.name AS entity_name"),
    "audit_type_id":        ("audit_type", "aty", "aty.name AS audit_type_name"),
    "category_id":          ("audit_category", "acat", "acat.name AS category_name"),
    "audit_sub_category_id":("audit_sub_category", "asub", "asub.name AS sub_category_name"),
    "audit_universe_id":    ("audit_universe", "auniv", "auniv.name AS universe_name"),
    # audit_cycle has NO name column (confirmed live) — its identity is its
    # date span + status, same pattern as inspection_cycle. Hallucinating
    # acyc.name/cycle_name/title/code burned 7 LLM calls in one eval_audit.py
    # run before the model gave up and dropped the column entirely.
    "cycle_id":             ("audit_cycle", "acyc", "acyc.start_date, acyc.end_date, acyc.status AS cycle_status"),
    "audit_cycle_id":       ("audit_cycle", "acyc", "acyc.start_date, acyc.end_date, acyc.status AS cycle_status"),
    "audit_plan_id":        ("audit_plan", "ap", "ap.title AS plan_title"),
    "schedule_id":          ("audit_schedule", "asch", "asch.schedule_id AS schedule_display_id"),
    "audit_id":             ("audit_report", "arep", "arep.audit_id AS report_display_id"),
    "audit_form_used_id":   ("audit_forms", "aform", "aform.title AS form_title"),
    "risk_level_id":        ("risk_level", "rl", "rl.name AS risk_level_name"),
    "impact_id":            ("impact", "imp", "imp.name AS impact_name"),
    "compliance_id":        ("compliance", "comp", "comp.name AS compliance_name"),
    "internal_stakeholder_id": ("users", "u_stk", "u_stk.first_name || ' ' || u_stk.last_name AS stakeholder_name"),
}

AUDIT_TABLES = [
    "audit_universe",
    "audit_category",
    "audit_sub_category",
    "audit_type",
    "audit_cycle",
    "audit_plan",
    "audit_schedule",
    "audit_report",
    "audit_corrective_action",
    "auditor_portfolio",
    "auditor_portfolio_detail",
    "audit_forms",
    "review",
    "review_level",
    "compliance",
]

# AI flat tables audit actually uses — ai_questions is SHARED with inspection
# (differentiated by module_name, not a domain column); ai_audit_answers is
# audit's own answer table (see AUDIT_BUSINESS_RULES "AUDIT FORM-ANSWER LAYER"
# for the column-rename history). Was [] until 2026-07-29 — introspect_audit_schema()
# previously assumed this layer was empty for audit; confirmed live it is not.
AUDIT_AI_TABLES = ["ai_questions", "ai_audit_answers"]

AUDIT_USEFUL_COLUMNS = {
    "ai_questions": [
        "element_id", "label", "module_name", "module_id", "entity_type",
    ],
    "ai_audit_answers": [
        "element_id", "submission_id", "answer_text", "answer_numeric", "score",
        "audit_report_id", "audit_id", "auditor_user_id", "module_name",
        # audit_report_id/audit_id/auditor_user_id were renamed from
        # inspection-flavored inspection_report_id/inspection_id/inspector_user_id
        # 2026-07-29 — see AUDIT_BUSINESS_RULES.
    ],
    "audit_plan": [
        "title", "status", "is_unplanned", "audit_type_id", "category_id", "cycle_id", "user_id",
    ],
    "audit_schedule": [
        "schedule_id", "schedule_status", "duration", "follow_up_date", "reschedule_count",
        "reschedule_requested", "is_unplanned", "auditor_id", "auditee_id", "facility_id",
        "project_id", "audit_plan_id", "audit_universe_id", "audit_sub_category_id",
        # audit_status DELIBERATELY excluded — dead column, see AUDIT_BUSINESS_RULES
    ],
    "audit_report": [
        "audit_id", "status", "audit_score", "total_audit_hours", "submitted_on", "closed_on",
        "schedule_id", "audit_form_used_id", "client_id", "entity_id",
        # audit_type_id, audit_cycle_id, facility_id, project_id DELIBERATELY excluded —
        # dead/absent, see AUDIT_BUSINESS_RULES. audit_cycle_id removed 2026-07-31: it was
        # still listed here (unlike audit_type_id, already excluded) despite being confirmed
        # 0-of-17 populated live — the LLM could still see it as an available column and
        # propose it, relying only on the reactive BLOCK rule to catch it after the fact
        # instead of never seeing it as an option in the first place.
    ],
    "audit_corrective_action": [
        "corrective_action_id", "status", "implementation_status", "adequacy_status",
        "progress_stage", "responsible", "pending_with", "cause", "correction",
        "corrective_action",
        # THREE SEPARATE TEXT FIELDS, confirmed live via DB introspection — do not
        # conflate: cause = why it happened, correction = the immediate fix already
        # applied, corrective_action = the (longer-term/root-cause) action proposed
        # to prevent recurrence. corrective_action was missing from this list
        # entirely until a live query asking for "corrective action text" only
        # got back aca.correction because the model had never been told this
        # column exists.
        "target_close_out_date", "close_on", "completed_on", "audit_id",
        "risk_level_id", "impact_id", "compliance_id", "internal_stakeholder_id",
        "external_stakeholder_email", "external_stakeholder_name", "external_stakeholder_phone",
    ],
    "auditor_portfolio": [
        "status", "submitted_date", "user_id", "audit_cycle_id",
    ],
    "auditor_portfolio_detail": [
        "facility_id", "project_id", "portfolio_id",
    ],
    "audit_cycle": [
        "start_date", "end_date", "due_date", "status", "send_notification_to_auditors",
    ],
    "audit_forms": [
        "title", "status", "audit_scope", "audit_type_id", "audit_categories_id",
        "audit_sub_category_id", "audit_universe_id", "audit_cycle_id", "client_id", "entity_id",
        # audit_schedule_id DELIBERATELY excluded — always NULL, this is a TEMPLATE library
    ],
    "review": [
        "entity_id", "entity_type", "review_type", "level", "status", "action",
        "assigned_to_id", "reviewed_by", "reviewed_on", "submission_id",
        # entity_type: 'AUDIT_PLAN' | 'PORTFOLIO' | 'SCHEDULE' | 'AUDIT_CORRECTIVE_ACTION'
        #   (also 'INSPECTION_REPORT'/'INSPECTION_PORTFOLIO'/'INSPECTION_SCHEDULE' for
        #   inspection domain — filter entity_type when only audit rows matter)
        # entity_id is POLYMORPHIC — no FK — always pair with entity_type
    ],
}

_AUDIT_SUPPRESS_COLUMNS: set[str] = {
    "audit_status",  # DEAD on audit_schedule — always NULL
    "audit_type_id",  # DEAD on audit_report — always NULL (real column exists on audit_plan too, kept there)
    "gp_score",  # near-dead on audit_report (mostly NULL, redundant with audit_score where present)
    "submission_answers",  # sparse JSONB, only a handful of rows populated
    "active", "created_by", "modified_by",
    # observation_id: suppressed from the generic column dump (mirrors inspection's
    # identical choice for inspection_corrective_action, see _SUPPRESS_COLUMNS above)
    # — it's real and heavily used, but taught via the dedicated OBSERVATION ->
    # CORRECTIVE ACTION business rule + seed examples instead of the noisy generic list.
    "observation_id",
}

_AUDIT_ENUM_QUERIES = {
    "audit_type":         ("SELECT name FROM audit_type ORDER BY name", "name"),
    "audit_category":     ("SELECT name FROM audit_category ORDER BY name", "name"),
    "audit_sub_category": ("SELECT name FROM audit_sub_category ORDER BY name", "name"),
    "audit_universe":     ("SELECT name FROM audit_universe ORDER BY name", "name"),
    "compliance":         ("SELECT name FROM compliance ORDER BY name", "name"),
    "audit_plan.status": (
        "SELECT DISTINCT status FROM audit_plan WHERE status IS NOT NULL ORDER BY status", "status"
    ),
    "audit_schedule.schedule_status": (
        "SELECT DISTINCT schedule_status FROM audit_schedule WHERE schedule_status IS NOT NULL "
        "ORDER BY schedule_status", "schedule_status"
    ),
    "audit_report.status": (
        "SELECT DISTINCT status FROM audit_report WHERE status IS NOT NULL ORDER BY status", "status"
    ),
    # audit_corrective_action's own status columns — previously missing here,
    # relying only on the static AUDIT_ENUM_COLUMNS fallback. Confirmed live:
    # this gap is what let the SQL-generation model guess
    # implementation_status = 'APPROVED' (plausible English, wrong enum
    # literal) instead of the real 'APPROVE' — silently 0 rows, no error.
    "audit_corrective_action.status": (
        "SELECT DISTINCT status FROM audit_corrective_action WHERE status IS NOT NULL ORDER BY status",
        "status"
    ),
    "audit_corrective_action.implementation_status": (
        "SELECT DISTINCT implementation_status FROM audit_corrective_action "
        "WHERE implementation_status IS NOT NULL ORDER BY implementation_status", "implementation_status"
    ),
    "audit_corrective_action.adequacy_status": (
        "SELECT DISTINCT adequacy_status FROM audit_corrective_action "
        "WHERE adequacy_status IS NOT NULL ORDER BY adequacy_status", "adequacy_status"
    ),
    "audit_corrective_action.progress_stage": (
        "SELECT DISTINCT progress_stage FROM audit_corrective_action "
        "WHERE progress_stage IS NOT NULL ORDER BY progress_stage", "progress_stage"
    ),
    # Added 2026-07-29 after "how many distinct audit forms are currently
    # available" hallucinated status = 'ACTIVE' (doesn't exist — only DRAFT/
    # SUBMITTED do) and silently returned 0 instead of 5. Live-refreshed rather
    # than relying only on the static AUDIT_ENUM_COLUMNS fallback, same fix
    # shape as audit_corrective_action's status columns above.
    "audit_forms.status": (
        "SELECT DISTINCT status FROM audit_forms WHERE status IS NOT NULL ORDER BY status", "status"
    ),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TableSchema:
    name: str
    columns: list
    foreign_keys: list
    primary_key: list


@dataclass
class DatabaseSchema:
    tables: dict = field(default_factory=dict)       # inspection_* tables (or domain-equivalent)
    ai_tables: dict = field(default_factory=dict)    # ai_question, ai_answer
    lookup_tables: dict = field(default_factory=dict)
    enum_values: dict = field(default_factory=dict)  # table → list of name values (dynamic)
    _system_cache: Optional[str] = field(default=None, repr=False)
    _reasoning_cache: Optional[str] = field(default=None, repr=False)

    # ── Domain-injected constant sets — default to the INSPECTION globals so
    # every existing caller (introspect_schema(), zero call-site changes) gets
    # byte-identical behavior. introspect_audit_schema() below passes the
    # AUDIT_* equivalents instead. This is dependency injection, not a second
    # class, since the introspection MECHANISM (BFS-free, just per-table
    # column/FK formatting) is identical between domains — only the content
    # sets differ. See agent/domains.py's DOMAIN_REGISTRY for how a domain
    # picks its schema instance.
    _domain_tables_list: list = field(default_factory=lambda: list(INSPECTION_TABLES), repr=False)
    _ai_tables_list: list = field(default_factory=lambda: list(AI_TABLES), repr=False)
    _lookup_config: dict = field(default_factory=lambda: dict(LOOKUP_TABLES), repr=False)
    _fk_resolution: dict = field(default_factory=lambda: dict(FK_RESOLUTION), repr=False)
    _optional_fks: set = field(default_factory=lambda: set(OPTIONAL_FKS), repr=False)
    _useful_columns: dict = field(default_factory=lambda: dict(USEFUL_COLUMNS), repr=False)
    _enum_columns: dict = field(default_factory=lambda: dict(ENUM_COLUMNS), repr=False)
    _suppress_columns: set = field(default_factory=lambda: set(_SUPPRESS_COLUMNS), repr=False)
    _business_rules: list = field(default_factory=lambda: list(BUSINESS_RULES), repr=False)
    _table_relationships: str = field(default=TABLE_RELATIONSHIPS, repr=False)
    # Core-only (topic-gated remainder surfaced via CONDITIONAL_BLOCKS instead)
    # — see the "Topic-gated core/topical split" section above. Used only by
    # orchestrator/loop.py's reasoning-system-prompt splice; _business_rules/
    # _table_relationships above are untouched for every other consumer
    # (for_system_prompt, for_schema_hint).
    _business_rules_core: list = field(
        default_factory=lambda: list(BUSINESS_RULES_CORE), repr=False)
    _table_relationships_core: str = field(default=TABLE_RELATIONSHIPS_CORE, repr=False)
    # for_schema_hint() only (Phase 3.15) — see TABLE_RELATIONSHIPS_SQL_GEN
    # above for why this differs from both _table_relationships (full,
    # still used by for_system_prompt) and _table_relationships_core (used
    # only by the reasoning-prompt splice). Phase 3.16: this is now also the
    # "no question supplied" / "domain doesn't support gating" fallback for
    # _table_relationships_topic_gated() below, not for_schema_hint()'s
    # only source of this content.
    _table_relationships_sql_gen: str = field(default=TABLE_RELATIONSHIPS_SQL_GEN, repr=False)
    # Phase 3.16: for_schema_hint()'s per-question topic gating of the same
    # 3 sections TABLE_RELATIONSHIPS_SQL_GEN unconditionally concatenates —
    # reuses the identical extracted chunks (_TR_USER_ROLES etc.) so a
    # gated call renders byte-identical section text to the unconditional
    # one, just fewer of them. Empty + _supports_tr_topic_gating=False for
    # audit (see the audit factory override) since AUDIT_TABLE_RELATIONSHIPS
    # was never split into per-topic chunks.
    _tr_user_roles: str = field(default=_TR_USER_ROLES, repr=False)
    _tr_approval_workflow: str = field(default=_TR_APPROVAL_WORKFLOW, repr=False)
    _tr_communication_groups: str = field(default=_TR_COMMUNICATION_GROUPS, repr=False)
    _supports_tr_topic_gating: bool = field(default=True, repr=False)
    _section_title: str = field(default="INSPECTION WORKFLOW TABLES", repr=False)
    _has_ai_section: bool = field(default=True, repr=False)
    _ai_join_pattern: str = field(
        default=(
            "FROM ai_answers aa\n"
            "  LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
            "  JOIN inspection_report ir ON aa.inspection_report_id = ir.id"
        ),
        repr=False,
    )
    _status_enum_keys: tuple = field(
        default=(
            "inspection_report.status", "inspection_cycle.status", "inspection_schedule.status",
            "inspection_corrective_action.status", "inspection_corrective_action.progress_stage",
        ),
        repr=False,
    )

    def for_system_prompt(self) -> str:
        if not self._system_cache:
            self._system_cache = self._build_system_block()
        return self._system_cache

    def for_reasoning_prompt(self) -> str:
        """
        Compact schema block for the reasoning LLM's system prompt — curated
        key columns per table (not the full per-column dump _build_system_block
        produces), plus the lookup/JOIN-pattern section. Mirrors the token-budget
        fix already applied to the SQL model in for_sql_prompt_slim(): dumping
        full columns for all 29 tables on every call pushed the reasoning LLM's
        prompt to 19953/20000 num_ctx tokens, causing truncated tool-call JSON.
        """
        if not self._reasoning_cache:
            self._reasoning_cache = self._build_reasoning_block()
        return self._reasoning_cache

    def _build_reasoning_block(self) -> str:
        lines = [f"━━━ {self._section_title} — key columns ━━━", ""]
        for tname in self._domain_tables_list + self._ai_tables_list:
            cols = self._useful_columns.get(tname)
            if cols:
                lines.append(f"  {tname}: {', '.join(cols)}")
            else:
                lines.append(f"  {tname} (call get_schema for columns)")
        lines.append("")
        lines.append(self._fmt_lookup())
        if self._has_ai_section:
            lines.append("")
            lines.append(f"AI FLAT TABLES join pattern:\n  {self._ai_join_pattern}")
        return "\n".join(lines)

    # Same set for_sql_prompt_slim's FK-joins section filters to — pulled out
    # to a module-level constant so _slim_schema_parts and for_sql_prompt_slim
    # share one definition instead of two copies that could drift.
    _CRITICAL_FKS = frozenset({
        "risk_level_id", "impact_id", "facility_id", "inspection_type_id",
        "inspector_user_id", "frequency_definition_id", "portfolio_details_id",
    })

    def _slim_schema_parts(self) -> dict:
        """
        Structural breakdown behind for_sql_prompt_slim() — one rendered
        line per table/FK, keyed by name, in the same order
        for_sql_prompt_slim() emits them. Extracted as its own method so
        Phase 3.10's schema-block composition profiling
        (profile_slim_schema_tokens below) can tokenize each line
        individually without re-deriving this rendering logic a second
        time with room to drift from the real one — same pattern
        generation.py's _planner_contract_parts used for Phase 3.9.
        for_sql_prompt_slim() itself is unchanged output (still the
        identical joined string) and now just assembles these parts under
        their section headers.

        Returns {"key_columns": {table: line}, "lookup_tables": {table:
        line}, "fk_joins": {fk_col: line}} — rendered line strings, not the
        section header/label text for_sql_prompt_slim() wraps them in.
        """
        key_columns = {}
        for tname, cols in self._useful_columns.items():
            ev = self.enum_values.get(f"{tname}.status")
            status_note = f" | status: {', '.join(repr(v) for v in ev)}" if ev else ""
            key_columns[tname] = f"  {tname}: {', '.join(cols)}{status_note}"

        lookup_tables = {}
        for tname, info in self._lookup_config.items():
            ev = self.enum_values.get(tname)
            if ev and isinstance(ev[0], str):
                vals = ", ".join(repr(v) for v in ev[:12])
                lookup_tables[tname] = f"  {tname}.name → [{vals}]"
            elif ev and isinstance(ev[0], dict):
                names = ", ".join(repr(d["name"]) for d in ev[:8])
                lookup_tables[tname] = f"  {tname}.name → [{names}]"
            elif tname == "users":
                lookup_tables[tname] = f"  users: {_USERS_ID_NOTE}"
            else:
                lookup_tables[tname] = f"  {tname}.name (UUID FK)"

        fk_joins = {}
        for fk_col, (table, alias, sel) in self._fk_resolution.items():
            if fk_col in self._CRITICAL_FKS:
                fk_joins[fk_col] = f"  {fk_col} → JOIN {table} {alias} ON ...{fk_col}={alias}.id → {sel}"

        return {"key_columns": key_columns, "lookup_tables": lookup_tables, "fk_joins": fk_joins}

    def for_sql_prompt_slim(self) -> str:
        """
        Compact schema block for deepseek — target ~600 tokens.
        Shows curated key columns + live enum values + critical FKs.
        Does NOT dump the full introspected column list for every table —
        that inflates context 3x and degrades 7B model generation quality.
        """
        parts = self._slim_schema_parts()
        lines = ["### Schema — key columns (get_schema for full DDL):"]
        lines.extend(parts["key_columns"].values())
        lines.append("")
        lines.append("Lookup tables (JOIN by UUID FK, never compare UUID to string):")
        lines.extend(parts["lookup_tables"].values())
        lines.append("")
        lines.append("FK joins:")
        lines.extend(parts["fk_joins"].values())
        return "\n".join(lines)

    def profile_slim_schema_tokens(self, base_url: str) -> dict:
        """
        Phase 3.10: real /tokenize count for each line of the slim SQL-gen
        schema block, computed ONCE — not per question. Unlike schema_hint
        as a whole (3.9), for_sql_prompt_slim()'s content doesn't depend on
        the question at all: the same _useful_columns/_lookup_config/
        _fk_resolution lines render on every single call (only live enum
        values feed in, and those are fixed at DatabaseSchema construction,
        not re-fetched per question). So a one-time per-line breakdown is
        exact, not an approximation, for the whole eval run — versus
        ~29 x 103 repeated /tokenize calls for the same unchanging text.

        Callers (eval.py/eval_audit.py, gated behind --profile-tokens) diff
        this against each query's _diagnostics["generated_tables"] to see
        which included tables the generated SQL actually referenced.
        """
        from src.llm.tokenize import count_tokens_llamacpp
        parts = self._slim_schema_parts()
        key_columns_tok = {k: count_tokens_llamacpp(v, base_url) for k, v in parts["key_columns"].items()}
        lookup_tables_tok = {k: count_tokens_llamacpp(v, base_url) for k, v in parts["lookup_tables"].items()}
        fk_joins_tok = {k: count_tokens_llamacpp(v, base_url) for k, v in parts["fk_joins"].items()}
        return {
            "key_columns": key_columns_tok,
            "lookup_tables": lookup_tables_tok,
            "fk_joins": fk_joins_tok,
            "key_columns_total": sum(key_columns_tok.values()),
            "lookup_tables_total": sum(lookup_tables_tok.values()),
            "fk_joins_total": sum(fk_joins_tok.values()),
            "static_block_total": sum(key_columns_tok.values()) + sum(lookup_tables_tok.values()) + sum(fk_joins_tok.values()),
        }

    def profile_targeted_schema_tables(self, base_url: str) -> dict:
        """
        Phase 3.11: real /tokenize breakdown of for_schema_hint()'s per-
        table rendering (_fmt_table), computed ONCE for every table this
        domain knows about (self.tables | self.ai_tables |
        self.lookup_tables) — same invariant-content rationale as
        profile_slim_schema_tokens (3.10): _fmt_table(ts)'s output for a
        given table doesn't depend on which OTHER tables the reasoning
        model also named that call, so a per-table cost computed once here
        lets eval.py reconstruct any specific "targeted"-variant call's
        exact schema-block total via dict lookup on its named tables —
        zero additional /tokenize calls in the hot path, however many
        tables a call names.

        Also tokenizes _fmt_lookup() and self._table_relationships ONCE —
        both are appended by for_schema_hint() UNCONDITIONALLY, regardless
        of which/how-many tables were selected, so they're fixed overhead
        on every targeted call. This method exists specifically to answer
        3.11's hypothesis B: is the targeted branch's ~4x cost (vs the slim
        block) driven by per-table rendering richness, or by this fixed
        baseline alone?
        """
        from src.llm.tokenize import count_tokens_llamacpp
        all_tables = {**self.tables, **self.ai_tables, **self.lookup_tables}
        per_table = {}
        for tname, ts in all_tables.items():
            fmt_parts = self._fmt_table_parts(ts)
            plain_text = "\n".join(l for c, l in fmt_parts if c == "plain")
            fk_text = "\n".join(l for c, l in fmt_parts if c == "fk")
            enum_text = "\n".join(l for c, l in fmt_parts if c == "enum")
            useful_line = (f"Useful output columns: {', '.join(self._useful_columns[tname])}"
                           if tname in self._useful_columns else "")
            table_full = self._fmt_table(ts)
            columns_tok = count_tokens_llamacpp(plain_text, base_url)
            fk_tok = count_tokens_llamacpp(fk_text, base_url)
            enum_tok = count_tokens_llamacpp(enum_text, base_url)
            useful_tok = count_tokens_llamacpp(useful_line, base_url)
            table_total_tok = count_tokens_llamacpp(table_full, base_url)
            per_table[tname] = {
                "columns_tokens": columns_tok,
                "fk_tokens": fk_tok,
                "enum_tokens": enum_tok,
                "useful_output_tokens": useful_tok,
                # "### {name}\nColumns:" header + reconciliation slack —
                # same by-subtraction treatment as 3.9/3.10's boilerplate.
                "header_tokens": max(0, table_total_tok - columns_tok - fk_tok - enum_tok - useful_tok),
                "table_total_tokens": table_total_tok,
            }
        return {
            "per_table": per_table,
            "lookup_section_tokens": count_tokens_llamacpp(self._fmt_lookup(), base_url),
            # Phase 3.15: for_schema_hint() appends _table_relationships_sql_gen
            # (COMMUNICATION GROUPS/USER ROLES/APPROVAL WORKFLOW only), not the
            # full _table_relationships — tokenize what's actually rendered.
            # STALE as of Phase 3.16: for_schema_hint() now further topic-gates
            # this per-question (0-3 of the 3 sections, not always all 3), so
            # this fixed "computed once" value is only correct again for calls
            # where the question matches all 3 topic patterns. --profile-tokens
            # reconciliation (eval.py) was not reworked for per-call gating —
            # use plain (non-profiled) eval runs for Phase 3.16+ correctness
            # validation, not --profile-tokens's token-composition breakdown.
            "table_relationships_tokens": count_tokens_llamacpp(self._table_relationships_sql_gen, base_url),
        }

    def profile_fixed_overhead_relevance(self, base_url: str) -> dict:
        """
        Phase 3.12: structural breakdown of the two FIXED blocks 3.11
        identified as ~78% of every "targeted" for_schema_hint() call's
        cost — _fmt_lookup() (~979 tok) and self._table_relationships
        (~1922 tok), both appended in full regardless of which/how-many
        tables were named. Computed ONCE (same invariant-content rationale
        as profile_slim_schema_tokens/profile_targeted_schema_tables) —
        returns each sub-part's real token count AND its raw rendered text,
        so eval.py can check per-call relevance (does this sub-part mention
        any of THIS call's named/generated tables?) via a simple text
        search, with zero additional /tokenize calls in the hot path.

        lookup_tables/status_enums/join_patterns entries carry a "table"
        field (the one table each line is about) for a direct relevance
        check; relationship_sections entries don't (a prose section can
        mention several tables) — eval.py substring-searches "text" for
        those instead.
        """
        from src.llm.tokenize import count_tokens_llamacpp
        parts = self._fmt_lookup_parts()

        lookup_tables_tok = {
            tname: {"text": line, "tokens": count_tokens_llamacpp(line, base_url), "table": tname}
            for tname, line in parts["lookup_tables"].items()
        }
        status_enums_tok = {
            key: {"text": line, "tokens": count_tokens_llamacpp(line, base_url), "table": key.split(".")[0]}
            for key, line in parts["status_enums"].items()
        }
        join_patterns_tok = {
            fk_col: {
                "text": line, "tokens": count_tokens_llamacpp(line, base_url),
                "table": self._fk_resolution[fk_col][0],
            }
            for fk_col, line in parts["join_patterns"].items()
        }

        relationship_sections_tok = {}
        for title, body in _split_relationship_sections(self._table_relationships):
            full = f"{title}\n{body}"
            relationship_sections_tok[title] = {
                "text": full, "tokens": count_tokens_llamacpp(full, base_url),
            }

        return {
            "lookup_tables": lookup_tables_tok,
            "status_enums": status_enums_tok,
            "join_patterns": join_patterns_tok,
            "relationship_sections": relationship_sections_tok,
        }

    def _table_relationships_topic_gated(self, question: str | None) -> str:
        """
        Phase 3.16: per-question gate on TABLE_RELATIONSHIPS_SQL_GEN's 3
        sections, reusing the EXACT same question_pattern regexes
        agent/prompts/builder.py's CONDITIONAL_BLOCKS already uses to gate
        the identical content (user_roles/approval_workflow/communication_
        groups topical rules) into the reasoning-LLM's prompt — single
        source of truth in knowledge/question_patterns.py, so the SQL-gen
        and reasoning prompts agree on which niche topics a question
        touches instead of one being unconditional and the other gated.

        Falls back to the full, unconditional _table_relationships_sql_gen
        when no question is supplied (preserves exact prior behavior for
        callers that don't have one — e.g. agent/tools.py's get_schema()
        on-demand single-table lookup) or when the domain doesn't support
        gating (audit — AUDIT_TABLE_RELATIONSHIPS was never split into
        per-topic chunks, see the audit factory's _supports_tr_topic_gating
        override).

        Zero-match questions get "" — no size penalty, same as
        build_conditional_blocks() on the reasoning side. This is a real
        behavior change from 3.15's unconditional-3 baseline, not
        instrumentation — validate with the standard full-eval-suite
        real_pass/sql_ok/truth_pass gate before trusting it, same as every
        other behavior-changing phase in this project.
        """
        if not question or not self._supports_tr_topic_gating:
            return self._table_relationships_sql_gen
        matched = []
        if re.search(USER_ROLES_TOPIC, question):
            matched.append(self._tr_user_roles)
        if re.search(APPROVAL_WORKFLOW_TOPIC, question):
            matched.append(self._tr_approval_workflow)
        if re.search(COMMUNICATION_GROUPS_TOPIC, question):
            matched.append(self._tr_communication_groups)
        return "\n".join(matched)

    def for_schema_hint(self, table_names: list, question: str | None = None) -> str:
        """
        Phase 3.13: _fmt_lookup() is scoped to the named tables' own FK
        neighborhood — confirmed by 3.12's relevance profiling (real 103-
        question suite, real /tokenize + generated-SQL-table diff): of
        _fmt_lookup()'s ~979 tok, the cleanly per-table-attributable lines
        (lookup tables / status enums / FK join-patterns) were irrelevant
        to the actual generated SQL 87% of the time on average, and 12 of
        them (149 tok, constant every call) were relevant on ZERO of 52
        real targeted-branch calls. Additionally: for any table that IS
        named, its own FK columns already get full join-resolution info
        rendered inline by _fmt_table (see _fmt_table_parts's "fk" branch,
        which checks the SAME self._fk_resolution _fmt_lookup's "JOIN
        patterns:" section draws from) — so the unscoped join-patterns
        list was largely re-stating information already present elsewhere
        in the prompt for named tables, on top of listing irrelevant FKs
        for tables that weren't named at all.

        Phase 3.15: renders self._table_relationships_sql_gen (COMMUNICATION
        GROUPS / USER ROLES / APPROVAL WORKFLOW only) instead of the full
        self._table_relationships — 3.14's causal ablation (full 105-
        question suite, all 6 other sections removed at once: WHERE
        CONCEPTS LIVE, HOW TABLES CONNECT, TABLE COLUMNS, CRITICAL ai_*
        JOIN PATTERN, MULTI-HOP EXAMPLES, RULE: TYPE names) showed zero
        correctness regression — real_pass/sql_ok held, and the specific
        entity/country/organisation-hierarchy questions those sections
        document (FLT-08, JOI-13, JOI-15) still resolved correctly. See
        TABLE_RELATIONSHIPS_SQL_GEN for why the 3 kept sections weren't
        also cut.

        Phase 3.16: those 3 kept sections are now further gated per-question
        via _table_relationships_topic_gated(question) — see that method's
        docstring. `question` is optional (defaults to None, preserving the
        3.15 unconditional-3 behavior) since not every caller has the
        natural-language question available (agent/tools.py's get_schema()
        on-demand lookup does not).
        """
        parts = []
        for tname in table_names:
            ts = self.tables.get(tname) or self.ai_tables.get(tname) or self.lookup_tables.get(tname)
            if ts:
                parts.append(self._fmt_table(ts))
        parts.append(self._fmt_lookup(scope=self.targeted_lookup_scope(table_names)))
        parts.append(self._table_relationships_topic_gated(question))
        return "\n".join(parts)

    def targeted_lookup_scope(self, table_names: list) -> dict:
        """
        The _fmt_lookup() scope for_schema_hint() uses for these
        table_names — extracted as its own method so eval.py's Phase 3.13
        validation (comparing each targeted call's real vs. reconstructed
        cost using Phase 3.12's per-item costs) can compute the identical
        scope without re-deriving the FK-neighbor logic a second time.
        """
        relevant_lookup_tables: set = set()
        relevant_fk_cols: set = set()
        for tname in table_names:
            ts = self.tables.get(tname) or self.ai_tables.get(tname) or self.lookup_tables.get(tname)
            if ts:
                for fk in ts.foreign_keys:
                    relevant_fk_cols.add(fk["column"])
                    relevant_lookup_tables.add(fk["referred_table"])
        return {
            "lookup_tables": relevant_lookup_tables,
            "status_enums": set(table_names),
            "join_patterns": relevant_fk_cols,
        }

    def _fmt_table_parts(self, ts: TableSchema) -> list[tuple[str, str]]:
        """
        Ordered (category, rendered_line) pairs behind _fmt_table's per-
        column rendering — category is one of "plain"/"fk"/"enum", in the
        same order and with the identical line text _fmt_table's inline
        loop produced. Extracted as its own method so Phase 3.11's
        targeted-branch (for_schema_hint) composition profiling
        (profile_targeted_schema_tables below) can tokenize each category
        separately without duplicating this column-rendering logic in a
        second place — same pattern _slim_schema_parts (3.10) and
        _planner_contract_parts (3.9) used. _fmt_table itself is unchanged
        output and now just extracts the line text from these pairs.
        """
        parts: list[tuple[str, str]] = []
        for col in ts.columns:
            col_name = col["name"]

            # Skip system/audit columns and known noisy columns
            if col_name in self._suppress_columns:
                continue

            ctype = str(col["type"]).lower()
            for long, short in [
                ("character varying", "varchar"),
                ("timestamp without time zone", "timestamp"),
                ("timestamp with time zone", "timestamptz"),
                ("double precision", "float"),
                ("boolean", "bool"),
            ]:
                ctype = ctype.replace(long, short)

            full_key = f"{ts.name}.{col_name}"

            if full_key in self._enum_columns:
                vals = self._enum_columns[full_key]
                if vals:
                    parts.append(("enum",
                        f"  {col_name} ({ctype} — ENUM: {', '.join(repr(v) for v in vals)})"))
                else:
                    parts.append(("enum", f"  {col_name} ({ctype} — plain string)"))
                continue

            fk = next((f for f in ts.foreign_keys if f["column"] == col_name), None)
            if fk:
                ref = fk["referred_table"]
                is_opt = col_name in self._optional_fks
                jtype = "LEFT JOIN" if is_opt else "JOIN"
                if col_name in self._fk_resolution:
                    _, alias, sel = self._fk_resolution[col_name]
                    parts.append(("fk",
                        f"  {col_name} (uuid FK → {ref}. "
                        f"{jtype} {ref} {alias} ON ...{col_name} = {alias}.id → {sel})"))
                else:
                    note = " — may be NULL, use LEFT JOIN" if is_opt else ""
                    parts.append(("fk", f"  {col_name} (uuid FK → {ref}{note})"))
            else:
                parts.append(("plain", f"  {col_name} ({ctype})"))

        return parts

    def _fmt_table(self, ts: TableSchema) -> str:
        lines = [f"### {ts.name}", "Columns:"]
        lines.extend(line for _, line in self._fmt_table_parts(ts))
        if ts.name in self._useful_columns:
            lines.append(f"Useful output columns: {', '.join(self._useful_columns[ts.name])}")
        return "\n".join(lines)

    def _fmt_ai_section(self) -> str:
        lines = ["", "### AI FLAT TABLES — form questions and answers:", ""]
        for tname in self._ai_tables_list:
            ts = self.ai_tables.get(tname)
            if ts:
                lines.append(self._fmt_table(ts))
                lines.append("")
        lines.append(f"Standard JOIN:\n  {self._ai_join_pattern}")
        return "\n".join(lines)

    def _fmt_lookup_parts(self) -> dict:
        """
        Structural breakdown behind _fmt_lookup() — one rendered line (or
        two-line block, for "users") per lookup table / status-enum key /
        FK join pattern, keyed by name, in the same order _fmt_lookup()'s
        three inline loops produce. Extracted for Phase 3.12's fixed-
        overhead relevance profiling (see profile_fixed_overhead_relevance)
        — _fmt_lookup() itself is unchanged output and now just assembles
        these parts under its header/section labels.

        Returns {"lookup_tables": {table: line}, "status_enums":
        {status_key: line}, "join_patterns": {fk_col: line}}.
        """
        lookup_tables = {}
        for tname, info in self._lookup_config.items():
            ts = self.lookup_tables.get(tname)
            if not ts:
                continue
            expr = info["name_expr"] or info["name_col"]
            ev = self.enum_values.get(tname)
            if ev and isinstance(ev[0], str):
                vals = ", ".join(repr(v) for v in ev[:20])  # cap at 20
                line = f"  {tname}: {expr}  [values: {vals}]"
            elif ev and isinstance(ev[0], dict):
                # frequency_definition — show names only in the lookup section
                names = ", ".join(repr(d["name"]) for d in ev[:10])
                line = f"  {tname}: {expr}  [definitions: {names}]"
            else:
                line = f"  {tname}: {expr}"
            if tname == "users":
                line += f"\n    ⚠ {_USERS_ID_NOTE}"
            lookup_tables[tname] = line

        status_enums = {}
        for col_key in self._status_enum_keys:
            vals = self.enum_values.get(col_key)
            if vals:
                status_enums[col_key] = f"  {col_key}: {', '.join(repr(v) for v in vals)}"

        join_patterns = {}
        for fk_col, (table, alias, sel) in self._fk_resolution.items():
            join_patterns[fk_col] = f"  {fk_col} → JOIN {table} {alias} ON ...{fk_col} = {alias}.id → {sel}"

        return {"lookup_tables": lookup_tables, "status_enums": status_enums, "join_patterns": join_patterns}

    def _fmt_lookup(self, scope: dict | None = None) -> str:
        """
        scope=None (the default — used by for_reasoning_prompt/
        for_system_prompt, both unaffected by Phase 3.13) renders every
        lookup table / status enum / FK join-pattern, unchanged from
        before. scope={"lookup_tables": {...}, "status_enums": {...},
        "join_patterns": {...}} (for_schema_hint's Phase 3.13 use) keeps
        only entries whose table (lookup_tables/status_enums, matched by
        table name) or FK column (join_patterns) is in the corresponding
        set — status_enums matches on the table name before the "."
        (e.g. "inspection_report" for the "inspection_report.status" key).
        """
        parts = self._fmt_lookup_parts()
        lookup_items = parts["lookup_tables"]
        status_items = parts["status_enums"]
        join_items = parts["join_patterns"]
        if scope is not None:
            lookup_items = {k: v for k, v in lookup_items.items() if k in scope.get("lookup_tables", ())}
            status_items = {k: v for k, v in status_items.items()
                            if k.split(".")[0] in scope.get("status_enums", ())}
            join_items = {k: v for k, v in join_items.items() if k in scope.get("join_patterns", ())}
        lines = [
            "", "### UUID RESOLUTION:",
            "NEVER show raw UUIDs. Always JOIN to lookup tables.",
            "'responsible', 'pending_with', 'status' are string enums — NOT FKs.",
            "", "Lookup tables:",
        ]
        lines.extend(lookup_items.values())
        lines.extend(status_items.values())
        lines.append("")
        lines.append("JOIN patterns:")
        lines.extend(join_items.values())
        return "\n".join(lines)

    def _build_system_block(self) -> str:
        lines = [f"━━━ {self._section_title} ━━━", ""]
        for tname in self._domain_tables_list:
            ts = self.tables.get(tname)
            if ts:
                lines.append(self._fmt_table(ts))
                lines.append("")
        lines.append(self._fmt_lookup())
        if self._has_ai_section:
            lines.append(self._fmt_ai_section())
        lines.append("")
        lines.append("BUSINESS RULES:")
        for rule in self._business_rules:
            lines.append(f"  - {rule}")
        lines.append("")
        lines.append(self._table_relationships)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Introspector
# ---------------------------------------------------------------------------

_INSPECTION_ENUM_QUERIES = {
    # Lookup tables — query their name column
    "risk_level":           ("SELECT name FROM risk_level ORDER BY name", "name"),
    "impact":               ("SELECT name FROM impact ORDER BY name", "name"),
    "inspection_type":      ("SELECT name FROM inspection_type ORDER BY name", "name"),
    "frequency_definition": (
        "SELECT name, repeat_count, repeat_interval, repeat_unit "
        "FROM frequency_definition ORDER BY name", "name"
    ),
    # Enum columns on main tables — query DISTINCT values
    "inspection_report.status": (
        "SELECT DISTINCT status FROM inspection_report WHERE status IS NOT NULL ORDER BY status",
        "status"
    ),
    "inspection_cycle.status": (
        "SELECT DISTINCT status FROM inspection_cycle WHERE status IS NOT NULL ORDER BY status",
        "status"
    ),
    "inspection_schedule.status": (
        "SELECT DISTINCT status FROM inspection_schedule WHERE status IS NOT NULL ORDER BY status",
        "status"
    ),
    # inspection_corrective_action's own status columns — same gap/fix as the
    # audit domain's equivalent addition to _AUDIT_ENUM_QUERIES above.
    "inspection_corrective_action.status": (
        "SELECT DISTINCT status FROM inspection_corrective_action WHERE status IS NOT NULL ORDER BY status",
        "status"
    ),
    "inspection_corrective_action.progress_stage": (
        "SELECT DISTINCT progress_stage FROM inspection_corrective_action "
        "WHERE progress_stage IS NOT NULL ORDER BY progress_stage", "progress_stage"
    ),
}
# Keys whose query returns multiple columns and should be stored as a list of
# dicts rather than a flat list of the "name" column — frequency_definition is
# the only inspection-domain case today.
_MULTI_COL_ENUM_KEYS = frozenset({"frequency_definition"})


def _introspect(
    db_engine: Engine,
    *,
    domain_tables: list,
    ai_tables: list,
    lookup_tables: dict,
    enum_queries: dict,
    multi_col_keys: frozenset = frozenset(),
    schema_kwargs: dict | None = None,
    label: str = "schema",
) -> DatabaseSchema:
    """Shared introspection mechanism — table/FK/enum discovery is identical
    between domains, only which tables/queries to run differs. See
    introspect_schema() / introspect_audit_schema() below."""
    print(f"  Introspecting {label}...")
    inspector = sa_inspect(db_engine)
    schema = DatabaseSchema(**(schema_kwargs or {}))
    all_tables = set(inspector.get_table_names())

    def _do(tname) -> Optional[TableSchema]:
        if tname not in all_tables:
            return None
        try:
            columns = inspector.get_columns(tname)
            fks_raw = inspector.get_foreign_keys(tname)
            pk = inspector.get_pk_constraint(tname)
            fks = []
            for fk in fks_raw:
                for i, col in enumerate(fk.get("constrained_columns", [])):
                    ref_cols = fk.get("referred_columns", [])
                    fks.append({
                        "column": col,
                        "referred_table": fk.get("referred_table", ""),
                        "referred_column": ref_cols[i] if i < len(ref_cols) else "id",
                    })
            return TableSchema(
                name=tname,
                columns=[{"name": c["name"], "type": str(c["type"]),
                           "nullable": c.get("nullable", True)} for c in columns],
                foreign_keys=fks,
                primary_key=pk.get("constrained_columns", []) if pk else [],
            )
        except Exception as e:
            print(f"    ⚠ Could not introspect {tname}: {e}")
            return None

    for tname in domain_tables:
        ts = _do(tname)
        if ts:
            schema.tables[tname] = ts

    for tname in ai_tables:
        ts = _do(tname)
        if ts:
            schema.ai_tables[tname] = ts
        else:
            print(f"    ⚠ AI table '{tname}' not found in database")

    for tname in lookup_tables:
        ts = _do(tname)
        if ts:
            schema.lookup_tables[tname] = ts

    # ── Dynamic enum extraction ───────────────────────────────────────────────
    # Query actual values from lookup tables and key enum columns at startup.
    # This means new risk_level tiers, inspection types, frequency definitions,
    # etc. are automatically visible without any code changes.
    try:
        from sqlalchemy import text as sa_text
        with db_engine.connect() as conn:
            for key, (query, col) in enum_queries.items():
                try:
                    rows = conn.execute(sa_text(query)).fetchall()
                    if key in multi_col_keys:
                        # Store full row for richer context (e.g. frequency_definition's
                        # repeat_count/repeat_interval/repeat_unit alongside name)
                        cols = list(rows[0]._mapping.keys()) if rows else [col]
                        schema.enum_values[key] = [dict(zip(cols, r)) for r in rows]
                    else:
                        schema.enum_values[key] = [r[0] for r in rows if r[0]]
                except Exception:
                    pass  # Non-fatal: hardcoded values in ENUM_COLUMNS still apply
    except Exception:
        pass  # DB not available — degrade gracefully

    print(
        f"  ✓ {label.title()}: {len(schema.tables)} domain tables, "
        f"{len(schema.ai_tables)} ai tables, "
        f"{len(schema.lookup_tables)} lookup tables"
    )
    if schema.enum_values:
        extracted = ", ".join(
            f"{k}={len(v)}" for k, v in schema.enum_values.items()
            if v and isinstance(v[0], str)
        )
        print(f"  ✓ Dynamic enums: {extracted}")
    return schema


def introspect_schema(db_engine: Engine) -> DatabaseSchema:
    """Inspection-domain introspection — unchanged behavior from before this
    module supported multiple domains (all params/defaults match exactly)."""
    return _introspect(
        db_engine,
        domain_tables=INSPECTION_TABLES,
        ai_tables=AI_TABLES,
        lookup_tables=LOOKUP_TABLES,
        enum_queries=_INSPECTION_ENUM_QUERIES,
        multi_col_keys=_MULTI_COL_ENUM_KEYS,
        label="inspection schema",
        # DatabaseSchema()'s field defaults already ARE the inspection globals —
        # no schema_kwargs override needed.
    )


def introspect_audit_schema(db_engine: Engine) -> DatabaseSchema:
    """Audit-domain introspection — same mechanism as introspect_schema(),
    AUDIT_* content instead. See DOMAIN_REGISTRY in agent/domains.py."""
    return _introspect(
        db_engine,
        domain_tables=AUDIT_TABLES,
        ai_tables=AUDIT_AI_TABLES,
        lookup_tables=AUDIT_LOOKUP_TABLES,
        enum_queries=_AUDIT_ENUM_QUERIES,
        multi_col_keys=frozenset(),
        label="audit schema",
        schema_kwargs=dict(
            _domain_tables_list=list(AUDIT_TABLES),
            _ai_tables_list=list(AUDIT_AI_TABLES),
            _lookup_config=dict(AUDIT_LOOKUP_TABLES),
            _fk_resolution=dict(AUDIT_FK_RESOLUTION),
            _optional_fks=set(),
            _useful_columns=dict(AUDIT_USEFUL_COLUMNS),
            _enum_columns=dict(AUDIT_ENUM_COLUMNS),
            _suppress_columns=set(_AUDIT_SUPPRESS_COLUMNS),
            _business_rules=list(AUDIT_BUSINESS_RULES),
            _table_relationships=AUDIT_TABLE_RELATIONSHIPS,
            _business_rules_core=list(AUDIT_BUSINESS_RULES_CORE),
            _table_relationships_core=AUDIT_TABLE_RELATIONSHIPS_CORE,
            # Audit's for_schema_hint() keeps the full block — unlike
            # inspection's, AUDIT_TABLE_RELATIONSHIPS was never split (see
            # the "left fully core" note above) and wasn't part of 3.14's
            # ablation, so there's no evidence any of it is safe to drop.
            _table_relationships_sql_gen=AUDIT_TABLE_RELATIONSHIPS,
            # Phase 3.16: audit has no per-topic TR chunks to gate on (see
            # above) — disable gating so _table_relationships_topic_gated()
            # always falls back to the unconditional full block regardless
            # of question, same as before 3.16 existed.
            _tr_user_roles="", _tr_approval_workflow="", _tr_communication_groups="",
            _supports_tr_topic_gating=False,
            _section_title="AUDIT WORKFLOW TABLES",
            _has_ai_section=True,
            _ai_join_pattern=(
                "FROM ai_audit_answers aaa\n"
                "  LEFT JOIN ai_questions aq ON aaa.element_id = aq.element_id\n"
                "  JOIN audit_report arep ON aaa.audit_report_id = arep.id"
            ),
            _status_enum_keys=(
                "audit_plan.status", "audit_schedule.schedule_status", "audit_report.status",
                "audit_corrective_action.status", "audit_corrective_action.implementation_status",
                "audit_corrective_action.adequacy_status", "audit_corrective_action.progress_stage",
                "audit_forms.status",
            ),
        ),
    )