"""
AUDIT_SEED_EXAMPLES — NL→SQL few-shot pairs for the audit domain, same dict
shape as sql/examples/seed_data.py's SEED_EXAMPLES ({'nl','sql','domain',
'intent','limit'}), consumed by a separate SeedExampleIndex instance (see
agent/domains.py's DOMAIN_REGISTRY), never merged with inspection's.

Starting set (~30 examples) proving the pipeline end-to-end and covering the
traps confirmed live against the DB — grows organically from real traffic the
same way inspection's did, per the domain-router plan's stated scope.
"""

AUDIT_SEED_EXAMPLES: list[dict] = [
    # ── audit_schedule ───────────────────────────────────────────────────────
    {
        'nl':     'list all audit schedules with auditor auditee facility and status',
        'sql':    """\
SELECT asch.schedule_id, asch.schedule_status, asch.duration, asch.follow_up_date,
       fac.name AS facility_name,
       u_aud.first_name || ' ' || u_aud.last_name AS auditor_name,
       u_ee.first_name || ' ' || u_ee.last_name AS auditee_name
FROM audit_schedule asch
LEFT JOIN facility fac ON asch.facility_id = fac.id
LEFT JOIN users u_aud ON asch.auditor_id = u_aud.id
LEFT JOIN users u_ee ON asch.auditee_id = u_ee.id
WHERE asch.active = true
ORDER BY asch.follow_up_date DESC NULLS LAST
LIMIT 100;
""",
        'domain': 'audit_schedule', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'which auditor has the most audit schedules assigned',
        'sql':    """\
SELECT u.first_name || ' ' || u.last_name AS auditor_name, COUNT(*) AS schedule_count
FROM audit_schedule asch
JOIN users u ON asch.auditor_id = u.id
WHERE asch.active = true AND u.active = true
GROUP BY u.id, u.first_name, u.last_name
ORDER BY schedule_count DESC
LIMIT 10;
""",
        'domain': 'audit_schedule', 'intent': 'ranked_list', 'limit': 10,
    },
    {
        'nl':     'show overdue audit schedules pending review not yet approved',
        'sql':    """\
-- 'pending review' = schedule_status = 'PENDING_REVIEW' — NEVER audit_status (dead column)
SELECT asch.schedule_id, asch.schedule_status, asch.follow_up_date,
       fac.name AS facility_name
FROM audit_schedule asch
LEFT JOIN facility fac ON asch.facility_id = fac.id
WHERE asch.schedule_status = 'PENDING_REVIEW' AND asch.active = true
ORDER BY asch.follow_up_date ASC NULLS LAST
LIMIT 100;
""",
        'domain': 'audit_schedule', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'who audited this facility last month which auditor conducted the audit in a date range',
        'sql':    """\
-- audit_schedule.follow_up_date is the schedule's own date — use THIS for 'last month'/date-range
-- questions about when a specific audit happened. audit_schedule has NO 'due_date' column.
-- NEVER use audit_cycle.due_date here — that's the whole CYCLE's period deadline, a different
-- concept, reached via a different 3-hop path (audit_plan_id -> audit_plan.cycle_id -> audit_cycle) —
-- substituting it for 'when this schedule happened' answers the wrong question and silently
-- returns 0 or the wrong rows.
SELECT u.first_name || ' ' || u.last_name AS auditor_name, asch.schedule_id, asch.follow_up_date
FROM audit_schedule asch
JOIN facility fac ON asch.facility_id = fac.id
JOIN users u ON asch.auditor_id = u.id
WHERE fac.name ILIKE '%[FACILITY]%'
  AND asch.follow_up_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
  AND asch.follow_up_date <  DATE_TRUNC('month', CURRENT_DATE)
  AND asch.active = true AND fac.active = true AND u.active = true
ORDER BY asch.follow_up_date DESC
LIMIT 100;
""",
        'domain': 'audit_schedule', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'audit schedules where auditee requested a reschedule',
        'sql':    """\
-- reschedule_requested/reschedule_count reflect the auditee's SCHEDULE-level
-- accept/reschedule review step, not a separate workflow.
SELECT asch.schedule_id, asch.follow_up_date, asch.reschedule_count,
       fac.name AS facility_name,
       u_ee.first_name || ' ' || u_ee.last_name AS auditee_name
FROM audit_schedule asch
LEFT JOIN facility fac ON asch.facility_id = fac.id
LEFT JOIN users u_ee ON asch.auditee_id = u_ee.id
WHERE asch.reschedule_requested = true AND asch.active = true
ORDER BY asch.follow_up_date ASC NULLS LAST
LIMIT 100;
""",
        'domain': 'audit_schedule', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'list unplanned ad hoc audits that were added outside the plan',
        'sql':    """\
-- is_unplanned exists on BOTH audit_schedule and audit_plan — same name, two
-- different levels. This is the SCHEDULE-level flag (an ad hoc visit added
-- outside an approved plan); audit_plan.is_unplanned marks the plan itself
-- as unplanned, a separate and much rarer case. Don't conflate the two.
SELECT asch.schedule_id, asch.follow_up_date, fac.name AS facility_name
FROM audit_schedule asch
LEFT JOIN facility fac ON asch.facility_id = fac.id
WHERE asch.is_unplanned = true AND asch.active = true
ORDER BY asch.follow_up_date DESC NULLS LAST
LIMIT 100;
""",
        'domain': 'audit_schedule', 'intent': 'filter', 'limit': 100,
    },

    {
        'nl':     'which auditor has the most overdue corrective actions assigned to their audits',
        'sql':    """\
-- aca.audit_id is a UUID FK to audit_report.id — NEVER audit_report.audit_id (that's the
-- VARCHAR display code, a different column despite the shared name) and NEVER directly to
-- audit_schedule.id (silently returns 0 rows — both happen to be UUID-typed so it doesn't error,
-- but audit_report is a mandatory intermediate hop to reach the auditor).
SELECT u.first_name || ' ' || u.last_name AS auditor_name, COUNT(*) AS overdue_ca_count
FROM audit_corrective_action aca
JOIN audit_report arep ON aca.audit_id = arep.id
JOIN audit_schedule asch ON arep.schedule_id = asch.id
JOIN users u ON asch.auditor_id = u.id
WHERE aca.status = 'OVERDUE' AND aca.active = true AND arep.active = true AND asch.active = true
  AND aca.adequacy_status = 'APPROVE'
GROUP BY u.id, u.first_name, u.last_name
ORDER BY overdue_ca_count DESC
LIMIT 10;
""",
        'domain': 'audit_corrective_action', 'intent': 'ranked_list', 'limit': 10,
    },

    # ── facility/project multi-hop trap (audit_report has no facility_id) ──
    {
        'nl':     'show audit reports with their facility and project',
        'sql':    """\
-- audit_report has NO facility_id/project_id of its own — two-hop via audit_schedule
SELECT arep.audit_id, arep.status, arep.audit_score, arep.submitted_on,
       fac.name AS facility_name, proj.name AS project_name
FROM audit_report arep
JOIN audit_schedule asch ON arep.schedule_id = asch.id
LEFT JOIN facility fac ON asch.facility_id = fac.id
LEFT JOIN project proj ON asch.project_id = proj.id
WHERE arep.active = true AND asch.active = true
ORDER BY arep.submitted_on DESC NULLS LAST
LIMIT 100;
""",
        'domain': 'audit_report', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'audits with no corrective actions raised against them',
        'sql':    """\
-- EXISTENCE check, not a listing — do NOT reuse the CA_DRAFT-exclusion default
-- here. A 'CA_DRAFT' row is a REAL corrective action that WAS raised, just not
-- yet adequacy-approved — it still counts as "raised". Check row existence
-- (aca.id IS NULL) only; never adequacy_status or corrective_action_id, which
-- would wrongly count "raised but not yet approved" as "never raised" and
-- inflate this result with audits that DO have one raised (confirmed live:
-- adequacy_status-based filtering here overcounts by the number of audits
-- whose only CA is still pending approval).
SELECT arep.audit_id, fac.name AS facility_name
FROM audit_report arep
JOIN audit_schedule asch ON arep.schedule_id = asch.id
LEFT JOIN facility fac ON asch.facility_id = fac.id
LEFT JOIN audit_corrective_action aca ON aca.audit_id = arep.id AND aca.active = true
WHERE arep.active = true AND asch.active = true AND aca.id IS NULL
ORDER BY arep.submitted_on DESC
LIMIT 100;
""",
        'domain': 'audit_report', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'corrective actions with their facility name',
        'sql':    """\
-- Three hops from a corrective action to facility — audit_report has no facility_id either
SELECT aca.corrective_action_id, aca.status, aca.cause, fac.name AS facility_name
FROM audit_corrective_action aca
JOIN audit_report arep ON aca.audit_id = arep.id
JOIN audit_schedule asch ON arep.schedule_id = asch.id
LEFT JOIN facility fac ON asch.facility_id = fac.id
WHERE aca.active = true AND arep.active = true AND asch.active = true
  AND aca.adequacy_status = 'APPROVE'
ORDER BY aca.created_on DESC
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'list all corrective actions with their audit form title',
        'sql':    """\
-- audit_report has NO title column, and audit_form_used_id is NOT on
-- audit_corrective_action either — two-hop via audit_report to audit_forms.
SELECT aca.corrective_action_id, aca.status, aca.implementation_status,
       aform.title AS form_title
FROM audit_corrective_action aca
JOIN audit_report arep ON aca.audit_id = arep.id
JOIN audit_forms aform ON arep.audit_form_used_id = aform.id
WHERE aca.active = true AND arep.active = true
  AND aca.adequacy_status = 'APPROVE'
ORDER BY aca.created_on DESC
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },

    # ── audit type 3-hop trap ────────────────────────────────────────────────
    {
        'nl':     'audit reports with their audit type internal external statutory',
        'sql':    """\
-- audit_report.audit_type_id is DEAD (always NULL) — real path is via audit_plan
SELECT arep.audit_id, arep.status, aty.name AS audit_type_name
FROM audit_report arep
JOIN audit_schedule asch ON arep.schedule_id = asch.id
JOIN audit_plan ap ON asch.audit_plan_id = ap.id
JOIN audit_type aty ON ap.audit_type_id = aty.id
WHERE arep.active = true AND asch.active = true AND ap.active = true
ORDER BY arep.submitted_on DESC NULLS LAST
LIMIT 100;
""",
        'domain': 'audit_report', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'how many audit schedules of each audit type',
        'sql':    """\
SELECT aty.name AS audit_type_name, COUNT(*) AS schedule_count
FROM audit_schedule asch
JOIN audit_plan ap ON asch.audit_plan_id = ap.id
JOIN audit_type aty ON ap.audit_type_id = aty.id
WHERE asch.active = true AND ap.active = true
GROUP BY aty.id, aty.name
ORDER BY schedule_count DESC;
""",
        'domain': 'audit_schedule', 'intent': 'aggregate', 'limit': None,
    },
    {
        'nl':     'how many audit schedules for each sub-category',
        'sql':    """\
SELECT asub.name AS sub_category_name, COUNT(*) AS schedule_count
FROM audit_schedule asch
JOIN audit_sub_category asub ON asch.audit_sub_category_id = asub.id
WHERE asch.active = true AND asub.active = true
GROUP BY asub.id, asub.name
ORDER BY schedule_count DESC;
""",
        'domain': 'audit_schedule', 'intent': 'aggregate', 'limit': None,
    },
    {
        'nl':     'which audit universe has the most audit plans',
        'sql':    """\
-- audit_plan has NO audit_universe_id and NO direct FK to audit_universe at
-- all — confirmed live the only wrong-but-executable shortcut is joining
-- ap.audit_type_id straight to auniv.id (both happen to be UUID-typed so it
-- doesn't error, it just silently matches nothing and returns 0 rows for
-- every universe). The only valid path is via audit_schedule, which carries
-- BOTH audit_universe_id and audit_plan_id.
SELECT auniv.name AS universe_name, COUNT(DISTINCT ap.id) AS plan_count
FROM audit_universe auniv
JOIN audit_schedule asch ON asch.audit_universe_id = auniv.id
JOIN audit_plan ap ON asch.audit_plan_id = ap.id
WHERE ap.active = true AND asch.active = true AND auniv.active = true
GROUP BY auniv.id, auniv.name
ORDER BY plan_count DESC
LIMIT 10;
""",
        'domain': 'audit_universe', 'intent': 'aggregate', 'limit': 10,
    },

    # ── audit_category / audit_sub_category taxonomy ─────────────────────────
    {
        'nl':     'list all audit sub-categories with their parent category and priority',
        'sql':    """\
-- audit_sub_category rolls up to audit_category ONLY (audit_category_id) —
-- it has no audit_universe_id of its own. audit_universe and audit_category
-- are two INDEPENDENT classification dimensions stored separately on
-- audit_schedule/audit_forms; sub-category does not imply a universe.
SELECT acat.name AS category_name, asub.name AS sub_category_name, asub.priority
FROM audit_sub_category asub
JOIN audit_category acat ON asub.audit_category_id = acat.id
WHERE asub.active = true AND acat.active = true
ORDER BY acat.name, asub.name
LIMIT 100;
""",
        'domain': 'audit_category', 'intent': 'filter', 'limit': 100,
    },

    # ── facility geography (region_id/organization_id mostly dead) ──────────
    {
        'nl':     'how many audit schedules per country',
        'sql':    """\
-- facility.region_id is populated for only a handful of rows and
-- organization_id is always NULL (dead) — country_id is the one geography
-- FK that's reliably populated. Use it for any "by region/location" ask.
SELECT co.name AS country_name, COUNT(*) AS schedule_count
FROM audit_schedule asch
JOIN facility fac ON asch.facility_id = fac.id
JOIN country co ON fac.country_id = co.id
WHERE asch.active = true AND fac.active = true
GROUP BY co.id, co.name
ORDER BY schedule_count DESC;
""",
        'domain': 'facility', 'intent': 'aggregate', 'limit': None,
    },

    # ── audit_plan ───────────────────────────────────────────────────────────
    {
        'nl':     'list all audit plans with status proposer cycle and category',
        'sql':    """\
SELECT ap.title, ap.status, acyc.start_date, acyc.end_date,
       acat.name AS category_name,
       u.first_name || ' ' || u.last_name AS proposed_by
FROM audit_plan ap
JOIN audit_cycle acyc ON ap.cycle_id = acyc.id
LEFT JOIN audit_category acat ON ap.category_id = acat.id
LEFT JOIN users u ON ap.user_id = u.id
WHERE ap.active = true
ORDER BY acyc.start_date DESC
LIMIT 100;
""",
        'domain': 'audit_plan', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'audit plans pending approval awaiting review not yet approved',
        'sql':    """\
SELECT ap.title, ap.status, acyc.start_date
FROM audit_plan ap
JOIN audit_cycle acyc ON ap.cycle_id = acyc.id
WHERE ap.status = 'SUBMITTED' AND ap.active = true
ORDER BY acyc.start_date ASC
LIMIT 100;
""",
        'domain': 'audit_plan', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'which audit plans were returned for modification',
        'sql':    """\
-- status = 'RETURNED_FOR_MODIFICATION' is a direct column value on audit_plan
-- itself (confirmed live: 2 rows) — simpler than going via the review table
-- when the question doesn't ask WHO rejected it, just which plans. For
-- "who rejected this plan and when", use review instead: entity_type='AUDIT_PLAN'
-- AND action='RETURN_FOR_MODIFICATION' (see the review-table examples above).
SELECT ap.title, ap.status, acyc.start_date, acyc.end_date
FROM audit_plan ap
JOIN audit_cycle acyc ON ap.cycle_id = acyc.id
WHERE ap.status = 'RETURNED_FOR_MODIFICATION' AND ap.active = true
ORDER BY acyc.start_date DESC
LIMIT 100;
""",
        'domain': 'audit_plan', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'who proposed the most audit plans',
        'sql':    """\
-- audit_plan.user_id is the proposer (confirmed live: always held by the
-- 'Internal Auditor' role in this dataset, but query the FK directly —
-- never filter by role name to find proposers).
SELECT u.first_name || ' ' || u.last_name AS proposer_name, COUNT(*) AS plan_count
FROM audit_plan ap
JOIN users u ON ap.user_id = u.id
WHERE ap.active = true AND u.active = true
GROUP BY u.id, u.first_name, u.last_name
ORDER BY plan_count DESC
LIMIT 10;
""",
        'domain': 'audit_plan', 'intent': 'ranked_list', 'limit': 10,
    },
    {
        'nl':     'how many schedules were created from each audit plan',
        'sql':    """\
SELECT ap.title, COUNT(asch.id) AS schedule_count
FROM audit_plan ap
LEFT JOIN audit_schedule asch ON asch.audit_plan_id = ap.id AND asch.active = true
WHERE ap.active = true
GROUP BY ap.id, ap.title
ORDER BY schedule_count DESC
LIMIT 100;
""",
        'domain': 'audit_plan', 'intent': 'aggregate', 'limit': 100,
    },
    {
        'nl':     'how many unplanned audit plans were raised',
        'sql':    """\
SELECT COUNT(*) AS unplanned_count
FROM audit_plan
WHERE is_unplanned = true AND active = true;
""",
        'domain': 'audit_plan', 'intent': 'aggregate', 'limit': None,
    },

    # ── review table — polymorphic approval workflow ────────────────────────
    {
        'nl':     'show pending level 2 reviews for audit plans awaiting audit in-charge approval',
        'sql':    """\
-- review.entity_id is POLYMORPHIC — always filter entity_type alongside entity_id.
-- AUDIT_PLAN has 2 review levels; level 2 is the final approval step.
-- Reviewer role is NOT hardcoded — it is dynamic/permission-driven, join to see who it's assigned to.
SELECT ap.title, rev.level, rev.status, rev.reviewed_on,
       u.first_name || ' ' || u.last_name AS assigned_to
FROM review rev
JOIN audit_plan ap ON rev.entity_id = ap.id AND rev.entity_type = 'AUDIT_PLAN'
LEFT JOIN users u ON rev.assigned_to_id = u.id
WHERE rev.level = 2 AND rev.status = 'PENDING' AND ap.active = true
ORDER BY ap.created_on ASC
LIMIT 100;
""",
        'domain': 'review', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'who approved this auditor portfolio and when',
        'sql':    """\
-- PORTFOLIO review is single-level, always level 1.
-- status='COMPLETED' alone is NOT enough — a completed review can be a rejection
-- (action='RETURN_FOR_MODIFICATION'), confirmed live (28% of completed reviews overall
-- are rejections). 'who APPROVED' must also filter action = 'APPROVE'.
-- auditor_portfolio has NO display column of its own (no title/code) —
-- confirmed live this previously leaked po.id (a raw UUID) into real answers.
-- Identify the portfolio by its OWNING auditor instead (JOIN users on
-- po.user_id) — never select po.id for display.
SELECT u_owner.first_name || ' ' || u_owner.last_name AS auditor_name,
       rev.status, rev.action, rev.reviewed_on,
       u_rev.first_name || ' ' || u_rev.last_name AS reviewed_by
FROM review rev
JOIN auditor_portfolio po ON rev.entity_id = po.id AND rev.entity_type = 'PORTFOLIO'
JOIN users u_owner ON po.user_id = u_owner.id
LEFT JOIN users u_rev ON rev.reviewed_by = u_rev.id
WHERE rev.status = 'COMPLETED' AND rev.action = 'APPROVE' AND po.active = true AND u_owner.active = true
ORDER BY rev.reviewed_on DESC
LIMIT 100;
""",
        'domain': 'review', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'audit schedules awaiting auditee acceptance or reschedule decision',
        'sql':    """\
-- SCHEDULE review = the auditee's accept/reschedule step on the scheduled visit.
SELECT asch.schedule_id, rev.level, rev.status,
       fac.name AS facility_name
FROM review rev
JOIN audit_schedule asch ON rev.entity_id = asch.id AND rev.entity_type = 'SCHEDULE'
LEFT JOIN facility fac ON asch.facility_id = fac.id
WHERE rev.status = 'PENDING' AND asch.active = true
ORDER BY asch.follow_up_date ASC NULLS LAST
LIMIT 100;
""",
        'domain': 'review', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'corrective actions pending adequacy review by the auditor',
        'sql':    """\
-- ADEQUACY review_type = auditor-side check on the proposed fix, level 1.
SELECT aca.corrective_action_id, aca.cause, aca.adequacy_status, rev.level, rev.status
FROM audit_corrective_action aca
JOIN review rev
  ON rev.entity_id = aca.id AND rev.entity_type = 'AUDIT_CORRECTIVE_ACTION'
  AND rev.review_type = 'ADEQUACY'
WHERE rev.status = 'PENDING' AND aca.active = true
ORDER BY aca.created_on ASC
LIMIT 100;
""",
        'domain': 'review', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'corrective actions pending implementation approval level 2 auditor sign off',
        'sql':    """\
-- IMPLEMENTATION review_type is two-level: level 1 = auditee-side, level 2 = auditor-side final sign-off.
SELECT aca.corrective_action_id, aca.correction, aca.implementation_status, rev.level, rev.status
FROM audit_corrective_action aca
JOIN review rev
  ON rev.entity_id = aca.id AND rev.entity_type = 'AUDIT_CORRECTIVE_ACTION'
  AND rev.review_type = 'IMPLEMENTATION'
WHERE rev.level = 2 AND rev.status = 'PENDING' AND aca.active = true
ORDER BY aca.created_on ASC
LIMIT 100;
""",
        'domain': 'review', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'how many audit plans are pending at each review level',
        'sql':    """\
SELECT rev.level, COUNT(*) AS pending_count
FROM review rev
WHERE rev.entity_type = 'AUDIT_PLAN' AND rev.status = 'PENDING'
GROUP BY rev.level
ORDER BY rev.level;
""",
        'domain': 'review', 'intent': 'aggregate', 'limit': None,
    },
    {
        'nl':     'who approved this audit plan',
        'sql':    """\
-- status='COMPLETED' alone is NOT 'approved' — a completed review can be a rejection
-- (action='RETURN_FOR_MODIFICATION'). Must filter BOTH status AND action.
SELECT ap.title, rev.level, rev.reviewed_on,
       u.first_name || ' ' || u.last_name AS approved_by
FROM review rev
JOIN audit_plan ap ON rev.entity_id = ap.id AND rev.entity_type = 'AUDIT_PLAN'
JOIN users u ON rev.reviewed_by = u.id
WHERE rev.status = 'COMPLETED' AND rev.action = 'APPROVE' AND ap.active = true
ORDER BY rev.reviewed_on DESC
LIMIT 100;
""",
        'domain': 'review', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'how many levels of approval does an audit corrective action implementation review go through',
        'sql':    """\
-- review_level table is NOT reliable here (says 1 level for AUDIT_CORRECTIVE_ACTION,
-- but real review rows reach level 2 for IMPLEMENTATION) — compute from review directly.
SELECT MAX(rev.level) AS max_level_seen, COUNT(DISTINCT rev.level) AS distinct_levels
FROM review rev
WHERE rev.entity_type = 'AUDIT_CORRECTIVE_ACTION' AND rev.review_type = 'IMPLEMENTATION';
""",
        'domain': 'review', 'intent': 'aggregate', 'limit': None,
    },
    {
        'nl':     'which corrective actions were returned for modification during adequacy review',
        'sql':    """\
SELECT aca.corrective_action_id, aca.cause, rev.level, rev.reviewed_on,
       u.first_name || ' ' || u.last_name AS returned_by
FROM audit_corrective_action aca
JOIN review rev
  ON rev.entity_id = aca.id AND rev.entity_type = 'AUDIT_CORRECTIVE_ACTION'
  AND rev.review_type = 'ADEQUACY'
LEFT JOIN users u ON rev.reviewed_by = u.id
WHERE rev.status = 'COMPLETED' AND rev.action = 'RETURN_FOR_MODIFICATION' AND aca.active = true
ORDER BY rev.reviewed_on DESC
LIMIT 100;
""",
        'domain': 'review', 'intent': 'filter', 'limit': 100,
    },

    # ── audit_corrective_action status/vocabulary ───────────────────────────
    {
        'nl':     'list all corrective actions',
        'sql':    """\
-- General listing — exclude CAs still pending adequacy approval BY DEFAULT.
-- corrective_action_id = 'CA_DRAFT' until adequacy_status = 'APPROVE' (confirmed
-- live) — NOT about submission status, a SUBMITTED-status CA can still be
-- 'CA_DRAFT'. Only omit this filter when the question explicitly asks for
-- pending/draft/unapproved corrective actions or adequacy review status.
SELECT aca.corrective_action_id, aca.status, aca.cause, aca.target_close_out_date
FROM audit_corrective_action aca
WHERE aca.active = true AND aca.adequacy_status = 'APPROVE'
ORDER BY aca.created_on DESC
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'list corrective actions pending adequacy approval',
        'sql':    """\
-- Explicit ask for pending-approval CAs — DO include them here (the inverse of
-- the default-exclude case above). Several rows may share
-- corrective_action_id = 'CA_DRAFT' in this result — that's expected, not a
-- bug. Never treat corrective_action_id as a unique row identifier here.
SELECT aca.corrective_action_id, aca.status, aca.adequacy_status, aca.cause
FROM audit_corrective_action aca
WHERE aca.active = true AND aca.adequacy_status IS DISTINCT FROM 'APPROVE'
ORDER BY aca.created_on DESC
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'count of corrective actions by status',
        'sql':    """\
-- GROUP BY status does NOT change the default-exclude rule above — it's
-- tempting to think grouping by status means "show every status including
-- CA_DRAFT/pending-adequacy rows", but the same general-listing default
-- applies: exclude rows still pending adequacy approval UNLESS the question
-- explicitly asks about pending/draft/adequacy status itself (confirmed live:
-- this exact phrasing pattern — "count/group by status" — is the single most
-- common way this default gets missed).
SELECT aca.status, COUNT(*) AS count
FROM audit_corrective_action aca
WHERE aca.active = true AND aca.adequacy_status = 'APPROVE'
GROUP BY aca.status
ORDER BY aca.status;
""",
        'domain': 'audit_corrective_action', 'intent': 'aggregate', 'limit': None,
    },
    {
        'nl':     'list open overdue corrective actions with risk level and facility',
        'sql':    """\
SELECT aca.corrective_action_id, aca.cause, aca.status, rl.name AS risk_level_name,
       fac.name AS facility_name, aca.target_close_out_date
FROM audit_corrective_action aca
JOIN audit_report arep ON aca.audit_id = arep.id
JOIN audit_schedule asch ON arep.schedule_id = asch.id
LEFT JOIN risk_level rl ON aca.risk_level_id = rl.id
LEFT JOIN facility fac ON asch.facility_id = fac.id
WHERE aca.status IN ('OPEN', 'OVERDUE') AND aca.active = true
  AND arep.active = true AND asch.active = true
  AND aca.adequacy_status = 'APPROVE'
ORDER BY aca.target_close_out_date ASC NULLS LAST
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        # Deliberately kept as its OWN example, not folded into the
        # 'list open overdue corrective actions' one above — confirmed live
        # 2026-07-29 (qid=987552fd): "lsit all corrective actions that are
        # overdue" retrieved that OPEN+OVERDUE example as its nearest
        # neighbor and copied its status IN ('OPEN', 'OVERDUE') filter
        # verbatim, silently widening a single-status ask into two statuses.
        # OPEN and OVERDUE are separate, mutually exclusive enum values
        # (see AUDIT_ENUM_COLUMNS) — "overdue" alone means status = 'OVERDUE'
        # only. Having both examples side by side gives the retriever a
        # closer-matching neighbor for the single-status phrasing instead of
        # forcing every "overdue" question through the open+overdue example.
        'nl':     'list all corrective actions that are overdue',
        'sql':    """\
SELECT aca.corrective_action_id, aca.cause, aca.status, rl.name AS risk_level_name,
       fac.name AS facility_name, aca.target_close_out_date
FROM audit_corrective_action aca
JOIN audit_report arep ON aca.audit_id = arep.id
JOIN audit_schedule asch ON arep.schedule_id = asch.id
LEFT JOIN risk_level rl ON aca.risk_level_id = rl.id
LEFT JOIN facility fac ON asch.facility_id = fac.id
WHERE aca.status = 'OVERDUE' AND aca.active = true
  AND arep.active = true AND asch.active = true
  AND aca.adequacy_status = 'APPROVE'
ORDER BY aca.target_close_out_date ASC NULLS LAST
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'corrective actions that are at risk behind schedule progress stage',
        'sql':    """\
-- 'at risk' = progress_stage = 'AT_RISK' ONLY — do NOT OR in NULL/NOT_STARTED.
SELECT aca.corrective_action_id, aca.cause, aca.progress_stage, aca.status
FROM audit_corrective_action aca
WHERE aca.progress_stage = 'AT_RISK' AND aca.active = true
  AND aca.adequacy_status = 'APPROVE'
ORDER BY aca.created_on DESC
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'how many corrective actions closed with deferred vs closed on time',
        'sql':    """\
-- status = 'CLOSE_WITH_DEFERRED' is a WORKFLOW STATE (deferred to next cycle),
-- NOT a timeliness judgment — do not confuse with "closed late" below, a
-- completely different concept derived from comparing dates.
SELECT
  COUNT(*) FILTER (WHERE status = 'CLOSED') AS closed_on_time,
  COUNT(*) FILTER (WHERE status = 'CLOSE_WITH_DEFERRED') AS closed_with_deferred
FROM audit_corrective_action
WHERE active = true AND adequacy_status = 'APPROVE';
""",
        'domain': 'audit_corrective_action', 'intent': 'aggregate', 'limit': None,
    },
    {
        # Caught live 2026-07-29 (qid=039d5f86): "which audit corrective actions
        # are closed on time vs late" retrieved the example above (STATUS-based,
        # CLOSED vs CLOSE_WITH_DEFERRED) as its nearest neighbor and adapted the
        # FILTER logic to compare dates instead — but kept the OLD alias
        # 'closed_with_deferred' unchanged, so the result was a column literally
        # named 'closed_with_deferred' whose actual value was "count of CLOSED
        # rows completed after target_close_out_date" — completely unrelated to
        # the real CLOSE_WITH_DEFERRED status. This is a DIFFERENT concept
        # (timeliness, derived from comparing two dates) from the workflow-state
        # example above — kept as its own seed example so retrieval can tell
        # them apart instead of blending one's filter logic with the other's
        # column names.
        'nl':     'which corrective actions are closed on time vs closed late',
        'sql':    """\
SELECT
  COUNT(*) FILTER (WHERE status = 'CLOSED' AND completed_on <= target_close_out_date) AS closed_on_time,
  COUNT(*) FILTER (WHERE status = 'CLOSED' AND completed_on > target_close_out_date) AS closed_late
FROM audit_corrective_action
WHERE active = true AND adequacy_status = 'APPROVE';
""",
        'domain': 'audit_corrective_action', 'intent': 'aggregate', 'limit': None,
    },
    {
        'nl':     'average number of days to close a corrective action',
        'sql':    """\
-- aca.age is DEAD (always 0, confirmed live) — never use it for duration.
-- Compute the real gap from close_on - created_on, and only for rows that
-- actually have a close_on (CLOSED / CLOSE_WITH_DEFERRED); still-open rows
-- have no close_on and would otherwise skew the average toward zero.
SELECT ROUND(AVG(EXTRACT(EPOCH FROM (close_on - created_on)) / 86400)::numeric, 1) AS avg_days_to_close,
       COUNT(*) AS closed_count
FROM audit_corrective_action
WHERE active = true AND close_on IS NOT NULL AND status IN ('CLOSED', 'CLOSE_WITH_DEFERRED')
  AND adequacy_status = 'APPROVE';
""",
        'domain': 'audit_corrective_action', 'intent': 'aggregate', 'limit': None,
    },
    {
        'nl':     'corrective actions assigned to which internal stakeholder',
        'sql':    """\
-- internal_stakeholder_id is a reliable FK to users, populated on every row
-- regardless of 'responsible' value — it's the Khidmah-side point of contact.
-- Don't confuse with external_stakeholder_name/email/phone, which are plain
-- free-text (not a users FK) and populated only for a small minority of rows.
SELECT aca.corrective_action_id, aca.status, aca.responsible,
       u.first_name || ' ' || u.last_name AS internal_stakeholder_name
FROM audit_corrective_action aca
JOIN users u ON aca.internal_stakeholder_id = u.id
WHERE aca.active = true AND u.active = true
  AND aca.adequacy_status = 'APPROVE'
ORDER BY aca.created_on DESC
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        # Caught live 2026-07-29 (qid=f87d531f): "who proposed the corrective
        # action" was answered with aca.responsible AS proposer — returning a
        # party/organization label (e.g. 'KHIDMAH_AND_CLIENT') as if it were a
        # person's name. created_by is the actual person who raised/logged the
        # CA, populated on every row, and a clean UUID FK to users.
        'nl':     'who proposed or raised this corrective action',
        'sql':    """\
SELECT aca.corrective_action_id,
       u.first_name || ' ' || u.last_name AS raised_by
FROM audit_corrective_action aca
JOIN users u ON aca.created_by = u.id
WHERE aca.active = true AND u.active = true
  AND aca.adequacy_status = 'APPROVE'
ORDER BY aca.created_on DESC
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'corrective actions by responsible party khidmah client contractor consultant',
        'sql':    """\
-- responsible values (confirmed live, corrected 2026-07-29 — this is NOT the
-- same enum as inspection_corrective_action.responsible): KHIDMAH,
-- KHIDMAH_AND_CLIENT, CLIENT, SERVICE_PROVIDER_CONTRACTOR, CONSULTANT.
-- This is WHICH PARTY/ORGANIZATION is responsible for executing the fix, not
-- a person — for "who proposed/raised this CA" use created_by -> users instead.
SELECT responsible, COUNT(*) AS ca_count
FROM audit_corrective_action
WHERE active = true AND adequacy_status = 'APPROVE'
GROUP BY responsible
ORDER BY ca_count DESC;
""",
        'domain': 'audit_corrective_action', 'intent': 'aggregate', 'limit': None,
    },

    # ── impact / compliance / risk_level disambiguation ─────────────────────
    {
        'nl':     'corrective actions with high impact classification',
        'sql':    """\
-- 'impact' = business dimension affected — use impact_id, NOT risk_level_id or compliance_id.
SELECT aca.corrective_action_id, imp.name AS impact_name, imp.scale
FROM audit_corrective_action aca
JOIN impact imp ON aca.impact_id = imp.id
WHERE aca.active = true AND imp.active = true
  AND aca.adequacy_status = 'APPROVE'
ORDER BY imp.scale DESC
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'corrective actions with a major non conformity compliance gap',
        'sql':    """\
-- 'compliance gap' = compliance_id, NOT impact_id — match by intent (compliance question),
-- not label text (confirmed live: risk_level/compliance/impact have no overlapping labels).
-- SPELLING TRAP: the compliance table stores 'Non Confirmity' (typo, missing 'o'),
-- NOT 'Non Conformity' — confirmed live. 'Conformity' matches zero rows.
SELECT aca.corrective_action_id, comp.name AS compliance_name
FROM audit_corrective_action aca
JOIN compliance comp ON aca.compliance_id = comp.id
WHERE comp.name ILIKE '%Non Confirmity%Major%' AND aca.active = true
  AND aca.adequacy_status = 'APPROVE'
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'show details of these high risk corrective actions',
        'sql':    """\
-- FOLLOW-UP 'show details' after a risk_level-filtered listing — reuse ONLY the
-- axis already established (risk_level) and add the plain-text columns; do NOT
-- also INNER JOIN impact/compliance just because 'details' sounds broad.
-- Confirmed live: adding those two unasked axes silently returned 0 rows for 6
-- real rows (impact_id/compliance_id were NULL), hiding aca.cause/correction/
-- corrective_action entirely even though the query executed without error.
SELECT aca.corrective_action_id, aca.status, aca.progress_stage,
       aca.cause, aca.correction, aca.corrective_action, aca.target_close_out_date,
       rl.name AS risk_level_name
FROM audit_corrective_action aca
JOIN risk_level rl ON aca.risk_level_id = rl.id
WHERE aca.active = true AND rl.active = true
  AND rl.name ILIKE '%High%' AND aca.adequacy_status = 'APPROVE'
ORDER BY aca.created_on DESC
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'what is the cause, correction, and corrective action for this',
        'sql':    """\
-- THREE SEPARATE TEXT COLUMNS on audit_corrective_action, confirmed live via DB
-- introspection — never conflate, never drop one because another sounds similar:
--   cause             = why it happened
--   correction        = the immediate fix already applied
--   corrective_action = the (root-cause/longer-term) action proposed to prevent
--                        recurrence — NOT the same value as correction, and NOT
--                        the same as corrective_action_id (the VARCHAR display
--                        code like '2026/06/ST041/AUD003_CA004').
-- No risk_level/impact/compliance JOIN needed at all to answer this — they're
-- plain columns directly on the table.
SELECT aca.corrective_action_id, aca.cause, aca.correction, aca.corrective_action, aca.status
FROM audit_corrective_action aca
WHERE aca.active = true AND aca.adequacy_status = 'APPROVE'
ORDER BY aca.created_on DESC
LIMIT 100;
""",
        'domain': 'audit_corrective_action', 'intent': 'filter', 'limit': 100,
    },

    # ── auditor_portfolio ────────────────────────────────────────────────────
    {
        'nl':     'show auditor portfolio coverage facilities and categories for this cycle',
        'sql':    """\
-- Portfolio coverage: auditor -> portfolio -> detail -> facility, and detail -> category/type junctions.
-- NEVER the dead audit_*_portfolio family — use portfolio_detail_audit_* here.
SELECT u.first_name || ' ' || u.last_name AS auditor_name,
       fac.name AS facility_name,
       acat.name AS category_name
FROM auditor_portfolio po
JOIN users u ON po.user_id = u.id
JOIN auditor_portfolio_detail pod ON pod.portfolio_id = po.id
LEFT JOIN facility fac ON pod.facility_id = fac.id
LEFT JOIN portfolio_detail_audit_category pdac ON pdac.detail_id = pod.id
LEFT JOIN audit_category acat ON pdac.audit_category_id = acat.id
WHERE po.active = true AND pod.active = true
ORDER BY u.last_name
LIMIT 100;
""",
        'domain': 'auditor_portfolio', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'which auditors have not submitted their portfolio for this cycle',
        'sql':    """\
SELECT u.first_name || ' ' || u.last_name AS auditor_name, po.status, po.submitted_date
FROM auditor_portfolio po
JOIN users u ON po.user_id = u.id
WHERE po.status != 'APPROVED' AND po.active = true
ORDER BY po.submitted_date ASC NULLS FIRST
LIMIT 100;
""",
        'domain': 'auditor_portfolio', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'which auditor portfolios need modification',
        'sql':    """\
-- status='MODIFICATION' — confirmed live (2 rows); the "returned, needs
-- rework" state distinct from APPROVED. auditor_portfolio.status is a
-- 4-value CHECK constraint: APPROVED, DRAFT, MODIFICATION, PENDING (confirmed
-- live via DB introspection) — there is NO 'SUBMITTED' value at all, that's
-- a live-caught hallucination (structurally impossible, not just unused).
-- DRAFT/PENDING are valid schema values but happen to have 0 populated rows
-- in this dataset right now (only APPROVED/MODIFICATION are currently used) —
-- don't assume DRAFT/PENDING can never appear just because today's data
-- doesn't have any; only treat SUBMITTED as invalid.
SELECT u.first_name || ' ' || u.last_name AS auditor_name, po.status, po.submitted_date
FROM auditor_portfolio po
JOIN users u ON po.user_id = u.id
WHERE po.status = 'MODIFICATION' AND po.active = true
ORDER BY po.submitted_date DESC
LIMIT 100;
""",
        'domain': 'auditor_portfolio', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'which portfolios are still pending submission',
        'sql':    """\
-- 'pending submission' = status IN ('DRAFT', 'PENDING') — the two
-- not-yet-submitted values in auditor_portfolio's 4-value CHECK constraint
-- (APPROVED, DRAFT, MODIFICATION, PENDING). NEVER 'SUBMITTED' — that value
-- does not exist on this table at all (confirmed live via DB introspection),
-- unlike audit_plan/audit_report which DO use 'SUBMITTED'. Don't reuse
-- another table's status vocabulary here.
SELECT u.first_name || ' ' || u.last_name AS auditor_name, po.status, po.submitted_date
FROM auditor_portfolio po
JOIN users u ON po.user_id = u.id
WHERE po.status IN ('DRAFT', 'PENDING') AND po.active = true
ORDER BY po.submitted_date ASC NULLS FIRST
LIMIT 100;
""",
        'domain': 'auditor_portfolio', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'how many facilities and projects does each auditor portfolio cover',
        'sql':    """\
SELECT u.first_name || ' ' || u.last_name AS auditor_name,
       COUNT(DISTINCT pod.facility_id) AS facility_count,
       COUNT(DISTINCT pod.project_id) AS project_count
FROM auditor_portfolio po
JOIN users u ON po.user_id = u.id
JOIN auditor_portfolio_detail pod ON pod.portfolio_id = po.id
WHERE po.active = true AND pod.active = true
GROUP BY u.id, u.first_name, u.last_name
ORDER BY facility_count DESC
LIMIT 100;
""",
        'domain': 'auditor_portfolio', 'intent': 'aggregate', 'limit': 100,
    },
    {
        'nl':     'which auditor covers the most facilities in their portfolio',
        'sql':    """\
SELECT u.first_name || ' ' || u.last_name AS auditor_name,
       COUNT(DISTINCT pod.facility_id) AS facility_count
FROM auditor_portfolio po
JOIN users u ON po.user_id = u.id
JOIN auditor_portfolio_detail pod ON pod.portfolio_id = po.id
WHERE po.active = true AND pod.active = true
GROUP BY u.id, u.first_name, u.last_name
ORDER BY facility_count DESC
LIMIT 10;
""",
        'domain': 'auditor_portfolio', 'intent': 'ranked_list', 'limit': 10,
    },
    {
        'nl':     'how many auditor portfolios were approved for this cycle',
        'sql':    """\
-- auditor_portfolio.audit_cycle_id is a DIRECT FK to the cycle — do not
-- route through any of the dead junction tables for cycle coverage.
SELECT COUNT(*) AS approved_count
FROM auditor_portfolio po
JOIN audit_cycle acyc ON po.audit_cycle_id = acyc.id
WHERE po.status = 'APPROVED' AND acyc.status = 'ONGOING' AND po.active = true;
""",
        'domain': 'auditor_portfolio', 'intent': 'aggregate', 'limit': None,
    },

    # ── audit_cycle progress ─────────────────────────────────────────────────
    {
        'nl':     'show audit cycles with their date range and status',
        'sql':    """\
SELECT acyc.start_date, acyc.end_date, acyc.due_date, acyc.status
FROM audit_cycle acyc
WHERE acyc.active = true
ORDER BY acyc.start_date DESC
LIMIT 100;
""",
        'domain': 'audit_cycle', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'how many audit plans schedules and reports in the current audit cycle',
        'sql':    """\
SELECT
  (SELECT COUNT(*) FROM audit_plan ap JOIN audit_cycle acyc ON ap.cycle_id = acyc.id
    WHERE acyc.status = 'ONGOING' AND ap.active = true) AS plan_count,
  (SELECT COUNT(*) FROM audit_schedule asch JOIN audit_plan ap ON asch.audit_plan_id = ap.id
    JOIN audit_cycle acyc ON ap.cycle_id = acyc.id
    WHERE acyc.status = 'ONGOING' AND asch.active = true) AS schedule_count,
  (SELECT COUNT(*) FROM audit_report arep JOIN audit_cycle acyc ON arep.audit_cycle_id = acyc.id
    WHERE acyc.status = 'ONGOING' AND arep.active = true) AS report_count;
""",
        'domain': 'audit_cycle', 'intent': 'aggregate', 'limit': None,
    },
    {
        'nl':     'which audit cycle is currently active',
        'sql':    """\
-- status = 'ONGOING' — confirmed live enum value (also seen: UPCOMING, COMPLETED).
SELECT acyc.start_date, acyc.end_date, acyc.due_date, acyc.status
FROM audit_cycle acyc
WHERE acyc.status = 'ONGOING' AND acyc.active = true
LIMIT 5;
""",
        'domain': 'audit_cycle', 'intent': 'filter', 'limit': 5,
    },
    {
        'nl':     'how many audit plans were proposed in each cycle',
        'sql':    """\
-- audit_cycle has NO display-name column (confirmed live) — identify cycles
-- by their date span and status only. Inventing a name/title/code field here
-- is a live-caught hallucination that once burned 7 LLM calls before this
-- trap was guarded.
SELECT acyc.start_date, acyc.end_date, acyc.status, COUNT(ap.id) AS plan_count
FROM audit_cycle acyc
LEFT JOIN audit_plan ap ON ap.cycle_id = acyc.id AND ap.active = true
WHERE acyc.active = true
GROUP BY acyc.id, acyc.start_date, acyc.end_date, acyc.status
ORDER BY acyc.start_date DESC
LIMIT 100;
""",
        'domain': 'audit_cycle', 'intent': 'aggregate', 'limit': 100,
    },
    {
        'nl':     'which audit cycles are overdue past their due date',
        'sql':    """\
-- due_date is the cycle's own end-of-period deadline — a DIFFERENT concept
-- from any individual schedule's follow_up_date. Only meaningful for cycles
-- not yet COMPLETED.
SELECT acyc.start_date, acyc.end_date, acyc.due_date, acyc.status
FROM audit_cycle acyc
WHERE acyc.due_date < CURRENT_DATE AND acyc.status != 'COMPLETED' AND acyc.active = true
ORDER BY acyc.due_date ASC
LIMIT 100;
""",
        'domain': 'audit_cycle', 'intent': 'filter', 'limit': 100,
    },

    # ── audit_report / audit_forms ───────────────────────────────────────────
    {
        'nl':     'audit reports with their form template used and score',
        'sql':    """\
-- audit_form_used_id is always populated (confirmed live) — the reliable form link.
SELECT arep.audit_id, arep.status, arep.audit_score, arep.total_audit_hours,
       aform.title AS form_title
FROM audit_report arep
JOIN audit_forms aform ON arep.audit_form_used_id = aform.id
WHERE arep.active = true
ORDER BY arep.submitted_on DESC NULLS LAST
LIMIT 100;
""",
        'domain': 'audit_report', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'average audit score by client and entity',
        'sql':    """\
-- client_id and entity_id are BOTH always populated and genuinely independent
-- dimensions on audit_report — not aliases of each other.
SELECT cl.name AS client_name, ent.name AS entity_name,
       ROUND(AVG(arep.audit_score)::numeric, 2) AS avg_score, COUNT(*) AS report_count
FROM audit_report arep
JOIN client cl ON arep.client_id = cl.id
JOIN entity ent ON arep.entity_id = ent.id
WHERE arep.active = true AND arep.audit_score IS NOT NULL
GROUP BY cl.id, cl.name, ent.id, ent.name
ORDER BY avg_score DESC
LIMIT 50;
""",
        'domain': 'audit_report', 'intent': 'aggregate', 'limit': 50,
    },
    {
        'nl':     'which audit reports scored the highest',
        'sql':    """\
-- gp_score and closed_on are BOTH always NULL on audit_report (confirmed
-- live, dead columns) — use audit_score for scoring, and don't try to
-- answer "when was this report closed" from report-level data at all.
SELECT arep.audit_id, arep.audit_score, cl.name AS client_name, ent.name AS entity_name
FROM audit_report arep
JOIN client cl ON arep.client_id = cl.id
JOIN entity ent ON arep.entity_id = ent.id
WHERE arep.active = true AND arep.audit_score IS NOT NULL
ORDER BY arep.audit_score DESC
LIMIT 10;
""",
        'domain': 'audit_report', 'intent': 'ranked_list', 'limit': 10,
    },
    {
        'nl':     'list all audit form templates by universe and audit type',
        'sql':    """\
SELECT aform.title, aform.status, auniv.name AS universe_name, aty.name AS audit_type_name
FROM audit_forms aform
LEFT JOIN audit_universe auniv ON aform.audit_universe_id = auniv.id
LEFT JOIN audit_type aty ON aform.audit_type_id = aty.id
WHERE aform.active = true
ORDER BY aform.title
LIMIT 100;
""",
        'domain': 'audit_forms', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'which facilities does this audit form template apply to',
        'sql':    """\
-- audit_form_facilities is a plain junction table (audit_form_id, facility_id
-- — no id/active/timestamps of its own, unlike most tables here).
SELECT fac.name AS facility_name
FROM audit_form_facilities aff
JOIN facility fac ON aff.facility_id = fac.id
WHERE aff.audit_form_id = '<AUDIT_FORM_UUID>' AND fac.active = true
ORDER BY fac.name
LIMIT 100;
""",
        'domain': 'audit_forms', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'which audit forms are still in draft status',
        'sql':    """\
-- status: DRAFT (template still being authored) vs SUBMITTED (published,
-- confirmed live: 1 DRAFT / 14 SUBMITTED) — the template's own publishing
-- state, unrelated to any specific audit visit's status.
SELECT aform.title, auniv.name AS universe_name, aty.name AS audit_type_name
FROM audit_forms aform
LEFT JOIN audit_universe auniv ON aform.audit_universe_id = auniv.id
LEFT JOIN audit_type aty ON aform.audit_type_id = aty.id
WHERE aform.status = 'DRAFT' AND aform.active = true
LIMIT 100;
""",
        'domain': 'audit_forms', 'intent': 'filter', 'limit': 100,
    },
    {
        'nl':     'how many audit form templates exist for each audit type',
        'sql':    """\
SELECT aty.name AS audit_type_name, COUNT(*) AS form_count
FROM audit_forms aform
JOIN audit_type aty ON aform.audit_type_id = aty.id
WHERE aform.active = true
GROUP BY aty.id, aty.name
ORDER BY form_count DESC;
""",
        'domain': 'audit_forms', 'intent': 'aggregate', 'limit': None,
    },

    # ── dead-table avoidance (BLOCK rule reinforcement via example) ─────────
    {
        'nl':     'which category and type does this auditor portfolio cover',
        'sql':    """\
-- NEVER audit_category_portfolio/audit_type_portfolio (always 0 rows, dead).
-- Real path is portfolio_detail_audit_category/_type via auditor_portfolio_detail.
SELECT acat.name AS category_name, aty.name AS audit_type_name
FROM auditor_portfolio_detail pod
LEFT JOIN portfolio_detail_audit_category pdac ON pdac.detail_id = pod.id
LEFT JOIN audit_category acat ON pdac.audit_category_id = acat.id
LEFT JOIN portfolio_detail_audit_type pdat ON pdat.detail_id = pod.id
LEFT JOIN audit_type aty ON pdat.audit_type_id = aty.id
WHERE pod.portfolio_id = '<PORTFOLIO_UUID>' AND pod.active = true
LIMIT 100;
""",
        'domain': 'auditor_portfolio', 'intent': 'filter', 'limit': 100,
    },

    # ── total counts / simple aggregates ─────────────────────────────────────
    {
        'nl':     'how many open audit corrective actions are there',
        'sql':    """\
SELECT COUNT(*) AS open_count
FROM audit_corrective_action
WHERE active = true AND status = 'OPEN' AND adequacy_status = 'APPROVE';
""",
        'domain': 'audit_corrective_action', 'intent': 'aggregate', 'limit': None,
    },
    {
        'nl':     'total number of audit schedules by status',
        'sql':    """\
SELECT schedule_status, COUNT(*) AS schedule_count
FROM audit_schedule
WHERE active = true
GROUP BY schedule_status
ORDER BY schedule_count DESC;
""",
        'domain': 'audit_schedule', 'intent': 'aggregate', 'limit': None,
    },
    {
        'nl':     'most common causes of audit corrective actions',
        'sql':    """\
SELECT cause, COUNT(*) AS ca_count
FROM audit_corrective_action
WHERE active = true AND cause IS NOT NULL AND adequacy_status = 'APPROVE'
GROUP BY cause
ORDER BY ca_count DESC
LIMIT 20;
""",
        'domain': 'audit_corrective_action', 'intent': 'ranked_list', 'limit': 20,
    },

    # ── ai_audit_answers form-answer layer (confirmed live 2026-07-29 — see
    # AUDIT_BUSINESS_RULES "AUDIT FORM-ANSWER LAYER" / "OBSERVATION -> CORRECTIVE
    # ACTION" and AUDIT_SEMANTICS["joins"]["observation_to_ca"] for the cast rule) ──
    {
        # Fixed 2026-07-29 (qid=9ce471eb): "for what observation was this raised"
        # matched this example and returned ALL 9 form fields in the submission
        # (Audit Checkpoint, Compliance, Impact, Requirement Type, Risk Level,
        # Schedule Date and Time, Supporting Documents, etc.) instead of just the
        # free-text finding — noisy for a question asking specifically "what
        # observation", and missing the aq.label filter AUDIT_BUSINESS_RULES'
        # OBSERVATION -> CORRECTIVE ACTION rule documents for PATH 1 ("to get the
        # free-text finding itself, further filter WHERE aq.label = 'Findings /
        # Remarks'"). Use the FULL-SUBMISSION variant below when the question asks
        # to see everything filled out, not just what triggered the CA.
        'nl':     'what was the observation for this corrective action observation text for a specific CA what triggered this corrective action',
        'sql':    """-- OBSERVATION -> CA PATH: aaa.submission_id = aca.observation_id::varchar
-- Cast UUID side to varchar — NOT aaa.submission_id::uuid (submission_id is varchar, observation_id is UUID)
-- This is the DIRECT link — NOT aca.audit_id = arep.id (that returns ALL CAs in the report)
-- Filter to the free-text finding itself — omitting this returns all 9 form
-- fields in the submission (Audit Checkpoint, Compliance, Impact, etc.), not
-- just the observation.
SELECT aca.corrective_action_id,
       aaa.answer_text AS observation_text
FROM audit_corrective_action aca
JOIN ai_audit_answers aaa ON aaa.submission_id = aca.observation_id::varchar
JOIN ai_questions aq ON aaa.element_id = aq.element_id
WHERE aca.corrective_action_id = '<CA_ID>'
  AND aca.observation_id IS NOT NULL
  AND aca.active = true
  AND aq.label = 'Findings / Remarks'
LIMIT 1;
""",
        'domain': 'ai_audit_answers', 'intent': 'join', 'limit': 1,
    },
    {
        'nl':     'show the full submission everything filled out for this observation entry',
        'sql':    """-- Full-submission variant — every form field for the observation entry
-- this CA was raised against, not just the free-text finding. Use when the
-- question asks to see everything filled out, not "what observation" alone.
SELECT aca.corrective_action_id,
       aq.label AS field,
       aaa.answer_text AS value
FROM audit_corrective_action aca
JOIN ai_audit_answers aaa ON aaa.submission_id = aca.observation_id::varchar
JOIN ai_questions aq ON aaa.element_id = aq.element_id
WHERE aca.corrective_action_id = '<CA_ID>'
  AND aca.observation_id IS NOT NULL
  AND aca.active = true
ORDER BY aq.label
LIMIT 100;
""",
        'domain': 'ai_audit_answers', 'intent': 'join', 'limit': 100,
    },
    {
        'nl':     'which observation triggered the most corrective actions observation with most CAs',
        'sql':    """-- CAST direction: aaa.submission_id = aca.observation_id::varchar
-- submission_id is varchar, observation_id is UUID — cast UUID to varchar, NOT varchar to uuid
SELECT aaa.answer_text AS observation,
       COUNT(aca.id)   AS ca_count,
       arep.audit_id
FROM audit_corrective_action aca
JOIN ai_audit_answers aaa ON aaa.submission_id = aca.observation_id::varchar
JOIN ai_questions aq ON aaa.element_id = aq.element_id
JOIN audit_report arep ON aca.audit_id = arep.id
WHERE aca.active = true AND aca.observation_id IS NOT NULL AND aq.label = 'Findings / Remarks'
-- NOTE: audit has NO question labeled 'Observation' (unlike inspection) — confirmed live.
-- 'Findings / Remarks' is the equivalent free-text field.
GROUP BY aaa.id, aaa.answer_text, arep.audit_id
ORDER BY ca_count DESC LIMIT 10;
""",
        'domain': 'ai_audit_answers', 'intent': 'ranked_list', 'limit': 10,
    },
    {
        'nl':     'reconstruct the form for this audit show all answers for this audit report full filled out form',
        'sql':    """-- PATH 2: whole-report reconstruction — aaa.audit_report_id = arep.id
-- Group by submission_id: each distinct value is one repeatable entry (e.g. one observation)
SELECT aq.label,
       aaa.answer_text,
       aaa.answer_numeric,
       aaa.submission_id
FROM ai_audit_answers aaa
JOIN ai_questions aq ON aaa.element_id = aq.element_id
JOIN audit_report arep ON aaa.audit_report_id = arep.id
WHERE arep.audit_id = '<AUDIT_DISPLAY_ID>'
ORDER BY aaa.submission_id, aaa.created_on
LIMIT 200;
""",
        'domain': 'ai_audit_answers', 'intent': 'join', 'limit': 200,
    },
]
