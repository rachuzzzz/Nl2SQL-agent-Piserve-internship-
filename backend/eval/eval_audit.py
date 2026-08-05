"""
Audit Domain — Evaluation Test Set (~50 queries)
=================================================
Audit-domain counterpart to eval.py — same runner shape, hardcoded to
domain="audit" (AgentOrchestrator.from_env(domain="audit")) instead of
inspection's default.

Usage:
  python3 eval_audit.py
  python3 eval_audit.py --skip-multiturn
  python3 eval_audit.py --category "Approval"
  python3 eval_audit.py --output my_results.json

Categories
──────────
  Plan        — audit_plan: status, cycle, type, category, proposer
  Schedule    — audit_schedule: status, facility, auditor/auditee, dates
  Status      — status vocabulary across plan/schedule/report/corrective action
  Cycle       — audit_cycle: date range, currently-active, plans-per-cycle, overdue
  Forms       — audit_forms: templates, draft status, facility/project applicability
  Admin       — org-wide/admin-level questions: counts, coverage, existence checks
  Portfolio   — auditor_portfolio / auditor_portfolio_detail coverage
  Approval    — review/review_level: who approved, levels of approval, rejections.
                Written from a corrected understanding of that relationship (see
                schema_introspector.py's BUSINESS_RULES and
                audit_rules.py's audit_block_review_completed_not_approved):
                  - review_level is NOT authoritative for "how many levels" —
                    it's a stale, incomplete 4-row config table (missing
                    AUDIT_PLAN entirely, wrong for AUDIT_CORRECTIVE_ACTION).
                    The real level count must come from review itself.
                  - status='COMPLETED' on review does NOT mean approved — 28%
                    of completed reviews (confirmed live) are rejections
                    (action='RETURN_FOR_MODIFICATION'). "Who approved" needs
                    status='COMPLETED' AND action='APPROVE'.
  Workflow    — cross-cutting lifecycle cases added 2026-07-28 after a full
                workflow walkthrough with the domain owner (see
                docs/audit_workflow.md): the unplanned-audit fast-track
                (skips plan + auditee review entirely, audit_plan_id not
                guaranteed populated), multi-facility plan→schedule mapping,
                post-approval portfolio edits, corrected progress_stage
                ownership (internal stakeholder/FM manager, not auditee), the
                auditor-triggered 2nd adequacy level, and manual (non-
                automatic) corrective-action closure.

Scoring notes — identical semantics to eval.py:
  keyword_pct : percentage of expected_contains keywords found in the answer.
  empty_result: True when the answer matches known "no data" phrases.
  real_pass   : True only when success=True AND not a forced-empty-fail.
"""

TEST_SET = [
    # ── Plans ────────────────────────────────────────────────────────────────
    {
        "id": "PLAN-01", "category": "Plan",
        "question": "list all audit plans",
        "expected_contains": ["plan"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Baseline listing",
    },
    {
        "id": "PLAN-02", "category": "Plan",
        "question": "which audit plans are still in draft",
        "expected_contains": ["draft"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "status = 'DRAFT'",
    },
    {
        "id": "PLAN-03", "category": "Plan",
        "question": "show audit plans pending approval",
        "expected_contains": ["plan"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "status = 'SUBMITTED'",
    },
    {
        "id": "PLAN-04", "category": "Plan",
        "question": "how many audit plans are approved for this cycle",
        "expected_contains": ["plan"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "status='APPROVED' + cycle_id join",
    },
    {
        "id": "PLAN-05", "category": "Plan",
        "question": "who proposed the most audit plans",
        "expected_contains": ["plan"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "audit_plan.user_id → users, ranked",
    },
    {
        "id": "PLAN-06", "category": "Plan",
        "question": "how many unplanned audit plans were raised",
        "expected_contains": ["plan"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "is_unplanned = true",
    },
    {
        "id": "PLAN-07", "category": "Plan",
        "question": "show audit plans by category and type",
        "expected_contains": ["plan"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "JOIN audit_category, audit_type",
    },
    {
        "id": "PLAN-09", "category": "Plan",
        "question": "which audit plan has the most schedules under it",
        "expected_contains": ["plan"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "audit_schedule.audit_plan_id, ranked",
    },

    # ── Schedules ────────────────────────────────────────────────────────────
    {
        "id": "SCHED-01", "category": "Schedule",
        "question": "list all audit schedules",
        "expected_contains": ["schedule"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Baseline listing",
    },
    {
        "id": "SCHED-02", "category": "Schedule",
        "question": "which audit schedules are ongoing",
        "expected_contains": ["schedule"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "schedule_status = 'ONGOING', NOT audit_status (dead)",
    },
    {
        "id": "SCHED-03", "category": "Schedule",
        "question": "show audit schedules awaiting auditee acceptance",
        "expected_contains": ["schedule"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "schedule_status = 'PENDING_REVIEW'",
    },
    {
        "id": "SCHED-04", "category": "Schedule",
        "question": "how many audit schedules were completed this year",
        "expected_contains": ["schedule"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "schedule_status = 'COMPLETED' + year filter",
    },
    {
        "id": "SCHED-05", "category": "Schedule",
        "question": "which auditor has the most scheduled visits",
        "expected_contains": ["auditor"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "auditor_id → users, ranked",
    },
    {
        "id": "SCHED-06", "category": "Schedule",
        "question": "show audit schedules for facility Al Ghadeer",
        "expected_contains": ["facility"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "facility_id join + name ILIKE",
    },
    {
        "id": "SCHED-07", "category": "Schedule",
        "question": "how many schedules have been rescheduled more than once",
        "expected_contains": ["schedule"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "reschedule_count > 1",
    },
    {
        "id": "SCHED-08", "category": "Schedule",
        "question": "show audit schedules happening next week",
        "expected_contains": ["schedule"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "follow_up_date range — schedule's own date, not cycle due_date",
    },
    {
        "id": "SCHED-09", "category": "Schedule",
        "question": "who is the auditee for the most audit schedules",
        "expected_contains": ["auditee"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "auditee_id → users, ranked",
    },

    # ── Statuses ─────────────────────────────────────────────────────────────
    {
        "id": "STAT-01", "category": "Status",
        "question": "how many audit reports are closed with exception",
        "expected_contains": ["report"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "status = 'CLOSED_WITH_EXCEPTION'",
    },
    {
        "id": "STAT-02", "category": "Status",
        "question": "what statuses can an audit corrective action have",
        "expected_contains": ["status"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "DISTINCT status vocabulary",
    },
    {
        "id": "STAT-03", "category": "Status",
        "question": "how many corrective actions are overdue vs open vs closed",
        "expected_contains": ["overdue"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "CASE/FILTER breakdown, OVERDUE is explicit state",
    },
    {
        "id": "STAT-04", "category": "Status",
        "question": "which audit reports are still in draft status",
        "expected_contains": ["report"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "audit_report.status = 'DRAFT'",
    },
    {
        "id": "STAT-05", "category": "Status",
        "question": "how many audit schedules are in each schedule status",
        "expected_contains": ["status"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "GROUP BY schedule_status",
    },
    {
        "id": "STAT-06", "category": "Status",
        "question": "how many corrective actions closed with deferred vs closed on time",
        "expected_contains": ["closed"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "CLOSED vs CLOSE_WITH_DEFERRED, FILTER clause",
    },

    # ── Cycle ────────────────────────────────────────────────────────────────
    # Previously zero dedicated coverage — cycle only appeared incidentally
    # inside Plan questions (PLAN-04/PLAN-08). Added alongside the 5 new
    # audit_cycle seed examples (2026-07-27) to actually exercise them.
    {
        "id": "CYCLE-01", "category": "Cycle",
        "question": "show all audit cycles with their date range and status",
        "expected_contains": ["cycle"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Baseline listing — acyc has no name, identify by dates/status",
    },
    {
        "id": "CYCLE-02", "category": "Cycle",
        "question": "which audit cycle is currently active",
        "expected_contains": ["cycle"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "status = 'ONGOING' — confirmed live enum value",
    },
    {
        "id": "CYCLE-03", "category": "Cycle",
        "question": "how many audit plans were proposed in each cycle",
        "expected_contains": ["cycle"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "REGRESSION: live-caught burning 7 LLM calls hallucinating acyc.name before the "
                 "no-name-column trap was guarded (WRONG_COLUMN / HALLUCINATED_AC_COLUMNS) and a "
                 "matching seed example was added — expect 1-2 calls now, not 7.",
    },
    {
        "id": "CYCLE-04", "category": "Cycle",
        "question": "which audit cycles are overdue past their due date",
        "expected_contains": ["due"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "due_date < CURRENT_DATE, excluding COMPLETED cycles",
    },
    {
        "id": "CYCLE-05", "category": "Cycle",
        "question": "how many audit plans schedules and reports are in the current audit cycle",
        "expected_contains": ["cycle"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Three-way scalar subquery rollup, status='ONGOING'",
    },

    # ── Forms ────────────────────────────────────────────────────────────────
    # Previously ZERO coverage of any kind — audit_forms never appeared in any
    # eval question. Added alongside the 4 new audit_forms seed examples
    # (2026-07-27), including the previously-untouched audit_form_facilities
    # junction table.
    {
        "id": "FORMS-01", "category": "Forms",
        "question": "list all audit form templates by universe and audit type",
        "expected_contains": ["form"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "Baseline listing",
    },
    {
        "id": "FORMS-02", "category": "Forms",
        "question": "which audit forms are still in draft status",
        "expected_contains": ["draft"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "status='DRAFT' AND active=true. The one DRAFT-status row in this dataset "
                 "('Test Form') has active=false (soft-deleted) — confirmed live via direct query — "
                 "so an empty result with active=true correctly applied is the RIGHT answer, not a "
                 "bug. expected_nonempty was wrongly set True initially; corrected after live "
                 "verification caught it as a forced-fail.",
    },
    {
        "id": "FORMS-03", "category": "Forms",
        "question": "how many audit form templates exist for each audit type",
        "expected_contains": ["type"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "GROUP BY audit_type_id",
    },
    {
        "id": "FORMS-04", "category": "Forms",
        "question": "which facilities does the QHSE audit form template apply to",
        "expected_contains": ["facilit"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "audit_form_facilities junction — previously zero seed/eval coverage of any kind. "
                 "Also confirms audit_forms.audit_schedule_id is dead (0/15 populated, confirmed live) "
                 "isn't mistakenly used as a shortcut here.",
    },

    # ── Admin-level ──────────────────────────────────────────────────────────
    {
        "id": "ADMIN-01", "category": "Admin",
        "question": "how many total audits have been conducted across all universes",
        "expected_contains": ["audit"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "org-wide count",
    },
    {
        "id": "ADMIN-02", "category": "Admin",
        "question": "which audit universe has the most audit plans",
        "expected_contains": ["universe"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ranked by universe",
    },
    {
        "id": "ADMIN-03", "category": "Admin",
        "question": "audits with no corrective actions raised against them",
        "expected_contains": ["audit"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "existence check — CA_DRAFT still counts as raised",
    },
    {
        "id": "ADMIN-04", "category": "Admin",
        "question": "which facility has been audited the most",
        "expected_contains": ["facility"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "two-hop via audit_schedule, ranked",
    },
    {
        "id": "ADMIN-05", "category": "Admin",
        "question": "list all active auditors",
        "expected_contains": ["auditor"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "distinct auditor_id from audit_schedule, not role table",
    },
    {
        "id": "ADMIN-06", "category": "Admin",
        "question": "how many audit categories and sub-categories are there",
        "expected_contains": ["categor"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "reference/taxonomy counts",
    },
    {
        "id": "ADMIN-07", "category": "Admin",
        "question": "what is the average audit score across all reports",
        "expected_contains": ["score"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "AVG(audit_score)",
    },
    {
        "id": "ADMIN-08", "category": "Admin",
        "question": "show audit schedules per country",
        "expected_contains": ["country"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "facility.country_id — the one reliably populated geo FK",
    },

    # ── Portfolios ───────────────────────────────────────────────────────────
    {
        "id": "PORT-01", "category": "Portfolio",
        "question": "show auditor portfolio coverage — facilities and categories covered by each auditor",
        "expected_contains": ["portfolio"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "auditor_portfolio + auditor_portfolio_detail. Originally phrased as 'for this "
                 "cycle', which the agent resolved to audit_cycle.status='ACTIVE' — no cycle has "
                 "that status in this dataset, so it returned empty. Not a schema bug; rephrased "
                 "to drop the 'current cycle' resolution requirement.",
    },
    {
        "id": "PORT-02", "category": "Portfolio",
        "question": "which facilities are covered under this auditor's portfolio",
        "expected_contains": ["facility"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "auditor_portfolio_detail.facility_id",
    },
    {
        "id": "PORT-03", "category": "Portfolio",
        "question": "which category does this auditor portfolio cover",
        "expected_contains": ["categor"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "portfolio_detail_audit_category — NOT the dead audit_category_portfolio junction",
    },
    {
        "id": "PORT-04", "category": "Portfolio",
        "question": "how many auditor portfolios were submitted this cycle",
        "expected_contains": ["portfolio"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "auditor_portfolio.audit_cycle_id direct FK",
    },
    {
        "id": "PORT-05", "category": "Portfolio",
        "question": "which auditor covers the most facilities in their portfolio",
        "expected_contains": ["portfolio"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "ranked by portfolio detail count",
    },
    {
        "id": "PORT-06", "category": "Portfolio",
        "question": "how many projects fall under each auditor's portfolio",
        "expected_contains": ["portfolio"],
        "expected_nonempty": True,
        "expected_llm_calls": 1, "notes": "auditor_portfolio_detail.project_id",
    },
    {
        "id": "PORT-07", "category": "Portfolio",
        "question": "which portfolios are still pending submission",
        "expected_contains": ["portfolio"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "auditor_portfolio.status",
    },

    # ── Approval / who-approved / levels of approval ───────────────────────────
    # Written from the corrected review/review_level understanding — see module
    # docstring. Any answer that relies on review_level for a level count, or on
    # status='COMPLETED' alone for an approval, should be treated as WRONG even
    # if the agent returns something confidently.
    {
        "id": "APPR-01", "category": "Approval",
        "question": "who approved this audit plan",
        "expected_contains": ["approv"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "review.entity_type='AUDIT_PLAN', MUST filter status='COMPLETED' AND action='APPROVE'",
    },
    {
        "id": "APPR-02", "category": "Approval",
        "question": "how many levels of approval does an audit corrective action go through",
        "expected_contains": ["level"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "MUST compute from review (MAX/COUNT DISTINCT level), NOT review_level — "
                 "review_level wrongly claims 1 level; real data reaches level 2 for both review_types",
    },
    {
        "id": "APPR-03", "category": "Approval",
        "question": "how many levels of approval does an audit plan require",
        "expected_contains": ["level"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "review_level has NO row at all for AUDIT_PLAN — must come from review itself (real answer: 2)",
    },
    {
        "id": "APPR-04", "category": "Approval",
        "question": "which audit plans were rejected and returned for modification",
        "expected_contains": ["return"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "review.action = 'RETURN_FOR_MODIFICATION', entity_type='AUDIT_PLAN'",
    },
    {
        "id": "APPR-05", "category": "Approval",
        "question": "who approved the adequacy review for this corrective action",
        "expected_contains": ["adequacy"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "review_type='ADEQUACY', status='COMPLETED' AND action='APPROVE'",
    },
    {
        "id": "APPR-06", "category": "Approval",
        "question": "which corrective actions passed level 2 implementation sign-off",
        "expected_contains": ["implementation"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "review_type='IMPLEMENTATION', level=2, status='COMPLETED' AND action='APPROVE'",
    },
    {
        "id": "APPR-07", "category": "Approval",
        "question": "how many audit plans are pending at each review level",
        "expected_contains": ["pending"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "status='PENDING' GROUP BY level — pending has no action yet, action IS NULL",
    },
    {
        "id": "APPR-08", "category": "Approval",
        "question": "what percentage of completed reviews were rejections rather than approvals",
        "expected_contains": ["%"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "COUNT FILTER (action='RETURN_FOR_MODIFICATION') / COUNT FILTER (status='COMPLETED') — "
                 "confirmed live around 28% overall; this question directly targets the trap",
    },
    {
        "id": "APPR-09", "category": "Approval",
        "question": "who is assigned to review this audit corrective action",
        "expected_contains": ["assign"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "review.assigned_to_id → users — distinct from reviewed_by (who actually completed it)",
    },
    {
        "id": "APPR-10", "category": "Approval",
        "question": "which auditor portfolios were approved and who approved them",
        "expected_contains": ["approv"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "entity_type='PORTFOLIO', status='COMPLETED' AND action='APPROVE' — PORTFOLIO is "
                 "auditor_portfolio, NOT INSPECTION_PORTFOLIO",
    },
    {
        "id": "APPR-11", "category": "Approval",
        "question": "show the full review history for this audit schedule including who reviewed it and the outcome",
        "expected_contains": ["review"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "entity_type='SCHEDULE' — the auditee's accept/reschedule step, level 1, rare level 2; "
                 "'outcome' means action column, not just status",
    },
    # ── Approval regression cases — added from live-traffic failures (2026-07-27) ──
    # These reproduce the EXACT phrasing that broke before their respective fixes,
    # so a future regression in the underlying rule/prompt shows up here directly.
    {
        "id": "APPR-12", "category": "Approval",
        "question": "show all corrective actions and who approved their implementation statuses level 1 and level 2",
        "expected_contains": ["approv"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "REGRESSION: live-caught burning 6 LLM calls / 3 SQL errors / 27s / 64K tokens — the "
                 "model joined rev.entity_id = aca.corrective_action_id (VARCHAR display code) instead "
                 "of aca.id (UUID PK), oscillating between an uncast type error and an invalid ::uuid "
                 "cast before self-correcting via two get_schema calls. Guarded now by "
                 "audit_block_review_entity_id_corrective_action_id — this case should resolve in 1 call.",
    },
    {
        "id": "APPR-13", "category": "Approval",
        "question": "list corrective actions with implementation status approved",
        "expected_contains": ["approv"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "REGRESSION: live-caught — model generated implementation_status = 'APPROVED' "
                 "(plausible English, wrong enum literal), silently matching 0 of 5 real rows with no "
                 "error. Real value is 'APPROVE'. Guarded now by both an AUTOCORRECT rule "
                 "(audit_autocorrect_implementation_status_approved) and dynamic enum introspection "
                 "(audit_corrective_action.implementation_status now startup-verified, not just static "
                 "prose) — expect a nonempty, single-call result.",
    },
    {
        "id": "APPR-14", "category": "Approval",
        "question": "which corrective actions got two level implementation approval and who approved each level",
        "expected_contains": ["level"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Two-join review pattern (r1/r2 both review_type='IMPLEMENTATION', "
                 "status='COMPLETED', action='APPROVE') — confirmed live working in 1 call the same "
                 "day APPR-12's join trap was fixed; kept as a positive-path companion to APPR-12.",
    },
    {
        "id": "APPR-15", "category": "Approval",
        "question": "which corrective actions were returned for modification during implementation review and by whom",
        "expected_contains": ["return"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "action = 'RETURN_FOR_MODIFICATION' (the rejection path, not COMPLETED+APPROVE) — "
                 "companion negative case to the approve-only questions above.",
    },

    # ── Workflow lifecycle ──────────────────────────────────────────────────
    # Added 2026-07-28 after a full audit-lifecycle walkthrough confirmed live
    # with the domain owner (see docs/audit_workflow.md) — exercises facts that
    # were previously undocumented or actively wrong in this codebase's own
    # comments: the unplanned-audit fast-track, multi-facility plan→schedule
    # mapping, post-approval portfolio edits, progress_stage ownership, the
    # adequacy 2nd-level trigger, and manual (non-automatic) CA closure.
    {
        "id": "WORKFLOW-01", "category": "Workflow",
        "question": "list unplanned audit schedules along with their auditor",
        "expected_contains": ["auditor"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "is_unplanned=true; audit_schedule.audit_plan_id is NOT guaranteed populated for "
                 "these rows — must not require joining audit_plan at all to answer this.",
    },
    {
        "id": "WORKFLOW-02", "category": "Workflow",
        "question": "how many audit schedules are unplanned versus tied to an approved plan",
        "expected_contains": ["unplanned"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "CASE WHEN is_unplanned THEN ... grouping — planned-side rows must use LEFT JOIN "
                 "audit_plan, never INNER JOIN, or every unplanned row silently vanishes with no "
                 "error (same failure class as the CA multi-axis INNER JOIN trap).",
    },
    {
        "id": "WORKFLOW-03", "category": "Workflow",
        "question": "which facilities does this audit plan cover",
        "expected_contains": ["facilit"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "audit_plan has no facility_id of its own — reached via audit_plan → "
                 "audit_schedule (1:many) → facility; one plan can span multiple facilities via "
                 "multiple schedule rows.",
    },
    {
        "id": "WORKFLOW-04", "category": "Workflow",
        "question": "which audit plans have schedules across more than one facility",
        "expected_contains": ["plan"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "COUNT(DISTINCT facility_id) per audit_plan_id HAVING > 1 — directly exercises the "
                 "multi-facility-per-plan structure rather than just the single-plan case above.",
    },
    {
        "id": "WORKFLOW-05", "category": "Workflow",
        "question": "which auditor portfolios are under modification after already being approved",
        "expected_contains": ["portfolio"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "Confirmed live: a super-admin/audit-in-charge-permission user can edit an "
                 "already-APPROVED auditor_portfolio row — status='MODIFICATION' is a real, "
                 "distinct branch here, not an error state implying the row must return to a "
                 "pending status first.",
    },
    {
        "id": "WORKFLOW-06", "category": "Workflow",
        "question": "which corrective actions are on track versus behind versus at risk right now",
        "expected_contains": ["track"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "progress_stage grouping — reflects the internal stakeholder's (FM manager's) work "
                 "progress, NOT the auditee's; corrects an earlier 'auditee only' assumption in this "
                 "codebase (see docs/audit_workflow.md §4 / semantics_registry.py's ca_workflow). "
                 "expected_nonempty=False is deliberate, not a shrug: confirmed live against the QA "
                 "staging DB (2026-07-28) that all 11 adequacy-APPROVEd CAs have already progressed "
                 "past this window (8 COMPLETED, 1 CLOSE_WITH_DEFERRED, 2 NULL) — zero currently sit "
                 "in ON_TRACK/BEHIND/AT_RISK with adequacy approved. The SQL is correct; an empty "
                 "result here is the true answer for this dataset, not a failure signal.",
    },
    {
        "id": "WORKFLOW-07", "category": "Workflow",
        "question": "which corrective actions needed a second level of adequacy approval",
        "expected_contains": ["adequacy"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "review.entity_type='AUDIT_CORRECTIVE_ACTION', review_type='ADEQUACY', level=2 — "
                 "confirmed live this only happens when the auditor explicitly requests further "
                 "approval (a frontend checkbox), not randomly; MUST compute from review itself, "
                 "never review_level (see APPR-02/APPR-03).",
    },
    {
        "id": "WORKFLOW-08", "category": "Workflow",
        "question": "which corrective actions have both implementation and adequacy approved but are not yet closed",
        "expected_contains": ["approv"],
        "expected_nonempty": False,
        "expected_llm_calls": 1,
        "notes": "implementation_status='APPROVE' AND adequacy_status='APPROVE' AND status NOT IN "
                 "('CLOSED','CLOSE_WITH_DEFERRED') — directly exercises the confirmed-live fact that "
                 "closing is a SEPARATE MANUAL auditor action, never automatic once both tracks clear.",
    },
]

# ── Multi-turn sequences ─────────────────────────────────────────────────────
# Category "Multi-turn" shares ONE ConversationSession across all its entries,
# run in TEST_SET order (see run_eval()'s session=session if cat=="Multi-turn"
# else None) — unlike every other category, these turns build on each other for
# real, the same way a live chat session does. This specific 3-turn sequence
# reproduces a live session (2026-07-27) that exercised, in order: the
# entity_id/aca.id join fix (turn 1), aggregate-vs-listing follow-up handling
# (turn 2 is a COUNT, turn 3 must NOT confuse it with turn 1's row-level list),
# and the REVIEW_COMPLETED_NOT_APPROVED safety net (turn 3 originally dropped
# action='APPROVE' and self-corrected in 1 retry) — all three turns succeeded
# live; kept here as a standing regression guard for the full chain, not just
# the individual fixes in isolation.
TEST_SET += [
    {
        "id": "MULTI-01", "category": "Multi-turn",
        "question": "show all corrective actions and who approved their implementation statuses level 1 and level 2",
        "expected_contains": ["approv"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Turn 1/3 — establishes the row-level list this sequence's later turns refer back to.",
    },
    {
        "id": "MULTI-02", "category": "Multi-turn",
        "question": "how many of the above are closed",
        "expected_contains": ["closed"],
        "expected_nonempty": True,
        "expected_llm_calls": 1,
        "notes": "Turn 2/3 — a scalar COUNT follow-up. Must reuse turn 1's filters, not drift to an "
                 "unrelated aggregate.",
    },
    {
        "id": "MULTI-03", "category": "Multi-turn",
        "question": "show details",
        "expected_contains": ["approv"],
        "expected_nonempty": True,
        "expected_llm_calls": 2,
        "notes": "Turn 3/3 — deliberately terse, no pronoun ('details' isn't in _FOLLOWUP_RE). Must "
                 "resolve to turn 1's 15-row listing, NOT turn 2's scalar COUNT result — live-confirmed "
                 "the model gets this right by reading raw conversation history even without an "
                 "explicit ACTIVE CONTEXT hint. Also live-confirmed to trip REVIEW_COMPLETED_NOT_APPROVED "
                 "on the first attempt (dropped action='APPROVE') and self-correct in 1 retry — "
                 "expected_llm_calls=2 reflects that known, acceptable one-retry pattern rather than a "
                 "regression if it recurs; only flag if it exceeds 2.",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

# Phrases that indicate the agent returned no usable data.
_EMPTY_PHRASES = (
    "the query returned no results",
    "no answers found",
    "no results found",
    "there were no",
    "there is no data",
    "no data to calculate",
    "no matching",
    "could not find",
    "found 0 result",
    "returned 0",
)


def _is_empty_result(answer: str) -> bool:
    al = answer.lower()
    return any(p in al for p in _EMPTY_PHRASES)


# ── Runner ───────────────────────────────────────────────────────────────────

import sys, os, time, json, argparse
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_eval(model_pair_label="llama3.1:8b + deepseek:6.7b",
             skip_multiturn=False, category_filter=None, output_file=None):

    from src.agent.orchestrator import AgentOrchestrator, ConversationSession
    from dotenv import load_dotenv
    load_dotenv()

    tests = TEST_SET
    if category_filter:
        tests = [t for t in tests if t["category"].lower() == category_filter.lower()]
    if skip_multiturn:
        tests = [t for t in tests if t["category"] != "Multi-turn"]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Audit Domain NL2SQL Evaluation — {model_pair_label}")
    print(f"  {len(tests)} questions  |  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{sep}\n")

    orch    = AgentOrchestrator.from_env(domain="audit")
    results = []
    session = ConversationSession()

    for i, test in enumerate(tests):
        cat  = test["category"]
        qid  = test["id"]
        q    = test["question"]
        print(f"[{i+1:02d}/{len(tests)}] {qid} ({cat})  Q: {q[:65]}")
        t0 = time.time()
        try:
            result = orch.query(q, session=session if cat == "Multi-turn" else None,
                                visibility_scope={})
            wall   = time.time() - t0
            al     = result.answer.lower()

            # ── keyword scoring ──────────────────────────────────────────────
            keywords = test["expected_contains"]
            kt = len(keywords)
            kh = sum(1 for kw in keywords if kw.lower() in al)
            kp = round(kh / kt * 100) if kt else None

            # ── sql / parse error counting ───────────────────────────────────
            se = sum(1 for s in result.steps
                     if s.tool == "execute_sql" and not s.result.get("success"))
            pe = sum(1 for s in result.steps if s.tool == "[parse_error]")

            # ── empty-result detection ───────────────────────────────────────
            empty = _is_empty_result(result.answer)

            # ── ground-truth comparison — no audit cases in eval_truth.py yet,
            # so this always reports covered=False; kept for parity with
            # eval.py and so a future audit GROUND_TRUTH addition needs no
            # runner change here. Looked up under an "AUDIT_" prefix, NOT the
            # bare qid — GROUND_TRUTH is a flat dict shared with eval.py's
            # inspection suite, and inspection already defines its own
            # "PORT-01" ("inspectors with portfolios", inspector_portfolio
            # table). Confirmed live: looking up bare "PORT-01" here silently
            # matched inspection's entry and ran inspector_portfolio.
            # inspector_id (a column that doesn't exist for audit) against
            # this domain's DB, surfacing as a spurious truth-layer SQL error
            # on an otherwise-correct audit answer. ─────────────────────────
            try:
                from eval_truth import run_truth_check, extract_agent_rows
                tm = run_truth_check(orch.db_engine, f"AUDIT_{qid}", extract_agent_rows(result))
            except Exception as _te:
                tm = {"covered": False, "truth_match": None, "detail": f"err: {_te}"}

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
                truth_covered=tm["covered"],
                truth_match=tm["truth_match"],
                truth_detail=tm["detail"],
                answer_preview=result.answer[:250],
                success=result.success,
                manual_score=None,
                notes=test["notes"],
            )
            results.append(row)

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

    ff_rows = [r for r in results if r["forced_fail"]]
    if ff_rows:
        print(f"\n  FORCED FAILS (empty result where data expected):")
        for r in ff_rows:
            print(f"    {r['id']:<8} {r['question'][:60]}")

    # ── save ─────────────────────────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = output_file or f"eval_audit_results_{ts}.json"
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
    p = argparse.ArgumentParser(description="Audit Domain NL2SQL Evaluation")
    p.add_argument("--model-pair", default="llama3.1:8b + deepseek:6.7b")
    p.add_argument("--skip-multiturn", action="store_true")
    p.add_argument("--category", default=None,
                   help="Plan | Schedule | Status | Cycle | Forms | Admin | Portfolio | Approval | Workflow")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    results = run_eval(a.model_pair, a.skip_multiturn, a.category, a.output)
    sys.exit(0 if all(r["real_pass"] for r in results) else 1)
