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
        "forbidden_contains": ["draft"],
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
        "forbidden_contains": ["draft"],
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
    {
        "id": "SCHED-10", "category": "Schedule",
        "question": "which audit schedules are still in draft",
        "expected_contains": ["draft"],
        "expected_nonempty": False,
        "expected_llm_calls": 1, "notes": "schedule_status = 'DRAFT' — explicit-ask carve-out, mirrors PLAN-02",
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

import sys, os, time, json, argparse, re
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_table_mention_re_cache: dict[str, "re.Pattern"] = {}


def _table_mentioned(text: str, tables: set) -> bool:
    """See eval.py's identical helper for the full rationale (Phase 3.12)."""
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
             profile_tokens=False):

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

    orch    = AgentOrchestrator.from_env(domain="audit", profile=profile_tokens)
    results = []
    session = ConversationSession()

    # ── Phase 3.10: static schema-block composition, computed ONCE — see
    # eval.py's identical block for the full rationale (DatabaseSchema.
    # profile_slim_schema_tokens' content is question-independent).
    _schema_profile = None
    _schema_tables_cost = {}
    _table_used_count = defaultdict(int)
    _table_unused_count = defaultdict(int)
    _n_utilization_queries = 0
    # See eval.py's identical accumulator for why "targeted" queries are
    # tracked and excluded separately from the slim-table diff.
    _n_targeted_variant_queries = 0

    # ── Phase 3.11: targeted-branch decomposition — see eval.py's identical
    # block for the full rationale.
    _targeted_schema_profile = None
    _targeted_call_samples = []

    # ── Phase 3.12: relevance of the two fixed blocks — see eval.py's
    # identical block for the full rationale.
    _fixed_overhead_profile = None
    _fixed_item_relevant_named = defaultdict(int)
    _fixed_item_relevant_used = defaultdict(int)
    _fixed_overhead_calls = []
    if profile_tokens:
        _sql_base_url = getattr(orch.sql_llm, "api_base", "") or getattr(orch.sql_llm, "base_url", "")
        _schema_profile = orch.db_schema.profile_slim_schema_tokens(_sql_base_url)
        _schema_tables_cost = {**_schema_profile["key_columns"], **_schema_profile["lookup_tables"]}
        _targeted_schema_profile = orch.db_schema.profile_targeted_schema_tables(_sql_base_url)
        _fixed_overhead_profile = orch.db_schema.profile_fixed_overhead_relevance(_sql_base_url)

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

            # ── forbidden-keyword scoring — e.g. "draft" must never appear in
            # a general listing's answer (PLAN-01/SCHED-01 regression guard
            # for the DRAFT-exclusion filter-dropping bug, 2026-08-18) ──────
            forbidden = test.get("forbidden_contains", [])
            forbidden_hit = next((kw for kw in forbidden if kw.lower() in al), None)

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

            forced_fail = bool(test.get("expected_nonempty") and empty) or bool(forbidden_hit)
            real_pass   = result.success and not forced_fail

            # ── Phase 3.10: schema-block utilization — see eval.py's
            # identical block for the full rationale, including why
            # "targeted"-variant queries are excluded from this diff.
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
                            # See eval.py's identical block for the full
                            # rationale (Phase 3.11 exact reconstruction).
                            if _targeted_schema_profile:
                                _tt = _sh_bd.get("targeted_tables") or []
                                _pt = _targeted_schema_profile["per_table"]
                                _cols_tok = sum(_pt.get(t, {}).get("columns_tokens", 0) for t in _tt)
                                _fk_tok = sum(_pt.get(t, {}).get("fk_tokens", 0) for t in _tt)
                                _enum_tok = sum(_pt.get(t, {}).get("enum_tokens", 0) for t in _tt)
                                _useful_tok = sum(_pt.get(t, {}).get("useful_output_tokens", 0) for t in _tt)
                                _header_tok = sum(_pt.get(t, {}).get("header_tokens", 0) for t in _tt)
                                # Phase 3.13 — see eval.py's identical block
                                # for the full rationale (for_schema_hint's
                                # _fmt_lookup() is now scoped per call).
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

                            # ── Phase 3.12 — see eval.py's identical block
                            # for the full rationale.
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
                forbidden_hit=forbidden_hit,
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
            status = ("✅" if real_pass else
                      f"❌ FORBIDDEN({forbidden_hit})" if forbidden_hit else
                      "❌ EMPTY" if forced_fail else "❌")
            print(f"         {perf} {wall:.1f}s | {result.stats.total_llm_calls} calls | "
                  f"kw {kw_str} | SQL {'OK' if se == 0 else str(se) + ' err'} | {status}")
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

    # ── latency breakdown — Phase 3.1 profiling ────────────────────────────
    # wall_s vs. (reasoning_ms + sql_gen_ms + sql_exec_ms) — the gap is real
    # unaccounted overhead (turn-classifier regex/fuzzy-vocab work, scope
    # filter building, JSON parsing, tool dispatch, network round-trips
    # themselves, etc.), NOT a bug — see QueryStats.total_sql_gen_ms.
    _ok = [r for r in results if r["llm_calls"] >= 0]
    _tot_wall_ms   = sum(r["wall_s"] * 1000 for r in _ok)
    _tot_reason_ms = sum(r["reasoning_ms"] for r in _ok)
    _tot_sqlgen_ms = sum(r["sql_gen_ms"]   for r in _ok)
    _tot_sqlexec_ms= sum(r["sql_exec_ms"]  for r in _ok)
    _tot_other_ms  = max(0, _tot_wall_ms - _tot_reason_ms - _tot_sqlgen_ms - _tot_sqlexec_ms)
    _pct = lambda part: round(100 * part / _tot_wall_ms) if _tot_wall_ms else 0

    # ── prefill vs decode split — Phase 3.2 profiling (llama.cpp only,
    # zero on other providers — see QueryStats.reasoning_prefill_ms) ──────
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
    # the full prompt. ──────────────────────────────────────────────────
    _tot_auto_tok = sum(r["sql_gen_auto_schema_tokens"]     for r in _ok)
    _tot_mh_tok   = sum(r["sql_gen_mandatory_hint_tokens"]  for r in _ok)
    _tot_notes_tok= sum(r["sql_gen_planner_notes_tokens"]   for r in _ok)
    _tot_jp_tok   = sum(r["sql_gen_join_path_tokens"]       for r in _ok)
    _tot_br_tok   = sum(r["sql_gen_business_rules_tokens"]  for r in _ok)
    _tot_draft_tok= sum(r["sql_gen_draft_sql_tokens"]       for r in _ok)
    _sh_pct = lambda part: round(100 * part / _tot_sh_tok, 1) if _tot_sh_tok else 0.0

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
        # derived by subtraction — if it goes negative or the printed total
        # doesn't equal _tot_sh_tok exactly, an atom was missed; stop and
        # debug before trusting the distribution.
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

    if _schema_profile and _n_utilization_queries:
        # Phase 3.10 — Part A: static composition (see eval.py's identical
        # block for the full rationale).
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
        # tables the block spent tokens describing?
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
        # Phase 3.11 — see eval.py's identical block for the full rationale.
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
        _tot_slack = _tot_real - _tot_reconstructed
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

        _avg_selected = sum(s["table_count"] for s in _targeted_call_samples) / _n_tgt
        _avg_sql = sum(s["sql_table_count"] for s in _targeted_call_samples) / _n_tgt
        print(f"\n  TARGETED-BRANCH TABLE SELECTION (Phase 3.11, hypothesis A)")
        print(f"    avg tables named (rendered): {_avg_selected:.1f}")
        print(f"    avg tables in generated SQL: {_avg_sql:.1f}")
        if _avg_sql:
            print(f"    over-selection ratio       : {_avg_selected/_avg_sql:.2f}x")
        else:
            print(f"    over-selection ratio       : n/a (no SQL tables extracted)")

        print(f"\n  TARGETED-BRANCH real_total DISTRIBUTION (Phase 3.11, {_n_tgt} calls)")
        print(f"    min={min(_real_totals):>5d}  p50={_pctile(_real_totals,0.50):>5d}  p75={_pctile(_real_totals,0.75):>5d}  "
              f"p90={_pctile(_real_totals,0.90):>5d}  p95={_pctile(_real_totals,0.95):>5d}  max={max(_real_totals):>5d}")

    if _fixed_overhead_calls:
        # Phase 3.12 — see eval.py's identical block for the full rationale.
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

    fb_rows = [r for r in results if r.get("forbidden_hit")]
    if fb_rows:
        print(f"\n  FORBIDDEN-KEYWORD HITS (leaked content that should be filtered out):")
        for r in fb_rows:
            print(f"    {r['id']:<8} {r['question'][:50]:<52} hit={r['forbidden_hit']!r}")

    ff_rows = [r for r in results if r["forced_fail"] and not r.get("forbidden_hit")]
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
            # Phase 3.10 profiling — absent unless run with --profile-tokens.
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
            # Phase 3.11 profiling — see eval.py's identical block for the
            # full field rationale.
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
            # Phase 3.12 profiling — see eval.py's identical block for the
            # full field rationale.
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
    p = argparse.ArgumentParser(description="Audit Domain NL2SQL Evaluation")
    p.add_argument("--model-pair", default="llama3.1:8b + deepseek:6.7b")
    p.add_argument("--skip-multiturn", action="store_true")
    p.add_argument("--category", default=None,
                   help="Plan | Schedule | Status | Cycle | Forms | Admin | Portfolio | Approval | Workflow")
    p.add_argument("--output", default=None)
    p.add_argument("--profile-tokens", action="store_true",
                   help="Phase 3.8: tokenize dynamic prompt components (question/"
                        "schema_hint/core_context/retrieved examples) via llama.cpp's "
                        "/tokenize endpoint and report the breakdown. Adds a few extra "
                        "HTTP round-trips per generate_sql call; off by default.")
    a = p.parse_args()
    results = run_eval(a.model_pair, a.skip_multiturn, a.category, a.output, a.profile_tokens)
    sys.exit(0 if all(r["real_pass"] for r in results) else 1)
