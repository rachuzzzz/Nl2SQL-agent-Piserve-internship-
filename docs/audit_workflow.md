# Audit Domain Workflow

This document captures the full audit lifecycle as confirmed live with the domain owner on 2026-07-28, reconciled against the actual schema and the NL2SQL codebase's existing rules. It's the narrative companion to the enforced/generated logic living in:

- `backend/src/knowledge/semantics_registry.py` — `AUDIT_SEMANTICS` (status/workflow data feeding LLM-facing prompt blocks)
- `backend/src/knowledge/schema_introspector.py` — `AUDIT_BUSINESS_RULES` (prose facts surfaced to the SQL-generation model)
- `backend/src/sql/audit_join_graph.py` — table join paths
- `backend/src/sql/rules/audit_rules.py` — hard-enforced validator BLOCK/WARN/INJECT rules

If this document and the code ever disagree, the code (and its "confirmed live" comments) is the source of truth for what's actually enforced — update this document to match, not the other way around.

## 1. Cycle → Portfolio → Plan (the planning chain)

1. **A cycle is created** (`audit_cycle`: `start_date`, `end_date`, `due_date`, `status`). There is no creator/owner column on this table and no RBAC gate on who may create one — "audit in charge creates cycles" is a real business convention with **no schema or permission enforcement** behind it in this codebase.
2. **Auditors submit portfolios** against a cycle (`auditor_portfolio.audit_cycle_id`, `user_id`). Live data only shows `status IN ('APPROVED', 'MODIFICATION')` — no `PENDING`/`SUBMITTED` value has been observed, so "submitted" just means the row exists, pre-approval.
3. **Portfolio approval** is tracked two ways in parallel: the polymorphic `review` table (`entity_type='PORTFOLIO'`, normally 1 level) **and** `auditor_portfolio.status` directly. As with every use of `review`, `status='COMPLETED'` is not the same as approved — always also check `action='APPROVE'` (28% of completed reviews are rejections, confirmed live).
4. **Post-approval edits are a real, separate branch**: a super-admin/audit-in-charge-permission user can modify an `auditor_portfolio` row *after* it already reached `status='APPROVED'`. This is not an error state and doesn't imply the row should have gone back through a pending/returned status first.
5. **The auditor submits an audit plan** (`audit_plan.status`: `DRAFT → SUBMITTED → APPROVED`, with `SUBMITTED → RETURNED_FOR_MODIFICATION → SUBMITTED` as a re-file loop). `DRAFT` is never counted in reporting.
6. **The plan goes through 2-level review** (`review.entity_type='AUDIT_PLAN'`). The two levels are **never named by role** anywhere in this codebase, deliberately — see §5 below on why role names are never hardcoded.

## 2. Plan → Schedule → Conduct (the execution chain)

1. **`audit_plan` has no facility of its own.** Facilities only exist at the `audit_schedule` level. One `audit_plan` can produce **multiple `audit_schedule` rows** (`audit_schedule.audit_plan_id → audit_plan.id`, 1:many) — this is how a single plan covers multiple facilities. Each individual schedule is scoped to exactly one `facility_id` and one `auditee_id`.
2. **The auditee accepts or reschedules** — this is `review.entity_type='SCHEDULE'` (normally 1 level, rare 2nd). `audit_schedule.reschedule_requested`/`reschedule_count` are not a separate mechanism; they mirror this same review step, denormalized onto the schedule row.
3. **The auditor conducts the audit** on the accepted date, choosing/using a form template (`audit_report.audit_form_used_id → audit_forms.id`, confirmed always populated). `audit_forms` is a template library — `audit_forms.audit_schedule_id` is always NULL by design. There is **no schema mechanism to tell whether the auditor modified the form's content** for a specific audit — only which template was used. Confirmed: this is not currently needed by the business.
4. **The audit report itself is never reviewed** via the polymorphic `review` table — unlike plan/portfolio/schedule/corrective-action, `audit_report.status` moves `DRAFT → SUBMITTED` directly, then eventually `CLOSED`/`CLOSED_WITH_EXCEPTION` (both schema-valid; not yet observed in live data at time of writing).

## 3. Unplanned Audits — a full fast-track, not a variant

`audit_schedule.is_unplanned = true` is **not** just "a schedule missing its plan." Confirmed live:

- Created **directly by a super-admin-permission user**.
- Skips the entire `audit_plan` / portfolio / cycle chain.
- Skips the auditee accept/reschedule review step entirely (no `SCHEDULE`-level review row).
- Goes straight to the auditor conducting it.

Consequence: `audit_schedule.audit_plan_id` is **not guaranteed populated**. Any query that needs to include unplanned audits alongside planned ones must `LEFT JOIN audit_plan`, never `INNER JOIN` — an inner join silently drops every unplanned row with no error or row-count signal (the same silent-loss failure class as the corrective-action multi-axis join trap documented in `audit_rules.py`).

`audit_plan.is_unplanned` is a **separate, much rarer** flag — the *plan itself* created ad hoc — not to be conflated with the schedule-level flag above.

## 4. Corrective Action Lifecycle

1. **The auditee (or whoever raises it) enters a corrective action** against the audit as a whole (`audit_corrective_action.audit_id → audit_report.id`). Unlike inspection's `ica.observation_id`, **there is currently no finding/observation-level link** — a CA cannot be traced back to one specific finding within the report, only to the report overall. This is a known, accepted current limitation: the audit-side form-builder tables aren't normalized yet, and an `ai_audit_answers` table (mirroring inspection's `ai_answers`) is planned, which is the natural point to revisit this.
2. Three independent text fields exist on the CA, not to be confused with each other: `cause` (why it happened), `correction` (the immediate fix already applied), `corrective_action` (the proposed root-cause/longer-term action).
3. **`corrective_action_id` displays as the literal string `'CA_DRAFT'`** until `adequacy_status = 'APPROVE'` — a CA_DRAFT row is still a real, existing corrective action, just not yet adequacy-approved. (For existence checks — "was any CA raised" — a CA_DRAFT row still counts; never gate existence on adequacy status.)
4. **Adequacy review** is normally a single auditor-side level. A 2nd level does occur, but **it is not random or a data quality issue** — it fires when the auditor explicitly checks a "needs further approval" option in the frontend. There is no dedicated schema column tracking this trigger.
5. **Implementation is carried out by the internal stakeholder** (the "FM manager" role, `aca.internal_stakeholder_id`) — they do the actual work. `progress_stage` reflects *their* progress (`NOT_STARTED → ON_TRACK/BEHIND/AT_RISK → COMPLETED`, or `CLOSE_WITH_DEFERRED`). This corrects an earlier assumption in this codebase that progress_stage was auditee-owned.
6. **The auditee then verifies** that work as their side of the two-level `implementation_status` approval (`NULL → PENDING_FOR_REVIEW → APPROVE`/`RETURN_FOR_MODIFICATION`) — verification, not the work itself.
7. **The auditor gives final implementation sign-off.**
8. **Closing is a separate, manual auditor action** — after both `implementation_status = 'APPROVE'` and `adequacy_status = 'APPROVE'` are in place, the auditor manually sets `status` to `CLOSED` or `CLOSE_WITH_DEFERRED`. This does **not** happen automatically as a side effect of the second approval landing — there is no trigger enforcing it, only an observed correlation.

Enum trap that recurs constantly: the exact string is `'APPROVE'`, never `'APPROVED'`, for both `implementation_status` and `adequacy_status`.

## 5. On Roles ("audit in charge", "super admin", "FM manager")

None of these are fixed, hardcoded roles anywhere in this codebase, and that's deliberate. Role names are labels attached to a **dynamically admin-configurable permission set** — the same permission bundle can be assigned under different labels, and occasionally an unexpected role label ends up holding a given permission ("weird roles doing everything," confirmed by the domain owner). `security/rbac.py` states this outright: *role names must never gate scope decisions, only permission strings do.* Every review-level description in this codebase deliberately describes the workflow **stage** (`entity_type` + `review_type` + `level`) rather than naming who sits at each level — join to `users`/`user_roles`/`role` only when a question explicitly asks *who* reviewed something, never as a correctness gate.

## Open / accepted limitations (not bugs)

- No observation/finding-level link for audit corrective actions (planned: `ai_audit_answers`).
- No tracking of whether a form template was modified for a specific audit — confirmed not needed currently.
- No creator/owner column on `audit_cycle`, no RBAC enforcement of who may create one.
