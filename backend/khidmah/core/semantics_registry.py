"""
semantics_registry.py — SINGLE SOURCE OF TRUTH for domain semantics.

Every semantic rule that multiple layers must agree on lives HERE and only here:
  - status groupings (conducted / completed / draft exclusion)
  - join paths and cast directions
  - column existence facts

Consumers (all import from this module — never hardcode these rules):
  1. agent/prompts.py          → SYSTEM_PROMPT status section   (llama_status_block)
  2. agent/seed_examples.py    → CORE_CONTEXT                   (core_context_blocks)
  3. core/business_rules.py    → INJECT rule hints              (inject_hint_*)
  4. agent/orchestrator.py     → _CONSTRAINT_NOTE_MAP           (constraint_note_*)
  5. core/validator.py         → (rules reference SEMANTICS for value lists)

To change a definition: edit SEMANTICS below, restart. All layers update together.
Run `python -m core.semantics_registry` to print all generated blocks for review.

History: before this module existed the conducted/completed definition was
hardcoded in 5 places and they desynced 3 times in one day (2026-06-10).
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# THE CANONICAL DATA — edit here, nowhere else
# ═══════════════════════════════════════════════════════════════════════════

SEMANTICS: dict = {

    # ── inspection_report.status lifecycle ─────────────────────────────────
    "status": {
        "all_values": [
            "DRAFT", "SUBMITTED", "UNDER_REVIEW",
            "RETURN_FOR_MODIFICATION", "CLOSED",
        ],
        "lifecycle": (
            "DRAFT → SUBMITTED → UNDER_REVIEW → CLOSED, "
            "with UNDER_REVIEW → RETURN_FOR_MODIFICATION → SUBMITTED (re-filed)"
        ),
        # Semantic groupings — the user-facing vocabulary
        "groupings": {
            "conducted": {
                "statuses": ["SUBMITTED", "UNDER_REVIEW", "RETURN_FOR_MODIFICATION"],
                "meaning":  "in-flight — inspector filed, not yet fully approved",
            },
            "completed": {
                "statuses": ["CLOSED"],
                "meaning":  "fully reviewed and approved (final)",
            },
        },
        # DRAFT is globally excluded — never counted, listed, or aggregated
        "never_counted": ["DRAFT"],
        # "total inspections" presentation contract
        "total_presentation": (
            "two-row category summary via "
            "CASE WHEN ir.status = 'CLOSED' THEN 'completed' ELSE 'conducted' END"
        ),
        "forbidden_filters": [
            {
                "pattern": "status IN ('CLOSED','SUBMITTED')",
                "why":     "mixes conducted and completed — has no domain meaning",
            },
            {
                "pattern": "status != 'DRAFT' as the only filter for conducted",
                "why":     "includes CLOSED, which is completed not conducted",
            },
        ],
    },

    # ── Join paths with non-obvious cast directions ─────────────────────────
    "joins": {
        "observation_to_ma": {
            "sql":   "aa.submission_id = ica.observation_id::varchar",
            "left":  ("ai_answers.submission_id", "varchar"),
            "right": ("inspection_corrective_action.observation_id", "uuid"),
            "rule":  "cast the UUID side to varchar — varchar→uuid cast can throw on bad data",
            "never": [
                "aa.submission_id::uuid = ica.observation_id",
                "ica.observation_id::text = aa.submission_id",
            ],
        },
        "ma_to_po": {
            "sql":   "po.mitigative_action_uuid = ica.id",
            "never": [
                "ica.mitigative_action_uuid = po.*  (column does not exist on ica)",
                "LEFT JOIN po_details directly when SUM(ica.capex) present (row multiplication)",
            ],
        },
    },

    # ── Counting contracts ───────────────────────────────────────────────────
    "counting": {
        "good_practice": {
            "expr": "COUNT(DISTINCT ir.id)",
            "why":  ("each inspection is Good Practice OR Deviation (mutually exclusive); "
                     "COUNT(*) on ai_answers rows overcounts"),
        },
        "deviation": {
            "expr": "COUNT(DISTINCT ir.id)",
            "why":  "same rule as good_practice",
        },
        "observations_per_inspection": {
            "expr": "COUNT(DISTINCT aa.submission_id)",
            "why":  ("one submission_id per observation group; "
                     "COUNT(*) counts every answer row (Risk Level, Impact, ...)"),
        },
    },

    # ── Financial routing — capex/opex definitions and join paths ────────────
    "financial": {
        "metrics": {
            "capex_generated": {
                "expr": "SUM(ica.capex)",
                "meaning": "inspectee's original estimate on ICA",
            },
            "capex_converted": {
                "expr": "SUM(po.po_amount)",
                "meaning": "actual PO raised (via po_details)",
            },
            "capex_lost": {
                "expr": "SUM(ica.capex) - COALESCE(SUM(po.po_amount), 0)",
                "meaning": "arithmetic difference — generated minus converted",
                "never": "capex_status='REJECTED'/'DEFERRED' is NOT the loss calculation",
            },
            "opex_generated": {
                "expr": "SUM(ica.opex)",
                "meaning": "inspectee's original estimate on ICA — parallel to capex_generated",
            },
            "opex_incurred": {
                "expr": "SUM(pr.pr_amount)",
                "meaning": "actual PR spend (via pr_details) — a.k.a. 'opex converted', parallel to capex_converted",
            },
        },
        "tables": {
            "po_details": {
                "columns": ["id", "po_amount", "po_date", "po_number", "mitigative_action_uuid"],
                "join": "po.mitigative_action_uuid = ica.id",
                "multiplicity": "one ICA can have MULTIPLE POs",
                "cte_pattern": (
                    "WITH po_by_ica AS (SELECT mitigative_action_uuid AS ica_id, "
                    "SUM(po_amount) AS total_po_amount FROM po_details GROUP BY 1)\n"
                    "  LEFT JOIN po_by_ica p ON p.ica_id = ica.id"
                ),
            },
            "pr_details": {
                "columns": ["id", "pr_amount", "pr_date", "pr_number", "mitigative_action_uuid"],
                "join": "pr.mitigative_action_uuid = ica.id",
                "multiplicity": "one ICA can have MULTIPLE PRs — always SUM(pr.pr_amount)",
            },
        },
        "capex_loss_form": {
            "module": "Inspection Mitigative Action Completion",
            "join": "ica.completion_submission_id = aa.inspection_report_id",
            "labels": {
                "Is CAPEX Lost": "aa.answer_text = 'true' (boolean flag)",
                "Amount Difference for CPAEX Loss": "aa.answer_numeric (NOTE: typo CPAEX not CAPEX)",
                "CAPEX Lost To(Company/Entity/Subcontractor)": "aa.answer_text (who did it cheaper)",
                "Justification for CAPEX Loss": "aa.answer_text (reason)",
            },
        },
        "pr_form": {
            "module": "Inspection Mitigative Action PR Entry",
            "join": "ica.pr_submission_id = aa.inspection_report_id",
            "labels": ["PR Amount", "PR Date", "PR Number"],
        },
        "ownership": [
            "CAPEX is ALWAYS tied to the FM Manager (inspectee_user_id)",
            "Inspectors only raise observations — they have NO role in CAPEX/OPEX",
            "FM Manager (inspectee_user_id) raises the ICA and assigns the CAPEX value",
            "Client raises PO to cover it (CAPEX Converted) or org raises PR (OPEX Incurred)",
            # VERIFIED against live data 2026-07-07: 26/26 distinct inspectee_user_id
            # values in inspection_report hold at least one FM-side role (FM Manager,
            # FM Engineer, AD FM, Asst FM Manager, FM Admin, FM Supervisor — see
            # schema_introspector.py's USER ROLES section for the full EXISTS pattern).
            # 0 exceptions found. This is a data snapshot, not a DB-enforced constraint
            # — if it ever needs re-checking, run the same EXISTS query again; it is
            # NOT currently applied to fm_manager queries (33 locations checked
            # 2026-07-07) since it changes zero results today and the role table
            # itself carries its own caveats (see schema_introspector.py).
        ],
    },

    # ── Mitigative action workflow — four independent-but-coupled tracks ─────
    # Discovered via pgAdmin state-matrix introspection 2026-06-11.
    "ma_workflow": {
        "status": {  # overall MA lifecycle (system-managed)
            "values": ["OPEN", "CLOSED", "OVERDUE", "CLOSE_WITH_DEFERRED"],
        },
        "progress_stage": {  # inspectee's SELF-REPORTED progress (inspectee-owned)
            "values": {
                "NOT_STARTED":         "Not Started",
                "ON_TRACK":            "On Track",
                "BEHIND":              "Behind",
                "AT_RISK":             "At Risk",
                "COMPLETED":           "Completed — triggers PENDING_FOR_REVIEW",
                "CLOSE_WITH_DEFERRED": "Close with deferred — carried to next cycle",
            },
            "owner": "inspectee (FM manager side) — only they update this",
            "trap": ("'at risk' = progress_stage='AT_RISK' (schedule health), "
                     "NOT risk_level='High' (severity) — completely different concepts"),
        },
        "implementation_status": {  # reviewer-owned approval track
            "values": {
                None:                      "inspectee hasn't marked complete yet",
                "PENDING_FOR_REVIEW":      "inspectee completed, awaiting FM manager + inspector",
                "APPROVE":                 "both approvers accepted — MA closes",
                "RETURN_FOR_MODIFICATION": "a reviewer sent it back to the inspectee",
            },
            "owner": "FM manager + inspector (both must approve)",
            "traps": [
                "exact string is 'APPROVE' — NOT 'APPROVED'",
                "exact string is 'PENDING_FOR_REVIEW' — NOT 'PENDING'",
            ],
        },
        "capex_status": {  # financial track — USER-ENTERED, not authoritative
            "values": [None, "ACCEPTED", "REJECTED", "DEFERRED"],
            "caveat": (
                "capex_status is entered by the user and is DECLARATIVE ONLY. "
                "po_details is the AUTHORITATIVE money record — POs legitimately "
                "exist on REJECTED rows (rejected-but-converted is valid data, "
                "not an error). Never infer PO existence from capex_status; "
                "never compute capex loss from it."
            ),
        },
        "invariants": [
            "status = 'CLOSED'  ⟺  implementation_status = 'APPROVE' (closure IS the approval)",
            "implementation_status = 'PENDING_FOR_REVIEW' only co-occurs with progress_stage = 'COMPLETED'",
            "RETURN_FOR_MODIFICATION resets progress_stage to NULL — "
            "so progress_stage = 'COMPLETED' MISSES returned MAs",
            "inspection_report.status = 'UNDER_REVIEW' is driven by MAs awaiting approval",
            "inspection_report closes only when ALL its MAs are closed",
        ],
        "unreliable_columns": {
            "pending_with":    "observed NULL across rows — derive 'who must act' from implementation_status",
            "adequacy_status": "currently NULL, future use",
            "close_on":        "dead column — confirmed always NULL (0/398, 2026-07-08). Use completed_on for closure timing.",
        },
        "timeliness": {
            # confirmed 2026-06-11: completed_on populated for CLOSED rows
            # confirmed 2026-07-08: close_on exists but is always NULL (0/398 rows) — dead, not missing
            "completed_on": "the actual closure timestamp — populated for status='CLOSED' rows",
            "overdue": {
                "meaning": "still OPEN and past deadline — has NOT closed yet",
                "expr": "status = 'OVERDUE'",
            },
            "closed_late": {
                "meaning": "DID close, but after the deadline — mutually exclusive with overdue",
                "expr": "status = 'CLOSED' AND completed_on > target_close_out_date",
            },
            "invariant": (
                "overdue and closed_late are MUTUALLY EXCLUSIVE buckets — "
                "NEVER combine with OR into a single count. "
                "'not closed on time' / 'closed late' = closed_late ONLY, not overdue+closed_late."
            ),
        },
        "vocabulary": {
            "pending approval / awaiting review": "implementation_status = 'PENDING_FOR_REVIEW'",
            "approved":                           "implementation_status = 'APPROVE'",
            "sent back / returned":               "implementation_status = 'RETURN_FOR_MODIFICATION'",
            "at risk / behind / on track":        "progress_stage (schedule health)",
            "not started":                        "progress_stage = 'NOT_STARTED' OR progress_stage IS NULL",
            "overdue":                            "status = 'OVERDUE'",
            "closed late / not closed on time":   "status = 'CLOSED' AND completed_on > target_close_out_date",
        },
    },

    # ── Column existence facts (validator-relevant) ──────────────────────────
    "columns": {
        "no_active_column": [
            "ai_answers", "ai_questions", "role", "user_roles",
        ],
        "has_active_column": [
            "inspection_report", "inspection_corrective_action", "facility", "users",
            "inspection_type", "inspector_portfolio", "inspector_portfolio_details",
            "frequency_definition", "po_details", "pr_details", "risk_level",
            "inspection_schedule",
            # confirmed 2026-07-08: 21/2209 rows are active=false — a real, used soft-delete flag
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Convenience accessors
# ═══════════════════════════════════════════════════════════════════════════

def conducted_statuses() -> list[str]:
    return SEMANTICS["status"]["groupings"]["conducted"]["statuses"]


def completed_statuses() -> list[str]:
    return SEMANTICS["status"]["groupings"]["completed"]["statuses"]


def conducted_in_list() -> str:
    """SQL IN-list literal: ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')"""
    return "(" + ",".join(f"'{s}'" for s in conducted_statuses()) + ")"


def completed_filter() -> str:
    """SQL equality/IN filter for completed. Single status → '= X', else IN-list."""
    sts = completed_statuses()
    if len(sts) == 1:
        return f"= '{sts[0]}'"
    return "IN (" + ",".join(f"'{s}'" for s in sts) + ")"


def case_category_expr() -> str:
    """The canonical two-row summary CASE expression."""
    sts = completed_statuses()
    if len(sts) == 1:
        cond = f"ir.status = '{sts[0]}'"
    else:
        cond = "ir.status IN (" + ",".join(f"'{s}'" for s in sts) + ")"
    return f"CASE WHEN {cond} THEN 'completed' ELSE 'conducted' END"


def draft_exclusion_filter() -> str:
    """SQL filter excluding globally-never-counted statuses (e.g. 'status != \\'DRAFT\\'')."""
    excluded = SEMANTICS["status"]["never_counted"]
    return " AND ".join(f"status != '{s}'" for s in excluded)


def completed_required_regex() -> str:
    """Regex matching the correct 'completed' filter form: status = 'CLOSED'."""
    sts = completed_statuses()
    if len(sts) == 1:
        return rf"status\s*=\s*'{sts[0]}'"
    in_list = "|".join(f"'{s}'" for s in sts)
    return rf"status\s*IN\s*\([^)]*(?:{in_list})[^)]*\)"


def conducted_required_regex() -> str:
    """Regex matching the correct 'conducted' filter form: status IN (...)."""
    in_list = "|".join(f"'{s}'" for s in conducted_statuses())
    return rf"status\s*IN\s*\([^)]*(?:{in_list})[^)]*\)"


def old_completed_forbidden_regex() -> str:
    """Regex matching the OLD WRONG 'completed = CLOSED+SUBMITTED' pattern."""
    return r"status\s*IN\s*\(\s*'CLOSED'\s*,\s*'SUBMITTED'\s*\)"


# ═══════════════════════════════════════════════════════════════════════════
# Generated blocks — one per consumer layer
# ═══════════════════════════════════════════════════════════════════════════

def llama_status_block() -> str:
    """For agent/prompts.py SYSTEM_PROMPT — Llama's understanding of status."""
    g = SEMANTICS["status"]["groupings"]
    return f"""IMPORTANT — inspection_report.status values and groupings:
  DRAFT                    → incomplete, never submitted — NEVER count or show
  SUBMITTED                → inspector filed the report (in-flight)
  UNDER_REVIEW             → under FM/admin review (in-flight)
  RETURN_FOR_MODIFICATION  → sent back to inspector (in-flight)
  CLOSED                   → fully reviewed and approved (final)

  STATUS GROUPINGS — always bucket into two categories for reporting:
    conducted = IN {conducted_in_list()}
    completed {completed_filter()}

  DEFAULT for "total / how many / count inspections":
    Show as TWO-ROW summary using CASE grouping:
      {case_category_expr()} AS category
    schema_hint must use this pattern — NOT a flat COUNT(*) with status != 'DRAFT'

  NEVER: status IN ('CLOSED','SUBMITTED') — mixes conducted and completed
  NEVER: status != 'DRAFT' as the only filter for conducted counts
  ALWAYS exclude DRAFT — add status != 'DRAFT' on all inspection_report queries"""


def core_context_status_block() -> str:
    """For agent/seed_examples.py CORE_CONTEXT — Deepseek's always-on context."""
    return f"""-- DRAFT RECORDS ARE NEVER COUNTED — GLOBAL RULE:
--   DRAFT inspections are incomplete — they must NEVER appear in any count, list, average, or aggregate.
--
-- STATUS GROUPINGS — always bucket into two categories (never expose raw status names to user):
--   conducted  = status IN {conducted_in_list()} — in-flight
--   completed  = status {completed_filter()}{' ' * max(0, 42 - len(completed_filter()))}— fully approved
--
-- DEFAULT for "total inspections / how many inspections / inspections in <month>":
--   Show as TWO-ROW summary using CASE grouping:
--     {case_category_expr()} AS category
--   NEVER show raw status breakdown (SUBMITTED/UNDER_REVIEW/etc) unless user explicitly asks
--   NEVER use status IN ('CLOSED','SUBMITTED') — that mixes conducted and completed
--   NEVER use status != 'DRAFT' alone for conducted — CLOSED is not conducted"""


def core_context_cast_block() -> str:
    """For agent/seed_examples.py CORE_CONTEXT — the observation↔MA cast rule."""
    j = SEMANTICS["joins"]["observation_to_ma"]
    nevers = "\n".join(f"--   NEVER: {n}" for n in j["never"])
    return f"""-- OBSERVATION → MA JOIN — exact form:
--   {j['sql']}
--   ({j['rule']})
{nevers}"""


def inject_hint_status() -> str:
    """For core/business_rules.py inject_completed_inspections rule hint."""
    return f"""
STATUS GROUPINGS — always show as TWO categories, never raw status values:
  conducted = status IN {conducted_in_list()}
  completed = status {completed_filter()}
  NEVER: status IN ('CLOSED','SUBMITTED') — mixes conducted and completed
  NEVER: status != 'DRAFT' alone for conducted — CLOSED is not conducted
  NEVER expose raw status names (SUBMITTED/UNDER_REVIEW/etc) in results
  UNLESS user explicitly asks to drill into a specific status

DEFAULT PATTERN for total/how many inspections:
SELECT
    {case_category_expr()} AS category,
    COUNT(*) AS inspection_count
FROM inspection_report ir
WHERE ir.active = true AND ir.status != 'DRAFT'
GROUP BY category ORDER BY category DESC;

If user asks ONLY for completed: WHERE ir.status {completed_filter()}
If user asks ONLY for conducted: WHERE ir.status IN {conducted_in_list()}
"""


def constraint_note_status() -> str:
    """For agent/orchestrator.py _CONSTRAINT_NOTE_MAP['MISSING_COMPLETED_STATUS']."""
    return (
        "-- STATUS GROUPINGS (never conflate):\n"
        f"--   completed  = status {completed_filter()} (fully reviewed and approved)\n"
        f"--   conducted  = status IN {conducted_in_list()} (in-flight)\n"
        "--   NEVER: status IN ('CLOSED','SUBMITTED') — mixes conducted and completed\n"
        "--   'total inspections' = two-row CASE summary:\n"
        f"--     {case_category_expr()} AS category\n"
        "--   ALWAYS exclude DRAFT: add status != 'DRAFT' (or a stricter status filter)"
    )


def constraint_note_cast() -> str:
    """For agent/orchestrator.py _CONSTRAINT_NOTE_MAP['WRONG_CAST_DIR']."""
    j = SEMANTICS["joins"]["observation_to_ma"]
    nevers = "\n".join(f"--   NEVER: {n}" for n in j["never"])
    return (
        f"-- CAST RULE: JOIN ON {j['sql']}\n"
        f"--   {j['rule']}\n"
        f"{nevers}"
    )


def core_context_counting_block() -> str:
    """For agent/seed_examples.py CORE_CONTEXT — good practice / deviation / observation counting."""
    c = SEMANTICS["counting"]
    gp = c["good_practice"]
    obs = c["observations_per_inspection"]
    return f"""-- GOOD PRACTICE / DEVIATION COUNTING — always {gp['expr']}, NEVER COUNT(*):
--   {gp['why']}.
--   Same rule applies to deviation counts.
--   WRONG:  SELECT COUNT(*) ... WHERE aq.label='Observation Type' AND aa.answer_text ILIKE '%Good Practice%'
--   CORRECT: SELECT {gp['expr']} ... same filters
-- OBSERVATIONS PER INSPECTION — use {obs['expr']}:
--   {obs['why']}"""


def core_context_columns_block() -> str:
    """For agent/seed_examples.py CORE_CONTEXT — soft-delete / active-column facts."""
    cols = SEMANTICS["columns"]
    has = cols["has_active_column"]
    no = cols["no_active_column"]
    # wrap the has-list at ~4 tables per line for readability
    lines, cur = [], []
    for t in has:
        cur.append(t)
        if len(cur) == 4:
            lines.append(", ".join(cur) + ",")
            cur = []
    if cur:
        lines.append(", ".join(cur))
    has_wrapped = "\n".join(f"--   {line}" for line in lines)
    return f"""-- SOFT-DELETE: these tables have an 'active' boolean column — ALWAYS filter WHERE active = true:
{has_wrapped}
-- NO active column: {", ".join(no)}
--   NEVER use aa.active or aq.active — these columns do not exist"""


def llama_columns_block() -> str:
    """For agent/prompts.py SYSTEM_PROMPT — soft-delete / active-column facts (no Deepseek comment prefixes)."""
    cols = SEMANTICS["columns"]
    has = cols["has_active_column"]
    no = cols["no_active_column"]
    lines, cur = [], []
    for t in has:
        cur.append(t)
        if len(cur) == 4:
            lines.append(", ".join(cur) + ",")
            cur = []
    if cur:
        lines.append(", ".join(cur))
    has_wrapped = "\n".join(f"    {line}" for line in lines)
    return f"""IMPORTANT — 'active' column existence (soft-delete):
  HAS active column — include WHERE <alias>.active = true:
{has_wrapped}
  NO active column — NEVER write <alias>.active for these tables:
    {", ".join(no)}
  In particular: inspection_schedule DOES have an active column (confirmed 2026-07-08,
  21/2209 rows are active=false) — include WHERE isched.active = true unless the
  question is specifically about cancelled/removed schedules."""


def core_context_ma_workflow_block() -> str:
    """For agent/seed_examples.py CORE_CONTEXT — the MA approval workflow."""
    w = SEMANTICS["ma_workflow"]
    ps = w["progress_stage"]["values"]
    ps_list = ", ".join(k for k in ps)
    inv = "\n".join(f"--   {i}" for i in w["invariants"])
    vocab = "\n".join(f"--   '{k}' → {v}" for k, v in w["vocabulary"].items())
    t = w["timeliness"]
    return f"""-- MITIGATIVE ACTION WORKFLOW — four independent tracks on ica:
--   status (system):            OPEN | CLOSED | OVERDUE | CLOSE_WITH_DEFERRED
--   progress_stage (inspectee): {ps_list}
--   implementation_status (reviewers): NULL | PENDING_FOR_REVIEW | APPROVE | RETURN_FOR_MODIFICATION
--     EXACT STRINGS: 'APPROVE' not 'APPROVED' — 'PENDING_FOR_REVIEW' not 'PENDING'
--   capex_status (USER-ENTERED, declarative only): NULL | ACCEPTED | REJECTED | DEFERRED
--     po_details is the authoritative money record — POs legitimately exist on REJECTED rows
-- INVARIANTS:
{inv}
-- TRAP: 'at risk' = progress_stage='AT_RISK' (schedule health), NOT risk_level='High' (severity)
-- VOCABULARY:
{vocab}
--
-- TIMELINESS — overdue vs closed-late are MUTUALLY EXCLUSIVE, never OR them together:
--   'overdue'                = {t['overdue']['expr']}
--                              ({t['overdue']['meaning']})
--   'not closed on time' / 'closed late' = {t['closed_late']['expr']}
--                              ({t['closed_late']['meaning']})
--   close_on is a dead column (always NULL) — use completed_on for closure timing
--   {t['invariant']}"""


def inject_hint_financial() -> str:
    """For core/business_rules.py inject_financial_routing rule hint."""
    f = SEMANTICS["financial"]
    m = f["metrics"]
    po = f["tables"]["po_details"]
    pr = f["tables"]["pr_details"]
    loss = f["capex_loss_form"]
    prf = f["pr_form"]

    loss_labels = "\n".join(
        f"    '{label}'{' ' * max(1, 42 - len(label))}— {desc}"
        for label, desc in loss["labels"].items()
    )
    ownership = "\n".join(f"  {line}" for line in f["ownership"])

    return (
        "\nFINANCIAL DATA ROUTING:\n"
        f"  CAPEX Generated  = {m['capex_generated']['expr']}       — {m['capex_generated']['meaning']}\n"
        f"  CAPEX Converted  = {m['capex_converted']['expr']}    — {m['capex_converted']['meaning']}\n"
        f"  CAPEX Lost       = CAPEX Generated - CAPEX Converted\n"
        f"                     i.e. {m['capex_lost']['expr']}\n"
        f"  DO NOT use capex_status for loss calculation — use the arithmetic difference\n"
        f"\n  CAPEX Loss Detail (from form answers — module: '{loss['module']}'):\n"
        f"{loss_labels}\n"
        f"    JOIN via: {loss['join']}\n"
        f"\n  OPEX Generated   = {m['opex_generated']['expr']}       — {m['opex_generated']['meaning']}\n"
        f"  OPEX Incurred    = {m['opex_incurred']['expr']}    — {m['opex_incurred']['meaning']}\n"
        f"    '{prf['labels'][0]}' / '{prf['labels'][1]}' / '{prf['labels'][2]}' also in form: '{prf['module']}'\n"
        f"    JOIN via: {prf['join']}\n"
        f"\npo_details: {', '.join(po['columns'])}\n"
        f"  {po['multiplicity'].capitalize()} — NEVER direct LEFT JOIN, use CTE pre-aggregation:\n"
        f"  {po['cte_pattern']}\n"
        f"\npr_details: {', '.join(pr['columns'])}\n"
        f"  JOIN: LEFT JOIN pr_details pr ON {pr['join']}\n"
        f"  {pr['multiplicity'].capitalize()}\n"
        f"\n{ownership}\n"
        "\nDATE RULE: NEVER hardcode years (2026, 2025). Use CURRENT_DATE or DATE_TRUNC('year', CURRENT_DATE).\n"
    )


def llama_ma_workflow_block() -> str:
    """For agent/prompts.py SYSTEM_PROMPT — Llama must route approval vocabulary correctly."""
    w = SEMANTICS["ma_workflow"]
    vocab = "\n".join(f"    \"{k}\" → {v}" for k, v in w["vocabulary"].items())
    t = w["timeliness"]
    return f"""IMPORTANT — mitigative action (ica) workflow columns (four independent tracks):
  status (system):       OPEN | CLOSED | OVERDUE | CLOSE_WITH_DEFERRED
  progress_stage:        NOT_STARTED | ON_TRACK | BEHIND | AT_RISK | COMPLETED | CLOSE_WITH_DEFERRED
                         (inspectee's self-reported SCHEDULE HEALTH — not severity)
  implementation_status: NULL | PENDING_FOR_REVIEW | APPROVE | RETURN_FOR_MODIFICATION
                         (reviewer approval track — EXACT strings: 'APPROVE' not 'APPROVED',
                          'PENDING_FOR_REVIEW' not 'PENDING')
  capex_status:          NULL | ACCEPTED | REJECTED | DEFERRED
                         (USER-ENTERED financial opinion — NEVER use for approval-workflow questions)

  VOCABULARY ROUTING (use these exact mappings in schema_hint):
{vocab}

  NEVER route "awaiting approval" / "pending approval" to capex_status — that is the
  financial track. Approval questions = implementation_status.
  'at risk' = progress_stage='AT_RISK' (schedule), NOT risk_level='High' (severity).
  status='CLOSED' ⟺ implementation_status='APPROVE' — closure IS the approval.

  TIMELINESS — 'overdue' and 'closed late' are MUTUALLY EXCLUSIVE, NEVER OR them:
    'overdue' (still open, past due)     → {t['overdue']['expr']}
    'closed late' / 'not closed on time' → {t['closed_late']['expr']}
    close_on is a dead column (always NULL) — use completed_on for closure timing."""


# ═══════════════════════════════════════════════════════════════════════════
# Consistency checker — catches stale hardcoded copies in consumer files
# ═══════════════════════════════════════════════════════════════════════════

# Patterns that indicate a STALE copy of an old definition. If any of these
# appear OUTSIDE a NEVER/forbidden context in a consumer file, the layers
# have desynced. Run in CI / at startup with strict=False.
_STALE_PATTERNS: list[tuple[str, str]] = [
    # (regex, description)
    (r"completed.{0,30}IN \('CLOSED',\s*'SUBMITTED'\)",
     "old definition: completed = CLOSED+SUBMITTED"),
    (r"IN \('CLOSED',\s*'SUBMITTED'\).{0,30}completed",
     "old definition: completed = CLOSED+SUBMITTED (reversed)"),
    (r"conducted\s*=\s*status\s*=\s*'SUBMITTED'",
     "old definition: conducted = SUBMITTED only"),
    (r"ica\.observation_id\s*=\s*aa\.submission_id::uuid",
     "old cast direction: varchar cast to uuid"),
    (r"ica\.mitigative_action_uuid\s*=",
     "wrong join: mitigative_action_uuid is on po/pr_details, not ica"),
    (r"COUNT\(\*\)[^\n]{0,80}Good Practice",
     "good-practice counting with COUNT(*) — must be COUNT(DISTINCT ir.id)"),
    (r"capex_lost\s*=.{0,40}REJECTED",
     "capex_lost defined via capex_status — must be arithmetic difference"),
    (r"forbidden_patterns.{0,80}status\s*=\s*'CLOSED'",
     "glossary forbids status='CLOSED' for completed — that is now the CORRECT pattern"),
    (r"mitigative_action_uuid\s*=\s*ir\.",
     "wrong join target: mitigative_action_uuid → ica.id, never ir.id (silent all-zero results)"),
    (r"\bclose_on\s*[<>]=?\s*\w",
     "close_on is a dead column (always NULL) — use completed_on for closure timing"),
    (r"implementation_status\s*=\s*'APPROVED'",
     "wrong enum string: it is 'APPROVE' not 'APPROVED'"),
    (r"implementation_status\s*=\s*'PENDING'(?!_)",
     "wrong enum string: it is 'PENDING_FOR_REVIEW' not 'PENDING'"),
    (r"status\s*NOT\s*IN\s*\(\s*'CLOSED'\s*,\s*'CLOSE_WITH_DEFERRED'\s*\)\s*AND\s*"
     r"[\w.]*target_close_out_date\s*<\s*(?:CURRENT_DATE|NOW\(\))",
     "old definition: overdue reconstructed via date arithmetic — must be status = 'OVERDUE' "
     "(found live in agent/seed_examples.py, agent/prompts.py, core/business_rules.py, "
     "and core/semantics_registry.py itself on 2026-07-07; cost every 'overdue' query an "
     "extra retry before the model self-corrected)"),
]


def verify_consistency(file_paths: list[str], strict: bool = False) -> list[str]:
    """
    Scan consumer files for stale definitions.
    Returns list of human-readable problems (empty = all in sync).
    Lines containing NEVER / WRONG / forbidden are skipped (they intentionally
    show the wrong form as a warning).
    """
    import re as _re
    problems: list[str] = []
    for path in file_paths:
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            problems.append(f"{path}: unreadable ({exc})")
            continue
        for lineno, line in enumerate(lines, 1):
            if _re.search(r"NEVER|WRONG|forbidden|never_counted|DO NOT", line, _re.IGNORECASE):
                continue  # intentional negative example
            for pat, desc in _STALE_PATTERNS:
                if _re.search(pat, line, _re.IGNORECASE):
                    problems.append(f"{path}:{lineno} — {desc}\n    {line.strip()}")
    if problems and strict:
        raise RuntimeError(
            "Semantics desync detected:\n" + "\n".join(problems)
        )
    return problems


# ═══════════════════════════════════════════════════════════════════════════
# CLI — print all generated blocks for review
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # python -m khidmah.core.semantics_registry check file1.py file2.py ...
        _khidmah_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        targets = sys.argv[2:] or [
            os.path.join(_khidmah_dir, "agent", "prompts.py"),
            os.path.join(_khidmah_dir, "agent", "seed_examples.py"),
            os.path.join(_khidmah_dir, "core", "business_rules.py"),
            os.path.join(_khidmah_dir, "agent", "orchestrator.py"),
        ]
        probs = verify_consistency(targets)
        if probs:
            print(f"✗ {len(probs)} desync(s) found:\n")
            for p in probs:
                print(f"  {p}\n")
            sys.exit(1)
        print(f"✓ all {len(targets)} consumer files in sync with registry")
        sys.exit(0)

    sections = [
        ("LLAMA STATUS BLOCK (prompts.py)",           llama_status_block()),
        ("CORE_CONTEXT STATUS (seed_examples.py)",     core_context_status_block()),
        ("CORE_CONTEXT CAST (seed_examples.py)",       core_context_cast_block()),
        ("CORE_CONTEXT COUNTING (seed_examples.py)",   core_context_counting_block()),
        ("CORE_CONTEXT COLUMNS (seed_examples.py)",    core_context_columns_block()),
        ("LLAMA COLUMNS (prompts.py)",                  llama_columns_block()),
        ("CORE_CONTEXT MA WORKFLOW (seed_examples.py)", core_context_ma_workflow_block()),
        ("INJECT HINT STATUS (business_rules.py)",     inject_hint_status()),
        ("INJECT HINT FINANCIAL (business_rules.py)",  inject_hint_financial()),
        ("CONSTRAINT NOTE STATUS (orchestrator.py)",   constraint_note_status()),
        ("CONSTRAINT NOTE CAST (orchestrator.py)",     constraint_note_cast()),
    ]
    for title, block in sections:
        print(f"\n{'═'*70}\n  {title}\n{'═'*70}\n{block}")