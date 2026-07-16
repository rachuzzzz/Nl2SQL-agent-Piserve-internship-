"""
Business Rules Registry
========================
All domain-specific rules for the inspection NL2SQL system live here as
structured data.  No business knowledge should be hardcoded in validator.py
or orchestrator.py — those files consume this registry.

Rule types
----------
  BLOCK        Hard validation error.  SQL is rejected and the error message
               is returned to the LLM for repair.
  WARN         Soft warning.  SQL executes but the warning is surfaced.
  AUTOCORRECT  The rule's `replacement` is applied automatically in clean_sql()
               before validation runs.  Logged but not surfaced to the user.
  INJECT       Question-level hint.  When `question_pattern` matches the user
               query, `hint` is appended to schema_hint before SQL generation.

Adding a new rule
-----------------
  1. Give it a unique `id`.
  2. Pick a type.
  3. Fill in the required fields for that type (see schema below).
  4. Run:  python -c "from business_rules import RulesEngine; RulesEngine()"
     A conflict check runs at import time — fix any warnings it reports.

Conflict detection
------------------
  At RulesEngine.__init__ the engine checks:
  - Duplicate rule IDs
  - BLOCK + AUTOCORRECT rules sharing the same sql_pattern (ambiguous precedence)
  - INJECT rules with identical question_pattern strings (redundant)

  Conflicts raise ValueError at startup so they are caught before the first query.
"""

from __future__ import annotations

import re


# ── Semantics registry (single source of truth for status/cast rules) ─────────
try:
    from khidmah.core.semantics_registry import (
        inject_hint_status as _inject_hint_status,
        inject_hint_financial as _inject_hint_financial,
        draft_exclusion_filter as _draft_exclusion_filter,
    )
    _STATUS_HINT = _inject_hint_status()
    _FINANCIAL_HINT = _inject_hint_financial()
    _DRAFT_EXCLUSION = _draft_exclusion_filter()
except ImportError:
    # Fallback — keep in sync with core/semantics_registry.py
    _DRAFT_EXCLUSION = "status != 'DRAFT'"
    _STATUS_HINT = (
        "\nSTATUS GROUPINGS — always show as TWO categories, never raw status values:\n"
        "  conducted = status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')\n"
        "  completed = status = 'CLOSED'\n"
        "  NEVER: status IN ('CLOSED','SUBMITTED') — mixes conducted and completed\n"
        "  NEVER: status != 'DRAFT' alone for conducted — CLOSED is not conducted\n"
        "  NEVER expose raw status names (SUBMITTED/UNDER_REVIEW/etc) in results\n"
        "  UNLESS user explicitly asks to drill into a specific status\n\n"
        "DEFAULT PATTERN for total/how many inspections:\n"
        "SELECT\n"
        "    CASE WHEN ir.status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category,\n"
        "    COUNT(*) AS inspection_count\n"
        "FROM inspection_report ir\n"
        "WHERE ir.active = true AND ir.status != 'DRAFT'\n"
        "GROUP BY category ORDER BY category DESC;\n\n"
        "If user asks ONLY for completed: WHERE ir.status = 'CLOSED'\n"
        "If user asks ONLY for conducted: WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')\n"
    )
    _FINANCIAL_HINT = (
        "\nFINANCIAL DATA ROUTING:\n"
        "  CAPEX Generated  = SUM(ica.capex)       — inspectee's original estimate on ICA\n"
        "  CAPEX Converted  = SUM(po.po_amount)    — actual PO raised (via po_details)\n"
        "  CAPEX Lost       = SUM(ica.capex) - COALESCE(SUM(po.po_amount), 0)\n"
        "  DO NOT use capex_status for loss calculation — use the arithmetic difference\n"
        "  OPEX Incurred    = SUM(pr.pr_amount)    — actual PR spend (via pr_details)\n"
        "po_details JOIN: po.mitigative_action_uuid = ica.id — one ICA can have MULTIPLE POs,\n"
        "  NEVER direct LEFT JOIN with SUM(ica.capex); use CTE pre-aggregation.\n"
        "pr_details JOIN: pr.mitigative_action_uuid = ica.id — always SUM(pr.pr_amount).\n"
        "CAPEX is tied to the FM Manager (inspectee_user_id); inspectors have no role in CAPEX/OPEX.\n"
        "DATE RULE: NEVER hardcode years. Use CURRENT_DATE or DATE_TRUNC('year', CURRENT_DATE).\n"
    )


# ── Rule type constants ────────────────────────────────────────────────────────

BLOCK       = "BLOCK"
WARN        = "WARN"
AUTOCORRECT = "AUTOCORRECT"
INJECT      = "INJECT"


# ── Registry ───────────────────────────────────────────────────────────────────

REGISTRY: list[dict] = [

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK — hard errors that cause SQL rejection
    # ══════════════════════════════════════════════════════════════════════════

    # ── po/pr_details wrong join column ─────
    {
        "id":          "block_po_pr_wrong_join_column",
        "type":        BLOCK,
        "description": "po/pr_details.mitigative_action_uuid must join to ica.id (UUID PK), NEVER to ica.corrective_action_id (varchar).",
        "sql_pattern": r"(?:po|pr)_details\b.*?\.mitigative_action_uuid\s*=\s*ica\.corrective_action_id",
        "message":     (
            "BLOCKED: po/pr_details.mitigative_action_uuid must join to ica.id (UUID primary key), "
            "NOT ica.corrective_action_id (varchar — causes type error). "
            "Correct: ON po.mitigative_action_uuid = ica.id. "
            "Pre-aggregate in CTEs to avoid row fan-out."
        ),
        "hard_fail":   True,
    },
    # ── po/pr_details joined to ir.id instead of ica.id ─────
    {
        "id":          "block_po_pr_join_to_ir",
        "type":        BLOCK,
        "code":        "PO_PR_JOIN_TO_IR",
        "description": "po/pr_details.mitigative_action_uuid must join to ica.id, NEVER to ir.id (inspection_report). Joining to ir.id matches zero rows.",
        "sql_pattern": r"mitigative_action_uuid\s*=\s*ir\.id",
        "message":     (
            "BLOCKED: po/pr_details.mitigative_action_uuid must join to ica.id (inspection_corrective_action UUID), "
            "NOT ir.id (inspection_report) — that matches ZERO rows. "
            "The ICA hop is mandatory: po_details → ica → inspection_report → facility. "
            "Use CTEs: WITH po_agg AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total FROM po_details WHERE active=true GROUP BY 1) "
            "then LEFT JOIN po_agg po ON po.ica_id = ica.id"
        ),
        "hard_fail":   True,
    },
    # ── po_details + pr_details both directly joined causes row fan-out ─────
    {
        "id":          "block_po_pr_double_direct_join",
        "type":        BLOCK,
        "code":        "PO_PR_DOUBLE_JOIN",
        "description": (
            "Joining po_details AND pr_details directly into the same query (not via separate "
            "pre-aggregated CTEs) causes a cross-join row multiplication when an MA has more than "
            "one PO or PR row, silently corrupting SUM() results (confirmed in production "
            "2026-06-15: an MA with 2 POs and 1 PR returned capex_generated=1000 instead of 2000, "
            "and capex_converted=0 instead of 1000)."
        ),
        "sql_pattern": (
            r"(?is)LEFT\s+JOIN\s+po_details\b.*?LEFT\s+JOIN\s+pr_details\b.*?GROUP\s+BY"
            r"|LEFT\s+JOIN\s+pr_details\b.*?LEFT\s+JOIN\s+po_details\b.*?GROUP\s+BY"
        ),
        "message": (
            "BLOCKED: po_details and pr_details are both directly LEFT JOINed into a GROUP BY query. "
            "If a mitigative action has more than one PO or PR row, this cross-joins them and corrupts "
            "every SUM() in the result (confirmed bug: 2 POs + 1 PR returned half the correct capex). "
            "ALWAYS pre-aggregate each table in its own CTE first:\n"
            "WITH po_agg AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po "
            "FROM po_details WHERE active=true GROUP BY 1),\n"
            "pr_agg AS (SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr "
            "FROM pr_details WHERE active=true GROUP BY 1)\n"
            "then LEFT JOIN po_agg ON po_agg.ica_id = ica.id and LEFT JOIN pr_agg ON pr_agg.ica_id = ica.id "
            "in the outer query — each CTE returns at most one row per ica.id, so no fan-out is possible."
        ),
        "hard_fail":   True,
    },
    # ── Organisation join always returns empty (FK unpopulated) ─────────────
    {
        "id":          "block_organisation_join",
        "type":        BLOCK,
        "description": "facility.organization_id is unpopulated (all NULL) — joining to organisation always returns zero rows.",
        "sql_pattern": r"\bJOIN\s+organisation\b",
        "message":     "UNSUPPORTED: facility.organization_id is not populated in the current dataset. All facilities belong to a single organisation (Khidmah). Grouping by organisation is not meaningful — use facility, region, country, project, or entity instead.",
    },

    # ── DML / DDL ────────────────────────────────────────────────────────────
    {
        "id":          "block_dml",
        "type":        BLOCK,
        "code":        "DML_BLOCKED",
        "description": "Only SELECT is allowed — no data modification.",
        "sql_pattern": r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE)\b",
        "message":     "BLOCKED: DML/DDL detected. Only SELECT is allowed.",
    },

    # ── frequency_definition has NO description column ───────────────────────
    {
        "id":          "block_frequency_definition_description",
        "type":        BLOCK,
        "description": "frequency_definition has no 'description' column — hallucinated.",
        # frequency_definition columns: id, name, repeat_count, repeat_interval, repeat_unit, active
        "sql_pattern": r"\b(fd|frequency_definition)\.description\b",
        "message": (
            "WRONG COLUMN: frequency_definition has no 'description' column. "
            "Its columns are: id, name, repeat_count, repeat_interval, repeat_unit, active. "
            "Use fd.name (e.g. 'Quarterly Twice') — repeat_count/repeat_interval/repeat_unit "
            "describe the cadence (e.g. repeat_count=2, repeat_interval=3, repeat_unit='MONTH')."
        ),
    },

    # ── inspection_cycle has NO name column ───────────────────────────────────
    {
        "id":          "block_inspection_cycle_name",
        "type":        BLOCK,
        "description": "inspection_cycle has no 'name' column — hallucinated.",
        # Confirmed 2026-06-12 (Llama self-correction after real DB error):
        # inspection_cycle columns: id, status, start_date, end_date, due_date,
        # send_notification_to_inspectors, created_on, modified_on — NO name.
        # "cycles" with human-readable names = frequency_definition.name instead
        # (see block_frequency_definition_description / "list all cycles" seed).
        "sql_pattern": r"\b(ic|inspection_cycle)\.name\b",
        "message": (
            "WRONG COLUMN: inspection_cycle has no 'name' column. Its columns are: "
            "id, status, start_date, end_date, due_date, send_notification_to_inspectors, "
            "created_on, modified_on. For a display label, use ic.start_date / ic.end_date / "
            "ic.status (e.g. 'Jan 2026 cycle'), or join to frequency_definition for a "
            "human-readable cadence name (fd.name via ip/ipd, see cycle-frequency seeds)."
        ),
    },

    # ── inspection_schedule has NO frequency_definition_id column ────────────
    {
        "id":          "block_isched_frequency_definition_id",
        "type":        BLOCK,
        "description": "inspection_schedule has no frequency_definition_id — hallucinated column causing infinite retry loop.",
        # Confirmed 2026-06-15: Deepseek repeatedly hallucinated
        # LEFT JOIN frequency_definition fd ON isched.frequency_definition_id = fd.id
        # across 5+ iterations because the retrieved seeds prominently feature
        # frequency_definition joins in other contexts. The correct path to
        # frequency is: isched.portfolio_details_id → ipd.frequency_definition_id → fd
        # inspection_schedule columns: id, active, schedule_id, status, due_date,
        # schedule_date, facility_id, inspector_id, inspectee_id, project_id,
        # portfolio_details_id, created_on, modified_on
        # NO: frequency_definition_id, inspection_report_id (reach the report via
        #     inspection_report.schedule_id = isched.id — the FK lives on the report)
        "sql_pattern": r"\bisched\.frequency_definition_id\b|\binspection_schedule\b[^;]*\bfrequency_definition_id\b",
        "message": (
            "WRONG COLUMN: inspection_schedule has no 'frequency_definition_id' column. "
            "inspection_schedule columns: id, active, schedule_id, status, due_date, schedule_date, "
            "facility_id, inspector_id, inspectee_id, project_id, portfolio_details_id — "
            "NO frequency_definition_id, NO inspection_report_id. "
            "To get frequency: JOIN inspector_portfolio_details ipd ON isched.portfolio_details_id = ipd.id "
            "JOIN frequency_definition fd ON ipd.frequency_definition_id = fd.id"
        ),
    },

    # ── AUTOCORRECT: strip the hallucinated frequency_definition join ─────────
    {
        "id":          "autocorrect_isched_frequency_definition_join",
        "type":        AUTOCORRECT,
        "description": "Remove hallucinated LEFT JOIN frequency_definition fd ON isched.frequency_definition_id = fd.id.",
        "sql_pattern": r"LEFT\s+JOIN\s+frequency_definition\s+\w+\s+ON\s+\w+\.frequency_definition_id\s*=\s*\w+\.id\s*",
        "replacement": "",
        "message": "[autocorrect] Removed hallucinated JOIN frequency_definition via non-existent isched.frequency_definition_id",
    },

    # ── BLOCK: fac.name ILIKE inside CTE WHERE ───────────
    {
        "id":          "block_fac_filter_in_cte",
        "type":        BLOCK,
        "description": "fac.name ILIKE inside a CTE body causes 'column fac.name does not exist' error. Facility filter must be in the outer query WHERE clause.",
        "sql_pattern": r"(?i)WITH\s+\w+\s+AS\s*\([^)]*AND\s+fac\.name\s+ILIKE",
        "message": (
            "BLOCKED: fac.name ILIKE inside a CTE WHERE clause causes 'column fac.name does not exist' "
            "because fac (facility) is NOT joined inside the CTE. "
            "The facility filter MUST go in the outer query WHERE clause, not inside a CTE. "
            "Pattern: WITH po_agg AS (SELECT ir.facility_id, SUM(...) FROM po_details JOIN ica ON ... JOIN ir ON ... "
            "WHERE ir.active=true GROUP BY ir.facility_id) "
            "SELECT ... FROM facility fac LEFT JOIN po_agg ON po_agg.facility_id=fac.id "
            "WHERE fac.name ILIKE '%[NAME]%' AND fac.active=true"
        ),
        "hard_fail": True,
    },
    # ── AUTOCORRECT: remove fac.name ILIKE from inside CTE bodies ─────
    {
        "id":          "autocorrect_fac_filter_in_cte",
        "type":        AUTOCORRECT,
        "description": "Remove fac.name ILIKE filter placed inside a CTE WHERE clause where fac is not in scope. Facility filter must be in the outer query.",
        "sql_pattern": r"\s+AND\s+fac\.name\s+ILIKE\s+'%[^']*%'(?=\s*(?:GROUP BY|ORDER BY|LIMIT|\)))",
        "replacement": "",
        "message": "[autocorrect] Removed fac.name ILIKE from CTE body where fac is not in scope; facility filter must be in outer WHERE clause",
    },
    # ── cycle-frequency join chain missing its defining subquery ─────────────
    {
        "id":          "block_cycle_frequency_missing_subquery",
        "type":        BLOCK,
        "description": "ic→ip→ipd→fd frequency join with status='COMPLETED' requires the most-recent-cycle subquery.",
        # This exact join chain (ic -> ip -> ipd -> fd) combined with
        # ic.status = 'COMPLETED' is ONLY valid as part of the "most recently
        # completed cycle" pattern, which REQUIRES:
        #   WHERE ic.id = (SELECT id FROM inspection_cycle WHERE status='COMPLETED'
        #                   ORDER BY end_date DESC LIMIT 1)
        # Without that subquery, "ic.status = 'COMPLETED'" matches MANY cycles,
        # turning "frequency of the MOST RECENT cycle" into "frequencies used
        # by ALL completed cycles" — a silently different, usually-DISTINCT-ed
        # answer. Observed twice in a row (2026-06-11) on "list all cycles".
        "sql_pattern": (
            r"(?=.*\binspection_cycle\s+\w+)"
            r"(?=.*\binspector_portfolio\b)"
            r"(?=.*\bfrequency_definition\b)"
            r"(?=.*status\s*=\s*'COMPLETED')"
            # Exempt: subquery form (most-recent-cycle pattern) ...
            r"(?!.*\(\s*select\s+id\s+from\s+inspection_cycle)"
            # ... or direct UUID/parameter filter (specific-cycle pattern):
            # ic.id = '<uuid-or-placeholder>' — any quoted literal, not a subquery
            r"(?!.*\bic\.id\s*=\s*'[^']*')"
        ),
        "message": (
            "WRONG SHAPE: the ic→ip→ipd→fd join with status='COMPLETED' is ONLY "
            "for 'the MOST RECENTLY COMPLETED cycle's frequency' — and REQUIRES "
            "the defining subquery:\n"
            "  WHERE ic.id = (SELECT id FROM inspection_cycle WHERE status = 'COMPLETED' "
            "ORDER BY end_date DESC LIMIT 1)\n"
            "Without it, this returns frequencies for ALL completed cycles — a different question.\n"
            "If the user asked 'list all cycles' / 'what cycles are available' / "
            "'cycle types', do NOT join inspection_cycle at all — use:\n"
            "  SELECT name AS frequency_name, repeat_count, repeat_interval, repeat_unit "
            "FROM frequency_definition WHERE active = true ORDER BY name;"
        ),
    },

    # ── mitigative_action_uuid must join to ica.id — nothing else ────────────
    {
        "id":          "block_mitigative_uuid_wrong_target",
        "type":        BLOCK,
        "code":        "MITIGATIVE_UUID_WRONG_TARGET",
        "description": "po/pr_details.mitigative_action_uuid joins to ica.id only.",
        # Catches the WRONG form: mitigative_action_uuid = ir.id / ir.<col> / fac.id etc.
        # Silent-zero failure mode: the join matches nothing, LEFT JOIN +
        # COALESCE(...,0) then paints every row with zeros (observed 2026-06-11).
        "sql_pattern": (
            r"mitigative_action_uuid\s*=\s*(?!ica\.id\b)(ir|fac|aa|aq|isched|u|cl|proj|reg|ent)\."
            r"|\b(ir|fac|aa|aq|isched|u|cl|proj|reg|ent)\.\w+\s*=\s*\w*\.?mitigative_action_uuid"
        ),
        "message": (
            "WRONG JOIN TARGET: mitigative_action_uuid points to inspection_corrective_action.id "
            "(the MA), NEVER inspection_report.id or any other table. "
            "Correct path: po_details/pr_details → ica → inspection_report → facility:\n"
            "  JOIN inspection_corrective_action ica ON pd.mitigative_action_uuid = ica.id\n"
            "  JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "Joining it to ir.id matches ZERO rows and silently produces all-zero results."
        ),
    },

    # ── DRAFT records must never be counted ──────────────────────────────────
    {
        "id":          "block_draft_not_excluded",
        "type":        BLOCK,
        "description": "inspection_report queries must always exclude DRAFT records.",
        # Fires when inspection_report is queried but no status filter operator is present.
        # status != 'DRAFT', status = 'CLOSED', status IN (...) all satisfy this.
        # SCOPE LIMIT (eval regression FLT-03): queries joining ai_answers are exempt —
        # observation/form-answer queries are answer-driven and repeatedly hard-blocking
        # them caused 8-retry death spirals. ai_answers rows only exist for submitted
        # inspections, so DRAFT leakage risk there is minimal.
        "sql_pattern": (
            r"(?=.*\bFROM\s+inspection_report\b)"              # inspection_report is the PRIMARY table
            r"(?!.*\bai_answers\b)"                            # exempt form-answer queries
            r"(?!.*\bstatus\s*(?:!=|=|IN\s*\(|NOT\s+IN))"     # no status comparison operator anywhere
        ),
        "message": (
            "MISSING FILTER: inspection_report queries must always exclude DRAFT records. "
            "DRAFT inspections are incomplete and must NEVER be counted or listed. "
            "Add: AND ir.status != 'DRAFT'  (or a stricter filter like = 'CLOSED' or = 'SUBMITTED'). "
            "A stricter status filter already excludes DRAFT implicitly — "
            "only add != 'DRAFT' when listing all non-draft statuses."
        ),
    },

    # ── Hallucinated tables ───────────────────────────────────────────────────
    {
        "id":          "block_table_ai_question_singular",
        "type":        BLOCK,
        "code":        "UNKNOWN_TABLE",
        "description": "ai_question (singular) does not exist — use ai_questions.",
        "sql_pattern": r"\bai_question\b(?!s)",
        "message":     "HALLUCINATION: table 'ai_question' does not exist. Use ai_questions (plural).",
    },
    {
        "id":          "block_table_ai_answer_singular",
        "type":        BLOCK,
        "code":        "UNKNOWN_TABLE",
        "description": "ai_answer (singular) does not exist — use ai_answers.",
        "sql_pattern": r"\bai_answer\b(?!s)",
        "message":     "HALLUCINATION: table 'ai_answer' does not exist. Use ai_answers (plural).",
    },
    {
        "id":          "block_table_fb_users",
        "type":        BLOCK,
        "code":        "UNKNOWN_TABLE",
        "description": "fb_users does not exist — use users.",
        "sql_pattern": r"\bfb_users?\b",
        "message":     "HALLUCINATION: table 'fb_users' does not exist. Use the 'users' table.",
    },
    {
        "id":          "block_table_fb_submissions",
        "type":        BLOCK,
        "code":        "UNKNOWN_TABLE",
        "description": "fb_submissions / fb_submission do not exist.",
        "sql_pattern": r"\bfb_submissions?\b",
        "message":     "HALLUCINATION: fb_submissions does not exist. Answer data is in ai_answers.",
    },
    {
        "id":          "block_table_fb_answers",
        "type":        BLOCK,
        "code":        "UNKNOWN_TABLE",
        "description": "fb_answers / fb_answer / fb_form_answers do not exist.",
        "sql_pattern": r"\bfb_(form_)?answers?\b",
        "message":     "HALLUCINATION: fb_answers does not exist. Use ai_answers.",
    },
    {
        "id":          "block_table_fb_responses",
        "type":        BLOCK,
        "code":        "UNKNOWN_TABLE",
        "description": "fb_responses / fb_response do not exist.",
        "sql_pattern": r"\bfb_responses?\b",
        "message":     "HALLUCINATION: fb_responses does not exist. Use ai_answers.",
    },
    {
        "id":          "block_table_dynamic_fb",
        "type":        BLOCK,
        "code":        "DYNAMIC_TABLE",
        "description": "Dynamic fb_{uuid} answer tables — all answer data is in ai_answers.",
        "sql_pattern": r"\bfb_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}\b",
        "message":     "BLOCKED: Dynamic fb_<uuid> table. Use ai_answers instead.",
    },
    {
        "id":          "block_ica_ir_varchar_uuid_join",
        "type":        BLOCK,
        "description": "ir.inspection_id = ica.inspection_id is a varchar=UUID type mismatch.",
        "sql_pattern": r"\bir\.inspection_id\s*=\s*ica\.inspection_id\b|\bica\.inspection_id\s*=\s*ir\.inspection_id\b",
        "message":     (
            "WRONG JOIN: ir.inspection_id (VARCHAR) ≠ ica.inspection_id (UUID). "
            "CORRECT: JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id "
            "(ica.inspection_id is a UUID FK → inspection_report.id UUID PK)"
        ),
    },
    {
        "id":          "block_ma_for_observation_wrong_path",
        "type":        BLOCK,
        "description": (
            "When fetching MAs for a specific observation text (ILIKE), must use "
            "observation_id → submission_id path, not inspection join."
        ),
        # Fires when: ai_answers ILIKE filter present + ICA joined via inspection_report
        # but observation_id → submission_id join is absent
        "sql_pattern": (
            r"(?=.*\bai_answers\b)"                          # ai_answers present
            r"(?=.*\bilike\b)"                               # ILIKE filter present
            r"(?=.*\binspection_corrective_action\b)"        # ICA present
            r"(?=.*\bica\.inspection_id\s*=\s*ir\.id\b)"     # wrong path: via inspection
            r"(?!.*\bica\.observation_id\b)"                 # correct path absent
        ),
        "message":     (
            "WRONG PATH: When filtering MAs by observation text (ILIKE), join via "
            "aa.submission_id = ica.observation_id::varchar — NOT via ica.inspection_id = ir.id. "
            "The inspection join returns ALL MAs in the inspection; the observation_id join "
            "returns only MAs directly linked to that specific observation."
        ),
    },

    # ── CAPEX Lost definition — block capex_status=REJECTED as loss proxy ──────
    {
        "id":          "block_capex_status_rejected_as_loss",
        "type":        BLOCK,
        "description": "capex_status REJECTED/DEFERRED is NOT how CAPEX Lost is calculated.",
        # Narrowed (2026-06-12): only fire when capex_status REJECTED/DEFERRED is
        # used to PRODUCE a capex_lost/capex_converted output column (a SUM/CASE
        # computing loss from capex_status), NOT when it's a plain WHERE filter
        # for listing/displaying rows (e.g. "capex lost from rejected mitigative
        # actions" — legitimately filters WHERE ica.capex_status = 'REJECTED'
        # to list those rows; capex_status is user-entered/declarative and
        # POs can legitimately exist on REJECTED rows).
        "sql_pattern": (
            r"(?:capex_lost|capex_converted)\s*=.*capex_status.*(?:'REJECTED'|'DEFERRED')"
            r"|capex_status.*(?:'REJECTED'|'DEFERRED').*(?:AS\s+capex_lost|AS\s+capex_converted)"
        ),
        "message":     (
            "WRONG: capex_status='REJECTED' or 'DEFERRED' does not mean CAPEX Lost. "
            "CAPEX Lost = SUM(ica.capex) - COALESCE(SUM(po.po_amount), 0) "
            "via LEFT JOIN po_details po ON po.mitigative_action_uuid = ica.id. "
            "For loss detail: join ai_answers via ica.completion_submission_id and "
            "filter HAVING MAX(CASE WHEN label='Is CAPEX Lost' THEN answer_text END) = 'true'."
        ),
    },
    {
        "id":          "block_po_wrong_join_col",
        "type":        BLOCK,
        "description": "po_details has no corrective_action_id column — join via mitigative_action_uuid",
        "sql_pattern": r"po\.corrective_action_id",
        "message":     "WRONG JOIN: po_details has no corrective_action_id. Use: po.mitigative_action_uuid = ica.id",
    },
    {
        "id":          "block_capex_left_join_multiplication",
        "type":        BLOCK,
        "code":        "CAPEX_LEFT_JOIN_MULT",
        "description": "Direct LEFT JOIN po_details inflates SUM(ica.capex) when one ICA has multiple POs",
        # Must match: SUM(ica.capex) AND LEFT JOIN po_details in the main query
        # but NOT when po_details only appears inside a CTE (WITH ... AS (...po_details...))
        # Detection: LEFT JOIN po_details must appear AFTER the main SELECT (outside CTE)
        # We detect this by requiring LEFT JOIN po_details to NOT be preceded by a closing paren
        # that belongs to a CTE definition.
        # Simpler reliable check: LEFT JOIN po_details appears AND no po_by_ica CTE alias follows it
        "sql_pattern": (
            r"(?=.*\bSUM\s*\(\s*ica\.capex)"        # SUM(ica.capex) present
            r"(?=.*\bLEFT\s+JOIN\s+po_details\b)"   # direct LEFT JOIN po_details
            r"(?!.*\bpo_by_ica\b)"                  # CTE alias po_by_ica NOT present
        ),
        "message":     (
            "WRONG: Direct LEFT JOIN po_details inflates SUM(ica.capex) — one ICA can have multiple POs. "
            "Use scalar subqueries instead: "
            "SELECT (SELECT SUM(capex) FROM inspection_corrective_action) AS capex_generated, "
            "(SELECT SUM(po_amount) FROM po_details) AS capex_converted. "
            "OR use CTE: WITH po_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total FROM po_details GROUP BY 1) "
            "LEFT JOIN po_by_ica p ON p.ica_id = ica.id"
        ),
    },
    {
        "id":          "block_expenditure_as_conversion",
        "type":        BLOCK,
        "code":        "EXPENDITURE_AS_CONVERSION",
        "description": "expenditure column is a category flag, NOT a conversion indicator",
        "sql_pattern": r"expenditure.*IN.*CAPEX.*converted|converted.*expenditure.*IN.*CAPEX|CASE.*expenditure.*CAPEX.*capex",
        "message":     (
            "WRONG: ica.expenditure is a category enum (CAPEX/OPEX/NO_EXPENDITURE), NOT a conversion indicator. "
            "CAPEX Converted = SUM(po.po_amount) via LEFT JOIN po_details po ON po.mitigative_action_uuid = ica.id. "
            "NEVER use expenditure to calculate converted amounts."
        ),
    },

    # ── Hallucinated columns on ai_questions ──────────────────────────────────
    {
        "id":          "block_col_aq_question_label",
        "type":        BLOCK,
        "description": "ai_questions.question_label does not exist — the column is called label.",
        "sql_pattern": r"\b\w*\.question_label\b",
        "table_guard": "ai_questions",
        "message":     "WRONG COLUMN: question_label does not exist on ai_questions. Use aq.label.",
    },
    {
        "id":          "block_col_aq_form_name",
        "type":        BLOCK,
        "description": "ai_questions.form_name does not exist — use module_name.",
        "sql_pattern": r"\b(aq|ai_questions)\b.*?\bform_name\b",
        "message":     "WRONG COLUMN: form_name does not exist on ai_questions. Use module_name.",
    },
    {
        "id":          "block_col_aq_question_type",
        "type":        BLOCK,
        "description": "ai_questions.question_type does not exist — use entity_type.",
        "sql_pattern": r"\b(aq|ai_questions)\b.*?\bquestion_type\b",
        "message":     "WRONG COLUMN: question_type does not exist on ai_questions. Use entity_type.",
    },
    {
        "id":          "block_col_aa_answer_value",
        "type":        BLOCK,
        "description": "ai_answers.answer_value does not exist — use answer_text or answer_numeric.",
        "sql_pattern": r"\b(aa|ai_answers)\b.*?\banswer_value\b",
        "message":     "WRONG COLUMN: answer_value does not exist on ai_answers. Use answer_text (text) or answer_numeric (numeric).",
    },

    # ── Hallucinated columns on inspection_report ─────────────────────────────
    {
        "id":          "block_col_ir_facility_name",
        "type":        BLOCK,
        "description": "inspection_report has no facility_name column — must JOIN facility table.",
        "sql_pattern": r"\bir\.facility_name\b",
        "message":     "WRONG COLUMN: ir.facility_name does not exist. JOIN facility fac ON ir.facility_id = fac.id → SELECT fac.name AS facility_name.",
    },
    {
        "id":          "block_col_ir_inspector_name",
        "type":        BLOCK,
        "description": "inspection_report has no inspector_name column — must JOIN users table.",
        "sql_pattern": r"\bir\.inspector_name\b",
        "message":     "WRONG COLUMN: ir.inspector_name does not exist. JOIN users u ON ir.inspector_user_id = u.id → u.first_name || ' ' || u.last_name AS inspector_name.",
    },
    {
        "id":          "block_col_ir_client_name",
        "type":        BLOCK,
        "description": "inspection_report has no client_name column — must JOIN client table.",
        "sql_pattern": r"\bir\.client_name\b",
        "message":     "WRONG COLUMN: ir.client_name does not exist. JOIN client cl ON ir.client_id = cl.id → cl.name AS client_name.",
    },
    {
        "id":          "block_col_ir_project_name",
        "type":        BLOCK,
        "description": "inspection_report has no project_name column — must JOIN project table.",
        "sql_pattern": r"\bir\.project_name\b",
        "message":     "WRONG COLUMN: ir.project_name does not exist. JOIN project proj ON ir.project_id = proj.id → proj.name AS project_name.",
    },
    {
        "id":          "block_col_ir_inspection_type",
        "type":        BLOCK,
        "description": "inspection_report has no inspection_type column — must JOIN inspection_type table.",
        "sql_pattern": r"\bir\.inspection_type\b(?!_id)",
        "message":     "WRONG COLUMN: ir.inspection_type does not exist. JOIN inspection_type it ON ir.inspection_type_id = it.id → it.name AS inspection_type_name.",
    },
    {
        "id":          "block_col_ir_id_in_select",
        "type":        BLOCK,
        "description": "ir.id is a UUID PK — never select for display. Use ir.inspection_id (varchar).",
        # Tightened to avoid false positive on JOIN conditions like ica.inspection_id = ir.id.
        # Only matches ir.id when it appears as a standalone SELECT expression:
        # SELECT ir.id,  or  SELECT ..., ir.id,  or  SELECT ..., ir.id AS ...
        # Does NOT match ir.id inside JOIN ON clauses or WHERE conditions.
        "sql_pattern": r"\bselect\b\s+(?:[^,]+,\s*)*ir\.id(?:\s+as\s+\w+)?(?:\s*,|\s+from\b)",
        "table_guard": "inspection_report",
        "message":     "WRONG COLUMN: ir.id is the UUID primary key — outputs UUID garbage. Use ir.inspection_id (varchar like '2026/04/ST001/INS001') for display. Note: ir.id in JOIN conditions is correct.",
    },

    # ── close_on misuse in date filters ──────────────────────────────────────
    {
        "id":          "block_close_on_date_filter",
        "type":        BLOCK,
        "description": "close_on is NULL on most ICA rows — filtering by EXTRACT/date_trunc on close_on returns near-empty results.",
        "sql_pattern": r"EXTRACT\s*\([^)]*FROM\s+\w*\.?close_on\b|date_trunc\s*\([^)]*,\s*\w*\.?close_on\b",
        "table_guard": "inspection_corrective_action",
        "message":     (
            "WRONG: close_on is NULL on most inspection_corrective_action rows (only set in special workflows). "
            "NEVER use EXTRACT or date_trunc on close_on for date filtering. "
            "For 'closed this month' use: WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED') "
            "AND date_trunc('month', target_close_out_date) = date_trunc('month', CURRENT_DATE). "
            "Or use created_on for when the ICA was raised."
        ),
    },
    {
        "id":          "block_col_ica_risk_level_bare",
        "type":        BLOCK,
        "description": "ica.risk_level does not exist — only risk_level_id (UUID FK) exists.",
        "sql_pattern": r"\b(ica|inspection_corrective_action)\.risk_level\b(?!_id)",
        "message":     "WRONG COLUMN: ica.risk_level does not exist. JOIN risk_level rl ON ica.risk_level_id = rl.id WHERE rl.name ILIKE '%High%'",
    },
    {
        "id":          "block_col_ica_impact_bare",
        "type":        BLOCK,
        "description": "ica.impact does not exist — only impact_id (UUID FK) exists.",
        "sql_pattern": r"\b(ica|inspection_corrective_action)\.impact\b(?!_id)",
        "message":     "WRONG COLUMN: ica.impact does not exist. JOIN impact im ON ica.impact_id = im.id WHERE im.name ILIKE '%Non Confirmity%'",
    },
    # ICA ghost columns proven in debug traces — never exist on inspection_corrective_action
    {
        "id":          "block_col_ica_times_deferred",
        "type":        BLOCK,
        "description": "times_deferred does not exist on inspection_corrective_action.",
        "sql_pattern": r"\btimes_deferred\b(?!\s+FROM|\s+AS\s+times_deferred)",
        "table_guard": "inspection_corrective_action",
        "message":     "WRONG COLUMN: times_deferred does not exist on inspection_corrective_action. Use COUNT(*) AS times_deferred in GROUP BY aggregation instead.",
    },
    # block_col_ica_capex_status removed — capex_status IS a valid column on
    # inspection_corrective_action (NULL=awaiting, ACCEPTED, REJECTED, DEFERRED).
    # It is a workflow flag only — NOT a financial metric.
    {
        "id":          "block_col_ica_adequacy_status",
        "type":        BLOCK,
        "description": "adequacy_status does not exist on inspection_corrective_action.",
        "sql_pattern": r"\badequacy_status\b",
        "table_guard": "inspection_corrective_action",
        "message":     "WRONG COLUMN: adequacy_status does not exist on inspection_corrective_action. For risk filtering: JOIN risk_level rl ON ica.risk_level_id = rl.id WHERE rl.name = 'High'",
    },

    # ── Hallucinated quarter-comparison columns on inspection_report ──────────
    {
        "id":          "block_col_ir_current_qtr_score",
        "type":        BLOCK,
        "description": "ir.current_qtr_score does not exist — use CTE with CASE WHEN date_trunc.",
        "sql_pattern": r"\bir\.current_qtr_score\b",
        "message":     "WRONG COLUMN: ir.current_qtr_score does not exist. Use: AVG(CASE WHEN ir.submitted_on >= date_trunc('quarter', NOW()) THEN ir.inspection_score END) AS this_q",
    },
    {
        "id":          "block_col_ir_last_qtr_score",
        "type":        BLOCK,
        "description": "ir.last_qtr_score does not exist — use CTE with CASE WHEN date_trunc.",
        "sql_pattern": r"\bir\.last_qtr_score\b",
        "message":     "WRONG COLUMN: ir.last_qtr_score does not exist. Use: AVG(CASE WHEN ir.submitted_on >= date_trunc('quarter', NOW()-INTERVAL '3 months') AND ir.submitted_on < date_trunc('quarter', NOW()) THEN ir.inspection_score END) AS last_q",
    },
    {
        "id":          "block_col_ir_score_improvement",
        "type":        BLOCK,
        "description": "ir.score_improvement does not exist — compute as this_q - last_q in a CTE.",
        "sql_pattern": r"\bir\.score_improvement\b",
        "message":     "WRONG COLUMN: ir.score_improvement does not exist. Compute it in a CTE: ROUND((this_q - last_q)::numeric, 1) AS improvement",
    },

    # ── Hallucinated schedule/portfolio join columns ──────────────────────────
    {
        "id":          "block_col_ipd_cycle_id",
        "type":        BLOCK,
        "description": "inspector_portfolio_details.cycle_id does not exist — cycle link goes through inspector_portfolio.",
        "sql_pattern": r"\bipd\.cycle_id\b",
        "message":     (
            "WRONG COLUMN: inspector_portfolio_details has no cycle_id. "
            "The cycle chain is: inspection_schedule.inspection_cycle_id → inspection_cycle.id "
            "OR: inspector_portfolio_details.portfolio_id → inspector_portfolio.cycle_id → inspection_cycle.id"
        ),
    },
    {
        "id":          "block_col_isched_inspection_schedule_id",
        "type":        BLOCK,
        "description": "inspection_schedule.inspection_schedule_id does not exist — use portfolio_details_id.",
        "sql_pattern": r"\bisched\.inspection_schedule_id\b",
        "message":     (
            "WRONG COLUMN: inspection_schedule has no inspection_schedule_id column. "
            "To join to inspector_portfolio_details: "
            "JOIN inspector_portfolio_details ipd ON isched.portfolio_details_id = ipd.id"
        ),
    },

    # ── UUID FK compared to string literal ───────────────────────────────────
    {
        "id":          "block_risk_level_id_string_compare",
        "type":        BLOCK,
        "description": "risk_level_id is a UUID FK — never compare to a string like 'High'.",
        "sql_pattern": r"\brisk_level_id\s*(?:=|ILIKE|LIKE|!=|<>)\s*'(?![0-9a-f]{8}-)[^']*'",
        "message":     "WRONG COMPARISON: risk_level_id is a UUID FK. Cannot compare to string. JOIN risk_level rl ON ica.risk_level_id = rl.id WHERE rl.name ILIKE '%High%'",
    },
    {
        "id":          "block_impact_id_string_compare",
        "type":        BLOCK,
        "description": "impact_id is a UUID FK — never compare to a string.",
        "sql_pattern": r"\bimpact_id\s*(?:=|ILIKE|LIKE|!=|<>)\s*'(?![0-9a-f]{8}-)[^']*'",
        "message":     "WRONG COMPARISON: impact_id is a UUID FK. JOIN impact im ON ica.impact_id = im.id WHERE im.name ILIKE '%Non Confirmity%'. NOTE: spelled 'Confirmity' not 'Conformity'.",
    },

    # ── Impact table spelling typo ────────────────────────────────────────────
    {
        "id":          "block_conformity_spelling",
        "type":        BLOCK,
        "description": "The impact table stores 'Non Confirmity' (with i) — 'conformity' returns zero rows.",
        "sql_pattern": r"\bconformity\b",
        "table_guard": "impact",
        "message":     "DATA TYPO: impact table stores 'Non Confirmity' (NOT 'Non Conformity'). ILIKE '%conformity%' always returns zero rows. Use: WHERE im.name ILIKE '%Non Confirmity%'",
    },

    # ── Date string arithmetic (AGG-02 bad first attempt) ────────────────────
    {
        "id":          "block_date_string_cast",
        "type":        BLOCK,
        "description": "Never construct date strings with ::TEXT || '-01-01' — use date_trunc or EXTRACT.",
        "sql_pattern": r"\bEXTRACT\s*\(.*CURRENT_DATE.*\)\s*::TEXT\s*\|\|",
        "message":     "BLOCKED: Date string arithmetic via ::TEXT || is fragile. Use: EXTRACT(YEAR FROM col) = EXTRACT(YEAR FROM CURRENT_DATE) or date_trunc('year', col) = date_trunc('year', NOW())",
    },
    {
        "id":          "block_date_concat_year",
        "type":        BLOCK,
        "description": "Never concatenate year strings to build date literals.",
        "sql_pattern": r"TO_CHAR\s*\(\s*CURRENT_DATE.*\)\s*\|\|\s*['\-]",
        "message":     "BLOCKED: String date construction. Use: EXTRACT(YEAR FROM col) = EXTRACT(YEAR FROM CURRENT_DATE)",
    },
    {
        "id":          "block_rolling_365_day_year",
        "type":        BLOCK,
        "description": "CURRENT_DATE - INTERVAL '1 year' is a rolling window, not a calendar year.",
        "sql_pattern": r"CURRENT_DATE\s*-\s*INTERVAL\s*'1\s*year'",
        "message":     (
            "BLOCKED: CURRENT_DATE - INTERVAL '1 year' gives a rolling 365-day window, not calendar year. "
            "Use: EXTRACT(YEAR FROM col) = EXTRACT(YEAR FROM CURRENT_DATE) "
            "or: col >= date_trunc('year', CURRENT_DATE)"
        ),
    },
    {
        "id":          "block_interval_30_days_for_month",
        "type":        BLOCK,
        "description": "INTERVAL '30 days' used as proxy for 'this month' — wrong semantics.",
        # Only fire when question asks about 'this month' or 'current month'
        # but SQL uses INTERVAL '30 days' as the filter (rolling window ≠ calendar month).
        # NEVER fire when user explicitly says 'last 30 days', 'past 30 days', 'in 30 days' etc.
        "sql_pattern": r"INTERVAL\s*'30\s*days?'",
        "question_pattern": r"\b(this\s+month|current\s+month|in\s+(?:the\s+)?month)\b",
        "message":     (
            "BLOCKED: INTERVAL '30 days' is a rolling 30-day window, NOT 'this month'. "
            "For 'this month' use: date_trunc('month', col) = date_trunc('month', NOW()). "
            "Only use INTERVAL '30 days' when the question says 'last 30 days' or 'past 30 days'."
        ),
    },
    {
        "id":          "block_inspector_not_in_pattern",
        "type":        BLOCK,
        "code":        "INSPECTOR_NOT_IN_PATTERN",
        "description": "NOT IN (SELECT inspector_user_id) misses new inspectors with zero inspections.",
        # Fires when: query asks about inactive/missing inspectors using NOT IN subquery
        # on inspection_report — this silently excludes brand-new inspectors.
        "sql_pattern": (
            r"\bNOT\s+IN\s*\(\s*SELECT\b.*?\binspector_user_id\b.*?\bFROM\s+inspection_report\b"
        ),
        "question_pattern": (
            r"\b(inspector|inspectors?)\b.{0,50}"
            r"\b(haven.t|not|no|never|inactive|zero|0)\b.{0,30}"
            r"\b(inspection|submitted|conducted|done|performed|activity)\b"
        ),
        "message": (
            "WRONG PATTERN: NOT IN (SELECT inspector_user_id FROM inspection_report) "
            "misses new inspectors who have ZERO inspections — they have no rows in "
            "inspection_report so they don't appear in either the subquery or the result. "
            "CORRECT: start from users JOIN user_roles JOIN role, "
            "LEFT JOIN inspection_report ON ir.inspector_user_id = u.id, "
            "HAVING MAX(ir.submitted_on) < CURRENT_DATE - INTERVAL '...' "
            "OR MAX(ir.submitted_on) IS NULL"
        ),
    },
    {
        "id":          "block_responsible_fk_join",
        "type":        BLOCK,
        "code":        "WRONG_JOIN",
        "description": "ica.responsible is a string ENUM — never JOIN it to users.",
        "sql_pattern": r"\bjoin\s+users\b.*?\b(?:responsible|pending_with)\s*=\s*\w+\.id\b",
        "message":     "HALLUCINATION: responsible and pending_with are plain string ENUMs, NOT UUID FKs. Never JOIN them to users. Use: WHERE ica.responsible = 'CLIENT' (uppercase).",
    },

    # ── close_on usage — DEAD COLUMN, confirmed all-NULL ─────────────────────
    {
        "id":          "block_close_on_arithmetic",
        "type":        BLOCK,
        "code":        "DEAD_COLUMN",
        "description": "close_on is a DEAD COLUMN (all values NULL, confirmed) — any comparison/arithmetic is meaningless.",
        # Catches arithmetic (+/-), EXTRACT/date_trunc, AND comparisons (>, <, >=, <=, =)
        "sql_pattern": (
            r"\bclose_on\s*[-+<>]=?\s*\w"
            r"|\bextract\s*\(.*?\bclose_on\b"
            r"|\bdate_trunc\s*\([^)]*,\s*\w*\.?close_on\b"
        ),
        "message": (
            "BLOCKED: close_on is a DEAD COLUMN — all values are NULL, confirmed. "
            "Any comparison or arithmetic on it always evaluates to NULL/false. "
            "Use completed_on for closure timing instead:\n"
            "  'overdue' (still open, past due)     → status = 'OVERDUE' (the authoritative field — "
            "do not recompute from target_close_out_date)\n"
            "  'closed late' / 'not closed on time' → status = 'CLOSED' AND completed_on > target_close_out_date\n"
            "These two are MUTUALLY EXCLUSIVE — never OR them into one count."
        ),
    },

    # ── Hardcoded year ────────────────────────────────────────────────────────
    {
        "id":          "block_responsible_groupby_fm",
        "type":        BLOCK,
        "description": "GROUP BY ica.responsible returns CLIENT/INTERNAL/SUB_CONTRACTOR — not FM managers",
        # Guarded (2026-06-12): only fire when the QUESTION is actually about
        # FM managers / performers — "breakdown by responsible party" is a
        # legitimate generic GROUP BY ica.responsible question and should pass.
        # This rule exists to catch the case where the user means PEOPLE
        # (FM managers) but the model groups by the enum instead.
        "question_pattern": r"\b(fm|facility)\s*manager|capex\s*(generator|converter)s?|top\s*perform",
        "sql_pattern": r"GROUP\s+BY\s+(?:[\w.]+\s*,\s*)*ica\.responsible",
        "message":     (
            "WRONG: GROUP BY ica.responsible gives CLIENT/INTERNAL_OPERATIONS/SUB_CONTRACTOR — "
            "these are who carries out the fix, NOT FM managers or users. "
            "For FM manager grouping: GROUP BY u.id, u.first_name, u.last_name "
            "with JOIN users u ON ir.inspectee_user_id = u.id. "
            "ica.responsible is an enum column, not a person."
        ),
    },
    {
        "id":          "block_hardcoded_year",
        "type":        WARN,
        "code":        "HARDCODED_YEAR",
        "description": (
            "Hardcoded year — consider EXTRACT(YEAR FROM CURRENT_DATE) for current-year queries. "
            "SOFT_WARN, not BLOCK: intentional historical-year queries (e.g. 'inspections in 2024') "
            "are legitimate and must not be hard-rejected — matches core/validator.py's existing, "
            "deliberate demotion of this same check (see validator.py's HARDCODED_YEAR comment)."
        ),
        "sql_pattern": r"\bextract\s*\(\s*year\s+from\s+\w[\w.]*\s*\)\s*=\s*20\d\d\b",
        "message":     "Hardcoded year '= 20XX' — consider EXTRACT(YEAR FROM CURRENT_DATE) for current-year queries",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # WARN — soft warnings (SQL executes but flag is surfaced)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id":          "warn_no_limit",
        "type":        WARN,
        "code":        "MISSING_LIMIT",
        "description": "Non-aggregate queries without LIMIT can return huge result sets.",
        "sql_pattern": r"^(?!.*\b(COUNT|SUM|AVG|MIN|MAX)\s*\()(?!.*\bLIMIT\b)",
        "message":     "WARNING: No LIMIT on non-aggregate query. Add LIMIT 100.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # AUTOCORRECT — applied automatically in clean_sql() before validation
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id":          "autocorrect_ica_ir_join_varchar_uuid",
        "type":        AUTOCORRECT,
        "description": "Fix ICA→IR join using varchar column instead of UUID PK.",
        "sql_pattern": r"\bir\.inspection_id\s*=\s*ica\.inspection_id\b",
        "replacement": "ica.inspection_id = ir.id",
        "message":     "[autocorrect] Fixed ICA→IR join direction (varchar=UUID → UUID=UUID)",
    },
    {
        "id":          "autocorrect_ica_ir_join_reversed",
        "type":        AUTOCORRECT,
        "description": "Fix reversed ICA→IR join.",
        "sql_pattern": r"\bica\.inspection_id\s*=\s*ir\.inspection_id\b",
        "replacement": "ica.inspection_id = ir.id",
        "message":     "[autocorrect] Fixed reversed ICA→IR join",
    },
    {
        "id":          "autocorrect_strip_responsible_users_join",
        "type":        AUTOCORRECT,
        "description": "Strip invalid LEFT JOIN users on ica.responsible enum comparison.",
        # Matches: LEFT JOIN users <alias> ON ica.responsible = '<ENUM>' AND <anything up to u.id>
        "sql_pattern": (
            r"\s*LEFT\s+JOIN\s+users\s+\w+\s+ON\s+ica\.responsible\s*=\s*'[A-Z_]+'"
            r"\s+AND\s+[\w.]+\s*=\s*\w+\.id"
        ),
        "replacement": "",
        "message":     "[autocorrect] Removed invalid users join on ica.responsible enum",
    },
    {
        "id":          "autocorrect_responsible_party_select",
        "type":        AUTOCORRECT,
        "description": "Replace u.first_name||' '||u.last_name AS responsible_party with ica.responsible.",
        "sql_pattern": (
            r"\w+\.first_name\s*\|\|\s*'\s+'\s*\|\|\s*\w+\.last_name\s+AS\s+responsible_party"
        ),
        "replacement": "ica.responsible AS responsible_party",
        "message":     "[autocorrect] responsible_party now selects ica.responsible enum, not a user name",
    },
    {
        "id":          "autocorrect_ica_ir_join_cast",
        "type":        AUTOCORRECT,
        "description": "Fix CAST(ir.inspection_id AS ...) = ica.inspection_id type mismatch.",
        "sql_pattern": r"CAST\s*\(\s*ir\.inspection_id\s+AS\s+\w+\s*\)\s*=\s*ica\.inspection_id\b",
        "replacement": "ica.inspection_id = ir.id",
        "message":     "[autocorrect] Fixed CAST ICA→IR join",
    },
    {
        "id":          "autocorrect_ica_ir_join_cast_typecast",
        "type":        AUTOCORRECT,
        "description": "Fix ir.inspection_id::type = ica.inspection_id type mismatch.",
        "sql_pattern": r"\bir\.inspection_id\s*::\s*\w+(?:\(\d+\))?\s*=\s*ica\.inspection_id\b",
        "replacement": "ica.inspection_id = ir.id",
        "message":     "[autocorrect] Fixed typecast ICA→IR join",
    },
    {
        "id":          "autocorrect_reserved_alias_is",
        "type":        AUTOCORRECT,
        "description": "Replace 'is' as table alias (SQL reserved word) with 'isch'.",
        "sql_pattern": r"(\b\w+\s+)is\b(?=\s*\.|\s+ON\b|\s+WHERE\b|\s+GROUP\b|\s+ORDER\b|\s+LEFT\b|\s+JOIN\b|\s+INNER\b|\s*\n|\s*,)",
        "replacement": r"\1isch",
        "message":     "[autocorrect] Replaced reserved alias 'is' with 'isch'",
    },

    {
        "id":          "autocorrect_impact_ilike",
        "type":        AUTOCORRECT,
        "description": "aq.label ILIKE '%impact%' matches wrong labels — use exact match",
        "sql_pattern": r"aq\.label\s+ILIKE\s+'%[Ii]mpact%'",
        "replacement": "aq.label = 'Impact'",
        "message":     "[autocorrect] Replaced ILIKE Impact with exact aq.label = 'Impact'",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # INJECT — append hint to schema_hint when question matches
    # Higher priority fires first; hints are concatenated in priority order.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id":             "inject_top_capex_converters",
        "type":           INJECT,
        "priority":       97,
        "description":    "Top N capex converters — inject exact CTE SQL, bypass embedding.",
        "required_tables": ["po_details"],
        "forbidden_patterns": [r"ORDER BY ica\.capex", r"capex_status\s*=\s*'ACCEPTED'", r"ica\.expenditure"],
        # Suppress the broader generic FM-manager routing hint — its routing
        # advice can conflict with this rule's "USE THIS EXACT SQL" template
        # when both fire on an ambiguous "top capex converter for FM manager" question.
        "suppresses":     ["inject_fm_manager_routing"],
        "question_pattern": (
            r"\b(top|best|highest|leading).{0,20}(capex.{0,15}converter|convert.{0,15}capex)s?\b"
            r"|\b(capex.{0,15}converter|convert.{0,15}capex)s?.{0,20}(top|best|rank|leaderboard)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SQL FOR TOP CAPEX CONVERTERS (do not deviate):\n"
            "WITH po_by_ica AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po\n"
            "    FROM po_details WHERE active = true GROUP BY 1\n"
            ")\n"
            "SELECT\n"
            "    u.first_name || ' ' || u.last_name  AS fm_manager,\n"
            "    COUNT(DISTINCT ica.id)              AS ca_count,\n"
            "    COALESCE(SUM(ica.capex), 0)         AS capex_generated,\n"
            "    COALESCE(SUM(p.total_po), 0)        AS capex_converted,\n"
            "    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "JOIN users u ON ir.inspectee_user_id = u.id\n"
            "LEFT JOIN po_by_ica p ON p.ica_id = ica.id\n"
            "WHERE ir.active = true AND ica.active = true AND u.active = true AND ica.capex > 0\n"
            "GROUP BY u.id, u.first_name, u.last_name\n"
            "ORDER BY capex_converted DESC LIMIT 5;\n"
            "NEVER: ORDER BY ica.capex — that ranks by GENERATED not CONVERTED\n"
            "NEVER: ica.expenditure filter — that is a category flag, not conversion\n"
        ),
    },
    {
        "id":             "inject_top_capex_generators",
        "type":           INJECT,
        "priority":       96,
        "description":    "Top N capex generators — ranked by SUM(ica.capex), includes po join for context.",
        "required_tables": ["po_details"],
        "forbidden_patterns": [r"ORDER BY.*capex_converted", r"ica\.expenditure"],
        "suppresses":     ["inject_fm_manager_routing"],
        "question_pattern": (
            # Must explicitly mention fm manager / facility manager context
            r"\b(top|best|highest|leading|most).{0,25}capex.{0,20}generat\w*.{0,20}(fm|facility.{0,5}manager|inspectee)\b"
            r"|\b(top|best|highest|leading|most).{0,25}(fm|facility.{0,5}manager).{0,25}capex.{0,20}generat\w*\b"
            r"|\bcapex.{0,20}generat\w*.{0,20}(top|best|rank|leaderboard).{0,20}(fm|facility|manager)\b"
            r"|\b(fm|facility).{0,10}manager.{0,30}generat.{0,20}(most|highest|top).{0,20}capex\b"
            r"|\bwhich.{0,10}(fm|facility).{0,10}manager.{0,30}(most|highest).{0,20}capex\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SQL FOR TOP CAPEX GENERATORS (ranked by generated, NOT converted):\n"
            "WITH po_by_ica AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po\n"
            "    FROM po_details WHERE active = true GROUP BY 1\n"
            ")\n"
            "SELECT\n"
            "    u.first_name || ' ' || u.last_name  AS fm_manager,\n"
            "    COUNT(DISTINCT ica.id)              AS ca_count,\n"
            "    COALESCE(SUM(ica.capex), 0)         AS capex_generated,\n"
            "    COALESCE(SUM(p.total_po), 0)        AS capex_converted,\n"
            "    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "JOIN users u ON ir.inspectee_user_id = u.id\n"
            "LEFT JOIN po_by_ica p ON p.ica_id = ica.id\n"
            "WHERE ir.active = true AND ica.active = true AND u.active = true AND ica.capex > 0\n"
            "GROUP BY u.id, u.first_name, u.last_name\n"
            "ORDER BY capex_generated DESC LIMIT 10;\n"
            "NEVER: ORDER BY capex_converted — that ranks converters not generators\n"
        ),
    },
    {
        "id":             "inject_top_fm_performers",
        "type":           INJECT,
        "priority":       96,
        "description":    "Top performing FM managers — inject CTE SQL ranking by conversion rate.",
        "required_tables": ["po_details"],
        "forbidden_patterns": [r"AVG\(ir\.inspection_score\)", r"ORDER BY avg_score", r"GROUP BY ica\.responsible"],
        "suppresses":     ["inject_fm_manager_routing"],
        "question_pattern": (
            r"\b(top|best|highest).{0,20}(perform|fm.{0,10}manager|facility.{0,10}manager).{0,30}"
            r"(manager|fm|perform|rank|leaderboard)\b"
            r"|\b(who.{0,20}(best|top).{0,20}(fm|facility).{0,10}manager"
            r"|best.{0,20}fm.{0,10}manager)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SQL FOR TOP FM MANAGERS — rank by conversion_pct, filter by FM role, NOT avg_score:\n"
            "NEVER: AVG(ir.inspection_score) — that is inspector quality, not FM performance\n"
            "WITH po_by_ica AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po\n"
            "    FROM po_details WHERE active = true GROUP BY 1\n"
            "),\n"
            "pr_by_ica AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr\n"
            "    FROM pr_details WHERE active = true GROUP BY 1\n"
            ")\n"
            "SELECT\n"
            "    u.first_name || ' ' || u.last_name          AS fm_manager,\n"
            "    COALESCE(SUM(ica.capex), 0)                  AS capex_generated,\n"
            "    COALESCE(SUM(p.total_po), 0)                 AS capex_converted,\n"
            "    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct,\n"
            "    COALESCE(SUM(pr.total_pr), 0)                AS opex_incurred\n"
            "FROM inspection_report ir\n"
            "JOIN users u ON ir.inspectee_user_id = u.id\n"
            "LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id AND ica.active = true AND ica.capex > 0\n"
            "LEFT JOIN po_by_ica p  ON p.ica_id  = ica.id\n"
            "LEFT JOIN pr_by_ica pr ON pr.ica_id = ica.id\n"
            "WHERE ir.active = true AND u.active = true\n"
            "  AND EXISTS (\n"
            "      SELECT 1 FROM user_roles ur JOIN role r ON ur.role_id = r.id\n"
            "      WHERE ur.user_id = u.id\n"
            "        AND r.name IN ('FM Manager','FM Engineer','AD FM',\n"
            "                       'Asst FM Manager','FM Admin','FM Supervisor')\n"
            "  )\n"
            "GROUP BY u.id, u.first_name, u.last_name\n"
            "HAVING SUM(ica.capex) > 0\n"
            "ORDER BY conversion_pct DESC NULLS LAST, opex_incurred ASC\n"
            "LIMIT 5;\n"
        ),
    },
    {
        "id":             "inject_avg_ma_close_time",
        "type":           INJECT,
        "priority":       85,
        "description":    "Average time to close mitigative action — use target_close_out_date not close_on.",
        "question_pattern": (
            r"\b(average|avg|mean).{0,20}(time|days?|hours?|duration).{0,20}"
            r"(close|clos|resolv|complet).{0,20}(mitigative|action|ma)\b"
            r"|\b(how.{0,10}long|time.{0,10}taken?).{0,30}(close|clos|resolv)\b"
        ),
        "hint": (
            "\nAVERAGE TIME TO CLOSE MITIGATIVE ACTION:\n"
            "NEVER use close_on — it is NULL on most rows (only set in special workflows).\n"
            "USE target_close_out_date - created_on for the planned close window:\n"
            "SELECT\n"
            "    AVG(EXTRACT(EPOCH FROM (ica.target_close_out_date - ica.created_on)) / 86400) AS avg_days,\n"
            "    COUNT(*) AS total_closed\n"
            "FROM inspection_corrective_action ica\n"
            "WHERE ica.status IN (\'CLOSED\', \'CLOSE_WITH_DEFERRED\')\n"
            "  AND ica.active = true\n"
            "  AND ica.target_close_out_date IS NOT NULL\n"
            "  AND ica.created_on IS NOT NULL;\n"
            "NEVER: EXTRACT(EPOCH FROM (close_on - created_on)) — close_on IS NULL on most rows\n"
        ),
    },
    {
        "id":             "inject_qa_recent_form",
        "type":           INJECT,
        "priority":       92,
        "description":    "Q&A from most recent form — provide complete 2-table SQL direct.",
        "suppresses":     [],
        "question_pattern": (
            r"\b(list|show|get|fetch|display|what).{0,20}"
            r"(question|answer|qa|form.{0,10}answer|answer.{0,10}form).{0,40}"
            r"(most.{0,10}recent(?:ly)?|latest|last)\b"
            r"|\b(most.{0,10}recent(?:ly)?|latest|last).{0,30}"
            r"(question|answer|qa|form.{0,10}answer)\b"
            # 'questions and answers for the most recently' — noun comes before recency qualifier
            r"|\b(question|answer)s?.{0,50}(most.{0,10}recent(?:ly)?|latest|last)\b"
        ),
        "hint": (
            "\nQ&A FROM MOST RECENT FORM — use this EXACT SQL. ONE query, NOT two separate queries:\n"
            "SELECT aq.label AS question, aa.answer_text AS answer, aa.module_name AS form_section\n"
            "FROM ai_answers aa\n"
            "JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
            "WHERE aa.inspection_report_id = (\n"
            "    SELECT aa2.inspection_report_id FROM ai_answers aa2\n"
            "    ORDER BY aa2.submitted_on DESC LIMIT 1\n"
            ")\n"
            "\n"
            "ORDER BY aa.module_name, aq.label\n"
            "LIMIT 200;\n"
            "CRITICAL: This is a SINGLE SQL query — do NOT first lookup the inspection_id and then\n"
            "do a second query. The subquery handles the inspection lookup internally.\n"
            "NEVER: SELECT ir.inspection_id first, then ask for Q&A — use the complete SQL above.\n"
        ),
    },
    {
        "id":             "inject_pct_with_ma",
        "type":           INJECT,
        "priority":       100,
        "description":    "Percentage of inspections with mitigative action — inject exact SQL.",
        "question_pattern": (
            r"\b(percentage|percent|pct|proportion|ratio).{0,40}"
            r"(mitigative|finding)\b"
            r"|\b(mitigative|finding).{0,40}"
            r"(percentage|percent|pct|proportion|ratio)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SQL — do not generate your own:\n"
            "SELECT ROUND(\n"
            "  100.0 * COUNT(DISTINCT CASE WHEN ica.id IS NOT NULL THEN ir.id END)\n"
            "  / NULLIF(COUNT(DISTINCT ir.id), 0), 1\n"
            ") AS pct_with_mitigative_action\n"
            "FROM inspection_report ir\n"
            "LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id\n"
            f"WHERE ir.{_DRAFT_EXCLUSION};\n"
        ),
    },
    {
        "id":             "inject_inspector_performance",
        "type":           INJECT,
        "priority":       93,
        "description":    "Inspector performance summary — assigned/conducted/observations — inject CTE pattern.",
        "question_pattern": (
            r"\b(inspector).{0,50}(performance|summary|breakdown|stats?|report|overview)\b"
            r"|\b(performance|summary|breakdown|stats?).{0,50}(inspector)\b"
            r"|\b(assigned|conducted).{0,30}(inspection|inspector)\b"
            r"|\b(inspector).{0,30}(assigned|conducted|observations?|high.?risk)\b"
        ),
        "hint": (
            "\nINSPECTOR PERFORMANCE SUMMARY — CRITICAL DEFINITIONS:\n"
            "  assigned   = COUNT(ir.id)                               — ALL reports for inspector\n"
            "  conducted  = COUNT FILTER (WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION'))\n"
            "               NEVER status != 'DRAFT' — that includes CLOSED\n"
            "               NEVER status = 'SUBMITTED' alone — misses UNDER_REVIEW and RETURN_FOR_MOD\n"
            "  total_observations = SUM(COUNT DISTINCT aa.submission_id per inspection)\n"
            "               NEVER COUNT(*) on ai_answers — overcounts (many rows per observation)\n"
            "  high_risk_observations = COUNT DISTINCT submission_id WHERE Risk Level = High\n"
            "  NEVER join ai_answers twice in main query — causes cross-join fan-out\n"
            "  NEVER join inspection_schedule — irrelevant to performance counts\n"
            "  Pre-aggregate observations in a CTE:\n"
            "WITH obs AS (\n"
            "    SELECT aa.inspection_report_id,\n"
            "           COUNT(DISTINCT aa.submission_id)   AS total_obs,\n"
            "           COUNT(DISTINCT CASE\n"
            "               WHEN aq.label = 'Risk Level' AND aa.answer_text ILIKE '%High%'\n"
            "               THEN aa.submission_id END)     AS high_risk_obs\n"
            "    FROM ai_answers aa\n"
            "    JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
            "    WHERE aa.module_name = 'Inspection Form'\n"
            "    GROUP BY aa.inspection_report_id\n"
            ")\n"
            "SELECT u.first_name || ' ' || u.last_name          AS inspector_name,\n"
            "       COUNT(ir.id)                                AS assigned,\n"
            "       COUNT(ir.id) FILTER (WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')) AS conducted,\n"
            "       COUNT(ir.id) FILTER (WHERE ir.status = 'CLOSED') AS completed,\n"
            "       COALESCE(SUM(obs.total_obs), 0)             AS total_observations,\n"
            "       COALESCE(SUM(obs.high_risk_obs), 0)         AS high_risk_observations\n"
            "FROM users u\n"
            "JOIN inspection_report ir ON ir.inspector_user_id = u.id AND ir.active = true\n"
            "LEFT JOIN obs ON obs.inspection_report_id = ir.id\n"
            "WHERE u.active = true\n"
            "GROUP BY u.id, u.first_name, u.last_name\n"
            "ORDER BY conducted DESC NULLS LAST LIMIT 100;\n"
        ),
    },
    {
        "id":             "inject_inspector_count",
        "type":           INJECT,
        "priority":       90,
        "description":    "Inspector count query — prevents hardcoded years.",
        "question_pattern": (
            r"\b(which|who|top|most|count|how\s+many|list).{0,30}"
            r"(inspector|inspectors?).{0,30}"
            r"(most|conducted|done|most\s+inspections?|count|number|year)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT PATTERN for inspector count queries:\n"
            "-- conducted = IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION') — in-flight\n"
            "-- completed = CLOSED only\n"
            "-- assigned  = all reports linked to inspector (any non-draft status)\n"
            "SELECT u.first_name || ' ' || u.last_name AS inspector_name,\n"
            "       COUNT(*) AS assigned,\n"
            "       COUNT(*) FILTER (WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')) AS conducted,\n"
            "       COUNT(*) FILTER (WHERE ir.status = 'CLOSED') AS completed\n"
            "FROM inspection_report ir\n"
            "JOIN users u ON ir.inspector_user_id = u.id\n"
            "WHERE ir.active = true AND ir.status != 'DRAFT'\n"
            "  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)\n"
            "GROUP BY u.id, u.first_name, u.last_name\n"
            "ORDER BY conducted DESC LIMIT 1;\n"
            "CRITICAL: NEVER write '= 2023' or '= 2026' — always use EXTRACT(YEAR FROM CURRENT_DATE).\n"
            "CRITICAL: conducted = IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION'). NEVER status = 'SUBMITTED' only.\n"
        ),
    },
    {
        "id":             "inject_financial_routing",
        "required_tables": [],
        "forbidden_patterns": [r"SUM\\(ica\\.opex\\)", r"capex_status\\s*=\\s*'ACCEPTED'"],

        "type":           INJECT,
        "priority":       88,
        "description":    "CAPEX/OPEX financial queries — route to po_details/pr_details/ai_answers.",
        "question_pattern": (
            r"\b(po[_\s]detail|pr[_\s]detail|purchase\s+order|purchase\s+requisition|"
            r"purchase\s+order\s+amount|po\s+amount|total\s+po|"
            r"capex.{0,30}(generat|convert|lost|loss|incurr)|"
            r"opex.{0,30}(incurr|facilit|spend|higher|exceed)|"
            r"(generat|convert|lost|incurr).{0,30}(capex|opex)|"
            r"how\s+much.{0,40}(convert|capex|opex|po|purchase)|"
            r"opex.{0,30}(higher|more|greater|exceed).{0,30}capex|"
            r"conversion\s+rate|capex\s+conversion|"
            r"(convert|capex).{0,20}(inspectee|fm|manager|facility)|"
            # plural 'converters' / 'generators' explicitly
            r"(capex|opex)\s+(converter|generator)s?|"
            r"(converter|generator)s?\s+(capex|opex)|"
            r"top.{0,15}(capex|opex).{0,15}(converter|generator)s?|"
            # facility cost / expenditure queries
            r"(facility|site).{0,30}(costs?|spend|expenditure|expensive)|"
            r"(cost|spend|expenditure).{0,30}(facility|site)|"
            r"most\s+expensive\s+(facility|site)|"
            r"rival.{0,20}(quote|bid|amount)|financial\s+summary)\b"
        ),
        "hint": _FINANCIAL_HINT,
    },
    {
        "id":             "inject_fm_manager_routing",
        "forbidden_patterns": [r"GROUP\\s+BY\\s+ica\\.responsible", r"AVG\\(ir\\.inspection_score\\)"],

        "type":           INJECT,
        "priority":       85,
        "description":    "FM Manager / inspectee queries — route to inspectee_user_id not inspector_user_id.",
        "question_pattern": (
            r"\b(fm\s+manager|facility\s+manager|inspectee|fm\s+engineer|"
            r"fm\s+auditee|fm\s+leaderboard|capex\s+leaderboard|"
            # 'top N capex converters/generators' — plural handled by 's?' suffix
            r"top.{0,25}(fm|manager|capex|opex).{0,25}(generator|converter|performer|leaderboard)s?|"
            # standalone 'capex converters/generators'
            r"capex\s+(generator|converter|generat|convert)s?|"
            r"(generator|converter)s?.{0,20}capex|"
            # 'who are the top capex converters' type phrasing
            r"(best|highest|top|leading|rank).{0,30}(capex|opex).{0,30}(converter|generator)s?|"
            r"(converter|generator)s?.{0,30}(capex|opex))\b"
        ),
        "hint": (
            "\nFM MANAGER ROUTING — CRITICAL RULES:\n"
            "RULE 1: ica.responsible is a STRING ENUM (CLIENT/INTERNAL_OPERATIONS/SUB_CONTRACTOR)\n"
            "  It is NOT a person. NEVER GROUP BY ica.responsible for FM manager queries.\n"
            "  NEVER SELECT ica.responsible AS fm_manager — this gives 3 rows, not people.\n"
            "RULE 2: NEVER rank FM managers by avg_inspection_score — that is inspector quality.\n"
            "  FM performance = CAPEX conversion rate + low OPEX (pr_details), not inspection scores.\n"
            "RULE 3: FM Manager = inspectee_user_id (NOT inspector_user_id)\n"
            "  CORRECT JOIN: JOIN users u ON ir.inspectee_user_id = u.id\n"
            "  CORRECT GROUP: GROUP BY u.id, u.first_name, u.last_name\n"
            "\nFM MANAGER PERFORMANCE — correct ranking:\n"
            "  GOOD: High conversion = SUM(po.po_amount) / NULLIF(SUM(ica.capex),0) * 100\n"
            "  BAD:  High OPEX = SUM(pr.pr_amount) from pr_details (Khidmah pays internally)\n"
            "  NEVER SUM(ica.opex) — use pr_details for actual OPEX paid\n"
            "  Best FM manager = highest conversion_pct DESC, opex_incurred ASC\n"
            "\nCAPEX LEADERBOARD:\n"
            "  CAPEX Generated = SUM(ica.capex)\n"
            "  CAPEX Converted = SUM(po.po_amount) via po_details — NOT capex_status='ACCEPTED'\n"
            "  CAPEX Lost      = SUM(ica.capex) - COALESCE(SUM(po.po_amount), 0)\n"
            "  OPEX Incurred   = SUM(pr.pr_amount) via pr_details\n"
            "  capex_status IS a valid column but is a workflow flag — NOT a financial metric\n"
            "\nPRE-AGGREGATE to avoid row multiplication:\n"
            "  WITH po_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total FROM po_details WHERE active=true GROUP BY 1)\n"
            "  WITH pr_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total FROM pr_details WHERE active=true GROUP BY 1)\n"
        ),
    },
    {
        "id":             "inject_risk_observation_routing",
        "required_tables": ["risk_level"],
        "forbidden_patterns": [r"ai_answers.*label.*Risk Level", r"aa\.answer_text.*ILIKE.*High"],

        "type":           INJECT,
        "priority":       87,
        "description":    "ICA mitigative action risk queries — route to risk_level table via UUID FK. Does NOT apply to form observation queries which correctly use ai_answers.",
        "suppresses":     [],
        "question_pattern": (
            # Only fires for MITIGATIVE ACTION risk queries.
            # Form observation queries ("high risk observations") use ai_answers correctly.
            r"\b(high.{0,10}risk|critical.{0,10}risk).{0,30}"
            r"(mitigative.{0,10}action)|"
            r"\b(mitigative.{0,10}action).{0,30}"
            r"(high.{0,10}risk|critical.{0,10}risk)|"
            r"\b(risk.{0,10}breakdown|breakdown.{0,10}risk|"
            r"risk.{0,10}distribut|risk.{0,10}count|count.{0,10}risk|"
            r"risk.{0,10}by.{0,20}facilit|by.{0,10}risk.{0,10}level)\b"
        ),
        "hint": (
            "\nICA RISK ROUTING — mitigative action risk levels use inspection_corrective_action:\n"
            "  For risk level COUNTS / BREAKDOWNS on mitigative actions:\n"
            "  ALWAYS use the risk_level table via inspection_corrective_action:\n"
            "  NOTE: form 'high risk observations' uses ai_answers correctly — this rule\n"
            "  only applies to mitigative-action risk queries.\n"
            "  ALWAYS use the risk_level table via inspection_corrective_action:\n"
            "  JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "  WHERE rl.name = \'High\'   -- or \'Medium\' / \'Low\'\n"
            "\nCorrect pattern for high-risk observations this month:\n"
            "SELECT COUNT(*) AS high_risk_count\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "WHERE rl.name = \'High\'\n"
            "  AND ir.active = true AND ica.active = true\n"
            "  AND date_trunc(\'month\', ir.submitted_on) = date_trunc(\'month\', CURRENT_DATE);\n"
            "\nNEVER: ai_answers WHERE label=\'Observation\' AND answer_text ILIKE \'%High%\'\n"
            "NEVER: ai_questions WHERE label=\'Risk Level\'\n"
        ),
    },
    {
        "id":             "inject_ica_join",
        "type":           INJECT,
        "priority":       80,
        "description":    "ICA + facility/grouping — prevent varchar=UUID join mismatch.",
        "question_pattern": (
            r"\b(mitigative.?action|capex|opex).{0,50}"
            r"(facility|client|inspector|type|count|by|group|breakdown)\b"
            r"|\b(facility|client|inspector|type|breakdown).{0,50}"
            r"(mitigative.?action|capex|opex)\b"
        ),
        "hint": (
            "\nCRITICAL JOIN — inspection_corrective_action to inspection_report:\n"
            "  ica.inspection_id → UUID FK to inspection_report.id (UUID PK)\n"
            "  ir.inspection_id  → VARCHAR human-readable ('2026/04/ST001/INS001') — DIFFERENT\n"
            "  ALWAYS: JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id\n"
            "  NEVER:  ON ir.inspection_id = ica.inspection_id  ← varchar=UUID type mismatch\n"
        ),
    },
    {
        "id":             "inject_most_recent_inspection",
        "type":           INJECT,
        "priority":       70,
        "description":    "Most recent inspection — force direct inspection_report query.",
        "question_pattern": (
            r"\b(most\s+recent|latest|last)\s+(inspection|site|facility|form)\b"
            r"|\b(which|what)\s+(facility|site).{0,30}(most\s+recent|latest|last)\b"
        ),
        "hint": (
            "\nDIRECT QUERY — do NOT route through ai_answers for this:\n"
            "SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on, ir.status\n"
            "FROM inspection_report ir\n"
            "JOIN facility fac ON ir.facility_id = fac.id\n"
            "WHERE ir.status != 'DRAFT'\n"
            "ORDER BY ir.submitted_on DESC LIMIT 1;\n"
            "CRITICAL: always SELECT ir.inspection_id — needed for multi-turn follow-up.\n"
        ),
    },
    {
        "id":             "inject_deferred_chain",
        "type":           INJECT,
        "priority":       80,
        "description":    "Deferred / recurring action chain — CLOSE_WITH_DEFERRED pattern.",
        "question_pattern": (
            r"\b(deferred|recurring|recur|carried.?forward|repeated.?finding|"
            r"same.?issue|chronic|persistent|repeat.?observation|"
            r"close.?with.?deferred|multiple.?cycles?)\b"
        ),
        "hint": (
            "\nCLOSE_WITH_DEFERRED — deferred/recurring action pattern:\n"
            "  Status value: 'CLOSE_WITH_DEFERRED' (exact uppercase)\n"
            "  ALWAYS join: ica → ir (ON ica.inspection_id = ir.id) → facility\n"
            "Example (recurring issues at a facility):\n"
            "SELECT ica.cause, COUNT(*) AS recurrence_count, fac.name AS facility_name\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "JOIN facility fac ON ir.facility_id = fac.id\n"
            "WHERE ica.status = 'CLOSE_WITH_DEFERRED'\n"
            "GROUP BY ica.cause, fac.name ORDER BY recurrence_count DESC LIMIT 20;\n"
        ),
    },
    {
        "id":             "inject_return_for_modification",
        "type":           INJECT,
        "priority":       70,
        "description":    "Return for modification — exact WHERE clause, no extra conditions.",
        "question_pattern": r"\breturn.{0,10}modif\b",
        "hint": (
            "\nUSE EXACTLY: WHERE ir.status = 'RETURN_FOR_MODIFICATION'"
            " — no date filter, no extra conditions.\n"
        ),
    },
    # Portfolio "sent back for modification" was getting hijacked by the
    # semantically-similar-but-wrong-table "inspections returned for modification"
    # seed example (2026-07-08: DeepSeek ignored Llama's correct ip.status='MODIFICATION'
    # schema_hint and generated ir.status='RETURN_FOR_MODIFICATION' SQL instead —
    # executed fine, silently answered the wrong question). required_tables forces
    # a guarded retry if inspector_portfolio doesn't end up in the generated SQL.
    {
        "id":             "inject_portfolio_modification",
        "type":           INJECT,
        "priority":       92,
        "description":    "Portfolio sent-back/modification questions — inspector_portfolio.status, NOT inspection_report.status.",
        "required_tables": ["inspector_portfolio"],
        "forbidden_patterns": [r"RETURN_FOR_MODIFICATION"],
        "suppresses":     ["inject_return_for_modification"],
        "question_pattern": (
            r"\bportfolio.{0,30}(modif|sent\s*back|returned)\b"
            r"|(modif|sent\s*back|returned).{0,30}\bportfolio\b"
        ),
        "hint": (
            "\nPORTFOLIO MODIFICATION — this is about inspector_portfolio, NOT inspection_report:\n"
            "  inspector_portfolio.status = 'MODIFICATION' means the portfolio was sent back for changes.\n"
            "  This is a DIFFERENT enum on a DIFFERENT table from inspection_report.status = "
            "'RETURN_FOR_MODIFICATION' — do not use ir.status here.\n"
            "USE: SELECT u.first_name || ' ' || u.last_name AS inspector_name, ip.submitted_date\n"
            "FROM inspector_portfolio ip\n"
            "JOIN users u ON ip.user_id = u.id\n"
            "WHERE ip.status = 'MODIFICATION' AND ip.active = true AND u.active = true\n"
            "ORDER BY ip.submitted_date DESC;\n"
        ),
    },
    {
        "id":             "inject_overdue_quarter",
        "type":           INJECT,
        "priority":       70,
        "description":    "Overdue inspections this quarter — route to inspection_schedule.",
        "question_pattern": (
            r"\boverdue.{0,20}(this\s+quarter|quarter)\b"
            r"|\b(this\s+quarter|quarter).{0,20}overdue\b"
        ),
        "hint": (
            "\nUSE: SELECT COUNT(*) AS overdue_count FROM inspection_schedule\n"
            "WHERE status = 'OVERDUE'\n"
            "  AND due_date >= date_trunc('quarter', CURRENT_DATE);\n"
        ),
    },
    {
        "id":             "inject_raised_vs_closed_quarter",
        "type":           INJECT,
        "priority":       85,
        "description":    "Raised vs closed mitigative actions — must use FILTER, not GROUP BY.",
        "question_pattern": (
            r"\b(raised|created|opened).{0,30}(closed|completed|resolved)\b"
            r"|\b(closed|completed|resolved).{0,30}(raised|created|opened)\b"
            r"|\bhow\s+many.{0,30}mitigative.{0,20}(raised|closed|quarter)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SINGLE-ROW FORM — NOT GROUP BY status:\n"
            "SELECT COUNT(*) AS total_raised,\n"
            "       COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS total_closed,\n"
            "       COUNT(*) FILTER (WHERE status IN ('OPEN','OVERDUE')) AS still_open\n"
            "FROM inspection_corrective_action\n"
            "WHERE created_on >= date_trunc('quarter', NOW());\n"
            "CRITICAL: NEVER use GROUP BY status — that gives multiple rows. Return ONE row.\n"
        ),
    },
    {
        "id":             "inject_avg_inspection_duration",
        "type":           INJECT,
        "priority":       80,
        "description":    "Average inspection duration — EPOCH extraction with IS NOT NULL guard.",
        "question_pattern": (
            r"\b(average|avg|mean).{0,20}(duration|time|hours?|length)\b"
            r"|\b(duration|time|hours?|length).{0,20}(average|avg|inspection)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT PATTERN — start_date_time has NULLs, filter them out:\n"
            "SELECT ROUND(\n"
            "    AVG(EXTRACT(EPOCH FROM (ir.submitted_on - ir.start_date_time)) / 3600)::numeric,\n"
            "    2\n"
            ") AS avg_duration_hours\n"
            "FROM inspection_report ir\n"
            "WHERE ir.status != 'DRAFT'\n"
            "  AND ir.start_date_time IS NOT NULL;\n"
            "NEVER use: AVG(ir.submitted_on - ir.start_date_time) — returns interval not hours.\n"
        ),
    },
    {
        "id":             "inject_most_recent_form_answers",
        "type":           INJECT,
        "priority":       90,
        "description":    "Most recent inspection form Q&A — use aa subquery not ir subquery.",
        # Suppresses inject_most_recent_inspection when both patterns match, preventing
        # contradictory hints (one says 'use ir.submitted_on', the other says 'use aa subquery').
        "suppresses":     ["inject_most_recent_inspection"],
        "question_pattern": (
            # verb + answer/question/form + recency qualifier
            r"\b(list|show|get|fetch|display|what|give).{0,30}"
            r"(question|answer(?:s|ed)?|form).{0,30}"
            r"(most\s+recent(?:ly)?|latest|last)"
            # recency qualifier + answer/question/form (any order for the noun)
            r"|\b(most\s+recent(?:ly)?|latest|last).{0,30}"
            r"(question|answer(?:s|ed)?|form)"
            # answer/question/form + recency qualifier (handles "answers for the latest")
            r"|\b(answer(?:s|ed)?|question|form).{0,30}"
            r"(most\s+recent(?:ly)?|latest|last)\b"
        ),
        "hint": (
            "\nMOST RECENT FORM — always anchor on ai_answers.submitted_on, not ir.submitted_on.\n"
            "ai_answers is a complete flattened snapshot so every inspection has rows,\n"
            "but the submission timestamps in ai_answers may differ from ir.submitted_on.\n"
            "Use the aa subquery to guarantee you get the inspection with the latest FORM submission:\n"
            "\n-- Pattern A: user wants Q&A from the most recent form\n"
            "SELECT aq.label AS question, aa.answer_text AS answer, aa.module_name\n"
            "FROM ai_answers aa\n"
            "JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
            "WHERE aa.inspection_report_id = (\n"
            "    SELECT aa2.inspection_report_id FROM ai_answers aa2\n"
            "    ORDER BY aa2.submitted_on DESC LIMIT 1\n"
            ")\n"
            "ORDER BY aa.module_name, aq.label LIMIT 200;\n"
            "\n-- Pattern B: user wants inspection metadata for the most recently answered inspection\n"
            "SELECT ir.inspection_id, fac.name AS facility_name,\n"
            "       it.name AS inspection_type_name,\n"
            "       u.first_name || ' ' || u.last_name AS inspector_name,\n"
            "       ir.submitted_on, ir.status, ir.inspection_score\n"
            "FROM inspection_report ir\n"
            "JOIN facility fac ON ir.facility_id = fac.id\n"
            "JOIN inspection_type it ON ir.inspection_type_id = it.id\n"
            "JOIN users u ON ir.inspector_user_id = u.id\n"
            "WHERE ir.id = (\n"
            "    SELECT aa2.inspection_report_id FROM ai_answers aa2\n"
            "    ORDER BY aa2.submitted_on DESC LIMIT 1\n"
            ");\n"
            "\nNEVER: ORDER BY ir.submitted_on DESC LIMIT 1 — use aa subquery instead.\n"
            "NEVER: WHERE aa2.module_name = 'Inspection Form' in the inner subquery — "
            "if the actual module_name differs the subquery returns NULL and the whole query returns empty.\n"
            "Only filter by module_name in the OUTER WHERE using ILIKE if the user names a specific form.\n"
        ),
    },
    # Fault 1 — "completed/conducted/total" semantic mapping
    # hint comes from khidmah.core.semantics_registry — edit there, not here
    # SCOPE LIMIT (eval regression FREQ-05): excludes schedule-context questions —
    # "completed vs overdue in the SCHEDULE" is an inspection_schedule question with
    # different status enums (COMPLETED/OVERDUE/PENDING), handled by the rule below.
    # SCOPE LIMIT 2 (2026-06-12): also excludes overdue/pending/cancelled vocabulary
    # even when "schedule" isn't said literally — "how many inspections are overdue
    # vs pending vs completed" is schedule-status vocabulary (inspection_schedule's
    # own enum), not inspection_report's conducted/completed split.
    # inject_schedule_status (priority 76, below) still requires the literal word
    # "schedule" by design — it is NOT broadened here. If NEITHER rule fires for
    # such a question, the model falls back to the relevant worked example in
    # prompts.py — considered acceptable.
    {
        "id":             "inject_completed_inspections",
        "type":           INJECT,
        "priority":       75,
        "description":    "Total/completed/conducted inspections — canonical two-row category summary.",
        "question_pattern": (
            r"(?i)^(?!.*\bschedule)(?!.*\b(overdue|pending|cancelled)\b)"
            r"(?:.*(?:\b(completed|conducted|total|how\s+many|count)\s*(inspections?|inspection\s+report)\b"
            r"|\binspections?\s*(completed|conducted|total|count|in\s+\w+)\b))"
        ),
        "hint": _STATUS_HINT,
    },
    # "responsible party" phrasing — proactively prevent the users-join instinct
    {
        "id":             "inject_responsible_party_enum",
        "type":           INJECT,
        "priority":       79,
        "description":    "'responsible party' = ica.responsible enum, never a users join.",
        "question_pattern": (
            r"(?i)\bresponsible\s*(party|parties)?\b"
        ),
        "hint": (
            "\n'responsible' / 'responsible party' = ica.responsible directly.\n"
            "  It is a VARCHAR ENUM: CLIENT | INTERNAL_OPERATIONS | SUB_CONTRACTOR — a category, not a person.\n"
            "  SELECT ica.responsible AS responsible_party — NO JOIN to users table.\n"
            "  NEVER: LEFT JOIN users u ON ica.responsible = '...' AND ...u.id — this is not a valid join,\n"
            "  ica.responsible has no FK relationship to users at all.\n"
        ),
    },
    # MA approval-workflow questions — route to implementation_status, never capex_status
    {
        "id":             "inject_ma_approval_workflow",
        "type":           INJECT,
        "priority":       78,
        "description":    "MA approval/progress questions — implementation_status & progress_stage routing.",
        "question_pattern": (
            r"(?i)\b(awaiting|pending)\s*(for\s+)?(approval|review)\b.{0,40}\b(action|MA)s?\b"
            r"|\b(action|MA)s?\b.{0,40}\b(awaiting|pending)\s*(for\s+)?(approval|review)\b"
            r"|\bapproval\b.{0,30}\b(action|MA)s?\b"
            r"|\b(approved|sent\s+back|returned)\b.{0,30}\b(action|MA)s?\b"
            r"|\b(at\s+risk|behind\s+schedule|on\s+track|not\s+started)\b.{0,30}\b(action|MA)s?\b"
            r"|\b(action|MA)s?\b.{0,40}\b(at\s+risk|behind|on\s+track|approved?|pending|sent\s+back|returned)\b"
        ),
        "hint": (
            "\nMA WORKFLOW ROUTING — four independent ica columns, do not confuse:\n"
            "  'awaiting/pending approval/review' → implementation_status = 'PENDING_FOR_REVIEW'\n"
            "      (EXACT string — NOT 'PENDING'. NEVER capex_status — that is the financial track)\n"
            "  'approved'                         → implementation_status = 'APPROVE' (NOT 'APPROVED')\n"
            "  'sent back / returned' (MAs)       → ica.implementation_status = 'RETURN_FOR_MODIFICATION'\n"
            "  'returned' (INSPECTIONS)           → ir.status = 'RETURN_FOR_MODIFICATION' — different table!\n"
            "  'at risk / behind / on track'      → progress_stage (schedule health)\n"
            "      'at risk' = progress_stage='AT_RISK' — NOT risk_level='High' (severity)\n"
            "  'not started'                      → progress_stage = 'NOT_STARTED' OR progress_stage IS NULL\n"
            "  status='CLOSED' ⟺ implementation_status='APPROVE' — closure IS the approval\n"
        ),
    },
    # "Needs attention" dashboard questions — scalar subqueries, never UNION
    {
        "id":             "inject_needs_attention_dashboard",
        "type":           INJECT,
        "priority":       77,
        "description":    "Ambiguous dashboard questions — scalar subquery summary, never UNION ALL.",
        "question_pattern": (
            r"(?i)\b(needs?\s+attention|critical\s+issues?|urgent\s+items?"
            r"|what\s+should\s+i\s+(look|focus)|action\s+items?)\b"
        ),
        "hint": (
            "\nDASHBOARD QUESTION — return ONE ROW of scalar subqueries.\n"
            "NEVER UNION ALL heterogeneous tables (MA rows + schedule rows) — "
            "'N/A' placeholders clash with date/enum column types and fail at runtime.\n\n"
            "USE THIS PATTERN:\n"
            "SELECT\n"
            "    (SELECT COUNT(*) FROM inspection_corrective_action\n"
            "     WHERE status = 'OVERDUE' AND active = true)  AS overdue_mitigative_actions,\n"
            "    (SELECT COUNT(*) FROM inspection_corrective_action\n"
            "     WHERE status = 'OPEN' AND active = true)     AS open_mitigative_actions,\n"
            "    (SELECT COUNT(*) FROM inspection_schedule\n"
            "     WHERE status = 'OVERDUE')                    AS overdue_scheduled_inspections,\n"
            "    (SELECT COUNT(*) FROM inspection_corrective_action ica\n"
            "     JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "     WHERE rl.name = 'High' AND ica.status IN ('OPEN','OVERDUE')\n"
            "       AND ica.active = true)                     AS high_risk_open_cas;\n"
        ),
    },
    # Schedule-status questions — inspection_schedule has its OWN status enum
    {
        "id":             "inject_schedule_status",
        "type":           INJECT,
        "priority":       76,
        "description":    "Schedule status questions — inspection_schedule enum, not inspection_report.",
        "question_pattern": (
            r"(?i)\b(completed|overdue|pending|cancelled)\b.{0,50}\bschedule\b"
            r"|\bschedule\b.{0,50}\b(completed|overdue|pending|cancelled)\b"
        ),
        "hint": (
            "\nSCHEDULE STATUS — this is an inspection_schedule question, NOT inspection_report:\n"
            "  inspection_schedule.status values: PENDING | COMPLETED | OVERDUE | CANCELLED\n"
            "  (different enum from inspection_report — 'COMPLETED' here, 'CLOSED' there)\n"
            "  inspection_schedule DOES have an active column — always filter isched.active = true\n"
            "  Columns: active, schedule_date (NOT scheduled_date), due_date, schedule_id, status\n\n"
            "PATTERN for completed vs overdue in the schedule this year:\n"
            "SELECT isched.status, COUNT(*) AS count\n"
            "FROM inspection_schedule isched\n"
            "WHERE isched.status IN ('COMPLETED','OVERDUE')\n"
            "  AND isched.active = true\n"
            "  AND EXTRACT(YEAR FROM isched.due_date) = EXTRACT(YEAR FROM CURRENT_DATE)\n"
            "GROUP BY isched.status;\n"
        ),
    },
    # Fault 3/4/9 — Single-row comparison queries
    {
        "id":             "inject_this_vs_last_comparison",
        "type":           INJECT,
        "priority":       85,
        "description":    "This X vs last X comparison — single-row FILTER aggregate, not GROUP BY.",
        "question_pattern": (
            r"\b(this|current)\s+(month|quarter|year|week).{0,20}"
            r"(vs\.?|versus|compared?\s+to|against).{0,20}"
            r"(last|previous|prior)\s+(month|quarter|year|week)\b"
            r"|\b(last|previous|prior)\s+(month|quarter|year|week).{0,20}"
            r"(vs\.?|versus|compared?\s+to)\b"
        ),
        "hint": (
            "\nSINGLE-ROW COMPARISON — use FILTER aggregate, NOT GROUP BY:\n"
            "SELECT\n"
            "  COUNT(*) FILTER (WHERE date_trunc('month', col) = date_trunc('month', NOW())) AS this_month,\n"
            "  COUNT(*) FILTER (WHERE date_trunc('month', col) = date_trunc('month', NOW()-INTERVAL '1 month')) AS last_month\n"
            "FROM table WHERE ...;\n"
            "CRITICAL: Do NOT add GROUP BY or TO_CHAR. This returns EXACTLY ONE ROW.\n"
        ),
    },
    # ICA risk level — prevent UUID drift (ICA-02/03 regression fix)
    {
        "id":             "inject_ica_risk_level",
        "type":           INJECT,
        "priority":       95,
        "description":    "ICA risk level queries — ALWAYS JOIN risk_level table, never select UUID.",
        "question_pattern": (
            # Must mention mitigative action or risk breakdown in context of actions/counts
            r"\b(high|medium|low)\s*risk\s*(mitigative.?action|action|count|breakdown|all\s+mitigative)\b"
            r"|\b(mitigative.?action).{0,40}(risk|high.?risk|medium.?risk|low.?risk)\b"
            r"|\brisk.?level.{0,30}(mitigative|action|breakdown|all\s+mitigative)\b"
            r"|\b(how\s+many|count|show|list).{0,20}(high|medium|low)\s*risk\s*(mitigative|action)\b"
        ),
        "hint": (
            "\nICA RISK LEVEL — risk_level_id is a UUID FK, NEVER a string:\n"
            "  ALWAYS: JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "  ALWAYS: SELECT rl.name AS risk_level_name\n"
            "  ALWAYS: WHERE rl.name = 'High'  (for high-risk filter)\n"
            "  NEVER:  WHERE ica.risk_level_id = 'High'\n"
            "  NEVER:  SELECT ica.risk_level_id  ← outputs UUID garbage\n"
            "  NEVER:  GROUP BY ica.risk_level_id  ← UUID groups, not readable names\n"
            "\nCount by risk level (ICA-02 canonical form):\n"
            "SELECT rl.name AS risk_level_name, COUNT(*) AS count\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "GROUP BY rl.name ORDER BY count DESC;\n"
            "\nCount only High risk (ICA-03 canonical form):\n"
            "SELECT COUNT(*) AS high_risk_count\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "WHERE rl.name = 'High';\n"
        ),
    },
    # impact table columns — risk_level and impact are structurally IDENTICAL
    # (id, active, name, scale numeric, is_positive_impact bool, short_code) which
    # caused a 6-iteration, 2-error struggle on "highest impact scale rating"
    # (2026-07-08): neither table's scale/is_positive_impact column is exposed by
    # the normal lookup-table schema hint (which only ever shows .name), so the
    # model guessed plausible-sounding names (impact_scale, impact_scale_rating).
    {
        "id":             "inject_impact_scale",
        "type":           INJECT,
        "priority":       94,
        "description":    "Impact scale/rating/positive-impact questions — impact table has scale + is_positive_impact, NOT risk_level.",
        "required_tables": ["impact"],
        "question_pattern": (
            r"\bimpact.{0,20}(scale|rating|score)\b"
            r"|\b(scale|rating|score).{0,20}\bimpact\b"
            r"|\bpositive\s+impact\b|\bimpact.{0,15}positive\b"
        ),
        "hint": (
            "\nIMPACT SCALE/RATING — impact and risk_level are TWIN tables with the SAME columns "
            "(id, name, scale numeric, is_positive_impact bool, short_code) but are DIFFERENT lookup "
            "tables reached via DIFFERENT FKs — do not confuse them:\n"
            "  risk_level = severity classification, JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "  impact     = impact classification,   JOIN impact im ON ica.impact_id = im.id\n"
            "ALWAYS: JOIN impact im ON ica.impact_id = im.id\n"
            "ALWAYS: SELECT im.scale, im.is_positive_impact, im.name AS impact_name\n"
            "NEVER:  im.impact_scale / im.impact_scale_rating / im.rating — these column names don't exist\n"
            "\nHighest impact scale (canonical form):\n"
            "SELECT ica.corrective_action_id, im.name AS impact_name, im.scale\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN impact im ON ica.impact_id = im.id\n"
            "WHERE ica.active = true AND im.active = true\n"
            "ORDER BY im.scale DESC LIMIT 50;\n"
            "\nCount with positive impact (canonical form):\n"
            "SELECT COUNT(*) AS positive_impact_count\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN impact im ON ica.impact_id = im.id\n"
            "WHERE ica.active = true AND im.active = true AND im.is_positive_impact = true;\n"
        ),
    },
    # EDG-02 — "how many questions" routing disambiguation
    {
        "id":             "inject_count_questions_in_form",
        "type":           INJECT,
        "priority":       95,
        "description":    "Count questions in a form — route to COUNT(ai_questions), NOT list_forms.",
        "question_pattern": (
            r"\b(how\s+many|count|number\s+of).{0,25}"
            r"(question|field|item|section).{0,25}"
            r"(form|inspection|module)\b"
            r"|\b(form|inspection|module).{0,25}"
            r"(how\s+many|count|number\s+of).{0,25}"
            r"(question|field|item)\b"
        ),
        "hint": (
            "\nCOUNT QUESTIONS — use generate_sql with COUNT(*) on ai_questions, NOT list_forms.\n"
            "list_forms returns a summary table; it cannot answer 'how many questions' naturally.\n"
            "\nIf user names a specific form:\n"
            "SELECT COUNT(*) AS question_count\n"
            "FROM ai_questions\n"
            "WHERE module_name ILIKE '%<form_name>%';\n"
            "\nIf user did not name a form, list all forms and their counts:\n"
            "SELECT module_name, COUNT(*) AS question_count\n"
            "FROM ai_questions\n"
            "WHERE module_name IS NOT NULL AND module_name != ''\n"
            "GROUP BY module_name\n"
            "ORDER BY question_count DESC;\n"
            "NEVER call list_forms then generate_sql separately — one generate_sql call is enough.\n"
        ),
    },
    # Join registry — schedule/cycle/frequency tables
    # These tables are visible in the schema but their join paths are non-obvious.
    # Without explicit guidance deepseek will hallucinate joins or miss the chain.
    {
        "id":             "inject_inspection_schedule_joins",
        "type":           INJECT,
        "priority":       80,
        "description":    "Schedule/cycle/frequency queries — correct join chain and column names.",
        "question_pattern": (
            r"\b(schedule[ds]?|due\s+date|upcoming|overdue\s+inspection)\b"
            r"|\b(frequency|cycle|recurring|period)\s+(of\s+)?(inspection|visit|audit)\b"
            r"|\b(inspection|visit|audit).{0,30}(frequency|cycle|recurring|period)\b"
            r"|\bhow\s+(often|frequent)\b"
        ),
        "hint": (
            "\nINSPECTION CYCLE / FREQUENCY — EXACT column names (errors occur with wrong names):\n"
            "  inspection_schedule.inspection_cycle_id  → inspection_cycle.id\n"
            "  inspection_schedule.portfolio_details_id → inspector_portfolio_details.id\n"
            "    NEVER: isched.portfolio_id or isched.inspection_schedule_id (don't exist)\n"
            "  inspector_portfolio_details.frequency_definition_id → frequency_definition.id\n"
            "  inspector_portfolio_details.portfolio_id → inspector_portfolio.id\n"
            "    NEVER: ipd.cycle_id (does not exist — cycle goes through inspector_portfolio)\n"
            "  inspector_portfolio.cycle_id → inspection_cycle.id\n"
            "\nfrequency_definition columns: id, name, repeat_count, repeat_interval,\n"
            "  repeat_unit (DAY/WEEK/MONTH)\n"
            "  'Daily'=repeat_count=1,repeat_interval=1,DAY\n"
            "  'Alternate Days'=repeat_count=1,repeat_interval=2,DAY\n"
            "\nFor frequency breakdown of a specific cycle (use GROUP BY, not flat join):\n"
            "SELECT fd.name AS frequency_name, fd.repeat_count, fd.repeat_interval,\n"
            "       fd.repeat_unit, COUNT(*) AS schedule_count\n"
            "FROM inspection_schedule isched\n"
            "JOIN inspector_portfolio_details ipd ON isched.portfolio_details_id = ipd.id\n"
            "JOIN frequency_definition fd ON ipd.frequency_definition_id = fd.id\n"
            "WHERE isched.inspection_cycle_id = '<cycle_id>'\n"
            "GROUP BY fd.name, fd.repeat_count, fd.repeat_interval, fd.repeat_unit\n"
            "ORDER BY schedule_count DESC;\n"
            "\nFor overdue schedule count:\n"
            "SELECT COUNT(*) AS overdue_count FROM inspection_schedule\n"
            "WHERE status = 'OVERDUE'\n"
            "  AND due_date >= date_trunc('quarter', NOW())\n"
            "  AND due_date < date_trunc('quarter', NOW()) + INTERVAL '3 months';\n"
        ),
    },
    # Scores across inspection type categories — group ALL types, never filter by name
    {
        "id":             "inject_inspection_scores_by_type",
        "type":           INJECT,
        "priority":       88,
        "description":    "Inspection scores across named categories — group all inspection types without name-filter.",
        "question_pattern": (
            # "inspection scores across safety, hygiene, ..."
            r"(?i)\binspection\s+score\w*\b.{0,60}"
            r"\b(across|by|per|for\s+each|breakdown|compare|versus|vs)\b"
            # "scores across / by <anything> / breakdown"
            r"|\bscore\w*\b.{0,30}\b(across|by\s+\w+|breakdown)\b"
            # two or more inspection-domain keywords in the same sentence with a score word
            r"|\b(safety|hygiene|structure|compliance|documentation|cleanliness|fire"
            r"|electrical|mechanical|health)\b.{0,100}"
            r"\b(safety|hygiene|structure|compliance|documentation|cleanliness|fire"
            r"|electrical|mechanical|health)\b.{0,60}"
            r"\b(score\w*|rating\w*|inspection)\b"
            r"|\b(score\w*|rating\w*)\b.{0,40}"
            r"\b(safety|hygiene|structure|compliance|documentation|cleanliness|fire"
            r"|electrical|mechanical|health)\b.{0,80}"
            r"\b(safety|hygiene|structure|compliance|documentation|cleanliness|fire"
            r"|electrical|mechanical|health)\b"
        ),
        "hint": (
            "\nINSPECTION SCORES BY TYPE — CRITICAL RULE:\n"
            "NEVER filter WHERE it.name IN (...) or WHERE it.name ILIKE '%safety%' — "
            "those names may not match the DB and return ZERO rows.\n"
            "The user's category words (safety, hygiene, structure, compliance, documentation) "
            "are INTENT labels, NOT exact strings stored in inspection_type.name.\n"
            "\nALWAYS group ALL inspection types without a name filter:\n"
            "SELECT it.name AS inspection_type,\n"
            "       ROUND(AVG(ir.inspection_score)::numeric, 1) AS avg_score\n"
            "FROM inspection_report ir\n"
            "JOIN inspection_type it ON ir.inspection_type_id = it.id\n"
            "WHERE ir.active = true AND ir.status != 'DRAFT' AND it.active = true\n"
            "GROUP BY it.name\n"
            "ORDER BY avg_score DESC;\n"
            "Return ONLY these 2 columns (inspection_type, avg_score) — no count column.\n"
            "This gives exactly 1 categorical + 1 score column so the chart advisor auto-selects radar.\n"
        ),
    },
    # P&L summary — single-row waterfall trigger
    {
        "id":          "inject_pl_summary_waterfall",
        "type":        INJECT,
        "priority":    89,
        "description": "Overall P&L / income-cost-profit summary — force 3-column single-row pattern for waterfall.",
        "question_pattern": (
            r"(?i)\b(total|overall|summary)\b.{0,30}\b(income|revenue)\b"
            r"|\b(income|revenue).{0,30}\b(cost|expense|expenditure)\b"
            r"|\b(profit|net\s+profit|p&l|profit\s+and\s+loss|profit\s+loss)\b.{0,40}"
            r"\b(total|overall|summary|income|revenue|cost)\b"
            r"|\b(income|revenue).{0,20}\b(cost|expense).{0,20}\b(profit|net)\b"
            r"|\bfinancial\s+summary\b"
        ),
        "hint": (
            "\nP&L SUMMARY — WATERFALL PATTERN (single row, 3 exact column aliases):\n"
            "ALWAYS return exactly 1 row with these 3 column names — no other columns:\n"
            "  total_income  → SUM of po_details.po_amount  (client paid Khidmah)\n"
            "  total_cost    → SUM of pr_details.pr_amount  (Khidmah paid out)\n"
            "  net_profit    → total_income - total_cost\n"
            "NEVER alias as: po_total, pr_total, revenue, expenses, balance — use EXACT names above.\n"
            "NEVER GROUP BY anything — one global row only.\n"
            "NEVER add a category/label column — only the 3 numeric columns.\n"
            "\nCORRECT SQL:\n"
            "WITH po_agg AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_income\n"
            "    FROM po_details WHERE active = true GROUP BY 1\n"
            "),\n"
            "pr_agg AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_cost\n"
            "    FROM pr_details WHERE active = true GROUP BY 1\n"
            ")\n"
            "SELECT\n"
            "    COALESCE(SUM(po.total_income), 0)                                   AS total_income,\n"
            "    COALESCE(SUM(pr.total_cost),  0)                                    AS total_cost,\n"
            "    COALESCE(SUM(po.total_income), 0) - COALESCE(SUM(pr.total_cost), 0) AS net_profit\n"
            "FROM inspection_corrective_action ica\n"
            "LEFT JOIN po_agg po ON po.ica_id = ica.id\n"
            "LEFT JOIN pr_agg pr ON pr.ica_id = ica.id\n"
            "WHERE ica.active = true;\n"
        ),
    },
]


# ── RulesEngine ────────────────────────────────────────────────────────────────

class RulesEngine:
    """
    Loads and applies the business rules registry.

    Usage
    -----
    engine = RulesEngine()                            # validates registry at init
    hint    = engine.hint_for_question(question)       # INJECT rules
    ids     = engine.matching_inject_ids(question)
    tables  = engine.required_tables_for(rule_ids)
    forbidden = engine.forbidden_patterns_for(rule_ids)

    BLOCK/WARN/AUTOCORRECT rules are consumed by core/validator.py's
    SQLValidator (_registry_block_warn_issues / _registry_autocorrects),
    restricted to an explicit allowlist — not through this class directly.
    """

    def __init__(self, rules: list[dict] | None = None):
        self._rules = rules if rules is not None else REGISTRY
        self._compiled: list[dict] = []
        self._compile_and_validate()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _compile_and_validate(self) -> None:
        ids_seen: set[str] = set()
        sql_block_patterns:  dict[str, str] = {}   # pattern → id
        sql_autocorr_patterns: dict[str, str] = {} # pattern → id
        inject_q_patterns:   dict[str, str] = {}   # pattern → id
        conflicts: list[str] = []

        for rule in self._rules:
            # Required fields
            rid  = rule["id"]
            rtype = rule["type"]

            # Duplicate ID check
            if rid in ids_seen:
                conflicts.append(f"Duplicate rule ID: '{rid}'")
            ids_seen.add(rid)

            # Compile pattern
            compiled = dict(rule)
            if "sql_pattern" in rule:
                compiled["_sql_re"] = re.compile(rule["sql_pattern"], re.IGNORECASE | re.DOTALL)
            if "question_pattern" in rule:
                compiled["_q_re"] = re.compile(rule["question_pattern"], re.IGNORECASE)

            # Conflict: same sql_pattern in BLOCK and AUTOCORRECT
            if rtype == BLOCK and "sql_pattern" in rule:
                p = rule["sql_pattern"]
                if p in sql_autocorr_patterns:
                    conflicts.append(
                        f"Conflict: BLOCK rule '{rid}' and AUTOCORRECT rule "
                        f"'{sql_autocorr_patterns[p]}' share sql_pattern '{p[:40]}...'"
                    )
                sql_block_patterns[p] = rid

            if rtype == AUTOCORRECT and "sql_pattern" in rule:
                p = rule["sql_pattern"]
                if p in sql_block_patterns:
                    conflicts.append(
                        f"Conflict: AUTOCORRECT rule '{rid}' and BLOCK rule "
                        f"'{sql_block_patterns[p]}' share sql_pattern '{p[:40]}...'"
                    )
                sql_autocorr_patterns[p] = rid

            # Conflict: identical INJECT question_patterns
            if rtype == INJECT and "question_pattern" in rule:
                p = rule["question_pattern"]
                if p in inject_q_patterns:
                    conflicts.append(
                        f"Conflict: INJECT rules '{rid}' and '{inject_q_patterns[p]}' "
                        f"share identical question_pattern"
                    )
                inject_q_patterns[p] = rid

            self._compiled.append(compiled)

        if conflicts:
            raise ValueError(
                "Business rules registry conflicts detected:\n  " +
                "\n  ".join(conflicts)
            )

        # Sort INJECT rules by priority descending so highest-priority hints inject first
        self._inject_rules = sorted(
            [r for r in self._compiled if r["type"] == INJECT],
            key=lambda r: r.get("priority", 0),
            reverse=True,
        )
        self._block_rules  = [r for r in self._compiled if r["type"] == BLOCK]
        self._warn_rules   = [r for r in self._compiled if r["type"] == WARN]
        self._autocorr_rules = [r for r in self._compiled if r["type"] == AUTOCORRECT]

    # ── Question-level hint injection ─────────────────────────────────────────

    def hint_for_question(self, question: str) -> str:
        """
        Return combined schema_hint for all INJECT rules matching the question.
        Rules fire in priority order (highest first) and hints are concatenated.

        If a matching rule declares ``"suppresses": ["rule_id", ...]``, those
        lower-priority rules are silenced even if their own pattern matches.
        This prevents contradictory hints from being concatenated.
        """
        suppressed: set[str] = set()
        parts: list[str] = []
        for rule in self._inject_rules:
            rid = rule["id"]
            if rid in suppressed:
                continue
            if rule["_q_re"].search(question):
                parts.append(rule["hint"])
                for sid in rule.get("suppresses", []):
                    suppressed.add(sid)
        return "".join(parts)

    def matching_inject_ids(self, question: str) -> list[str]:
        """Return IDs of all INJECT rules that match the question (for debugging)."""
        return [r["id"] for r in self._inject_rules if r["_q_re"].search(question)]

    # ── Introspection ─────────────────────────────────────────────────────────

    def rules_by_type(self, rule_type: str) -> list[dict]:
        return [r for r in self._rules if r["type"] == rule_type]

    def required_tables_for(self, rule_ids: list) -> list:
        """Return all required_tables for a set of fired INJECT rule IDs."""
        tables = []
        for rule in self._rules:
            if rule.get("id") in rule_ids and rule.get("type") == INJECT:
                tables.extend(rule.get("required_tables", []))
        return list(set(tables))

    def forbidden_patterns_for(self, rule_ids: list) -> list:
        """Return all forbidden_patterns for a set of fired INJECT rule IDs."""
        patterns = []
        for rule in self._rules:
            if rule.get("id") in rule_ids and rule.get("type") == INJECT:
                patterns.extend(rule.get("forbidden_patterns", []))
        return patterns

    def summary(self) -> str:
        counts = {BLOCK: 0, WARN: 0, AUTOCORRECT: 0, INJECT: 0}
        for r in self._rules:
            counts[r["type"]] += 1
        return (
            f"RulesEngine: {len(self._rules)} rules — "
            + ", ".join(f"{v} {k}" for k, v in counts.items())
        )


# ── Singleton for import convenience ─────────────────────────────────────────
# Validated at import time — if the registry has conflicts this raises
# immediately rather than at first query.

_default_engine: RulesEngine | None = None


def get_engine() -> RulesEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = RulesEngine()
    return _default_engine


if __name__ == "__main__":
    engine = RulesEngine()
    print(engine.summary())
    print("\nBLOCK rules:")
    for r in engine.rules_by_type(BLOCK):
        print(f"  {r['id']}: {r['description']}")
    print("\nINJECT rules (by priority):")
    for r in engine._inject_rules:
        print(f"  [{r.get('priority',0)}] {r['id']}: {r['description']}")