"""
BLOCK_RULES: extracted verbatim from the original
core/business_rules.py REGISTRY list (all "type": BLOCK entries,
in original relative order). See core.business_rules.engine for how
these are aggregated and consumed.
"""

from src.sql.rules._types import BLOCK

BLOCK_RULES: list[dict] = [

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
        "code":        "MA_OBSERVATION_WRONG_PATH",
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

    {
        "id":          "block_degenerate_label_groupby",
        "type":        BLOCK,
        "code":        "DEGENERATE_LABEL_GROUPBY",
        "description": (
            "GROUP BY aq.label is meaningless when WHERE already pins aq.label to one "
            "literal value — collapses every row into a single bucket instead of grouping "
            "by the actual answer content (e.g. which specific observation recurred)."
        ),
        # Caught live 2026-07-29 (audit domain, same shared ai_questions/aq alias):
        # "which observation triggered the most MAs/CAs" generated
        # WHERE aq.label = 'X' ... GROUP BY aq.label — every matching row already has
        # aq.label = 'X', so GROUP BY aq.label produces exactly one group; COUNT(*)
        # just counts total matching rows, not "which observation" at all.
        "sql_pattern": (
            r"(?=.*\baq\.label\s*=\s*'[^']+')"
            r"(?=.*\bgroup\s+by\b[^;]*\baq\.label\b)"
        ),
        "message": (
            "WRONG GROUP BY: aq.label is already pinned to one value by the WHERE filter, "
            "so 'GROUP BY aq.label' collapses everything into a single meaningless group. "
            "Group by the actual answer instead, e.g.: "
            "GROUP BY aa.id, aa.answer_text, ir.inspection_id "
            "(or aaa.id, aaa.answer_text, arep.audit_id for audit)."
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
        "message":     "WRONG COLUMN: ica.impact does not exist. JOIN impact im ON ica.impact_id = im.id WHERE im.name ILIKE '%Financial%'",
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
        "message":     "WRONG COMPARISON: impact_id is a UUID FK. JOIN impact im ON ica.impact_id = im.id WHERE im.name ILIKE '%Financial%'",
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

    # ── review approval workflow — status='COMPLETED' != approved ───────────
    # Mirrors audit_rules.py's audit_block_review_completed_not_approved —
    # 'review' is the SAME shared polymorphic table in both domains, so the
    # same trap applies: a completed review can be a rejection
    # (action = 'RETURN_FOR_MODIFICATION'), not an approval. See
    # schema_introspector.py's BUSINESS_RULES for the confirmed-live numbers.
    {
        "id":          "inspection_block_review_completed_not_approved",
        "type":        BLOCK,
        "code":        "REVIEW_COMPLETED_NOT_APPROVED",
        "description": (
            "review.status = 'COMPLETED' does NOT mean approved — a completed review can "
            "have action = 'RETURN_FOR_MODIFICATION' (rejected/sent back) instead of "
            "'APPROVE'. 'who/which approved' questions must filter "
            "status = 'COMPLETED' AND action = 'APPROVE', not status alone."
        ),
        # MUST be anchored with ^ — see audit_rules.py's identical rule for why:
        # without it, re.search() can find a starting position PAST an earlier
        # exclude-word (e.g. "rejections") where the negative lookahead no longer
        # sees it while a later approve-word is still ahead, producing a false match.
        "question_pattern": (
            r"(?i)^(?=.*\bapprov\w*\b)"
            r"(?!.*\b(reject\w*|return\w*|declin\w*|pending)\b)"
        ),
        "sql_pattern": (
            r"(?=.*\b(?:FROM|JOIN)\s+review\b)"
            r"(?=.*\bstatus\s*=\s*'COMPLETED')"
            r"(?!.*\baction\s*=\s*'APPROVE')"
        ),
        "message": (
            "BLOCKED: review.status = 'COMPLETED' includes REJECTED reviews "
            "(action = 'RETURN_FOR_MODIFICATION'), not just approved ones. For "
            "'who/which approved' questions add AND action = 'APPROVE' alongside "
            "status = 'COMPLETED'."
        ),
        "hard_fail": True,
    },

]
