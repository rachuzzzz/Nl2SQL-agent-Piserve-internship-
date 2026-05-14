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
from dataclasses import dataclass, field
from typing import Optional


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

    # ── DML / DDL ────────────────────────────────────────────────────────────
    {
        "id":          "block_dml",
        "type":        BLOCK,
        "description": "Only SELECT is allowed — no data modification.",
        "sql_pattern": r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE)\b",
        "message":     "BLOCKED: DML/DDL detected. Only SELECT is allowed.",
    },

    # ── Hallucinated tables ───────────────────────────────────────────────────
    {
        "id":          "block_table_ai_question_singular",
        "type":        BLOCK,
        "description": "ai_question (singular) does not exist — use ai_questions.",
        "sql_pattern": r"\bai_question\b(?!s)",
        "message":     "HALLUCINATION: table 'ai_question' does not exist. Use ai_questions (plural).",
    },
    {
        "id":          "block_table_ai_answer_singular",
        "type":        BLOCK,
        "description": "ai_answer (singular) does not exist — use ai_answers.",
        "sql_pattern": r"\bai_answer\b(?!s)",
        "message":     "HALLUCINATION: table 'ai_answer' does not exist. Use ai_answers (plural).",
    },
    {
        "id":          "block_table_fb_users",
        "type":        BLOCK,
        "description": "fb_users does not exist — use users.",
        "sql_pattern": r"\bfb_users?\b",
        "message":     "HALLUCINATION: table 'fb_users' does not exist. Use the 'users' table.",
    },
    {
        "id":          "block_table_fb_submissions",
        "type":        BLOCK,
        "description": "fb_submissions / fb_submission do not exist.",
        "sql_pattern": r"\bfb_submissions?\b",
        "message":     "HALLUCINATION: fb_submissions does not exist. Answer data is in ai_answers.",
    },
    {
        "id":          "block_table_fb_answers",
        "type":        BLOCK,
        "description": "fb_answers / fb_answer / fb_form_answers do not exist.",
        "sql_pattern": r"\bfb_(form_)?answers?\b",
        "message":     "HALLUCINATION: fb_answers does not exist. Use ai_answers.",
    },
    {
        "id":          "block_table_fb_responses",
        "type":        BLOCK,
        "description": "fb_responses / fb_response do not exist.",
        "sql_pattern": r"\bfb_responses?\b",
        "message":     "HALLUCINATION: fb_responses does not exist. Use ai_answers.",
    },
    {
        "id":          "block_table_dynamic_fb",
        "type":        BLOCK,
        "description": "Dynamic fb_{uuid} answer tables — all answer data is in ai_answers.",
        "sql_pattern": r"\bfb_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}\b",
        "message":     "BLOCKED: Dynamic fb_<uuid> table. Use ai_answers instead.",
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
        "sql_pattern": r"\bselect\b.*?\bir\.id\b.*?\bfrom\b",
        "table_guard": "inspection_report",
        "message":     "WRONG COLUMN: ir.id is the UUID primary key — outputs UUID garbage. Use ir.inspection_id (varchar like '2026/04/ST001/INS001') for display. Note: ir.id in JOIN conditions is correct.",
    },

    # ── Hallucinated columns on inspection_corrective_action ─────────────────
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
    {
        "id":          "block_col_ica_capex_status",
        "type":        BLOCK,
        "description": "capex_status does not exist on inspection_corrective_action.",
        "sql_pattern": r"\b(ica|ics?|ca|inspection_corrective_action)\.capex_status\b",
        "message":     "WRONG COLUMN: capex_status does not exist on inspection_corrective_action. For risk filtering: JOIN risk_level rl ON ica.risk_level_id = rl.id WHERE rl.name = 'High'",
    },
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
        "description": "INTERVAL '30 days' is not 'this month' — it's a rolling window.",
        "sql_pattern": r"INTERVAL\s*'30\s*days?'",
        "message":     (
            "BLOCKED: INTERVAL '30 days' is a rolling 30-day window, NOT 'this month'. "
            "For 'this month' use: date_trunc('month', col) = date_trunc('month', NOW()). "
            "Only use INTERVAL '30 days' if the question literally says 'last 30 days' or 'past 30 days'."
        ),
    },
    {
        "id":          "block_responsible_fk_join",
        "type":        BLOCK,
        "description": "ica.responsible is a string ENUM — never JOIN it to users.",
        "sql_pattern": r"\bjoin\s+users\b.*?\b(?:responsible|pending_with)\s*=\s*\w+\.id\b",
        "message":     "HALLUCINATION: responsible and pending_with are plain string ENUMs, NOT UUID FKs. Never JOIN them to users. Use: WHERE ica.responsible = 'CLIENT' (uppercase).",
    },

    # ── close_on arithmetic ───────────────────────────────────────────────────
    {
        "id":          "block_close_on_arithmetic",
        "type":        BLOCK,
        "description": "close_on is NULL on most rows — arithmetic on it is unreliable.",
        "sql_pattern": r"\bclose_on\s*[-+]\s*\w|\bextract\s*\(.*?\bclose_on\b",
        "message":     "BLOCKED: Arithmetic on close_on is unreliable — NULL on most rows. Use status-based counting: COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED')).",
    },

    # ── Hardcoded year ────────────────────────────────────────────────────────
    {
        "id":          "block_hardcoded_year",
        "type":        BLOCK,
        "description": "Never hardcode years — use EXTRACT(YEAR FROM CURRENT_DATE).",
        "sql_pattern": r"\bextract\s*\(\s*year\s+from\s+\w[\w.]*\s*\)\s*=\s*20\d\d\b",
        "message":     "HARDCODED YEAR: Never use '= 20XX'. Always use: EXTRACT(YEAR FROM submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # WARN — soft warnings (SQL executes but flag is surfaced)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id":          "warn_no_limit",
        "type":        WARN,
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

    # ══════════════════════════════════════════════════════════════════════════
    # INJECT — append hint to schema_hint when question matches
    # Higher priority fires first; hints are concatenated in priority order.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id":             "inject_pct_with_ca",
        "type":           INJECT,
        "priority":       100,
        "description":    "Percentage of inspections with corrective action — inject exact SQL.",
        "question_pattern": (
            r"\b(percentage|percent|pct|proportion|ratio).{0,40}"
            r"(corrective.?action|mitigative|finding)\b"
            r"|\b(corrective.?action|mitigative|finding).{0,40}"
            r"(percentage|percent|pct|proportion|ratio)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SQL — do not generate your own:\n"
            "SELECT ROUND(\n"
            "  100.0 * COUNT(DISTINCT CASE WHEN ica.id IS NOT NULL THEN ir.id END)\n"
            "  / NULLIF(COUNT(DISTINCT ir.id), 0), 1\n"
            ") AS pct_with_corrective_action\n"
            "FROM inspection_report ir\n"
            "LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id\n"
            "WHERE ir.status != 'DRAFT';\n"
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
            "SELECT u.first_name || ' ' || u.last_name AS inspector_name,\n"
            "       COUNT(*) AS num_inspections\n"
            "FROM inspection_report ir\n"
            "JOIN users u ON ir.inspector_user_id = u.id\n"
            "WHERE ir.status != 'DRAFT'\n"
            "  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)\n"
            "GROUP BY u.first_name, u.last_name\n"
            "ORDER BY num_inspections DESC LIMIT 1;\n"
            "CRITICAL: NEVER write '= 2023' or '= 2026' — always use EXTRACT(YEAR FROM CURRENT_DATE).\n"
        ),
    },
    {
        "id":             "inject_ica_join",
        "type":           INJECT,
        "priority":       80,
        "description":    "ICA + facility/grouping — prevent varchar=UUID join mismatch.",
        "question_pattern": (
            r"\b(corrective.?action|capex|opex).{0,50}"
            r"(facility|client|inspector|type|count|by|group|breakdown)\b"
            r"|\b(facility|client|inspector|type|breakdown).{0,50}"
            r"(corrective.?action|capex|opex)\b"
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
        "description":    "Raised vs closed corrective actions — must use FILTER, not GROUP BY.",
        "question_pattern": (
            r"\b(raised|created|opened).{0,30}(closed|completed|resolved)\b"
            r"|\b(closed|completed|resolved).{0,30}(raised|created|opened)\b"
            r"|\bhow\s+many.{0,30}corrective.{0,20}(raised|closed|quarter)\b"
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
        "question_pattern": (
            r"\b(list|show|get|fetch|display).{0,30}"
            r"(question|answer|form).{0,30}"
            r"(most\s+recent|latest|last)\b"
            r"|\b(most\s+recent|latest|last).{0,30}"
            r"(question|answer|form)\b"
        ),
        "hint": (
            "\nCRITICAL: The most recent inspection_report row may NOT have ai_answers yet.\n"
            "Use the aa subquery to find the most recent inspection WITH Inspection Form answers:\n"
            "SELECT aq.label AS question, aa.answer_text AS answer\n"
            "FROM ai_answers aa\n"
            "JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
            "WHERE aa.inspection_report_id = (\n"
            "    SELECT aa2.inspection_report_id\n"
            "    FROM ai_answers aa2\n"
            "    WHERE aa2.module_name = 'Inspection Form'\n"
            "    ORDER BY aa2.submitted_on DESC LIMIT 1\n"
            ")\n"
            "  AND aa.module_name = 'Inspection Form'\n"
            "ORDER BY aq.label LIMIT 100;\n"
            "NEVER: (SELECT id FROM inspection_report WHERE status != 'DRAFT' "
            "ORDER BY submitted_on DESC LIMIT 1) — that row may have no form answers.\n"
        ),
    },
    # Fault 1 — "completed" semantic mapping
    {
        "id":             "inject_completed_inspections",
        "type":           INJECT,
        "priority":       75,
        "description":    "Completed inspections — canonical mapping to CLOSED and SUBMITTED.",
        "question_pattern": (
            r"\b(completed|finished|done|finalized)\s+(inspection|inspections?)\b"
            r"|\binspections?\s+(completed|finished|done)\b"
            r"|\bhow\s+many\s+inspections?.{0,20}(completed|finished)\b"
        ),
        "hint": (
            "\nSEMANTIC MAPPING: 'completed inspection' = status IN ('CLOSED', 'SUBMITTED').\n"
            "  SUBMITTED = inspector filed and submitted — counts as completed.\n"
            "  CLOSED = fully reviewed and approved — counts as completed.\n"
            "  UNDER_REVIEW = still being reviewed — NOT completed.\n"
            "  RETURN_FOR_MODIFICATION = sent back — NOT completed.\n"
            "  DRAFT = not yet submitted — NOT completed.\n"
            "USE: WHERE ir.status IN ('CLOSED', 'SUBMITTED')\n"
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
            # Must mention corrective action, ICA, or risk breakdown in context of actions/counts
            r"\b(high|medium|low)\s*risk\s*(corrective.?action|action|count|breakdown|all\s+corrective)\b"
            r"|\b(corrective.?action|ica).{0,40}(risk|high.?risk|medium.?risk|low.?risk)\b"
            r"|\brisk.?level.{0,30}(corrective|action|breakdown|all\s+corrective)\b"
            r"|\b(how\s+many|count|show|list).{0,20}(high|medium|low)\s*risk\s*(corrective|action)\b"
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
]


# ── RulesEngine ────────────────────────────────────────────────────────────────

@dataclass
class RuleMatch:
    rule_id:  str
    rule_type: str
    message:  str
    fix:      Optional[str] = None


class RulesEngine:
    """
    Loads and applies the business rules registry.

    Usage
    -----
    engine = RulesEngine()                       # validates registry at init
    errors  = engine.validate_sql(sql)           # BLOCK + WARN rules
    sql     = engine.autocorrect_sql(sql)        # AUTOCORRECT rules
    hint    = engine.hint_for_question(question) # INJECT rules
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

    # ── SQL validation ────────────────────────────────────────────────────────

    def validate_sql(self, sql: str) -> tuple[bool, list[str]]:
        """
        Apply BLOCK and WARN rules to the given SQL.
        Returns (passed, [error/warning strings]).
        Real errors (BLOCK) cause passed=False.
        """
        errors: list[str] = []
        sql_lower = sql.lower()

        for rule in self._block_rules:
            pattern_re: re.Pattern = rule["_sql_re"]
            # Optional table_guard: rule only fires if a specific table is present
            guard = rule.get("table_guard")
            if guard and guard.lower() not in sql_lower:
                continue
            if pattern_re.search(sql):
                errors.append(rule["message"])

        warnings: list[str] = []
        for rule in self._warn_rules:
            pattern_re = rule["_sql_re"]
            guard = rule.get("table_guard")
            if guard and guard.lower() not in sql_lower:
                continue
            if pattern_re.search(sql):
                warnings.append(rule["message"])

        all_messages = errors + warnings
        return len(errors) == 0, all_messages

    # ── SQL autocorrection ────────────────────────────────────────────────────

    def autocorrect_sql(self, sql: str) -> tuple[str, list[str]]:
        """
        Apply AUTOCORRECT rules to the given SQL.
        Returns (corrected_sql, [log_messages]).
        """
        logs: list[str] = []
        for rule in self._autocorr_rules:
            pattern_re: re.Pattern = rule["_sql_re"]
            replacement = rule.get("replacement", "")
            if pattern_re.search(sql):
                sql = pattern_re.sub(replacement, sql)
                logs.append(rule.get("message", f"[autocorrect] applied rule '{rule['id']}'"))
        return sql, logs

    # ── Question-level hint injection ─────────────────────────────────────────

    def hint_for_question(self, question: str) -> str:
        """
        Return combined schema_hint for all INJECT rules matching the question.
        Rules fire in priority order (highest first) and hints are concatenated.
        """
        parts: list[str] = []
        for rule in self._inject_rules:
            if rule["_q_re"].search(question):
                parts.append(rule["hint"])
        return "".join(parts)

    def matching_inject_ids(self, question: str) -> list[str]:
        """Return IDs of all INJECT rules that match the question (for debugging)."""
        return [r["id"] for r in self._inject_rules if r["_q_re"].search(question)]

    # ── Introspection ─────────────────────────────────────────────────────────

    def rules_by_type(self, rule_type: str) -> list[dict]:
        return [r for r in self._rules if r["type"] == rule_type]

    def get_rule(self, rule_id: str) -> dict | None:
        for r in self._rules:
            if r["id"] == rule_id:
                return r
        return None

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