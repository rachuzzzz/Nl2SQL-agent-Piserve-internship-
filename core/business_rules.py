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

    # ── responsible is a plain ENUM, not a FK ────────────────────────────────
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
        "id":          "autocorrect_close_on_quarter_scope",
        "type":        AUTOCORRECT,
        "description": (
            "close_on is NULL for all OPEN/OVERDUE rows. When deepseek uses "
            "date_trunc('quarter', close_on) or EXTRACT(QUARTER FROM close_on) "
            "to scope 'this quarter's actions', replace with created_on which "
            "is always populated."
        ),
        "sql_pattern": r"date_trunc\s*\(\s*'quarter'\s*,\s*(?:ica\.)?close_on\s*\)",
        "replacement": "date_trunc('quarter', created_on)",
        "message":     "[autocorrect] close_on→created_on for quarter scope (close_on NULL on OPEN rows)",
    },
    {
        "id":          "autocorrect_extract_quarter_close_on",
        "type":        AUTOCORRECT,
        "description": "EXTRACT(QUARTER FROM close_on) → EXTRACT(QUARTER FROM created_on)",
        "sql_pattern": r"EXTRACT\s*\(\s*(?:QUARTER|quarter)\s+FROM\s+(?:ica\.)?close_on\s*\)",
        "replacement": "EXTRACT(QUARTER FROM created_on)",
        "message":     "[autocorrect] EXTRACT(QUARTER FROM close_on)→created_on",
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
        "id":             "inject_ica_risk_level",
        "type":           INJECT,
        "priority":       85,
        "description":    "ICA risk/impact level queries — must use JOIN lookup, not bare column or string compare.",
        "question_pattern": (
            r"\b(high|medium|low|no\s+active)\s+risk\b"
            r"|\brisk\s+(level|breakdown|distribution|count|ranking|group)\b"
            r"|\bby\s+risk\s+level\b"
            r"|\bhigh.{0,20}(corrective|action|finding)\b"
            r"|\b(corrective|action|finding).{0,20}high\s+risk\b"
        ),
        "hint": (
            "\nICA RISK LEVEL — risk_level_id is a UUID FK, not a string column:\n"
            "  WRONG: WHERE ica.risk_level = 'High'          ← column does not exist\n"
            "  WRONG: WHERE ica.risk_level_id = 'High'       ← comparing UUID FK to string\n"
            "  WRONG: WHERE ica.cause ILIKE '%High Risk%'    ← cause is free text, not risk level\n"
            "  WRONG: WHERE ica.capex_status = 'High Risk'   ← capex_status is not risk level\n"
            "  CORRECT: JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "           WHERE rl.name ILIKE '%High%'\n"
            "Risk level values (exact): 'High', 'Medium', 'Low', 'No Active Risk'\n"
            "Example — filter high risk:\n"
            "SELECT ica.corrective_action_id, ica.cause, ica.status, rl.name AS risk_level\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "WHERE rl.name ILIKE '%High%' LIMIT 100;\n"
            "Example — count by risk level:\n"
            "SELECT rl.name AS risk_level, COUNT(*) AS action_count\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "GROUP BY rl.name ORDER BY action_count DESC;\n"
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
            "\nMOST-RECENT INSPECTION WITH FORM DATA — anchor on ai_answers.submitted_on:\n"
            "The most recent inspection_report.submitted_on may have NO linked ai_answers.\n"
            "Always use the ai_answers table to find the most recently answered inspection.\n"
            "\nFor 'most recent inspection form questions and answers':\n"
            "SELECT ir.inspection_id, fac.name AS facility_name,\n"
            "       aq.label AS question, aa.answer_text AS answer, aa.module_name\n"
            "FROM ai_answers aa\n"
            "JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
            "JOIN inspection_report ir ON aa.inspection_report_id = ir.id\n"
            "JOIN facility fac ON ir.facility_id = fac.id\n"
            "WHERE aa.inspection_report_id = (\n"
            "    SELECT inspection_report_id FROM ai_answers\n"
            "    WHERE inspection_report_id IS NOT NULL\n"
            "      AND module_name = 'Inspection Form'\n"
            "    ORDER BY submitted_on DESC LIMIT 1\n"
            ")\n"
            "ORDER BY aq.label LIMIT 100;\n"
            "\nFor 'show most recent inspection' (metadata only):\n"
            "SELECT ir.inspection_id, fac.name AS facility_name,\n"
            "       ir.submitted_on, ir.status, ir.inspection_score\n"
            "FROM inspection_report ir\n"
            "JOIN facility fac ON ir.facility_id = fac.id\n"
            "WHERE ir.id = (\n"
            "    SELECT inspection_report_id FROM ai_answers\n"
            "    WHERE inspection_report_id IS NOT NULL\n"
            "    ORDER BY submitted_on DESC LIMIT 1\n"
            ");\n"
            "CRITICAL: join aa.inspection_report_id = ir.id (UUID FK), NOT ir.inspection_id.\n"
            "CRITICAL: always SELECT ir.inspection_id (varchar) not ir.id (UUID) for display.\n"
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
            "SCHEMA CONSTRAINT: corrective_action_id is UNIQUE per row — there is NO\n"
            "  history/audit table. There is no way to count 'how many times' a single\n"
            "  action was carried forward. Instead, count DISTINCT occurrences by cause:\n"
            "Example (recurring issues at a facility):\n"
            "SELECT ica.cause, COUNT(*) AS recurrence_count, fac.name AS facility_name\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "JOIN facility fac ON ir.facility_id = fac.id\n"
            "WHERE ica.status = 'CLOSE_WITH_DEFERRED'\n"
            "GROUP BY ica.cause, fac.name ORDER BY recurrence_count DESC LIMIT 20;\n"
            "Example (facility with most deferred issues):\n"
            "SELECT fac.name AS facility_name, COUNT(*) AS deferred_count\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "JOIN facility fac ON ir.facility_id = fac.id\n"
            "WHERE ica.status = 'CLOSE_WITH_DEFERRED'\n"
            "GROUP BY fac.name ORDER BY deferred_count DESC LIMIT 10;\n"
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