"""
SQL Validator — catches common LLM mistakes before execution.
"""

import re


class SQLValidator:

    # Tables the LLM might hallucinate that don't exist.
    # NOTE: correct table names are ai_questions and ai_answers (plural).
    HALLUCINATED_TABLES = {
        # singular forms (the most common LLM mistake)
        "ai_question", "ai_answer",
        # other fb_* hallucinations
        "fb_users", "fb_user",
        "fb_answers", "fb_answer", "fb_form_answer", "fb_form_answers",
        "fb_submissions", "fb_submission",
        "fb_responses", "fb_response",
    }

    # Columns that don't exist on ai_questions — LLM keeps guessing these
    HALLUCINATED_AI_COLUMNS = {
        "ai_questions.question_label",  # real name: label
        "ai_questions.form_name",       # doesn't exist, use module_name
        "ai_questions.question_type",   # doesn't exist, use entity_type
        "ai_questions.page_name",       # doesn't exist
        "ai_questions.required",        # doesn't exist
        "ai_answers.answer_value",      # doesn't exist: use answer_text or answer_numeric
        "ai_answers.form_name",         # doesn't exist, use module_name
    }

    # Columns the LLM hallucinates on inspection_report (they don't exist as direct columns —
    # you must JOIN to users/facility/etc. to get these values)
    HALLUCINATED_IR_COLUMNS = {
        "inspection_report.facility_name":   "JOIN facility fac ON ir.facility_id = fac.id → fac.name",
        "inspection_report.inspector_name":  "JOIN users u ON ir.inspector_user_id = u.id → u.first_name || ' ' || u.last_name",
        "inspection_report.inspectee_name":  "JOIN users u ON ir.inspectee_user_id = u.id → u.first_name || ' ' || u.last_name",
        "inspection_report.client_name":     "JOIN client cl ON ir.client_id = cl.id → cl.name",
        "inspection_report.project_name":    "JOIN project proj ON ir.project_id = proj.id → proj.name",
        "inspection_report.type_name":       "JOIN inspection_type it ON ir.inspection_type_id = it.id → it.name",
        "inspection_report.inspection_type": "JOIN inspection_type it ON ir.inspection_type_id = it.id → it.name",
    }

    # ir.id is the UUID PK — selecting it for display produces UUID garbage.
    # The correct column is ir.inspection_id (varchar like '2026/04/ST001/INS001').
    # Match ir.id only when it appears in the SELECT clause (between SELECT and FROM).
    # Using ir.id in a JOIN condition (ON aa.inspection_report_id = ir.id) is CORRECT
    # and must NOT be flagged.
    _IR_ID_SELECT_RE = re.compile(
        r'\bselect\b(.*?)\bfrom\b',
        re.IGNORECASE | re.DOTALL
    )

    # Dynamic fb_{uuid} pattern — these should never appear in generated SQL;
    # all answer data is now in ai_answer.
    _DYNAMIC_TABLE_RE = re.compile(
        r'\bfb_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}\b'
    )

    def validate(self, sql: str) -> tuple[bool, list[str]]:
        errors = []

        if not sql or sql.startswith("-- ERROR:"):
            return False, [sql or "Empty SQL"]

        # Block DML/DDL
        if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE)\b", sql, re.IGNORECASE):
            errors.append("BLOCKED: DML/DDL detected. Only SELECT allowed.")
            return False, errors

        sql_lower = sql.lower()

        # Block hallucinated tables
        for table in self.HALLUCINATED_TABLES:
            if re.search(rf"\b{re.escape(table)}\b", sql_lower):
                errors.append(
                    f"HALLUCINATION: table '{table}' does not exist. "
                    f"Use ai_answers / ai_questions (plural) for form data."
                )

        # Block hallucinated columns on ai_* tables
        for col_path in self.HALLUCINATED_AI_COLUMNS:
            table, col = col_path.split(".")
            if re.search(rf"\b{re.escape(table)}\b", sql_lower) and \
               re.search(rf"\b\w*\.{re.escape(col)}\b", sql_lower):
                hints = {
                    "question_label": "use aq.label instead",
                    "form_name":      "use module_name instead (no form_name column exists)",
                    "question_type":  "use entity_type instead",
                    "answer_value":   "use answer_text (text) or answer_numeric (numeric) instead",
                    "page_name":      "column does not exist on ai_questions",
                    "required":       "column does not exist on ai_questions",
                }
                hint = hints.get(col, "column does not exist")
                errors.append(f"WRONG COLUMN: {col_path} — {hint}")

        # Block hallucinated columns on inspection_report
        # These values require JOINs — they are not direct columns on the table.
        ir_aliases = ["ir", "inspection_report"]
        for col_path, join_hint in self.HALLUCINATED_IR_COLUMNS.items():
            _, col = col_path.split(".")
            for alias in ir_aliases:
                if re.search(rf"\b{re.escape(alias)}\.{re.escape(col)}\b", sql_lower):
                    errors.append(
                        f"WRONG COLUMN: {col_path} does not exist. "
                        f"To get this value: {join_hint}"
                    )
                    break

        # Block ir.id in SELECT clause — outputs UUID instead of human-readable inspection_id
        # Only check the SELECT clause (not JOIN/WHERE where ir.id is legitimately used)
        select_clause_match = self._IR_ID_SELECT_RE.search(sql_lower)
        if select_clause_match:
            select_clause = select_clause_match.group(1)
            # ir.id appears in SELECT clause AND is not just part of a longer column name
            if re.search(r'\bir\.id\b', select_clause) and                re.search(r'\binspection_report\b', sql_lower):
                errors.append(
                    "WRONG COLUMN: ir.id is the UUID primary key — never use it for display. "
                    "Use ir.inspection_id instead (varchar like '2026/04/ST001/INS001'). "
                    "Note: using ir.id in JOIN conditions (ON ... = ir.id) is correct and fine."
                )

        # Block dynamic fb_{uuid} tables — use ai_answer instead
        if self._DYNAMIC_TABLE_RE.search(sql):
            errors.append(
                "BLOCKED: Dynamic fb_<uuid> table in SQL. "
                "Use ai_answer table or the get_answers tool instead."
            )

        # Warn about missing LIMIT
        has_agg = bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql, re.IGNORECASE))
        if not has_agg and not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
            errors.append("WARNING: No LIMIT on non-aggregate query.")

        real_errors = [e for e in errors if not e.startswith("WARNING:")]
        return len(real_errors) == 0, errors

    # FK columns that must be resolved via JOIN — never selected raw
    UUID_FK_COLUMNS = {
        "facility_id":             ("facility",         "fac", "fac.name AS facility_name"),
        "inspector_user_id":       ("users",            "u",   "u.first_name || ' ' || u.last_name AS inspector_name"),
        "inspectee_user_id":       ("users",            "insp","insp.first_name || ' ' || insp.last_name AS inspectee_name"),
        "client_id":               ("client",           "cl",  "cl.name AS client_name"),
        "project_id":              ("project",          "proj","proj.name AS project_name"),
        "inspection_type_id":      ("inspection_type",  "it",  "it.name AS inspection_type_name"),
        "inspection_sub_type_id":  ("inspection_sub_type","ist","ist.name AS subtype_name"),
        "module_id":               ("fb_modules",       "mod", "mod.name AS module_name"),
    }

    def validate_semantic(self, sql: str, question: str) -> list[str]:
        warnings = []
        sql_lower = sql.lower()
        q_lower = question.lower()

        asks_about_answers = bool(re.search(
            r'\b(answer|submission|response|submitted|filled|responded)\b', q_lower))
        uses_inspection = bool(re.search(
            r'\binspection_(report|corrective_action|schedule|cycle)\b', sql_lower))
        uses_ai_answer = "ai_answers" in sql_lower

        if asks_about_answers and not uses_inspection and not uses_ai_answer:
            warnings.append(
                "WARNING: Question asks about answers/submissions but SQL doesn't use "
                "ai_answers. Consider using the get_answers tool or join via ai_answers."
            )

        # Detect UUID FK columns being selected without their lookup JOIN.
        # e.g. SELECT facility_id FROM inspection_report GROUP BY facility_id
        # — should be: JOIN facility fac ON ir.facility_id = fac.id → SELECT fac.name
        for fk_col, (lookup_table, alias, select_expr) in self.UUID_FK_COLUMNS.items():
            col_in_select = bool(re.search(
                rf'\bselect\b.*\b{re.escape(fk_col)}\b', sql_lower, re.DOTALL
            ))
            col_in_groupby = bool(re.search(
                rf'\bgroup\s+by\b.*\b{re.escape(fk_col)}\b', sql_lower, re.DOTALL
            ))
            lookup_joined = lookup_table in sql_lower

            if (col_in_select or col_in_groupby) and not lookup_joined:
                warnings.append(
                    f"WARNING: UUID LEAK — '{fk_col}' is selected/grouped without "
                    f"joining '{lookup_table}'. This returns raw UUIDs instead of names. "
                    f"Fix: JOIN {lookup_table} {alias} ON ...{fk_col} = {alias}.id "
                    f"then SELECT {select_expr}"
                )

        return warnings

    def fix_reserved_aliases(self, sql: str) -> str:
        """Fix 'is' used as table alias (common with inspection_schedule)."""
        replacements = {"is": "isch"}
        for reserved, safe in replacements.items():
            alias_pattern = re.compile(
                r'(\b\w+\s+)' + r'\b' + re.escape(reserved) + r'\b'
                r'(?=\s*\.|\s+ON\b|\s+WHERE\b|\s+GROUP\b|\s+ORDER\b'
                r'|\s+LEFT\b|\s+JOIN\b|\s+INNER\b|\s*\n|\s*,)',
                re.IGNORECASE,
            )
            if alias_pattern.search(sql):
                sql = alias_pattern.sub(lambda m, s=safe: m.group(1) + s, sql)
                sql = re.sub(r'\b' + re.escape(reserved) + r'\.', safe + '.', sql)
        return sql

    def clean_sql(self, sql: str) -> str:
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```", "", sql)
        sql = re.sub(r"\[/?[A-Z_]+\]", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^###.*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"-\s*\[.*?\]\(.*?\)", "", sql)
        sql = re.sub(r"^[^\n;]*:\s*$", "", sql, flags=re.MULTILINE)
        sql = sql.strip()

        if "SELECT" in sql.upper():
            select_pos = sql.upper().find("SELECT")
            cte_match = re.search(r'\bWITH\s+(?:RECURSIVE\s+)?\w+\s+AS\s*\(', sql, re.IGNORECASE)
            if cte_match and cte_match.start() < select_pos:
                sql = sql[cte_match.start():]
            else:
                sql = sql[select_pos:]

        semi = re.search(r";", sql)
        if semi:
            sql = sql[:semi.end()]

        sql = self.fix_reserved_aliases(sql)
        sql = re.sub(r"\s*\[/?[A-Z_]+\]\s*$", "", sql, flags=re.IGNORECASE).rstrip()

        if sql and not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"

        return sql

    def is_dynamic_answer_table(self, table_name: str) -> bool:
        return bool(re.match(
            r'^fb_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}$',
            table_name
        ))