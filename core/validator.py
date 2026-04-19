"""
SQL Validator — catches common LLM mistakes before execution.
Reused from the original pipeline with all learned patterns.
"""

import re


class SQLValidator:
    """
    Validates generated SQL before it hits the database.
    Catches hallucinated columns, wrong JOIN patterns, JSONB mistakes.
    """

    # Columns that DO NOT EXIST but LLMs constantly hallucinate
    HALLUCINATED_COLUMNS = {
        "fb_question.label", "fb_question.title",
        "fb_question.text", "fb_question.description",
        "fb_question.question_json",
        "fb_section.title", "fb_section.label", "fb_section.description",
        "fb_sub_form.title", "fb_sub_form.label", "fb_sub_form.description",
        "fb_page.title", "fb_page.label",
        "fb_question_group.title",
        "fb_multifield.label",
    }

    # Tables that DO NOT EXIST but LLMs constantly hallucinate.
    # NOTE: fb_section IS a real table (it has insertion_order) — the problem
    # is its columns, which the HALLUCINATED_COLUMNS check handles. Do NOT
    # include fb_section here or legitimate section queries will be rejected.
    # Only plural/nonexistent names belong here.
    HALLUCINATED_TABLES = {
        "fb_users", "fb_user", "fb_creators",
        "fb_sections",          # plural — doesn't exist (singular fb_section does)
        "fb_questions",         # plural — doesn't exist (singular fb_question does)
        # Prevent LLMs from guessing a single global answer table
        "fb_answers", "fb_answer", "fb_form_answer",
        "fb_form_answers", "fb_submissions", "fb_submission",
        "fb_responses", "fb_response",
        # NOTE: "users" is a REAL table (has first_name, last_name, email).
        # Do NOT add it here. fb_users and fb_user are fake — those stay.
    }

    # Regex patterns that indicate known mistakes
    WRONG_PATTERNS = [
        (r"(\w+_appearance)\.id\s*=\s*\w+\.form_element_id",
         "Shared PK wrong: use subtype.id = base.id, not .form_element_id"),
        (r"(\w+_validation)\.id\s*=\s*\w+\.form_element_id",
         "Shared PK wrong: use subtype.id = base.id, not .form_element_id"),
        (r"translations\s*->>\s*'",
         "JSONB: must use jsonb_array_elements() to unpack the array first"),
        (r"elem\s*->>\s*'translated_text'",
         "JSONB key: use camelCase 'translatedText' not 'translated_text'"),
        (r"elem\s*->>\s*'entity_type'",
         "JSONB key: use camelCase 'entityType' not 'entity_type'"),
        (r"elem\s*->>\s*'element_id'",
         "JSONB key: use camelCase 'elementId' not 'element_id'"),
        (r"elem->(?!>)'(translatedText|elementId|entityType|attribute|language)'",
         "JSONB: use ->> (double arrow) not -> for text extraction"),
    ]

    # Dynamic answer tables match this pattern: fb_ + UUID with underscores
    _DYNAMIC_TABLE_RE = re.compile(
        r'^fb_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}'
    )

    def validate(self, sql: str) -> tuple[bool, list[str]]:
        """
        Returns (is_valid, list_of_errors).
        Warnings (prefixed with WARNING:) don't count as errors.
        """
        errors = []

        if not sql or sql.startswith("-- ERROR:"):
            return False, [sql or "Empty SQL"]

        # Block DML/DDL
        if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE)\b", sql, re.IGNORECASE):
            errors.append("BLOCKED: DML/DDL detected. Only SELECT allowed.")
            return False, errors

        sql_lower = sql.lower()

        # Check hallucinated columns
        for col in self.HALLUCINATED_COLUMNS:
            table, column = col.split(".")
            if re.search(rf"\b{re.escape(table)}\b", sql_lower) and \
               re.search(rf"\b\w*\.{column}\b", sql_lower):
                errors.append(f"HALLUCINATION: '{col}' doesn't exist. Use JSONB translations.")

        # Check hallucinated tables
        for table in self.HALLUCINATED_TABLES:
            if re.search(rf"\b{re.escape(table)}\b", sql_lower):
                errors.append(
                    f"HALLUCINATION: table '{table}' does not exist. "
                    f"Use get_schema() to find the correct table. "
                    f"For answer/submission data, use resolve_answer_table tool."
                )

        # Check wrong patterns
        for pattern, msg in self.WRONG_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                errors.append(f"WRONG: {msg}")

        # Warn about missing LIMIT
        has_agg = bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql, re.IGNORECASE))
        if not has_agg and not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
            errors.append("WARNING: No LIMIT on non-aggregate query.")

        real_errors = [e for e in errors if not e.startswith("WARNING:")]
        return len(real_errors) == 0, errors

    def validate_semantic(self, sql: str, question: str) -> list[str]:
        """
        Cross-check the generated SQL against the intent of the question.
        """
        warnings = []
        sql_upper = sql.upper()
        q_lower = question.lower()

        has_jsonb = "JSONB_ARRAY_ELEMENTS" in sql_upper
        asks_about_questions = bool(re.search(r'\bquestion', q_lower))
        asks_about_pages     = bool(re.search(r'\bpage', q_lower))
        asks_about_elements  = asks_about_questions or asks_about_pages

        if asks_about_elements and not has_jsonb:
            entity = "questions" if asks_about_questions else "pages"
            warnings.append(
                f"WARNING: Question asks about {entity} but SQL has no "
                f"jsonb_array_elements(). "
                f"Use: SELECT COUNT(*) FROM fb_translation_json, "
                f"jsonb_array_elements(translations) AS elem "
                f"WHERE elem->>'language'='eng' AND elem->>'attribute'='NAME' "
                f"AND elem->>'entityType'='{'QUESTION' if asks_about_questions else 'PAGE'}';"
            )

        # Warn if question asks about answers/submissions but SQL doesn't
        # use dynamic tables (agent should use answer tools instead).
        # EXCEPTION: if the SQL references inspection_report or
        # inspection_corrective_action, those are proper relational tables
        # where scores/submissions live — no warning needed.
        asks_about_answers = bool(re.search(
            r'\b(answer|submission|response|score|submitted|filled)\b', q_lower))
        uses_inspection_tables = bool(re.search(
            r'\binspection_(report|corrective_action|schedule|cycle)\b', sql.lower()))
        if asks_about_answers and "ANSWER_DATA" not in sql_upper and not uses_inspection_tables:
            warnings.append(
                "WARNING: Question asks about answers/submissions. "
                "Use resolve_answer_table + query_answers tools instead of generate_sql."
            )

        return warnings

    def is_dynamic_answer_table(self, table_name: str) -> bool:
        """Check if a table name matches the dynamic answer table pattern."""
        return bool(self._DYNAMIC_TABLE_RE.match(table_name))

    def fix_jsonb_arrows(self, sql: str) -> str:
        """Auto-fix -> to ->> for known JSONB text keys."""
        keys = ['translatedText', 'elementId', 'entityType', 'attribute', 'language']
        for key in keys:
            sql = sql.replace(f"->'{key}'", f"->>'{key}'")
        return sql

    def fix_jsonb_key_case(self, sql: str) -> str:
        """Fix incorrect casing of known JSONB keys."""
        canonical = {
            'translatedtext': 'translatedText',
            'elementid':      'elementId',
            'entitytype':     'entityType',
        }
        for wrong, right in canonical.items():
            sql = re.sub(
                r"(->>'?)" + re.escape(wrong) + r"('?)",
                lambda m, r=right: m.group(1) + r + m.group(2),
                sql,
                flags=re.IGNORECASE,
            )
        return sql

    def fix_reserved_aliases(self, sql: str) -> str:
        """
        Fix SQL reserved words used as table aliases.
        qwen2.5 commonly uses 'is' for inspection_schedule, etc.
        Replaces both the alias declaration and all column references.
        """
        replacements = {
            'is': 'isch',   # inspection_schedule is → isch
        }
        for reserved, safe in replacements.items():
            # Step 1: Find "table_name <reserved>" used as alias
            # (after a table name, before ON/WHERE/./comma/newline)
            alias_pattern = re.compile(
                r'(\b\w+\s+)' + r'\b' + re.escape(reserved) + r'\b'
                r'(?=\s*\.|\s+ON\b|\s+WHERE\b|\s+GROUP\b|\s+ORDER\b'
                r'|\s+LEFT\b|\s+JOIN\b|\s+INNER\b|\s*\n|\s*,)',
                re.IGNORECASE,
            )
            if alias_pattern.search(sql):
                # Step 2: Replace the alias declaration
                sql = alias_pattern.sub(
                    lambda m, s=safe: m.group(1) + s, sql)
                # Step 3: Replace all "reserved." column references
                # Use word boundary to avoid replacing inside words
                sql = re.sub(
                    r'\b' + re.escape(reserved) + r'\.',
                    safe + '.',
                    sql,
                )
        return sql

    def clean_sql(self, sql: str) -> str:
        """Clean up model output: remove markdown, extract SELECT, fix arrows."""
        # Remove markdown fences ANYWHERE (not just at end)
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```", "", sql)  # strip ALL remaining fences
        sql = re.sub(r"\[/?[A-Z_]+\]", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^###.*$", "", sql, flags=re.MULTILINE)
        # Remove markdown links/references that sqlcoder sometimes appends
        sql = re.sub(r"-\s*\[.*?\]\(.*?\)", "", sql)
        # Remove preamble lines ending with colon — qwen2.5 often outputs
        # "SELECT query can be used:" or "The SQL query is:" before the
        # actual SQL. These lines always end with ':' and contain no SQL.
        sql = re.sub(r"^[^\n;]*:\s*$", "", sql, flags=re.MULTILINE)
        sql = sql.strip()

        if "SELECT" in sql.upper():
            # Preserve WITH ... AS (...) CTEs — but only if WITH is followed
            # by a valid CTE name (identifier + AS), not prose like "with them."
            sql_upper = sql.upper()
            select_pos = sql_upper.find("SELECT")
            # Look for "WITH <identifier> AS" pattern (valid CTE)
            cte_match = re.search(
                r'\bWITH\s+(?:RECURSIVE\s+)?\w+\s+AS\s*\(',
                sql, re.IGNORECASE
            )
            if cte_match and cte_match.start() < select_pos:
                sql = sql[cte_match.start():]
            else:
                sql = sql[select_pos:]

        # Truncate after the first semicolon — everything after is garbage
        # (handles sqlcoder appending links, comments, references after the SQL)
        semi_match = re.search(r";", sql)
        if semi_match:
            sql = sql[:semi_match.end()]

        sql = self.fix_jsonb_arrows(sql)
        sql = self.fix_jsonb_key_case(sql)
        sql = self.fix_reserved_aliases(sql)
        sql = re.sub(r"\s*\[/?[A-Z_]+\]\s*$", "", sql, flags=re.IGNORECASE).rstrip()

        if sql and not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"

        return sql

    def _fix_jsonb_select(self, sql: str) -> str:
        """
        Catches ONE specific sqlcoder mistake: the model copies a form-listing
        SELECT clause but then adds a jsonb_array_elements JOIN.
        """
        sql_upper = sql.upper()
        if "JSONB_ARRAY_ELEMENTS" not in sql_upper:
            return sql
        if "TRANSLATEDTEXT" in sql_upper and "ELEM" in sql_upper:
            return sql

        select_match = re.match(r"(SELECT\s+)(.*?)(\s+FROM\s+)", sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return sql

        select_clause = select_match.group(2).strip()
        skip_indicators = [
            r"\b(COUNT|AVG|SUM|MIN|MAX|DISTINCT)\b",
            r"\b(created_on|modified_on|created_at|updated_at|creation_date|date|timestamp)\b",
            r"\*", r"->>?", r"\(",
        ]
        for pattern in skip_indicators:
            if re.search(pattern, select_clause, re.IGNORECASE):
                return sql

        if sql_upper.count("SELECT") > 1 or "GROUP BY" in sql_upper:
            return sql

        new_select = "elem->>'translatedText' AS label, elem->>'elementId' AS element_id"
        sql = select_match.group(1) + new_select + select_match.group(3) + sql[select_match.end():]
        return sql