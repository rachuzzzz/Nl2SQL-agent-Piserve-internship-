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
    # NOTE: the validator checks table_short ("question") in sql_lower, which
    # matches entityType='QUESTION' in JSONB queries. Only include columns
    # whose names are unique enough to not cause false positives.
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

    # Tables that DO NOT EXIST but LLMs constantly hallucinate
    HALLUCINATED_TABLES = {
        "fb_users", "fb_user", "users", "fb_creators",
        "fb_sections", "fb_section",
        "fb_questions",
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

        # Check hallucinated columns
        sql_lower = sql.lower()
        for col in self.HALLUCINATED_COLUMNS:
            table, column = col.split(".")
            # Check that the full table name appears as a table reference (not inside a string)
            # and the column name appears with a dot prefix
            if re.search(rf"\b{re.escape(table)}\b", sql_lower) and \
               re.search(rf"\b\w*\.{column}\b", sql_lower):
                errors.append(f"HALLUCINATION: '{col}' doesn't exist. Use JSONB translations.")

        # Check hallucinated tables
        for table in self.HALLUCINATED_TABLES:
            if re.search(rf"\b{re.escape(table)}\b", sql_lower):
                errors.append(
                    f"HALLUCINATION: table '{table}' does not exist. "
                    f"Use get_schema() to find the correct table. "
                    f"For creator info, check fb_forms.created_by column."
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
        Returns a list of WARNING strings (does not block execution).

        Catches the common sqlcoder mistake of counting fb_forms rows when
        the question asks about questions/pages/sections.
        """
        warnings = []
        sql_upper = sql.upper()
        q_lower = question.lower()

        has_jsonb = "JSONB_ARRAY_ELEMENTS" in sql_upper
        # Also check for subqueries that might contain jsonb_array_elements
        has_subquery = sql_upper.count("SELECT") > 1
        # Only QUESTION and PAGE exist as entityTypes in this database
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

        return warnings

    def fix_jsonb_arrows(self, sql: str) -> str:
        """Auto-fix -> to ->> for known JSONB text keys."""
        keys = ['translatedText', 'elementId', 'entityType', 'attribute', 'language']
        for key in keys:
            sql = sql.replace(f"->'{key}'", f"->>'{key}'")
        return sql

    def fix_jsonb_key_case(self, sql: str) -> str:
        """
        Fix incorrect casing of known JSONB keys inside ->> '...' expressions.
        PostgreSQL JSONB is case-sensitive — 'translatedtext' returns NULL,
        only 'translatedText' works.
        """
        # Map every known wrong-case variant to the correct camelCase key
        canonical = {
            'translatedtext': 'translatedText',
            'elementid':      'elementId',
            'entitytype':     'entityType',
        }
        for wrong, right in canonical.items():
            # Match ->> 'wrongkey' or -> 'wrongkey' (case-insensitive on the key part)
            sql = re.sub(
                r"(->>'?)" + re.escape(wrong) + r"('?)",
                lambda m, r=right: m.group(1) + r + m.group(2),
                sql,
                flags=re.IGNORECASE,
            )
        return sql

    def clean_sql(self, sql: str) -> str:
        """Clean up model output: remove markdown, extract SELECT, fix arrows, fix SELECT clause."""
        # Remove markdown fences
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```\s*$", "", sql)
        # Remove bracket tags sqlcoder emits ([SQL], [/SQL], [QUESTION], [/ANSWER], etc.)
        sql = re.sub(r"\[/?[A-Z_]+\]", "", sql, flags=re.IGNORECASE)
        # Remove ### headers
        sql = re.sub(r"^###.*$", "", sql, flags=re.MULTILINE)
        sql = sql.strip()

        # Extract first SELECT if model generated extra stuff
        if "SELECT" in sql.upper():
            select_start = sql.upper().index("SELECT")
            sql = sql[select_start:]

        # Fix JSONB arrows
        sql = self.fix_jsonb_arrows(sql)

        # Fix JSONB key casing (translatedtext → translatedText, etc.)
        sql = self.fix_jsonb_key_case(sql)

        # Strip any trailing bracket tags that survived
        sql = re.sub(r"\s*\[/?[A-Z_]+\]\s*$", "", sql, flags=re.IGNORECASE).rstrip()

        # Ensure semicolon
        if sql and not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"

        return sql

    def _fix_jsonb_select(self, sql: str) -> str:
        """
        Catches ONE specific sqlcoder mistake: the model copies a form-listing
        SELECT clause (f.name, f.status, f.active) but then adds a
        jsonb_array_elements JOIN, producing nonsensical output.

        Only rewrites when the SELECT clause contains exclusively simple
        fb_forms columns (name, status, active) with no other content.
        Leaves all other queries untouched — timestamps, DISTINCT, functions,
        aggregates, etc. are never rewritten.
        """
        sql_upper = sql.upper()

        # Only apply if the query uses jsonb_array_elements
        if "JSONB_ARRAY_ELEMENTS" not in sql_upper:
            return sql

        # Already has elem->>'translatedText' — nothing to fix
        if "TRANSLATEDTEXT" in sql_upper and "ELEM" in sql_upper:
            return sql

        # Find the SELECT ... FROM boundary
        select_match = re.match(r"(SELECT\s+)(.*?)(\s+FROM\s+)", sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return sql

        select_clause = select_match.group(2).strip()

        # Only rewrite if the SELECT clause looks like the known bad pattern:
        # just simple column references (f.name, f.status, etc.) with no
        # functions, timestamps, DISTINCT, *, subqueries, or JSONB operators
        skip_indicators = [
            r"\b(COUNT|AVG|SUM|MIN|MAX|DISTINCT)\b",  # aggregates
            r"\b(created_on|modified_on|created_at|updated_at|creation_date|date|timestamp)\b",  # dates
            r"\*",                                       # SELECT *
            r"->>?",                                     # JSONB operators
            r"\(",                                       # any function call
        ]
        for pattern in skip_indicators:
            if re.search(pattern, select_clause, re.IGNORECASE):
                return sql

        # Also skip subqueries and GROUP BY
        if sql_upper.count("SELECT") > 1 or "GROUP BY" in sql_upper:
            return sql

        # At this point the SELECT clause has only simple column refs and the
        # query unpacks JSONB — this is the known bad pattern. Fix it.
        new_select = "elem->>'translatedText' AS label, elem->>'elementId' AS element_id"
        sql = select_match.group(1) + new_select + select_match.group(3) + sql[select_match.end():]

        return sql