"""
ShapeMixin: validate_shape() -- structural query-shape checks (single-row
intent, LIMIT presence, etc). Extracted verbatim from the original
core/validator.py.
"""

import re

try:
    import sqlglot
    import sqlglot.expressions as exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False

from src.core.validator.models import ValidationResult, ValidationIssue, ValidatorStats


class ShapeMixin:

    # ------------------------------------------------------------------ #
    # Shape validation — HARD_FAIL for structural mismatches
    # Call after generate_sql when the expected shape is known.
    # ------------------------------------------------------------------ #

    # Map of question patterns → expected shape
    _SINGLE_ROW_PATTERNS = re.compile(
        r"\b(this\s+(month|quarter|year|week)\s+vs\.?\s+last|"
        r"last\s+(month|quarter|year|week)\s+vs\.?|"
        r"how\s+many.{0,30}(vs\.?|versus|compared?\s+to)|"
        r"raised\s+vs\.?\s+closed|"
        r"comparison|"
        r"faster\s+or\s+slower)\b",
        re.IGNORECASE,
    )


    def validate_shape(self, sql: str, question: str) -> ValidationResult:
        """
        AST-level shape validation. Called when query intent implies a
        specific result structure that the generated SQL must match.

        Currently enforces:
          - SINGLE_ROW intent: GROUP BY is forbidden (use FILTER aggregates)

        Returns ValidationResult with HARD_FAIL if shape is violated.
        This triggers a retry with an explicit structural correction message.
        """
        if not _HAS_SQLGLOT:
            return ValidationResult(passed=True, issues=[])

        issues: list[ValidationIssue] = []
        q_lower = question.lower()

        # ── Completed status filter ───────────────────────────────────────
        # "completed inspections" must filter status IN ('CLOSED','SUBMITTED').
        # status != 'DRAFT' alone is insufficient — it includes IN_PROGRESS etc.
        # NOTE: SOFT_WARN only — deepseek cannot fix this via retry because it
        # copies the seed example. Fix is in the seed SQL and CORE_CONTEXT.
        _COMPLETED_Q = re.compile(
            r"\b(completed|finished|done|finalized)\s+(inspection|inspections?)\b"
            r"|\binspections?\s+(were\s+)?(completed|finished|done|finalized)\b"
            r"|\bhow\s+many\s+inspections?.{0,20}(completed|finished|done)\b",
            re.IGNORECASE,
        )
        if _COMPLETED_Q.search(question):
            has_submitted = "'SUBMITTED'" in sql.upper() or "\"SUBMITTED\"" in sql
            has_closed = "'CLOSED'" in sql.upper() or "\"CLOSED\"" in sql
            if not (has_submitted or has_closed):
                issues.append(ValidationIssue(
                    "SOFT_WARN", "MISSING_COMPLETED_STATUS",
                    "Query asks about 'completed' inspections but SQL has neither 'CLOSED' nor 'SUBMITTED'",
                    detail="completed_inspection = status IN ('CLOSED','SUBMITTED')",
                    retryable=False,
                ))
                ValidatorStats.record("MISSING_COMPLETED_STATUS", is_retry_trigger=False)
                return ValidationResult(passed=True, issues=issues)  # warn but don't block

        # Detect single-row comparison intent
        is_single_row_intent = bool(self._SINGLE_ROW_PATTERNS.search(q_lower))

        if not is_single_row_intent:
            return ValidationResult(passed=True, issues=[])

        # Parse SQL AST and check for GROUP BY on the outermost SELECT
        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            return ValidationResult(passed=True, issues=[])

        # Find outermost SELECT
        outermost = None
        for node in tree.walk():
            if isinstance(node, sqlglot.exp.Select):
                outermost = node
                break

        if outermost is None:
            return ValidationResult(passed=True, issues=[])

        has_group_by = outermost.args.get("group") is not None

        # Allow GROUP BY for legitimate trend queries (quarter/month breakdown)
        _is_trend_groupby = bool(re.search(
            r'GROUP\s+BY\s+(date_trunc|EXTRACT|TO_CHAR|\d+)',
            sql, re.IGNORECASE
        ))
        # Allow GROUP BY when FILTER aggregates are already present — the query
        # is using the correct FILTER pattern and GROUP BY just adds a breakdown
        # dimension (e.g. CASE WHEN status='CLOSED' GROUP BY that CASE alias).
        _has_filter_aggregates = bool(re.search(r'\bFILTER\s*\(', sql, re.IGNORECASE))

        if has_group_by and not _is_trend_groupby and not _has_filter_aggregates:
            # Build domain-specific retry message
            q_lower_shape = question.lower()
            _is_ca_comparison = bool(re.search(
                r'\braised.{0,20}closed\b|\bclosed.{0,20}raised\b'
                r'|\b(faster|slower|closure.{0,20}rate)\b', q_lower_shape))
            _is_time_comparison = bool(re.search(
                r'\b(vs\.?|versus|this.{0,10}(month|quarter).{0,10}last'
                r'|last.{0,10}(month|quarter).{0,10}this)\b', q_lower_shape))

            if _is_ca_comparison:
                _fix_example = (
                    "  SELECT 'Q2 2026' AS quarter,\n"
                    "    COUNT(*) AS total_raised,\n"
                    "    COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS closed_count\n"
                    "  FROM inspection_corrective_action\n"
                    "  WHERE target_close_out_date >= date_trunc('quarter', CURRENT_DATE);"
                )
                _msg = "Use FILTER aggregates for raised/closed comparison (single row, no GROUP BY):\n"
            elif _is_time_comparison:
                _fix_example = (
                    "  SELECT\n"
                    "    COUNT(*) FILTER (WHERE date_trunc('month',col)=date_trunc('month',NOW())) AS this_month,\n"
                    "    COUNT(*) FILTER (WHERE date_trunc('month',col)=date_trunc('month',NOW()-INTERVAL '1 month')) AS last_month\n"
                    "  FROM table WHERE ...;"
                )
                _msg = "Use FILTER aggregates for time comparison (single row, no GROUP BY):\n"
            else:
                # Non-comparison GROUP BY: likely simple aggregate with unnecessary grouping
                _fix_example = (
                    "  -- For simple counts/aggregates, do NOT add GROUP BY:\n"
                    "  SELECT COUNT(DISTINCT col) AS count FROM table WHERE ...;\n"
                    "  -- Only use GROUP BY when the question asks for breakdown BY something."
                )
                _msg = "WRONG_SHAPE: remove unnecessary GROUP BY for a simple aggregate query:\n"

            issue = ValidationIssue(
                "HARD_FAIL",
                "WRONG_SHAPE",
                _msg + _fix_example,
                retryable=True,
            )
            ValidatorStats.record("WRONG_SHAPE", is_retry_trigger=True)
            return ValidationResult(passed=False, issues=[issue])

        return ValidationResult(passed=True, issues=[])
