"""
SemanticMixin: validate_semantic() -- SOFT_WARN-only advisory heuristics.
Extracted verbatim from the original core/validator.py.
"""

import re

try:
    import sqlglot
    import sqlglot.expressions as exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False

try:
    from src.core.business_glossary import check_glossary as _check_glossary
    _HAS_GLOSSARY = True
except ImportError:
    _HAS_GLOSSARY = False

from src.core.validator.models import ValidationIssue, ValidatorStats
from src.core.validator.ast_checks import _check_uuid_leaks_ast, _check_uuid_leaks_regex


class SemanticMixin:

    def validate_semantic(self, sql: str, question: str) -> list[ValidationIssue]:
        """
        Advisory heuristics.

        Returns SOFT_WARN issues only.
        These are for observability (logs, stats) only.
        They must NEVER be forwarded to the LLM retry prompt.
        They must NEVER cause retries.

        Callers: log these, count them via ValidatorStats, then discard.
        """
        warns: list[ValidationIssue] = []
        sql_lower = sql.lower()
        q_lower = question.lower()

        # Routing mismatch: asks about form answers but SQL avoids ai_answers
        asks_about_answers = bool(re.search(
            r'\b(answer|submission|response|submitted|filled|responded)\b', q_lower))
        uses_ai_answers = "ai_answers" in sql_lower
        uses_inspection = bool(re.search(
            r'\binspection_(report|corrective_action|schedule|cycle)\b', sql_lower))

        if asks_about_answers and not uses_inspection and not uses_ai_answers:
            warns.append(ValidationIssue(
                "SOFT_WARN", "ROUTING_MISMATCH",
                "Question mentions answers but SQL avoids ai_answers",
                detail="Consider get_answers tool or join via ai_answers.element_id",
                retryable=False,
            ))
            ValidatorStats.record("ROUTING_MISMATCH", is_retry_trigger=False)

        # UUID FK columns selected/grouped without their lookup JOIN.
        # Uses AST when available for precise SELECT clause detection.
        if _HAS_SQLGLOT:
            _check_uuid_leaks_ast(sql, self.UUID_FK_COLUMNS, warns)
        else:
            _check_uuid_leaks_regex(sql_lower, self.UUID_FK_COLUMNS, warns)

        # Required predicate checks — verify semantic constraints are present.
        # These are SOFT_WARN because the query may still be correct (e.g. user
        # is explicitly asking about DRAFT inspections).

        # "completed" inspections must include SUBMITTED or CLOSED
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
                warns.append(ValidationIssue(
                    "SOFT_WARN", "MISSING_COMPLETED_STATUS",
                    "Query asks about 'completed' inspections but SQL has neither 'CLOSED' nor 'SUBMITTED'",
                    detail="completed_inspection = status IN ('CLOSED','SUBMITTED')",
                    retryable=False,
                ))
                ValidatorStats.record("MISSING_COMPLETED_STATUS", is_retry_trigger=False)

        # NOTE: "overdue mitigative actions" via date arithmetic is now a HARD_FAIL,
        # enforced in validate() (see OVERDUE_DATE_ARITHMETIC) — promoted out of this
        # advisory-only method because it was firing consistently (both 3B and 7B,
        # in eval) with no retry and no user-visible signal.

        # ── Business glossary term checks ─────────────────────────────────────
        if _HAS_GLOSSARY:
            for severity, msg in _check_glossary(question, sql):
                warns.append(ValidationIssue(
                    "SOFT_WARN", "GLOSSARY_VIOLATION", msg,
                    detail="business_glossary check",
                    retryable=False,
                ))
                ValidatorStats.record("GLOSSARY_VIOLATION", is_retry_trigger=False)

        return warns
