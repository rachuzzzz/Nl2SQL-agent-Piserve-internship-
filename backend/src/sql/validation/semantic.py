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
    from src.knowledge.business_glossary import check_glossary as _check_glossary
    _HAS_GLOSSARY = True
except ImportError:
    _HAS_GLOSSARY = False

try:
    from src.knowledge.semantics_registry import (
        completed_statuses as _completed_statuses,
        AUDIT_REPORT_SEMANTICS as _AUDIT_REPORT_SEMANTICS,
    )
    # Bare "completed audits" (no qualifier) means audit_report.status, NOT
    # audit_plan.status — confirmed live, see business_glossary.py's matching
    # completed_audits/conducted_audits terms for the full rationale.
    _AUDIT_COMPLETED_STATUSES = _completed_statuses(_AUDIT_REPORT_SEMANTICS)
except ImportError:
    # Fallback — keep in sync with knowledge/semantics_registry.py's AUDIT_REPORT_SEMANTICS
    _AUDIT_COMPLETED_STATUSES = ["CLOSED", "CLOSED_WITH_EXCEPTION"]

from src.sql.validation.models import ValidationIssue, ValidatorStats
from src.sql.validation.ast_checks import _check_uuid_leaks_ast, _check_uuid_leaks_regex


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

        # Routing mismatch: asks about form answers but SQL avoids ai_answers/
        # ai_audit_answers. uses_domain_table recognizes EITHER domain's real
        # tables as "not a routing mismatch" — a bare word like "submitted"
        # (audit_report/schedule questions legitimately use this word
        # constantly) must not trigger this check just because the SQL
        # correctly has no ai_answers/ai_audit_answers JOIN. ai_audit_answers
        # was added 2026-07-29 once confirmed live/populated for audit —
        # previously audit had no ai_answers equivalent to route to at all.
        asks_about_answers = bool(re.search(
            r'\b(answer|submission|response|submitted|filled|responded)\b', q_lower))
        uses_ai_answers = "ai_answers" in sql_lower or "ai_audit_answers" in sql_lower
        uses_domain_table = bool(re.search(
            r'\b(?:inspection|audit)_(report|corrective_action|schedule|cycle)\b', sql_lower))

        if asks_about_answers and not uses_domain_table and not uses_ai_answers:
            warns.append(ValidationIssue(
                "SOFT_WARN", "ROUTING_MISMATCH",
                "Question mentions answers but SQL avoids ai_answers/ai_audit_answers",
                detail="Consider get_answers tool or join via ai_answers.element_id "
                       "(or ai_audit_answers.element_id for audit)",
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

        # "completed" audits — TWO SEPARATE checks, matching business_glossary
        # .py's completed_audits (audit_report) vs completed_audit_plans
        # (audit_plan) split. Bare "completed audits" (no qualifier) means
        # audit_report.status almost every time — was the audit itself
        # carried out and closed out — NOT whether its planning document was
        # approved. Explicit "audit plan(s)" wording is the separate,
        # genuinely-different question. Both reuse the inspection check's
        # code (MISSING_COMPLETED_STATUS) — just gated on audit wording and
        # each other's own status literal.
        _COMPLETED_AUDIT_PLAN_Q = re.compile(
            r"\b(completed|finished|done|finalized|approved)\s+audit\s+plans?\b"
            r"|\baudit\s+plans?\s+(were\s+)?(completed|finished|done|finalized|approved)\b"
            r"|\bhow\s+many\s+audit\s+plans?.{0,20}(completed|finished|done|approved)\b",
            re.IGNORECASE,
        )
        _COMPLETED_AUDIT_Q = re.compile(
            r"\b(completed|finished|done|finalized)\s+(audit|audits)\b(?!\s+plans?\b)"
            r"|\baudits?\s+(were\s+)?(completed|finished|done|finalized)\b"
            r"|\bhow\s+many\s+audits?\b(?!\s+plans?).{0,20}(completed|finished|done)\b",
            re.IGNORECASE,
        )
        if _COMPLETED_AUDIT_PLAN_Q.search(question):
            has_plan_completed_status = "'APPROVED'" in sql.upper()
            if not has_plan_completed_status:
                warns.append(ValidationIssue(
                    "SOFT_WARN", "MISSING_COMPLETED_STATUS",
                    "Query asks about 'completed' audit plans but SQL has no 'APPROVED'",
                    detail="completed_audit_plan = audit_plan.status = 'APPROVED'",
                    retryable=False,
                ))
                ValidatorStats.record("MISSING_COMPLETED_STATUS", is_retry_trigger=False)
        elif _COMPLETED_AUDIT_Q.search(question):
            has_completed_status = any(
                f"'{s}'" in sql.upper() for s in _AUDIT_COMPLETED_STATUSES
            )
            if not has_completed_status:
                warns.append(ValidationIssue(
                    "SOFT_WARN", "MISSING_COMPLETED_STATUS",
                    f"Query asks about 'completed' audits but SQL has none of {_AUDIT_COMPLETED_STATUSES}",
                    detail=f"completed_audit (audit_report) = status IN {_AUDIT_COMPLETED_STATUSES}",
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
