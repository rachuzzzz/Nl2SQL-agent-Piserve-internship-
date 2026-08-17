"""
Unit tests for loop.py's _validation_issue_signature() — the Bug 1 fix from
the 2026-08-07 capex/opex audit (qid=e63f03af). See loop.py's docstring on
_validation_issue_signature for the full incident writeup.
"""

from src.agent.orchestrator.loop import _validation_issue_signature
from src.sql.validation.models import ValidationIssue


def test_same_code_different_quoted_alias_gives_different_signatures():
    a = ValidationIssue("HARD_FAIL", "UNDEFINED_ALIAS", "Table alias 'ir' is used but not joined")
    b = ValidationIssue("HARD_FAIL", "UNDEFINED_ALIAS", "Table alias 'proj' is used but not joined")
    assert _validation_issue_signature(a) != _validation_issue_signature(b)


def test_same_code_same_quoted_alias_gives_identical_signatures():
    a = ValidationIssue("HARD_FAIL", "UNDEFINED_ALIAS", "Table alias 'ir' is used but not joined")
    b = ValidationIssue("HARD_FAIL", "UNDEFINED_ALIAS", "Table alias 'ir' is used but not joined")
    assert _validation_issue_signature(a) == _validation_issue_signature(b)


def test_code_with_no_quoted_identifier_falls_back_to_bare_code():
    # PO_PR_DOUBLE_JOIN's real message is static prose with no quoted
    # identifier — a genuinely stable, repeatable signature is desired here.
    issue = ValidationIssue(
        "HARD_FAIL", "PO_PR_DOUBLE_JOIN",
        "po_details and pr_details are both directly LEFT JOINed into a GROUP BY query",
    )
    assert _validation_issue_signature(issue) == "PO_PR_DOUBLE_JOIN"


def test_original_code_and_message_are_untouched():
    issue = ValidationIssue("HARD_FAIL", "UNDEFINED_ALIAS", "Table alias 'ir' is used but not joined")
    _validation_issue_signature(issue)
    assert issue.code == "UNDEFINED_ALIAS"
    assert issue.message == "Table alias 'ir' is used but not joined"
