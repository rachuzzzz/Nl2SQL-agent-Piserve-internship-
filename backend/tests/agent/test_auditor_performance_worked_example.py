"""
Regression test for the audit domain's "Auditor performance ranking" Tier-2
worked example (examples_tier2_audit.py) — added 2026-07-29 after an observed
inconsistency: "who is the worst inspector" correctly triggered a clarifying
question (inspection domain already had this guard), but "who is the worst
auditor" had no equivalent and silently picked its own metric instead of
asking. Mirrors inspection's "Inspector performance ranking" example.

No LLM, no live DB — pure regex/string matching against the real registered
EXAMPLES_TIER2_AUDIT data via build_worked_examples_audit().
"""

from src.agent.prompts.builder_audit import build_worked_examples_audit

_HEADER = "ADDITIONAL WORKED EXAMPLES"


def test_worst_auditor_matches_performance_ranking_example():
    result = build_worked_examples_audit("who is the worst auditor")
    assert _HEADER in result
    assert "audit_score" in result
    assert "Could you clarify" in result


def test_best_performing_auditor_matches():
    result = build_worked_examples_audit("which auditor performed best this cycle")
    assert _HEADER in result


def test_top_auditors_leaderboard_matches():
    result = build_worked_examples_audit("auditor leaderboard")
    assert _HEADER in result


def test_plain_auditor_assignment_question_does_not_match_ranking_example():
    # "who is the auditor for this schedule" is a plain assignment lookup, not
    # a performance-ranking question — must not pull in the ranking guidance.
    result = build_worked_examples_audit("who is the auditor for this schedule")
    assert "Auditor performance ranking" not in result


def test_no_match_for_unrelated_question():
    result = build_worked_examples_audit("list all audit schedules")
    assert "Auditor performance ranking" not in result
