"""
Direct regex unit tests for session.py's module-level _FOLLOWUP_RE (hoisted
and broadened) and the new _COUNT_FOLLOWUP_RE — 2026-08-07 fix
(debug_llm_20260807_153128, qid=3dbcdaf3): "how many of the ones listed
above are completed" was answered with zero tool calls, in part because
_FOLLOWUP_RE (then a local inside build_context_block) had no "these"/
"those"/"ones" coverage at all.
"""

from src.agent.orchestrator.session import _FOLLOWUP_RE, _COUNT_FOLLOWUP_RE


def test_followup_re_matches_the_live_phrasing_and_named_variants():
    positives = [
        "how many of the ones listed above are completed",
        "how many of these are completed",
        "how many of those are completed",
        "which ones are completed",
        "show the open ones",
        "the ones listed above",
        "shown above",
    ]
    for q in positives:
        assert _FOLLOWUP_RE.search(q), f"expected _FOLLOWUP_RE to match: {q!r}"


def test_followup_re_still_excludes_the_documented_fresh_negatives():
    """Regression guard for the exclusion rationale this module's own
    comments document — broadening must not touch these."""
    negatives = [
        "mitigative actions that are overdue",
        "show me this month's data",
        "list all audit plans",
        "list all conducted audits",
        "how many audits were conducted last month",
    ]
    for q in negatives:
        assert not _FOLLOWUP_RE.search(q), f"expected _FOLLOWUP_RE NOT to match: {q!r}"


def test_count_followup_re_matches_conjunctive_phrasings_either_order():
    positives = [
        "how many of the ones listed above are completed",
        "how many of these are completed",
        "how many of those are completed",
        "of these 18 how many are closed",
        "what number of them are closed",
        "total count of those that are closed",
    ]
    for q in positives:
        assert _COUNT_FOLLOWUP_RE.search(q), f"expected _COUNT_FOLLOWUP_RE to match: {q!r}"


def test_count_followup_re_requires_both_halves():
    """Deliberately conjunctive — an aggregate cue alone or a backref cue
    alone must NOT match, keeping false positives near zero."""
    negatives = [
        "how many audits were conducted last month",
        "show me those",
        "list the ones from Abu Dhabi",
        "which ones are completed",
        "list all conducted audits",
    ]
    for q in negatives:
        assert not _COUNT_FOLLOWUP_RE.search(q), f"expected _COUNT_FOLLOWUP_RE NOT to match: {q!r}"
