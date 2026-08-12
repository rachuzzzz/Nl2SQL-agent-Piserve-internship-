"""
Regression test for loop.py's LIMIT-preservation hint regex
(the `limit_match` pattern inside the generate_sql branch) — found to have
a real false-positive bug during Phase 5's evaluation: the requested-N
digit was matched identically whether it named a row count ("the last 5
inspections") or a date window ("the last 30 days"), silently injecting a
bogus "use LIMIT 30" hint into a pure date-filter question.

Fix: a negative lookahead excluding time-unit nouns immediately after the
digit in the first regex alternative (the second alternative was never
reachable through a date phrase to begin with — it requires the digit to
be followed by a specific count-shaped noun, none of which are time units).
"""

import re

_LIMIT_RE_1 = re.compile(
    r'\b(?:last|top|first|recent|latest)\s+(\d+)'
    r'(?!\s*(?:days?|weeks?|months?|hours?|minutes?|years?)\b)\b',
    re.IGNORECASE,
)
_LIMIT_RE_2 = re.compile(
    r'\b(\d+)\s+(?:most\s+recent|latest|last|inspections?|actions?|reports?)\b',
    re.IGNORECASE,
)


def _extract(question: str) -> str | None:
    m = _LIMIT_RE_1.search(question) or _LIMIT_RE_2.search(question)
    return m.group(1) if m else None


def test_date_window_phrases_do_not_trigger_a_limit_hint():
    assert _extract("show inspections from the last 30 days") is None
    assert _extract("inspections in the last 7 days") is None
    assert _extract("last 90 days of audit reports") is None
    assert _extract("last 2 weeks of data") is None
    assert _extract("last 12 months") is None
    assert _extract("show the last 6 hours") is None
    assert _extract("last 45 minutes") is None
    assert _extract("last 3 years") is None


def test_row_count_phrases_still_trigger_a_limit_hint():
    assert _extract("show the last 5 inspections") == "5"
    assert _extract("top 10 facilities") == "10"
    assert _extract("the first 3 audits") == "3"
    assert _extract("most recent 20 reports") == "20"
    assert _extract("show 3 reports") == "3"
    assert _extract("the 5 most recent audits") == "5"
    assert _extract("first 20 results") == "20"


def test_no_match_when_no_count_or_date_phrase_present():
    assert _extract("list all audit plans") is None
    assert _extract("how many inspections are overdue") is None
