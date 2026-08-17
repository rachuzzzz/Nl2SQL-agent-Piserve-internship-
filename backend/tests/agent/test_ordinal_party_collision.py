"""
Regression test for a confirmed-live false-positive ordinal match:
"First Party", "Second Party", and "Third party" (and the hyphenated
"Third-party") are real audit_type/inspection_type values in the live DB —
not ordinal references — but the ordinal-word list ("first", "second",
"third", ...) matched the leading word of each of these regardless.

Confirmed live 2026-08-04 (qid=57b076d1): "list all schedules under Second
party audit" matched "Second" as an ordinal follow-up, which used to
resolve against an unrelated prior turn's list and inject a spurious
resolved-id instruction, even though the question is a plain name filter
with no ordinal follow-up intent at all.

Confirmed against the live DB that this collision hits BOTH domains:
audit_type has 'First Party', 'Second Party', 'Third party / ISO  audit';
inspection_type has 'Third-party inspection'.

Ported onto the new architecture: the party/parties exclusion now lives in
loop.py's _ORDINAL_BACKSTOP_RE — a pure DETECTOR deciding whether to
redirect the LLM toward calling resolve_ordinal_reference, not a resolver
itself (resolve_ordinal_reference never sees free question text at all,
so it has no party-collision surface of its own — the LLM supplies a
clean `position` int directly). The "legitimate ordinal still resolves"
half of the original test is ported onto resolve_ordinal_reference
directly, confirming the fix didn't also suppress genuine references.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn
from src.agent.orchestrator.ordinal import resolve_ordinal_reference
from src.agent.orchestrator.loop import _ORDINAL_BACKSTOP_RE


def _session_after_plan_listing() -> ConversationSession:
    """Mirrors the real 'list all audit plans' turn preceding the live bug —
    any prior turn with row_titles is enough to make ordinal resolution
    attempt to fire."""
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="list all audit plans",
        answer="25 plans found",
        tool_used="generate_sql",
        total_count=25,
        columns=["title", "status"],
        row_titles=[f"Plan {i}" for i in range(1, 26)],
        row_ids=[f"aaaa000{i}-0000-0000-0000-00000000000{i}" for i in range(1, 26)],
        row_id_columns=["id"] * 25,
    ))
    return session


def test_second_party_audit_does_not_trigger_backstop():
    assert _ORDINAL_BACKSTOP_RE.search("list all schedules under Second party audit") is None


def test_first_party_does_not_trigger_backstop():
    assert _ORDINAL_BACKSTOP_RE.search("First Party audits this year") is None


def test_third_party_slash_iso_does_not_trigger_backstop():
    assert _ORDINAL_BACKSTOP_RE.search("show me Third party / ISO  audit schedules") is None


def test_hyphenated_third_party_inspection_does_not_trigger_backstop():
    """The inspection-domain mirror of the same collision."""
    assert _ORDINAL_BACKSTOP_RE.search("how many third-party inspections were overdue") is None


def test_legitimate_ordinal_phrase_still_triggers_backstop():
    """The fix must not break genuine ordinal detection — the negative
    lookahead excludes ONLY the party-collision, not ordinal words
    generally."""
    assert _ORDINAL_BACKSTOP_RE.search("show the approval status for the first one") is not None
    assert _ORDINAL_BACKSTOP_RE.search("show details for the 6th one") is not None


def test_legitimate_ordinal_follow_up_still_resolves():
    session = _session_after_plan_listing()
    r = resolve_ordinal_reference(session, "audit", position=1, entity_type_hint=None)
    assert r["result"]["target_id"] == "aaaa0001-0000-0000-0000-000000000001"


def test_legitimate_numeric_ordinal_still_resolves():
    session = _session_after_plan_listing()
    r = resolve_ordinal_reference(session, "audit", position=6, entity_type_hint=None)
    assert r["result"]["target_id"] == "aaaa0006-0000-0000-0000-000000000006"
