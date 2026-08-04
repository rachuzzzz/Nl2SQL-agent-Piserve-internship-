"""
Regression test for a confirmed-live false-positive ordinal match in
session.py's _ORDINAL_RE: "First Party", "Second Party", and "Third party"
(and the hyphenated "Third-party") are real audit_type/inspection_type
values in the live DB — not ordinal references — but the ordinal-word list
("first", "second", "third", ...) matched the leading word of each of
these regardless.

Confirmed live 2026-08-04 (qid=57b076d1): "list all schedules under Second
party audit" matched "Second" as an ordinal follow-up, which would resolve
against an unrelated prior turn's list (e.g. row 2 of an audit_plan
listing) and inject a spurious "ORDINAL RESOLVED ... WHERE <alias>.id =
'<unrelated uuid>'" instruction plus a "Showing results for ..." disclosure
prefix, even though the question is a plain name filter with no ordinal
follow-up intent at all.

Confirmed against the live DB that this collision hits BOTH domains:
audit_type has 'First Party', 'Second Party', 'Third party / ISO  audit';
inspection_type has 'Third-party inspection'.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


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


def test_second_party_audit_does_not_trigger_ordinal_resolution():
    session = _session_after_plan_listing()
    ctx = session.build_context_block(
        current_question="list all schedules under Second party audit",
        current_domain="audit",
    )
    assert "ORDINAL" not in ctx.upper()
    assert "Showing results for" not in ctx


def test_first_party_does_not_trigger_ordinal_resolution():
    session = _session_after_plan_listing()
    ctx = session.build_context_block(
        current_question="First Party audits this year",
        current_domain="audit",
    )
    assert "ORDINAL" not in ctx.upper()


def test_third_party_slash_iso_does_not_trigger_ordinal_resolution():
    session = _session_after_plan_listing()
    ctx = session.build_context_block(
        current_question="show me Third party / ISO  audit schedules",
        current_domain="audit",
    )
    assert "ORDINAL" not in ctx.upper()


def test_hyphenated_third_party_inspection_does_not_trigger_ordinal_resolution():
    """The inspection-domain mirror of the same collision."""
    session = _session_after_plan_listing()
    ctx = session.build_context_block(
        current_question="how many third-party inspections were overdue",
        current_domain="inspection",
    )
    assert "ORDINAL" not in ctx.upper()


def test_legitimate_ordinal_follow_up_still_resolves():
    """The fix must not break genuine ordinal references — the whole point
    of the negative lookahead is to exclude ONLY the party-collision, not
    ordinal detection generally."""
    session = _session_after_plan_listing()
    ctx = session.build_context_block(
        current_question="show the approval status for the first one",
        current_domain="audit",
    )
    assert "ORDINAL RESOLVED" in ctx
    assert "aaaa0001-0000-0000-0000-000000000001" in ctx


def test_legitimate_numeric_ordinal_still_resolves():
    session = _session_after_plan_listing()
    ctx = session.build_context_block(
        current_question="show details for the 6th one",
        current_domain="audit",
    )
    assert "ORDINAL RESOLVED" in ctx
    assert "aaaa0006-0000-0000-0000-000000000006" in ctx
