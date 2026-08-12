"""
Regression test for chat.py's bare-domain-pick short-circuit.

Confirmed live: after a bare greeting ("Hi") triggers the router's
AMBIGUOUS clarifying question ("audits or inspections?"), replying with
just "audit" does NOT qualify for the existing origin-question carryover
(see test_domain_redirect.py's test_resume_conditions_match_a_redirect_
domain_turn) — "Hi" isn't a real question worth resuming — so the literal
word "audit" fell through to being sent to the reasoning model as if it
were an actual data question. The model's judgment on what to do with a
bare domain word wasn't consistent: one real session got a fast clarifying
question (~5s, 1 LLM call), another got an unrequested default "recent
audit reports" query (~16.5s, 3 LLM calls, a validation retry) for the
exact same input.

The fix: after computing _effective_message (the origin-question swap, if
any, already applied), if it's STILL just a bare domain word, chat.py
short-circuits with a deterministic acknowledgment instead of calling the
reasoning model at all. This test verifies the exact boolean condition
that decision is built on — same style as test_domain_redirect.py's
test_resume_conditions_match_a_redirect_domain_turn, which verifies
chat.py's logic without spinning up the full FastAPI route.
"""

from src.agent.orchestrator.models import ConversationTurn
from src.api.routes.chat import _BARE_DOMAIN_PICK_RE, _GREETING_RE


def _effective_message(prev: ConversationTurn | None, reply: str) -> str:
    """Mirrors chat.py's exact _effective_message computation."""
    origin_is_real_question = bool(
        prev and not _GREETING_RE.match(prev.question.strip())
        and len(prev.question.strip().split()) >= 2
    )
    is_pending_domain_pick = bool(
        prev and prev.tool_used == "domain_clarification"
        and _BARE_DOMAIN_PICK_RE.match(reply.strip())
        and origin_is_real_question
    )
    return prev.question if is_pending_domain_pick else reply


def test_bare_domain_reply_after_a_greeting_short_circuits():
    """"Hi" is never a real question to resume — the bare "audit" reply
    that follows it must still look bare after the swap, so chat.py's new
    short-circuit fires instead of sending it to the reasoning model."""
    prev = ConversationTurn(
        question="Hi", answer="Are you looking for information about facility inspections or audits?",
        tool_used="domain_clarification",
    )
    eff = _effective_message(prev, "audit")
    assert eff == "audit"
    assert _BARE_DOMAIN_PICK_RE.match(eff.strip())


def test_bare_domain_reply_with_no_prior_turn_at_all_short_circuits():
    """A session's very first message being just "audit" — no prior turn
    to even consider — must also short-circuit."""
    eff = _effective_message(None, "audit")
    assert eff == "audit"
    assert _BARE_DOMAIN_PICK_RE.match(eff.strip())


def test_bare_domain_reply_after_a_real_question_does_not_short_circuit():
    """The existing carryover case (test_domain_redirect.py's core
    scenario) — "show all schedules" IS a real question, so the swap
    replaces "audit" with it, and the result must NOT look like a bare
    domain pick anymore."""
    prev = ConversationTurn(
        question="show all schedules",
        answer="Are you asking about audit schedules or inspection schedules?",
        tool_used="domain_clarification",
    )
    eff = _effective_message(prev, "audit")
    assert eff == "show all schedules"
    assert not _BARE_DOMAIN_PICK_RE.match(eff.strip())


def test_ordinary_bare_domain_word_after_a_normal_completed_turn_still_short_circuits():
    """Not just the domain-clarification-reply case — a user typing just
    "audit" any time (e.g. mid-conversation, switching topics) with
    nothing that looks like an unresolved question to resume should also
    short-circuit, since _effective_message falls back to the literal
    reply itself in every case other than a genuine pending-pick."""
    prev = ConversationTurn(
        question="how many inspections happened last month", answer="42 inspections.",
        tool_used="execute_sql", domain="inspection",
    )
    eff = _effective_message(prev, "audit")
    assert eff == "audit"
    assert _BARE_DOMAIN_PICK_RE.match(eff.strip())
