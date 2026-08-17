"""
Unit tests for api/intent_recap.py — the "here's what I understood" recap
shown alongside every chat answer (ChatResponse.intent_recap). Covers the
behaviors that matter for reliability: an ordinal/disambiguation disclosure
always wins over the generic paraphrase, a failed/slow LLM call falls back
to the deterministic template rather than blocking or erroring the
response, a model completion that looks like it answered the question
instead of paraphrasing it (or rambled on too long) is caught and replaced
with the template rather than shown to the user as-is, and the template
itself reads sensibly with/without a resolved active-subject label.

Contract: the Future started by start_intent_recap() resolves to the RAW
paraphrase text only (e.g. "how many inspections happened this month.") —
resolve_intent_recap() always supplies the "Here's what I understood: "
prefix itself, never trusts the model to include it (see intent_recap.py's
docstring for why: a bare "start your sentence with X" instruction proved
unreliable with the small reasoning model used here).
"""

import concurrent.futures

from src.api.intent_recap import resolve_intent_recap, _template_recap, _looks_like_a_real_paraphrase
from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


def _done_future(value=None, exc=None) -> concurrent.futures.Future:
    fut: concurrent.futures.Future = concurrent.futures.Future()
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(value)
    return fut


def test_pending_resolved_disclosure_takes_priority_over_future():
    session = ConversationSession()
    session.pending_resolved_disclosure = "Showing results for 'Golf Gardens' (3rd item in your previous list):"
    fut = _done_future("Here's what I understood: something else entirely.")

    recap = resolve_intent_recap(fut, "the 3rd one", session)

    assert recap == "Showing results for 'Golf Gardens' (3rd item in your previous list):"


def test_successful_future_result_is_used_when_no_disclosure_pending():
    """The Future resolves to raw paraphrase text (no prefix) — resolve_
    intent_recap supplies "Here's what I understood: " itself."""
    session = ConversationSession()
    fut = _done_future("you're asking how many inspections happened this month.")

    recap = resolve_intent_recap(fut, "how many inspections this month", session)

    assert recap == "Here's what I understood: you're asking how many inspections happened this month."


def test_model_output_that_looks_like_an_answer_falls_back_to_template():
    """Confirmed live: the small reasoning model sometimes ANSWERED the
    question instead of paraphrasing it (numbers, row counts, SQL-flavored
    phrasing) — this must be caught and replaced with the template, not
    shown to the user as an unrelated "recap"."""
    session = ConversationSession()
    fut = _done_future("There are 42 rows matching SELECT COUNT(*) FROM inspection_report WHERE ...")

    recap = resolve_intent_recap(fut, "how many inspections happened", session)

    assert recap == "Here's what I understood: how many inspections happened."


def test_model_output_that_rambles_past_one_sentence_falls_back_to_template():
    session = ConversationSession()
    rambling = "Well, this is an interesting question. " * 10
    fut = _done_future(rambling)

    recap = resolve_intent_recap(fut, "how many inspections happened", session)

    assert recap == "Here's what I understood: how many inspections happened."


def test_looks_like_a_real_paraphrase_accepts_clean_short_text():
    assert _looks_like_a_real_paraphrase("you're asking about overdue mitigative actions.") is True


def test_looks_like_a_real_paraphrase_rejects_empty_or_answer_shaped_text():
    assert _looks_like_a_real_paraphrase("") is False
    assert _looks_like_a_real_paraphrase("there are 5 results") is False
    assert _looks_like_a_real_paraphrase("SELECT * FROM inspection_report") is False


def test_future_exception_falls_back_to_template():
    session = ConversationSession()
    fut = _done_future(exc=RuntimeError("LLM call failed"))

    recap = resolve_intent_recap(fut, "how many inspections happened", session)

    assert recap == "Here's what I understood: how many inspections happened."


def test_future_timeout_falls_back_to_template():
    """A Future that never completes (still running past the resolve
    timeout) must not block/fail the response — falls back immediately per
    the short timeout, same as an outright error."""
    session = ConversationSession()
    never_done: concurrent.futures.Future = concurrent.futures.Future()  # no result set

    recap = resolve_intent_recap(never_done, "how many inspections happened", session, timeout=0.05)

    assert recap == "Here's what I understood: how many inspections happened."


def test_none_future_falls_back_to_template_directly():
    session = ConversationSession()

    recap = resolve_intent_recap(None, "how many inspections happened", session)

    assert recap == "Here's what I understood: how many inspections happened."


def test_template_includes_active_subject_label_when_present():
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="show me Golf Gardens", answer="...", tool_used="execute_sql",
        active_subject_type="facility", active_subject_value="Golf Gardens",
        active_subject_label="Golf Gardens",
    ))

    recap = _template_recap("how many inspections there", session)

    assert recap == "Here's what I understood: how many inspections there — focusing on Golf Gardens."


def test_template_handles_empty_question():
    assert _template_recap("", None) == ""
    assert _template_recap("   ", None) == ""
