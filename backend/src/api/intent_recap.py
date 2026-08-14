"""
Intent recap: a short, plain-English "here's what I understood" sentence
shown alongside every chat answer (see api/schemas.py's ChatResponse.
intent_recap) — built from a SEPARATE, minimal LLM call, not the reasoning
loop's own internal "thought" (that's full of internal tool-planning
jargon unsuited for non-technical end users — see chat.py's `reasoning`
field, which still carries the raw thought for potential future debug use
but is never shown to end users).

start_intent_recap() is called BEFORE orch.query() runs and
resolve_intent_recap() AFTER it returns (see routes/chat.py) so the recap
call's latency (~1-3s for a tiny prompt) overlaps with the main
reasoning/SQL-gen/execution loop instead of adding to it — by the time
chat.py needs the result, the background call has usually already finished.
"""

import concurrent.futures
import logging
import re

_log_query = logging.getLogger("inspectai.query")

# Small, dedicated pool — this module's calls are tiny (a few hundred
# tokens in, one sentence out), never the ~9K-token reasoning prompt, so a
# handful of workers is plenty even under several concurrent chat requests.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="intent-recap")

_RESOLVE_TIMEOUT_S = 2.0

# Confirmed live (2026-08-06): the raw LLM paraphrase sometimes came back
# as content unrelated to a "here's what I understood" recap — the small
# reasoning model, given a bare instruction to paraphrase, would instead
# try to ANSWER the question (numbers, "there are X", a snippet of SQL) or
# ramble well past one sentence. Both are signs it ignored the "paraphrase,
# don't answer" instruction rather than a signal the paraphrase is simply
# unusual wording — caught here and routed to the deterministic template
# instead of shown to the user as-is.
_LOOKS_LIKE_AN_ANSWER_RE = re.compile(
    r"\b(SELECT\s|WHERE\s|FROM\s|COUNT\s*\(|there\s+(is|are)\s+\d|"
    r"\d+\s+(rows?|results?|records?)|total\s+(of\s+)?\d)\b",
    re.IGNORECASE,
)
_MAX_RECAP_CHARS = 220  # a genuine one-sentence recap never needs more than this

# Second failure mode, confirmed live 2026-08-12 (debug_llm_20260812_104507,
# qid=76afb7b8): the one-shot example below used REAL, plausible domain
# vocabulary ("Golf Gardens", "mitigative actions") — for the actual
# question "can you show them along with their respective observations"
# (an audit-domain corrective-actions turn with no facility ever mentioned),
# the completion came back "You're asking to display the overdue mitigative
# actions for Golf Gardens alongside their corresponding observations" — the
# model latched onto the example's entities instead of the real question's.
# Because they're realistic, on-topic terms, _LOOKS_LIKE_AN_ANSWER_RE and the
# length check both waved this straight through and it was shown to the user
# as-is. _EXAMPLE_SENTINEL_RE below catches any leak of the example's own
# fixed wording (the fix in _run() also swaps the example to deliberately
# unrealistic placeholder terms, so a genuine leak is now both far less
# likely to happen AND, if it does, obviously wrong rather than a plausible
# but false-sounding recap).
_EXAMPLE_SENTINEL_RE = re.compile(r"\bzylotech\b|\bwidget\s+cycle\b", re.IGNORECASE)


def _looks_like_a_real_paraphrase(text: str) -> bool:
    return (
        bool(text)
        and len(text) <= _MAX_RECAP_CHARS
        and not _LOOKS_LIKE_AN_ANSWER_RE.search(text)
        and not _EXAMPLE_SENTINEL_RE.search(text)
    )


def _template_recap(question: str, session=None) -> str:
    """Deterministic, zero-latency fallback — used when the LLM paraphrase
    call errors or doesn't finish within _RESOLVE_TIMEOUT_S. This is the
    exact wording that would have shipped had the simpler "no LLM call"
    option been chosen instead, so it's a genuine reliability net, not a
    degraded experience."""
    q_clean = (question or "").strip().rstrip("?.!")
    if not q_clean:
        return ""
    last_turn = getattr(session, "last_turn", None) if session else None
    label = getattr(last_turn, "active_subject_label", "") if last_turn else ""
    if label:
        return f"Here's what I understood: {q_clean} — focusing on {label}."
    return f"Here's what I understood: {q_clean}."


_RECAP_SYSTEM_PROMPT = """\
You paraphrase a user's database-chat question into ONE short, plain-English \
sentence describing what they're asking for. Do NOT answer the question — no \
numbers, no SQL, no row counts, just a description of what they want. \
Respond with ONLY the paraphrase sentence — no "Paraphrase:" label, no \
restating the question, no explanation.

Example (illustrative only — ignore its subject matter; your paraphrase must \
describe the REAL question given below, never the example's):
Question: how many widgets are overdue in the current widget cycle for Zylotech
Paraphrase: You're asking how many overdue widgets there are in the current cycle for Zylotech.
"""


def start_intent_recap(reasoning_llm, question: str, session=None) -> concurrent.futures.Future:
    """Fire the recap LLM call in a background thread and return the Future
    immediately (non-blocking) — call this BEFORE orch.query(), not after,
    so it runs concurrently with the expensive SQL-gen/execution loop
    instead of adding its own latency on top of it."""
    def _run() -> str:
        last_turn = getattr(session, "last_turn", None) if session else None
        context = f"Previous question in this conversation: {last_turn.question!r}\n" if last_turn else ""

        # Structured SYSTEM/USER chat call — matches TurnClassifier
        # (orchestrator/turn_classifier.py) and DomainRouter
        # (agent/domain_router.py), the other two call sites sharing this
        # same reasoning_llm client. Previously used raw .complete(prompt)
        # with the one-shot example and the real question concatenated into
        # one undifferentiated text blob for the model to continue —
        # confirmed live 2026-08-12 (qid=76afb7b8) that a real-sounding
        # example ("Golf Gardens", "mitigative actions") leaked into the
        # paraphrase for an unrelated question. A raw completion prompt
        # gives the model no structural signal that the example is an
        # *example* rather than literal prior context to keep continuing
        # from; a SYSTEM-role instruction does. The example itself still
        # uses deliberately unrealistic placeholder entities ("Zylotech",
        # "widget cycle") from that same fix, and _EXAMPLE_SENTINEL_RE below
        # still guards against a leak either way — this is a second,
        # independent layer addressing the more likely root cause, not a
        # replacement for that backstop.
        from llama_index.core.llms import ChatMessage, MessageRole
        user_content = f"{context}Question: {question}"
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_RECAP_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ]
        response = reasoning_llm.chat(messages)
        raw = response.message.content.strip()
        # Defensive strip — some models echo the prefix/label despite the
        # system instruction above; harmless either way since this is just
        # cleanup, not the mechanism the instruction relies on.
        raw = re.sub(r'^(paraphrase|question)\s*:\s*', '', raw, flags=re.IGNORECASE).strip()
        # Take only the first line/sentence-ish chunk — a rambling
        # completion that runs on past one sentence is exactly the "looks
        # like it ignored the instruction" signal _looks_like_a_real_
        # paraphrase checks for below, so trimming here doesn't hide a
        # real problem, it just avoids showing 3 paragraphs when the
        # model's first line was actually fine.
        raw = raw.splitlines()[0].strip() if raw else raw
        _log_query.debug("[INTENT-RECAP] raw model output: %r", raw)
        return raw

    return _EXECUTOR.submit(_run)


def resolve_intent_recap(
    future: concurrent.futures.Future | None, question: str, session=None,
    timeout: float = _RESOLVE_TIMEOUT_S,
) -> str | None:
    """Join the Future started by start_intent_recap().

    A resolved ordinal/disambiguation disclosure
    (session.pending_resolved_disclosure, set inside orch.query() — see
    ordinal.py) always takes priority over the generic paraphrase for this
    turn: its wording ("Showing results for X...") is already clean and
    specific — this is just its new home, previously shown inline in the
    answer text by synthesis.py's _synthesize().

    Falls back to the deterministic template on timeout or any error — the
    recap is never worth blocking or failing the whole chat response over.
    """
    disclosure = getattr(session, "pending_resolved_disclosure", "") if session else ""
    if disclosure:
        return disclosure
    if future is None:
        return _template_recap(question, session) or None
    try:
        raw = future.result(timeout=timeout)
        if not _looks_like_a_real_paraphrase(raw):
            _log_query.debug(
                "[INTENT-RECAP] model output failed the paraphrase sanity check "
                "(looked like an answer, or too long) — falling back to template: %r",
                raw,
            )
            return _template_recap(question, session) or None
        # The fixed prefix is always supplied here, never trusted from the
        # model (see start_intent_recap's docstring) — guarantees
        # consistent wording regardless of what the completion contains.
        return f"Here's what I understood: {raw}"
    except Exception as exc:
        _log_query.debug("[INTENT-RECAP] falling back to template — %s", exc)
        return _template_recap(question, session) or None
