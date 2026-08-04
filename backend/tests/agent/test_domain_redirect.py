"""
Regression tests for the domain-mismatch redirect mechanism — the fix for a
confirmed-live bug (2026-08-03): a user typo'd "mittigative actions" instead
of "mitigative actions" (the INSPECTION-domain term) mid-conversation in the
AUDIT domain. domain_router.py's Tier 1 keyword regex requires an exact
match, so the typo missed it; with no Tier 1 signal, the router fell back to
Tier 2 (session.last_domain stickiness), which silently kept the
conversation in AUDIT, and the AUDIT reasoning model reinterpreted
"mitigative actions" as "corrective actions" — wrong domain, no error,
nothing caught it.

Fix: each domain's own system prompt is now aware of the OTHER domain's
vocabulary, framed as a semantic-confidence check rather than a keyword
lookup (so it also catches genuine synonyms like "remedial actions", which
matches neither domain's literal keyword list). When genuinely uncertain,
the model calls a new `redirect_domain` tool instead of guessing —
loop.py tags the resulting turn identically to the router's own Tier-3
AMBIGUOUS outcome (tool_used="domain_clarification", domain=""), so
chat.py's existing "ask once, resume from the reply" mechanism picks it up
completely unchanged.
"""

import json

from src.agent.orchestrator.loop import AgentOrchestrator
from src.agent.orchestrator.session import ConversationSession
from src.agent.tool_dispatcher import _KNOWN_TOOLS, parse_tool_call
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.prompts.builder_audit import SYSTEM_PROMPT_AUDIT
from src.api.routes.chat import _BARE_DOMAIN_PICK_RE, _GREETING_RE


def _fake_orchestrator(domain: str, canned_tool_call: dict) -> AgentOrchestrator:
    """A minimal AgentOrchestrator that bypasses __init__ (no DB/embedding/
    Ollama connections) — just enough state for query() to reach the
    redirect_domain branch, with _call_llm stubbed to return a canned
    tool-call JSON instead of hitting a real reasoning LLM."""
    fake = object.__new__(AgentOrchestrator)
    fake.domain = domain
    fake._system_prompt = "SYSTEM PROMPT"
    fake._builder_module = type(
        "B", (), {"build_dynamic_prompt_additions": staticmethod(lambda q, score_fn=None: "")}
    )()
    fake.embed_model = None
    fake._scope_primary_table = "audit_corrective_action"
    fake._scope_primary_alias = "aca"
    fake._debug = None
    _raw = json.dumps(canned_tool_call)
    fake._call_llm = lambda messages, stats, iteration=None: _raw
    return fake


def test_redirect_domain_produces_clarification_tagged_turn():
    """The exact live-caught bug: the reasoning model is uncertain and calls
    redirect_domain instead of guessing. Verify the resulting turn is tagged
    identically to the router's own Tier-3 AMBIGUOUS turn."""
    orch = _fake_orchestrator("audit", {
        "thought": "Wording is generic, could mean either domain.",
        "tool": "redirect_domain",
        "args": {
            "suspected_domain": "INSPECTION",
            "answer": "Just to confirm - are you asking about audits or facility inspections?",
        },
    })
    session = ConversationSession()
    result = orch.query(
        "show all mittigative actions that are overdue",
        session=session, visibility_scope={},
    )
    assert result.success is True
    assert result.answer == "Just to confirm - are you asking about audits or facility inspections?"
    assert len(session.turns) == 1
    assert session.turns[-1].tool_used == "domain_clarification"
    # domain="" (not self.domain="audit") — deliberate, so Tier-2 stickiness
    # on the NEXT ambiguous follow-up doesn't stick right back to the domain
    # that just admitted uncertainty.
    assert session.turns[-1].domain == ""


def test_redirect_domain_answer_is_not_mangled_by_final_answer_repair():
    """Regression for a bug caught during implementation: tool_dispatcher.py's
    _reconstruct_fragmented_answer assumes ANY key besides "answer" is
    corruption debris to merge back into the answer text — true for
    final_answer (which only ever has one meaningful arg), but WRONG for
    redirect_domain, which legitimately always has a second key
    (suspected_domain). Applying that repair to redirect_domain would
    mangle "suspected_domain: INSPECTION" onto the end of every single
    well-formed answer. Must not happen."""
    orch = _fake_orchestrator("audit", {
        "thought": "...",
        "tool": "redirect_domain",
        "args": {"suspected_domain": "INSPECTION", "answer": "Which domain did you mean?"},
    })
    session = ConversationSession()
    result = orch.query("remedial actions overdue", session=session, visibility_scope={})
    assert result.answer == "Which domain did you mean?"
    assert "suspected_domain" not in result.answer


def test_redirect_domain_in_known_tools_for_regex_fallback_recovery():
    """_KNOWN_TOOLS feeds parse_tool_call's Step-4 regex-fallback recovery,
    used on every malformed/truncated LLM response — without this,
    redirect_domain calls that get truncated mid-JSON would be unrecoverable."""
    assert "redirect_domain" in _KNOWN_TOOLS


def test_parse_tool_call_recovers_truncated_redirect_domain():
    """End-to-end check of the regex-fallback path itself: a truncated
    redirect_domain call (missing closing braces, as Qwen sometimes emits)
    must still resolve to the right tool name."""
    truncated = (
        '{"thought": "uncertain", "tool": "redirect_domain", '
        '"args": {"suspected_domain": "AUDIT", "answer": "which domain'
    )
    parsed = parse_tool_call(truncated)
    assert parsed["tool"] == "redirect_domain"


def test_both_system_prompts_document_redirect_domain():
    """Cheap static check, no LLM/Ollama needed — catches a template-loading
    regression on every run."""
    assert "redirect_domain" in SYSTEM_PROMPT
    assert "redirect_domain" in SYSTEM_PROMPT_AUDIT
    assert "DOMAIN-MISMATCH CHECK" in SYSTEM_PROMPT
    assert "DOMAIN-MISMATCH CHECK" in SYSTEM_PROMPT_AUDIT


def test_resume_conditions_match_a_redirect_domain_turn():
    """Verifies chat.py's EXISTING resume-from-reply logic (unchanged by
    this feature) recognizes a turn produced by loop.py's new redirect_domain
    block exactly as it would a router-originated AMBIGUOUS turn — both are
    plain ConversationTurn field checks with no knowledge of which code path
    set them, so tagging the new turn with the identical
    tool_used="domain_clarification" literal is what makes this work with
    zero chat.py changes."""
    from src.agent.orchestrator.models import ConversationTurn

    prev = ConversationTurn(
        question="show all mittigative actions that are overdue",
        answer="Just to confirm - are you asking about audits or facility inspections?",
        tool_used="domain_clarification",
    )
    reply = "Inspection"

    origin_is_real_question = bool(
        prev and not _GREETING_RE.match(prev.question.strip())
        and len(prev.question.strip().split()) >= 2
    )
    is_pending_domain_pick = bool(
        prev and prev.tool_used == "domain_clarification"
        and _BARE_DOMAIN_PICK_RE.match(reply.strip())
        and origin_is_real_question
    )
    assert origin_is_real_question is True
    assert is_pending_domain_pick is True
    # This is exactly chat.py's _effective_message swap (line 237) — the
    # ORIGINAL mismatched question gets replayed, not the bare "Inspection".
    effective_message = prev.question if is_pending_domain_pick else reply
    assert effective_message == "show all mittigative actions that are overdue"
