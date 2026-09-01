"""
TurnClassifier — decides what kind of conversational turn the current message
is (fresh question vs. some kind of follow-up) and, for the follow-up kinds
where something was actually added/changed, extracts what changed as
structured fields.

Built the same way agent/domain_router.py's DomainRouter is: a small system
prompt, a fixed JSON output shape, and the same three-tier parse strategy
(json.loads -> json_repair -> regex extraction of just the enum fields) that
domain_router.py::_parse_domain_json already proves out in this codebase.

Replaces a bank of hand-authored regexes that used to live in session.py
(_FOLLOWUP_RE, _SELECTION_FOLLOWUP_RE, _COUNT_FOLLOWUP_RE, _FRESH_SUBJECT_RE,
_BACK_RE, the inline PAGINATION/REFINEMENT/AUGMENTATION gate) and loop.py
(_ORDINAL_BACKSTOP_RE, _PAGINATION_ONLY_RE/_HAS_NEW_SUBJECT_RE) — every one of
those had a comment citing a specific "confirmed live" regression it was
hand-tuned to fix. That doesn't scale: each new phrasing a real user tries
needs a developer to notice the miss and add another keyword. The regression
history those regexes encoded is preserved below, in the system prompt's
worked examples, so the classifier is primed on the same cases without a
per-case code change being required for the next one.

Output is two ORTHOGONAL fields, not one enum — `action` (what the user
wants done) and `reference` (how the target is identified) — because they
are genuinely independent questions and collapsing them into one enum causes
real misclassification. Confirmed live 2026-08-12 (debug_llm_20260812_104507,
qid=26c6d0f4): "list all schedules under the fourth one shown above" was
classified action=MODIFY_QUERY (reasonably — the user does want a new,
modified query), but with only one field there was nowhere to also record
that the target is identified BY ORDINAL POSITION. loop.py's ordinal backstop
checked action=="ORDINAL_REFERENCE", didn't fire, and the reasoning LLM
fabricated the 4th cycle's specific date values from its own (unreliable)
reading of the truncated conversation history instead of calling
resolve_ordinal_reference to get the real row id — exactly the
guess-instead-of-resolve failure mode the original _ORDINAL_BACKSTOP_RE
existed to prevent. `reference` now fires its backstop independently of
whatever `action` co-occurs with it.

Known regressions the old regex bank (and the single-enum first version of
this classifier) were built to catch, kept here as the classifier's own
worked-example knowledge instead of as separate patterns:
  - "the ones listed above are completed" (aggregate + backward reference,
    conjunctive: neither cue alone is enough) -> action=COUNT_FOLLOWUP.
  - "show along with auditees" -> action=MODIFY_QUERY, NOT a full query
    rewrite — only the added column/join belongs in `filters`/`entity`, the
    rest of the previous query's filters and join directions must be left
    alone.
  - "which cycle is currently active" after a facility was the active
    subject -> action=NEW_QUERY, not a follow-up — a stale scope must not
    leak into an unrelated global question just because a subject is still
    active.
  - "list all schedules under the fourth one shown above" ->
    reference=ORDINAL (action can be NEW_QUERY or MODIFY_QUERY — the point
    is reference alone must trigger resolve_ordinal_reference, regardless).
  - "for the second person show a breakdown of..." -> reference=ORDINAL,
    even though the head noun is "person", not "one". Confirmed live
    2026-08-12 (debug_llm_20260812_111917, qid=76947d25): the ORDINAL
    prompt's examples were all "...one"-shaped ("the 3rd one", "the first
    one", "the 6th auditor" — itself already a non-"one" noun, but every
    example still paired the ordinal directly with a domain noun already
    seen elsewhere in the prompt), and the classifier returned
    reference=NONE with a guessed entity value instead — the exact same
    reference/action collision the qid=26c6d0f4 fix above was built to
    prevent, just reopened by an under-specified prompt rather than a
    schema limitation. The reasoning LLM then fabricated a name
    ("neenu inspectee2" — a plausible-looking variant of row 1's REAL name,
    not row 2's actual name) instead of calling resolve_ordinal_reference,
    and the query never recovered (3 failed SQL-gen retries later, the
    existing "don't show an unverified answer" safety net correctly
    refused rather than showing wrong data — but the user got a failure
    after ~52s and ~87K prompt tokens for what should have been a normal,
    fast ordinal lookup). Fix: the ORDINAL prompt section now says
    explicitly that ANY noun after an ordinal word/number qualifies, not
    just "one" — a person, a facility, a plan, a row, an item.
  - "list all audits under this" with rows checked via UI checkboxes ->
    reference=SELECTION, even though "this" isn't followed by a noun.
  - "go back to that inspector" -> reference=BACK, re-anchoring to an
    earlier turn's subject rather than the most recent one.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

from src.agent.orchestrator.debug import _dbg
from src.llm.base import extract_token_counts, extract_llamacpp_timings, TokenUsage
from src.llm.prefix_cache import PrefixCacheManager
from src.llm.anthropic import build_cached_system_message

_log_turn = logging.getLogger("inspectai.turn_classifier")

_VALID_ACTIONS = frozenset({"NEW_QUERY", "MODIFY_QUERY", "PAGINATE", "COUNT_FOLLOWUP"})
_VALID_REFERENCES = frozenset({"NONE", "ORDINAL", "SELECTION", "BACK"})

# Only these actions carry real "what changed" content, and only when the
# target isn't ALSO being identified by position/selection/back-reference
# (reference == NONE) — when a reference is present, the actual entity
# resolution happens via the resolve_* tool call downstream (LLM-supplied
# position/pick as a structured tool arg), so entity/filters here would be
# redundant or, worse, actively misleading (see module docstring's
# qid=26c6d0f4 regression: an "entity.value" carrying an ordinal position
# like "4" instead of a real name/id looks like an extracted fact but isn't
# one — it's exactly the kind of guess the resolve_* tools exist to avoid).
_ACTIONS_WITH_CHANGE_FIELDS = frozenset({"MODIFY_QUERY", "COUNT_FOLLOWUP"})


@dataclass
class TurnAction:
    action: str = "NEW_QUERY"
    reference: str = "NONE"
    entity: dict = field(default_factory=dict)   # {"type": "inspector", "value": "Kavya"}
    filters: dict = field(default_factory=dict)  # {"status": "COMPLETED"}
    time_range: str = ""
    confidence: float = 0.0


_CLASSIFIER_SYSTEM_PROMPT = """\
You are a turn classifier for a multi-turn database chat assistant covering \
facility inspections and audits. Given the conversation state so far and the \
user's CURRENT MESSAGE, decide TWO independent things: what the user wants \
done (action), and how the target of their message is identified (reference) \
— these are separate questions and a message can have any combination of the \
two, e.g. "list schedules under the fourth cycle" is action=NEW_QUERY (a new \
table/subject) with reference=ORDINAL (the cycle is picked by position).

Respond with ONLY a single JSON object, no other text, no markdown fences:
{"action": "<one of the ACTION categories below>",
 "reference": "<one of the REFERENCE categories below>",
 "entity": {"type": "<e.g. inspector, facility, inspection, audit, plan>", "value": "<name or id text>"},
 "filters": {"<short filter description>": "<value>"},
 "time_range": "<free text time range, or empty string>",
 "confidence": <0.0-1.0>}

Only populate "entity"/"filters"/"time_range" when action is MODIFY_QUERY or \
COUNT_FOLLOWUP AND reference is NONE — leave them as {} / {} / "" otherwise. \
In particular, do NOT put an ordinal position, "the selected ones", or a \
prior-turn label into "entity" — that belongs in "reference" instead, since \
resolving it is a separate tool's job, not something to guess here.

ACTION — what the user wants done:

- NEW_QUERY — a fresh question, or one asking about a genuinely different \
  table/subject than the previous turn(s), even if it reuses something from \
  earlier context (e.g. via an ORDINAL/SELECTION/BACK reference — see below). \
  Example: after several turns scoped to one facility, "which cycle is \
  currently active" is NEW_QUERY, not a follow-up about that facility — a \
  global question must never inherit a stale scope just because something is \
  still "active".

- MODIFY_QUERY — the user wants to filter, sort, or add to the SAME previous \
  query, or is asking an ordinary pronoun-style follow-up about the current \
  active subject ("how many inspections did they do", "what about last \
  month"). Extract what changed into entity/filters/time_range (only when \
  reference is NONE). Example: "show along with auditees" is MODIFY_QUERY \
  with filters describing ONLY the added column — it does not mean \
  re-deriving the whole query from scratch, and every existing filter/join \
  direction from the previous query must be assumed unchanged unless the \
  user says otherwise.

- PAGINATE — the user wants to see more of the exact same previous result \
  and nothing else — "show more", "show the rest", "continue", "next page". \
  If the message ALSO introduces any new subject, filter, or condition \
  alongside "show more"/"continue"-type language, that is MODIFY_QUERY, not \
  PAGINATE — PAGINATE means literally nothing changed except how much of the \
  same result is shown.

- COUNT_FOLLOWUP — the user wants to count/aggregate over the previous \
  result (which may only be a capped sample of a much larger total). \
  Example: "how many of the ones listed above are completed" — this needs \
  BOTH an aggregate request (how many / count / percentage) AND a backward \
  reference (these / those / the above) to qualify; a bare "how many audits \
  were conducted last month" with no backward reference is NEW_QUERY instead.

REFERENCE — how the target is identified (independent of action above; use \
NONE unless one of the other three clearly applies):

- NONE — the target isn't identified via position, UI selection, or an \
  earlier superseded subject (the default for most messages).

- ORDINAL — the user refers to a specific item BY ITS POSITION in a list \
  shown earlier, REGARDLESS of what noun follows the ordinal word/number — \
  "the 3rd one", "the first one", "the 6th auditor", "under the fourth one \
  shown above", "for the second person", "the 2nd facility", "the third \
  plan". A number/ordinal word (first/second/third/2nd/3rd/...) immediately \
  before ANY noun describing something from a previous list — a person, a \
  facility, a plan, an item, a row — is this category; do not require the \
  specific word "one" to recognize it. This applies EVEN IF the rest of the \
  message asks for something new about that item (e.g. a different table) \
  — the ordinal identification is what matters here, not the action.

- SELECTION — the user refers to row(s) they explicitly selected via UI \
  checkboxes (only plausible when the conversation state says rows are \
  currently selected). Example: "list all audits under this" with rows \
  checked — even a bare "this"/"that" with nothing else can mean the \
  selection here, not a request for clarification.

- BACK — the user explicitly wants to return to an earlier subject that's \
  since been superseded by more recent turns: "go back to that inspector", \
  "back to the facility we discussed earlier".
"""


def _parse_turn_json(raw: str) -> Optional[dict]:
    """Same "try json.loads, then json_repair, then regex" strategy as
    domain_router.py::_parse_domain_json, adapted to this classifier's
    {"action": ..., "reference": ..., "entity": ..., "filters": ...,
    "time_range": ..., "confidence": ...} shape instead of the router's
    {"domain": ...}."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    if _HAS_JSON_REPAIR:
        try:
            repaired = json_repair.repair_json(candidate, return_objects=True)
            objs = repaired if isinstance(repaired, list) else [repaired]
            for obj in objs:
                if isinstance(obj, dict) and "action" in obj:
                    return obj
        except Exception:
            pass

    # Last-resort regex extraction, same degrade-gracefully precedent as
    # domain_router.py — pulls both enum fields independently since either
    # can be present without the other in a badly-mangled response.
    m_action = re.search(r'"action"\s*:\s*"(' + "|".join(_VALID_ACTIONS) + r')"', text, re.IGNORECASE)
    m_ref = re.search(r'"reference"\s*:\s*"(' + "|".join(_VALID_REFERENCES) + r')"', text, re.IGNORECASE)
    if m_action or m_ref:
        out = {}
        if m_action:
            out["action"] = m_action.group(1).upper()
        if m_ref:
            out["reference"] = m_ref.group(1).upper()
        return out
    return None


class TurnClassifier:
    def __init__(self, reasoning_llm=None, debug_logger=None, domain: str = ""):
        # reasoning_llm: the SAME warm client AgentOrchestrator uses for
        # planning (see agent/orchestrator/loop.py's self.reasoning_llm) and
        # DomainRouter reuses for its own Tier-3 — passed in explicitly, not
        # built here, so classification costs one extra short completion
        # rather than a model swap. None is valid (e.g. unit tests) — the
        # classifier then falls back to NEW_QUERY/NONE instead of raising.
        self.reasoning_llm = reasoning_llm
        self._debug_logger = debug_logger
        # role="turn_classifier", not "reasoning" — this call sends a
        # completely different, much shorter system prompt
        # (_CLASSIFIER_SYSTEM_PROMPT) than the main loop's own reasoning-role
        # calls (self._system_prompt in loop.py, the one actually warmed via
        # AgentOrchestrator's own PrefixCacheManager). A distinct role label
        # here means [PREFIX-CACHE] log lines can show whether THIS call is
        # cold/warm separately from the main loop's — previously invisible,
        # since this class never called .record() at all. Reuses the SAME
        # llm_client the main loop warms (self.reasoning_llm), so a cold
        # prefill here is itself evidence worth having: it's what you'd
        # expect if this call's differently-shaped prompt is evicting the
        # main loop's own warmed prefix from Ollama's slot cache every turn
        # (see the TurnClassifier section of the token/caching audit) —
        # record() alone doesn't warm anything or fix that, it only makes
        # the pattern visible in logs so it can be measured before acting on it.
        self._prefix_cache = PrefixCacheManager(reasoning_llm, domain=domain, role="turn_classifier")

    def _render_state(self, session) -> str:
        lines = ["CONVERSATION STATE:"]

        turns = session.turns
        lines.append(f"Recent turns (most recent last, {len(turns)} of up to {session.MAX_TURNS} kept):")
        for i, t in enumerate(turns, 1):
            _subj = (
                f" [active_subject: {t.active_subject_type}='{t.active_subject_label or t.active_subject_value}']"
                if t.active_subject_type and t.active_subject_value else ""
            )
            _q = t.question[:200] + ("…" if len(t.question) > 200 else "")
            _a = t.answer[:150] + ("…" if len(t.answer) > 150 else "")
            lines.append(f"  Turn {i} [domain={t.domain or '?'}]: User: {_q!r} -> {_a!r}{_subj}")

        if session.current_scope:
            cs = session.current_scope
            lines.append(
                f"Current active scope: {cs['type']} = '{cs.get('label') or cs.get('value')}' "
                f"(domain={cs.get('domain') or '?'})"
            )
        else:
            lines.append("Current active scope: (none)")

        if session.entity_memory:
            _mem = [f"{e['type']} '{e.get('label') or e.get('value')}'" for e in session.entity_memory]
            lines.append(f"Entities mentioned earlier this session (oldest to newest): {_mem}")

        if session.pending_selected_positions:
            lines.append(
                f"The user has {len(session.pending_selected_positions)} row(s) currently "
                f"SELECTED via checkboxes in the UI from the last shown result."
            )

        last = turns[-1] if turns else None
        if last and last.sql:
            _sql_snippet = " ".join(last.sql.split())[:400]
            lines.append(f"Last query's SQL: {_sql_snippet}")
            lines.append(f"Last query's total row count: {last.total_count}")

        return "\n".join(lines)

    def classify(self, question: str, session=None, stats=None, qid: str = "") -> TurnAction:
        # stats: the SAME QueryStats instance AgentOrchestrator.query() passes
        # to its own _call_llm (see synthesis.py) — without this, this call's
        # tokens/latency/count were silently invisible to stats.total_llm_calls/
        # total_prompt_tokens/total_completion_tokens/total_llm_latency_ms
        # (confirmed live 2026-08-12: the debug log's own [LLM] iter=1 latency=
        # line, and the [QUERY] END tokens=X/Y summary, both undercounted by
        # however much this call cost, because DomainRouter's Tier-3 — the
        # precedent this was modeled on — legitimately runs OUTSIDE query()'s
        # stats entirely, but this call runs INSIDE it and was never wired up
        # the same way _call_llm's other callers are). Mirrors
        # synthesis.py::_call_llm's accumulation exactly, just inlined here
        # since TurnClassifier has no access to that bound method.
        if session is None or not session.turns:
            # Nothing to be a follow-up to yet.
            return TurnAction(action="NEW_QUERY", reference="NONE", confidence=1.0)

        if self.reasoning_llm is None:
            return TurnAction(action="NEW_QUERY", reference="NONE", confidence=0.0)

        from llama_index.core.llms import ChatMessage, MessageRole
        state_block = self._render_state(session)
        user_content = f"{state_block}\n\nCURRENT MESSAGE: {question}"
        # _CLASSIFIER_SYSTEM_PROMPT is fully static (a module constant, no
        # per-turn content) — no separate dynamic tail to keep out of the
        # cached region, unlike loop.py's system message. Still a no-op on
        # every non-Anthropic client (see build_cached_system_message).
        messages = [
            build_cached_system_message(self.reasoning_llm, _CLASSIFIER_SYSTEM_PROMPT, ""),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ]
        _dbg(self._debug_logger, "TURN CLASSIFIER PROMPT", f"{_CLASSIFIER_SYSTEM_PROMPT}\n\n{user_content}")
        _t0 = time.time()
        try:
            response = self.reasoning_llm.chat(messages)
            raw = response.message.content.strip()
        except Exception as exc:
            _log_turn.warning("qid=%s [TURN CLASSIFIER] LLM call failed: %s", qid, exc)
            _dbg(self._debug_logger, "TURN CLASSIFIER EXCEPTION", str(exc))
            return TurnAction(action="NEW_QUERY", reference="NONE", confidence=0.0)
        _latency_ms = int((time.time() - _t0) * 1000)

        _usage = TokenUsage()
        if stats is not None:
            stats.total_llm_calls += 1
            stats.total_llm_latency_ms += _latency_ms
            _raw_usage = getattr(response, "raw", {}) or {}
            _usage = extract_token_counts(_raw_usage)
            stats.total_prompt_tokens += _usage.prompt_tokens
            stats.total_completion_tokens += _usage.completion_tokens
            stats.total_cache_read_tokens += _usage.cache_read_tokens
            stats.total_cache_creation_tokens += _usage.cache_creation_tokens
            self._prefix_cache.record(_raw_usage)
            _t = extract_llamacpp_timings(_raw_usage)
            stats.reasoning_prefill_ms += int(_t.prompt_ms)
            stats.reasoning_decode_ms += int(_t.predicted_ms)

        _dbg(self._debug_logger, "TURN CLASSIFIER RAW OUTPUT", raw or "(empty)")

        parsed = _parse_turn_json(raw)
        if not parsed:
            _log_turn.warning("qid=%s [TURN CLASSIFIER] could not parse LLM output: %r", qid, raw[:200])
            return TurnAction(action="NEW_QUERY", reference="NONE", confidence=0.0)

        action = str(parsed.get("action", "NEW_QUERY")).upper()
        if action not in _VALID_ACTIONS:
            action = "NEW_QUERY"

        reference = str(parsed.get("reference", "NONE")).upper()
        if reference not in _VALID_REFERENCES:
            reference = "NONE"

        entity = parsed.get("entity")
        entity = entity if isinstance(entity, dict) else {}
        filters = parsed.get("filters")
        filters = filters if isinstance(filters, dict) else {}
        time_range = str(parsed.get("time_range", "") or "")

        if action not in _ACTIONS_WITH_CHANGE_FIELDS or reference != "NONE":
            entity, filters, time_range = {}, {}, ""

        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        _log_turn.info(
            "qid=%s [TURN CLASSIFIER] action=%s reference=%s confidence=%.2f "
            "latency=%dms prompt_tok=%d completion_tok=%d cache_read=%d cache_write=%d question=%r",
            qid, action, reference, confidence, _latency_ms,
            _usage.prompt_tokens, _usage.completion_tokens,
            _usage.cache_read_tokens, _usage.cache_creation_tokens,
            question[:120],
        )
        return TurnAction(
            action=action, reference=reference, entity=entity, filters=filters,
            time_range=time_range, confidence=confidence,
        )
