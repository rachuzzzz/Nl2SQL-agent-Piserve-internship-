"""
DomainRouter — 3-tier hybrid router deciding whether a question belongs to
the "inspection" or "audit" domain, so api/routes/chat.py can pick the right
AgentOrchestrator instance (see agent/domains.py's DOMAIN_REGISTRY).

Tiers, first that resolves wins:
  Tier 1 — keyword match (instant, no LLM call). Most real questions carry an
    unambiguous domain word ("inspection"/"inspector"/"mitigative action" vs
    "audit"/"auditor"/"corrective action"), so most turns cost nothing extra.
  Tier 2 — session stickiness (instant, no LLM call). No domain word at all —
    stays in whichever domain the conversation is already in. Makes
    follow-ups ("show the overdue ones") feel natural without re-asking.
  Tier 3 — LLM judge (~3-4s, rare). Only when there's no keyword AND no
    conversation yet — reuses the already-warm reasoning_llm client (one
    extra short completion, not a model swap) rather than a dedicated router
    model. Returns AMBIGUOUS + a clarifying question when it can't tell.

This supersedes the earlier docx design (docs/Domain_Router_Plan.docx §1),
which called the LLM once per message unconditionally — Tiers 1/2 make the
common case free.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

from src.agent.orchestrator.debug import _dbg

_log_router = logging.getLogger("inspectai.router")


# ── Tier 1 — keyword sets ────────────────────────────────────────────────────
# Deliberately narrow and high-precision: these words essentially never appear
# in the *other* domain's real questions. "corrective action" is the audit
# term where inspection says "mitigative action" — same underlying table
# shape, different vocabulary per domain, per the user's worked example.
#
# Verb forms (audited/auditing, inspected/inspecting, bare "inspect"/"audit")
# are included alongside the noun forms — confirmed live this isn't
# optional: a session sticky to AUDIT, asked "who inspected this ADJD",
# fell through Tier 1 (no noun-form match), landed on Tier 2 stickiness
# (AUDIT), and the audit-domain LLM — which has no inspection concept at
# all — silently reinterpreted "inspected" as "audited" and returned a real
# auditor name with success=True. A wrong-but-confident answer, not an
# error, and nothing would have caught it. The verb forms below close that
# gap so a domain word of EITHER conjugation always wins Tier 1 regardless
# of whatever the session is currently sticky to.
_AUDIT_KEYWORDS_RE = re.compile(
    r"\baudit(?:s|or|ors|ee|ees|ed|ing)?\b|\bcorrective\s+actions?\b",
    re.IGNORECASE,
)
_INSPECTION_KEYWORDS_RE = re.compile(
    r"\binspect(?:ion|ions|or|ors|ed|ing)?\b|\bmitigative\s+actions?\b",
    re.IGNORECASE,
)


@dataclass
class DomainRoute:
    domain: str                       # "INSPECTION" | "AUDIT" | "AMBIGUOUS"
    tier: int                         # 1, 2, or 3 — which tier resolved it (debugging/logging)
    clarifying_question: str = ""     # populated only when domain == "AMBIGUOUS"
    confidence: float = 1.0


_ROUTER_SYSTEM_PROMPT = """\
You are a domain classifier for a database chat assistant that serves two \
separate domains from the same chat surface: INSPECTION (facility \
inspections, inspectors, mitigative actions) and AUDIT (audits, auditors, \
audit corrective actions, audit plans/schedules/reports).

Given the user's message, decide which domain it belongs to. If it could \
reasonably belong to either domain and there's no way to tell which one the \
user means, respond AMBIGUOUS with a short, friendly one-sentence \
clarifying question asking the user to specify.

Respond with ONLY a single JSON object, no other text, no markdown fences:
{"domain": "INSPECTION" | "AUDIT" | "AMBIGUOUS", "clarifying_question": "<string, only when AMBIGUOUS, else \"\">", "confidence": <0.0-1.0>}
"""


def _parse_domain_json(raw: str) -> Optional[dict]:
    """Robust JSON extraction for the router's small, fixed output shape —
    same "try json.loads, then json_repair" strategy as
    agent/tool_dispatcher.py's _try_parse, but not routed through that
    function directly since it validates against the tool-call schema
    ({"tool": ..., "args": ...}), a different shape than the router's
    {"domain": ..., "clarifying_question": ..., "confidence": ...}."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "domain" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    if _HAS_JSON_REPAIR:
        try:
            repaired = json_repair.repair_json(candidate, return_objects=True)
            objs = repaired if isinstance(repaired, list) else [repaired]
            for obj in objs:
                if isinstance(obj, dict) and "domain" in obj:
                    return obj
        except Exception:
            pass

    # Last resort: pull just the domain value out with a regex — matches the
    # tool_dispatcher.py precedent of degrading gracefully rather than
    # raising when the model's JSON is unrecoverable.
    m = re.search(r'"domain"\s*:\s*"(INSPECTION|AUDIT|AMBIGUOUS)"', text, re.IGNORECASE)
    if m:
        return {"domain": m.group(1).upper(), "clarifying_question": "", "confidence": 0.5}
    return None


class DomainRouter:
    def __init__(self, reasoning_llm=None, debug_logger=None):
        # reasoning_llm: the SAME warm client AgentOrchestrator uses for
        # planning (see agent/orchestrator/loop.py's self.reasoning_llm) —
        # passed in explicitly, not built here, so Tier 3 costs one extra
        # short completion rather than a model swap. None is valid (e.g.
        # unit tests exercising Tiers 1/2 only) — Tier 3 then falls back to
        # AMBIGUOUS with a generic clarifying question instead of raising.
        self.reasoning_llm = reasoning_llm
        # debug_logger: the same debug_llm_*.log logger AgentOrchestrator
        # uses (agent/orchestrator/debug.py's _make_debug_logger/_dbg) — pass
        # one of the orchestrators' self._debug in so Tier-3's prompt/raw
        # output land in the same file instead of being invisible beyond a
        # terse WARNING on parse failure (see api/app.py's lifespan).
        self._debug_logger = debug_logger

    @classmethod
    def from_env(cls, reasoning_llm=None, debug_logger=None) -> "DomainRouter":
        return cls(reasoning_llm=reasoning_llm, debug_logger=debug_logger)

    def classify(self, question: str, session=None) -> DomainRoute:
        # ── Tier 1 — keyword (instant) ──────────────────────────────────────
        audit_hit = bool(_AUDIT_KEYWORDS_RE.search(question or ""))
        inspection_hit = bool(_INSPECTION_KEYWORDS_RE.search(question or ""))

        if audit_hit and not inspection_hit:
            return DomainRoute(domain="AUDIT", tier=1, confidence=1.0)
        if inspection_hit and not audit_hit:
            return DomainRoute(domain="INSPECTION", tier=1, confidence=1.0)
        # Both fire (e.g. "compare audit corrective actions vs mitigative
        # actions") or neither fires — not a hard Tier-1 signal either way,
        # fall through to stickiness/LLM rather than guess.

        # ── Tier 2 — session stickiness (instant) ───────────────────────────
        if session is not None:
            sticky = getattr(session, "last_domain", "")
            if sticky:
                return DomainRoute(domain=sticky.upper(), tier=2, confidence=1.0)

        # ── Tier 3 — LLM judge (rare: no keyword AND no conversation yet) ──
        if self.reasoning_llm is None:
            return DomainRoute(
                domain="AMBIGUOUS", tier=3, confidence=0.0,
                clarifying_question=(
                    "Could you clarify — is this about facility inspections or audits?"
                ),
            )

        from llama_index.core.llms import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_ROUTER_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=question),
        ]
        _dbg(self._debug_logger, "ROUTER TIER-3 PROMPT", f"{_ROUTER_SYSTEM_PROMPT}\n\nUSER: {question}")
        try:
            response = self.reasoning_llm.chat(messages)
            raw = response.message.content.strip()
        except Exception as exc:
            _log_router.warning("[ROUTER] Tier-3 LLM call failed: %s", exc)
            _dbg(self._debug_logger, "ROUTER TIER-3 EXCEPTION", str(exc))
            return DomainRoute(
                domain="AMBIGUOUS", tier=3, confidence=0.0,
                clarifying_question=(
                    "Could you clarify — is this about facility inspections or audits?"
                ),
            )

        _dbg(self._debug_logger, "ROUTER TIER-3 RAW OUTPUT", raw or "(empty)")

        parsed = _parse_domain_json(raw)
        if not parsed:
            _log_router.warning("[ROUTER] Tier-3 could not parse LLM output: %r", raw[:200])
            return DomainRoute(
                domain="AMBIGUOUS", tier=3, confidence=0.0,
                clarifying_question=(
                    "Could you clarify — is this about facility inspections or audits?"
                ),
            )

        domain = str(parsed.get("domain", "AMBIGUOUS")).upper()
        if domain not in ("INSPECTION", "AUDIT", "AMBIGUOUS"):
            domain = "AMBIGUOUS"
        clarifying = str(parsed.get("clarifying_question", "") or "")
        if domain == "AMBIGUOUS" and not clarifying:
            clarifying = "Could you clarify — is this about facility inspections or audits?"
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        _log_router.info(
            "[ROUTER] tier=3 domain=%s confidence=%.2f question=%r",
            domain, confidence, question[:120],
        )
        return DomainRoute(domain=domain, tier=3, confidence=confidence, clarifying_question=clarifying)
