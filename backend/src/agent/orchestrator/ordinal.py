"""
resolve_ordinal_reference — deterministic resolution of "the first one"/
"the 3rd item"-style follow-up references to a specific prior turn's row,
given an explicit 1-based `position` and optional `entity_type_hint`
supplied by the reasoning LLM itself (via the resolve_ordinal_reference
tool — see agent/tool_defs/ordinal.py) rather than re-derived by regex
from raw question text.

Ported from ConversationSession.build_context_block()'s former ordinal
block (session.py, prompt-injection era, retired — see that file's
history) — same underlying data (ConversationTurn.row_titles/row_ids/
row_id_columns/row_entity_type, populated by synthesis.py's turn-finish
extraction) and the same duplicate-title trap, restructured around a
clean int position instead of a regex match object. The regex's own
trailing-noun-capture group mis-scanning past the ordinal word into the
next clause (misclassifying "the first one show all corrective actions
raised" as being ABOUT corrective actions, not the true antecedent type)
was the actual bug this refactor exists to remove — an LLM-supplied
position has no text left to mis-scan.
"""

from src.agent.orchestrator.disambiguation import entity_type_for_ordinal_noun, _entity_noun_tables


# entity_type_for_ordinal_noun's regex table (disambiguation.py's
# _entity_noun_tables) was built to scan a TAIL of raw question text,
# where a domain-qualifying word naturally sits next to the noun ("the
# 6th AUDIT PLAN"). A short, LLM-supplied entity_type_hint has no such
# surrounding context — an LLM asked for "the first one" naming what kind
# of item it means will naturally say just "plan", not "audit plan", and
# several of the underlying regexes (audit_plan/audit_report/audit_cycle/
# audit_forms/inspection_report) require that domain prefix and silently
# return "" for the bare noun alone. Confirmed live while wiring this
# tool: passing entity_type_hint="plan" in the audit domain fell through
# to bare-ordinal resolution every time — the type check was silently a
# no-op. This alias table covers the bare nouns an LLM will actually send;
# entity_type_for_ordinal_noun remains the fallback for anything phrased
# more like a sentence fragment.
_AUDIT_HINT_ALIASES = {
    "plan": "plan", "audit plan": "plan",
    "report": "audit", "audit report": "audit", "audit": "audit",
    "cycle": "cycle", "audit cycle": "cycle",
    "form": "form", "audit form": "form",
    "schedule": "schedule", "audit schedule": "schedule",
    "corrective action": "corrective_action", "corrective_action": "corrective_action",
    "portfolio": "portfolio", "auditor portfolio": "portfolio",
    "auditor": "person", "inspector": "person", "person": "person",
}
_INSPECTION_HINT_ALIASES = {
    "inspection": "inspection", "inspection report": "inspection",
    "corrective action": "corrective_action", "corrective_action": "corrective_action",
    "mitigative action": "corrective_action",
    "schedule": "schedule", "inspection schedule": "schedule",
    "portfolio": "portfolio", "inspector portfolio": "portfolio",
    "cycle": "cycle", "inspection cycle": "cycle",
    "auditor": "person", "inspector": "person", "person": "person",
}

# Substring fallback, ordered most-specific first — catches snake_case
# table-name-style hints the schema context in the system prompt naturally
# primes an LLM to echo back (confirmed live: entity_type_hint="audit_plan"
# for "list all audit plans" -> "for the first one..."). Low-risk: this
# only ever WIDENS what resolves via the type-checked path instead of
# silently falling to the untyped bare-ordinal fallback — it can never
# cause a wrong-type match (a real mismatch still returns "" and the
# caller's needs_clarification/bare-ordinal handling is unchanged), only
# a missed one, same failure direction as no match at all.
_AUDIT_HINT_SUBSTRINGS = (
    ("corrective action", "corrective_action"),
    ("auditor portfolio", "portfolio"), ("portfolio", "portfolio"),
    ("plan", "plan"), ("schedule", "schedule"), ("cycle", "cycle"),
    ("form", "form"), ("report", "audit"),
    ("auditor", "person"), ("inspector", "person"),
)
_INSPECTION_HINT_SUBSTRINGS = (
    ("corrective action", "corrective_action"), ("mitigative action", "corrective_action"),
    ("inspector portfolio", "portfolio"), ("portfolio", "portfolio"),
    ("schedule", "schedule"), ("cycle", "cycle"),
    ("inspection", "inspection"),
    ("auditor", "person"), ("inspector", "person"),
)


def _normalize_entity_type_hint(hint: str, domain: str) -> str:
    if not hint:
        return ""
    hint_norm = hint.strip().lower().replace("_", " ")
    hint_norm = " ".join(hint_norm.split())  # collapse repeated whitespace
    aliases = _AUDIT_HINT_ALIASES if domain == "audit" else _INSPECTION_HINT_ALIASES
    if hint_norm in aliases:
        return aliases[hint_norm]
    substrings = _AUDIT_HINT_SUBSTRINGS if domain == "audit" else _INSPECTION_HINT_SUBSTRINGS
    for needle, etype in substrings:
        if needle in hint_norm:
            return etype
    return entity_type_for_ordinal_noun(hint, domain)


def _ord_suffix(idx0: int) -> str:
    pos = idx0 + 1
    if 10 <= pos % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(pos % 10, "th")


def _table_for_entity_type(entity_type: str, domain: str) -> str:
    if not entity_type:
        return ""
    for _pat, table, etype in _entity_noun_tables(domain):
        if etype == entity_type and table:
            return table
    return ""


def _needs_clarification(position: int, entity_type: str, message: str) -> dict:
    return {
        "success": True,
        "result": {
            "resolved": False, "target_table": "", "target_column": "",
            "target_id": "", "display_title": "", "position": position,
            "entity_type": entity_type, "warning": None,
            "needs_clarification": True, "clarification_message": message,
        },
        "error": None,
    }


def resolve_ordinal_reference(session, domain, position, entity_type_hint=None) -> dict:
    """
    session: ConversationSession (live object, threaded in via the tool's
        context_provider — see agent/tool_defs/ordinal.py's context_provider).
    domain: "inspection" | "audit".
    position: 1-based index into the target turn's row list (the LLM
        already converts "first"/"the 2nd"/"the 15th" to this integer —
        no ordinal-word parsing happens here).
    entity_type_hint: optional short noun the LLM already extracted from
        the question ("plan", "corrective action", "auditor", ...) — fed
        through the same noun -> entity-type vocabulary used elsewhere in
        this app (disambiguation.py's entity_type_for_ordinal_noun) for
        consistency, rather than re-implemented here. None/unrecognized
        => bare-ordinal resolution: most recent turn with any row_titles,
        regardless of type — identical semantics to the retired
        session.py block's fallback.
    """
    try:
        position = int(position)
    except (TypeError, ValueError):
        return {"success": False, "result": None,
                "error": f"position must be an integer, got {position!r}"}
    if position < 1:
        return {"success": False, "result": None,
                "error": f"position must be >= 1 (1-based), got {position}"}
    idx = position - 1

    if not session or not session.turns:
        return _needs_clarification(
            position, "",
            "There's no earlier result in this conversation to resolve a "
            "positional reference against — ask the user what list they "
            "mean, or list it first.",
        )

    entity_type = _normalize_entity_type_hint(str(entity_type_hint), domain) if entity_type_hint else ""

    if entity_type:
        target_turn = next(
            (t for t in reversed(session.turns)
             if t.row_titles and t.row_entity_type == entity_type),
            None,
        )
        if target_turn is None:
            return _needs_clarification(
                position, entity_type,
                f"The question refers to an ordinal position among "
                f"'{entity_type}'-type items, but no earlier turn in this "
                f"conversation actually listed items of that type. Do NOT "
                f"resolve this against any OTHER list — it is a different "
                f"kind of item, and guessing would silently answer the wrong "
                f"question. Ask the user to first show/list the "
                f"'{entity_type}' items they mean.",
            )
    else:
        target_turn = next((t for t in reversed(session.turns) if t.row_titles), None)
        if target_turn is None:
            return _needs_clarification(
                position, "",
                "No earlier turn in this conversation listed multiple rows "
                "to resolve a positional reference against — ask the user "
                "what list they mean.",
            )

    if idx >= len(target_turn.row_titles):
        return _needs_clarification(
            position, entity_type or target_turn.row_entity_type,
            f"Position {position} is out of range — the most recent "
            f"matching list only has {len(target_turn.row_titles)} item(s).",
        )

    display_title = target_turn.row_titles[idx]

    # DUPLICATE-WITHIN-LIST TRAP — confirmed live: a single listing can
    # contain the SAME display title twice for two genuinely different rows
    # (e.g. two audit_plan rows sharing an identical title). Resolving to
    # this title string and filtering follow-up SQL by WHERE ... = title
    # (or ILIKE) would silently match BOTH and merge two different records'
    # data into one answer — flagged here regardless of whether a real id
    # was resolved below, since it's the caller's job to prefer the id.
    warning = None
    if target_turn.row_titles.count(display_title) > 1:
        warning = (
            f"'{display_title}' appears MORE THAN ONCE in that same list "
            f"(not a data error — distinct rows can share an identical "
            f"title/name). Filtering by WHERE ... = '{display_title}' alone "
            f"would match ALL of them and silently merge multiple different "
            f"records into one answer."
        )

    target_id = ""
    target_column = ""
    if idx < len(target_turn.row_ids):
        target_id = target_turn.row_ids[idx] or ""
        if target_id:
            target_column = "id"
            if idx < len(target_turn.row_id_columns):
                target_column = target_turn.row_id_columns[idx] or "id"

    resolved_entity_type = entity_type or target_turn.row_entity_type or ""

    session.pending_resolved_disclosure = (
        f"Showing results for {display_title!r} "
        f"({position}{_ord_suffix(idx)} item in your previous list):"
    )

    return {
        "success": True,
        "result": {
            "resolved": True,
            "target_table": _table_for_entity_type(resolved_entity_type, domain),
            "target_column": target_column,
            "target_id": target_id,
            "display_title": display_title,
            "position": position,
            "entity_type": resolved_entity_type,
            "warning": warning,
            "needs_clarification": False,
            "clarification_message": None,
        },
        "error": None,
    }
