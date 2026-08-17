"""
resolve_selected_rows — deterministic resolution of checkbox row selections
made in the UI (see agent/tool_defs/row_selection.py) to real row ids,
mirroring resolve_ordinal_reference's per-index lookup (ordinal.py) but
generalized to a LIST of positions and with stricter safety rules, since a
checkbox selection is structured input (not a word the LLM has to
interpret) and a wrong resolution here would silently scope an aggregate
answer to the wrong rows rather than just misname one item.

Deliberate deviations from resolve_ordinal_reference, see inline comments:
  1. Always targets session.turns[-1] directly — never searches backward.
  2. Any single invalid/unresolvable/ambiguous position hard-fails the
     WHOLE selection (needs_clarification) rather than resolving the good
     ones and dropping or guessing the bad one.
"""

from src.agent.orchestrator.ordinal import _table_for_entity_type


def _needs_clarification(message: str) -> dict:
    return {
        "success": True,
        "result": {
            "resolved": False, "target_table": "", "target_column": "",
            "target_ids": [], "display_titles": [], "positions": [],
            "entity_type": "", "needs_clarification": True,
            "clarification_message": message,
        },
        "error": None,
    }


def resolve_selected_rows(session, domain, **_ignored) -> dict:
    """
    session: ConversationSession (live object, threaded in via the tool's
        context_provider — see agent/tool_defs/row_selection.py).
    domain: "inspection" | "audit".
    Reads session.pending_selected_positions directly (NOT an LLM-supplied
    arg) — routes/chat.py sets it fresh from the request before the
    reasoning loop starts, so the LLM never has to guess/re-derive which
    rows were checked; it only decides WHETHER this question refers to
    them and calls this tool with no arguments if so.
    """
    positions = list(getattr(session, "pending_selected_positions", None) or [])
    if not positions:
        return _needs_clarification(
            "No rows are currently selected in the UI — ask the user to "
            "check the rows they mean, or resolve this some other way "
            "(e.g. resolve_ordinal_reference, or a fresh generate_sql search)."
        )

    if not session or not session.turns:
        return _needs_clarification(
            "There's no earlier result in this conversation to resolve the "
            "selected rows against — ask the user what list they mean, or "
            "list it first."
        )

    # Target the LAST turn ONLY — unlike resolve_ordinal_reference's
    # backward search (appropriate for a freeform NL reference, which could
    # genuinely mean an older list), a checkbox selection is physically tied
    # to whichever table is currently rendered on screen, which is always
    # the most recent turn. Falling back to an older turn on a miss would
    # silently resolve the user's clicks against the wrong table.
    target_turn = session.turns[-1]
    if not target_turn.row_titles:
        return _needs_clarification(
            "The latest result in this conversation has no selectable rows "
            "recorded — do not resolve this selection against an earlier "
            "list; ask the user to clarify or re-run their list."
        )

    n_rows = len(target_turn.row_titles)
    invalid = [p for p in positions if not isinstance(p, int) or p < 1 or p > n_rows]
    if invalid:
        return _needs_clarification(
            f"Position(s) {invalid} are out of range — the latest result "
            f"only has {n_rows} row(s). Do NOT resolve the valid positions "
            f"and drop the invalid ones; ask the user to clarify."
        )

    items = []
    for position in positions:
        idx = position - 1
        title = target_turn.row_titles[idx]
        row_id = target_turn.row_ids[idx] if idx < len(target_turn.row_ids) else ""
        id_column = (
            target_turn.row_id_columns[idx]
            if idx < len(target_turn.row_id_columns) else "id"
        )
        items.append({
            "position": position, "title": title,
            "id": row_id or "", "id_column": id_column or "id",
        })

    # Missing target_id (e.g. a GROUP BY turn with no identity injection)
    # — no per-item title fallback here, unlike ordinal.py's single-pick
    # case. A title IN (...) filter across a multi-row selection risks
    # silently pulling in unchecked sibling rows sharing that title (see
    # the duplicate check right below) — worse than ordinal's "ambiguous
    # WHICH row" case, since it's silent over-inclusion into an aggregate.
    missing_id = [it["position"] for it in items if not it["id"]]
    if missing_id:
        return _needs_clarification(
            f"Position(s) {missing_id} have no resolvable row id in the "
            f"latest result (likely a grouped/aggregated listing) — do NOT "
            f"fall back to filtering by display title/name for this "
            f"selection. Ask the user to clarify, or re-run a listing that "
            f"returns individual rows."
        )

    # Duplicate-title NOTE — warn, don't block. Every item here already has
    # a real resolved id (checked above), so WHERE id IN (...) is exact
    # regardless of a shared display value — unlike ordinal.py's single-pick
    # case (where a duplicate title is a genuine "which one?" ambiguity),
    # here the id already disambiguates every selected row individually.
    # Confirmed live: this is NOT a rare edge case — audit_corrective_action
    # rows commonly share their displayed corrective_action_id (the
    # CA_DRAFT-placeholder trap documented extensively in sql/rules/
    # audit_rules.py is one instance of a broader pattern of non-unique
    # display identifiers in this schema) — hard-failing here made ordinary
    # multi-row corrective-action selections unusable. A prior version of
    # this function hard-failed on this instead; that broke the exact
    # "select a few corrective actions, ask about them" flow this tool
    # exists for.
    duplicate_title_positions = [
        it["position"] for it in items
        if target_turn.row_titles.count(it["title"]) > 1
    ] or None

    entity_type = target_turn.row_entity_type or ""
    target_column = items[0]["id_column"] if items else "id"

    return {
        "success": True,
        "result": {
            "resolved": True,
            "target_table": _table_for_entity_type(entity_type, domain),
            "target_column": target_column,
            "target_ids": [it["id"] for it in items],
            "display_titles": [it["title"] for it in items],
            "positions": [it["position"] for it in items],
            "entity_type": entity_type,
            "duplicate_title_positions": duplicate_title_positions,
            "needs_clarification": False,
            "clarification_message": None,
        },
        "error": None,
    }
