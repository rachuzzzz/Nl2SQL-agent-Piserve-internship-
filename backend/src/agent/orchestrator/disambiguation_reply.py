"""
resolve_disambiguation_reply — deterministic resolution of a user's reply
to a pending "which one did you mean?" clarification (people OR entity
disambiguation), given an explicit `picked_name` and/or `picked_position`
supplied by the reasoning LLM itself (via the resolve_disambiguation_reply
tool — see agent/tool_defs/disambiguation_reply.py) rather than re-derived
by regex from raw question text.

Ported from ConversationSession.build_context_block()'s former two
disambiguation-reply blocks (session.py, prompt-injection era, retired —
see that file's history) — same underlying data (ConversationTurn.
disambiguation_name/disambiguation_candidates/disambiguation_origin_question
for people; entity_disambiguation_candidates/_table/_alias for entities)
and the same ambiguous-duplicate-name STOP safety behavior, restructured
around a clean LLM-supplied pick instead of regex matching against raw
question text. Because the LLM already reads the reply and extracts a
clean name (or position) itself, the old query-verb gate, filler-word
stripping, and substring-containment fallback tier all become unnecessary
— there is no leftover sentence text left to fuzzy-match against; only
an EXACT match against candidate names is needed.

`_find_exact_candidate()` is also reused directly by synthesis.py's
automatic active_subject_* fallback (a DIFFERENT consumer — see that
file's `_finish()` — which runs post-hoc with no LLM decision point
involved, so it can't itself become a tool call, but must not silently
diverge from this module's matching rule again the way it previously did
with its own independent exact-whole-question-equality implementation).
"""


def _find_exact_candidate(candidates: list[dict], name_field: str, text: str) -> list[dict]:
    if not text:
        return []
    needle = text.strip().lower()
    return [c for c in candidates if c.get(name_field, "").strip().lower() == needle]


def _needs_clarification(kind: str | None, message: str) -> dict:
    return {
        "success": True,
        "result": {
            "resolved": False, "kind": kind or "", "target_table": "", "target_alias": "",
            "target_column": "", "target_id": "", "display_name": "", "total_candidates": 0,
            "picked_via": "", "warning": None,
            "needs_clarification": True, "clarification_message": message,
        },
        "error": None,
    }


def resolve_disambiguation_reply(session, picked_name: str | None = None,
                                  picked_position: int | None = None) -> dict:
    """
    session: ConversationSession (live object, threaded in via the tool's
        context_provider — see agent/tool_defs/disambiguation_reply.py).
    picked_name: the specific name the LLM extracted from the user's reply
        as their intended pick, exactly as shown in the previous candidate
        list (e.g. "George Smith"). Matched by EXACT equality only.
    picked_position: 1-based position in the previously-shown candidate
        list ("option 2", "the second one"). Resolves directly — unambiguous
        even when two candidates share a display name, since position
        indexes the row, not the (possibly duplicated) name.
    """
    if picked_name is None and picked_position is None:
        return {"success": False, "result": None,
                "error": "resolve_disambiguation_reply requires picked_name or picked_position."}

    if not session or not session.turns:
        return _needs_clarification(
            None,
            "There's no pending disambiguation in this conversation to resolve — "
            "do not call this tool unless the previous turn asked which one you meant.",
        )

    last = session.turns[-1]

    if last.disambiguation_name:
        kind = "person"
        candidates = last.disambiguation_candidates
        id_field, name_field = "user_id", "name"
        target_table, target_alias = "users", "u"
        origin_question = last.disambiguation_origin_question
        searched_name = last.disambiguation_name
    elif last.entity_disambiguation_candidates:
        kind = "entity"
        candidates = last.entity_disambiguation_candidates
        id_field, name_field = "id", "name"
        target_table = last.entity_disambiguation_table or "entity"
        target_alias = last.entity_disambiguation_alias or "e"
        origin_question = ""
        searched_name = ""
    else:
        return _needs_clarification(
            None,
            "There's no pending disambiguation on the previous turn to resolve — "
            "do not call this tool unless the previous turn asked which one you meant.",
        )

    total_count = last.total_count

    def _resolved(match: dict, picked_via: str) -> dict:
        display_name = match.get(name_field, "")
        session.pending_resolved_disclosure = f"Showing results for {display_name!r}:"
        origin_note = (
            f" Once resolved, go on to actually answer the ORIGINAL request — "
            f"'{origin_question}' — not just return this record's info."
            if origin_question and origin_question.strip().lower() != searched_name.lower()
            else ""
        )
        return {
            "success": True,
            "result": {
                "resolved": True, "kind": kind,
                "target_table": target_table, "target_alias": target_alias,
                "target_column": "id", "target_id": match.get(id_field, ""),
                "display_name": display_name, "total_candidates": len(candidates),
                "picked_via": picked_via, "warning": None,
                "needs_clarification": False,
                "clarification_message": origin_note or None,
            },
            "error": None,
        }

    if picked_position is not None:
        try:
            idx = int(picked_position) - 1
        except (TypeError, ValueError):
            return {"success": False, "result": None,
                    "error": f"picked_position must be an integer, got {picked_position!r}."}
        if 0 <= idx < len(candidates):
            return _resolved(candidates[idx], "position")
        return _needs_clarification(
            kind,
            f"picked_position {picked_position} is out of range — only "
            f"{len(candidates)} candidates were shown.",
        )

    matches = _find_exact_candidate(candidates, name_field, picked_name)

    if len(matches) == 1:
        return _resolved(matches[0], "name")

    if len(matches) > 1:
        # Genuine data-shape ambiguity, not a text-parsing artifact — several
        # distinct DB rows can share an identical display name (confirmed
        # live: 8 different "George Joseph" users). A clean, LLM-extracted
        # name is exactly as ambiguous here as raw question text was; must
        # still hard-stop rather than guess.
        dup_ids = ", ".join(f"'{c.get(id_field, '')}'" for c in matches)
        noun = "people" if kind == "person" else "records"
        return {
            "success": True,
            "result": {
                "resolved": False, "kind": kind, "target_table": "", "target_alias": "",
                "target_column": "", "target_id": "", "display_name": picked_name,
                "total_candidates": len(candidates), "picked_via": "name", "warning": None,
                "needs_clarification": True,
                "clarification_message": (
                    f"'{picked_name}' matches {len(matches)} DIFFERENT {noun} from the "
                    f"previous list (same display name, different records: "
                    f"{id_field} IN ({dup_ids})). Do NOT guess — call final_answer now "
                    f"asking the user to disambiguate further (e.g. by email address); "
                    f"do NOT pick one of these arbitrarily."
                ),
            },
            "error": None,
        }

    # Zero matches.
    msg = (
        f"'{picked_name}' does not exactly match any of the "
        f"{len(candidates)} previously-shown candidates."
    )
    if total_count > len(candidates):
        # Candidate lists are capped at 20 rows (see disambiguation.py's
        # _extract_disambiguation/_extract_entity_disambiguation) while
        # total_count may be larger — zero matches here doesn't mean the
        # real pick doesn't exist, only that it wasn't in the captured page.
        msg += (
            f" Only {len(candidates)} of {total_count} total matches were captured — "
            f"the real match may be outside this list. Ask the user for a more "
            f"specific name rather than assuming it doesn't exist."
        )
    return _needs_clarification(kind, msg)
