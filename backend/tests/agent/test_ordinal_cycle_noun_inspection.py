"""
Regression test for the confirmed-live ordinal-resolution failure on
inspection_cycle listings.

disambiguation.py's _entity_noun_tables() had an audit_cycle -> "cycle"
entry for the audit domain but no inspection_cycle equivalent for
inspection — so entity_type_for_ordinal_noun("cycle", "inspection") always
returned "" (unrecognized). A later ordinal follow-up naming "cycle"
therefore fell to the untyped "bare ordinal" branch, which binds to the
MOST RECENTLY LISTED turn regardless of type, instead of the cycles
listing specifically.

Confirmed live 2026-08-04: "list all inspection cycles" -> "list all
inspections" -> "what is the status of the 3rd cycle" produced a wrong,
self-invented answer (the reasoning model got no usable deterministic
resolution, distrusted whatever context it did get, and fell back to a
fragile LIMIT 1 OFFSET 2 guess against a completely different question).

synthesis.py's capture side was already correct and domain-agnostic before
this fix — a cycle listing gets row_entity_type="cycle" via Priority 1.5
(the date-range + status composite branch) regardless of domain. The gap
was purely the missing noun registration in disambiguation.py.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn
from src.agent.orchestrator.ordinal import resolve_ordinal_reference

_CYCLE_TITLES = [
    "2026-06-03 to 2026-06-04 (COMPLETED)",
    "2026-06-05 to 2026-06-15 (COMPLETED)",
    "2026-06-16 to 2026-06-30 (COMPLETED)",
    "2026-07-01 to 2026-07-05 (COMPLETED)",
    "2026-07-06 to 2026-07-17 (COMPLETED)",
    "2026-07-18 to 2026-07-25 (COMPLETED)",
    "2026-07-27 to 2026-08-07 (ONGOING)",
    "2026-08-08 to 2026-08-14 (UPCOMING)",
    "2026-08-17 to 2026-08-21 (UPCOMING)",
]
_CYCLE_3_ID = "ba26ddb4-f3c4-451b-9876-777691e0b8be"
_CYCLE_ROW_IDS = [
    "4fc2e71a-8e0f-4aa5-bfc6-2d956804b6fb",
    "c3df2404-ad93-48e6-b822-3b90930befac",
    _CYCLE_3_ID,
    "a18c81ad-ca2c-4d37-a9e2-fddd9ca79350",
    "52fbb30a-e3c0-4487-9424-63c8ac7046c6",
    "61c90f78-2cde-4763-b3fa-65917a51c42d",
    "1ffa9d9f-c5a9-46cd-8ae9-2e6cb96f20e8",
    "7038523b-dcd9-48c2-95e4-924260840288",
    "30b5b917-c77b-4647-9f57-d772af8c49a6",
]

_INSPECTION_TITLES = [f"2026/07/ST{i:03d}/INS001" for i in range(1, 51)]
_INSPECTION_ROW_IDS = [f"bbbb{i:04d}-0000-0000-0000-00000000000{i % 10}" for i in range(1, 51)]


def _session_with_cycle_then_inspection_listing() -> ConversationSession:
    """Mirrors the exact live repro: a cycles listing, then an unrelated
    (more recent) inspections listing — the intervening list that caused
    the live bug."""
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="list all inspection cycles",
        answer="9 cycles found",
        tool_used="generate_sql",
        total_count=9,
        columns=["start_date", "end_date", "status"],
        row_titles=list(_CYCLE_TITLES),
        row_ids=list(_CYCLE_ROW_IDS),
        row_id_columns=["id"] * 9,
        row_entity_type="cycle",
    ))
    session.add_turn(ConversationTurn(
        question="list all inspections",
        answer="50 inspections found",
        tool_used="generate_sql",
        total_count=50,
        columns=["inspection_id"],
        row_titles=list(_INSPECTION_TITLES),
        row_ids=list(_INSPECTION_ROW_IDS),
        row_id_columns=["id"] * 50,
        row_entity_type="inspection",
    ))
    return session


def test_third_cycle_resolves_against_the_cycles_list_not_the_intervening_inspections_list():
    session = _session_with_cycle_then_inspection_listing()
    r = resolve_ordinal_reference(session, "inspection", position=3, entity_type_hint="cycle")
    result = r["result"]
    assert result["resolved"] is True
    assert result["target_id"] == _CYCLE_3_ID
    # Must NOT have fallen back to the untyped "most recent list" (the
    # inspections listing) — none of the inspection row ids should match.
    for other_id in _INSPECTION_ROW_IDS:
        assert result["target_id"] != other_id


def test_bare_ordinal_with_no_noun_still_falls_back_to_most_recent_list_unchanged():
    """Regression guard: this fix must not touch the untyped bare-ordinal
    path other tests already cover — "the 3rd one" (no hint) should still
    resolve against the most recently listed turn regardless of type."""
    session = _session_with_cycle_then_inspection_listing()
    r = resolve_ordinal_reference(session, "inspection", position=3, entity_type_hint=None)
    result = r["result"]
    assert result["resolved"] is True
    assert result["target_id"] == _INSPECTION_ROW_IDS[2]
