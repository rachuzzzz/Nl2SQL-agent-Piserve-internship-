"""
Regression tests for resolve_disambiguation_reply (agent/orchestrator/
disambiguation_reply.py) — the tool-based replacement for the retired
regex reply-resolution blocks in session.py's build_context_block()
(people-name disambiguation, entity disambiguation) — see that module's
docstring for the full history.

No prior test coverage existed for either retired block (confirmed via
repo-wide grep before this file was written), so these tests are written
from scratch against the new function's contract, not ported from an
older test.
"""

from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn
from src.agent.orchestrator.disambiguation_reply import resolve_disambiguation_reply


def _session_with_people(candidates, total_count=None, origin_question=""):
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="who is george", answer="...", tool_used="generate_sql",
        disambiguation_name="george",
        total_count=total_count if total_count is not None else len(candidates),
        disambiguation_candidates=candidates,
        disambiguation_origin_question=origin_question,
    ))
    return session


def _session_with_entities(candidates, table="facility", alias="fac", total_count=None):
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="mitigative actions at golf", answer="...", tool_used="generate_sql",
        total_count=total_count if total_count is not None else len(candidates),
        entity_disambiguation_candidates=candidates,
        entity_disambiguation_table=table,
        entity_disambiguation_alias=alias,
    ))
    return session


def test_exact_people_pick_by_name():
    session = _session_with_people([
        {"name": "George Smith", "user_id": "uid-1"},
        {"name": "George Joseph", "user_id": "uid-2"},
    ])
    r = resolve_disambiguation_reply(session, picked_name="George Smith")
    result = r["result"]
    assert result["resolved"] is True
    assert result["kind"] == "person"
    assert result["target_table"] == "users"
    assert result["target_alias"] == "u"
    assert result["target_column"] == "id"
    assert result["target_id"] == "uid-1"
    assert result["picked_via"] == "name"


def test_ambiguous_picked_name_matching_multiple_candidates_stops():
    """Genuine data-shape ambiguity (two rows share a display name) is
    preserved even though the old text-parsing tiers are gone — a clean
    LLM-extracted name is exactly as ambiguous as raw text was."""
    session = _session_with_people([
        {"name": "George Joseph", "user_id": "uid-2"},
        {"name": "George Joseph", "user_id": "uid-3"},
    ])
    r = resolve_disambiguation_reply(session, picked_name="George Joseph")
    result = r["result"]
    assert result["resolved"] is False
    assert result["needs_clarification"] is True
    assert result["target_id"] == ""
    assert "uid-2" in result["clarification_message"]
    assert "uid-3" in result["clarification_message"]


def test_entity_pick_by_name_uses_turn_table_and_alias():
    session = _session_with_entities(
        [{"name": "Golf Gardens", "count": 5, "id": "fac-1"},
         {"name": "Golf Course Villas", "count": 3, "id": "fac-2"}],
        table="facility", alias="fac",
    )
    r = resolve_disambiguation_reply(session, picked_name="Golf Gardens")
    result = r["result"]
    assert result["resolved"] is True
    assert result["kind"] == "entity"
    assert result["target_table"] == "facility"
    assert result["target_alias"] == "fac"
    assert result["target_id"] == "fac-1"


def test_positional_pick_people():
    session = _session_with_people([
        {"name": "A", "user_id": "uid-a"},
        {"name": "B", "user_id": "uid-b"},
        {"name": "C", "user_id": "uid-c"},
    ])
    r = resolve_disambiguation_reply(session, picked_position=2)
    assert r["result"]["target_id"] == "uid-b"
    assert r["result"]["picked_via"] == "position"


def test_positional_pick_entity():
    session = _session_with_entities([
        {"name": "X", "count": 1, "id": "id-x"},
        {"name": "Y", "count": 1, "id": "id-y"},
    ])
    r = resolve_disambiguation_reply(session, picked_position=1)
    assert r["result"]["target_id"] == "id-x"


def test_positional_pick_out_of_range():
    session = _session_with_people([{"name": "A", "user_id": "uid-a"}])
    r = resolve_disambiguation_reply(session, picked_position=99)
    result = r["result"]
    assert result["resolved"] is False
    assert result["needs_clarification"] is True
    assert "out of range" in result["clarification_message"]


def test_positional_pick_uses_position_even_when_duplicate_names_present():
    """Position sidesteps name-matching entirely — unambiguous even across
    duplicate display names, unlike a name-based pick."""
    session = _session_with_people([
        {"name": "George Joseph", "user_id": "uid-2"},
        {"name": "George Joseph", "user_id": "uid-3"},
    ])
    r = resolve_disambiguation_reply(session, picked_position=2)
    assert r["result"]["resolved"] is True
    assert r["result"]["target_id"] == "uid-3"


def test_no_match_found():
    session = _session_with_people([{"name": "George Smith", "user_id": "uid-1"}])
    r = resolve_disambiguation_reply(session, picked_name="Nobody Here")
    result = r["result"]
    assert result["resolved"] is False
    assert result["needs_clarification"] is True
    assert "Nobody Here" in result["clarification_message"]


def test_no_match_warns_when_candidate_list_was_capped():
    """Candidate lists are capped at 20 rows while total_count may be
    larger — zero matches doesn't mean the real pick doesn't exist."""
    candidates = [{"name": f"Person {i}", "user_id": f"uid-{i}"} for i in range(20)]
    session = _session_with_people(candidates, total_count=25)
    r = resolve_disambiguation_reply(session, picked_name="Someone Not Listed")
    msg = r["result"]["clarification_message"]
    assert "Only 20 of 25" in msg


def test_no_match_does_not_warn_when_list_was_complete():
    candidates = [{"name": "George Smith", "user_id": "uid-1"}]
    session = _session_with_people(candidates, total_count=1)
    r = resolve_disambiguation_reply(session, picked_name="Nobody Here")
    assert "Only" not in r["result"]["clarification_message"]


def test_neither_arg_supplied_is_malformed():
    session = _session_with_people([{"name": "A", "user_id": "uid-a"}])
    r = resolve_disambiguation_reply(session)
    assert r["success"] is False
    assert r["result"] is None
    assert "requires" in r["error"]


def test_no_pending_disambiguation_on_last_turn():
    session = ConversationSession()
    session.add_turn(ConversationTurn(question="x", answer="y", tool_used="generate_sql"))
    r = resolve_disambiguation_reply(session, picked_name="anything")
    result = r["result"]
    assert result["resolved"] is False
    assert result["needs_clarification"] is True
    assert "no pending disambiguation" in result["clarification_message"].lower() \
        or "no pending disambiguation" in result["clarification_message"]


def test_empty_session_needs_clarification_not_an_error():
    session = ConversationSession()
    r = resolve_disambiguation_reply(session, picked_name="anything")
    assert r["success"] is True
    assert r["result"]["needs_clarification"] is True


def test_pending_resolved_disclosure_set_on_success():
    session = _session_with_people([{"name": "George Smith", "user_id": "uid-1"}])
    assert session.pending_resolved_disclosure == ""
    resolve_disambiguation_reply(session, picked_name="George Smith")
    assert "George Smith" in session.pending_resolved_disclosure


def test_pending_resolved_disclosure_not_set_on_failure():
    session = _session_with_people([{"name": "George Smith", "user_id": "uid-1"}])
    resolve_disambiguation_reply(session, picked_name="Nobody Here")
    assert session.pending_resolved_disclosure == ""


def test_origin_question_reminder_included_when_different_from_search_term():
    session = _session_with_people(
        [{"name": "George Smith", "user_id": "uid-1"}],
        origin_question="show mitigative actions assigned to george",
    )
    r = resolve_disambiguation_reply(session, picked_name="George Smith")
    msg = r["result"]["clarification_message"]
    assert msg is not None
    assert "mitigative actions" in msg


def test_origin_question_reminder_omitted_when_same_as_search_term():
    session = _session_with_people(
        [{"name": "George Smith", "user_id": "uid-1"}],
        origin_question="george",  # same as disambiguation_name, not a real "original ask"
    )
    r = resolve_disambiguation_reply(session, picked_name="George Smith")
    assert r["result"]["clarification_message"] is None
