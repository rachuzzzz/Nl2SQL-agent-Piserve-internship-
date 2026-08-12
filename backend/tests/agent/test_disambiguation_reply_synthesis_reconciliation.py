"""
Regression tests for synthesis.py's active_subject_* inheritance branch
reconciliation (synthesis.py's `_finish()`, the `elif session and
session.turns:` block) — the fix for a confirmed real divergence between
two previously-independent implementations of "resolve a disambiguation
reply for scope-continuity purposes":

  - The retired session.py prompt-injection block matched by SUBSTRING
    containment against the raw question (so a full-sentence drilldown
    reply like "show mitigative actions at Adnoc Schools- Ghayathi"
    resolved fine for SQL-generation purposes).
  - synthesis.py's OWN independent implementation matched by EXACT
    whole-question equality — stricter, so that same full-sentence reply
    never set active_subject_type here, even though the id was used
    correctly in the SQL. A real continuity gap for the turn after.

Fix: prefer this turn's own resolve_disambiguation_reply tool-call result
(available in `steps`) over re-deriving a match from `question` text a
second time — works for both people and entity kinds, and doesn't depend
on the follow-up sentence's shape at all. The old text-match path is kept
as a fallback (for turns where the tool wasn't invoked, e.g. a forced
finish), now reusing disambiguation_reply.py's own exact-match helper
instead of an independently-diverging implementation.
"""

from src.agent.orchestrator.synthesis import SynthesisMixin
from src.agent.orchestrator.disambiguation import DisambiguationMixin
from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


class _FakeOrchestrator(SynthesisMixin, DisambiguationMixin):
    domain = "audit"
    db_engine = None
    _debug = None


class _FakeStep:
    def __init__(self, tool, args, result):
        self.tool = tool
        self.args = args
        self.result = result


def _resolved_disambig_step(kind, target_table, target_alias, target_id, display_name):
    return _FakeStep(
        "resolve_disambiguation_reply",
        {"picked_name": display_name},
        {"success": True, "result": {
            "resolved": True, "kind": kind,
            "target_table": target_table, "target_alias": target_alias,
            "target_column": "id", "target_id": target_id,
            "display_name": display_name, "total_candidates": 2,
            "picked_via": "name", "warning": None,
            "needs_clarification": False, "clarification_message": None,
        }},
    )


def _session_with_entity_candidates(table="facility", alias="fac"):
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="mitigative actions at golf", answer="...", tool_used="generate_sql",
        total_count=2,
        entity_disambiguation_candidates=[
            {"name": "Adnoc Schools- Ghayathi", "count": 5, "id": "fac-1"},
            {"name": "Adnoc Schools- Ruwais", "count": 3, "id": "fac-2"},
        ],
        entity_disambiguation_table=table,
        entity_disambiguation_alias=alias,
    ))
    return session


def test_resolved_tool_call_sets_active_subject_for_person_kind():
    """New capability: people-picks never fed into this inheritance
    mechanism before — only entities did."""
    orch = _FakeOrchestrator()
    session = ConversationSession()
    session.add_turn(ConversationTurn(
        question="who is george", answer="...", tool_used="generate_sql",
        disambiguation_name="george", total_count=2,
        disambiguation_candidates=[
            {"name": "George Smith", "user_id": "uid-1"},
            {"name": "George Joseph", "user_id": "uid-2"},
        ],
    ))
    steps = [_resolved_disambig_step("person", "users", "u", "uid-1", "George Smith")]
    orch._finish("George Smith", "answer", steps, iteration=1, session=session)
    turn = session.turns[-1]
    assert turn.active_subject_type == "inspector"
    assert turn.active_subject_id == "uid-1"
    assert turn.active_subject_value == "George Smith"


def test_resolved_tool_call_sets_active_subject_for_facility_kind():
    orch = _FakeOrchestrator()
    session = _session_with_entity_candidates(table="facility", alias="fac")
    steps = [_resolved_disambig_step("entity", "facility", "fac", "fac-1", "Adnoc Schools- Ghayathi")]
    orch._finish("Adnoc Schools- Ghayathi", "answer", steps, iteration=1, session=session)
    turn = session.turns[-1]
    assert turn.active_subject_type == "facility"
    assert turn.active_subject_id == "fac-1"


def test_resolved_tool_call_sets_active_subject_for_generic_entity_kind():
    """project/client/organisation use the generic 'entity' type, carrying
    table/alias — same convention the retired session.py comment documented."""
    orch = _FakeOrchestrator()
    session = _session_with_entity_candidates(table="project", alias="proj")
    steps = [_resolved_disambig_step("entity", "project", "proj", "fac-1", "Some Project")]
    orch._finish("Some Project", "answer", steps, iteration=1, session=session)
    turn = session.turns[-1]
    assert turn.active_subject_type == "entity"
    assert turn.active_subject_table == "project"
    assert turn.active_subject_alias == "proj"


def test_fallback_exact_whole_question_match_still_works_without_a_tool_call():
    """Regression guard: when the tool wasn't invoked this turn (e.g. a
    forced finish bypassing normal dispatch), the old text-match fallback
    must still resolve a bare, exact-whole-question reply."""
    orch = _FakeOrchestrator()
    session = _session_with_entity_candidates()
    steps = []  # no resolve_disambiguation_reply step this turn
    orch._finish("Adnoc Schools- Ghayathi", "answer", steps, iteration=1, session=session)
    turn = session.turns[-1]
    assert turn.active_subject_type == "facility"
    assert turn.active_subject_id == "fac-1"


def test_full_sentence_reply_with_tool_call_now_sets_active_subject():
    """The concrete regression test for the divergence this phase closes:
    a FULL-SENTENCE drilldown reply (not a bare name) used to silently
    fail synthesis.py's old exact-whole-question-equality fallback even
    though session.py's substring rule resolved it fine for SQL purposes.
    Now that the backstop forces the tool to be called first, the
    structured result is available regardless of the sentence shape."""
    orch = _FakeOrchestrator()
    session = _session_with_entity_candidates()
    steps = [_resolved_disambig_step(
        "entity", "facility", "fac", "fac-1", "Adnoc Schools- Ghayathi",
    )]
    # The actual full-sentence question — NOT equal to the candidate name.
    orch._finish(
        "show mitigative actions at Adnoc Schools- Ghayathi", "answer",
        steps, iteration=1, session=session,
    )
    turn = session.turns[-1]
    assert turn.active_subject_type == "facility"
    assert turn.active_subject_id == "fac-1"


def test_full_sentence_reply_without_tool_call_still_fails_the_fallback():
    """Confirms the divergence this phase fixes was real: WITHOUT a
    resolved tool-call step, the fallback's exact-whole-question match
    still can't resolve a full-sentence reply — this is exactly why the
    backstop (loop.py) exists to force the tool to be called."""
    orch = _FakeOrchestrator()
    session = _session_with_entity_candidates()
    steps = []
    orch._finish(
        "show mitigative actions at Adnoc Schools- Ghayathi", "answer",
        steps, iteration=1, session=session,
    )
    turn = session.turns[-1]
    assert turn.active_subject_type != "facility"
