"""
Regression tests for the stale ACTIVE CONTEXT scope-leak bug.

Confirmed live: an AUDIT-domain turn incidentally returned 2 rows sharing
one facility (neither row about the facility itself) and got tagged
active_subject_type="facility". Five completely unrelated INSPECTION-domain
turns later (observations, corrective actions, capex rankings — none
referencing a facility at all), synthesis.py's inheritance branch had kept
blindly re-copying that stale scope onto each of them, then a 6th turn
("show top 5 of the same", meant to repeat the immediately prior GLOBAL
FM-manager ranking) got the facility filter silently reapplied — narrowing
a real 5-row answer down to 1, with no error.

Fix: synthesis.py's inheritance branch now checks whether the CURRENT
turn's own executed SQL shows any sign of relating to the scope's entity
(_sql_still_relevant_to_scope) before inheriting it — and clears
session.current_scope entirely when it doesn't, so it can't perpetuate onto
the *next* unrelated turn either.
"""

from src.agent.orchestrator.synthesis import SynthesisMixin, _sql_still_relevant_to_scope
from src.agent.orchestrator.disambiguation import DisambiguationMixin
from src.agent.orchestrator.session import ConversationSession
from src.agent.orchestrator.models import ConversationTurn


class _FakeOrchestrator(SynthesisMixin, DisambiguationMixin):
    domain = "inspection"
    db_engine = None
    _debug = None


class _FakeStep:
    def __init__(self, tool, args, result):
        self.tool = tool
        self.args = args
        self.result = result


def _execute_sql_step(rows: list[dict], sql: str = "SELECT ..."):
    cols = list(rows[0].keys()) if rows else []
    return _FakeStep(
        "execute_sql",
        {"sql": sql},
        {"success": True, "result": {
            "total_count": len(rows), "row_count": len(rows),
            "columns": cols, "rows": rows,
        }},
    )


_FACILITY_SCOPE = {
    "type": "facility", "value": "Aldar HQ", "label": "Aldar HQ",
    "table": "", "alias": "", "domain": "audit",
    "id": "52611ecb-8be8-4507-a25c-df5807ccf2b4",
}


def _session_with_stale_facility_scope() -> ConversationSession:
    session = ConversationSession()
    # Seed a turn so self.turns isn't empty (build_context_block() no-ops
    # on an empty session), then stamp the stale scope directly onto
    # current_scope — exactly what add_turn() would have done for the
    # original facility-tagging turn several turns back.
    session.add_turn(ConversationTurn(
        question="show all audits conducted in the last two weeks of june",
        answer="2 results found", tool_used="generate_sql",
        active_subject_type="facility", active_subject_value="Aldar HQ",
        active_subject_label="Aldar HQ", active_subject_domain="audit",
        active_subject_id=_FACILITY_SCOPE["id"],
    ))
    assert session.current_scope == _FACILITY_SCOPE
    return session


def test_sql_relevance_helper_facility():
    assert _sql_still_relevant_to_scope(
        "SELECT * FROM facility fac JOIN ...", "facility"
    ) is True
    assert _sql_still_relevant_to_scope(
        "SELECT u.first_name FROM inspection_corrective_action ica JOIN users u ON ...",
        "facility",
    ) is False


def test_sql_relevance_helper_empty_sql_is_dont_disturb():
    assert _sql_still_relevant_to_scope("", "facility") is True


def test_sql_relevance_helper_ungated_type_always_relevant():
    assert _sql_still_relevant_to_scope("SELECT 1", "inspector") is True


def test_facility_blind_turn_clears_the_stale_scope():
    """The exact live shape: a global FM-manager capex ranking with no
    facility table anywhere in its SQL — must NOT inherit the stale
    facility scope, and must clear current_scope entirely."""
    orch = _FakeOrchestrator()
    session = _session_with_stale_facility_scope()
    rows = [
        {"fm_manager": "neenu inspectee", "capex_converted": 24758},
        {"fm_manager": "neenu inspectee2", "capex_converted": 8706},
    ]
    steps = [_execute_sql_step(
        rows,
        sql=(
            "SELECT u.first_name AS fm_manager, SUM(p.total_po) AS capex_converted "
            "FROM inspection_corrective_action ica "
            "JOIN inspection_report ir ON ica.inspection_id = ir.id "
            "JOIN users u ON ir.inspectee_user_id = u.id "
            "LEFT JOIN po_details p ON p.mitigative_action_uuid = ica.id "
            "GROUP BY u.id, u.first_name ORDER BY capex_converted DESC LIMIT 20"
        ),
    )]
    orch._finish(
        "which fm managers convert the most capex by amount",
        "answer", steps, iteration=1, session=session,
    )
    turn = session.turns[-1]
    assert turn.active_subject_type != "facility"
    assert session.current_scope is None

    # And the NEXT turn's context block must not carry the facility scope
    # forward either — this is the actual user-visible fix.
    ctx = session.build_context_block(
        current_question="show top 5 of the same", current_domain="inspection",
    )
    assert "ACTIVE CONTEXT" not in ctx or "facility" not in ctx.lower()


def test_facility_relevant_turn_still_preserves_the_scope():
    """The mechanism must keep working for genuine multi-turn drilldowns —
    a turn whose SQL DOES reference facility must still inherit/preserve
    the scope, no regression to the intended behavior."""
    orch = _FakeOrchestrator()
    session = _session_with_stale_facility_scope()
    rows = [{"corrective_action_id": "CA001", "status": "OPEN"}]
    steps = [_execute_sql_step(
        rows,
        sql=(
            "SELECT ica.corrective_action_id, ica.status "
            "FROM inspection_corrective_action ica "
            "JOIN inspection_report ir ON ica.inspection_id = ir.id "
            "JOIN facility fac ON ir.facility_id = fac.id "
            "WHERE fac.id = '52611ecb-8be8-4507-a25c-df5807ccf2b4'"
        ),
    )]
    orch._finish(
        "what about the open ones", "answer", steps, iteration=1, session=session,
    )
    assert session.current_scope is not None
    assert session.current_scope["type"] == "facility"


def test_turn_with_no_sql_leaves_scope_untouched():
    """get_answers-only turns (no generate_sql) can't be judged either way
    — the existing pagination-continuation mechanism owns that case, this
    fix must not disturb it."""
    orch = _FakeOrchestrator()
    session = _session_with_stale_facility_scope()
    steps = [_FakeStep(
        "get_answers", {},
        {"success": True, "result": {"rows": [{"answer_text": "some answer"}]}},
    )]
    orch._finish("show more", "answer", steps, iteration=1, session=session)
    assert session.current_scope is not None
    assert session.current_scope["type"] == "facility"
