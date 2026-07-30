"""
Regression tests for synthesis.py's _finish() capturing real row ids
(row_ids, active_subject_id) alongside display values — the M2 half of the
row-identity fix (see src/sql/validation/identity.py's inject_row_identity()
for M1, the AST injection that produces these __id-suffixed columns).

No LLM/DB required — feeds fabricated `steps` with an execute_sql result
straight into _finish() and inspects the ConversationTurn it produces.
"""

from src.agent.orchestrator.synthesis import SynthesisMixin
from src.agent.orchestrator.disambiguation import DisambiguationMixin
from src.agent.orchestrator.session import ConversationSession


class _FakeOrchestrator(SynthesisMixin, DisambiguationMixin):
    domain = "audit"
    db_engine = None
    _debug = None


class _FakeStep:
    def __init__(self, tool, args, result):
        self.tool = tool
        self.args = args
        self.result = result


def _execute_sql_step(rows: list[dict]):
    cols = list(rows[0].keys()) if rows else []
    return _FakeStep(
        "execute_sql",
        {"sql": "SELECT ..."},
        {"success": True, "result": {
            "total_count": len(rows), "row_count": len(rows),
            "columns": cols, "rows": rows,
        }},
    )


def _run(question: str, rows: list[dict]) -> ConversationSession:
    orch = _FakeOrchestrator()
    session = ConversationSession()
    steps = [_execute_sql_step(rows)]
    orch._finish(question, "answer text", steps, iteration=1, session=session)
    return session


def test_row_ids_paired_with_row_titles_multi_row_list():
    """The exact live-caught shape: a plan list where facility_name AND
    audit_id (identity-injected as audit_id__id) both appear on the same
    row — row_titles must prefer audit_id (the grammatical subject of "show
    all audits"), and row_ids must carry that SAME row's real id, not the
    facility's."""
    rows = [
        {"plan_title": "Delegation retirement plan", "schedule_id": "SCH001",
         "facility_name": "ADGM Square", "audit_id": "2026/07/ST034/AUD001",
         "audit_id__id": "aaaa1111-0000-0000-0000-000000000001"},
        {"plan_title": "Delegation retirement plan", "schedule_id": "SCH002",
         "facility_name": "Al Oyoun Village", "audit_id": "2026/07/ST157/AUD001",
         "audit_id__id": "aaaa2222-0000-0000-0000-000000000002"},
        {"plan_title": "Delegation retirement plan", "schedule_id": "SCH003",
         "facility_name": "Al Ghadeer", "audit_id": None, "audit_id__id": None},
    ]
    session = _run("show all audits part of this plan", rows)
    turn = session.turns[-1]

    assert turn.row_titles[0] == "2026/07/ST034/AUD001"
    assert turn.row_ids[0] == "aaaa1111-0000-0000-0000-000000000001"
    assert turn.row_titles[1] == "2026/07/ST157/AUD001"
    assert turn.row_ids[1] == "aaaa2222-0000-0000-0000-000000000002"
    # No __id column ever leaks into the columns list shown to the LLM
    assert not any(c.endswith("__id") for c in turn.columns)


def test_plan_active_subject_id_resolved_from_duplicate_rows_same_plan():
    """2 rows, same plan (e.g. review level 1 + level 2), column aliased
    audit_plan_title with a paired audit_plan_title__id — active_subject_id
    should resolve to that real UUID, not require a DB round trip."""
    rows = [
        {"audit_plan_title": "Delegation retirement plan",
         "audit_plan_title__id": "c58b5540-6cac-41ea-a308-2020ee9497b2",
         "approval_level": 1},
        {"audit_plan_title": "Delegation retirement plan",
         "audit_plan_title__id": "c58b5540-6cac-41ea-a308-2020ee9497b2",
         "approval_level": 2},
    ]
    session = _run("approval status for the second one", rows)
    turn = session.turns[-1]

    assert turn.active_subject_type == "plan"
    assert turn.active_subject_value == "Delegation retirement plan"
    assert turn.active_subject_id == "c58b5540-6cac-41ea-a308-2020ee9497b2"


def test_active_subject_id_empty_when_injection_did_not_run():
    """GROUP BY-style result (identity.py skips injection for these) — no
    __id column present at all. active_subject_id must stay empty, not
    fabricate one; downstream should fall back to title/ILIKE resolution."""
    rows = [
        {"title": "Delegation testing", "cnt": 2},
    ]
    session = _run("plan counts", rows)
    turn = session.turns[-1]

    assert turn.active_subject_type == "plan"
    assert turn.active_subject_id == ""


def test_facility_prefers_existing_facility_id_over_identity_injection():
    # facility_ids only captures values that actually look like a UUID (see
    # the _UUID_PAT guard in synthesis.py) — use a real-shaped one here so
    # this test exercises the "prefer facility_ids" branch, not the "wasn't
    # a real UUID, fall through to identity injection" branch.
    rows_single = [
        {"facility_name": "ADGM Square",
         "facility_id": "bbbb0001-0000-0000-0000-000000000001",
         "facility_name__id": "should-not-be-used"},
    ]
    session = _run("show facility details", rows_single)
    turn = session.turns[-1]
    assert turn.active_subject_type == "facility"
    assert turn.active_subject_id == "bbbb0001-0000-0000-0000-000000000001"
