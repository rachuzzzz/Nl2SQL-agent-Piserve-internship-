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


def test_row_titles_captures_cycle_composite_display():
    """audit_cycle has no name/title column — row_titles must fall back to
    the synthetic "<start_date> to <end_date> (<status>)" composite display,
    gated on the "<alias>_cycle_row__id" sibling that identity.py's
    _CYCLE_TABLES branch injects (see test_identity_injection.py's
    test_injects_cycle_composite_id_for_audit_cycle for the M1 half). This is
    the fix for the confirmed-live ordinal-resolution failure on cycle
    listings (qid=f2ec1739/f13ab827/4d1d0bf4)."""
    rows = [
        {"start_date": "2026-01-01", "end_date": "2026-01-31", "status": "COMPLETED",
         "acyc_cycle_row__id": "cccc0001-0000-0000-0000-000000000001"},
        {"start_date": "2026-02-01", "end_date": "2026-02-28", "status": "ONGOING",
         "acyc_cycle_row__id": "cccc0002-0000-0000-0000-000000000002"},
    ]
    session = _run("list all cycles", rows)
    turn = session.turns[-1]

    assert turn.row_titles[0] == "2026-01-01 to 2026-01-31 (COMPLETED)"
    assert turn.row_ids[0] == "cccc0001-0000-0000-0000-000000000001"
    assert turn.row_id_columns[0] == "id"
    assert turn.row_titles[1] == "2026-02-01 to 2026-02-28 (ONGOING)"
    assert turn.row_ids[1] == "cccc0002-0000-0000-0000-000000000002"
    # No __id column ever leaks into the columns list shown to the LLM
    assert not any(c.endswith("__id") for c in turn.columns)


def test_row_entity_type_audit_priority1():
    """Priority 1's matched column name (audit_id/schedule_id/inspection_id/
    corrective_action_id) IS the entity-type signal — no new plumbing
    needed beyond a lookup keyed on which column matched."""
    rows = [
        {"plan_title": "Delegation retirement plan", "audit_id": "2026/07/ST034/AUD001",
         "audit_id__id": "aaaa1111-0000-0000-0000-000000000001"},
        {"plan_title": "Delegation retirement plan", "audit_id": "2026/07/ST157/AUD001",
         "audit_id__id": "aaaa2222-0000-0000-0000-000000000002"},
    ]
    session = _run("show all audits part of this plan", rows)
    assert session.turns[-1].row_entity_type == "audit"


def test_row_entity_type_cycle_priority1_5():
    rows = [
        {"start_date": "2026-01-01", "end_date": "2026-01-31", "status": "COMPLETED",
         "acyc_cycle_row__id": "cccc0001-0000-0000-0000-000000000001"},
        {"start_date": "2026-02-01", "end_date": "2026-02-28", "status": "ONGOING",
         "acyc_cycle_row__id": "cccc0002-0000-0000-0000-000000000002"},
    ]
    session = _run("list all cycles", rows)
    assert session.turns[-1].row_entity_type == "cycle"


def test_row_entity_type_plan_priority2():
    rows = [
        {"title": "Delegation testing", "status": "ACTIVE"},
        {"title": "Delegation plan -2", "status": "ACTIVE"},
    ]
    session = _run("list all plans", rows)
    assert session.turns[-1].row_entity_type == "plan"


def test_row_entity_type_person_for_auditor_name():
    """auditor_name/inspector_name are unified to 'person' — both draw from
    the identical `users` table, just via a different FK role."""
    rows = [
        {"auditor_name": "George Auditor", "auditor_name__id": "bbbb0001-0000-0000-0000-000000000001"},
        {"auditor_name": "Jane Auditor", "auditor_name__id": "bbbb0002-0000-0000-0000-000000000002"},
    ]
    session = _run("list all auditors", rows)
    assert session.turns[-1].row_entity_type == "person"


def test_row_entity_type_empty_for_ambiguous_bare_name():
    """A bare 'name' column could be a facility, a portfolio, or anything
    else with a generic display column — deliberately left untyped rather
    than guessing, since a wrong type label would block a legitimate later
    ordinal match just as badly as no label at all."""
    rows = [
        {"name": "Something A"},
        {"name": "Something B"},
    ]
    session = _run("list all somethings", rows)
    assert session.turns[-1].row_entity_type == ""


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
