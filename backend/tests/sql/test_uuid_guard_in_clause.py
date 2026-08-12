"""
Regression test for the confirmed-live bug (2026-08-06): generation.py's
UUID-filter guard (~line 486) only recognized a single-uuid "col = 'uuid'"
shape in schema_hint. resolve_selected_rows (agent/orchestrator/
row_selection.py) emits a multi-row "col IN ('u1','u2',...)" shape for a
checkbox selection — the guard's regex never matched that, so it never
fired to catch/retry a model that silently dropped the whole filter. Live
symptom: selecting 2 corrective actions and asking "internal stakeholders
for these" returned 18 rows (every approved corrective action) instead of
2, because the coder model fell back to a generic unfiltered listing and
nothing caught it.
"""

from types import SimpleNamespace

from src.sql.generation import generate_sql
from src.sql.validation import SQLValidator


class _FakeResponse:
    def __init__(self, text, raw):
        self._text = text
        self.raw = raw

    def __str__(self):
        return self._text


class _FakeSqlLlm:
    def __init__(self, responses):
        self.model = "fake-model"
        self._responses = list(responses)
        self.calls = 0

    def complete(self, prompt):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _validator():
    return SQLValidator(domain="audit")


_UUID_1 = "ae8c734b-2f13-4749-bc54-734df340dfb0"
_UUID_2 = "26098896-88bc-48af-b222-2651c4dad810"


def test_in_clause_guard_fires_when_model_drops_the_whole_filter():
    """First completion ignores schema_hint's IN clause entirely (the exact
    live-caught failure shape) — the guard must force a retry pinning both
    uuids as a single IN (...) clause, not two AND'd equalities (which
    could never match any row)."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT aca.corrective_action_id, u.first_name FROM audit_corrective_action aca "
            "JOIN users u ON aca.internal_stakeholder_id = u.id WHERE aca.active = true;",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=20)},
        ),
        _FakeResponse(
            f"SELECT aca.corrective_action_id, u.first_name FROM audit_corrective_action aca "
            f"JOIN users u ON aca.internal_stakeholder_id = u.id "
            f"WHERE aca.id IN ('{_UUID_1}', '{_UUID_2}') AND aca.active = true;",
            {"usage": SimpleNamespace(input_tokens=150, output_tokens=25)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="internal stakeholders for these corrective actions",
        schema_hint=(
            f"SELECT aca.corrective_action_id, u.first_name || ' ' || u.last_name AS stakeholder_name "
            f"FROM audit_corrective_action aca JOIN users u ON aca.internal_stakeholder_id = u.id "
            f"WHERE aca.id IN ('{_UUID_1}', '{_UUID_2}') AND aca.active = true;"
        ),
    )
    assert sql_llm.calls == 2
    final_sql = result["result"]["sql"].lower()
    assert _UUID_1.lower() in final_sql
    assert _UUID_2.lower() in final_sql
    # Never two AND'd equalities on the same column for a multi-uuid IN source.
    assert f"aca.id = '{_UUID_1.lower()}' and aca.id = '{_UUID_2.lower()}'" not in final_sql


def test_in_clause_guard_does_not_fire_when_model_already_kept_it():
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            f"SELECT aca.corrective_action_id FROM audit_corrective_action aca "
            f"WHERE aca.id IN ('{_UUID_1}', '{_UUID_2}') AND aca.active = true;",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=20)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="internal stakeholders for these corrective actions",
        schema_hint=(
            f"SELECT aca.corrective_action_id FROM audit_corrective_action aca "
            f"WHERE aca.id IN ('{_UUID_1}', '{_UUID_2}') AND aca.active = true;"
        ),
    )
    assert sql_llm.calls == 1
    assert result["success"] is True
