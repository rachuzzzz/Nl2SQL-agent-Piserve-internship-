"""
Regression tests for generation.py's _guard_scan_text decoupling — Item 2
of the 2026-08-07 generation-quality plan (qid=e63f03af).

Before this fix, the offset-guard scanned only the raw LLM-authored
schema_hint and the uuid-guard scanned the post-concat schema_hint (which
used to include INJECT-rule hints appended onto it). Splitting mandatory
INJECT-rule/LIMIT-N content into a separate _mandatory_hint argument (so it
can render as its own "Required" sub-section, see
test_additional_context_sections.py) would silently stop both guards from
seeing a resolved OFFSET/UUID literal that originated in a mandatory hint
rather than the LLM's own draft — these tests prove that regression did not
happen: both guards now scan a shared _guard_scan_text covering BOTH
schema_hint and _mandatory_hint.
"""

from types import SimpleNamespace

from src.sql.generation import generate_sql
from src.sql.validation import SQLValidator

from tests.sql.test_generation_token_accounting import _FakeResponse, _FakeSqlLlm


def _validator():
    return SQLValidator(domain="audit")


def test_offset_guard_fires_when_offset_only_in_mandatory_hint():
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT ac.status FROM audit_cycle ac;",  # no OFFSET -> should trigger retry
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)},
        ),
        _FakeResponse(
            "SELECT ac.status FROM audit_cycle ac LIMIT 1 OFFSET 3;",
            {"usage": SimpleNamespace(input_tokens=150, output_tokens=15)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="the 4th cycle's status",
        schema_hint="",  # deliberately empty — OFFSET lives only in the mandatory hint
        _mandatory_hint="SELECT ac.id FROM audit_cycle ac ORDER BY ac.start_date LIMIT 1 OFFSET 3",
    )
    assert sql_llm.calls == 2, "offset-guard did not fire for an OFFSET that only appeared in _mandatory_hint"
    assert result["result"]["sql"] is not None


def test_uuid_guard_fires_when_uuid_only_in_mandatory_hint():
    uid = "12345678-1234-1234-1234-123456789012"
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT u.first_name FROM users u WHERE u.active = true;",  # missing UUID filter
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)},
        ),
        _FakeResponse(
            f"SELECT u.first_name FROM users u WHERE u.id = '{uid}';",
            {"usage": SimpleNamespace(input_tokens=150, output_tokens=15)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="that specific user's name",
        schema_hint="",  # deliberately empty — the UUID filter lives only in the mandatory hint
        _mandatory_hint=f"u.id = '{uid}'",
    )
    assert sql_llm.calls == 2, "uuid-guard did not fire for a UUID that only appeared in _mandatory_hint"
    assert uid in result["result"]["sql"]


def test_offset_guard_still_fires_when_offset_only_in_schema_hint():
    """Unchanged-behavior guard: the pre-existing case (OFFSET in the LLM's
    own schema_hint) must keep working exactly as before the split."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT ac.status FROM audit_cycle ac;",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)},
        ),
        _FakeResponse(
            "SELECT ac.status FROM audit_cycle ac LIMIT 1 OFFSET 3;",
            {"usage": SimpleNamespace(input_tokens=150, output_tokens=15)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="the 4th cycle's status",
        schema_hint="SELECT ac.id FROM audit_cycle ac ORDER BY ac.start_date LIMIT 1 OFFSET 3",
    )
    assert sql_llm.calls == 2
    assert result["result"]["sql"] is not None
