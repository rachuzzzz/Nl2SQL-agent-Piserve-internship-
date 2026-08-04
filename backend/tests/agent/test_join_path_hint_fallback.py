"""
Regression tests for src/sql/generation.py's fallback_tables mechanism —
_extract_table_names_from_sql() + _select_schema_block()'s fallback_tables
param.

Confirmed live: relevant_tables (the ONLY thing that used to trigger the
deterministic JOIN PATH hint) is essentially never populated in practice —
checked 58 real generate_sql calls across 5 sessions spanning 5 days
(2026-07-30 through 2026-08-04), zero hits. Root cause: the system prompt
frames relevant_tables as needed only for tables "beyond the default schema
context", but audit_report/audit_schedule/audit_corrective_action — the
exact hub tables both confirmed-live subquery bugs (see
test_subquery_uuid_display_code_mismatch.py) joined through — are already
in that default context, so the reasoning model never has a reason to list
them.

Fix: derive a fallback table list from the reasoning model's OWN
schema_hint (supplied on nearly every call, unlike relevant_tables) and use
it ONLY to still append the join-path hint onto the slim schema block —
deliberately not switching to the larger per-table for_schema_hint() block,
to preserve the token-budget tradeoff the slim block exists for.
"""

from src.knowledge.schema_introspector import DatabaseSchema, TableSchema
from src.sql import audit_join_graph
from src.sql.generation import _extract_table_names_from_sql, _select_schema_block


def _make_schema() -> DatabaseSchema:
    schema = DatabaseSchema()
    schema.tables["facility"] = TableSchema(
        name="facility",
        columns=[
            {"name": "id", "type": "uuid", "nullable": False},
            {"name": "name", "type": "varchar", "nullable": True},
        ],
        foreign_keys=[],
        primary_key=["id"],
    )
    return schema


# ---------------------------------------------------------------------------
# _extract_table_names_from_sql
# ---------------------------------------------------------------------------

def test_extracts_from_and_join_tables_in_order_deduped():
    hint = (
        "SELECT asch.schedule_id FROM audit_schedule asch "
        "JOIN audit_report arep ON arep.schedule_id = asch.id "
        "JOIN ai_audit_answers aaa ON aaa.audit_report_id = arep.id "
        "JOIN audit_report arep2 ON true"
    )
    tables = _extract_table_names_from_sql(hint)
    assert tables == ["audit_schedule", "audit_report", "ai_audit_answers"]


def test_extracts_the_exact_20260804_live_schema_hint():
    hint = (
        "SELECT arep.audit_id, aq.label, aaa.answer_text, aaa.answer_numeric, aaa.score "
        "FROM audit_schedule asch "
        "JOIN audit_report arep ON arep.schedule_id = asch.id "
        "JOIN ai_audit_answers aaa ON aaa.audit_report_id = arep.id "
        "JOIN ai_questions aq ON aaa.element_id = aq.element_id "
        "WHERE asch.schedule_id = (SELECT id FROM audit_schedule WHERE schedule_id = '2026/08/ST040/SCH001') "
        "AND aq.label = 'Findings / Remarks' AND arep.active = true AND asch.active = true "
        "ORDER BY aaa.submission_id, aq.label;"
    )
    tables = _extract_table_names_from_sql(hint)
    assert tables[:2] == ["audit_schedule", "audit_report"]


def test_returns_empty_list_for_blank_or_none_input():
    assert _extract_table_names_from_sql("") == []
    assert _extract_table_names_from_sql(None) == []


def test_does_not_crash_on_malformed_sql():
    assert _extract_table_names_from_sql("not really sql at all $$$ FROM (") == []


# ---------------------------------------------------------------------------
# _select_schema_block's fallback_tables wiring
# ---------------------------------------------------------------------------

def test_fallback_tables_adds_join_path_hint_when_relevant_tables_absent():
    schema = _make_schema()
    out = _select_schema_block(
        schema, None, fallback_tables=["audit_schedule", "audit_report"],
        join_graph_module=audit_join_graph,
    )
    assert "JOIN PATH" in out
    assert "arep.schedule_id = asch.id" in out


def test_fallback_tables_ignored_when_relevant_tables_explicitly_given():
    """Explicit relevant_tables still wins and switches to the full-detail
    block, exactly as before this fix — fallback_tables must never override
    or blend with an explicit, non-empty relevant_tables list."""
    schema = _make_schema()
    out = _select_schema_block(
        schema, ["facility"], fallback_tables=["audit_schedule", "audit_report"],
    )
    assert "name" in out  # facility's own column detail block
    assert "JOIN PATH" not in out  # only 1 explicit table — no hint expected


def test_fallback_tables_with_fewer_than_two_produces_no_hint():
    schema = _make_schema()
    out = _select_schema_block(schema, None, fallback_tables=["audit_schedule"])
    assert "JOIN PATH" not in out


def test_fallback_tables_none_preserves_original_slim_behavior():
    schema = _make_schema()
    baseline = schema.for_sql_prompt_slim()
    out = _select_schema_block(schema, None, fallback_tables=None)
    assert out == baseline


def test_unresolvable_fallback_tables_degrade_gracefully():
    schema = _make_schema()
    out = _select_schema_block(schema, None, fallback_tables=["not_a_real_table", "also_fake"])
    assert isinstance(out, str)
    assert "JOIN PATH" not in out


# ---------------------------------------------------------------------------
# End-to-end reproduction of both confirmed-live bugs — the hint that should
# have been available (but wasn't, before this fix) for the exact SQL each
# reasoning-model call actually produced.
# ---------------------------------------------------------------------------

def test_20260804_live_bug_now_gets_the_correct_hint():
    schema = _make_schema()
    hint = (
        "SELECT arep.audit_id, aq.label, aaa.answer_text "
        "FROM audit_schedule asch "
        "JOIN audit_report arep ON arep.schedule_id = asch.id "
        "WHERE asch.schedule_id = (SELECT id FROM audit_schedule WHERE schedule_id = '2026/08/ST040/SCH001');"
    )
    fallback = _extract_table_names_from_sql(hint)
    out = _select_schema_block(
        schema, None, fallback_tables=fallback,
        join_graph_module=audit_join_graph,
    )
    assert "arep.schedule_id = asch.id" in out


def test_20260730_live_bug_now_gets_the_correct_hint():
    schema = _make_schema()
    hint = (
        "SELECT DISTINCT u_stk.first_name || ' ' || u_stk.last_name AS stakeholder_name "
        "FROM audit_plan AS ap "
        "JOIN audit_schedule AS asch ON asch.audit_plan_id = ap.id "
        "JOIN audit_corrective_action AS aca ON aca.audit_id = "
        "(SELECT arep.audit_id FROM audit_report arep WHERE arep.schedule_id = asch.id) "
        "JOIN users u_stk ON aca.internal_stakeholder_id = u_stk.id;"
    )
    fallback = _extract_table_names_from_sql(hint)
    out = _select_schema_block(
        schema, None, fallback_tables=fallback,
        join_graph_module=audit_join_graph,
    )
    assert "aca.audit_id = arep.id" in out
    # The buggy form the model actually produced must never appear as "the fix":
    assert "aca.audit_id = arep.audit_id" not in out
