"""
Deterministic tests for agent/tools.py::_select_schema_block() — the
relevant_tables wiring for knowledge/schema_introspector.py::for_schema_hint().

No LLM, no live DB — DatabaseSchema is constructed directly with fake table
data. Run directly (`python tests/test_schema_hint_selection.py`) or via
pytest if it's ever added to this project (test_*() functions are
pytest-discoverable as-is).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.knowledge.schema_introspector import DatabaseSchema, TableSchema
from src.sql.generation import _select_schema_block, _join_path_hints, _join_graph_source

FAILURES = []


def _check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def _make_schema() -> DatabaseSchema:
    schema = DatabaseSchema()
    schema.tables["inspection_cycle"] = TableSchema(
        name="inspection_cycle",
        columns=[
            {"name": "id", "type": "uuid", "nullable": False},
            {"name": "status", "type": "varchar", "nullable": True},
            {"name": "start_date", "type": "date", "nullable": True},
            {"name": "end_date", "type": "date", "nullable": True},
            {"name": "due_date", "type": "date", "nullable": True},
        ],
        foreign_keys=[],
        primary_key=["id"],
    )
    return schema


def test_none_falls_back_to_slim_unchanged():
    schema = _make_schema()
    baseline = schema.for_sql_prompt_slim()
    out = _select_schema_block(schema, None)
    _check("relevant_tables=None -> for_sql_prompt_slim() unchanged", out == baseline)


def test_valid_list_uses_targeted_block():
    schema = _make_schema()
    out = _select_schema_block(schema, ["inspection_cycle"])
    _check(
        "relevant_tables=['inspection_cycle'] -> real columns present",
        all(col in out for col in ("start_date", "end_date", "due_date", "status")),
        detail=out[:300],
    )
    # Confirm the gap this closes: inspection_cycle has no OWN entry in the
    # default slim block (it's not in USEFUL_COLUMNS or LOOKUP_TABLES) — check
    # for the table-listing pattern "  inspection_cycle:" specifically, since
    # bare substring checks collide with real content: "start_date" is inside
    # inspection_report's "start_date_time" column, and "inspection_cycle" is
    # inside inspection_schedule's "inspection_cycle_id" FK column name.
    slim = schema.for_sql_prompt_slim()
    _check(
        "inspection_cycle has no own table-listing entry in the default slim block (confirms the gap existed)",
        "inspection_cycle:" not in slim,
        detail=slim[:300],
    )


def test_bare_string_falls_back_safely():
    schema = _make_schema()
    baseline = schema.for_sql_prompt_slim()
    # A bare string instead of a JSON array — plausible LLM mistake, this is the
    # only list-typed tool argument in the codebase per this session's research.
    out = _select_schema_block(schema, "inspection_cycle")
    _check(
        "relevant_tables as bare string -> falls back to slim, does not iterate characters",
        out == baseline,
        detail=out[:200],
    )


def test_empty_list_falls_back():
    schema = _make_schema()
    baseline = schema.for_sql_prompt_slim()
    out = _select_schema_block(schema, [])
    _check("relevant_tables=[] -> falls back to slim", out == baseline)


def test_unknown_table_name_degrades_gracefully():
    schema = _make_schema()
    out = _select_schema_block(schema, ["not_a_real_table"])
    _check(
        "relevant_tables=['not_a_real_table'] -> no crash, non-empty output",
        isinstance(out, str) and len(out) > 0,
        detail=out[:200],
    )
    _check(
        "unknown table produces no fabricated column list for it",
        "not_a_real_table" not in out or "### not_a_real_table" not in out,
    )


def test_db_schema_none_falls_through_to_db_engine_or_empty():
    out = _select_schema_block(None, ["inspection_cycle"], db_engine=None)
    _check("db_schema=None, db_engine=None -> empty string", out == "")


# ---------------------------------------------------------------------------
# agent/join_graph.py wiring — path_as_hint() chained across relevant_tables
# ---------------------------------------------------------------------------

def test_two_tables_adds_join_path_hint():
    schema = _make_schema()
    out = _select_schema_block(schema, ["facility", "inspection_cycle"])
    _check(
        "relevant_tables=['facility','inspection_cycle'] -> JOIN PATH hint present",
        "JOIN PATH (fac" in out and "inspection_cycle ic" in out,
        detail=out[-400:],
    )
    _check(
        "column-detail block still present alongside the join-path hint",
        "due_date" in out,
    )


def test_users_normalizes_to_users_inspector():
    _check(
        "_join_graph_source('users') -> 'users_inspector'",
        _join_graph_source("users") == "users_inspector",
    )
    hints = _join_path_hints(["users", "inspection_schedule"])
    _check(
        "relevant_tables=['users','inspection_schedule'] -> real join-path hint, not silently empty",
        "JOIN PATH" in hints,
        detail=hints,
    )


def test_single_table_has_no_join_path_hint():
    schema = _make_schema()
    out = _select_schema_block(schema, ["inspection_cycle"])
    _check(
        "single-table relevant_tables -> no JOIN PATH hint (needs 2+)",
        "JOIN PATH" not in out,
    )


def test_bogus_table_in_pair_degrades_gracefully():
    schema = _make_schema()
    out = _select_schema_block(schema, ["facility", "not_a_real_table"])
    _check(
        "relevant_tables=['facility','not_a_real_table'] -> no crash",
        isinstance(out, str) and len(out) > 0,
        detail=out[:200],
    )
    _check(
        "no JOIN PATH hint fabricated for the bogus table",
        "JOIN PATH" not in out,
    )


if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in test_fns:
        try:
            fn()
        except Exception as e:
            print(f"[ERROR] {fn.__name__} raised {e!r}")
            FAILURES.append(fn.__name__)

    print()
    if FAILURES:
        print(f"{len(FAILURES)}/{len(test_fns)} FAILED: {FAILURES}")
        sys.exit(1)
    else:
        print(f"ALL {len(test_fns)} CHECKS PASSED")
