"""
Regression tests for tool_dispatcher.py's _reconstruct_fragmented_schema_hint
generalization — Item 3 of the 2026-08-07 generation-quality plan
(qid=e63f03af).

Before this fix, the function bailed out unconditionally whenever
"schema_hint" wasn't a key in args (`if "schema_hint" not in args: return
args`). Since the new planner->SQL contract moves SQL out of schema_hint
and into draft_sql, a planner emitting only draft_sql got zero repair for
Qwen's Python-style string-concatenation JSON corruption — the exact
corruption class this function exists to fix. These tests prove draft_sql
now gets the same repair, and that the legacy schema_hint-only shape still
works exactly as before.
"""

from src.agent.tool_dispatcher import _validate_tool_call


def _fragmented(key: str) -> dict:
    """The corruption shape documented in tool_dispatcher.py: Qwen writes
    Python-style string concatenation, json_repair splits it into separate
    top-level keys that look like SQL clause continuations."""
    return {
        "thought": "generating sql",
        "tool": "generate_sql",
        "args": {
            "question": "list audit plans",
            key: "SELECT ap.title",
            "FROM audit_plan ap": "WHERE ap.active = true",
        },
    }


def test_fragmented_draft_sql_is_reassembled():
    result = _validate_tool_call(_fragmented("draft_sql"), raw_output="<raw>")
    args = result["args"]
    assert set(args.keys()) == {"question", "draft_sql"}
    assert "SELECT ap.title" in args["draft_sql"]
    assert "FROM audit_plan ap" in args["draft_sql"]
    assert "WHERE ap.active = true" in args["draft_sql"]


def test_fragmented_schema_hint_still_reassembled():
    """Unchanged-behavior guard: the pre-existing legacy shape (all SQL
    crammed into schema_hint) must keep working exactly as before."""
    result = _validate_tool_call(_fragmented("schema_hint"), raw_output="<raw>")
    args = result["args"]
    assert "FROM audit_plan ap" not in args
    assert "SELECT ap.title" in args["schema_hint"]
    assert "FROM audit_plan ap" in args["schema_hint"]
    assert "WHERE ap.active = true" in args["schema_hint"]


def test_both_present_prefers_draft_sql():
    parsed = {
        "thought": "generating sql",
        "tool": "generate_sql",
        "args": {
            "question": "list audit plans",
            "draft_sql": "SELECT ap.title",
            "schema_hint": "some unrelated notes",
            "FROM audit_plan ap": "WHERE ap.active = true",
        },
    }
    result = _validate_tool_call(parsed, raw_output="<raw>")
    args = result["args"]
    assert "FROM audit_plan ap" not in args
    assert "FROM audit_plan ap" in args["draft_sql"]
    assert "WHERE ap.active = true" in args["draft_sql"]
    # schema_hint is untouched — the fragments fold onto draft_sql only.
    assert args["schema_hint"] == "some unrelated notes"


def test_no_fragmentation_leaves_args_untouched():
    parsed = {
        "thought": "generating sql",
        "tool": "generate_sql",
        "args": {"question": "list audit plans", "draft_sql": "SELECT 1"},
    }
    result = _validate_tool_call(parsed, raw_output="<raw>")
    assert result["args"] == {"question": "list audit plans", "draft_sql": "SELECT 1"}
