"""
ToolDef declaration for resolve_selected_rows — the resolution logic itself
lives in agent/orchestrator/row_selection.py; this module is purely the
wiring metadata (see agent/tool_defs package docstring).
"""

from src.agent.tool_defs.base import ArgSpec, ToolDef
from src.agent.orchestrator.row_selection import resolve_selected_rows as _resolve


def _context_provider(loop_locals: dict) -> dict:
    return {"session": loop_locals["session"], "domain": loop_locals["domain"]}


def _format_result(tool_result: dict, question: str) -> str:
    if not tool_result.get("success"):
        return (
            f"resolve_selected_rows failed: {tool_result.get('error')}. "
            f"Ask the user to clarify which rows they mean — do not guess."
        )

    r = tool_result.get("result") or {}

    if r.get("needs_clarification"):
        return (
            f"resolve_selected_rows could not resolve the selected rows: "
            f"{r.get('clarification_message')}\n"
            f"Call final_answer now asking the user to clarify — do NOT "
            f"guess via generate_sql."
        )

    target_table = r.get("target_table") or ""
    target_column = r.get("target_column") or ""
    target_ids = r.get("target_ids") or []
    display_titles = r.get("display_titles") or []
    entity_type = r.get("entity_type") or "item"

    id_list = ", ".join(f"'{i}'" for i in target_ids)
    _table_note = f" in table {target_table}" if target_table else ""
    msg = (
        f"Resolved {len(target_ids)} selected row(s) ({entity_type}) -> "
        f"{target_column} IN ({id_list}){_table_note} "
        f"(display values: {display_titles!r}).\n"
        f"Use WHERE <alias>.{target_column} IN ({id_list}) in your NEXT "
        f"generate_sql call — NOT title/ILIKE matching, even if a display "
        f"value looks unique."
    )
    if r.get("duplicate_title_positions"):
        msg += (
            f"\nNOTE: position(s) {r['duplicate_title_positions']} share a "
            f"display value with another row in that list — this is fine, "
            f"resolution already used the real row id above, not the "
            f"display value, so it's still exact."
        )
    return msg


RESOLVE_SELECTED_ROWS = ToolDef(
    name="resolve_selected_rows",
    args=(),
    fn=_resolve,
    doc=(
        '  12. resolve_selected_rows — { }\n'
        '                         Call this FIRST — before generate_sql — whenever the\n'
        '                         question refers to rows the user explicitly checked via\n'
        '                         checkboxes in the UI ("these", "them", "the selected ones",\n'
        '                         "those") rather than a position in words or a fresh name\n'
        '                         search. Takes no arguments — the selected rows are already\n'
        '                         known server-side. Returns the exact row ids to filter on.\n'
        '                         Do NOT use resolve_ordinal_reference or guess via title/ILIKE\n'
        '                         matching for this — the SELECTED ROWS context block (when\n'
        '                         present) tells you when this applies.\n'
    ),
    terminal=False,
    domains=("inspection", "audit"),
    context_provider=_context_provider,
    format_result=_format_result,
)
