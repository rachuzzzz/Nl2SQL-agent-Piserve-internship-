"""
ToolDef declaration for resolve_ordinal_reference — the resolution logic
itself lives in agent/orchestrator/ordinal.py; this module is purely the
wiring metadata (see agent/tool_defs package docstring).
"""

from src.agent.tool_defs.base import ArgSpec, ToolDef
from src.agent.orchestrator.ordinal import resolve_ordinal_reference as _resolve


def _context_provider(loop_locals: dict) -> dict:
    return {"session": loop_locals["session"], "domain": loop_locals["domain"]}


def _format_result(tool_result: dict, question: str) -> str:
    if not tool_result.get("success"):
        return (
            f"resolve_ordinal_reference failed: {tool_result.get('error')}. "
            f"Ask the user to clarify which item they mean — do not guess."
        )

    r = tool_result.get("result") or {}

    if r.get("needs_clarification"):
        return (
            f"resolve_ordinal_reference could not resolve this reference: "
            f"{r.get('clarification_message')}\n"
            f"Call final_answer now asking the user to clarify — do NOT "
            f"guess via generate_sql."
        )

    target_table = r.get("target_table") or ""
    target_column = r.get("target_column") or ""
    target_id = r.get("target_id") or ""
    display_title = r.get("display_title") or ""
    warning = r.get("warning")

    if not target_id:
        # No real row id was resolvable for this position (e.g. the prior
        # turn's SQL had a GROUP BY, which id-injection skips) — fall back
        # to the display value, but the duplicate-title warning (when
        # present) is the load-bearing safety note here.
        msg = (
            f"Resolved position {r.get('position')} to display value "
            f"{display_title!r} — no real row id was available for this "
            f"item, so filter on this exact display value in your NEXT "
            f"generate_sql call."
        )
    else:
        _table_note = f" in table {target_table}" if target_table else ""
        msg = (
            f"Resolved position {r.get('position')} ({r.get('entity_type') or 'item'}) "
            f"-> {target_column} = '{target_id}'{_table_note} "
            f"(display value {display_title!r}).\n"
            f"Use WHERE <alias>.{target_column} = '{target_id}' in your NEXT "
            f"generate_sql call — NOT title/ILIKE, even if this display value "
            f"appears more than once in that list."
        )

    if warning:
        msg += f"\nWARNING: {warning}"

    return msg


RESOLVE_ORDINAL_REFERENCE = ToolDef(
    name="resolve_ordinal_reference",
    args=(
        ArgSpec("position", "int", required=True,
                doc="1-based ordinal position, e.g. 1 for 'the first one', 3 for 'the 3rd item'"),
        ArgSpec("entity_type_hint", "string|null", required=False,
                doc="optional short noun naming what kind of item is meant, e.g. 'plan', 'corrective action', 'auditor' — omit for a bare reference like 'the first one'"),
    ),
    fn=_resolve,
    doc=(
        '  10. resolve_ordinal_reference — { "position": 1, "entity_type_hint": null }\n'
        '                         Call this FIRST — before generate_sql — whenever the\n'
        '                         question refers to a specific item from a PREVIOUS\n'
        '                         result by position ("the first one", "the 2nd item",\n'
        '                         "the last one") rather than by name. Returns the exact\n'
        '                         row id to filter on. Do NOT guess it yourself via\n'
        '                         title/ILIKE matching in generate_sql — display values\n'
        '                         are not guaranteed unique.\n'
    ),
    terminal=False,
    domains=("inspection", "audit"),
    context_provider=_context_provider,
    format_result=_format_result,
)
