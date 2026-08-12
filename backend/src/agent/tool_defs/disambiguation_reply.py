"""
ToolDef declaration for resolve_disambiguation_reply — the resolution
logic itself lives in agent/orchestrator/disambiguation_reply.py; this
module is purely the wiring metadata (see agent/tool_defs package
docstring).
"""

from src.agent.tool_defs.base import ArgSpec, ToolDef
from src.agent.orchestrator.disambiguation_reply import resolve_disambiguation_reply as _resolve


def _context_provider(loop_locals: dict) -> dict:
    return {"session": loop_locals["session"]}


def _format_result(tool_result: dict, question: str) -> str:
    if not tool_result.get("success"):
        return (
            f"resolve_disambiguation_reply failed: {tool_result.get('error')}. "
            f"Ask the user to clarify which one they mean — do not guess."
        )

    r = tool_result.get("result") or {}

    if r.get("needs_clarification"):
        return (
            f"resolve_disambiguation_reply could not resolve this reply: "
            f"{r.get('clarification_message')}\n"
            f"Call final_answer now asking the user to clarify — do NOT "
            f"guess via generate_sql."
        )

    target_alias = r.get("target_alias") or "<alias>"
    target_id = r.get("target_id") or ""
    display_name = r.get("display_name") or ""

    msg = (
        f"Resolved to {display_name!r} ({r.get('kind') or 'item'}) -> "
        f"{target_alias}.id = '{target_id}'.\n"
        f"Use WHERE {target_alias}.id = '{target_id}' in your NEXT generate_sql "
        f"call — NOT ILIKE/substring matching on the name, even though it's "
        f"now resolved to a specific id."
    )
    if r.get("clarification_message"):
        # Non-empty here on a *resolved* result carries the "don't forget
        # the original request" reminder (disambiguation_reply.py's
        # origin_note), not an actual clarification ask.
        msg += f"\n{r['clarification_message']}"
    return msg


RESOLVE_DISAMBIGUATION_REPLY = ToolDef(
    name="resolve_disambiguation_reply",
    args=(
        ArgSpec("picked_name", "string|null", required=False,
                doc="the specific name the user picked from the previous candidate list, exactly as intended (e.g. 'George Smith') — omit if picking by position"),
        ArgSpec("picked_position", "int|null", required=False,
                doc="1-based position in the previously-shown candidate list ('option 2', 'the second one') — omit if picking by name"),
    ),
    fn=_resolve,
    doc=(
        '  11. resolve_disambiguation_reply — { "picked_name": "George Smith", "picked_position": null }\n'
        '                         Call this FIRST — before generate_sql or final_answer —\n'
        '                         whenever the previous turn asked "which one did you mean?"\n'
        '                         (a people-name or facility/project/client/organisation\n'
        '                         disambiguation list) and this message is the user picking\n'
        '                         one, by name or by position. Do NOT guess the id yourself\n'
        '                         via ILIKE/substring matching — call this tool to get the\n'
        '                         exact row id.\n'
    ),
    terminal=False,
    domains=("inspection", "audit"),
    context_provider=_context_provider,
    format_result=_format_result,
)
