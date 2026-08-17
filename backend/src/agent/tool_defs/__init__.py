"""
DECLARED_TOOLS — a shared declaration point for reasoning-LLM tools that
don't fit the hand-written ToolRegistry.call() if/elif chain in
agent/tools.py (see that file's module docstring for why the original 8
tools stay there rather than migrating).

A tool declared here is registered ONCE (name, arg schema, prompt doc,
terminal flag, and optional context/format-result hooks — see base.py's
ToolDef) and that single declaration drives:
  - ToolRegistry.call()'s dispatch (agent/tools.py)
  - tool_dispatcher.py's regex-fallback recovery set
  - the system prompt's AVAILABLE TOOLS block (agent/prompts/builder*.py,
    via the <<<DYNAMIC_TOOLS_BLOCK>>> sentinel)
  - AgentOrchestrator's terminal-vs-non-terminal tool handling (loop.py)
  - synthesis.py's next-turn tool-result formatting

Registry is built lazily on first access (mirrors agent/domains.py's
get_domain_config pattern) rather than importing every tool module at
package-import time — avoids import-order/cycle issues between this
package and orchestrator modules that a tool's implementation function
may live in.
"""

from src.agent.tool_defs.base import ArgSpec, ToolDef

_DECLARED: dict[str, ToolDef] | None = None


def _build_registry() -> dict[str, ToolDef]:
    from src.agent.tool_defs.ordinal import RESOLVE_ORDINAL_REFERENCE
    from src.agent.tool_defs.disambiguation_reply import RESOLVE_DISAMBIGUATION_REPLY
    from src.agent.tool_defs.row_selection import RESOLVE_SELECTED_ROWS

    tools = (RESOLVE_ORDINAL_REFERENCE, RESOLVE_DISAMBIGUATION_REPLY, RESOLVE_SELECTED_ROWS)
    return {t.name: t for t in tools}


def _registry() -> dict[str, ToolDef]:
    global _DECLARED
    if _DECLARED is None:
        _DECLARED = _build_registry()
    return _DECLARED


def get_declared(name: str) -> ToolDef | None:
    return _registry().get(name)


def declared_tools() -> tuple[ToolDef, ...]:
    return tuple(_registry().values())


def declared_names() -> set[str]:
    return set(_registry().keys())


__all__ = ["ArgSpec", "ToolDef", "get_declared", "declared_tools", "declared_names"]
