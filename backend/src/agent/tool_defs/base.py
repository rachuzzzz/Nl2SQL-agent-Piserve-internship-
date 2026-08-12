"""
ToolDef / ArgSpec — the declaration shape for tools registered through
agent/tool_defs (see package __init__ docstring). Plain frozen dataclasses,
not a decorator-based registry: a tool's *implementation* (the function
that does the work, e.g. agent/orchestrator/ordinal.py::resolve_ordinal_reference)
stays a normal, independently-testable function; a ToolDef in
agent/tool_defs/<name>.py is just the wiring metadata pointing at it.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str                      # prose only ("int", "string", "string|null") — not validated
    required: bool = False
    doc: str = ""                  # one-line description used when building example JSON


@dataclass(frozen=True)
class ToolDef:
    name: str
    args: tuple[ArgSpec, ...]
    fn: Callable[..., dict]        # called as fn(**call_kwargs) -> {"success","result","error"}
    doc: str                       # prose block appended to the AVAILABLE TOOLS prompt section
    terminal: bool = False         # True => success short-circuits straight to synthesis (loop.py)
    domains: tuple[str, ...] = ("inspection", "audit")
    # Given a small explicit dict of loop-local values (e.g. {"session":...,
    # "domain":...}), returns extra kwargs to merge into the dispatch call
    # for THIS tool only. Kept separate from the tool's own `args` dict
    # (which is stored verbatim on AgentStep and assumed JSON-safe) — see
    # loop.py's dispatch_args construction.
    context_provider: Optional[Callable[[dict], dict]] = None
    # Formats a successful/failed tool result into the message the LLM sees
    # on its NEXT iteration. Falls back to synthesis.py's generic
    # json.dumps(tool_result) when absent.
    format_result: Optional[Callable[[dict, str], str]] = None
