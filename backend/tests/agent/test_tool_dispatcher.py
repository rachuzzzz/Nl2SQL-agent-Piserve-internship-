"""
Regression tests for src/agent/tool_dispatcher.py's JSON-repair fragmentation
fixes — parse_tool_call()/_validate_tool_call() shouldn't lose part of a
model's answer to json_repair's truncated-JSON heuristics.
"""

from src.agent.tool_dispatcher import _validate_tool_call, _KNOWN_TOOLS
from src.agent.tools import ToolRegistry
from src.agent.tool_defs import declared_names


def test_fragmented_final_answer_is_reassembled():
    """
    Confirmed live 2026-07-30 (debug_llm_20260730_104043.log, question='audit'):
    the reasoning LLM's raw output was truncated mid-answer (missing closing
    brace). The answer text contained "...proposed audits, status: DRAFT,
    SUBMITTED, APPROVED)..." — json_repair's recovery mistook "status" for a
    new JSON key instead of prose inside the still-open answer string,
    producing args={"answer": "...proposed audits", "status": "DRAFT, ..."}.
    Since dispatch() only ever reads args["answer"] for final_answer, the
    user was silently shown the truncated half — missing the rest of the
    list and the closing prompt sentence.
    """
    parsed = {
        "thought": "recovered",
        "tool": "final_answer",
        "args": {
            "answer": (
                "I can help you with various aspects of audits. What "
                "specifically would you like to know?\n\nI can provide "
                "information about:\n- **Audit Plans** (proposed audits"
            ),
            "status": (
                "DRAFT, SUBMITTED, APPROVED)\n- **Audit Schedules** "
                "(scheduled audit visits with dates, auditors, facilities)"
            ),
        },
    }

    result = _validate_tool_call(parsed, raw_output="<raw>")

    assert list(result["args"].keys()) == ["answer"]
    answer = result["args"]["answer"]
    assert "proposed audits, status: DRAFT, SUBMITTED, APPROVED)" in answer
    assert "**Audit Schedules**" in answer


def test_final_answer_without_fragmentation_is_untouched():
    parsed = {
        "thought": "",
        "tool": "final_answer",
        "args": {"answer": "5 audit plans found."},
    }

    result = _validate_tool_call(parsed, raw_output="<raw>")

    assert result["args"] == {"answer": "5 audit plans found."}


def test_non_final_answer_tool_keeps_multiple_args():
    """generate_sql legitimately has more than one arg — must not be touched
    by the final_answer-specific reassembly."""
    parsed = {
        "thought": "",
        "tool": "generate_sql",
        "args": {"question": "list plans", "schema_hint": "SELECT ..."},
    }

    result = _validate_tool_call(parsed, raw_output="<raw>")

    assert result["args"] == {"question": "list plans", "schema_hint": "SELECT ..."}


def test_known_tools_stays_in_sync_with_registry_and_declared_tools():
    """Regression guard for the "5 places to touch" drift risk the
    agent/tool_defs framework exists to prevent: _KNOWN_TOOLS (used for
    parse_tool_call's regex-fallback recovery) must always be derivable
    from ToolRegistry.KNOWN_TOOLS + agent/tool_defs' declared tools —
    never a fourth, silently-diverging hand-maintained list."""
    expected = ToolRegistry.KNOWN_TOOLS | declared_names() | {"final_answer", "redirect_domain"}
    assert _KNOWN_TOOLS == expected
    # Concretely confirms declared tools (not just the original 8) are
    # actually reachable via this set, not merely that the set equation
    # happens to hold.
    assert "resolve_ordinal_reference" in _KNOWN_TOOLS
