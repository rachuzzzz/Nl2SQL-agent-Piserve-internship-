"""
Tool Dispatcher — JSON parsing and routing layer.

Responsibilities:
  parse_tool_call(raw_output) — extract a tool-call dict from the reasoning
      LLM's raw text output, which may contain preamble, markdown fences, or extra prose.
  dispatch(tool_call, registry) — validate the tool-call dict and route it to
      the correct ToolRegistry function, returning a standardised result.

Both functions are stateless and independently testable.
"""

import json
import re
from typing import Any

from agent.tools import ToolRegistry


# ---------------------------------------------------------------------------
# parse_tool_call
# ---------------------------------------------------------------------------

# Only "tool" is truly required — "args" defaults to {} when absent
_REQUIRED_KEYS = {"tool"}

# Patterns that LLMs wrap around their JSON output
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Match the outermost {...} block (greedy — captures the longest valid object)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_tool_call(raw_output: str) -> dict[str, Any]:
    """
    Extract a tool-call JSON object from the reasoning LLM's raw text output.

    Handles:
      - Clean JSON (ideal case)
      - JSON wrapped in ```json ... ``` fences
      - JSON with preamble/trailing prose
      - Nested JSON (finds the outermost {} block)

    Returns a dict with at least:
      { "thought": str, "tool": str, "args": dict }

    Raises ValueError with a descriptive message if no valid JSON is found
    or if required keys are missing.
    """
    if not raw_output or not raw_output.strip():
        raise ValueError("Empty output from reasoning LLM.")

    text = raw_output.strip()

    # 1. Try to unwrap markdown fences first
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Try direct parse (ideal case — model output is clean JSON)
    try:
        parsed = json.loads(text)
        return _validate_tool_call(parsed, raw_output)
    except json.JSONDecodeError:
        pass

    # 3. Extract the outermost {...} block and retry
    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        candidate = block_match.group(0)
        try:
            parsed = json.loads(candidate)
            return _validate_tool_call(parsed, raw_output)
        except json.JSONDecodeError:
            pass

    # 4. Last resort — walk through the text trying every possible JSON boundary
    for start in range(len(text)):
        if text[start] != "{":
            continue
        for end in range(len(text), start, -1):
            if text[end - 1] != "}":
                continue
            try:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, dict) and "tool" in parsed:
                    return _validate_tool_call(parsed, raw_output)
            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"Could not extract a valid JSON tool call from model output.\n"
        f"Raw output (first 400 chars): {raw_output[:400]!r}"
    )


def _validate_tool_call(parsed: Any, raw_output: str) -> dict[str, Any]:
    """
    Ensure the parsed object has the required structure.
    Fills in missing optional fields with safe defaults.
    """
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a JSON object (dict), got {type(parsed).__name__}. "
            f"Raw: {raw_output[:200]!r}"
        )

    missing = _REQUIRED_KEYS - parsed.keys()
    if missing:
        raise ValueError(
            f"Tool call JSON is missing required keys: {missing}. "
            f"Got keys: {set(parsed.keys())}. Raw: {raw_output[:200]!r}"
        )

    # "args" is optional — default to empty dict when absent or null
    if "args" not in parsed or parsed["args"] is None:
        parsed["args"] = {}
    elif not isinstance(parsed["args"], dict):
        raise ValueError(
            f"'args' must be a JSON object, got: {parsed.get('args')!r}"
        )

    # Normalise: ensure "thought" key exists (model sometimes omits it)
    parsed.setdefault("thought", "")

    return parsed


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def dispatch(tool_call: dict[str, Any], registry: ToolRegistry) -> dict[str, Any]:
    """
    Route a validated tool-call dict to the ToolRegistry.

    Does NOT raise — exceptions from the tool are caught and returned as
    {"success": False, "result": None, "error": "<message>"}.

    Special tool names handled here (not in ToolRegistry):
      - "final_answer": passes through directly.
    """
    tool_name = tool_call.get("tool", "")
    args = tool_call.get("args", {})

    # final_answer is handled by the orchestrator, not the registry
    if tool_name == "final_answer":
        return {
            "success": True,
            "result": args.get("answer", ""),
            "error": None,
        }

    try:
        result = registry.call(tool_name, args)
    except Exception as exc:
        result = {
            "success": False,
            "result": None,
            "error": f"Dispatcher caught unhandled exception in '{tool_name}': {exc}",
        }

    return result
