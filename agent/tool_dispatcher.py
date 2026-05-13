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

try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

from agent.tools import ToolRegistry


# ---------------------------------------------------------------------------
# parse_tool_call
# ---------------------------------------------------------------------------

# Only "tool" is truly required — "args" defaults to {} when absent
_REQUIRED_KEYS = {"tool"}

# Markdown fence stripper
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Finds the first { and everything after — used to strip leading prose
_FIRST_BRACE_RE = re.compile(r"\{", re.DOTALL)

# Known tool names — used for regex fallback extraction
_KNOWN_TOOLS = {
    "list_forms", "semantic_search", "search_questions",
    "get_answers", "get_answer_stats",
    "generate_sql", "execute_sql", "get_schema",
    "final_answer",
}
_TOOL_NAMES_PATTERN = "|".join(re.escape(t) for t in sorted(_KNOWN_TOOLS, key=len, reverse=True))
_TOOL_RE = re.compile(
    rf'"tool"\s*:\s*"({_TOOL_NAMES_PATTERN})"',
    re.IGNORECASE,
)


# SQL clause keywords that appear as fragment keys when Qwen emits Python-style
# multiline string concatenation.  Example of the corruption:
#
#   Qwen intent (Python source):
#       "schema_hint": (
#           "SELECT ...\n"
#           + "FROM inspection_report ir\n"
#           + "WHERE ir.status != 'DRAFT'"
#       )
#
#   JSON received by parser (json_repair sees separate string literals):
#       "schema_hint": "SELECT ...",
#       "FROM inspection_report ir": "WHERE ir.status != 'DRAFT'",
#       ...
#
# We reconstruct the full schema_hint by detecting these SQL fragment keys and
# appending them in document order to the first "schema_hint" value.

_SQL_FRAGMENT_PREFIXES = (
    "FROM ", "JOIN ", "LEFT JOIN ", "RIGHT JOIN ", "INNER JOIN ", "OUTER JOIN ",
    "WHERE ", "GROUP BY ", "HAVING ", "ORDER BY ", "LIMIT ", "OFFSET ",
    "UNION ALL", "UNION ", "EXCEPT ", "INTERSECT ",
    "ON ", "AND ", "OR ",
    "WITH ", "AS (", "AS(",
    "SELECT ",  # second SELECT in UNION
)


def _reconstruct_fragmented_schema_hint(args: dict) -> dict:
    """
    Detect and repair fragmented schema_hint caused by Python-style string
    concatenation in Qwen's JSON output.

    When Qwen writes:
        "schema_hint": "SELECT x\n" + "FROM t\n" + "WHERE ..."
    the parser receives THREE separate keys instead of one full string.

    We detect keys that look like SQL clause continuations (starting with
    FROM, WHERE, GROUP BY, etc.) and append them to schema_hint in order.

    Returns a cleaned args dict with the full schema_hint and fragment keys removed.
    """
    if "schema_hint" not in args:
        return args

    # Collect keys in insertion order — Python dicts preserve insertion order
    fragment_keys = [
        k for k in args
        if k != "schema_hint"
        and isinstance(k, str)
        and any(k.upper().startswith(prefix.upper()) for prefix in _SQL_FRAGMENT_PREFIXES)
    ]

    if not fragment_keys:
        return args  # no corruption detected

    # Reconstruct: start with existing (partial) schema_hint, append each fragment
    parts = [str(args["schema_hint"])]
    for key in fragment_keys:
        # The fragment value is the clause that follows this key
        # e.g. key="FROM inspection_report ir", value="WHERE ir.status != 'DRAFT'"
        # → append "\nFROM inspection_report ir\nWHERE ir.status != 'DRAFT'"
        parts.append("\n" + key)
        val = args[key]
        if val and str(val).strip():
            parts.append("\n" + str(val))

    cleaned = dict(args)  # shallow copy preserving other args
    cleaned["schema_hint"] = "".join(parts)
    for key in fragment_keys:
        cleaned.pop(key, None)

    return cleaned


def _try_parse(text: str, raw_output: str) -> dict[str, Any] | None:
    """Try json.loads then json_repair on a candidate string. Returns None on failure."""
    # Standard parse first — zero overhead when the model is well-behaved
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool" in parsed:
            return _validate_tool_call(parsed, raw_output)
    except (json.JSONDecodeError, ValueError):
        pass

    # json_repair: handles truncated JSON, unquoted keys, real newlines in strings,
    # single quotes, trailing commas, missing closing brackets — all common LLM bugs
    if _HAS_JSON_REPAIR:
        try:
            repaired = json_repair.repair_json(text, return_objects=True)
            # repair_json may return a list if it found multiple objects; take first dict with "tool"
            candidates = repaired if isinstance(repaired, list) else [repaired]
            for obj in candidates:
                if isinstance(obj, dict) and "tool" in obj:
                    return _validate_tool_call(obj, raw_output)
        except Exception:
            pass

    return None


def parse_tool_call(raw_output: str) -> dict[str, Any]:
    """
    Extract a tool-call JSON object from the reasoning LLM's raw text output.

    Strategy (in order of cost):
      1. Strip markdown fences, try direct parse + json_repair on full text
      2. Find the first '{' — strip leading prose — try parse + json_repair
      3. Scan all '{' start positions, try each suffix with json_repair
         (handles truncated JSON where the closing '}' is missing)
      4. Regex fallback — extract "tool" and reconstruct minimal dict

    Returns a dict with at least { "thought": str, "tool": str, "args": dict }
    Raises ValueError only if no tool name can be recovered at all.
    """
    if not raw_output or not raw_output.strip():
        raise ValueError("Empty output from reasoning LLM.")

    text = raw_output.strip()

    # ── Step 1: strip markdown fences ────────────────────────────────────────
    fence_match = _FENCE_RE.search(text)
    candidate = fence_match.group(1).strip() if fence_match else text

    result = _try_parse(candidate, raw_output)
    if result:
        return result

    # ── Step 2: strip leading prose (everything before first '{') ─────────────
    brace_match = _FIRST_BRACE_RE.search(candidate)
    if brace_match and brace_match.start() > 0:
        from_brace = candidate[brace_match.start():]
        result = _try_parse(from_brace, raw_output)
        if result:
            return result

    # ── Step 3: try each '{' position as a start — handles trailing prose and
    #           truncated JSON (json_repair closes open brackets/strings) ──────
    for m in _FIRST_BRACE_RE.finditer(candidate):
        suffix = candidate[m.start():]
        result = _try_parse(suffix, raw_output)
        if result:
            return result
        # Only scan first 5 '{' positions — avoid O(n) on deeply nested SQL
        if m.start() > 500:
            break

    # ── Step 4: regex fallback — if we can at least find the tool name, build
    #           a minimal valid dict so the orchestrator can continue ──────────
    tool_match = _TOOL_RE.search(raw_output)
    if tool_match:
        tool_name = tool_match.group(1)

        # Try to recover args from whatever JSON fragment exists
        args: dict = {}
        if _HAS_JSON_REPAIR:
            try:
                repaired = json_repair.repair_json(raw_output, return_objects=True)
                objs = repaired if isinstance(repaired, list) else [repaired]
                for obj in objs:
                    if isinstance(obj, dict):
                        args = obj.get("args", {}) or {}
                        break
            except Exception:
                pass

        return _validate_tool_call(
            {"thought": "[recovered]", "tool": tool_name, "args": args},
            raw_output,
        )

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

    # Repair fragmented schema_hint before returning — Qwen sometimes emits
    # Python-style multiline string concatenation which the JSON parser splits
    # into separate keys (FROM, WHERE, GROUP BY become top-level args keys).
    parsed["args"] = _reconstruct_fragmented_schema_hint(parsed["args"])

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