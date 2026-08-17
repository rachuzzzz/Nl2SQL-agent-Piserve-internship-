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

from src.agent.tools import ToolRegistry
from src.agent.tool_defs import declared_names as _declared_names


# ---------------------------------------------------------------------------
# parse_tool_call
# ---------------------------------------------------------------------------

# Only "tool" is truly required — "args" defaults to {} when absent
_REQUIRED_KEYS = {"tool"}

# Markdown fence stripper
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Finds the first { and everything after — used to strip leading prose
_FIRST_BRACE_RE = re.compile(r"\{", re.DOTALL)

# Known tool names — used for regex fallback extraction. Derived from
# ToolRegistry.KNOWN_TOOLS + agent/tool_defs' declared tools instead of a
# third independently-maintained list, so a new declared tool is
# automatically recoverable here without an extra edit.
#
# Computed LAZILY (module __getattr__ below, PEP 562) rather than at
# import time — a declared tool's implementation can live under
# agent/orchestrator/*, and agent/orchestrator/__init__.py eagerly imports
# loop.py, which imports THIS module; resolving tool_defs' registry at
# tool_dispatcher's own import time re-enters tool_defs mid-build (a real
# cycle, confirmed live 2026-08-05 while wiring resolve_ordinal_reference).
# Deferring until first actual use (parse_tool_call, well after all
# top-level imports have finished) breaks the cycle.
_known_tools_cache: set[str] | None = None
_tool_re_cache: "re.Pattern | None" = None


def _compute_known_tools() -> set[str]:
    return ToolRegistry.KNOWN_TOOLS | _declared_names() | {"final_answer", "redirect_domain"}


def _get_known_tools() -> set[str]:
    global _known_tools_cache
    if _known_tools_cache is None:
        _known_tools_cache = _compute_known_tools()
    return _known_tools_cache


def _get_tool_re() -> "re.Pattern":
    global _tool_re_cache
    if _tool_re_cache is None:
        _pattern = "|".join(re.escape(t) for t in sorted(_get_known_tools(), key=len, reverse=True))
        _tool_re_cache = re.compile(rf'"tool"\s*:\s*"({_pattern})"', re.IGNORECASE)
    return _tool_re_cache


def __getattr__(name: str):
    # PEP 562 module-level lazy attribute — lets `from src.agent.tool_dispatcher
    # import _KNOWN_TOOLS` keep working unchanged for existing callers/tests
    # while the underlying set is computed lazily, not at import time.
    if name == "_KNOWN_TOOLS":
        return _get_known_tools()
    if name == "_TOOL_RE":
        return _get_tool_re()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    Detect and repair fragmented schema_hint/draft_sql caused by
    Python-style string concatenation in Qwen's JSON output.

    When Qwen writes:
        "schema_hint": "SELECT x\n" + "FROM t\n" + "WHERE ..."
    the parser receives THREE separate keys instead of one full string.

    We detect keys that look like SQL clause continuations (starting with
    FROM, WHERE, GROUP BY, etc.) and append them to whichever of
    draft_sql/schema_hint is present, preferring draft_sql.

    draft_sql is the generation-quality plan's Item 3 field for
    planner-authored SQL (schema_hint is now reserved for freeform notes,
    see generation.py's _build_additional_context). Confirmed by direct
    read: this function used to bail out unconditionally whenever
    "schema_hint" was absent (`if "schema_hint" not in args: return args`)
    — once a planner emits draft_sql instead of cramming SQL into
    schema_hint, that left this exact corruption class with zero repair,
    silently. Checking draft_sql first (falling back to schema_hint for
    any caller not yet updated to the new shape) closes that gap without
    changing behavior for legacy callers.

    Returns a cleaned args dict with the full draft_sql/schema_hint and
    fragment keys removed.
    """
    target_key = "draft_sql" if "draft_sql" in args else ("schema_hint" if "schema_hint" in args else None)
    if target_key is None:
        return args

    # Collect keys in insertion order — Python dicts preserve insertion order
    fragment_keys = [
        k for k in args
        if k != target_key
        and isinstance(k, str)
        and any(k.upper().startswith(prefix.upper()) for prefix in _SQL_FRAGMENT_PREFIXES)
    ]

    if not fragment_keys:
        return args  # no corruption detected

    # Reconstruct: start with existing (partial) draft_sql/schema_hint, append each fragment
    parts = [str(args[target_key])]
    for key in fragment_keys:
        # The fragment value is the clause that follows this key
        # e.g. key="FROM inspection_report ir", value="WHERE ir.status != 'DRAFT'"
        # → append "\nFROM inspection_report ir\nWHERE ir.status != 'DRAFT'"
        parts.append("\n" + key)
        val = args[key]
        if val and str(val).strip():
            parts.append("\n" + str(val))

    cleaned = dict(args)  # shallow copy preserving other args
    cleaned[target_key] = "".join(parts)
    for key in fragment_keys:
        cleaned.pop(key, None)

    return cleaned


def _reconstruct_fragmented_answer(args: dict) -> dict:
    """
    Repair a final_answer's `answer` string fragmented into spurious extra
    keys by json_repair's heuristics.

    Confirmed live 2026-07-30: the reasoning LLM's raw output was truncated
    mid-answer (missing closing brace — "TRUNCATED — 1 unclosed brace(s)" in
    debug_llm log). The answer text itself happened to contain a
    "word: value"-shaped substring ("...proposed audits, status: DRAFT,
    SUBMITTED, APPROVED)..."), and json_repair's truncated-JSON recovery
    mistook "status" for a new key instead of prose inside the still-open
    answer string — splitting one continuous sentence into
    args={"answer": "...proposed audits", "status": "DRAFT, SUBMITTED, ..."}.
    dispatch() only ever reads args["answer"] for final_answer (see below),
    so the user silently received the answer text truncated mid-sentence —
    missing the rest of the list and the closing prompt — while the log's
    "PARSED TOOL CALL" line showed the (unused) "status" fragment sitting
    right next to it, looking harmless.

    Unlike the schema_hint fragmentation above, the spurious key here isn't
    drawn from a fixed SQL-clause vocabulary — it's an arbitrary word from
    the model's prose, so there's no prefix list to match against. But
    final_answer only has ONE meaningful arg ("answer"); ANY other key
    present on a final_answer call is necessarily this same corruption, not
    a real separate field, so it's always safe to fold every extra key back
    onto the end of the answer in insertion order (dicts preserve it).
    """
    if "answer" not in args:
        return args
    extra_keys = [k for k in args if k != "answer" and isinstance(k, str)]
    if not extra_keys:
        return args

    parts = [str(args["answer"])]
    for key in extra_keys:
        val = args[key]
        parts.append(f", {key}: {val}" if val not in (None, "") else f", {key}")

    cleaned = dict(args)
    cleaned["answer"] = "".join(parts)
    for key in extra_keys:
        cleaned.pop(key, None)
    return cleaned


def _escape_newlines_in_strings(text: str) -> str:
    """Escape literal newlines/carriage-returns inside JSON string values.

    LLMs sometimes emit multi-line answers as literal newlines rather than \\n,
    producing invalid JSON that json_repair cannot always recover when combined
    with a truncated (missing closing brace) output.
    """
    result = []
    in_string = False
    escape_next = False
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == '\\':
            result.append(char)
            escape_next = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif in_string and char == '\n':
            result.append('\\n')
        elif in_string and char == '\r':
            result.append('\\r')
        else:
            result.append(char)
    return ''.join(result)


def _try_parse(text: str, raw_output: str) -> dict[str, Any] | None:
    """Try json.loads then json_repair on a candidate string. Returns None on failure."""
    # Build candidate list: preprocessed version first (escapes literal newlines in strings),
    # then original as fallback. Preprocessing is tried first because json_repair on the
    # original with literal newlines + a missing closing brace returns args:{} — appearing
    # to succeed while silently losing the answer content.
    preprocessed = _escape_newlines_in_strings(text)
    candidates_to_try = [preprocessed, text] if preprocessed != text else [text]

    for candidate in candidates_to_try:
        # Standard parse — zero overhead when model output is well-formed
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "tool" in parsed:
                return _validate_tool_call(parsed, raw_output)
        except (json.JSONDecodeError, ValueError):
            pass

        # json_repair: handles truncated JSON, unquoted keys, trailing commas,
        # missing closing brackets — all common LLM bugs
        if _HAS_JSON_REPAIR:
            try:
                repaired = json_repair.repair_json(candidate, return_objects=True)
                # repair_json may return a list if it found multiple objects
                objs = repaired if isinstance(repaired, list) else [repaired]
                for obj in objs:
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
    tool_match = _get_tool_re().search(raw_output)
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

    # Same repair, targeted at final_answer's `answer` string — see
    # _reconstruct_fragmented_answer's docstring for the live-caught case.
    # Deliberately NOT applied to redirect_domain: that repair's whole premise
    # is "final_answer only has ONE meaningful arg, so any other key is
    # necessarily corruption" — but redirect_domain legitimately has a SECOND
    # meaningful arg (suspected_domain) always present, so running it through
    # this repair would unconditionally mangle suspected_domain into the
    # answer text on every single well-formed call, not just fragmented ones.
    if parsed.get("tool") == "final_answer":
        parsed["args"] = _reconstruct_fragmented_answer(parsed["args"])

    # Normalise: ensure "thought" key exists (model sometimes omits it)
    parsed.setdefault("thought", "")

    return parsed


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def dispatch(tool_call: dict[str, Any], registry: ToolRegistry,
             override_args: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Route a validated tool-call dict to the ToolRegistry.

    Does NOT raise — exceptions from the tool are caught and returned as
    {"success": False, "result": None, "error": "<message>"}.

    Special tool names handled here (not in ToolRegistry):
      - "final_answer": passes through directly.
      - "redirect_domain": passes through directly, same shape as
        final_answer — not actually reached via loop.py's main iteration
        (which intercepts and returns before dispatch() is called, exactly
        like final_answer), kept here for consistency/defensiveness so a
        future direct caller of dispatch() doesn't hit an unhandled tool name.

    override_args: when given, used INSTEAD of tool_call["args"] for the
    actual registry.call() — lets loop.py merge in per-call, non-JSON-safe
    context (e.g. a live ConversationSession object for a declared tool's
    context_provider) without mutating tool_call["args"] itself, since that
    dict is stored verbatim on AgentStep and assumed JSON-safe/loggable
    elsewhere. Defaults to tool_call["args"], preserving existing behavior
    for every caller that doesn't pass it.
    """
    tool_name = tool_call.get("tool", "")
    args = tool_call.get("args", {})
    dispatch_args = override_args if override_args is not None else args

    # final_answer is handled by the orchestrator, not the registry
    if tool_name == "final_answer":
        return {
            "success": True,
            "result": args.get("answer", ""),
            "error": None,
        }

    if tool_name == "redirect_domain":
        return {
            "success": True,
            "result": args.get("answer", ""),
            "error": None,
        }

    try:
        result = registry.call(tool_name, dispatch_args)
    except Exception as exc:
        result = {
            "success": False,
            "result": None,
            "error": f"Dispatcher caught unhandled exception in '{tool_name}': {exc}",
        }

    return result