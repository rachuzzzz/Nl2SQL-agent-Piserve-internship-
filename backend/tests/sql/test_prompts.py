"""
Deterministic tests for sql/prompts.py's build_core_context_additions() —
the SQL-generation ("Deepseek-side") counterpart to
agent/prompts/builder.py's build_conditional_blocks() (see
tests/agent/test_prompt_builder.py for the reasoning-side equivalent).

No LLM, no live DB — pure function tested against the real
CORE_CONTEXT_CONDITIONAL data, not fixtures. Run directly
(`python tests/sql/test_prompts.py`) or via pytest if it's ever added to
this project (test_*() functions are pytest-discoverable as-is).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.sql import prompts as _prompts_mod
from src.sql.prompts import build_core_context_additions

FAILURES = []

# A question that matches none of the real CORE_CONTEXT_CONDITIONAL patterns
# (verified empirically against the live registered patterns).
_NO_MATCH_QUESTION = "asdkfjaslkdfj random gibberish"


def _check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def test_no_match_returns_empty():
    _check("no-match question -> empty string", build_core_context_additions(_NO_MATCH_QUESTION) == "")


def test_empty_question_returns_empty():
    _check("empty question -> empty string", build_core_context_additions("") == "")


def test_ma_workflow_only_match():
    # Matches MA_WORKFLOW_TOPIC but none of the counting pattern's keywords.
    result = build_core_context_additions("how many mitigative actions are open")
    expected = "\n--\n" + _prompts_mod._MA_WORKFLOW_BLOCK
    _check(
        "MA-only question returns exactly the MA workflow block",
        result == expected,
        detail=f"got {result[:60]!r}",
    )


def test_counting_only_match():
    # Matches the counting pattern ("observations") but not MA_WORKFLOW_TOPIC.
    result = build_core_context_additions("how many observations were recorded")
    expected = "\n--\n" + _prompts_mod._COUNTING_BLOCK
    _check(
        "counting-only question returns exactly the counting block",
        result == expected,
        detail=f"got {result[:60]!r}",
    )


def test_both_blocks_match_in_declared_order():
    # Matches both MA_WORKFLOW_TOPIC ("mitigative actions") and the counting
    # pattern ("observations") — CORE_CONTEXT_CONDITIONAL declares MA workflow
    # first, so it must appear first in the joined result.
    result = build_core_context_additions(
        "how many mitigative actions were opened based on observations"
    )
    expected = "\n--\n" + "\n--\n".join(
        [_prompts_mod._MA_WORKFLOW_BLOCK, _prompts_mod._COUNTING_BLOCK]
    )
    _check(
        "question matching both patterns returns both blocks in declared order",
        result == expected,
        detail=f"got {result[:60]!r}",
    )


def test_malformed_regex_skipped():
    bogus = {
        "title": "bogus test entry",
        "question_pattern": "(unclosed",
        "get_text": lambda: "SHOULD_NEVER_APPEAR",
    }
    _prompts_mod.CORE_CONTEXT_CONDITIONAL.append(bogus)
    try:
        result = build_core_context_additions("mitigative actions bogus regex test")
        _check(
            "malformed question_pattern is skipped, not raised",
            "SHOULD_NEVER_APPEAR" not in result,
        )
    finally:
        _prompts_mod.CORE_CONTEXT_CONDITIONAL.remove(bogus)


if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in test_fns:
        try:
            fn()
        except Exception as e:
            print(f"[ERROR] {fn.__name__} raised {e!r}")
            FAILURES.append(fn.__name__)

    print()
    if FAILURES:
        print(f"{len(FAILURES)}/{len(test_fns)} FAILED: {FAILURES}")
        sys.exit(1)
    else:
        print(f"ALL {len(test_fns)} CHECKS PASSED")
