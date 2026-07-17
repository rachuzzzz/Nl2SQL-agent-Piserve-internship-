"""
Deterministic tests for agent/prompts/builder.py's dynamic prompt-assembly
functions: build_worked_examples, build_conditional_blocks,
build_dynamic_prompt_additions, and the shared _select_within_budget
budget-capping algorithm.

No LLM, no live DB — these are pure functions (string in, string out) tested
against the real EXAMPLES_TIER2/CONDITIONAL_BLOCKS data, not fixtures, so a
regression in an actual registered question_pattern is caught here. Run
directly (`python tests/agent/test_prompt_builder.py`) or via pytest if it's
ever added to this project (test_*() functions are pytest-discoverable
as-is).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.prompts import builder as _builder_mod
from src.agent.prompts.builder import (
    build_worked_examples,
    build_conditional_blocks,
    build_dynamic_prompt_additions,
    _select_within_budget,
)

FAILURES = []

# A question that matches none of the real EXAMPLES_TIER2/CONDITIONAL_BLOCKS
# patterns (verified empirically against the live registered patterns).
_NO_MATCH_QUESTION = "asdkfjaslkdfj random gibberish"

_WORKED_HEADER = "ADDITIONAL WORKED EXAMPLES"
_CONDITIONAL_HEADER = "ADDITIONAL CONTEXT"


def _check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# build_worked_examples
# ---------------------------------------------------------------------------

def test_worked_examples_no_match():
    _check("no-match question -> empty string", build_worked_examples(_NO_MATCH_QUESTION) == "")


def test_worked_examples_empty_question():
    _check("empty question -> empty string", build_worked_examples("") == "")


def test_worked_examples_ma_approval_match():
    result = build_worked_examples("show mitigative actions awaiting approval")
    _check(
        "MA-approval question includes worked-examples header",
        _WORKED_HEADER in result and len(result) > len(_WORKED_HEADER),
        detail=result[:80],
    )


def test_worked_examples_malformed_regex_skipped():
    bogus = {"title": "bogus test entry", "question_pattern": "(unclosed", "text": "SHOULD_NEVER_APPEAR"}
    _builder_mod.EXAMPLES_TIER2.append(bogus)
    try:
        result = build_worked_examples("mitigative actions awaiting approval bogus regex test")
        _check(
            "malformed question_pattern is skipped, not raised",
            "SHOULD_NEVER_APPEAR" not in result,
        )
    finally:
        _builder_mod.EXAMPLES_TIER2.remove(bogus)


# ---------------------------------------------------------------------------
# build_conditional_blocks
# ---------------------------------------------------------------------------

def test_conditional_blocks_no_match():
    _check("no-match question -> empty string", build_conditional_blocks(_NO_MATCH_QUESTION) == "")


def test_conditional_blocks_empty_question():
    _check("empty question -> empty string", build_conditional_blocks("") == "")


def test_conditional_blocks_ma_workflow_match():
    result = build_conditional_blocks("how many mitigative actions are open")
    _check(
        "MA-workflow question includes conditional-context header",
        _CONDITIONAL_HEADER in result and len(result) > len(_CONDITIONAL_HEADER),
        detail=result[:80],
    )


def test_conditional_blocks_malformed_regex_skipped():
    bogus = {"title": "bogus test entry", "question_pattern": "(unclosed", "get_text": lambda: "SHOULD_NEVER_APPEAR"}
    _builder_mod.CONDITIONAL_BLOCKS.append(bogus)
    try:
        result = build_conditional_blocks("mitigative actions bogus regex test")
        _check(
            "malformed question_pattern is skipped, not raised",
            "SHOULD_NEVER_APPEAR" not in result,
        )
    finally:
        _builder_mod.CONDITIONAL_BLOCKS.remove(bogus)


# ---------------------------------------------------------------------------
# build_dynamic_prompt_additions — the combined entry point
# ---------------------------------------------------------------------------

def test_dynamic_additions_no_match():
    _check("no-match question -> empty string", build_dynamic_prompt_additions(_NO_MATCH_QUESTION) == "")


def test_dynamic_additions_combines_both():
    result = build_dynamic_prompt_additions("show mitigative actions awaiting approval")
    _check(
        "MA-approval question includes BOTH conditional and worked-example headers",
        _CONDITIONAL_HEADER in result and _WORKED_HEADER in result,
        detail=result[:120],
    )


# ---------------------------------------------------------------------------
# _select_within_budget — shared budget-capping algorithm
# ---------------------------------------------------------------------------

def test_budget_no_score_fn_returns_unchanged():
    candidates = [("A", "x" * 100), ("B", "y" * 100)]
    kept = _select_within_budget(candidates, "q", None, budget_chars=10)
    _check(
        "no score_fn -> every candidate returned unchanged regardless of budget",
        kept == ["x" * 100, "y" * 100],
    )


def test_budget_single_candidate_returns_unchanged():
    candidates = [("A", "x" * 500)]
    kept = _select_within_budget(candidates, "q", lambda q, t: 1.0, budget_chars=10)
    _check(
        "single candidate -> returned unchanged even if score_fn supplied and over budget",
        kept == ["x" * 500],
    )


def test_budget_caps_lower_scored_candidate():
    candidates = [("A", "a" * 50), ("B", "b" * 50)]
    # B scores higher than A.
    score_fn = lambda q, title: 2.0 if title == "B" else 1.0
    kept = _select_within_budget(candidates, "q", score_fn, budget_chars=60)
    _check(
        "best-scored candidate (B) kept, second (A) dropped once budget is exceeded",
        kept == ["b" * 50],
        detail=str(kept),
    )


def test_budget_always_keeps_best_even_if_oversized():
    candidates = [("A", "a" * 200), ("B", "b" * 10)]
    # A scores higher than B, but A alone already exceeds the budget.
    score_fn = lambda q, title: 2.0 if title == "A" else 1.0
    kept = _select_within_budget(candidates, "q", score_fn, budget_chars=50)
    _check(
        "single best-scored candidate is always kept even if it alone exceeds budget_chars",
        kept == ["a" * 200],
        detail=str(kept),
    )


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
