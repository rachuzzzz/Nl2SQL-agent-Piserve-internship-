"""
NL2SQL Agent — Evaluation Test Set
====================================
Run this against each model pair and compare:
  - Did it answer correctly? (manual check)
  - How many LLM calls did it take?
  - How long did it take?
  - Did SQL execute without errors?
  - Was the result sensible?

Usage:
  python3 eval.py --model-pair "llama3.1:8b + deepseek:6.7b"

Scoring per question (fill in manually after running):
  2 = correct answer, 1 LLM call
  1 = correct answer, multiple LLM calls / retries
  0 = wrong answer or crash
"""

TEST_SET = [

    # ── CATEGORY 1: Simple aggregates (should be 1 LLM call, <8s) ──────────
    {
        "id": "AGG-01",
        "category": "Aggregate",
        "question": "what is the average inspection score",
        "expected_contains": ["86", "avg", "score"],   # approximate expected value
        "expected_llm_calls": 1,
        "notes": "Simplest possible query — baseline test",
    },
    {
        "id": "AGG-02",
        "category": "Aggregate",
        "question": "how many inspections were completed this year",
        "expected_contains": ["inspection"],
        "expected_llm_calls": 1,
        "notes": "Date filter + count",
    },
    {
        "id": "AGG-03",
        "category": "Aggregate",
        "question": "what is the total capex and opex across all corrective actions",
        "expected_contains": ["capex", "opex"],
        "expected_llm_calls": 1,
        "notes": "Two aggregates from corrective action table",
    },
    {
        "id": "AGG-04",
        "category": "Aggregate",
        "question": "how many open corrective actions are there",
        "expected_contains": ["open", "action"],
        "expected_llm_calls": 1,
        "notes": "Simple count with status filter",
    },

    # ── CATEGORY 2: Filtering + sorting ────────────────────────────────────
    {
        "id": "FLT-01",
        "category": "Filter",
        "question": "show the last 5 inspections by submission date",
        "expected_contains": ["inspection"],
        "expected_llm_calls": 1,
        "notes": "ORDER BY + LIMIT — must return exactly 5",
    },
    {
        "id": "FLT-02",
        "category": "Filter",
        "question": "show all overdue corrective actions",
        "expected_contains": ["overdue", "action"],
        "expected_llm_calls": 1,
        "notes": "Date comparison — target_close_out_date < CURRENT_DATE",
    },
    {
        "id": "FLT-03",
        "category": "Filter",
        "question": "show high risk observations",
        "expected_contains": ["high", "observation"],
        "expected_llm_calls": 1,
        "notes": "Requires answer_text ILIKE filter — tests encoded value handling",
    },
    {
        "id": "FLT-04",
        "category": "Filter",
        "question": "which inspections were returned for modification",
        "expected_contains": ["RETURN_FOR_MODIFICATION", "inspection"],
        "expected_llm_calls": 1,
        "notes": "Enum status filter",
    },

    # ── CATEGORY 3: JOINs across tables ─────────────────────────────────────
    {
        "id": "JOI-01",
        "category": "Join",
        "question": "show inspection scores grouped by facility",
        "expected_contains": ["facility", "score"],
        "expected_llm_calls": 1,
        "notes": "JOIN inspection_report + facility, GROUP BY",
    },
    {
        "id": "JOI-02",
        "category": "Join",
        "question": "which inspector has conducted the most inspections",
        "expected_contains": ["inspector", "count"],
        "expected_llm_calls": 1,
        "notes": "JOIN users, GROUP BY, ORDER BY count DESC",
    },
    {
        "id": "JOI-03",
        "category": "Join",
        "question": "show average inspection score by inspection type",
        "expected_contains": ["type", "score"],
        "expected_llm_calls": 1,
        "notes": "JOIN inspection_type, GROUP BY type name",
    },
    {
        "id": "JOI-04",
        "category": "Join",
        "question": "show corrective actions with their inspection facility and inspector",
        "expected_contains": ["facility", "inspector", "action"],
        "expected_llm_calls": 1,
        "notes": "Multi-table join — corrective_action + report + facility + users",
    },

    # ── CATEGORY 4: Form answers (ai_answers + ai_questions) ──────────────
    {
        "id": "ANS-01",
        "category": "Form Answers",
        "question": "what are the most common observation types across all inspections",
        "expected_contains": ["deviation", "observation"],
        "expected_llm_calls": 1,
        "notes": "get_answer_stats or GROUP BY on answer_text — tests label matching",
    },
    {
        "id": "ANS-02",
        "category": "Form Answers",
        "question": "list all questions and answers from the most recent inspection form",
        "expected_contains": ["question", "answer"],
        "expected_llm_calls": 1,
        "notes": "Must filter module_name = Inspection Form — key regression test",
    },
    {
        "id": "ANS-03",
        "category": "Form Answers",
        "question": "show all risk levels from the last 5 inspections",
        "expected_contains": ["risk", "high"],
        "expected_llm_calls": 1,
        "notes": "ai_answers filtered by question label + recent inspection",
    },
    {
        "id": "ANS-04",
        "category": "Form Answers",
        "question": "what did inspectors write in the observation field for Al Ghadeer",
        "expected_contains": ["observation", "al ghadeer"],
        "expected_llm_calls": 1,
        "notes": "Cross-domain: ai_answers + inspection_report + facility JOIN",
    },

    # ── CATEGORY 5: Scores from ai_answers ───────────────────────────────
    {
        "id": "SCR-01",
        "category": "Scores",
        "question": "show me questions with their scores for the last inspection by George",
        "expected_contains": ["score", "george"],
        "expected_llm_calls": 1,
        "notes": "Joins ai_answers.score + users — key test for backfill fix",
    },
    {
        "id": "SCR-02",
        "category": "Scores",
        "question": "what is the average score for the risk level question",
        "expected_contains": ["score", "risk"],
        "expected_llm_calls": 1,
        "notes": "get_answer_stats — numeric aggregation on ai_answers.score",
    },
    {
        "id": "SCR-03",
        "category": "Scores",
        "question": "which inspection had the highest total score",
        "expected_contains": ["score", "inspection"],
        "expected_llm_calls": 1,
        "notes": "SUM/MAX of ai_answers.score grouped by inspection",
    },

    # ── CATEGORY 6: Multi-turn (run these sequentially) ───────────────────
    {
        "id": "MT-01",
        "category": "Multi-turn",
        "question": "which facility underwent the most recent inspection",
        "expected_contains": ["facility"],
        "expected_llm_calls": 1,
        "notes": "Turn 1 of multi-turn sequence — stores facility + inspection_id",
    },
    {
        "id": "MT-02",
        "category": "Multi-turn",
        "question": "who inspected it",
        "expected_contains": ["inspector", "name"],
        "expected_llm_calls": 1,
        "notes": "Turn 2 — must use facility context from turn 1, not re-search",
        "depends_on": "MT-01",
    },
    {
        "id": "MT-03",
        "category": "Multi-turn",
        "question": "fetch all questions and answers she filled for that site",
        "expected_contains": ["question", "answer"],
        "expected_llm_calls": 1,
        "notes": "Turn 3 — MUST use inspection_id from context, not facility name (was 12584 bug)",
        "depends_on": "MT-02",
    },
    {
        "id": "MT-04",
        "category": "Multi-turn",
        "question": "show more",
        "expected_contains": [],
        "expected_llm_calls": 0,  # should be pagination fast-path, no LLM call
        "notes": "Pagination — must NOT re-generate SQL, must re-run last query",
        "depends_on": "MT-03",
    },

    # ── CATEGORY 7: Edge cases ────────────────────────────────────────────
    {
        "id": "EDG-01",
        "category": "Edge Case",
        "question": "show all high risk observations this month",
        "expected_contains": ["high", "observation"],
        "expected_llm_calls": 1,
        "notes": "Combines: answer filter + date filter — two constraints at once",
    },
    {
        "id": "EDG-02",
        "category": "Edge Case",
        "question": "how many questions are in the inspection form",
        "expected_contains": ["question", "count"],
        "expected_llm_calls": 1,
        "notes": "Count from ai_questions, filter by module_name",
    },
    {
        "id": "EDG-03",
        "category": "Edge Case",
        "question": "show corrective actions where responsible is client",
        "expected_contains": ["client", "action"],
        "expected_llm_calls": 1,
        "notes": "Enum value filter — responsible = 'CLIENT'",
    },
    {
        "id": "EDG-04",
        "category": "Edge Case",
        "question": "xyz abc nonsense query that makes no sense",
        "expected_contains": ["no", "found", "not"],
        "expected_llm_calls": 1,
        "notes": "Garbage input — should fail gracefully, not crash",
    },
]


# ── Runner ─────────────────────────────────────────────────────────────────

import sys
import time
import json
import argparse

def run_eval(model_pair_label: str, skip_multiturn: bool = False):
    """
    Imports and runs the agent against each test question.
    Records timing, LLM calls, and whether expected keywords appear.
    Saves results to eval_results_{timestamp}.json
    """
    from agent.orchestrator import AgentOrchestrator, ConversationSession
    from dotenv import load_dotenv
    load_dotenv()

    print(f"\n{'='*60}")
    print(f"  EVAL: {model_pair_label}")
    print(f"  Questions: {len(TEST_SET)}")
    print(f"{'='*60}\n")

    orch = AgentOrchestrator.from_env()
    results = []
    session = ConversationSession()  # shared session for multi-turn

    for i, test in enumerate(TEST_SET):
        if skip_multiturn and test["category"] == "Multi-turn":
            print(f"  [{test['id']}] SKIPPED (multi-turn)")
            continue

        print(f"\n[{i+1}/{len(TEST_SET)}] {test['id']} — {test['question'][:60]}")

        t0 = time.time()
        try:
            result = orch.query(
                test["question"],
                session=session if test["category"] == "Multi-turn" else None,
            )
            wall = time.time() - t0

            # Check expected keywords
            answer_lower = result.answer.lower()
            keyword_hits = sum(
                1 for kw in test["expected_contains"]
                if kw.lower() in answer_lower
            )
            keyword_total = len(test["expected_contains"])
            keyword_pct = (keyword_hits / keyword_total * 100) if keyword_total > 0 else 100

            # Check SQL errors
            sql_errors = sum(
                1 for step in result.steps
                if step.tool == "execute_sql" and not step.result.get("success")
            )

            row = {
                "id": test["id"],
                "category": test["category"],
                "question": test["question"],
                "model_pair": model_pair_label,
                "wall_s": round(wall, 2),
                "llm_calls": result.stats.total_llm_calls,
                "prompt_tokens": result.stats.total_prompt_tokens,
                "completion_tokens": result.stats.total_completion_tokens,
                "sql_errors": sql_errors,
                "keyword_hits": f"{keyword_hits}/{keyword_total}",
                "keyword_pct": round(keyword_pct),
                "answer_preview": result.answer[:200],
                "success": result.success,
                "manual_score": None,  # fill in after reviewing
                "notes": test["notes"],
            }
            results.append(row)

            # Print summary line
            status = "✓" if sql_errors == 0 else "✗"
            print(f"  {status} {wall:.1f}s | {result.stats.total_llm_calls} LLM calls | "
                  f"keywords {keyword_hits}/{keyword_total} | "
                  f"{'OK' if sql_errors == 0 else f'{sql_errors} SQL errors'}")
            print(f"    → {result.answer[:120]}")

        except Exception as e:
            wall = time.time() - t0
            print(f"  ✗ CRASHED in {wall:.1f}s: {e}")
            results.append({
                "id": test["id"],
                "category": test["category"],
                "question": test["question"],
                "model_pair": model_pair_label,
                "wall_s": round(wall, 2),
                "llm_calls": -1,
                "sql_errors": -1,
                "keyword_hits": "0/0",
                "keyword_pct": 0,
                "answer_preview": f"CRASH: {e}",
                "success": False,
                "manual_score": 0,
                "notes": test["notes"],
            })

    # Save results
    ts = int(time.time())
    fname = f"eval_results_{ts}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {model_pair_label}")
    print(f"{'='*60}")
    print(f"  {'ID':<10} {'Cat':<14} {'Time':>6} {'LLM':>5} {'KW%':>5} {'SQL Err':>8}")
    print(f"  {'-'*55}")
    for r in results:
        flag = "✓" if r["sql_errors"] == 0 else "✗"
        print(f"  {flag} {r['id']:<9} {r['category']:<14} "
              f"{r['wall_s']:>5.1f}s {r['llm_calls']:>4} "
              f"{r['keyword_pct']:>4}% {r['sql_errors']:>7}")

    total = len(results)
    crashed = sum(1 for r in results if r["llm_calls"] == -1)
    sql_ok = sum(1 for r in results if r["sql_errors"] == 0)
    avg_time = sum(r["wall_s"] for r in results) / total if total else 0
    avg_calls = sum(r["llm_calls"] for r in results if r["llm_calls"] >= 0)
    single_call = sum(1 for r in results if r["llm_calls"] == 1)

    print(f"\n  Total: {total} | Crashed: {crashed} | SQL OK: {sql_ok}/{total}")
    print(f"  Avg time: {avg_time:.1f}s | Single-call: {single_call}/{total}")
    print(f"  Results saved → {fname}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pair", default="unknown", help="Label for this model pair")
    parser.add_argument("--skip-multiturn", action="store_true", help="Skip multi-turn tests")
    args = parser.parse_args()
    run_eval(args.model_pair, skip_multiturn=args.skip_multiturn)