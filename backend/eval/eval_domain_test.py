"""
Domain Router Evaluation — Tier 1/2/3 test set
================================================
Lightweight, fast test harness for agent/domain_router.py's DomainRouter.
Does NOT spin up the full AgentOrchestrator (DB connection, embedding model,
seed index, SQL LLM) the way eval.py does — Tier 1/2 are pure regex/session-
state checks and need no LLM at all; only the (rare) Tier-3 cold-start cases
need the reasoning LLM client, built directly and cheaply via
src.llm.factory.get_reasoning_llm.

Usage:
  python3 eval_domain_test.py
  python3 eval_domain_test.py --skip-tier3        # offline, no Ollama needed
  python3 eval_domain_test.py --category tier1
  python3 eval_domain_test.py --output my_results.json

Two kinds of test cases:
  SINGLE_CASES  — one question, optionally with a synthetic prior turn to
                  simulate session stickiness, checked independently.
  SEQUENCES     — an ordered list of turns run against ONE shared session,
                  each turn's resolved domain feeding stickiness for the
                  next — this is what actually exercises Tier 2 and
                  cross-domain switching the way real usage does.
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Ensure `backend/` (parent of this file's `eval/` dir) is importable as the
# `src` package root regardless of the invocation directory/style
# (`python eval_domain_test.py`, `python eval/eval_domain_test.py`, etc.) —
# more robust than relying on an implicit CWD convention.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── SINGLE_CASES ──────────────────────────────────────────────────────────────

SINGLE_CASES = [
    # ── Tier 1 — clear-cut keyword, no session needed ──────────────────────────
    {"id": "T1-01", "category": "tier1_inspection", "question": "list all inspections",
     "prior_domain": None, "expected_domain": "INSPECTION", "expected_tier": 1},
    {"id": "T1-02", "category": "tier1_inspection", "question": "which inspector has the most overdue mitigative actions",
     "prior_domain": None, "expected_domain": "INSPECTION", "expected_tier": 1},
    {"id": "T1-03", "category": "tier1_inspection", "question": "show inspection schedules for next week",
     "prior_domain": None, "expected_domain": "INSPECTION", "expected_tier": 1},
    {"id": "T1-04", "category": "tier1_audit", "question": "which auditor has the most overdue corrective actions",
     "prior_domain": None, "expected_domain": "AUDIT", "expected_tier": 1},
    {"id": "T1-05", "category": "tier1_audit", "question": "show audit plans pending approval",
     "prior_domain": None, "expected_domain": "AUDIT", "expected_tier": 1},
    {"id": "T1-06", "category": "tier1_audit", "question": "list all audit schedules",
     "prior_domain": None, "expected_domain": "AUDIT", "expected_tier": 1},
    {"id": "T1-07", "category": "tier1_audit", "question": "how many corrective actions are open",
     "prior_domain": None, "expected_domain": "AUDIT", "expected_tier": 1,
     "notes": "'corrective action' alone (no 'audit' word) must still hit Tier 1"},

    # ── Tier 1 — verb forms (live-caught gap, fixed) ────────────────────────────
    # Sticky domain is deliberately set to the OPPOSITE of the expected answer —
    # these must win on the verb-form keyword alone, not fall through to Tier 2.
    {"id": "T1-VERB-01", "category": "tier1_verb", "question": "who audited this facility last month",
     "prior_domain": "inspection", "expected_domain": "AUDIT", "expected_tier": 1,
     "notes": "bare verb form 'audited' — was a live-caught silent misroute before the regex fix"},
    {"id": "T1-VERB-02", "category": "tier1_verb", "question": "who inspected this ADJD site",
     "prior_domain": "audit", "expected_domain": "INSPECTION", "expected_tier": 1,
     "notes": "bare verb form 'inspected' — the exact failing case from live testing"},
    {"id": "T1-VERB-03", "category": "tier1_verb", "question": "please inspect this facility",
     "prior_domain": "audit", "expected_domain": "INSPECTION", "expected_tier": 1},
    {"id": "T1-VERB-04", "category": "tier1_verb", "question": "auditing the schedule now",
     "prior_domain": "inspection", "expected_domain": "AUDIT", "expected_tier": 1},

    # ── Tier 1 — false-positive guard (should NOT match either domain) ─────────
    {"id": "T1-FP-01", "category": "tier1_false_positive", "question": "show me the auditorium bookings",
     "prior_domain": "inspection", "expected_domain": "INSPECTION", "expected_tier": 2,
     "notes": "'auditorium' must not false-positive as 'audit' — falls through to stickiness"},
    {"id": "T1-FP-02", "category": "tier1_false_positive", "question": "check inspectable items",
     "prior_domain": "audit", "expected_domain": "AUDIT", "expected_tier": 2,
     "notes": "'inspectable' must not false-positive as 'inspect' — falls through to stickiness"},

    # ── Both keywords fire at once — falls through to stickiness/LLM ───────────
    {"id": "BOTH-01", "category": "both_keywords",
     "question": "compare audit corrective actions vs inspection mitigative actions",
     "prior_domain": "inspection", "expected_domain": "INSPECTION", "expected_tier": 2},
    {"id": "BOTH-02", "category": "both_keywords",
     "question": "show me inspections and audits for ADJD facility",
     "prior_domain": "audit", "expected_domain": "AUDIT", "expected_tier": 2},

    # ── Tier 3 — cold start, ambiguous, no prior session (needs real LLM) ──────
    {"id": "T3-01", "category": "tier3", "question": "list all schedules",
     "prior_domain": None, "expected_domain": "AMBIGUOUS", "expected_tier": 3, "needs_llm": True},
    {"id": "T3-02", "category": "tier3", "question": "show me all plans",
     "prior_domain": None, "expected_domain": "AMBIGUOUS", "expected_tier": 3, "needs_llm": True},
    {"id": "T3-03", "category": "tier3", "question": "what's pending review",
     "prior_domain": None, "expected_domain": "AMBIGUOUS", "expected_tier": 3, "needs_llm": True},

    # ── Ambiguous-synonym backstop (Tier 1, deterministic, no LLM) ─────────────
    # Confirmed live 2026-08-03: the in-prompt DOMAIN-MISMATCH CHECK (a
    # semantic judgment call by the reasoning LLM) is NOT reliably
    # consistent — the same model caught "remedial actions" but silently
    # missed "curative actions" in the same session, once a nearby specific
    # cue ("at risk") pulled its attention away from the ambiguous word.
    # This deterministic router-level backstop can't be inconsistently
    # skipped the way the in-prompt check can.
    {"id": "SYN-01", "category": "ambiguous_synonym", "question": "show all remedial actions",
     "prior_domain": "audit", "expected_domain": "AMBIGUOUS", "expected_tier": 1},
    {"id": "SYN-02", "category": "ambiguous_synonym",
     "question": "show all curative actions that are at risk",
     "prior_domain": "audit", "expected_domain": "AMBIGUOUS", "expected_tier": 1},
    {"id": "SYN-03", "category": "ambiguous_synonym", "question": "list all preventive actions",
     "prior_domain": "inspection", "expected_domain": "AMBIGUOUS", "expected_tier": 1},
    {"id": "SYN-04", "category": "ambiguous_synonym",
     "question": "any rectification actions outstanding",
     "prior_domain": None, "expected_domain": "AMBIGUOUS", "expected_tier": 1},
    # An explicit domain keyword elsewhere in the same message still wins
    # outright — the backstop only fires when Tier 1's own keyword check
    # found nothing to resolve on.
    {"id": "SYN-05", "category": "ambiguous_synonym", "question": "audit's remedial actions",
     "prior_domain": None, "expected_domain": "AUDIT", "expected_tier": 1},
]


# ── SEQUENCES ─────────────────────────────────────────────────────────────────
# Each turn's ACTUAL resolved domain is fed forward as the next turn's sticky
# state (mirroring how api/routes/chat.py tags each ConversationTurn after a
# real query) — this is what actually exercises Tier 2, not a hardcoded guess.

SEQUENCES = [
    {
        "id": "SEQ-stickiness-pairs",
        "turns": [
            {"question": "show audit schedules", "expected_domain": "AUDIT", "expected_tier": 1},
            {"question": "show the overdue ones", "expected_domain": "AUDIT", "expected_tier": 2},
            {"question": "list all inspections", "expected_domain": "INSPECTION", "expected_tier": 1},
            {"question": "what about last month", "expected_domain": "INSPECTION", "expected_tier": 2},
            {"question": "show audit corrective actions", "expected_domain": "AUDIT", "expected_tier": 1},
            {"question": "which ones are closed", "expected_domain": "AUDIT", "expected_tier": 2},
        ],
    },
    {
        "id": "SEQ-scenario-A-hard-switch-and-back",
        "turns": [
            {"question": "show audit corrective actions", "expected_domain": "AUDIT", "expected_tier": 1},
            {"question": "is inspection there for this facility?", "expected_domain": "INSPECTION", "expected_tier": 1},
            {"question": "ok back to those corrective actions", "expected_domain": "AUDIT", "expected_tier": 1,
             "notes": "audit thread must still be reachable — nothing gets wiped by the switch"},
        ],
    },
    {
        "id": "SEQ-verb-form-mid-conversation",
        "turns": [
            {"question": "how many corrective actions are open", "expected_domain": "AUDIT", "expected_tier": 1},
            {"question": "who audited ADJD facility last month", "expected_domain": "AUDIT", "expected_tier": 1},
            {"question": "who inspected ADJD site", "expected_domain": "INSPECTION", "expected_tier": 1,
             "notes": "the exact live-caught bug — verb form must win over audit stickiness"},
        ],
    },
]


# ── Runner ───────────────────────────────────────────────────────────────────

def _make_router(needs_llm: bool):
    from src.agent.domain_router import DomainRouter
    if not needs_llm:
        return DomainRouter(reasoning_llm=None)
    from dotenv import load_dotenv
    load_dotenv()
    from src.config.settings import get_settings
    from src.llm.factory import get_reasoning_llm
    s = get_settings()
    llm = get_reasoning_llm(s.ollama_chat_model, s.ollama_url)
    return DomainRouter(reasoning_llm=llm)


def _fresh_session(prior_domain: str | None = None):
    from src.agent.orchestrator.session import ConversationSession
    from src.agent.orchestrator.models import ConversationTurn
    sess = ConversationSession()
    if prior_domain:
        sess.add_turn(ConversationTurn(
            question="(synthetic prior turn)", answer="", tool_used="test",
            domain=prior_domain,
        ))
    return sess


def _check(route, case) -> tuple[bool, str]:
    domain_ok = route.domain == case["expected_domain"]
    tier_ok = route.tier == case["expected_tier"]
    if domain_ok and tier_ok:
        return True, ""
    parts = []
    if not domain_ok:
        parts.append(f"domain={route.domain} (expected {case['expected_domain']})")
    if not tier_ok:
        parts.append(f"tier={route.tier} (expected {case['expected_tier']})")
    return False, ", ".join(parts)


def run_eval(skip_tier3=False, category_filter=None, output_file=None):
    from src.agent.orchestrator.models import ConversationTurn

    singles = SINGLE_CASES
    if skip_tier3:
        singles = [c for c in singles if not c.get("needs_llm")]
    if category_filter:
        singles = [c for c in singles if c["category"] == category_filter]
        sequences = [] if category_filter != "sequence" else SEQUENCES
    else:
        sequences = SEQUENCES

    needs_llm = any(c.get("needs_llm") for c in singles)
    router_no_llm = _make_router(needs_llm=False)
    router_with_llm = _make_router(needs_llm=True) if needs_llm else None

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Domain Router Evaluation")
    print(f"  {len(singles)} single case(s), {len(sequences)} sequence(s)  |  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{sep}\n")

    results = []

    # ── single cases ─────────────────────────────────────────────────────────
    for case in singles:
        router = router_with_llm if case.get("needs_llm") else router_no_llm
        sess = _fresh_session(case.get("prior_domain"))
        route = router.classify(case["question"], session=sess)
        passed, detail = _check(route, case)
        results.append({
            "id": case["id"], "category": case["category"], "question": case["question"],
            "expected_domain": case["expected_domain"], "expected_tier": case["expected_tier"],
            "actual_domain": route.domain, "actual_tier": route.tier,
            "passed": passed, "detail": detail, "notes": case.get("notes", ""),
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {case['id']:<12} tier={route.tier} domain={route.domain:<11} "
              f"| {case['question'][:55]}")
        if not passed:
            print(f"         -> {detail}")

    # ── sequences ────────────────────────────────────────────────────────────
    for seq in sequences:
        print(f"\n  --- sequence: {seq['id']} ---")
        sess = _fresh_session()
        for i, turn in enumerate(seq["turns"]):
            route = router_no_llm.classify(turn["question"], session=sess)
            passed, detail = _check(route, turn)
            case_id = f"{seq['id']}#{i+1}"
            results.append({
                "id": case_id, "category": "sequence", "question": turn["question"],
                "expected_domain": turn["expected_domain"], "expected_tier": turn["expected_tier"],
                "actual_domain": route.domain, "actual_tier": route.tier,
                "passed": passed, "detail": detail, "notes": turn.get("notes", ""),
            })
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] turn {i+1}  tier={route.tier} domain={route.domain:<11} "
                  f"| {turn['question'][:55]}")
            if not passed:
                print(f"         -> {detail}")
            # Feed the ACTUAL resolved domain forward as sticky state for the
            # next turn — mirrors api/routes/chat.py tagging each ConversationTurn.
            resolved = route.domain.lower() if route.domain != "AMBIGUOUS" else ""
            sess.add_turn(ConversationTurn(
                question=turn["question"], answer="", tool_used="test", domain=resolved,
            ))

    # ── summary ──────────────────────────────────────────────────────────────
    total = len(results)
    passed_n = sum(1 for r in results if r["passed"])
    print(f"\n{sep}")
    print(f"  SUMMARY")
    print(f"{sep}")
    from collections import defaultdict
    by_cat = defaultdict(lambda: {"count": 0, "passed": 0})
    for r in results:
        by_cat[r["category"]]["count"] += 1
        by_cat[r["category"]]["passed"] += int(r["passed"])
    for cat, s in sorted(by_cat.items()):
        print(f"  {cat:<22} {s['passed']:>3}/{s['count']:<3}")
    print(f"\n  TOTAL: {passed_n}/{total} ({100 * passed_n // total if total else 0}%)")

    fails = [r for r in results if not r["passed"]]
    if fails:
        print(f"\n  FAILURES:")
        for r in fails:
            print(f"    {r['id']:<28} {r['detail']}")
            print(f"      Q: {r['question']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = output_file or f"eval_domain_results_{ts}.json"
    with open(fname, "w") as f:
        json.dump({"total": total, "passed": passed_n, "results": results}, f, indent=2)
    print(f"\n  Results saved -> {fname}\n")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Domain Router Evaluation")
    p.add_argument("--skip-tier3", action="store_true",
                    help="Skip cold-start ambiguous cases that need the reasoning LLM (fully offline)")
    p.add_argument("--category", default=None,
                    help="tier1_inspection | tier1_audit | tier1_verb | tier1_false_positive | "
                         "both_keywords | tier3 | sequence")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    results = run_eval(a.skip_tier3, a.category, a.output)
    sys.exit(0 if all(r["passed"] for r in results) else 1)
