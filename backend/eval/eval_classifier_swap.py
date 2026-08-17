"""
Classifier-swap accuracy eval — the hard gate for the "give turn-classifier
and domain-router their own lightweight Ollama model" change (see the
project plan). Before qwen3.5:0.8b is ever wired into a real call site, this
replays REAL logged inputs (from debug_llm_*.log) that qwen3.5:9b already
classified, against qwen3.5:0.8b, and diffs the two models' decisions on the
fields that actually gate correctness — action/reference for the turn
classifier, domain for the router. A latency win here is worthless if it
silently misroutes questions or breaks follow-up handling.

Reuses the real system prompts and parse functions directly from source
(src.agent.orchestrator.turn_classifier / src.agent.domain_router) rather
than re-deriving them from log text, so there's no drift risk between what
this eval tests and what the app actually sends/parses.

Usage:
  python3 eval_classifier_swap.py
  python3 eval_classifier_swap.py --limit-turn 60 --limit-router 30
  python3 eval_classifier_swap.py --model qwen3.5:0.8b --output my_eval.json
"""

from __future__ import annotations

import sys
import os
import re
import json
import glob
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from src.agent.orchestrator.turn_classifier import (
    _CLASSIFIER_SYSTEM_PROMPT, _parse_turn_json, _VALID_ACTIONS, _VALID_REFERENCES,
)
from src.agent.domain_router import _ROUTER_SYSTEM_PROMPT, _parse_domain_json

CLASSIFIER_NUM_CTX = 8192

_TURN_PROMPT_RE = re.compile(
    r">>> TURN CLASSIFIER PROMPT\n={10,}\n(.*?)\n\n\n={10,}\n>>> TURN CLASSIFIER RAW OUTPUT\n={10,}\n(.*?)\n\n",
    re.DOTALL,
)
_ROUTER_PROMPT_RE = re.compile(
    r">>> ROUTER TIER-3 PROMPT\n={10,}\n(.*?)\n\n\n={10,}\n>>> ROUTER TIER-3 RAW OUTPUT\n={10,}\n(.*?)\n\n",
    re.DOTALL,
)


def _extract_turn_examples(text: str) -> list[dict]:
    """Each match pairs a logged TURN CLASSIFIER PROMPT (system+user combined,
    per turn_classifier.py's _dbg call) with the RAW OUTPUT qwen3.5:9b
    actually produced for it — that raw output, parsed, is ground truth."""
    out = []
    prefix = _CLASSIFIER_SYSTEM_PROMPT + "\n\n"
    for prompt_block, raw_output in _TURN_PROMPT_RE.findall(text):
        if not prompt_block.startswith(prefix):
            continue  # prompt shape drifted from current source — skip rather than mis-split
        user_content = prompt_block[len(prefix):]
        expected = _parse_turn_json(raw_output)
        if not expected or "action" not in expected or "reference" not in expected:
            continue  # can't compare against ground truth we can't even parse
        out.append({"user_content": user_content, "expected": expected})
    return out


def _extract_router_examples(text: str) -> list[dict]:
    prefix = _ROUTER_SYSTEM_PROMPT + "\n\nUSER: "
    out = []
    for prompt_block, raw_output in _ROUTER_PROMPT_RE.findall(text):
        if not prompt_block.startswith(prefix):
            continue
        question = prompt_block[len(prefix):]
        expected = _parse_domain_json(raw_output)
        if not expected or "domain" not in expected:
            continue
        out.append({"question": question, "expected": expected})
    return out


def _collect_examples(repo_root: str) -> tuple[list[dict], list[dict]]:
    turn_examples, router_examples = [], []
    for path in sorted(glob.glob(os.path.join(repo_root, "debug_llm_*.log"))):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        turn_examples += _extract_turn_examples(text)
        router_examples += _extract_router_examples(text)
    return turn_examples, router_examples


def _dedupe(examples: list[dict], key_fn) -> list[dict]:
    seen = set()
    out = []
    for ex in examples:
        k = key_fn(ex)
        if k in seen:
            continue
        seen.add(k)
        out.append(ex)
    return out


def _chat(client: httpx.Client, base_url: str, model: str, messages: list[dict]) -> str:
    # think=False and format="json" mirror what llm/ollama.py's build_ollama_client
    # sets for the real reasoning_llm client (json_mode=True, thinking=False) — the
    # real call sites (turn_classifier.py/domain_router.py) get these for free via
    # that client; a raw HTTP call here must set them explicitly or qwen3.5 models
    # burn the entire num_predict budget on hidden <thinking> content and return an
    # empty content string (documented incident in ollama.py's build_ollama_client).
    resp = client.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"num_predict": 300, "temperature": 0.0, "num_ctx": CLASSIFIER_NUM_CTX},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return (resp.json().get("message", {}) or {}).get("content", "").strip()


def run_turn_eval(client, base_url, model, examples: list[dict]) -> list[dict]:
    rows = []
    for ex in examples:
        raw = _chat(client, base_url, model, [
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": ex["user_content"]},
        ])
        actual = _parse_turn_json(raw) or {}
        exp_action = str(ex["expected"].get("action", "")).upper()
        exp_ref = str(ex["expected"].get("reference", "")).upper()
        act_action = str(actual.get("action", "")).upper()
        act_ref = str(actual.get("reference", "")).upper()
        rows.append({
            "kind": "turn",
            "question_snippet": ex["user_content"][-200:],
            "expected_action": exp_action, "actual_action": act_action,
            "expected_reference": exp_ref, "actual_reference": act_ref,
            "action_match": exp_action == act_action,
            "reference_match": exp_ref == act_ref,
            "full_match": exp_action == act_action and exp_ref == act_ref,
            "actual_raw": raw[:300],
        })
    return rows


def run_router_eval(client, base_url, model, examples: list[dict]) -> list[dict]:
    rows = []
    for ex in examples:
        raw = _chat(client, base_url, model, [
            {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ])
        actual = _parse_domain_json(raw) or {}
        exp_domain = str(ex["expected"].get("domain", "")).upper()
        act_domain = str(actual.get("domain", "")).upper()
        rows.append({
            "kind": "router",
            "question": ex["question"][:200],
            "expected_domain": exp_domain, "actual_domain": act_domain,
            "full_match": exp_domain == act_domain,
            "actual_raw": raw[:300],
        })
    return rows


def _print_summary(label: str, rows: list[dict]) -> float:
    if not rows:
        print(f"  {label}: no examples found in logs — nothing to evaluate.")
        return float("nan")
    matches = sum(1 for r in rows if r["full_match"])
    rate = matches / len(rows)
    print(f"  {label}: {matches}/{len(rows)} full matches ({rate:.1%})")
    mismatches = [r for r in rows if not r["full_match"]]
    for r in mismatches[:10]:
        if r["kind"] == "turn":
            print(f"    MISMATCH  ...{r['question_snippet'][-80:]!r}")
            print(f"      expected action={r['expected_action']} reference={r['expected_reference']}")
            print(f"      actual   action={r['actual_action']} reference={r['actual_reference']}")
        else:
            print(f"    MISMATCH  {r['question']!r}")
            print(f"      expected domain={r['expected_domain']}  actual domain={r['actual_domain']}")
    if len(mismatches) > 10:
        print(f"    ... and {len(mismatches) - 10} more mismatches (see saved JSON)")
    return rate


def main():
    p = argparse.ArgumentParser(description="Accuracy-diff eval: qwen3.5:0.8b vs qwen3.5:9b on real turn-classifier/router decisions")
    p.add_argument("--model", default="qwen3.5:0.8b")
    p.add_argument("--limit-turn", type=int, default=60)
    p.add_argument("--limit-router", type=int, default=30)
    p.add_argument("--pass-threshold", type=float, default=0.90,
                    help="full-match rate below this is flagged as NOT SAFE TO SHIP")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    from src.config.settings import get_settings
    settings = get_settings()
    base_url = settings.ollama_url

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    turn_examples, router_examples = _collect_examples(repo_root)
    turn_examples = _dedupe(turn_examples, lambda e: e["user_content"])[: args.limit_turn]
    router_examples = _dedupe(router_examples, lambda e: e["question"])[: args.limit_router]

    sep = "=" * 70
    print(f"\n{sep}\n  Classifier-swap accuracy eval — {base_url}\n"
          f"  candidate model: {args.model}\n"
          f"  turn-classifier examples found: {len(turn_examples)}\n"
          f"  router-tier3 examples found:    {len(router_examples)}\n{sep}\n")

    if not turn_examples and not router_examples:
        print("No real examples extracted from debug_llm_*.log at the repo root — "
              "nothing to evaluate. Exercise the app for some multi-turn conversations "
              "first so there's real qwen3.5:9b traffic to diff against.")
        return

    with httpx.Client() as client:
        print("Running turn-classifier comparisons...")
        turn_rows = run_turn_eval(client, base_url, args.model, turn_examples)
        print("Running router-tier3 comparisons...")
        router_rows = run_router_eval(client, base_url, args.model, router_examples)

    print(f"\n{sep}\n  RESULTS\n{sep}")
    turn_rate = _print_summary("turn-classifier", turn_rows)
    router_rate = _print_summary("router-tier3", router_rows)

    print(f"\n{sep}")
    rates = [r for r in (turn_rate, router_rate) if r == r]  # drop NaN
    if rates and min(rates) < args.pass_threshold:
        print(f"  RESULT: NOT SAFE TO SHIP — full-match rate below {args.pass_threshold:.0%} threshold.")
        print(f"  Do not wire {args.model} into turn_classifier.py/domain_router.py as-is.")
    elif not rates:
        print("  RESULT: INCONCLUSIVE — no examples of either kind were found.")
    else:
        print(f"  RESULT: PASS — both rates at/above {args.pass_threshold:.0%} threshold.")
    print(f"{sep}\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = args.output or f"eval_classifier_swap_{ts}.json"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": ts, "base_url": base_url, "model": args.model,
            "turn_match_rate": turn_rate, "router_match_rate": router_rate,
            "turn_rows": turn_rows, "router_rows": router_rows,
        }, f, indent=2)
    print(f"Results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
