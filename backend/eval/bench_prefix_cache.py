"""
PrefixCache benchmark — measures whether Ollama's own slot-based prefix
caching is actually being exploited by this app's real prompts.

Talks to Ollama's raw HTTP API directly (not through llama_index) so it can
control exact prefix conditions and read prompt_eval_count/prompt_eval_duration
straight off the response — the same fields PrefixCacheManager.record() will
read from live traffic (see llm/prefix_cache.py).

Cases (per domain, per role):
  cold                    — a never-seen prefix (nonce-suffixed). Full prefill.
  warm_repeat             — the identical full prompt sent immediately after.
  warm_prefix_varied_tail — same static prefix, a different question. This is
                            the real-world win Option A is built for.
  cross_domain_thrash     — inspection/audit requests interleaved on the same
                            shared Ollama server, quantifying slot eviction
                            risk from running both domains' orchestrators in
                            one process (see api/app.py).

Run this BEFORE applying the prompt-ordering fix / keep_alive change to get a
true baseline, and again AFTER to quantify the actual win.

Usage:
  python3 bench_prefix_cache.py
  python3 bench_prefix_cache.py --repeats 5
  python3 bench_prefix_cache.py --output my_bench.json
"""

from __future__ import annotations

import sys
import os
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

# Mirrors llm/factory.py's get_reasoning_llm/get_sql_llm num_ctx values —
# must cover the real prompt size for each role or Ollama truncates silently.
REASONING_NUM_CTX = 20000
SQL_NUM_CTX = 16384

QUESTIONS = {
    "inspection": [
        "how many inspections were completed this year",
        "show the last 5 inspections by submission date",
        "which facility has the highest number of overdue mitigative actions",
    ],
    "audit": [
        "list all audit plans",
        "which audit plans are still in draft",
        "show audit plans pending approval",
    ],
}


def _chat(client: httpx.Client, base_url: str, model: str, messages: list[dict],
          num_predict: int = 8, num_ctx: int = 20000) -> dict:
    # num_ctx MUST cover the real prompt size or Ollama silently truncates
    # (default context window is only 2048-4096) — a truncated prompt makes
    # every case report the same token count/duration regardless of cold vs.
    # warm, which looks like "no caching effect" but is actually a
    # measurement artifact, not a real finding. Mirrors the num_ctx values
    # llm/factory.py's get_reasoning_llm (20000) / get_sql_llm (16384) use
    # for the real roles — callers pass the matching value per role.
    t0 = time.time()
    resp = client.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0.0, "num_ctx": num_ctx},
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    data = resp.json()
    wall_ms = (time.time() - t0) * 1000
    # Ollama reports duration fields in nanoseconds.
    return {
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "prompt_eval_duration_ms": round(data.get("prompt_eval_duration", 0) / 1e6, 1),
        "eval_duration_ms": round(data.get("eval_duration", 0) / 1e6, 1),
        "load_duration_ms": round(data.get("load_duration", 0) / 1e6, 1),
        "wall_ms": round(wall_ms, 1),
    }


def _build_reasoning_prefix(domain: str):
    from src.agent.orchestrator import AgentOrchestrator
    orch = AgentOrchestrator.from_env(domain=domain)
    return orch._system_prompt, orch


def _build_sql_static_block(domain: str, orch):
    if domain == "audit":
        from src.sql.prompts_audit import SQL_GENERATION_PROMPT_AUDIT as template, CORE_CONTEXT_AUDIT as core
    else:
        from src.sql.prompts import SQL_GENERATION_PROMPT as template, CORE_CONTEXT as core
    schema_hint = orch.db_schema.for_sql_prompt_slim()
    return template, schema_hint, core


def _fill_sql_prompt(template: str, schema_hint: str, dynamic_examples: str, question: str) -> str:
    return (
        template
        .replace("{question}", question)
        .replace("{schema_hint}", schema_hint)
        .replace("{dynamic_examples}", dynamic_examples)
    )


def _print_row(r: dict) -> None:
    print(f"    {r['case']:<24} prefill={r['prompt_eval_duration_ms']:>7.0f}ms "
          f"tokens={r['prompt_eval_count']:>5} wall={r['wall_ms']:>7.0f}ms")


def run_case_reasoning(client, base_url, model, domain, prefix, questions, repeats):
    rows = []
    nonce_prefix = prefix + f"\n\n<!-- bench nonce {time.time()} -->"
    rows.append(dict(case="cold", domain=domain, role="reasoning",
                      **_chat(client, base_url, model,
                              [{"role": "system", "content": nonce_prefix},
                               {"role": "user", "content": questions[0]}],
                              num_ctx=REASONING_NUM_CTX)))
    for _ in range(repeats):
        rows.append(dict(case="warm_repeat", domain=domain, role="reasoning",
                          **_chat(client, base_url, model,
                                  [{"role": "system", "content": prefix},
                                   {"role": "user", "content": questions[0]}],
                                  num_ctx=REASONING_NUM_CTX)))
    for q in questions[1:]:
        rows.append(dict(case="warm_prefix_varied_tail", domain=domain, role="reasoning",
                          **_chat(client, base_url, model,
                                  [{"role": "system", "content": prefix},
                                   {"role": "user", "content": q}],
                                  num_ctx=REASONING_NUM_CTX)))
    return rows


def run_case_sql(client, base_url, model, domain, template, schema_hint, core, questions, repeats):
    rows = []
    nonce_core = core + f"\n-- bench nonce {time.time()}"
    rows.append(dict(case="cold", domain=domain, role="sql",
                      **_chat(client, base_url, model,
                              [{"role": "user", "content": _fill_sql_prompt(template, schema_hint, nonce_core, questions[0])}],
                              num_ctx=SQL_NUM_CTX)))
    for _ in range(repeats):
        rows.append(dict(case="warm_repeat", domain=domain, role="sql",
                          **_chat(client, base_url, model,
                                  [{"role": "user", "content": _fill_sql_prompt(template, schema_hint, core, questions[0])}],
                                  num_ctx=SQL_NUM_CTX)))
    for q in questions[1:]:
        rows.append(dict(case="warm_prefix_varied_tail", domain=domain, role="sql",
                          **_chat(client, base_url, model,
                                  [{"role": "user", "content": _fill_sql_prompt(template, schema_hint, core, q)}],
                                  num_ctx=SQL_NUM_CTX)))
    return rows


def run_cross_domain_thrash(client, base_url, model, role, domains, build_messages_fn, cycles=2):
    """Interleave requests across domains on the same shared Ollama server —
    directly quantifies slot eviction risk from api/app.py running both
    domains' orchestrators against one Ollama instance concurrently."""
    num_ctx = REASONING_NUM_CTX if role == "reasoning" else SQL_NUM_CTX
    rows = []
    for cycle in range(cycles):
        for d in domains:
            rows.append(dict(case="cross_domain_thrash", domain=d, role=role, cycle=cycle,
                              **_chat(client, base_url, model, build_messages_fn(d), num_ctx=num_ctx)))
    return rows


def main():
    p = argparse.ArgumentParser(description="PrefixCache benchmark against the real Ollama deployment")
    p.add_argument("--repeats", type=int, default=3, help="warm_repeat samples per role/domain")
    p.add_argument("--cycles", type=int, default=2, help="cross_domain_thrash interleave cycles")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    from src.config.settings import get_settings
    settings = get_settings()
    base_url = settings.ollama_url

    sep = "=" * 70
    print(f"\n{sep}\n  PrefixCache benchmark — {base_url}\n"
          f"  reasoning model: {settings.ollama_chat_model}\n"
          f"  sql model:       {settings.ollama_sql_model}\n{sep}\n")

    all_rows = []
    prefixes = {}
    sql_blocks = {}

    with httpx.Client() as client:
        for domain in ("inspection", "audit"):
            print(f"Building real prompts for domain={domain}...")
            prefix, orch = _build_reasoning_prefix(domain)
            prefixes[domain] = prefix
            sql_blocks[domain] = _build_sql_static_block(domain, orch)

        for domain in ("inspection", "audit"):
            print(f"\n-- reasoning role, domain={domain} --")
            rows = run_case_reasoning(client, base_url, settings.ollama_chat_model, domain,
                                       prefixes[domain], QUESTIONS[domain], args.repeats)
            all_rows += rows
            for r in rows:
                _print_row(r)

            print(f"-- sql role, domain={domain} --")
            template, schema_hint, core = sql_blocks[domain]
            rows = run_case_sql(client, base_url, settings.ollama_sql_model, domain,
                                 template, schema_hint, core, QUESTIONS[domain], args.repeats)
            all_rows += rows
            for r in rows:
                _print_row(r)

        print(f"\n-- cross_domain_thrash, reasoning role --")
        def _reasoning_messages(d):
            return [{"role": "system", "content": prefixes[d]},
                    {"role": "user", "content": QUESTIONS[d][0]}]
        rows = run_cross_domain_thrash(client, base_url, settings.ollama_chat_model, "reasoning",
                                        list(QUESTIONS.keys()), _reasoning_messages, args.cycles)
        all_rows += rows
        for r in rows:
            print(f"    cycle={r['cycle']} domain={r['domain']:<10} "
                  f"prefill={r['prompt_eval_duration_ms']:.0f}ms")

        print(f"\n-- cross_domain_thrash, sql role --")
        def _sql_messages(d):
            template, schema_hint, core = sql_blocks[d]
            return [{"role": "user", "content": _fill_sql_prompt(template, schema_hint, core, QUESTIONS[d][0])}]
        rows = run_cross_domain_thrash(client, base_url, settings.ollama_sql_model, "sql",
                                        list(QUESTIONS.keys()), _sql_messages, args.cycles)
        all_rows += rows
        for r in rows:
            print(f"    cycle={r['cycle']} domain={r['domain']:<10} "
                  f"prefill={r['prompt_eval_duration_ms']:.0f}ms")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = args.output or f"bench_prefix_cache_{ts}.json"
    with open(fname, "w") as f:
        json.dump(dict(timestamp=ts, base_url=base_url, rows=all_rows), f, indent=2)
    print(f"\n{sep}\nResults saved -> {fname}\n{sep}\n")


if __name__ == "__main__":
    main()
