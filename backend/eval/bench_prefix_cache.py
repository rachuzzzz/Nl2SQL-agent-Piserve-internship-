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
  classifier_thrash       — TurnClassifier's short, differently-shaped system
                            prompt interleaved with the reasoning role's own
                            (warmed) system prompt on the SAME model, in the
                            same order query() actually calls them (classify()
                            always runs immediately before the main loop's
                            first iteration). Compares the reasoning call's
                            prefill speed here against warm_repeat's (same
                            prefix, NOT preceded by a classifier call) — a
                            large gap is the direct evidence for the
                            TurnClassifier cache-thrash hypothesis in the
                            token/caching audit (see agent/orchestrator/
                            turn_classifier.py and loop.py's classify() call
                            site).

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
import concurrent.futures
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


def _simulated_per_question_schema_hint(schema_hint: str, question: str) -> str:
    """
    The real generate_sql() builds {schema_hint} per-question — it's not a
    fixed block. _build_additional_context() (sql/generation.py) appends a
    "Planner draft — editable" section carrying the reasoning LLM's own
    per-question draft SQL/mandatory hints on top of the schema reference,
    so two different questions in the same domain get DIFFERENT schema_hint
    text even though the underlying table/column reference is the same.
    warm_prefix_varied_tail exists specifically to measure whether the
    prefix cache survives that per-question variation — using a byte-
    identical schema_hint across every question (as this benchmark did
    before this function existed) can't detect the bug Option A/the
    token-caching-audit fix addresses at all, since nothing before
    {question} would ever actually change. This stub doesn't need to be a
    real SQL draft, only a plausible-sized, genuinely question-dependent
    addition, so the benchmark exercises the real caching boundary.
    """
    return (
        f"{schema_hint}\n\n"
        f"#### Planner draft — editable\n"
        f"-- draft for: {question}\n"
        f"SELECT 1 -- placeholder draft, length/content varies per question\n"
    )


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
    # Real per-question schema_hint variation (see
    # _simulated_per_question_schema_hint's docstring) — NOT the same
    # fixed schema_hint used above, since holding it constant across
    # questions can't exercise the byte-prefix-shift bug this case exists
    # to catch.
    for q in questions[1:]:
        rows.append(dict(case="warm_prefix_varied_tail", domain=domain, role="sql",
                          **_chat(client, base_url, model,
                                  [{"role": "user", "content": _fill_sql_prompt(
                                      template, _simulated_per_question_schema_hint(schema_hint, q), core, q)}],
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


def run_classifier_thrash(client, base_url, model, domain, reasoning_prefix,
                           classifier_prompt, questions, cycles=3):
    """Interleaves a classifier-shaped call (short, distinct system prompt)
    between reasoning-shaped calls on the SAME model, in the same order
    query() actually issues them — classify() (loop.py) always runs
    immediately before the main loop's own first .chat() call, every turn
    after the first. Compare this case's classifier_thrash_reasoning rows
    against run_case_reasoning's warm_repeat rows (identical reasoning_prefix,
    but NOT preceded by a classifier call): a much slower/colder prefill here
    is the direct evidence that TurnClassifier's differently-shaped prompt is
    evicting the reasoning role's own warmed prefix from Ollama's slot cache
    before the main loop gets to use it — the root-cause hypothesis in the
    token/caching audit, not yet confirmed by any measurement before this."""
    rows = []
    # A plausible-shaped CONVERSATION STATE block — real content doesn't
    # matter for a prefill-speed measurement, only that it's roughly the
    # size real traffic sends (see turn_classifier.py's _render_state).
    fake_state = (
        "CONVERSATION STATE:\n"
        "Recent turns (most recent last, 1 of up to 8 kept):\n"
        f"  Turn 1 [domain={domain}]: User: 'sample prior question' -> 'sample prior answer'\n"
        "Current active scope: (none)\n"
    )
    for cycle in range(cycles):
        q = questions[cycle % len(questions)]
        rows.append(dict(case="classifier_thrash_classifier", domain=domain, role="turn_classifier", cycle=cycle,
                          **_chat(client, base_url, model,
                                  [{"role": "system", "content": classifier_prompt},
                                   {"role": "user", "content": f"{fake_state}\n\nCURRENT MESSAGE: {q}"}],
                                  num_ctx=REASONING_NUM_CTX)))
        rows.append(dict(case="classifier_thrash_reasoning", domain=domain, role="reasoning", cycle=cycle,
                          **_chat(client, base_url, model,
                                  [{"role": "system", "content": reasoning_prefix},
                                   {"role": "user", "content": q}],
                                  num_ctx=REASONING_NUM_CTX)))
    return rows


def run_concurrent_warmup_thrash(client, base_url, model, domain, reasoning_prefix,
                                  classifier_prompt, questions, cycles=3):
    """
    Tests a DIFFERENT hypothesis than classifier_thrash: that one is fully
    SEQUENTIAL (classifier call completes, then the reasoning call starts) —
    with no two requests ever genuinely in flight together, num_parallel has
    no concurrency to schedule across a second slot, which is the confirmed
    reason NUM_PARALLEL=2 measured zero improvement there (see the
    token/caching audit's classifier_thrash results).

    This case creates REAL concurrency instead: at the moment the classifier
    call fires, a second request — just the reasoning role's static prefix,
    num_predict=1, fire-and-forget — is issued AT THE SAME TIME, on a
    separate thread, so both are genuinely in flight together. The
    classifier's own output isn't needed to build this warmup request (it
    only needs reasoning_prefix, which never depends on the classifier's
    action/reference), so this doesn't change the real data dependency
    that forces the classifier before the real reasoning call — the real
    call still fires only after both of these complete.

    If Ollama's slot-routing keys on prefix match for concurrent requests,
    the warmup should land on a different slot than the classifier and
    leave the reasoning prefix hot there, so the REAL reasoning call
    (fired after, same prefix) hits warm. If it doesn't — if concurrent
    requests still land on the same slot regardless of prefix, or slot
    assignment doesn't work this way at all — the real reasoning call's
    prefill should look the same as classifier_thrash's cold number.
    """
    rows = []
    fake_state = (
        "CONVERSATION STATE:\n"
        "Recent turns (most recent last, 1 of up to 8 kept):\n"
        f"  Turn 1 [domain={domain}]: User: 'sample prior question' -> 'sample prior answer'\n"
        "Current active scope: (none)\n"
    )
    for cycle in range(cycles):
        q = questions[cycle % len(questions)]
        classifier_messages = [
            {"role": "system", "content": classifier_prompt},
            {"role": "user", "content": f"{fake_state}\n\nCURRENT MESSAGE: {q}"},
        ]
        warmup_messages = [
            {"role": "system", "content": reasoning_prefix},
            {"role": "user", "content": "warmup"},
        ]
        t_concurrent_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_classifier = pool.submit(_chat, client, base_url, model, classifier_messages,
                                        num_predict=8, num_ctx=REASONING_NUM_CTX)
            f_warmup = pool.submit(_chat, client, base_url, model, warmup_messages,
                                    num_predict=1, num_ctx=REASONING_NUM_CTX)
            classifier_result = f_classifier.result()
            warmup_result = f_warmup.result()
        concurrent_wall_ms = round((time.time() - t_concurrent_start) * 1000, 1)

        rows.append(dict(case="concurrent_warmup_classifier", domain=domain, role="turn_classifier",
                          cycle=cycle, concurrent_wall_ms=concurrent_wall_ms, **classifier_result))
        rows.append(dict(case="concurrent_warmup_prefetch", domain=domain, role="reasoning_prefetch",
                          cycle=cycle, concurrent_wall_ms=concurrent_wall_ms, **warmup_result))

        # The REAL reasoning call — fired only after both above complete,
        # same as production (classify() must finish before the reasoning
        # prompt's session-context block, which needs its output, can be
        # built). This is the number that matters: is it warm now?
        rows.append(dict(case="concurrent_warmup_reasoning", domain=domain, role="reasoning", cycle=cycle,
                          **_chat(client, base_url, model,
                                  [{"role": "system", "content": reasoning_prefix},
                                   {"role": "user", "content": q}],
                                  num_ctx=REASONING_NUM_CTX)))
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

        print(f"\n-- classifier_thrash (TurnClassifier vs. reasoning role, same model) --")
        from src.agent.orchestrator.turn_classifier import _CLASSIFIER_SYSTEM_PROMPT
        for domain in ("inspection", "audit"):
            rows = run_classifier_thrash(
                client, base_url, settings.ollama_chat_model, domain,
                prefixes[domain], _CLASSIFIER_SYSTEM_PROMPT, QUESTIONS[domain], args.cycles,
            )
            all_rows += rows
            for r in rows:
                print(f"    domain={domain:<10} case={r['case']:<28} cycle={r['cycle']} "
                      f"prefill={r['prompt_eval_duration_ms']:>7.0f}ms tokens={r['prompt_eval_count']:>5}")

        print(f"\n-- concurrent_warmup_thrash (classifier + reasoning-prefix warmup fired concurrently) --")
        for domain in ("inspection", "audit"):
            rows = run_concurrent_warmup_thrash(
                client, base_url, settings.ollama_chat_model, domain,
                prefixes[domain], _CLASSIFIER_SYSTEM_PROMPT, QUESTIONS[domain], args.cycles,
            )
            all_rows += rows
            for r in rows:
                cw = f" concurrent_wall={r['concurrent_wall_ms']:.0f}ms" if "concurrent_wall_ms" in r else ""
                print(f"    domain={domain:<10} case={r['case']:<28} cycle={r['cycle']} "
                      f"prefill={r['prompt_eval_duration_ms']:>7.0f}ms tokens={r['prompt_eval_count']:>5}{cw}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = args.output or f"bench_prefix_cache_{ts}.json"
    with open(fname, "w") as f:
        json.dump(dict(timestamp=ts, base_url=base_url, rows=all_rows), f, indent=2)
    print(f"\n{sep}\nResults saved -> {fname}\n{sep}\n")


if __name__ == "__main__":
    main()
