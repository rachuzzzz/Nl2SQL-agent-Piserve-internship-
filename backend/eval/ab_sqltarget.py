"""
3.26b — SQL TARGET-MODEL A/B harness.

Runs the full 105-question eval N times against whatever SQL target model
:10001 currently serves, with the 3.26 SQL-gen tee active so per-call
speculative-decode stats (draft acceptance, decode tok/s) are captured
alongside real_pass / sql_ok / truth_pass.

This is a CORRECTNESS experiment. Everything except the SQL target model
is held fixed: reasoning 9B, 0.5B draft, draft-simple, n-max=16, all
prompt/context machinery, the same eval suite. The target swap itself is
server-side on :10001 and is the operator's job (see steps below).

OPERATOR STEPS
  1. Current Coder-7B-Instruct live on :10001:
       python ab_sqltarget.py --label baseline
  2. Load candidate GGUF on :10001 (llama-server restart, user-side).
  3. python ab_sqltarget.py --label junmingg-text2sql
  4. python ab_sqltarget_compare.py --a baseline --b junmingg-text2sql
  5. Restore Coder-7B-Instruct on :10001.

Outputs per run i (1..N), written to repo root:
  bench_sqltarget_<label>_run<i>.json     — the eval result JSON
  sqlgen_ab_<label>_run<i>.jsonl          — per SQL-gen-call spec-decode trace
"""
import sys, os, json, time, argparse, statistics as st, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _server_model(url):
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
            d = json.loads(r.read())
        if isinstance(d, dict) and d.get("models"):
            return d["models"][0].get("model") or d["models"][0].get("name")
        if isinstance(d, dict) and d.get("data"):
            return d["data"][0].get("id")
    except Exception as e:
        return f"(probe failed: {e})"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="tag for this target model, e.g. 'baseline'")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--category", default=None, help="restrict to one eval category (quick sanity check)")
    ap.add_argument("--nmax", type=int, default=16, help="server draft n-max (round-estimation only)")
    ap.add_argument("--outdir", default=REPO)
    a = ap.parse_args()

    os.environ["DEBUG_LLM"] = "0"
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    os.environ["DEBUG_LLM"] = "0"

    from src.agent.orchestrator import AgentOrchestrator
    import importlib.util
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
    _s = importlib.util.spec_from_file_location("eval_script", _p)
    _m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_m)
    run_eval = _m.run_eval

    sql_url = os.environ.get("LLAMACPP_SQL_URL", "http://10.77.20.13:10001/v1")
    served = _server_model(sql_url)
    print(f"[ab] SQL target on {sql_url}: {served}")
    print(f"[ab] label={a.label!r}  runs={a.runs}")

    orch = AgentOrchestrator.from_env()

    box = {"fh": None, "q": None, "ix": 0}

    def emit(rec):
        if box["fh"] is None:
            return
        rec["ts"] = round(time.time(), 3)
        box["fh"].write(json.dumps(rec, ensure_ascii=False) + "\n")
        box["fh"].flush()

    _orig_query = orch.query

    def _query_wrap(question, *args, **kw):
        box["q"] = question
        emit({"type": "question", "q": question})
        return _orig_query(question, *args, **kw)
    orch.query = _query_wrap

    real = orch.sql_llm

    class Tee:
        def __init__(self, r): object.__setattr__(self, "_r", r)
        def __getattr__(self, n): return getattr(object.__getattribute__(self, "_r"), n)
        def __setattr__(self, n, v): setattr(object.__getattribute__(self, "_r"), n, v)
        def chat(self, *x, **k): return object.__getattribute__(self, "_r").chat(*x, **k)
        def complete(self, prompt, *x, **k):
            t0 = time.time()
            resp = object.__getattribute__(self, "_r").complete(prompt, *x, **k)
            wall_ms = int((time.time() - t0) * 1000)
            try:
                raw = getattr(resp, "raw", {}) or {}
                tm = raw.get("timings") if isinstance(raw, dict) else getattr(raw, "timings", {})
                tm = tm or {}
                g = (lambda kk: tm.get(kk) if isinstance(tm, dict) else getattr(tm, kk, None))
                box["ix"] += 1
                emit({"type": "call", "ix": box["ix"], "q": box["q"], "wall_ms": wall_ms,
                      "out_chars": len(getattr(resp, "text", "") or ""),
                      "prompt_n": g("prompt_n"), "prompt_ms": g("prompt_ms"), "cache_n": g("cache_n"),
                      "predicted_n": g("predicted_n"), "predicted_ms": g("predicted_ms"),
                      "predicted_per_second": g("predicted_per_second"),
                      "predicted_per_token_ms": g("predicted_per_token_ms"),
                      "draft_n": g("draft_n"), "draft_n_accepted": g("draft_n_accepted")})
            except Exception:
                emit({"type": "record_error"})
            return resp

    tee = Tee(real)
    orch.sql_llm = tee
    restored = []
    if hasattr(orch, "registry") and hasattr(orch.registry, "_sql_llm"):
        restored.append((orch.registry, "_sql_llm", orch.registry._sql_llm))
        orch.registry._sql_llm = tee

    summaries = []
    try:
        for i in range(1, a.runs + 1):
            box["ix"] = 0
            jl = os.path.join(a.outdir, f"sqlgen_ab_{a.label}_run{i}.jsonl")
            box["fh"] = open(jl, "w", encoding="utf-8")
            emit({"type": "meta", "label": a.label, "run": i, "sql_server_model": served,
                  "sql_url": sql_url})
            out = os.path.join(a.outdir, f"bench_sqltarget_{a.label}_run{i}.json")
            print(f"\n[ab] === {a.label} run {i}/{a.runs} ===")
            run_eval(model_pair_label=f"AB sqltarget={a.label} | {served}",
                     skip_multiturn=False, category_filter=a.category,
                     output_file=out, profile_tokens=False, orch=orch)
            box["fh"].close(); box["fh"] = None
            d = json.load(open(out))
            calls = [json.loads(l) for l in open(jl, encoding="utf-8")]
            calls = [c for c in calls if c.get("type") == "call" and c.get("draft_n")]
            acc = [100 * c["draft_n_accepted"] / c["draft_n"] for c in calls if c["draft_n"]]
            tps = [c["predicted_per_second"] for c in calls if c.get("predicted_per_second")]
            lb = d["latency_breakdown_ms"]
            s = dict(run=i, wall_s=round(lb["total_wall"] / 1000, 1),
                     sqlgen_decode_s=round(lb["sql_gen_decode"] / 1000, 1),
                     real_pass=d["real_pass"], sql_ok=d["sql_ok"], truth_pass=d["truth_pass"],
                     draft_acc_med=round(st.median(acc), 1) if acc else None,
                     decode_tps_med=round(st.median(tps), 1) if tps else None,
                     sql_calls=len(calls))
            summaries.append(s)
            print(f"[ab] run {i}: wall={s['wall_s']}s  real={s['real_pass']} sql_ok={s['sql_ok']} "
                  f"truth={s['truth_pass']}  draft_acc(med)={s['draft_acc_med']}%  "
                  f"decode(med)={s['decode_tps_med']} tok/s")
    finally:
        orch.sql_llm = real
        for h, at, ov in restored:
            try: setattr(h, at, ov)
            except Exception: pass
        if box["fh"]:
            box["fh"].close()

    agg = os.path.join(a.outdir, f"bench_sqltarget_{a.label}_summary.json")
    with open(agg, "w") as f:
        json.dump({"label": a.label, "sql_server_model": served, "runs": summaries}, f, indent=2)
    print(f"\n[ab] {a.label} medians: "
          f"wall={st.median(s['wall_s'] for s in summaries)}s  "
          f"real={st.median(s['real_pass'] for s in summaries)}  "
          f"sql_ok={st.median(s['sql_ok'] for s in summaries)}  "
          f"truth={st.median(s['truth_pass'] for s in summaries)}  "
          f"draft_acc={st.median(s['draft_acc_med'] for s in summaries if s['draft_acc_med'])}%")
    print(f"[ab] summary -> {agg}")


if __name__ == "__main__":
    main()
