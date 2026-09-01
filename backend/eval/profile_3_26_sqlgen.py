"""
Phase 3.26 — SQL-gen speculative-decoding characterisation (measurement).

3.22/3.23 established: reasoning decode (105s, 43% of wall) is optimised;
SQL decode (39s, 16%) at ~27.5% draft acceptance is the last serving lever.
Before recommending a different draft model (a server-side swap on :10001,
user-driven), measure the CURRENT 0.5B->7B pairing per call:

  - draft_n / draft_n_accepted  (real, from llama.cpp `timings`)
  - predicted_n / predicted_ms / predicted_per_token_ms / predicted_per_second
  - derived: est. draft rounds (draft_n / n_max), avg accepted per round,
    wasted draft forward-passes per round (n_max - accepted - ... )
  - distribution across calls + correlation with completion length

Observation only. Tee proxy on orch.sql_llm.complete(). DEBUG_LLM off.

Usage:  python profile_3_26_sqlgen.py [--category X] [--nmax 16]
"""
import sys, os, json, time, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sqlgen_3_26.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--nmax", type=int, default=16, help="server draft n-max, for round estimation only")
    ap.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    os.environ["DEBUG_LLM"] = "0"
    from dotenv import load_dotenv
    load_dotenv(r"C:\Users\thoma\Desktop\Pi\khidmah-nl2sql-prototype\.env")
    os.environ["DEBUG_LLM"] = "0"

    from src.agent.orchestrator import AgentOrchestrator
    import importlib.util
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
    _s = importlib.util.spec_from_file_location("eval_script", _p)
    _m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_m)
    run_eval = _m.run_eval

    orch = AgentOrchestrator.from_env()

    fh = open(OUT, "w", encoding="utf-8")
    state = {"q": None, "ix": 0}

    def emit(rec):
        rec["ts"] = round(time.time(), 3)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()

    _orig_query = orch.query

    def _query_wrap(question, *args, **kw):
        state["q"] = question
        emit({"type": "question", "q": question})
        return _orig_query(question, *args, **kw)
    orch.query = _query_wrap

    real = orch.sql_llm

    class Tee:
        def __init__(self, r): object.__setattr__(self, "_r", r)
        def __getattr__(self, n): return getattr(object.__getattribute__(self, "_r"), n)
        def __setattr__(self, n, v): setattr(object.__getattribute__(self, "_r"), n, v)
        def chat(self, *a, **k): return object.__getattribute__(self, "_r").chat(*a, **k)
        def complete(self, prompt, *a, **k):
            t0 = time.time()
            resp = object.__getattribute__(self, "_r").complete(prompt, *a, **k)
            wall_ms = int((time.time() - t0) * 1000)
            try:
                raw = getattr(resp, "raw", {}) or {}
                tm = raw.get("timings") if isinstance(raw, dict) else getattr(raw, "timings", {})
                tm = tm or {}
                g = (lambda k: tm.get(k) if isinstance(tm, dict) else getattr(tm, k, None))
                txt = getattr(resp, "text", "") or ""
                state["ix"] += 1
                emit({"type": "call", "ix": state["ix"], "q": state["q"],
                      "wall_ms": wall_ms, "out_chars": len(txt),
                      "prompt_n": g("prompt_n"), "prompt_ms": g("prompt_ms"),
                      "cache_n": g("cache_n"),
                      "predicted_n": g("predicted_n"), "predicted_ms": g("predicted_ms"),
                      "predicted_per_token_ms": g("predicted_per_token_ms"),
                      "predicted_per_second": g("predicted_per_second"),
                      "draft_n": g("draft_n"), "draft_n_accepted": g("draft_n_accepted")})
            except Exception:
                emit({"type": "record_error", "trace": traceback.format_exc()})
            return resp

    tee = Tee(real)
    orch.sql_llm = tee
    # ToolRegistry captured sql_llm at construction (tools.py: self._sql_llm);
    # generate_sql() reads it from there, so patch that holder too.
    restored = []
    if hasattr(orch, "registry") and hasattr(orch.registry, "_sql_llm"):
        restored.append((orch.registry, "_sql_llm", orch.registry._sql_llm))
        orch.registry._sql_llm = tee
    print(f"[3.26] tee on sql_llm + {len(restored)} registry holder(s)")

    out = os.path.join(a.output_dir, "profile_3_26_sqlgen.json")
    try:
        run_eval(model_pair_label="3.26 sqlgen spec-decode profiling",
                 skip_multiturn=False, category_filter=a.category,
                 output_file=out, profile_tokens=False, orch=orch)
    finally:
        orch.sql_llm = real
        for h, at, ov in restored:
            try: setattr(h, at, ov)
            except Exception: pass
        fh.close()
        print(f"\n[3.26] sqlgen trace -> {OUT}  (assumed n-max={a.nmax})")


if __name__ == "__main__":
    main()
