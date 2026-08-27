"""
Phase 3.23 — reasoning-model OUTPUT analysis (measurement pass).

The clean benchmark (3.22) put reasoning DECODE at 105.3s = 43% of the
242.7s suite wall — the single largest cost, and unlike prefill it is not
touched by prompt trimming. Before deciding "tighten the output contract"
vs "swap to a smaller reasoner" we need to know what the reasoning model
is actually generating, per question, token by token.

This script is OBSERVATION ONLY. It wraps orch.reasoning_llm.chat() with
a tee that records, for every reasoning call the suite makes (main ReAct
loop + TurnClassifier + any prefix-cache warm calls — all share the one
warm client):

  - raw completion text
  - llama.cpp `timings`: predicted_n / predicted_ms / prompt_n /
    prompt_ms / cache_n / draft_n / draft_n_accepted  (real, not est.)
  - openai `usage.completion_tokens` (cross-check)
  - caller function name (from the stack) -> tags main-loop vs aux calls
  - the question text in flight (from a wrapper on orch.query)

No eval.py changes, no prompt changes, DEBUG_LLM forced off. Output is a
JSONL at backend/eval/reasoning_output_3_23.jsonl — analyze_3_23.py turns
it into the per-segment token breakdown + distribution.

Usage:
  python profile_3_23_reasoning_output.py
  python profile_3_23_reasoning_output.py --category Aggregate
"""
import sys, os, json, time, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reasoning_output_3_23.jsonl")


def _caller_tag():
    """Walk the stack; report the first frame that looks like a real
    reasoning-call site, so aux calls (turn classify, domain route, cache
    warm) are separable from the main ReAct loop (_call_llm)."""
    names = []
    f = sys._getframe(2)
    for _ in range(12):
        if f is None:
            break
        names.append(f.f_code.co_name)
        f = f.f_back
    for n in names:
        if n in ("_call_llm",):
            return "main_loop"
        if n in ("classify",):
            return "turn_classifier"
        if n in ("route", "classify_domain", "_route"):
            return "domain_router"
        if n in ("warm", "record", "_warm", "prime"):
            return "prefix_cache_warm"
    return "other:" + ";".join(names[:4])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", default=None)
    p.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = p.parse_args()

    os.environ["DEBUG_LLM"] = "0"
    from dotenv import load_dotenv
    load_dotenv(r"C:\Users\thoma\Desktop\Pi\khidmah-nl2sql-prototype\.env")
    os.environ["DEBUG_LLM"] = "0"  # beat dotenv (override=False keeps ours)

    from src.agent.orchestrator import AgentOrchestrator

    import importlib.util
    _eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
    _spec = importlib.util.spec_from_file_location("eval_script", _eval_path)
    _eval_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_eval_module)
    run_eval = _eval_module.run_eval

    orch = AgentOrchestrator.from_env()

    fh = open(OUT, "w", encoding="utf-8")
    state = {"q": None, "call_ix": 0}

    def _emit(rec):
        rec["ts"] = round(time.time(), 3)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()

    # --- wrap orch.query to mark question boundaries -----------------------
    _orig_query = orch.query

    def _query_wrap(question, *args, **kw):
        state["q"] = question
        _emit({"type": "question", "q": question})
        return _orig_query(question, *args, **kw)

    orch.query = _query_wrap

    # --- wrap the shared reasoning client with a tee proxy ---------------
    # orch.reasoning_llm is a llama_index OpenAILike (pydantic + an
    # instrumentation-dispatcher-wrapped .chat), so neither instance nor
    # class monkeypatching of .chat is reliable. Wrap it in a plain proxy
    # that delegates everything via __getattr__ and only intercepts
    # chat()/complete(), then reassign it into every holder that captured
    # the raw client at construction time.
    _real_llm = orch.reasoning_llm

    def _record_chat(resp, wall_ms):
        try:
            content = resp.message.content
        except Exception:
            content = getattr(getattr(resp, "message", None), "content", "") or ""
        raw = getattr(resp, "raw", {}) or {}
        tm = raw.get("timings") if isinstance(raw, dict) else getattr(raw, "timings", {})
        tm = tm or {}
        usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
        uct = None
        if usage is not None:
            uct = usage.get("completion_tokens") if isinstance(usage, dict) \
                else getattr(usage, "completion_tokens", None)
        state["call_ix"] += 1
        _emit({
            "type": "call",
            "call_ix": state["call_ix"],
            "q": state["q"],
            "tag": _caller_tag(),
            "wall_ms": wall_ms,
            "content": content,
            "usage_completion_tokens": uct,
            "timings": {k: (tm.get(k) if isinstance(tm, dict) else getattr(tm, k, None))
                        for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
                                  "cache_n", "draft_n", "draft_n_accepted",
                                  "predicted_per_second", "predicted_per_token_ms")},
        })

    class _TeeLLM:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_real"), name, value)

        def chat(self, messages, *args, **kw):
            t0 = time.time()
            resp = object.__getattribute__(self, "_real").chat(messages, *args, **kw)
            try:
                _record_chat(resp, int((time.time() - t0) * 1000))
            except Exception:
                _emit({"type": "record_error", "trace": traceback.format_exc()})
            return resp

        def complete(self, *args, **kw):
            return object.__getattribute__(self, "_real").complete(*args, **kw)

    _tee = _TeeLLM(_real_llm)
    orch.reasoning_llm = _tee
    _restored = []
    for holder, attr in (
        (getattr(orch, "_turn_classifier", None), "reasoning_llm"),
        (getattr(orch, "_prefix_cache", None), "_llm_client"),
        (getattr(getattr(orch, "_turn_classifier", None), "_prefix_cache", None), "_llm_client"),
    ):
        if holder is not None and hasattr(holder, attr):
            _restored.append((holder, attr, getattr(holder, attr)))
            try:
                setattr(holder, attr, _tee)
            except Exception:
                pass
    print(f"[3.23] tee installed on reasoning_llm + {len(_restored)} aux holders")

    out = os.path.join(a.output_dir, "profile_3_23_reasoning_output.json")
    try:
        run_eval(model_pair_label="3.23 reasoning-output profiling | Qwen3.5-9B Q4_K_M + Qwen2.5-Coder-7B Q4_K_M",
                 skip_multiturn=False, category_filter=a.category, output_file=out,
                 profile_tokens=False, orch=orch)
    except Exception:
        _emit({"type": "error", "trace": traceback.format_exc()})
        raise
    finally:
        orch.reasoning_llm = _real_llm
        for holder, attr, orig in _restored:
            try:
                setattr(holder, attr, orig)
            except Exception:
                pass
        fh.close()
        print(f"\n[3.23] reasoning-call trace -> {OUT}")


if __name__ == "__main__":
    main()
