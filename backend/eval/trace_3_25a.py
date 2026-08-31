"""
3.25A — retry attribution. Run a targeted question subset through the
orchestrator (whatever reasoning model :10000 currently serves) and dump
the full AgentStep sequence per question: tool, ok/err, PG error class,
row count, generated SQL, and the reasoning 'thought' for each step.

Compares against nothing directly — pair its output with
reasoning_output_3_23.jsonl (the 9B run) by question text.

Usage:  python trace_3_25a.py            # default churn+truth subset
        python trace_3_25a.py --ids AGG-06,PORT-01
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_IDS = ["AGG-06", "PORT-01", "AGG-EXT-04",           # truth changed
               "CONV-03", "FLT-04", "ETL-02", "MT-02", "MT-04",  # +calls
               "DEF-02", "EDG-01", "JOI-14", "EDG-03"]       # +calls / big -calls
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_3_25a.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS))
    a = ap.parse_args()
    want = [s.strip() for s in a.ids.split(",") if s.strip()]

    os.environ["DEBUG_LLM"] = "0"
    from dotenv import load_dotenv
    load_dotenv(r"C:\Users\thoma\Desktop\Pi\khidmah-nl2sql-prototype\.env")
    os.environ["DEBUG_LLM"] = "0"

    from src.agent.orchestrator import AgentOrchestrator, ConversationSession
    from src.agent.orchestrator.loop import _failure_signature

    import importlib.util
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
    _s = importlib.util.spec_from_file_location("eval_script", _p)
    _m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_m)
    TEST_SET = _m.TEST_SET
    tests = {t["id"]: t for t in TEST_SET}

    orch = AgentOrchestrator.from_env()
    rmodel = getattr(orch.reasoning_llm, "model", "?")

    fh = open(OUT, "w", encoding="utf-8")

    def dump(qid, q, res, turn_ix):
        steps = []
        for s in getattr(res, "steps", []):
            env = s.result if isinstance(s.result, dict) else {}
            inner = env.get("result") if isinstance(env.get("result"), dict) else {}
            sql = ""
            if s.tool == "generate_sql":
                sql = (inner.get("sql") or "")
            elif s.tool == "execute_sql":
                sql = (s.args or {}).get("sql", "")
            err = env.get("error") or ""
            steps.append({
                "iter": s.iteration, "tool": s.tool,
                "ok": env.get("success"),
                "err_class": sorted(_failure_signature(err))[0][:60] if err else None,
                "err": err[:160] if err else None,
                "rows": (inner.get("row_count")
                         if inner.get("row_count") is not None
                         else (len(inner["rows"]) if isinstance(inner.get("rows"), list) else None)),
                "val_passed": (inner.get("validation") or {}).get("passed") if s.tool == "generate_sql" else None,
                "thought": (s.thought or "")[:200],
                "sql": " ".join(sql.split())[:400],
            })
        fh.write(json.dumps({
            "qid": qid, "turn_ix": turn_ix, "q": q,
            "reasoning_model": rmodel,
            "success": getattr(res, "success", None),
            "n_steps": len(steps),
            "n_llm_calls": getattr(getattr(res, "stats", None), "total_llm_calls", None),
            "answer": (getattr(res, "answer", "") or "")[:300],
            "steps": steps,
        }, ensure_ascii=False) + "\n")
        fh.flush()

    def chain_for(qid):
        """Build the depends_on chain ending at qid (for Multi-turn)."""
        seq = []
        cur = tests.get(qid)
        while cur:
            seq.append(cur)
            dep = cur.get("depends_on")
            cur = tests.get(dep) if dep else None
        return list(reversed(seq))

    done = set()
    for qid in want:
        t = tests.get(qid)
        if not t:
            print(f"  ?? unknown id {qid}")
            continue
        if qid in done:
            continue
        if t.get("category") == "Multi-turn" or t.get("depends_on"):
            seq = chain_for(qid)
            sess = ConversationSession()
            print(f"  running MT chain -> {qid}: {[s['id'] for s in seq]}")
            try:
                for k, step in enumerate(seq):
                    res = orch.query(step["question"], session=sess, visibility_scope={})
                    dump(step["id"], step["question"], res, k)
                    done.add(step["id"])
            except Exception as e:
                fh.write(json.dumps({"qid": qid, "error": repr(e)}) + "\n"); fh.flush()
        else:
            print(f"  running {qid} ({t['category']}) ...")
            try:
                res = orch.query(t["question"], session=None, visibility_scope={})
                dump(qid, t["question"], res, 0)
                done.add(qid)
            except Exception as e:
                fh.write(json.dumps({"qid": qid, "error": repr(e)}) + "\n"); fh.flush()

    fh.close()
    print(f"\n[3.25A] trace -> {OUT}  (reasoning model: {rmodel})")


if __name__ == "__main__":
    main()
