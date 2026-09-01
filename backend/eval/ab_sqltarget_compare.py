"""
3.26b — compare two SQL-target-model A/B runs produced by ab_sqltarget.py.

  python ab_sqltarget_compare.py --a baseline --b junmingg-text2sql

Prints median wall / sql-decode / draft-acceptance / correctness deltas,
per-question real_pass / sql_ok / truth_pass set-diffs (consistent across
all runs of each label), and flags any question that regressed.
"""
import sys, os, json, argparse, glob, statistics as st

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_label(label, outdir):
    benches = sorted(glob.glob(os.path.join(outdir, f"bench_sqltarget_{label}_run*.json")))
    if not benches:
        sys.exit(f"no runs found for label {label!r} in {outdir}")
    runs = [json.load(open(b)) for b in benches]
    jsonls = sorted(glob.glob(os.path.join(outdir, f"sqlgen_ab_{label}_run*.jsonl")))
    accs = []
    for jl in jsonls:
        cs = [json.loads(x) for x in open(jl, encoding="utf-8")]
        cs = [c for c in cs if c.get("type") == "call" and c.get("draft_n")]
        if cs:
            accs.append(100 * sum(c["draft_n_accepted"] for c in cs) / sum(c["draft_n"] for c in cs))
    return runs, accs


def med(runs, k):
    return st.median(d[k] for d in runs)


def med_lat(runs, k):
    return st.median(d["latency_breakdown_ms"][k] for d in runs) / 1000


def consistent_set(runs, pred):
    """IDs passing `pred` in EVERY run of this label."""
    sets = []
    for d in runs:
        sets.append({r["id"] for r in d["results"] if pred(r)})
    return set.intersection(*sets) if sets else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--outdir", default=REPO)
    args = ap.parse_args()

    A, Aacc = load_label(args.a, args.outdir)
    B, Bacc = load_label(args.b, args.outdir)
    print(f"A = {args.a}  ({len(A)} runs)   model: {A[0].get('model_pair')}")
    print(f"B = {args.b}  ({len(B)} runs)   model: {B[0].get('model_pair')}")

    print(f"\n{'metric':<22}{'A':>12}{'B':>12}{'B-A':>12}")
    for lab, fn in [
        ("wall s", lambda R: med_lat(R, "total_wall")),
        ("sql-gen s", lambda R: med_lat(R, "sql_gen")),
        ("  sql-gen decode s", lambda R: med_lat(R, "sql_gen_decode")),
        ("  sql-gen prefill s", lambda R: med_lat(R, "sql_gen_prefill")),
        ("reasoning s", lambda R: med_lat(R, "reasoning")),
        ("real_pass", lambda R: med(R, "real_pass")),
        ("sql_ok", lambda R: med(R, "sql_ok")),
        ("truth_pass", lambda R: med(R, "truth_pass")),
    ]:
        va, vb = fn(A), fn(B)
        print(f"{lab:<22}{va:>12.1f}{vb:>12.1f}{vb - va:>+12.1f}")
    if Aacc and Bacc:
        print(f"{'draft acceptance %':<22}{st.median(Aacc):>12.1f}{st.median(Bacc):>12.1f}"
              f"{st.median(Bacc) - st.median(Aacc):>+12.1f}")

    print("\n=== per-question set-diffs (consistent across all runs of each label) ===")
    for lab, pred in [
        ("real_pass", lambda r: bool(r.get("real_pass"))),
        ("sql_ok", lambda r: r.get("sql_errors", 0) == 0),
        ("truth_pass", lambda r: r.get("truth_match") is True),
    ]:
        sa, sb = consistent_set(A, pred), consistent_set(B, pred)
        lost, gained = sorted(sa - sb), sorted(sb - sa)
        flag = "  <-- REGRESSION" if lost else ""
        print(f"  {lab:<11} lost(B vs A)={lost}  gained={gained}{flag}")

    # flaky (differs across runs within a label) — context for the diffs above
    for name, runs in [(args.a, A), (args.b, B)]:
        ids = [r["id"] for r in runs[0]["results"]]
        flaky = {}
        for lab, pred in [("real_pass", lambda r: bool(r.get("real_pass"))),
                          ("truth_pass", lambda r: r.get("truth_match") is True)]:
            per = [{r["id"] for r in d["results"] if pred(r)} for d in runs]
            fl = [i for i in ids if len({i in s for s in per}) > 1]
            if fl:
                flaky[lab] = fl
        if flaky:
            print(f"  [{name}] intra-label flaky: {flaky}")


if __name__ == "__main__":
    main()
