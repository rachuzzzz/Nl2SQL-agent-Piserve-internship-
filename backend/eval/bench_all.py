"""
bench_all.py — one command to run the full NL2SQL regression gate.

Product-agnostic. Its job is a trustworthy, repeatable baseline so any future
change (accuracy, latency, model swap, prompt edit) is measured against the
same frozen point.

WHAT IT RUNS (no behavioural change to the existing runners — subprocess only):
  1. eval_truth.py                (comparator self-test, DB-free)   — gate
  2. eval_truth.py --smoke        (inspection canonical SQL)        — gate
  3. eval_truth.py --smoke --prefix AUDIT_  (audit canonical SQL)   — gate
  4. eval.py       --skip-multiturn                                 — inspection suite
  5. eval_audit.py --skip-multiturn                                 — audit suite
  (data-health preflight is already run inside 4 & 5 and copied out of their JSON.)

OUTPUT: a single machine-readable run_<ts>.json — stack config, gate results,
per-domain aggregates, and the full per-question record (id / domain / category
/ real_pass / truth_match / snapshot_blocked / sql_errors / llm_calls).

USAGE
  python bench_all.py                          # fresh run -> run_<ts>.json
  python bench_all.py --out baseline.json      # name it
  python bench_all.py --baseline baseline.json # run + diff vs baseline, gated
  python bench_all.py --test-diff baseline.json  # exercise the gate logic only (no suites)
  python bench_all.py --full                   # include multi-turn (default: --skip-multiturn)

REGRESSION GATE (exit 1 on any hard failure):
  PASS -> FAIL on real_pass_adjusted   ... hard   (unless newly snapshot_blocked)
  PASS -> FAIL on truth_match          ... hard   (unless newly snapshot_blocked)
  sql_ok count decreased               ... hard
  truth_pass_adjusted count decreased  ... hard
  real_pass_adjusted count decreased   ... hard
  snapshot_blocked set changed         ... report only (not a model regression)
  new / removed questions, coverage +/-, llm_calls moves, FAIL->PASS ... report only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_PY = sys.executable


# ── helpers ─────────────────────────────────────────────────────────────────
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env")
    except Exception:
        pass


def _stack_config() -> dict:
    g = os.environ.get
    return {
        "reasoning_provider": g("REASONING_LLM_PROVIDER", "ollama"),
        "sql_provider": g("SQL_LLM_PROVIDER", "ollama"),
        "reasoner_model": g("LLAMACPP_REASONING_MODEL")
                          or g("VLLM_REASONING_MODEL")
                          or g("OLLAMA_CHAT_MODEL", ""),
        "coder_model": g("LLAMACPP_SQL_MODEL")
                       or g("VLLM_SQL_MODEL")
                       or g("OLLAMA_SQL_MODEL", ""),
        "db_name": g("DB_NAME", ""),
        "db_host": g("DB_HOST", ""),
        "db_schema": g("DB_SCHEMA", "public"),
    }


def _run(cmd: list[str], *, cwd: Path = _HERE) -> tuple[int, str]:
    """Run a subprocess, stream nothing, return (exit_code, combined_output)."""
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    p = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _gate_line(out: str, needle: str) -> str:
    for ln in out.splitlines():
        if needle in ln:
            return ln.strip()
    return f"(marker {needle!r} not found)"


# ── suite normalisation ─────────────────────────────────────────────────────
_SUITE_KEYS = ("total", "real_pass", "real_pass_adjusted", "sql_ok",
               "truth_covered", "truth_covered_adjusted", "truth_pass",
               "truth_pass_adjusted", "snapshot_blocked", "avg_time_s")


def _summarise_suite(raw: dict, domain: str) -> dict:
    s = {k: raw.get(k) for k in _SUITE_KEYS}
    s["snapshot_blocked_ids"] = sorted(raw.get("snapshot_blocked_ids") or [])
    # fall back gracefully if a runner predates the adjusted keys
    if s["real_pass_adjusted"] is None:
        s["real_pass_adjusted"] = s["real_pass"]
    if s["truth_pass_adjusted"] is None:
        s["truth_pass_adjusted"] = s["truth_pass"]
    if s["truth_covered_adjusted"] is None:
        s["truth_covered_adjusted"] = s["truth_covered"]
    return s


def _per_question(raw: dict, domain: str) -> list[dict]:
    rows = []
    for r in raw.get("results", []):
        rows.append({
            "id": r["id"],
            "domain": domain,
            "category": r.get("category"),
            "real_pass": bool(r.get("real_pass")),
            "truth_covered": bool(r.get("truth_covered")),
            "truth_match": r.get("truth_match"),          # True / False / None
            "snapshot_blocked": bool(r.get("snapshot_blocked")),
            "sql_errors": r.get("sql_errors"),
            "llm_calls": r.get("llm_calls"),
        })
    return rows


def _truth_state(row: dict) -> str:
    if not row["truth_covered"]:
        return "not-cov"
    tm = row["truth_match"]
    return "PASS" if tm is True else ("FAIL" if tm is False else "ERR")


# ── run all ─────────────────────────────────────────────────────────────────
def run_all(*, skip_multiturn: bool) -> dict:
    _load_env()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    work = Path(tempfile.mkdtemp(prefix="bench_all_"))

    gates: dict = {}

    print("[1/5] eval_truth.py  (comparator self-test) ...", flush=True)
    rc, out = _run([_PY, "eval_truth.py"])
    gates["comparator_selftest"] = {"line": _gate_line(out, "self-tests passed"),
                                    "exit": rc, "ok": rc == 0}

    print("[2/5] eval_truth.py --smoke  (inspection canonical SQL) ...", flush=True)
    rc, out = _run([_PY, "eval_truth.py", "--smoke"])
    gates["smoke_inspection"] = {"line": _gate_line(out, "executed cleanly"),
                                 "exit": rc, "ok": rc == 0}

    print("[3/5] eval_truth.py --smoke --prefix AUDIT_  (audit canonical SQL) ...", flush=True)
    rc, out = _run([_PY, "eval_truth.py", "--smoke", "--prefix", "AUDIT_"])
    gates["smoke_audit"] = {"line": _gate_line(out, "executed cleanly"),
                            "exit": rc, "ok": rc == 0}

    mt = ["--skip-multiturn"] if skip_multiturn else []

    ins_out = work / "inspection.json"
    print(f"[4/5] eval.py {' '.join(mt)}  (inspection suite — this takes a few minutes) ...", flush=True)
    rc_i, _ = _run([_PY, "eval.py", *mt, "--output", str(ins_out),
                    "--model-pair", "bench_all"])
    aud_out = work / "audit.json"
    print(f"[5/5] eval_audit.py {' '.join(mt)}  (audit suite — this takes a few minutes) ...", flush=True)
    rc_a, _ = _run([_PY, "eval_audit.py", *mt, "--output", str(aud_out),
                    "--model-pair", "bench_all"])

    ins_raw = json.loads(ins_out.read_text(encoding="utf-8"))
    aud_raw = json.loads(aud_out.read_text(encoding="utf-8"))

    report = {
        "bench_all_version": 1,
        "timestamp": ts,
        "elapsed_s": round(time.time() - t0, 1),
        "skip_multiturn": skip_multiturn,
        "stack": _stack_config(),
        "gates": gates,
        "runner_exit": {"inspection": rc_i, "audit": rc_a},
        "health": {
            "inspection": ins_raw.get("data_health"),
            "audit": aud_raw.get("data_health"),
        },
        "suites": {
            "inspection": _summarise_suite(ins_raw, "inspection"),
            "audit": _summarise_suite(aud_raw, "audit"),
        },
        "per_question": _per_question(ins_raw, "inspection")
                        + _per_question(aud_raw, "audit"),
    }
    return report


# ── diff / gate ─────────────────────────────────────────────────────────────
def diff(baseline: dict, current: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (hard_failures, soft_reports, table_lines)."""
    hard: list[str] = []
    soft: list[str] = []
    tbl: list[str] = []

    B = {(r["domain"], r["id"]): r for r in baseline["per_question"]}
    C = {(r["domain"], r["id"]): r for r in current["per_question"]}

    # suite-level
    for dom in ("inspection", "audit"):
        b, c = baseline["suites"].get(dom, {}), current["suites"].get(dom, {})
        for k in ("sql_ok", "truth_pass_adjusted", "real_pass_adjusted"):
            bv, cv = b.get(k), c.get(k)
            if isinstance(bv, int) and isinstance(cv, int):
                if cv < bv:
                    hard.append(f"{dom}.{k} decreased: {bv} -> {cv}")
                elif cv != bv:
                    soft.append(f"{dom}.{k}: {bv} -> {cv}")
        # coverage / snapshot are report-only
        for k in ("truth_covered", "snapshot_blocked"):
            if b.get(k) != c.get(k):
                soft.append(f"{dom}.{k}: {b.get(k)} -> {c.get(k)}")
        if b.get("snapshot_blocked_ids") != c.get("snapshot_blocked_ids"):
            soft.append(f"{dom}.snapshot_blocked_ids: {b.get('snapshot_blocked_ids')} "
                        f"-> {c.get('snapshot_blocked_ids')}")

    # per-question
    for key in sorted(set(B) | set(C)):
        dom, qid = key
        b, c = B.get(key), C.get(key)
        if b is None:
            soft.append(f"NEW question {dom}/{qid} ({_truth_state(c)})")
            continue
        if c is None:
            soft.append(f"REMOVED question {dom}/{qid}")
            continue

        newly_blocked = c["snapshot_blocked"] and not b["snapshot_blocked"]

        # real_pass regression
        if b["real_pass"] and not c["real_pass"]:
            msg = f"{dom}/{qid}: real_pass PASS -> FAIL"
            (soft if newly_blocked else hard).append(
                msg + ("  (newly snapshot_blocked — reported, not gated)" if newly_blocked else ""))
            tbl.append(f"  {dom}/{qid:14} real_pass  PASS -> FAIL"
                       f"{'  [snapshot_blocked]' if newly_blocked else '  <== HARD'}")

        # truth regression
        sb, sc = _truth_state(b), _truth_state(c)
        if sb != sc:
            tbl.append(f"  {dom}/{qid:14} truth      {sb:8} -> {sc:8}"
                       + ("  <== HARD" if (sb == "PASS" and sc in ("FAIL", "ERR") and not newly_blocked)
                          else "  [snapshot_blocked]" if (sb == "PASS" and sc in ("FAIL", "ERR"))
                          else "  (improvement)" if (sc == "PASS")
                          else ""))
            if sb == "PASS" and sc in ("FAIL", "ERR"):
                msg = f"{dom}/{qid}: truth {sb} -> {sc}"
                (soft if newly_blocked else hard).append(
                    msg + ("  (newly snapshot_blocked — reported, not gated)" if newly_blocked else ""))

        # llm_calls move (report only)
        if b["llm_calls"] != c["llm_calls"]:
            soft.append(f"{dom}/{qid}: llm_calls {b['llm_calls']} -> {c['llm_calls']}")

    return hard, soft, tbl


def _print_summary(report: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  bench_all — {report['timestamp']}  ({report['elapsed_s']}s)")
    print(f"  stack: {report['stack']['reasoner_model']} + {report['stack']['coder_model']}"
          f"  | db {report['stack']['db_name']}")
    print("=" * 70)
    for name, g in report["gates"].items():
        mark = "OK " if g["ok"] else "FAIL"
        print(f"  [{mark}] {name:22} {g['line']}")
    for dom in ("inspection", "audit"):
        s = report["suites"][dom]
        print(f"  {dom:11}  real_pass {s['real_pass']}/{s['total']}"
              f" (adj {s['real_pass_adjusted']})"
              f"  | sql_ok {s['sql_ok']}/{s['total']}"
              f"  | truth {s['truth_pass']}/{s['truth_covered']}"
              f" (adj {s['truth_pass_adjusted']}/{s['truth_covered_adjusted']})"
              f"  | snap_blocked {s['snapshot_blocked']}")
        h = (report["health"][dom] or {}).get("checks", {})
        for dep, ch in h.items():
            print(f"                health: {dep} = {ch.get('status')}")
    print("=" * 70)


# ── gate-logic self test (no suites) ───────────────────────────────────────
def _test_diff(baseline_path: str) -> int:
    base = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    import copy

    clean = copy.deepcopy(base)
    hard, soft, _ = diff(base, clean)
    ok_clean = (len(hard) == 0)
    print(f"[test-diff] baseline vs itself: {len(hard)} hard, {len(soft)} soft "
          f"-> {'PASS (0 hard, exit 0)' if ok_clean else 'UNEXPECTED HARD FAILURES'}")

    bad = copy.deepcopy(base)
    flipped = []
    for r in bad["per_question"]:
        if r["truth_match"] is True and not r["snapshot_blocked"]:
            r["truth_match"] = False
            flipped.append(f"{r['domain']}/{r['id']}")
            break
    for r in bad["per_question"]:
        if r["real_pass"] and not r["snapshot_blocked"] and r["id"] not in flipped:
            r["real_pass"] = False
            flipped.append(f"{r['domain']}/{r['id']} (real_pass)")
            break
    for dom in ("inspection", "audit"):
        if isinstance(bad["suites"][dom].get("sql_ok"), int):
            bad["suites"][dom]["sql_ok"] -= 1
            bad["suites"][dom]["truth_pass_adjusted"] = \
                max(0, (bad["suites"][dom].get("truth_pass_adjusted") or 1) - 1)
            break
    hard2, soft2, _ = diff(base, bad)
    ok_bad = (len(hard2) >= 1)
    print(f"[test-diff] simulated regression (flipped {flipped}, sql_ok-1): "
          f"{len(hard2)} hard -> {'PASS (>=1 hard, exit 1)' if ok_bad else 'GATE FAILED TO TRIP'}")
    for h in hard2:
        print(f"             HARD: {h}")

    return 0 if (ok_clean and ok_bad) else 1


# ── main ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="NL2SQL full regression gate")
    ap.add_argument("--out", default=None, help="output path (default run_<ts>.json)")
    ap.add_argument("--baseline", default=None,
                    help="diff the run against this baseline JSON and gate on it")
    ap.add_argument("--full", action="store_true",
                    help="include multi-turn (default: --skip-multiturn, matches the frozen baseline)")
    ap.add_argument("--test-diff", default=None, metavar="BASELINE",
                    help="exercise the gate logic against a baseline (no suites run) and exit")
    a = ap.parse_args()

    if a.test_diff:
        return _test_diff(a.test_diff)

    report = run_all(skip_multiturn=not a.full)
    out_path = Path(a.out) if a.out else _HERE / f"run_{report['timestamp']}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _print_summary(report)
    print(f"\n  wrote {out_path}")

    # non-model gates first
    exit_code = 0
    for name, g in report["gates"].items():
        if not g["ok"]:
            print(f"\n  GATE FAILED: {name} ({g['line']})")
            exit_code = 1
    for dom, rc in report["runner_exit"].items():
        # eval_audit exits non-zero on a real_pass fail; that's informational here,
        # the diff gate below is authoritative for regressions.
        if rc not in (0, 1):
            print(f"\n  RUNNER ERROR: {dom} exited {rc}")
            exit_code = 1

    if a.baseline:
        base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
        hard, soft, tbl = diff(base, report)
        print("\n" + "-" * 70)
        print(f"  DIFF vs {a.baseline}")
        print("-" * 70)
        for ln in tbl:
            print(ln)
        if soft:
            print("\n  report-only changes:")
            for s in soft:
                print(f"    - {s}")
        if hard:
            print("\n  HARD FAILURES:")
            for h in hard:
                print(f"    ✗ {h}")
            exit_code = 1
        else:
            print("\n  no hard failures — regression gate PASSED")
    else:
        print("\n  (no --baseline given — this run can be frozen as the baseline)")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
