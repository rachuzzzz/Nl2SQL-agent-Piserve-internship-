"""
Phase 3.23 analysis — turn reasoning_output_3_23.jsonl into the
per-segment token breakdown + distribution.

For every reasoning call the suite made, segment the raw completion:

  think        <think>...</think>              (Qwen3.5 thinking mode)
  pre_json     prose before the first '{'      (CoT / preamble)
  thought_val  value of the JSON "thought" key (in-band reasoning prose)
  tool+args    the tool name + args object     (necessary: the action)
  json_struct  braces / keys / whitespace      (necessary: format overhead)
  post_json    trailing prose after the JSON   (usually junk)

"necessary output"  = tool+args + json_struct  (+ final_answer text lives in args)
"reasoning prose"   = think + pre_json + thought_val + post_json

Real token counts via llama.cpp /tokenize on the reasoning server. Then
distribution (not just mean) across questions, and the decode-time and
tok/s split those segments imply against 3.22's 105.3s reasoning decode.

Usage:  python analyze_3_23.py
"""
import json, os, re, statistics as st, urllib.request

JSONL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reasoning_output_3_23.jsonl")
TOK_URL = "http://10.77.20.13:10000/tokenize"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

_tok_cache = {}


def ntok(s: str) -> int:
    if not s:
        return 0
    if s in _tok_cache:
        return _tok_cache[s]
    body = json.dumps({"content": s}).encode()
    req = urllib.request.Request(TOK_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        n = len(json.loads(r.read())["tokens"])
    _tok_cache[s] = n
    return n


def segment(content: str):
    """Return dict of segment strings for one completion."""
    seg = {"think": "", "pre_json": "", "thought_val": "", "tool_args": "",
           "post_json": "", "raw": content}
    rest = content
    m = _THINK.search(rest)
    if m:
        seg["think"] = m.group(1)
        rest = rest[:m.start()] + rest[m.end():]
    fm = _FENCE.search(rest)
    if fm:
        rest = fm.group(1)
    i = rest.find("{")
    if i < 0:
        seg["pre_json"] = rest
        return seg
    seg["pre_json"] = rest[:i]
    # find matching close brace
    depth = 0
    end = None
    for j in range(i, len(rest)):
        c = rest[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    blob = rest[i:end] if end else rest[i:]
    seg["post_json"] = rest[end:] if end else ""
    try:
        obj = json.loads(blob)
    except Exception:
        try:
            from json_repair import repair_json
            obj = json.loads(repair_json(blob))
        except Exception:
            obj = None
    if isinstance(obj, dict):
        seg["thought_val"] = str(obj.get("thought", "") or "")
        slim = {k: v for k, v in obj.items() if k != "thought"}
        seg["tool_args"] = json.dumps(slim, ensure_ascii=False)
        seg["_tool"] = obj.get("tool", "")
    else:
        seg["tool_args"] = blob
        seg["_tool"] = "?"
    return seg


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    return xs[lo] if lo == k or lo + 1 >= len(xs) else xs[lo] + (xs[lo + 1] - xs[lo]) * (k - lo)


def dist(name, xs, unit=""):
    xs = [x for x in xs if x is not None]
    if not xs:
        print(f"  {name:<26} (no data)")
        return
    print(f"  {name:<26} n={len(xs):<4} sum={sum(xs):>8.0f}{unit}  "
          f"min={min(xs):>6.1f}  p25={pct(xs,25):>6.1f}  med={st.median(xs):>6.1f}  "
          f"p75={pct(xs,75):>6.1f}  p90={pct(xs,90):>6.1f}  max={max(xs):>7.1f}  mean={st.mean(xs):>6.1f}")


def main():
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
    # group calls by question
    questions = []
    cur = None
    for r in rows:
        if r["type"] == "question":
            cur = {"q": r["q"], "calls": []}
            questions.append(cur)
        elif r["type"] == "call" and cur is not None:
            cur["calls"].append(r)

    main_calls = [c for q in questions for c in q["calls"] if c["tag"] == "main_loop"]
    other_tags = {}
    for q in questions:
        for c in q["calls"]:
            if c["tag"] != "main_loop":
                other_tags[c["tag"]] = other_tags.get(c["tag"], 0) + 1
    print(f"questions={len(questions)}  main_loop calls={len(main_calls)}  "
          f"other calls={other_tags or '{}'}")

    # ---- per-call segmentation + tokenisation ----
    per_call = []
    for c in main_calls:
        s = segment(c["content"])
        tk = {k: ntok(s[k]) for k in ("think", "pre_json", "thought_val", "tool_args", "post_json")}
        raw_tok = ntok(s["raw"])
        struct = max(0, raw_tok - sum(tk.values()))
        pred_n = (c["timings"] or {}).get("predicted_n") or c.get("usage_completion_tokens") or raw_tok
        pred_ms = (c["timings"] or {}).get("predicted_ms") or 0
        dn = (c["timings"] or {}).get("draft_n") or 0
        da = (c["timings"] or {}).get("draft_n_accepted") or 0
        per_call.append({
            "q": c["q"], "tool": s.get("_tool", "?"),
            "raw_tok": raw_tok, "pred_n": pred_n, "pred_ms": pred_ms,
            "think": tk["think"], "pre_json": tk["pre_json"],
            "thought_val": tk["thought_val"], "tool_args": tk["tool_args"],
            "post_json": tk["post_json"], "json_struct": struct,
            "prose": tk["think"] + tk["pre_json"] + tk["thought_val"] + tk["post_json"],
            "necessary": tk["tool_args"] + struct,
            "toks_per_s": (pred_n / pred_ms * 1000) if pred_ms else None,
            "draft_acc": (da / dn) if dn else None,
        })

    # ---- per-question aggregation ----
    byq = {}
    for pc in per_call:
        d = byq.setdefault(pc["q"], {"calls": 0, "pred_n": 0, "pred_ms": 0,
                                     "prose": 0, "necessary": 0, "thought_val": 0,
                                     "tool_args": 0, "json_struct": 0, "think": 0,
                                     "pre_json": 0, "post_json": 0})
        d["calls"] += 1
        for k in ("pred_n", "pred_ms", "prose", "necessary", "thought_val",
                  "tool_args", "json_struct", "think", "pre_json", "post_json"):
            d[k] += pc[k]

    qs = list(byq.values())
    tot_pred = sum(d["pred_n"] for d in qs)
    tot_prose = sum(d["prose"] for d in qs)
    tot_nec = sum(d["necessary"] for d in qs)
    tot_thought = sum(d["thought_val"] for d in qs)

    print("\n=== PER-QUESTION distribution (summed across that question's calls) ===")
    dist("reasoning calls / q", [d["calls"] for d in qs])
    dist("completion tokens / q", [d["pred_n"] for d in qs], " tok")
    dist("decode ms / q", [d["pred_ms"] for d in qs], " ms")
    dist("  prose tokens / q", [d["prose"] for d in qs], " tok")
    dist("  thought-field tok / q", [d["thought_val"] for d in qs], " tok")
    dist("  necessary tok / q", [d["necessary"] for d in qs], " tok")

    print("\n=== PER-CALL distribution ===")
    dist("completion tokens / call", [p["pred_n"] for p in per_call], " tok")
    dist("thought-field tok / call", [p["thought_val"] for p in per_call], " tok")
    dist("tool+args tok / call", [p["tool_args"] for p in per_call], " tok")
    dist("json struct tok / call", [p["json_struct"] for p in per_call], " tok")
    dist("think tok / call", [p["think"] for p in per_call], " tok")
    dist("pre/post prose tok/call", [p["pre_json"] + p["post_json"] for p in per_call], " tok")
    dist("decode tok/s / call", [p["toks_per_s"] for p in per_call])
    dist("draft acceptance / call", [p["draft_acc"] * 100 for p in per_call if p["draft_acc"] is not None], " %")

    print("\n=== COMPOSITION of all reasoning completion tokens ===")
    total = tot_pred or 1
    comp = [
        ("thought field (in-band prose)", tot_thought),
        ("pre/post-JSON prose", sum(d["pre_json"] + d["post_json"] for d in qs)),
        ("<think> block", sum(d["think"] for d in qs)),
        ("tool + args (the action)", sum(d["tool_args"] for d in qs)),
        ("JSON structural overhead", sum(d["json_struct"] for d in qs)),
    ]
    for label, v in comp:
        print(f"  {label:<32} {v:>8d} tok  {100*v/total:5.1f}%")
    print(f"  {'-'*32} {'-'*8}")
    print(f"  {'PROSE subtotal':<32} {tot_prose:>8d} tok  {100*tot_prose/total:5.1f}%")
    print(f"  {'NECESSARY subtotal':<32} {tot_nec:>8d} tok  {100*tot_nec/total:5.1f}%")
    print(f"  {'measured predicted_n total':<32} {tot_pred:>8d} tok")

    # ---- project onto 3.22's 105.3s reasoning decode ----
    tot_decode_ms = sum(d["pred_ms"] for d in qs)
    if tot_decode_ms:
        prose_ms = tot_decode_ms * tot_prose / total
        nec_ms = tot_decode_ms * tot_nec / total
        print(f"\n=== decode-time split (measured {tot_decode_ms/1000:.1f}s of reasoning decode this run) ===")
        print(f"  prose      ~{prose_ms/1000:6.1f}s")
        print(f"  necessary  ~{nec_ms/1000:6.1f}s")
        thought_ms = tot_decode_ms * tot_thought / total
        print(f"  (thought field alone ~{thought_ms/1000:.1f}s)")

    # ---- worst offenders ----
    print("\n=== top 12 questions by completion tokens ===")
    ranked = sorted(byq.items(), key=lambda kv: -kv[1]["pred_n"])[:12]
    print(f"  {'question':<52} {'calls':>5} {'pred_n':>7} {'prose':>6} {'thought':>7} {'nec':>6}")
    for q, d in ranked:
        print(f"  {q[:50]:<52} {d['calls']:>5} {d['pred_n']:>7} {d['prose']:>6} {d['thought_val']:>7} {d['necessary']:>6}")

    # sample the biggest single thought field
    big = sorted(per_call, key=lambda p: -p["thought_val"])[:3]
    print("\n=== 3 largest 'thought' fields (verbatim) ===")
    for pc in big:
        for r in rows:
            if r["type"] == "call" and r["q"] == pc["q"]:
                s = segment(r["content"])
                if abs(ntok(s["thought_val"]) - pc["thought_val"]) <= 1:
                    print(f"\n  [{pc['thought_val']} tok | tool={pc['tool']} | q={pc['q'][:55]}]")
                    print("   " + s["thought_val"][:600].replace("\n", "\n   "))
                    break


if __name__ == "__main__":
    main()
