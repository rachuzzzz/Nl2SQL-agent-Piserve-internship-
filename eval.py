"""
NL2SQL Agent — Evaluation Test Set (57 queries)
================================================
Usage:
  python3 eval.py
  python3 eval.py --model-pair "llama3.1:8b + deepseek:6.7b"
  python3 eval.py --skip-multiturn
  python3 eval.py --category "Aggregate"
  python3 eval.py --output my_results.json
"""

TEST_SET = [
    # Aggregates
    {"id":"AGG-01","category":"Aggregate","question":"what is the average inspection score","expected_contains":["86","avg","score"],"expected_llm_calls":1,"notes":"Baseline"},
    {"id":"AGG-02","category":"Aggregate","question":"how many inspections were completed this year","expected_contains":["inspection"],"expected_llm_calls":1,"notes":"Date filter + count"},
    {"id":"AGG-03","category":"Aggregate","question":"what is the total capex and opex across all corrective actions","expected_contains":["capex","opex"],"expected_llm_calls":1,"notes":"Two aggregates"},
    {"id":"AGG-04","category":"Aggregate","question":"how many open corrective actions are there","expected_contains":["open"],"expected_llm_calls":1,"notes":"Status filter count"},
    {"id":"AGG-05","category":"Aggregate","question":"how many inspections were completed this month vs last month","expected_contains":["month"],"expected_llm_calls":1,"notes":"date_trunc comparison"},
    {"id":"AGG-06","category":"Aggregate","question":"how many inspections are overdue this quarter","expected_contains":["overdue"],"expected_llm_calls":1,"notes":"Quarter + date filter"},
    {"id":"AGG-07","category":"Aggregate","question":"how many corrective actions were raised vs closed this quarter","expected_contains":["open","closed"],"expected_llm_calls":1,"notes":"FILTER clause"},
    {"id":"AGG-08","category":"Aggregate","question":"what is the average inspection duration in hours","expected_contains":["hour"],"expected_llm_calls":1,"notes":"start_date_time / end_date_time"},
    {"id":"AGG-09","category":"Aggregate","question":"what percentage of inspections resulted in at least one corrective action","expected_contains":["%"],"expected_llm_calls":1,"notes":"LEFT JOIN + COUNT DISTINCT ~66.8%"},
    {"id":"AGG-10","category":"Aggregate","question":"what is the average number of observations per inspection","expected_contains":["4","observation"],"expected_llm_calls":1,"notes":"Subquery AVG ~4.2"},
    # Filters
    {"id":"FLT-01","category":"Filter","question":"show the last 5 inspections by submission date","expected_contains":["inspection"],"expected_llm_calls":1,"notes":"ORDER BY + LIMIT"},
    {"id":"FLT-02","category":"Filter","question":"show all overdue corrective actions","expected_contains":["action"],"expected_llm_calls":1,"notes":"Date comparison filter"},
    {"id":"FLT-03","category":"Filter","question":"show high risk observations","expected_contains":["high","observation"],"expected_llm_calls":1,"notes":"answer_text filter"},
    {"id":"FLT-04","category":"Filter","question":"which inspections were returned for modification","expected_contains":["inspection"],"expected_llm_calls":1,"notes":"Enum status filter"},
    {"id":"FLT-05","category":"Filter","question":"show me inspections with a score below 70","expected_contains":["score","inspection"],"expected_llm_calls":1,"notes":"Numeric threshold ~16"},
    {"id":"FLT-06","category":"Filter","question":"how many inspections were done on weekends","expected_contains":["2","inspection"],"expected_llm_calls":1,"notes":"EXTRACT(ISODOW)"},
    {"id":"FLT-07","category":"Filter","question":"which facilities haven't been inspected in the last 30 days","expected_contains":["facility"],"expected_llm_calls":1,"notes":"NOT IN subquery"},
    # Joins
    {"id":"JOI-01","category":"Join","question":"show inspection scores grouped by facility","expected_contains":["facility","score"],"expected_llm_calls":1,"notes":"GROUP BY + JOIN"},
    {"id":"JOI-02","category":"Join","question":"which inspector has conducted the most inspections this year","expected_contains":["inspector","neenu"],"expected_llm_calls":1,"notes":"JOIN users + COUNT"},
    {"id":"JOI-03","category":"Join","question":"show average inspection score by inspection type","expected_contains":["type","score"],"expected_llm_calls":1,"notes":"JOIN inspection_type"},
    {"id":"JOI-04","category":"Join","question":"show corrective actions with their inspection facility and inspector","expected_contains":["facility","inspector"],"expected_llm_calls":1,"notes":"4-table join"},
    {"id":"JOI-05","category":"Join","question":"what is the current open corrective action count by facility","expected_contains":["facility","open"],"expected_llm_calls":1,"notes":"Correct JOIN direction"},
    {"id":"JOI-06","category":"Join","question":"which facility has the highest number of overdue corrective actions","expected_contains":["facility","overdue"],"expected_llm_calls":1,"notes":"ADJD AA -Al Roya: 45"},
    {"id":"JOI-07","category":"Join","question":"which client has the most overdue corrective actions","expected_contains":["client"],"expected_llm_calls":1,"notes":"Kingfield: 72"},
    {"id":"JOI-08","category":"Join","question":"what is the total capex and opex by facility","expected_contains":["facility","capex","opex"],"expected_llm_calls":1,"notes":"ica->ir->facility chain"},
    {"id":"JOI-09","category":"Join","question":"show me the top 5 facilities by average inspection score","expected_contains":["facility","score","bridges"],"expected_llm_calls":1,"notes":"ORDER BY avg DESC LIMIT 5"},
    {"id":"JOI-10","category":"Join","question":"show facilities where every inspection this year scored above 90","expected_contains":["facility","score"],"expected_llm_calls":1,"notes":"HAVING MIN(score) > 90"},
    {"id":"JOI-11","category":"Join","question":"show facilities that have been inspected more than 5 times this year","expected_contains":["facility"],"expected_llm_calls":1,"notes":"HAVING COUNT > 5"},
    {"id":"JOI-12","category":"Join","question":"which projects have not had any inspections this year","expected_contains":["project"],"expected_llm_calls":1,"notes":"NOT IN subquery ~42 projects"},
    # Form Answers
    {"id":"ANS-01","category":"Form Answers","question":"what are the most common observation types across all inspections","expected_contains":["deviation","observation"],"expected_llm_calls":1,"notes":"get_answer_stats Deviation:359"},
    {"id":"ANS-02","category":"Form Answers","question":"list all questions and answers from the most recent inspection form","expected_contains":["question","answer"],"expected_llm_calls":1,"notes":"module_name filter regression"},
    {"id":"ANS-03","category":"Form Answers","question":"show all risk levels from the last 5 inspections","expected_contains":["risk"],"expected_llm_calls":1,"notes":"May be empty if DRAFT"},
    {"id":"ANS-04","category":"Form Answers","question":"what did inspectors write in the observation field for Al Ghadeer","expected_contains":["observation","ghadeer"],"expected_llm_calls":1,"notes":"Cross-domain 4-table JOIN"},
    {"id":"ANS-05","category":"Form Answers","question":"what are the most common causes of corrective actions","expected_contains":["cause"],"expected_llm_calls":1,"notes":"ica.cause NOT form answers"},
    {"id":"ANS-06","category":"Form Answers","question":"what are the most common risk levels across all inspections","expected_contains":["high","medium","low"],"expected_llm_calls":1,"notes":"get_answer_stats High:230"},
    {"id":"ANS-07","category":"Form Answers","question":"what are the repetitive observations in the last 6 months","expected_contains":["observation"],"expected_llm_calls":1,"notes":"GROUP BY HAVING COUNT > 1"},
    {"id":"ANS-08","category":"Form Answers","question":"which observation appears most frequently as high risk","expected_contains":["observation"],"expected_llm_calls":1,"notes":"Subquery: risk High -> obs text"},
    # Scores
    {"id":"SCR-01","category":"Scores","question":"show me questions with their scores for the last inspection by George","expected_contains":["score","george"],"expected_llm_calls":1,"notes":"ai_answers.score + users JOIN"},
    {"id":"SCR-02","category":"Scores","question":"what is the average score for the risk level question","expected_contains":["score","risk"],"expected_llm_calls":1,"notes":"get_answer_stats avg 3.24"},
    {"id":"SCR-03","category":"Scores","question":"which inspection had the highest total score","expected_contains":["score","inspection"],"expected_llm_calls":1,"notes":"SUM ai_answers.score = 102"},
    {"id":"SCR-04","category":"Scores","question":"which inspection type has the highest average score","expected_contains":["type","score"],"expected_llm_calls":1,"notes":"Hygiene and grooming: 94"},
    {"id":"SCR-05","category":"Scores","question":"which inspector has the lowest average inspection score","expected_contains":["inspector","score"],"expected_llm_calls":1,"notes":"George Inspector: 69.5"},
    # Trends
    {"id":"TRD-01","category":"Trends","question":"how has the average inspection score changed month by month this year","expected_contains":["january","february"],"expected_llm_calls":1,"notes":"TO_CHAR + date_trunc"},
    {"id":"TRD-02","category":"Trends","question":"are corrective actions being closed faster or slower compared to last quarter","expected_contains":["quarter","days"],"expected_llm_calls":1,"notes":"close_on - created_on"},
    {"id":"TRD-03","category":"Trends","question":"which month had the most high risk observations this year","expected_contains":["january","month"],"expected_llm_calls":1,"notes":"GROUP BY month Jan:73"},
    {"id":"TRD-04","category":"Trends","question":"show inspections where the score dropped compared to the previous inspection at the same facility","expected_contains":["score","facility"],"expected_llm_calls":1,"notes":"LAG() PARTITION BY facility_id"},
    {"id":"TRD-05","category":"Trends","question":"which facility improved the most in score compared to last quarter","expected_contains":["facility"],"expected_llm_calls":1,"notes":"LAG() quarter filter"},
    {"id":"TRD-06","category":"Trends","question":"which inspection type has the most corrective actions raised","expected_contains":["type","action"],"expected_llm_calls":1,"notes":"Regulatory/Insurance:103"},
    # People
    {"id":"PPL-01","category":"People","question":"which inspector has conducted the most inspections this year","expected_contains":["inspector","neenu"],"expected_llm_calls":1,"notes":"neenu extinsp1:49"},
    {"id":"PPL-02","category":"People","question":"show breakdown of corrective action status by responsible party","expected_contains":["client","internal"],"expected_llm_calls":1,"notes":"responsible is ENUM not FK"},
    {"id":"PPL-03","category":"People","question":"which inspectors haven't submitted any inspections in the last 30 days","expected_contains":["inspector"],"expected_llm_calls":1,"notes":"NOT IN subquery"},
    # Multi-turn
    {"id":"MT-01","category":"Multi-turn","question":"which facility underwent the most recent inspection","expected_contains":["facility"],"expected_llm_calls":1,"notes":"Turn 1 stores context"},
    {"id":"MT-02","category":"Multi-turn","question":"who inspected it","expected_contains":["inspector"],"expected_llm_calls":1,"notes":"Turn 2 uses facility context","depends_on":"MT-01"},
    {"id":"MT-03","category":"Multi-turn","question":"fetch all questions and answers she filled for that site","expected_contains":["question","answer"],"expected_llm_calls":1,"notes":"Turn 3 inspection_id context (12584 regression)","depends_on":"MT-02"},
    {"id":"MT-04","category":"Multi-turn","question":"show more","expected_contains":[],"expected_llm_calls":0,"notes":"Pagination no SQL","depends_on":"MT-03"},
    # Edge Cases
    {"id":"EDG-01","category":"Edge Case","question":"show all high risk observations this month","expected_contains":["high","observation"],"expected_llm_calls":1,"notes":"Answer + date filter"},
    {"id":"EDG-02","category":"Edge Case","question":"how many questions are in the inspection form","expected_contains":["240","question"],"expected_llm_calls":1,"notes":"COUNT ai_questions = 240"},
    {"id":"EDG-03","category":"Edge Case","question":"show corrective actions where responsible is client","expected_contains":["client","action"],"expected_llm_calls":1,"notes":"Enum value filter"},
    {"id":"EDG-04","category":"Edge Case","question":"xyz abc nonsense query that makes no sense","expected_contains":["not","don"],"expected_llm_calls":1,"notes":"Garbage input graceful decline"},
]


# ── Runner ──────────────────────────────────────────────────────────────────

import sys, os, time, json, argparse
from datetime import datetime
from collections import defaultdict


def run_eval(model_pair_label="llama3.1:8b + deepseek:6.7b",
             skip_multiturn=False, category_filter=None, output_file=None):

    from agent.orchestrator import AgentOrchestrator, ConversationSession
    from dotenv import load_dotenv
    load_dotenv()

    tests = TEST_SET
    if category_filter:
        tests = [t for t in tests if t["category"].lower() == category_filter.lower()]
    if skip_multiturn:
        tests = [t for t in tests if t["category"] != "Multi-turn"]

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  InspectAI NL2SQL Evaluation — {model_pair_label}")
    print(f"  {len(tests)} questions  |  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{sep}\n")

    orch    = AgentOrchestrator.from_env()
    results = []
    session = ConversationSession()

    for i, test in enumerate(tests):
        cat, qid, q = test["category"], test["id"], test["question"]
        print(f"[{i+1:02d}/{len(tests)}] {qid} ({cat})  Q: {q[:65]}")
        t0 = time.time()
        try:
            result = orch.query(q, session=session if cat == "Multi-turn" else None)
            wall   = time.time() - t0
            al     = result.answer.lower()
            kh     = sum(1 for kw in test["expected_contains"] if kw.lower() in al)
            kt     = len(test["expected_contains"])
            kp     = round(kh / kt * 100) if kt else 100
            se     = sum(1 for s in result.steps
                         if s.tool == "execute_sql" and not s.result.get("success"))
            pe     = sum(1 for s in result.steps if s.tool == "[parse_error]")
            row    = dict(
                id=qid, category=cat, question=q, model_pair=model_pair_label,
                wall_s=round(wall, 2), llm_calls=result.stats.total_llm_calls,
                prompt_tokens=result.stats.total_prompt_tokens,
                completion_tokens=result.stats.total_completion_tokens,
                sql_errors=se, parse_errors=pe,
                keyword_hits=f"{kh}/{kt}", keyword_pct=kp,
                answer_preview=result.answer[:250], success=result.success,
                manual_score=None, notes=test["notes"],
            )
            results.append(row)
            perf = ("🟢" if result.stats.total_llm_calls <= 1
                    else "🟡" if result.stats.total_llm_calls <= 3 else "🔴")
            print(f"         {perf} {wall:.1f}s | {result.stats.total_llm_calls} calls | "
                  f"kw {kh}/{kt} | SQL {'OK' if se==0 else str(se)+' err'}")
            print(f"         -> {result.answer[:100]}\n")
        except Exception as e:
            wall = time.time() - t0
            print(f"         CRASH {wall:.1f}s: {e}\n")
            results.append(dict(
                id=qid, category=cat, question=q, model_pair=model_pair_label,
                wall_s=round(wall, 2), llm_calls=-1, prompt_tokens=0, completion_tokens=0,
                sql_errors=-1, parse_errors=-1, keyword_hits="0/0", keyword_pct=0,
                answer_preview=f"CRASH:{e}", success=False, manual_score=0, notes=test["notes"],
            ))

    total   = len(results)
    sql_ok  = sum(1 for r in results if r["sql_errors"] == 0)
    single  = sum(1 for r in results if r["llm_calls"] == 1)
    avg_t   = sum(r["wall_s"] for r in results) / total if total else 0
    tot_se  = sum(r["sql_errors"]   for r in results if r["sql_errors"]   >= 0)
    tot_pe  = sum(r["parse_errors"] for r in results if r["parse_errors"] >= 0)

    cs = defaultdict(lambda: {"count": 0, "sql_ok": 0, "single": 0, "time": 0.0})
    for r in results:
        c = r["category"]
        cs[c]["count"]  += 1
        cs[c]["sql_ok"] += (r["sql_errors"] == 0)
        cs[c]["single"] += (r["llm_calls"] == 1)
        cs[c]["time"]   += r["wall_s"]

    print(f"\n{sep}")
    print(f"  SUMMARY — {model_pair_label}")
    print(f"{sep}")
    print(f"  {'Category':<18} {'Tests':>5} {'SQL OK':>7} {'1-call':>7} {'AvgTime':>9}")
    print(f"  {'-'*50}")
    for cat, s in sorted(cs.items()):
        avg = s["time"] / s["count"] if s["count"] else 0
        print(f"  {cat:<18} {s['count']:>5} "
              f"{s['sql_ok']:>5}/{s['count']:<2} "
              f"{s['single']:>5}/{s['count']:<2} "
              f"{avg:>7.1f}s")
    print(f"\n  SQL success:  {sql_ok}/{total} ({100*sql_ok//total if total else 0}%)")
    print(f"  Single-call:  {single}/{total} ({100*single//total if total else 0}%)")
    print(f"  Avg time:     {avg_t:.1f}s")
    print(f"  SQL errors:   {tot_se}  |  Parse errors: {tot_pe}")

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = output_file or f"eval_results_{ts}.json"
    with open(fname, "w") as f:
        json.dump(dict(
            model_pair=model_pair_label, timestamp=ts,
            total=total, sql_ok=sql_ok, single_call=single,
            avg_time_s=round(avg_t, 2),
            sql_errors_total=tot_se, parse_errors_total=tot_pe,
            results=results,
        ), f, indent=2)
    print(f"\n  Results saved -> {fname}\n")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="InspectAI NL2SQL Evaluation")
    p.add_argument("--model-pair", default="llama3.1:8b + deepseek:6.7b")
    p.add_argument("--skip-multiturn", action="store_true")
    p.add_argument("--category", default=None,
                   help="Aggregate | Filter | Join | Form Answers | Scores | Trends | People | Multi-turn | Edge Case")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    run_eval(a.model_pair, a.skip_multiturn, a.category, a.output)