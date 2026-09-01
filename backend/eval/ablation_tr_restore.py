"""
Causal ablation (RESTORE direction) — does putting the Phase 3.15-cut
table_relationships sections back into the SQL-gen schema_hint change
result-level correctness?

Context
-------
Phase 3.15 permanently stopped for_schema_hint() from rendering 6 of the 9
table_relationships sections (WHERE CONCEPTS LIVE, HOW TABLES CONNECT,
TABLE COLUMNS, CRITICAL ai_* JOIN PATTERN, MULTI-HOP EXAMPLES, RULE: TYPE
names). Phase 3.16 then per-question-gated the remaining 3. That cut was
validated only against real_pass/sql_ok — the truth layer was silently
returning 0/29 for everyone at the time (extract_agent_rows bug, fixed
2026-08-27). With the truth layer now live (16/29), three aggregate cases
diverge semantically: AGG-07, AGG-08, AGG-EXT-03.

This script restores the FULL self._table_relationships (all 9 sections,
unconditional — exact pre-3.15 behavior) into for_schema_hint() by
overriding _table_relationships_topic_gated on the live db_schema
instance, then runs the full 105-question suite. Compare truth_match and
real_pass/sql_ok per-question against truth_fix_20260827.json:

  - if AGG-07/08/EXT-03 flip to truth-pass -> the 3.15 cut caused a
    real semantic regression the old gate couldn't see; revisit the cut.
  - if they still fail identically -> the divergence predates 3.15 (or is
    a GROUND_TRUTH spec issue), and the cut is exonerated.

No --profile-tokens (pure correctness ablation, same as ablation_3_14.py).

Usage:
  python ablation_tr_restore.py                 # full suite
  python ablation_tr_restore.py --category Aggregate
"""
import sys, os, types, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description="table_relationships RESTORE ablation")
    p.add_argument("--category", default=None, help="restrict to one eval category")
    p.add_argument("--output-dir", default="../..")
    a = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    from src.agent.orchestrator import AgentOrchestrator

    import importlib.util
    _eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
    _spec = importlib.util.spec_from_file_location("eval_script", _eval_path)
    _eval_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_eval_module)
    run_eval = _eval_module.run_eval

    orch = AgentOrchestrator.from_env()
    ds = orch.db_schema

    full = ds._table_relationships or ""
    gated_default = ds._table_relationships_sql_gen or ""
    assert len(full) > len(gated_default) > 0, (
        f"unexpected sizes: full={len(full)} sql_gen={len(gated_default)}"
    )

    # Exact pre-3.15 behavior: for_schema_hint() appends the full,
    # unconditional table_relationships regardless of question.
    def _restored(self, question=None):
        return self._table_relationships

    ds._table_relationships_topic_gated = types.MethodType(_restored, ds)

    # Sanity: show what for_schema_hint now emits for a representative case.
    sample = ds.for_schema_hint(["inspection_corrective_action", "inspection_report"],
                                question="how many mitigative actions were raised vs closed this quarter")
    print(f"[ablation] full table_relationships   = {len(full):>6} chars")
    print(f"[ablation] 3.16 gated default (unused) = {len(gated_default):>6} chars")
    print(f"[ablation] for_schema_hint() sample    = {len(sample):>6} chars "
          f"(contains 'HOW TABLES CONNECT' -> {'HOW TABLES CONNECT' in sample}, "
          f"'TABLE COLUMNS' -> {'TABLE COLUMNS' in sample})")

    out = os.path.join(a.output_dir, "profile_tr_restore_20260827.json")
    run_eval(model_pair_label="ABLATION tr-restore (full table_relationships) "
                              "| Qwen3.5-9B Q4_K_M + Qwen2.5-Coder-7B Q4_K_M",
             skip_multiturn=False, category_filter=a.category, output_file=out,
             profile_tokens=False, orch=orch)


if __name__ == "__main__":
    main()
