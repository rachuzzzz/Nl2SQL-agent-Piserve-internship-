"""
Phase 3.14 — causal ablation of table_relationships sections.

3.11/3.12 identified table_relationships (~1922 tok, ~58% of every
"targeted" schema_hint call) as the last unaddressed chunk of the fixed
overhead 3.13 partially cut. 3.12's word-match relevance signal for this
prose content was judged too blunt to act on (a table name appearing
anywhere in a paragraph isn't evidence the paragraph was actually needed).
This script answers the sharper question directly instead of via another
textual proxy: for each of the 9 sections table_relationships splits into
(see schema_introspector._split_relationship_sections), does REMOVING it
from the prompt change real_pass/sql_ok across the full 103-question
suite? A section whose removal causes zero measured correctness drop is
a genuine cut candidate; a section whose removal breaks questions is
load-bearing regardless of how rarely its table names get mentioned.

No --profile-tokens here — this is a pure correctness ablation (already
covered by 3.9-3.13's token-composition work), and skipping the extra
/tokenize round-trips keeps each run faster and keeps timing measurement
uncontaminated for once.

Usage:
  python ablation_3_14.py --list                 # print section indices/titles
  python ablation_3_14.py --variant baseline      # unmodified table_relationships
  python ablation_3_14.py --variant 3             # excludes section index 3
"""
import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _slug(title: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in title.lower())[:40].strip("_")


def main():
    p = argparse.ArgumentParser(description="Phase 3.14 table_relationships section ablation")
    p.add_argument("--variant", default=None,
                   help="'baseline', 'all' (removes every section), or a 0-based section index to exclude")
    p.add_argument("--list", action="store_true", help="print section indices/titles and exit")
    p.add_argument("--output-dir", default="../..", help="where to write the result JSON (default: repo root)")
    p.add_argument("--category", default=None, help="restrict to one eval category (for smoke-testing)")
    a = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    from src.knowledge.schema_introspector import _split_relationship_sections
    from src.agent.orchestrator import AgentOrchestrator
    # backend/eval/__init__.py makes this a package, so `import eval` here
    # would resolve to the package (shadowing the sibling eval.py module) —
    # load eval.py directly by file path instead.
    import importlib.util
    _eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
    _spec = importlib.util.spec_from_file_location("eval_script", _eval_path)
    _eval_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_eval_module)
    run_eval = _eval_module.run_eval

    orch = AgentOrchestrator.from_env()
    # Phase 3.15: for_schema_hint() now renders _table_relationships_sql_gen
    # (COMMUNICATION GROUPS/USER ROLES/APPROVAL WORKFLOW only — the 6 other
    # sections were cut into production after this script's original 9-way
    # ablation showed zero regression on the pre-3.15 103-question suite).
    # Ablating against the OLD _table_relationships attribute here would be
    # a no-op now that for_schema_hint() no longer reads it.
    sections = _split_relationship_sections(orch.db_schema._table_relationships_sql_gen)

    if a.list:
        for i, (title, body) in enumerate(sections):
            print(f"{i}: {title!r}  ({len(body)} body chars)")
        return

    if a.variant is None:
        print("--variant required ('baseline' or a 0-based section index) — see --list")
        sys.exit(1)

    if a.variant == "baseline":
        label = "baseline"
        print("Running baseline (table_relationships unmodified).")
    elif a.variant == "all":
        # Every individual section showed zero measured regression — this
        # checks for a combinatorial/additive effect none of the 9 solo
        # ablations could reveal (e.g. two sections being mutually
        # redundant with each other but not with anything else).
        label = "all"
        orch.db_schema._table_relationships_sql_gen = ""
        print(f"Ablating ALL {len(sections)} sections (table_relationships emptied entirely).")
    elif "," in a.variant:
        # Combined ablation — validate a specific multi-section cut (e.g. the
        # "well-tested six") in one shot, against the CURRENT eval suite,
        # rather than trusting old single-section ablation results run
        # against a since-changed question set.
        idxs = sorted(int(x) for x in a.variant.split(","))
        label = "combo_" + "_".join(str(i) for i in idxs)
        removed_titles = [sections[i][0] for i in idxs]
        kept = [(t, b) for i, (t, b) in enumerate(sections) if i not in idxs]
        orch.db_schema._table_relationships_sql_gen = "\n".join(f"{t}\n{b}" for t, b in kept)
        print(f"Ablating {len(idxs)} sections {idxs}: {removed_titles}")
    else:
        idx = int(a.variant)
        title, body = sections[idx]
        label = f"{idx}_{_slug(title)}"
        # Rejoin every OTHER section unchanged — not a byte-exact
        # reconstruction of the original (confirmed acceptable: multi-line
        # headers collapse to one space-joined title line, verified earlier
        # to cover every source line with none dropped or duplicated), just
        # a fair "with this section's content vs. without it" comparison.
        kept = [(t, b) for i, (t, b) in enumerate(sections) if i != idx]
        orch.db_schema._table_relationships_sql_gen = "\n".join(f"{t}\n{b}" for t, b in kept)
        print(f"Ablating section {idx}: {title!r} ({len(body)} body chars removed).")

    out = os.path.join(a.output_dir, f"profile_3_14_ablate_{label}.json")
    run_eval(skip_multiturn=False, category_filter=a.category, output_file=out,
             profile_tokens=False, orch=orch)


if __name__ == "__main__":
    main()
