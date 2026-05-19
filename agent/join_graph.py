"""
join_graph.py
=============
Deterministic FK join path resolution for the OptimaX inspection schema.

Replaces LLM join guessing with code knowledge for known table relationships.
Startup validation checks every edge against the live DB — schema changes are
caught immediately rather than silently breaking queries.

Usage:
    from agent.join_graph import JOIN_GRAPH, find_join_path, validate_join_graph

    path = find_join_path("inspection_cycle", "frequency_definition")
    # Returns:
    # [
    #   ("inspection_cycle",           "ic",   None),
    #   ("inspector_portfolio",        "ip",   "ip.cycle_id = ic.id"),
    #   ("inspector_portfolio_details","ipd",  "ipd.portfolio_id = ip.id"),
    #   ("frequency_definition",       "fd",   "fd.id = ipd.frequency_definition_id"),
    # ]
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# JOIN_GRAPH
# ---------------------------------------------------------------------------
# Format per edge:
#   source_table → {
#       target_table: (target_alias, join_condition, [intermediate_tables])
#   }
#
# join_condition uses the aliases defined in the path, not raw column names.
# intermediate_tables lists any tables that must be traversed (multi-hop).
#
# Single-hop edges only — multi-hop paths are built by path-finding.

JOIN_GRAPH: dict[str, dict[str, tuple]] = {

    # ── inspection_report (hub table) ─────────────────────────────────────
    "inspection_report": {
        "facility":              ("fac",  "fac.id = ir.facility_id"),
        "inspection_type":       ("it",   "it.id = ir.inspection_type_id"),
        "inspection_sub_type":   ("ist",  "ist.id = ir.inspection_sub_type_id"),
        "project":               ("proj", "proj.id = ir.project_id"),
        "client":                ("cl",   "cl.id = ir.client_id"),
        "inspection_cycle":      ("ic",   "ic.id = ir.cycle_id"),
        "inspection_schedule":   ("isched","isched.id = ir.schedule_id"),
        # Two user roles — different aliases to allow both in same query
        "users_inspector":       ("u",    "u.id = ir.inspector_user_id"),
        "users_inspectee":       ("uinsp","uinsp.id = ir.inspectee_user_id"),
        "inspection_corrective_action": (
            "ica", "ica.inspection_id = ir.id"
        ),
    },

    # ── inspection_corrective_action ──────────────────────────────────────
    "inspection_corrective_action": {
        "inspection_report":     ("ir",   "ir.id = ica.inspection_id"),
        "risk_level":            ("rl",   "rl.id = ica.risk_level_id"),
        "impact":                ("imp",  "imp.id = ica.impact_id"),
        "po_details":            ("po",   "po.mitigative_action_uuid = ica.id"),
        "pr_details":            ("pr",   "pr.mitigative_action_uuid = ica.id"),
    },
    "po_details": {
        "inspection_corrective_action": ("ica", "ica.id = po.mitigative_action_uuid"),
    },
    "pr_details": {
        "inspection_corrective_action": ("ica", "ica.id = pr.mitigative_action_uuid"),
    },

    # ── inspection_cycle → frequency path ─────────────────────────────────
    "inspection_cycle": {
        "inspection_report":     ("ir",   "ir.cycle_id = ic.id"),
        "inspector_portfolio":   ("ip",   "ip.cycle_id = ic.id"),
    },

    "inspector_portfolio": {
        "inspection_cycle":          ("ic",  "ic.id = ip.cycle_id"),
        "inspector_portfolio_details":("ipd","ipd.portfolio_id = ip.id"),
        "users_inspector":           ("u",   "u.id = ip.user_id"),
    },

    "inspector_portfolio_details": {
        "inspector_portfolio":    ("ip",  "ip.id = ipd.portfolio_id"),
        "frequency_definition":   ("fd",  "fd.id = ipd.frequency_definition_id"),
        "facility":               ("fac", "fac.id = ipd.facility_id"),
    },

    # ── inspection_schedule ───────────────────────────────────────────────
    "inspection_schedule": {
        "inspection_report":          ("ir",  "ir.schedule_id = isched.id"),
        "facility":                   ("fac", "fac.id = isched.facility_id"),
        "users_inspector":            ("u",   "u.id = isched.inspector_id"),
        "inspector_portfolio_details":("ipd", "ipd.id = isched.portfolio_details_id"),
        "inspection_cycle":           ("ic",  "ic.id = isched.inspection_cycle_id"),
    },

    # ── ai tables ─────────────────────────────────────────────────────────
    "ai_answers": {
        "inspection_report": ("ir",  "ir.id = aa.inspection_report_id"),
        "ai_questions":      ("aq",  "aq.element_id = aa.element_id"),
    },
    "ai_questions": {
        "ai_answers":        ("aa",  "aa.element_id = aq.element_id"),
    },

    # ── lookup tables (typically leaf nodes) ──────────────────────────────
    "facility":            {"inspection_report": ("ir",  "ir.facility_id = fac.id")},
    "inspection_type":     {"inspection_report": ("ir",  "ir.inspection_type_id = it.id")},
    "frequency_definition":{"inspector_portfolio_details": ("ipd","ipd.frequency_definition_id = fd.id")},
    "risk_level":          {"inspection_corrective_action": ("ica","ica.risk_level_id = rl.id")},
    "users_inspector":     {"inspection_report": ("ir",  "ir.inspector_user_id = u.id")},
    "users_inspectee":     {"inspection_report": ("ir",  "ir.inspectee_user_id = uinsp.id")},
}


# ---------------------------------------------------------------------------
# Canonical aliases — used when rendering JOIN clauses
# ---------------------------------------------------------------------------
CANONICAL_ALIAS: dict[str, str] = {
    "inspection_report":            "ir",
    "inspection_corrective_action": "ica",
    "inspection_cycle":             "ic",
    "inspector_portfolio":          "ip",
    "inspector_portfolio_details":  "ipd",
    "frequency_definition":         "fd",
    "inspection_schedule":          "isched",
    "facility":                     "fac",
    "inspection_type":              "it",
    "inspection_sub_type":          "ist",
    "project":                      "proj",
    "client":                       "cl",
    "risk_level":                   "rl",
    "impact":                       "imp",
    "ai_answers":                   "aa",
    "ai_questions":                 "aq",
    "users_inspector":              "u",
    "users_inspectee":              "uinsp",
    "inspector_portfolio":          "ip",
}


# ---------------------------------------------------------------------------
# Path-finding (BFS)
# ---------------------------------------------------------------------------

def find_join_path(
    source: str,
    target: str,
    max_hops: int = 5,
) -> Optional[list[tuple[str, str, Optional[str]]]]:
    """
    Find shortest join path between two tables using BFS over JOIN_GRAPH.

    Returns list of (table_name, alias, join_condition) tuples where:
    - First entry has join_condition=None (it's the FROM table)
    - Subsequent entries have join_condition for the JOIN ON clause

    Returns None if no path found within max_hops.

    Example:
        find_join_path("inspection_cycle", "frequency_definition")
        → [
            ("inspection_cycle",            "ic",  None),
            ("inspector_portfolio",         "ip",  "ip.cycle_id = ic.id"),
            ("inspector_portfolio_details", "ipd", "ipd.portfolio_id = ip.id"),
            ("frequency_definition",        "fd",  "fd.id = ipd.frequency_definition_id"),
          ]
    """
    if source == target:
        alias = CANONICAL_ALIAS.get(source, source[:3])
        return [(source, alias, None)]

    # BFS
    from collections import deque
    queue = deque()
    queue.append((source, [(source, CANONICAL_ALIAS.get(source, source[:3]), None)]))
    visited = {source}

    while queue:
        current, path = queue.popleft()
        if len(path) > max_hops:
            continue

        neighbors = JOIN_GRAPH.get(current, {})
        for neighbor, (alias, condition) in neighbors.items():
            if neighbor in visited:
                continue
            new_path = path + [(neighbor, alias, condition)]
            if neighbor == target:
                return new_path
            visited.add(neighbor)
            queue.append((neighbor, new_path))

    return None


def render_join_path(path: list[tuple[str, str, Optional[str]]]) -> str:
    """
    Render a join path as a SQL fragment.

    Example output:
        FROM inspection_cycle ic
        JOIN inspector_portfolio ip ON ip.cycle_id = ic.id
        JOIN inspector_portfolio_details ipd ON ipd.portfolio_id = ip.id
        JOIN frequency_definition fd ON fd.id = ipd.frequency_definition_id
    """
    if not path:
        return ""
    lines = []
    for i, (table, alias, condition) in enumerate(path):
        # Strip _inspector/_inspectee suffix for actual table name
        actual_table = table.replace("_inspector", "").replace("_inspectee", "")
        if i == 0:
            lines.append(f"FROM {actual_table} {alias}")
        else:
            lines.append(f"JOIN {actual_table} {alias} ON {condition}")
    return "\n".join(lines)


def path_as_hint(source: str, target: str) -> Optional[str]:
    """
    Return a compact join path hint string for injection into schema_hint.
    Returns None if path not found.

    Example:
        "JOIN PATH (ic→ip→ipd→fd):
          FROM inspection_cycle ic
          JOIN inspector_portfolio ip ON ip.cycle_id = ic.id
          JOIN inspector_portfolio_details ipd ON ipd.portfolio_id = ip.id
          JOIN frequency_definition fd ON fd.id = ipd.frequency_definition_id"
    """
    path = find_join_path(source, target)
    if not path:
        return None
    arrows = "→".join(p[1] for p in path)
    rendered = render_join_path(path)
    return f"\nJOIN PATH ({arrows}):\n{rendered}\n"


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_join_graph(engine) -> list[str]:
    """
    Validate every edge in JOIN_GRAPH against the live DB.
    Returns list of error strings (empty = all OK).

    Checks:
    1. Source table exists in DB
    2. Target table exists in DB (stripping _inspector/_inspectee suffix)
    3. FK columns referenced in join conditions exist on their tables

    Call this at orchestrator startup — fail loud on schema drift.
    """
    from sqlalchemy import inspect as sa_inspect, text as sa_text

    errors = []
    insp = sa_inspect(engine)
    existing_tables = set(insp.get_table_names())

    def real_table(name: str) -> str:
        return name.replace("_inspector", "").replace("_inspectee", "")

    def get_columns(table: str) -> set[str]:
        try:
            return {c["name"] for c in insp.get_columns(table)}
        except Exception:
            return set()

    for source, edges in JOIN_GRAPH.items():
        src_real = real_table(source)
        if src_real not in existing_tables:
            errors.append(f"JOIN_GRAPH: source table '{src_real}' not found in DB")
            continue

        for target, (alias, condition) in edges.items():
            tgt_real = real_table(target)
            if tgt_real not in existing_tables:
                errors.append(
                    f"JOIN_GRAPH: target table '{tgt_real}' (from {source}) not found in DB"
                )
                continue

            # Check columns referenced in join condition
            # Extract col refs like "ip.cycle_id", "fac.id", etc.
            import re
            col_refs = re.findall(r'\b(\w+)\.(\w+)\b', condition)
            src_cols = get_columns(src_real)
            tgt_cols = get_columns(tgt_real)

            for tbl_alias, col in col_refs:
                # Try to match alias to real table
                if tbl_alias == alias:
                    check_table, check_cols = tgt_real, tgt_cols
                else:
                    check_table, check_cols = src_real, src_cols
                if check_cols and col not in check_cols:
                    errors.append(
                        f"JOIN_GRAPH: column '{tbl_alias}.{col}' in condition "
                        f"'{condition}' not found on table '{check_table}'"
                    )

    return errors