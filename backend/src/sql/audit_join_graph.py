"""
audit_join_graph.py
====================
Deterministic FK join path resolution for the audit domain — the audit
equivalent of sql/join_graph.py. Kept as a separate module (mirroring the
inspection domain rather than parameterizing a shared one) so each domain's
graph is independently reviewable and neither can accidentally corrupt the
other's join paths.

Every edge below was confirmed live against the DB — see
docs/Audit_Domain_Schema_Reference.docx and the domain-router plan's §1b.

Usage:
    from src.sql.audit_join_graph import AUDIT_JOIN_GRAPH, find_join_path, path_as_hint

    path = find_join_path("audit_corrective_action", "facility")
    # audit_corrective_action → audit_report → audit_schedule → facility
    # (audit_report has NO facility_id of its own — confirmed live, see rules)
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# AUDIT_JOIN_GRAPH
# ---------------------------------------------------------------------------
# Same format as sql/join_graph.py's JOIN_GRAPH:
#   source_table → { target_table: (target_alias, join_condition) }
# Single-hop edges only — multi-hop paths built by path-finding (find_join_path).

AUDIT_JOIN_GRAPH: dict[str, dict[str, tuple]] = {

    # ── audit_plan ─────────────────────────────────────────────────────────
    "audit_plan": {
        "audit_cycle":       ("acyc", "acyc.id = ap.cycle_id"),
        "audit_type":        ("aty",  "aty.id = ap.audit_type_id"),
        "audit_category":    ("acat", "acat.id = ap.category_id"),
        "users_proposer":    ("u",    "u.id = ap.user_id"),
        "audit_schedule":    ("asch", "asch.audit_plan_id = ap.id"),
    },

    # ── audit_schedule (hub for facility/project/auditor/auditee) ─────────
    # NOTE: asch.audit_plan_id is NOT guaranteed populated — confirmed live,
    # unplanned audits (asch.is_unplanned = true) are created directly by a
    # super-admin-permission user and never go through the audit_plan chain
    # at all. A query that needs unplanned audits alongside planned ones must
    # LEFT JOIN audit_plan here, not INNER JOIN — an INNER JOIN silently drops
    # every unplanned row with no error/row-count signal (same failure class
    # as the audit_corrective_action multi-axis INNER JOIN trap elsewhere in
    # this codebase). See schema_introspector.py's AUDIT_BUSINESS_RULES for
    # the full "unplanned audits are a fast-track" writeup.
    "audit_schedule": {
        "audit_plan":             ("ap",     "ap.id = asch.audit_plan_id"),
        "facility":                ("fac",    "fac.id = asch.facility_id"),
        "project":                 ("proj",   "proj.id = asch.project_id"),
        "audit_universe":          ("auniv",  "auniv.id = asch.audit_universe_id"),
        "audit_sub_category":      ("asub",   "asub.id = asch.audit_sub_category_id"),
        "users_auditor":           ("u_aud",  "u_aud.id = asch.auditor_id"),
        "users_auditee":           ("u_ee",   "u_ee.id = asch.auditee_id"),
        "audit_report":            ("arep",   "arep.schedule_id = asch.id"),
    },

    # ── audit_report ────────────────────────────────────────────────────────
    "audit_report": {
        "audit_schedule":    ("asch",  "asch.id = arep.schedule_id"),
        "audit_cycle":       ("acyc",  "acyc.id = arep.audit_cycle_id"),
        "audit_forms":       ("aform", "aform.id = arep.audit_form_used_id"),
        "client":            ("cl",    "cl.id = arep.client_id"),
        "entity":            ("ent",   "ent.id = arep.entity_id"),
        "audit_corrective_action": ("aca", "aca.audit_id = arep.id"),
        "ai_audit_answers": ("aaa", "aaa.audit_report_id = arep.id"),
    },

    # ── audit_corrective_action ─────────────────────────────────────────────
    "audit_corrective_action": {
        "audit_report":  ("arep", "arep.id = aca.audit_id"),
        "risk_level":    ("rl",   "rl.id = aca.risk_level_id"),
        "impact":        ("imp",  "imp.id = aca.impact_id"),
        "compliance":    ("comp", "comp.id = aca.compliance_id"),
        "users_internal_stakeholder": ("u_stk", "u_stk.id = aca.internal_stakeholder_id"),
        # NOTE: the observation->CA link (aaa.submission_id = aca.observation_id::varchar)
        # is deliberately NOT an edge here — same reasoning as inspection's join_graph.py,
        # which also excludes ica.observation_id/ai_answers from BFS. It's a CAST join
        # with two easily-confused directions (specific observation vs whole-report), so
        # it's taught via AUDIT_BUSINESS_RULES + AUDIT_SEED_EXAMPLES + the validator's
        # IMPOSSIBLE_JOIN_PATTERNS cast-direction check instead of blind pathfinding.
    },

    # ── ai flat tables (form questions/answers) — report-hop only; see the
    # observation->CA note above for why the cast-based link isn't an edge here ──
    "ai_audit_answers": {
        "audit_report": ("arep", "arep.id = aaa.audit_report_id"),
        "ai_questions":  ("aq",   "aq.element_id = aaa.element_id"),
    },
    "ai_questions": {
        "ai_audit_answers": ("aaa", "aaa.element_id = aq.element_id"),
    },

    # ── auditor_portfolio ────────────────────────────────────────────────────
    "auditor_portfolio": {
        "users_auditor":            ("u_aud", "u_aud.id = po.user_id"),
        "audit_cycle":              ("acyc",  "acyc.id = po.audit_cycle_id"),
        "auditor_portfolio_detail": ("pod",   "pod.portfolio_id = po.id"),
    },
    "auditor_portfolio_detail": {
        "auditor_portfolio": ("po",  "po.id = pod.portfolio_id"),
        "facility":          ("fac", "fac.id = pod.facility_id"),
        "project":           ("proj","proj.id = pod.project_id"),
        "portfolio_detail_audit_category":    ("pdac", "pdac.detail_id = pod.id"),
        "portfolio_detail_audit_subcategory": ("pdas", "pdas.detail_id = pod.id"),
        "portfolio_detail_audit_type":        ("pdat", "pdat.detail_id = pod.id"),
    },
    "portfolio_detail_audit_category": {
        "auditor_portfolio_detail": ("pod",  "pod.id = pdac.detail_id"),
        "audit_category":            ("acat", "acat.id = pdac.audit_category_id"),
    },
    "portfolio_detail_audit_subcategory": {
        "auditor_portfolio_detail": ("pod",  "pod.id = pdas.detail_id"),
        # NOTE: column is audit_subcategory_id (no underscore before "category"),
        # unlike audit_sub_category the TABLE name — confirmed live, a real trap.
        "audit_sub_category":        ("asub", "asub.id = pdas.audit_subcategory_id"),
    },
    "portfolio_detail_audit_type": {
        "auditor_portfolio_detail": ("pod", "pod.id = pdat.detail_id"),
        "audit_type":                 ("aty", "aty.id = pdat.audit_type_id"),
    },

    # ── audit_cycle ──────────────────────────────────────────────────────────
    "audit_cycle": {
        "audit_plan":        ("ap",   "ap.cycle_id = acyc.id"),
        "audit_report":      ("arep", "arep.audit_cycle_id = acyc.id"),
        "auditor_portfolio": ("po",   "po.audit_cycle_id = acyc.id"),
    },

    # ── audit_forms (template library — audit_schedule_id always NULL) ────
    # NOTE: audit_type intentionally NOT an edge here even though
    # audit_forms.audit_type_id is a real FK — that gives the form
    # TEMPLATE's designed type, not necessarily the report's actual
    # assigned type. The correct "audit type of a report" path is via
    # audit_plan (see audit_report's docstring trap and AUDIT_BUSINESS_RULES)
    # — omitting this edge stops BFS path-finding from silently offering
    # the shorter-but-wrong shortcut.
    "audit_forms": {
        "audit_report":    ("arep", "arep.audit_form_used_id = aform.id"),
        "audit_universe":   ("auniv","auniv.id = aform.audit_universe_id"),
        "client":            ("cl",   "cl.id = aform.client_id"),
        "entity":            ("ent",  "ent.id = aform.entity_id"),
    },

    # ── review (polymorphic — entity_id has no real FK, see rules) ────────
    # Not represented as graph edges since entity_id's target varies by
    # entity_type — handled via explicit seed examples instead (one per
    # entity_type), not BFS path-finding. See AUDIT_SEED_EXAMPLES.

    # ── lookup tables (leaf nodes) ─────────────────────────────────────────
    "audit_type":         {"audit_plan": ("ap", "ap.audit_type_id = aty.id")},
    "audit_category":     {"audit_plan": ("ap", "ap.category_id = acat.id")},
    "audit_sub_category":  {"audit_schedule": ("asch", "asch.audit_sub_category_id = asub.id")},
    "audit_universe":      {"audit_schedule": ("asch", "asch.audit_universe_id = auniv.id")},
    "facility": {
        "audit_schedule":            ("asch", "asch.facility_id = fac.id"),
        "auditor_portfolio_detail":  ("pod",  "pod.facility_id = fac.id"),
    },
    "project": {
        "audit_schedule":            ("asch", "asch.project_id = proj.id"),
        "auditor_portfolio_detail":  ("pod",  "pod.project_id = proj.id"),
    },
    "client":  {"audit_report": ("arep", "arep.client_id = cl.id")},
    "entity":  {"audit_report": ("arep", "arep.entity_id = ent.id")},
    "risk_level":  {"audit_corrective_action": ("aca", "aca.risk_level_id = rl.id")},
    "impact":      {"audit_corrective_action": ("aca", "aca.impact_id = imp.id")},
    "compliance":  {"audit_corrective_action": ("aca", "aca.compliance_id = comp.id")},
    "users_auditor": {"audit_schedule": ("asch", "asch.auditor_id = u_aud.id")},
    "users_auditee": {"audit_schedule": ("asch", "asch.auditee_id = u_ee.id")},
    "users_proposer": {"audit_plan": ("ap", "ap.user_id = u.id")},
}


# ---------------------------------------------------------------------------
# Canonical aliases
# ---------------------------------------------------------------------------
AUDIT_CANONICAL_ALIAS: dict[str, str] = {
    "audit_plan":                          "ap",
    "audit_schedule":                      "asch",
    "audit_report":                        "arep",
    "audit_corrective_action":             "aca",
    "auditor_portfolio":                   "po",
    "auditor_portfolio_detail":            "pod",
    "portfolio_detail_audit_category":     "pdac",
    "portfolio_detail_audit_subcategory":  "pdas",
    "portfolio_detail_audit_type":         "pdat",
    "audit_cycle":                         "acyc",
    "audit_forms":                         "aform",
    "audit_type":                          "aty",
    "audit_category":                      "acat",
    "audit_sub_category":                  "asub",
    "audit_universe":                      "auniv",
    "facility":                            "fac",
    "project":                             "proj",
    "client":                              "cl",
    "entity":                              "ent",
    "risk_level":                          "rl",
    "impact":                              "imp",
    "compliance":                          "comp",
    "users_auditor":                       "u_aud",
    "users_auditee":                       "u_ee",
    "users_proposer":                      "u",
    "users_internal_stakeholder":          "u_stk",
    "review":                              "rev",
    "ai_audit_answers":                    "aaa",
    "ai_questions":                        "aq",
}


# ---------------------------------------------------------------------------
# Path-finding (BFS) — identical algorithm to sql/join_graph.py, operating
# over AUDIT_JOIN_GRAPH/AUDIT_CANONICAL_ALIAS instead. Duplicated rather than
# parameterized to avoid any shared mutable state between domains.
# ---------------------------------------------------------------------------

def find_join_path(
    source: str,
    target: str,
    max_hops: int = 5,
) -> Optional[list[tuple[str, str, Optional[str]]]]:
    if source == target:
        alias = AUDIT_CANONICAL_ALIAS.get(source, source[:3])
        return [(source, alias, None)]

    from collections import deque
    queue = deque()
    queue.append((source, [(source, AUDIT_CANONICAL_ALIAS.get(source, source[:3]), None)]))
    visited = {source}

    while queue:
        current, path = queue.popleft()
        if len(path) > max_hops:
            continue

        neighbors = AUDIT_JOIN_GRAPH.get(current, {})
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
    if not path:
        return ""
    lines = []
    for i, (table, alias, condition) in enumerate(path):
        actual_table = table.replace("_auditor", "").replace("_auditee", "").replace("_proposer", "").replace("_internal_stakeholder", "")
        if actual_table.startswith("users"):
            actual_table = "users"
        if i == 0:
            lines.append(f"FROM {actual_table} {alias}")
        else:
            lines.append(f"JOIN {actual_table} {alias} ON {condition}")
    return "\n".join(lines)


def path_as_hint(source: str, target: str) -> Optional[str]:
    path = find_join_path(source, target)
    if not path:
        return None
    arrows = "→".join(p[1] for p in path)
    rendered = render_join_path(path)
    return f"\nJOIN PATH ({arrows}):\n{rendered}\n"


# ---------------------------------------------------------------------------
# Startup validation — same shape as sql/join_graph.py's validate_join_graph
# ---------------------------------------------------------------------------

def validate_join_graph(engine) -> list[str]:
    """Validate every edge in AUDIT_JOIN_GRAPH against the live DB. Returns
    list of error strings (empty = all OK). Call at audit-orchestrator
    startup — fail loud on schema drift, matching the inspection domain."""
    from sqlalchemy import inspect as sa_inspect
    import re as _re

    errors = []
    insp = sa_inspect(engine)
    existing_tables = set(insp.get_table_names())

    def real_table(name: str) -> str:
        for suffix in ("_auditor", "_auditee", "_proposer", "_internal_stakeholder"):
            name = name.replace(suffix, "")
        return "users" if name == "users" else name

    def get_columns(table: str) -> set[str]:
        try:
            return {c["name"] for c in insp.get_columns(table)}
        except Exception:
            return set()

    for source, edges in AUDIT_JOIN_GRAPH.items():
        src_real = real_table(source)
        if src_real not in existing_tables:
            errors.append(f"AUDIT_JOIN_GRAPH: source table '{src_real}' not found in DB")
            continue

        for target, (alias, condition) in edges.items():
            tgt_real = real_table(target)
            if tgt_real not in existing_tables:
                errors.append(
                    f"AUDIT_JOIN_GRAPH: target table '{tgt_real}' (from {source}) not found in DB"
                )
                continue

            col_refs = _re.findall(r'\b(\w+)\.(\w+)\b', condition)
            src_cols = get_columns(src_real)
            tgt_cols = get_columns(tgt_real)

            for tbl_alias, col in col_refs:
                if tbl_alias == alias:
                    check_table, check_cols = tgt_real, tgt_cols
                else:
                    check_table, check_cols = src_real, src_cols
                if check_cols and col not in check_cols:
                    errors.append(
                        f"AUDIT_JOIN_GRAPH: column '{tbl_alias}.{col}' in condition "
                        f"'{condition}' not found on table '{check_table}'"
                    )

    return errors
