"""
Business Rules Registry
========================
All domain-specific rules for the inspection NL2SQL system live here as
structured data.  No business knowledge should be hardcoded in validator.py
or orchestrator.py — those files consume this registry.

Rule types
----------
  BLOCK        Hard validation error.  SQL is rejected and the error message
               is returned to the LLM for repair.
  WARN         Soft warning.  SQL executes but the warning is surfaced.
  AUTOCORRECT  The rule's `replacement` is applied automatically in clean_sql()
               before validation runs.  Logged but not surfaced to the user.
  INJECT       Question-level hint.  When `question_pattern` matches the user
               query, `hint` is appended to schema_hint before SQL generation.

Adding a new rule
-----------------
  1. Give it a unique `id`.
  2. Pick a type.
  3. Fill in the required fields for that type (see schema below).
  4. Add it to the matching file under rules/ (block_rules.py, warn_rules.py,
     autocorrect_rules.py, or inject_rules.py).
  5. Run:  python -c "from src.sql.rules import RulesEngine; RulesEngine()"
     A conflict check runs at import time — fix any warnings it reports.

Conflict detection
------------------
  At RulesEngine.__init__ the engine checks:
  - Duplicate rule IDs
  - BLOCK + AUTOCORRECT rules sharing the same sql_pattern (ambiguous precedence)
  - INJECT rules with identical question_pattern strings (redundant)

  Conflicts raise ValueError at startup so they are caught before the first query.
"""

from __future__ import annotations

import re

from src.sql.rules._types import BLOCK, WARN, AUTOCORRECT, INJECT


# ── Semantics registry (single source of truth for status/cast rules) ─────────
try:
    from src.knowledge.semantics_registry import (
        inject_hint_status as _inject_hint_status,
        inject_hint_financial as _inject_hint_financial,
        draft_exclusion_filter as _draft_exclusion_filter,
    )
    _STATUS_HINT = _inject_hint_status()
    _FINANCIAL_HINT = _inject_hint_financial()
    _DRAFT_EXCLUSION = _draft_exclusion_filter()
except ImportError:
    # Fallback — keep in sync with knowledge/semantics_registry.py
    _DRAFT_EXCLUSION = "status != 'DRAFT'"
    _STATUS_HINT = (
        "\nSTATUS GROUPINGS — always show as TWO categories, never raw status values:\n"
        "  conducted = status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')\n"
        "  completed = status = 'CLOSED'\n"
        "  NEVER: status IN ('CLOSED','SUBMITTED') — mixes conducted and completed\n"
        "  NEVER: status != 'DRAFT' alone for conducted — CLOSED is not conducted\n"
        "  NEVER expose raw status names (SUBMITTED/UNDER_REVIEW/etc) in results\n"
        "  UNLESS user explicitly asks to drill into a specific status\n\n"
        "DEFAULT PATTERN for total/how many inspections:\n"
        "SELECT\n"
        "    CASE WHEN ir.status = 'CLOSED' THEN 'completed' ELSE 'conducted' END AS category,\n"
        "    COUNT(*) AS inspection_count\n"
        "FROM inspection_report ir\n"
        "WHERE ir.active = true AND ir.status != 'DRAFT'\n"
        "GROUP BY category ORDER BY category DESC;\n\n"
        "If user asks ONLY for completed: WHERE ir.status = 'CLOSED'\n"
        "If user asks ONLY for conducted: WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')\n"
    )
    _FINANCIAL_HINT = (
        "\nFINANCIAL DATA ROUTING:\n"
        "  CAPEX Generated  = SUM(ica.capex)       — inspectee's original estimate on ICA\n"
        "  CAPEX Converted  = SUM(po.po_amount)    — actual PO raised (via po_details)\n"
        "  CAPEX Lost       = SUM(ica.capex) - COALESCE(SUM(po.po_amount), 0)\n"
        "  DO NOT use capex_status for loss calculation — use the arithmetic difference\n"
        "  OPEX Incurred    = SUM(pr.pr_amount)    — actual PR spend (via pr_details)\n"
        "po_details JOIN: po.mitigative_action_uuid = ica.id — one ICA can have MULTIPLE POs,\n"
        "  NEVER direct LEFT JOIN with SUM(ica.capex); use CTE pre-aggregation.\n"
        "pr_details JOIN: pr.mitigative_action_uuid = ica.id — always SUM(pr.pr_amount).\n"
        "CAPEX is tied to the FM Manager (inspectee_user_id); inspectors have no role in CAPEX/OPEX.\n"
        "DATE RULE: NEVER hardcode years. Use CURRENT_DATE or DATE_TRUNC('year', CURRENT_DATE).\n"
    )


# ── Registry ───────────────────────────────────────────────────────────────────
# Aggregated from the 4 per-type files under rules/ — see business_rules/
# package docstring (§8 of the restructure plan) for the proof that
# concatenation order here doesn't matter: RulesEngine._compile_and_validate
# below filters by type via list comprehension, which preserves each type's
# own within-type relative order regardless of how the 4 lists are joined.

from src.sql.rules.block_rules import BLOCK_RULES
from src.sql.rules.warn_rules import WARN_RULES
from src.sql.rules.autocorrect_rules import AUTOCORRECT_RULES
from src.sql.rules.inject_rules import INJECT_RULES

REGISTRY: list[dict] = BLOCK_RULES + WARN_RULES + AUTOCORRECT_RULES + INJECT_RULES


# ── RulesEngine ────────────────────────────────────────────────────────────────

class RulesEngine:
    """
    Loads and applies the business rules registry.

    Usage
    -----
    engine = RulesEngine()                            # validates registry at init
    hint    = engine.hint_for_question(question)       # INJECT rules
    ids     = engine.matching_inject_ids(question)
    tables  = engine.required_tables_for(rule_ids)
    forbidden = engine.forbidden_patterns_for(rule_ids)

    BLOCK/WARN/AUTOCORRECT rules are consumed by core/validator.py's
    SQLValidator (_registry_block_warn_issues / _registry_autocorrects),
    restricted to an explicit allowlist — not through this class directly.
    """

    def __init__(self, rules: list[dict] | None = None):
        self._rules = rules if rules is not None else REGISTRY
        self._compiled: list[dict] = []
        self._compile_and_validate()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _compile_and_validate(self) -> None:
        ids_seen: set[str] = set()
        sql_block_patterns:  dict[str, str] = {}   # pattern → id
        sql_autocorr_patterns: dict[str, str] = {} # pattern → id
        inject_q_patterns:   dict[str, str] = {}   # pattern → id
        conflicts: list[str] = []

        for rule in self._rules:
            # Required fields
            rid  = rule["id"]
            rtype = rule["type"]

            # Duplicate ID check
            if rid in ids_seen:
                conflicts.append(f"Duplicate rule ID: '{rid}'")
            ids_seen.add(rid)

            # Compile pattern
            compiled = dict(rule)
            if "sql_pattern" in rule:
                compiled["_sql_re"] = re.compile(rule["sql_pattern"], re.IGNORECASE | re.DOTALL)
            if "question_pattern" in rule:
                compiled["_q_re"] = re.compile(rule["question_pattern"], re.IGNORECASE)

            # Conflict: same sql_pattern in BLOCK and AUTOCORRECT
            if rtype == BLOCK and "sql_pattern" in rule:
                p = rule["sql_pattern"]
                if p in sql_autocorr_patterns:
                    conflicts.append(
                        f"Conflict: BLOCK rule '{rid}' and AUTOCORRECT rule "
                        f"'{sql_autocorr_patterns[p]}' share sql_pattern '{p[:40]}...'"
                    )
                sql_block_patterns[p] = rid

            if rtype == AUTOCORRECT and "sql_pattern" in rule:
                p = rule["sql_pattern"]
                if p in sql_block_patterns:
                    conflicts.append(
                        f"Conflict: AUTOCORRECT rule '{rid}' and BLOCK rule "
                        f"'{sql_block_patterns[p]}' share sql_pattern '{p[:40]}...'"
                    )
                sql_autocorr_patterns[p] = rid

            # Conflict: identical INJECT question_patterns
            if rtype == INJECT and "question_pattern" in rule:
                p = rule["question_pattern"]
                if p in inject_q_patterns:
                    conflicts.append(
                        f"Conflict: INJECT rules '{rid}' and '{inject_q_patterns[p]}' "
                        f"share identical question_pattern"
                    )
                inject_q_patterns[p] = rid

            self._compiled.append(compiled)

        if conflicts:
            raise ValueError(
                "Business rules registry conflicts detected:\n  " +
                "\n  ".join(conflicts)
            )

        # Sort INJECT rules by priority descending so highest-priority hints inject first
        self._inject_rules = sorted(
            [r for r in self._compiled if r["type"] == INJECT],
            key=lambda r: r.get("priority", 0),
            reverse=True,
        )
        self._block_rules  = [r for r in self._compiled if r["type"] == BLOCK]
        self._warn_rules   = [r for r in self._compiled if r["type"] == WARN]
        self._autocorr_rules = [r for r in self._compiled if r["type"] == AUTOCORRECT]

    # ── Question-level hint injection ─────────────────────────────────────────

    def hint_for_question(self, question: str) -> str:
        """
        Return combined schema_hint for all INJECT rules matching the question.
        Rules fire in priority order (highest first) and hints are concatenated.

        If a matching rule declares ``"suppresses": ["rule_id", ...]``, those
        lower-priority rules are silenced even if their own pattern matches.
        This prevents contradictory hints from being concatenated.
        """
        suppressed: set[str] = set()
        parts: list[str] = []
        for rule in self._inject_rules:
            rid = rule["id"]
            if rid in suppressed:
                continue
            if rule["_q_re"].search(question):
                parts.append(rule["hint"])
                for sid in rule.get("suppresses", []):
                    suppressed.add(sid)
        return "".join(parts)

    def matching_inject_ids(self, question: str) -> list[str]:
        """Return IDs of all INJECT rules that match the question (for debugging)."""
        return [r["id"] for r in self._inject_rules if r["_q_re"].search(question)]

    # ── Introspection ─────────────────────────────────────────────────────────

    def rules_by_type(self, rule_type: str) -> list[dict]:
        return [r for r in self._rules if r["type"] == rule_type]

    def required_tables_for(self, rule_ids: list) -> list:
        """Return all required_tables for a set of fired INJECT rule IDs."""
        tables = []
        for rule in self._rules:
            if rule.get("id") in rule_ids and rule.get("type") == INJECT:
                tables.extend(rule.get("required_tables", []))
        return list(set(tables))

    def forbidden_patterns_for(self, rule_ids: list) -> list:
        """Return all forbidden_patterns for a set of fired INJECT rule IDs."""
        patterns = []
        for rule in self._rules:
            if rule.get("id") in rule_ids and rule.get("type") == INJECT:
                patterns.extend(rule.get("forbidden_patterns", []))
        return patterns

    def summary(self) -> str:
        counts = {BLOCK: 0, WARN: 0, AUTOCORRECT: 0, INJECT: 0}
        for r in self._rules:
            counts[r["type"]] += 1
        return (
            f"RulesEngine: {len(self._rules)} rules — "
            + ", ".join(f"{v} {k}" for k, v in counts.items())
        )


# ── Singleton for import convenience ─────────────────────────────────────────
# Validated at import time — if the registry has conflicts this raises
# immediately rather than at first query.

_default_engine: RulesEngine | None = None


def get_engine() -> RulesEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = RulesEngine()
    return _default_engine


if __name__ == "__main__":
    engine = RulesEngine()
    print(engine.summary())
    print("\nBLOCK rules:")
    for r in engine.rules_by_type(BLOCK):
        print(f"  {r['id']}: {r['description']}")
    print("\nINJECT rules (by priority):")
    for r in engine._inject_rules:
        print(f"  [{r.get('priority',0)}] {r['id']}: {r['description']}")
