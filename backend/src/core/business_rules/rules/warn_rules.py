"""
WARN_RULES: extracted verbatim from the original
core/business_rules.py REGISTRY list (all "type": WARN entries,
in original relative order). See core.business_rules.engine for how
these are aggregated and consumed.
"""

from src.core.business_rules._types import WARN

WARN_RULES: list[dict] = [
    {
        "id":          "block_hardcoded_year",
        "type":        WARN,
        "code":        "HARDCODED_YEAR",
        "description": (
            "Hardcoded year — consider EXTRACT(YEAR FROM CURRENT_DATE) for current-year queries. "
            "SOFT_WARN, not BLOCK: intentional historical-year queries (e.g. 'inspections in 2024') "
            "are legitimate and must not be hard-rejected — matches core/validator.py's existing, "
            "deliberate demotion of this same check (see validator.py's HARDCODED_YEAR comment)."
        ),
        "sql_pattern": r"\bextract\s*\(\s*year\s+from\s+\w[\w.]*\s*\)\s*=\s*20\d\d\b",
        "message":     "Hardcoded year '= 20XX' — consider EXTRACT(YEAR FROM CURRENT_DATE) for current-year queries",
    },


    # ══════════════════════════════════════════════════════════════════════════
    # WARN — soft warnings (SQL executes but flag is surfaced)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id":          "warn_no_limit",
        "type":        WARN,
        "code":        "MISSING_LIMIT",
        "description": "Non-aggregate queries without LIMIT can return huge result sets.",
        "sql_pattern": r"^(?!.*\b(COUNT|SUM|AVG|MIN|MAX)\s*\()(?!.*\bLIMIT\b)",
        "message":     "WARNING: No LIMIT on non-aggregate query. Add LIMIT 100.",
    },

]
