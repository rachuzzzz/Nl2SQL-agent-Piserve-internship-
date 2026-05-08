"""
SQL Validator — catches common LLM mistakes before execution.

All domain-specific rules (hallucinated columns, UUID FK comparisons, wrong
join directions, spelling typos, etc.) now live in business_rules.REGISTRY.

This file contains only the mechanics:
  - clean_sql()         strips markdown, extracts SELECT, applies autocorrections
  - validate()          runs BLOCK + WARN rules via the engine
  - validate_semantic() checks for UUID leaks and missing lookup JOINs
"""

import re
from typing import Optional

from core.business_rules import get_engine, RulesEngine


class SQLValidator:

    def __init__(self, engine: Optional[RulesEngine] = None):
        self._engine = engine or get_engine()

    # ── UUID FK columns that must be resolved via JOIN ─────────────────────
    UUID_FK_COLUMNS = {
        "facility_id":             ("facility",           "fac",  "fac.name AS facility_name"),
        "inspector_user_id":       ("users",              "u",    "u.first_name || ' ' || u.last_name AS inspector_name"),
        "inspectee_user_id":       ("users",              "insp", "insp.first_name || ' ' || insp.last_name AS inspectee_name"),
        "client_id":               ("client",             "cl",   "cl.name AS client_name"),
        "project_id":              ("project",            "proj", "proj.name AS project_name"),
        "inspection_type_id":      ("inspection_type",    "it",   "it.name AS inspection_type_name"),
        "inspection_sub_type_id":  ("inspection_sub_type","ist",  "ist.name AS subtype_name"),
        "module_id":               ("fb_modules",         "mod",  "mod.name AS module_name"),
        "risk_level_id":           ("risk_level",         "rl",   "rl.name AS risk_level_name"),
        "impact_id":               ("impact",             "im",   "im.name AS impact_name"),
    }

    _DYNAMIC_TABLE_RE = re.compile(
        r'\bfb_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}\b'
    )

    # ── Public API ──────────────────────────────────────────────────────────

    def validate(self, sql: str) -> tuple[bool, list[str]]:
        if not sql or sql.startswith("-- ERROR:"):
            return False, [sql or "Empty SQL"]
        return self._engine.validate_sql(sql)

    def validate_semantic(self, sql: str, question: str) -> list[str]:
        warnings: list[str] = []
        sql_lower = sql.lower()
        q_lower   = question.lower()

        for fk_col, (lookup_table, alias, select_expr) in self.UUID_FK_COLUMNS.items():
            col_in_select  = bool(re.search(rf'\bselect\b.*\b{re.escape(fk_col)}\b', sql_lower, re.DOTALL))
            col_in_groupby = bool(re.search(rf'\bgroup\s+by\b.*\b{re.escape(fk_col)}\b', sql_lower, re.DOTALL))
            lookup_joined  = lookup_table in sql_lower
            if (col_in_select or col_in_groupby) and not lookup_joined:
                warnings.append(
                    f"WARNING: UUID LEAK — '{fk_col}' selected/grouped without joining "
                    f"'{lookup_table}'. Returns raw UUIDs. "
                    f"Fix: JOIN {lookup_table} {alias} ON ...{fk_col} = {alias}.id → SELECT {select_expr}"
                )

        asks_about_answers = bool(re.search(
            r'\b(answer|submission|response|submitted|filled|responded)\b', q_lower))
        uses_inspection = bool(re.search(
            r'\binspection_(report|corrective_action|schedule|cycle)\b', sql_lower))
        uses_ai_answer  = "ai_answers" in sql_lower
        if asks_about_answers and not uses_inspection and not uses_ai_answer:
            warnings.append(
                "WARNING: Question asks about answers but SQL doesn't use ai_answers. "
                "Consider get_answers or joining via ai_answers."
            )

        return warnings

    def clean_sql(self, sql: str) -> str:
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```", "", sql)
        sql = re.sub(r"\[/?[A-Z_]+\]", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^###.*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"-\s*\[.*?\]\(.*?\)", "", sql)
        sql = re.sub(r"^[^\n;]*:\s*$", "", sql, flags=re.MULTILINE)
        sql = sql.strip()

        # Apply AUTOCORRECT rules from the registry
        sql, _ = self._engine.autocorrect_sql(sql)

        if "SELECT" in sql.upper():
            select_pos = sql.upper().find("SELECT")
            cte_match  = re.search(r'\bWITH\s+(?:RECURSIVE\s+)?\w+\s+AS\s*\(', sql, re.IGNORECASE)
            if cte_match and cte_match.start() < select_pos:
                sql = sql[cte_match.start():]
            else:
                sql = sql[select_pos:]

        semi = re.search(r";", sql)
        if semi:
            sql = sql[:semi.end()]

        sql = re.sub(r"\s*\[/?[A-Z_]+\]\s*$", "", sql, flags=re.IGNORECASE).rstrip()

        if sql and not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"

        return sql

    def is_dynamic_answer_table(self, table_name: str) -> bool:
        return bool(re.match(
            r'^fb_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}$',
            table_name
        ))