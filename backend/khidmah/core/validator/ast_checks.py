"""
Module-level sqlglot-AST helper functions shared by core.py (validate())
and semantic.py (validate_semantic()). Extracted verbatim from the
original core/validator.py.
"""

import re

try:
    import sqlglot
    import sqlglot.expressions as exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False

from khidmah.core.validator.models import ValidationIssue




# ---------------------------------------------------------------------------
# AST helpers (sqlglot) — used when available, regex fallback otherwise
# ---------------------------------------------------------------------------

def _strip_nested_parens(sql: str) -> str:
    """
    Replace content inside balanced parentheses with spaces, preserving string length.
    Used to isolate the outermost SELECT clause before regex matching,
    preventing false-positive matches on subquery content.

    Example:
      "SELECT ir.id FROM t WHERE x = (SELECT ir.id FROM t2)"
      → "SELECT ir.id FROM t WHERE x =                      "
    The outer ir.id is preserved; the inner one (in the subquery) is blanked.
    """
    result = list(sql)
    depth = 0
    for i, ch in enumerate(sql):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        elif depth > 0:
            result[i] = ' '
    return ''.join(result)




def _check_undefined_table_aliases(sql: str, issues: list[ValidationIssue]) -> None:
    """
    Use sqlglot AST to verify every table-qualified column reference (alias.col)
    corresponds to a table/alias actually introduced by FROM/JOIN in some scope
    that contains it (including CTEs and subqueries).

    This catches the class of error where Deepseek drops a JOIN but leaves a
    WHERE/SELECT reference to that table's alias — SQL that LOOKS syntactically
    plausible to regex checks but Postgres rejects with:
        missing FROM-clause entry for table "ir"

    Conservative by design — false negatives are OK (we'd rather miss a case
    than block valid SQL): we collect ALL aliases/table-names introduced
    ANYWHERE in the statement (any scope) into one set, plus a few universal
    aliases that legitimately appear without local definition in some dialects.
    A reference is only flagged if its qualifier appears NOWHERE as an
    introduced table/alias/CTE-name in the entire statement.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return

    introduced: set[str] = set()

    # Every table reference contributes its alias (if any) and its bare name.
    for t in tree.find_all(exp.Table):
        if t.alias:
            introduced.add(t.alias.lower())
        # bare table name is also a valid qualifier (e.g. "inspection_report.facility_id")
        if t.name:
            introduced.add(t.name.lower())

    # CTE names (WITH foo AS (...) — "foo" can be referenced like a table)
    for cte in tree.find_all(exp.CTE):
        if cte.alias:
            introduced.add(cte.alias.lower())

    # Subquery aliases: (SELECT ...) AS sub
    for sq in tree.find_all(exp.Subquery):
        if sq.alias:
            introduced.add(sq.alias.lower())

    if not introduced:
        return  # couldn't determine scope — fail safe, don't block

    # Collect all qualified column references: <alias>.<col>
    undefined: set[str] = set()
    for col in tree.find_all(exp.Column):
        if col.table and col.table.lower() not in introduced:
            undefined.add(col.table.lower())

    if undefined:
        bad = ", ".join(sorted(undefined))
        issues.append(ValidationIssue(
            "HARD_FAIL", "UNDEFINED_ALIAS",
            f"Table alias '{bad}' is used in WHERE/SELECT but not introduced "
            f"by any FROM/JOIN — add the missing JOIN or remove the reference",
            detail=(
                f"Aliases referenced but never joined: {bad}. "
                f"Tables/aliases actually in scope: {', '.join(sorted(introduced))}. "
                f"Postgres would reject this with 'missing FROM-clause entry for table \"{sorted(undefined)[0]}\"'."
            ),
        ))
        ValidatorStats.record("UNDEFINED_ALIAS", is_retry_trigger=True)




def _check_ir_id_in_select_ast(sql: str, issues: list[ValidationIssue]) -> None:
    """
    Use sqlglot AST to detect ir.id in the OUTERMOST SELECT column list only.

    CRITICAL: only check the top-level Select node's direct expressions.
    find_all(exp.Select) walks ALL Select nodes including subqueries — if we
    used that, we would false-positive on valid patterns like:
        WHERE aa.inspection_report_id = (SELECT ir.id FROM inspection_report ir ...)
    where ir.id is correctly used as a UUID FK key, never displayed.

    CTE structure: With → this (the final SELECT after WITH definitions).
    Simple query: Select directly.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")

        # Navigate to the outermost SELECT only — do NOT use find_all()
        top_select = None
        if isinstance(tree, exp.Select):
            top_select = tree
        elif isinstance(tree, exp.With):
            # CTE: tree.this is the final SELECT after the CTE definitions
            final = tree.this
            if isinstance(final, exp.Select):
                top_select = final
            elif isinstance(final, exp.Subquery):
                inner = final.this
                if isinstance(inner, exp.Select):
                    top_select = inner

        if top_select is None:
            return

        # Inspect only the direct column expressions of the outermost SELECT.
        # Do NOT recurse into nested Subquery or Anonymous nodes.
        for col_expr in top_select.expressions:
            # Unwrap alias: SELECT ir.id AS something → col_expr is Alias, col_expr.this is Column
            col = col_expr.this if isinstance(col_expr, exp.Alias) else col_expr
            if isinstance(col, exp.Column):
                if (col.table and col.table.lower() == "ir"
                        and col.name.lower() == "id"):
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "UUID_IN_SELECT",
                        "ir.id is the UUID primary key — use ir.inspection_id for display",
                        detail="ir.inspection_id is the human-readable varchar '2026/04/ST001/INS001'",
                    ))
                    ValidatorStats.record("UUID_IN_SELECT", is_retry_trigger=True)
                    return  # one report is enough

    except Exception:
        # AST parse failed — fail safe: do not block on uncertainty
        pass




def _check_uuid_leaks_ast(
    sql: str,
    uuid_fk_cols: dict,
    warns: list[ValidationIssue],
) -> None:
    """
    Use sqlglot AST to detect UUID FK columns selected/grouped without their lookup JOIN.
    AST approach avoids the DOTALL regex false positive where a subquery's GROUP BY
    triggers a warning on a parent query that correctly uses the lookup JOIN.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        # Fallback to regex on parse failure
        _check_uuid_leaks_regex(sql.lower(), uuid_fk_cols, warns)
        return

    # Collect all table names/aliases present in any FROM/JOIN
    joined_tables: set[str] = set()
    for node in tree.find_all(exp.Table):
        if node.name:
            joined_tables.add(node.name.lower())

    # Check outermost SELECT columns and GROUP BY for raw FK column references
    for node in tree.find_all(exp.Select):
        select_cols = {
            c.name.lower()
            for c in node.expressions
            if isinstance(c, exp.Column)
        }
        group_cols: set[str] = set()
        group = node.find(exp.Group)
        if group:
            group_cols = {
                c.name.lower()
                for c in group.find_all(exp.Column)
            }

        exposed = select_cols | group_cols

        for fk_col, (lookup_table, alias, select_expr) in uuid_fk_cols.items():
            if fk_col in exposed and lookup_table.lower() not in joined_tables:
                warns.append(ValidationIssue(
                    "SOFT_WARN", "UUID_LEAK",
                    f"'{fk_col}' selected/grouped without joining '{lookup_table}' — returns raw UUIDs",
                    detail=f"Fix: JOIN {lookup_table} {alias} ON ...{fk_col} = {alias}.id  "
                           f"SELECT {select_expr}",
                    retryable=False,
                ))
                ValidatorStats.record("UUID_LEAK", is_retry_trigger=False)




def _check_uuid_leaks_regex(
    sql_lower: str,
    uuid_fk_cols: dict,
    warns: list[ValidationIssue],
) -> None:
    """
    Regex fallback for UUID leak detection when sqlglot is unavailable.
    Less precise than AST — restricts DOTALL scope to reduce false positives.
    """
    for fk_col, (lookup_table, alias, select_expr) in uuid_fk_cols.items():
        # Check SELECT list only — use non-greedy, stop at first newline cluster
        col_in_select = bool(re.search(
            rf'\bSELECT\b[^\n]{{0,200}}\b{re.escape(fk_col)}\b',
            sql_lower, re.IGNORECASE,
        ))
        col_in_groupby = bool(re.search(
            rf'\bGROUP\s+BY\b[^\n]{{0,200}}\b{re.escape(fk_col)}\b',
            sql_lower, re.IGNORECASE,
        ))
        lookup_joined = lookup_table.lower() in sql_lower

        if (col_in_select or col_in_groupby) and not lookup_joined:
            warns.append(ValidationIssue(
                "SOFT_WARN", "UUID_LEAK",
                f"'{fk_col}' selected/grouped without joining '{lookup_table}' — returns raw UUIDs",
                detail=f"Fix: JOIN {lookup_table} {alias} ON ...{fk_col} = {alias}.id  "
                       f"SELECT {select_expr}",
                retryable=False,
            ))
            ValidatorStats.record("UUID_LEAK", is_retry_trigger=False)