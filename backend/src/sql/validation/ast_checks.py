"""
Module-level sqlglot-AST helper functions shared by engine.py (validate())
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

from src.sql.validation.models import ValidationIssue, ValidatorStats




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




def _check_uuid_pk_in_select_ast(
    sql: str,
    issues: list[ValidationIssue],
    pk_display_map: dict,
) -> None:
    """
    Use sqlglot AST to detect <alias>.id in the OUTERMOST SELECT column list
    only, for any alias in pk_display_map. Generalizes what was originally a
    single hardcoded ir.id-only check (see _check_ir_id_in_select_ast below)
    — confirmed live 2026-07-28 that the SAME "bare UUID PK leaked into the
    display columns" failure happens routinely in the audit domain too
    (acyc.id/audit_cycle, po.id/auditor_portfolio, ap.id/audit_plan — the
    reasoning model habitually adds `.id` to its schema_hint "just in case"
    even when a proper display column/seed example is right in front of it).

    pk_display_map: {alias: display_expr_or_None}. display_expr_or_None is a
    human-readable suggestion for the error message (e.g. "ap.title") — pass
    None for tables with NO display substitute at all (e.g. audit_cycle,
    auditor_portfolio — confirmed live neither has a name/code column; the
    row must be identified by its other columns instead, not by inventing an
    id-based label).

    CRITICAL: only check the top-level Select node's direct expressions.
    find_all(exp.Select) walks ALL Select nodes including subqueries — if we
    used that, we would false-positive on valid patterns like:
        WHERE aa.inspection_report_id = (SELECT ir.id FROM inspection_report ir ...)
    where ir.id is correctly used as a UUID FK key, never displayed.

    CTE structure: With → this (the final SELECT after WITH definitions).
    Simple query: Select directly.
    """
    if not pk_display_map:
        return
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
            if isinstance(col, exp.Column) and col.name.lower() == "id":
                # Exempt columns identity.py's inject_row_identity() added
                # (always output-aliased as "<display_col>__id") — confirmed
                # live 2026-07-30: this check runs a SECOND time in
                # execution.py's execute_sql() (a defense-in-depth
                # re-validation gate independent of generation.py's own
                # post-injection call), and without this exemption it HARD_
                # FAILs on the injected column exactly like a real "bare .id
                # leaked into display" mistake — except the reasoning LLM
                # never wrote it and has no way to remove something it
                # didn't select, so every retry regenerates clean SQL,
                # injection reintroduces the same column, and the query
                # fails every iteration up to the retry cap. This is a
                # deliberately internal-only column (see
                # response_formatting.py's strip_row_identity_columns,
                # which removes it before it ever reaches a user or a
                # prompt) — never a genuine display leak.
                if (col_expr.output_name or "").lower().endswith("__id"):
                    continue
                alias = (col.table or "").lower()
                if alias in pk_display_map:
                    display = pk_display_map[alias]
                    if display:
                        message = f"{alias}.id is the UUID primary key — use {display} for display instead"
                    else:
                        message = (
                            f"{alias}.id is the UUID primary key with no display substitute — "
                            f"identify this row by its other columns instead, do not select the raw id"
                        )
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "UUID_IN_SELECT", message,
                    ))
                    ValidatorStats.record("UUID_IN_SELECT", is_retry_trigger=True)
                    return  # one report is enough

    except Exception:
        # AST parse failed — fail safe: do not block on uncertainty
        pass




_UUID_LITERAL_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _check_display_code_uuid_compare_ast(
    sql: str,
    issues: list[ValidationIssue],
    display_code_map: dict,
) -> None:
    """
    Inverse of _check_uuid_pk_in_select_ast: catches a UUID-shaped string
    literal compared against a VARCHAR display-code column instead of the
    row's real primary key — confirmed live: ordinal-reference resolution
    ("the 1st one in the list") correctly resolved a row's real UUID from
    session context, but the reasoning model wrote
    `WHERE arep.audit_id = '<uuid>'` instead of `WHERE arep.id = '<uuid>'`
    (audit_report.audit_id is a varchar business code like
    '2026/06/ST001/AUD001', never a UUID). The query executed without error
    and silently returned 0 rows instead of the 4 real ones — confirmed
    directly against the database.

    Scoped to the exact (alias, column) pair, NOT the bare column name —
    audit_id/schedule_id/inspection_id are each a genuine UUID foreign key
    on a DIFFERENT table while being the display code on their OWN table
    (e.g. audit_corrective_action.audit_id is a real UUID FK to
    audit_report.id, audit_report.schedule_id is a real UUID FK to
    audit_schedule.id). A bare column-name check would false-block those
    legitimate FK comparisons — see display_code_map's caller in engine.py
    for the full alias/table reasoning.

    display_code_map: {(alias, column): real_id_column_for_message}, aliases
    lowercased at lookup time.
    """
    if not display_code_map:
        return
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        lowered_map = {
            (a.lower(), c.lower()): v for (a, c), v in display_code_map.items()
        }

        for cmp in tree.find_all(exp.EQ, exp.NEQ):
            left, right = cmp.left, cmp.right
            for col, lit in ((left, right), (right, left)):
                if not (isinstance(col, exp.Column)
                        and isinstance(lit, exp.Literal) and lit.is_string):
                    continue
                key = ((col.table or "").lower(), col.name.lower())
                if key in lowered_map and _UUID_LITERAL_RE.match(lit.this or ""):
                    alias, column = key
                    real_id_col = lowered_map[key]
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "DISPLAY_CODE_UUID_COMPARE",
                        f"'{alias}.{column}' is a VARCHAR display code, never a "
                        f"UUID — you likely mean '{real_id_col}' (the row's "
                        f"actual primary key)",
                    ))
                    ValidatorStats.record("DISPLAY_CODE_UUID_COMPARE", is_retry_trigger=True)
                    return  # one report is enough

    except Exception:
        # AST parse failed — fail safe: do not block on uncertainty
        pass




def _check_uuid_fk_compared_to_display_code_ast(
    sql: str,
    issues: list[ValidationIssue],
    uuid_fk_display_code_map: dict,
) -> None:
    """
    Inverse of _check_display_code_uuid_compare_ast: catches a display-code-
    shaped (i.e. non-UUID) string literal compared against a column that is
    a genuine UUID FK on THIS table, which merely happens to share its bare
    column name with the display-code column on a DIFFERENT table —
    confirmed live twice (2026-07-29 and 2026-08-04): the reasoning model
    wrote `WHERE aca.audit_id = '2026/06/ST074/AUD001'` (a display code),
    but audit_corrective_action.audit_id is a UUID FK to audit_report.id —
    the display code lives on audit_report.audit_id instead, reached only
    by joining through it. The query failed at DB execution with
    `psycopg2.errors.InvalidTextRepresentation: invalid input syntax for
    type uuid`, burning 2-4 LLM iterations to self-correct.

    Since these columns can only ever legitimately hold a UUID, ANY
    string literal compared against them that is not UUID-shaped is
    unconditionally wrong — no risk of false-blocking a legitimate FK
    comparison the way a bare column-name check would (see the sibling
    function's docstring for why this family of checks stays alias-scoped).

    uuid_fk_display_code_map: {(alias, column): (real_table, real_column)}
    where real_table/real_column identify the table to join through and
    the display-code column to filter on instead, aliases lowercased at
    lookup time.
    """
    if not uuid_fk_display_code_map:
        return
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        lowered_map = {
            (a.lower(), c.lower()): v for (a, c), v in uuid_fk_display_code_map.items()
        }

        for cmp in tree.find_all(exp.EQ, exp.NEQ):
            left, right = cmp.left, cmp.right
            for col, lit in ((left, right), (right, left)):
                if not (isinstance(col, exp.Column)
                        and isinstance(lit, exp.Literal) and lit.is_string):
                    continue
                key = ((col.table or "").lower(), col.name.lower())
                if key in lowered_map and not _UUID_LITERAL_RE.match(lit.this or ""):
                    alias, column = key
                    real_table, real_column = lowered_map[key]
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "UUID_FK_COMPARED_TO_DISPLAY_CODE",
                        f"'{alias}.{column}' is a UUID foreign key, never a display "
                        f"code — join {real_table} and filter on its "
                        f"'{real_column}' column instead",
                    ))
                    ValidatorStats.record("UUID_FK_COMPARED_TO_DISPLAY_CODE", is_retry_trigger=True)
                    return  # one report is enough

    except Exception:
        # AST parse failed — fail safe: do not block on uncertainty
        pass




_OWN_DISPLAY_CODE_RE = re.compile(
    r"^\d{4}/\d{2}/ST\d+/([A-Za-z]+)\d+$",
)


def _check_display_code_wrong_table_ast(
    sql: str,
    issues: list[ValidationIssue],
    own_display_code_token_map: dict,
) -> None:
    """
    Catches a display code that is well-formed but belongs to the WRONG
    table — e.g. `arep.audit_id = '2026/08/ST040/SCH001'`. That literal is
    a genuine, valid display code, just not audit_report's own: the 'SCH'
    token marks it as an audit_schedule code, not the 'AUD' audit_report
    expects. Neither existing display-code/UUID check catches this — both
    sides here are ordinary non-UUID strings, so there's no UUID-shape
    mismatch to flag.

    Confirmed live 2026-08-04: ordinal resolution ("the second one") handed
    the model a schedule's row/code correctly, but the reasoning model then
    filtered audit_report.audit_id with the SCHEDULE's code instead of
    joining audit_schedule and going through audit_report.schedule_id — a
    schedule and its resulting report have DIFFERENT, non-interchangeable
    codes ('.../SCH001' vs '.../AUD001'). The query executed without error
    and silently returned 0 rows instead of the report's real answers —
    confirmed directly against the database (verified: the schedule DOES
    have a linked report with a different, correct audit_id code).

    Scoped to the exact (alias, column) pair for the same reason as the
    sibling checks above — a bare column-name check would false-block
    audit_corrective_action.corrective_action_id / inspection_corrective_
    action.corrective_action_id, whose OWN values embed a parent code plus
    a '_CAnnn'/'_MAnnn' suffix (e.g. '2026/06/ST041/AUD003_CA002') and so
    never fullmatch this pattern in the first place — no special-casing
    needed, they're simply never in this map.

    own_display_code_token_map: {(alias, column): expected_token}, e.g.
    {("arep","audit_id"): "AUD", ("asch","schedule_id"): "SCH",
     ("ir","inspection_id"): "INS"}. Token compared case-insensitively.
    """
    if not own_display_code_token_map:
        return
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        lowered_map = {
            (a.lower(), c.lower()): v for (a, c), v in own_display_code_token_map.items()
        }

        for cmp in tree.find_all(exp.EQ, exp.NEQ):
            left, right = cmp.left, cmp.right
            for col, lit in ((left, right), (right, left)):
                if not (isinstance(col, exp.Column)
                        and isinstance(lit, exp.Literal) and lit.is_string):
                    continue
                key = ((col.table or "").lower(), col.name.lower())
                if key not in lowered_map:
                    continue
                m = _OWN_DISPLAY_CODE_RE.match(lit.this or "")
                if not m:
                    continue
                found_token = m.group(1).upper()
                expected_token = lowered_map[key]
                if found_token != expected_token:
                    alias, column = key
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "DISPLAY_CODE_WRONG_TABLE",
                        f"'{lit.this}' is a {found_token} code, not an "
                        f"{expected_token} code — {alias}.{column} needs its "
                        f"own table's code; join the other table instead",
                    ))
                    ValidatorStats.record("DISPLAY_CODE_WRONG_TABLE", is_retry_trigger=True)
                    return  # one report is enough

    except Exception:
        # AST parse failed — fail safe: do not block on uncertainty
        pass




# A fourth failure shape in the same column family as the three checks
# above, but through a scalar subquery instead of a literal — so none of
# them catch it (all three require a Literal on one side).
#
# Confirmed live TWICE, ~5 days apart, two different sub-shapes of the same
# root confusion:
#   2026-07-30 (qid=d5423641): JOIN audit_corrective_action aca ON
#     aca.audit_id = (SELECT arep.audit_id FROM audit_report arep WHERE ...)
#     — aca.audit_id is a genuine UUID FK to audit_report.id, but the
#     subquery selected arep.audit_id (audit_report's OWN varchar display
#     code) instead of arep.id. Failed with "operator does not exist:
#     uuid = character varying".
#   2026-08-04 (qid=9caecfb2): WHERE asch.schedule_id = (SELECT id FROM
#     audit_schedule WHERE schedule_id = '2026/08/ST040/SCH001') —
#     asch.schedule_id is itself the varchar display code (a direct literal
#     filter would have sufficed), but the subquery resolved audit_schedule's
#     UUID id instead. Failed with "operator does not exist: character
#     varying = uuid".
#
# Both are the SAME underlying ambiguity as the three checks above —
# audit_id/schedule_id/corrective_action_id/inspection_id is a genuine UUID
# FK on some tables and the VARCHAR display code on others — just with one
# side of the comparison routed through a subquery. Unlike the literal-based
# checks (which can be semantically wrong but still execute, e.g. silently
# returning 0 rows), a uuid/varchar type mismatch is a guaranteed Postgres
# execution error with zero false-positive risk: these two kinds NEVER
# implicitly cast to each other, so flagging any confirmed mismatch here is
# strictly beneficial — it converts a certain DB round-trip failure into an
# immediate, informative retry.
_UUID_KIND = "uuid"
_DISPLAY_CODE_KIND = "display_code"

# (table, column) -> kind, spanning both domains — deliberately keyed by
# real table name (not alias), since a subquery's own FROM can use any
# alias (or none) independent of the outer query's convention.
_COLUMN_KIND_BY_TABLE: dict[tuple[str, str], str] = {
    ("audit_report", "id"): _UUID_KIND,
    ("audit_report", "audit_id"): _DISPLAY_CODE_KIND,
    ("audit_schedule", "id"): _UUID_KIND,
    ("audit_schedule", "schedule_id"): _DISPLAY_CODE_KIND,
    ("audit_corrective_action", "id"): _UUID_KIND,
    ("audit_corrective_action", "corrective_action_id"): _DISPLAY_CODE_KIND,
    ("audit_corrective_action", "audit_id"): _UUID_KIND,  # FK to audit_report.id
    ("inspection_report", "id"): _UUID_KIND,
    ("inspection_report", "inspection_id"): _DISPLAY_CODE_KIND,
    ("inspection_corrective_action", "id"): _UUID_KIND,
    ("inspection_corrective_action", "corrective_action_id"): _DISPLAY_CODE_KIND,
    ("inspection_corrective_action", "inspection_id"): _UUID_KIND,  # FK to inspection_report.id
}


def _direct_from_join_tables(select_node) -> list:
    """Tables introduced directly by THIS select's own FROM/JOIN clauses —
    deliberately shallow (does not recurse into nested subqueries), so each
    query level gets its own independent alias scope."""
    tables = []
    from_ = select_node.args.get("from_")
    if from_ is not None and isinstance(from_.this, exp.Table):
        tables.append(from_.this)
    for j in (select_node.args.get("joins") or []):
        if isinstance(j.this, exp.Table):
            tables.append(j.this)
    return tables


def _alias_table_map(select_node) -> dict:
    m = {}
    for t in _direct_from_join_tables(select_node):
        name = (t.name or "").lower()
        if not name:
            continue
        alias = (t.alias or t.name or "").lower()
        m[alias] = name
    return m


def _classify_column(col, alias_map: dict) -> str | None:
    colname = (col.name or "").lower()
    alias = (col.table or "").lower()
    if alias:
        table = alias_map.get(alias)
    elif len(alias_map) == 1:
        # Unqualified column, single-table scope — e.g. "SELECT id FROM
        # audit_schedule WHERE ...", the exact shape of the 2026-08-04 bug.
        table = next(iter(alias_map.values()))
    else:
        table = None
    if not table:
        return None
    return _COLUMN_KIND_BY_TABLE.get((table, colname))


def _check_subquery_uuid_display_code_mismatch_ast(
    sql: str,
    issues: list[ValidationIssue],
) -> None:
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        top_level = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        if top_level is None:
            return
        outer_alias_map = _alias_table_map(top_level)

        for cmp in tree.find_all(exp.EQ, exp.NEQ):
            left, right = cmp.left, cmp.right
            for col_side, other_side in ((left, right), (right, left)):
                if not isinstance(col_side, exp.Column):
                    continue
                if not isinstance(other_side, exp.Subquery):
                    continue
                inner_select = other_side.this
                if (not isinstance(inner_select, exp.Select)
                        or len(inner_select.expressions) != 1):
                    continue
                inner_expr = inner_select.expressions[0]
                if not isinstance(inner_expr, exp.Column):
                    continue

                inner_alias_map = _alias_table_map(inner_select)
                outer_kind = _classify_column(col_side, outer_alias_map)
                inner_kind = _classify_column(inner_expr, inner_alias_map)

                if outer_kind and inner_kind and outer_kind != inner_kind:
                    qualified = (
                        f"{col_side.table}.{col_side.name}" if col_side.table
                        else col_side.name
                    )
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "SUBQUERY_UUID_DISPLAY_CODE_MISMATCH",
                        f"'{qualified}' is {outer_kind}, but the subquery "
                        f"selects '{inner_expr.name}' ({inner_kind}) — "
                        f"Postgres has no uuid/varchar cast between them",
                    ))
                    ValidatorStats.record(
                        "SUBQUERY_UUID_DISPLAY_CODE_MISMATCH", is_retry_trigger=True,
                    )
                    return  # one report is enough

    except Exception:
        # AST parse failed — fail safe: do not block on uncertainty
        pass




def _check_ir_id_in_select_ast(sql: str, issues: list[ValidationIssue]) -> None:
    """Inspection-domain wrapper — kept for backward-compatible callers.
    See _check_uuid_pk_in_select_ast for the general implementation."""
    _check_uuid_pk_in_select_ast(
        sql, issues,
        {"ir": "ir.inspection_id (the human-readable varchar '2026/04/ST001/INS001')"},
    )




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




_AGG_FUNCS = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)


def _check_ungrouped_select_columns_ast(sql: str, issues: list[ValidationIssue]) -> None:
    """
    Postgres requires every SELECT-list column that isn't wrapped in an
    aggregate function to appear in GROUP BY. Deepseek routinely violates
    this by tacking on one extra "detail" column next to an aggregate
    (e.g. a row-level id alongside COUNT(...)) — syntactically plausible,
    so the cleaner's regex/shape checks pass it, but Postgres rejects it at
    execution with "column must appear in the GROUP BY clause or be used
    in an aggregate function".

    Confirmed live: eval_audit.py's ADMIN-08 run ("audit schedules per
    country") selected asch.schedule_id alongside COUNT(DISTINCT asch.id)
    with GROUP BY fac.country_id, c.name — asch.schedule_id belongs to
    neither, and the query only failed at the DB round-trip, burning an
    LLM call a pre-execution check could have caught for free.

    Deliberately narrow — only flags a BARE exp.Column directly in the
    outermost SELECT list (unwrapped from a simple Alias), never columns
    buried inside CASE/window/concat expressions, where Postgres' actual
    functional-dependency rules are harder to reproduce safely without
    false positives. Also bails out on positional GROUP BY (e.g. "GROUP BY
    1, 2") since ordinal positions can't be reliably correlated back here.
    Fail-safe on anything ambiguous — false negatives are fine, a false
    positive would block valid SQL.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return

    top_select = None
    if isinstance(tree, exp.Select):
        top_select = tree
    elif isinstance(tree, exp.With):
        final = tree.this
        if isinstance(final, exp.Select):
            top_select = final

    if top_select is None:
        return

    group = top_select.args.get("group")
    if group is None:
        return
    if any(isinstance(g, exp.Literal) and g.is_int for g in group.expressions):
        return

    grouped_keys = {g.sql(dialect="postgres").lower() for g in group.expressions}

    has_aggregate_sibling = any(
        isinstance((e.this if isinstance(e, exp.Alias) else e), _AGG_FUNCS)
        or list((e.this if isinstance(e, exp.Alias) else e).find_all(*_AGG_FUNCS))
        for e in top_select.expressions
    )
    if not has_aggregate_sibling:
        return

    for col_expr in top_select.expressions:
        inner = col_expr.this if isinstance(col_expr, exp.Alias) else col_expr
        if not isinstance(inner, exp.Column):
            continue
        key = inner.sql(dialect="postgres").lower()
        if key not in grouped_keys:
            rendered = inner.sql(dialect="postgres")
            issues.append(ValidationIssue(
                "HARD_FAIL", "UNGROUPED_COLUMN",
                f"'{rendered}' is selected alongside an aggregate but is not in "
                f"GROUP BY — Postgres will reject this with 'column must appear "
                f"in the GROUP BY clause or be used in an aggregate function'. "
                f"Either add it to GROUP BY or remove it from SELECT.",
                detail=f"GROUP BY currently has: {group.sql(dialect='postgres')}",
            ))
            ValidatorStats.record("UNGROUPED_COLUMN", is_retry_trigger=True)
            return