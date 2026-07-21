"""
ScopeMixin: enforce_scope() and _build_secondary_predicate() -- RBAC
visibility-scope SQL rewriting. Extracted verbatim from the original
core/validator.py.
"""

import re
import logging

try:
    import sqlglot
    import sqlglot.expressions as exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False

_log_security = logging.getLogger("inspectai.security")


class ScopeMixin:

    # facility is intentionally excluded — public reference data.
    _SECONDARY_SCOPE_TABLES: dict = {
        'inspection_cycle': {
            'mode': 'forward', 'fk_col': 'cycle_id',
        },
        'inspection_corrective_action': {
            'mode': 'reverse', 'own_col': 'inspection_id',
        },
        'po_details': {
            'mode': 'reverse2', 'own_col': 'mitigative_action_uuid',
            'mid_table': 'inspection_corrective_action', 'mid_fk': 'inspection_id',
        },
        'pr_details': {
            'mode': 'reverse2', 'own_col': 'mitigative_action_uuid',
            'mid_table': 'inspection_corrective_action', 'mid_fk': 'inspection_id',
        },
        'inspection_schedule': {
            'mode': 'direct',
            'column_map': {'inspector_user_id': 'inspector_id', 'inspectee_user_id': 'inspectee_id'},
        },
        'inspector_portfolio': {
            'mode': 'inspector_only', 'own_col': 'user_id',
        },
        'inspector_portfolio_details': {
            'mode': 'inspector_only_reverse', 'own_col': 'portfolio_id',
            'mid_table': 'inspector_portfolio', 'mid_col': 'user_id',
        },
        'users': {
            'mode': 'connected_users', 'own_col': 'id',
        },
    }


    _USER_SUBQ_RE = re.compile(
        r"\(SELECT id FROM users WHERE id_fk = \d+ AND active = true AND is_synced = true\)"
    )


    @classmethod
    def _build_secondary_predicate(cls, tbl_name: str, spec: dict, alias: str,
                                    scope_filter: str, ir_predicate: str) -> str:
        mode = spec['mode']

        if mode == 'forward':
            return (
                f"{alias}.id IN (SELECT DISTINCT {spec['fk_col']} "
                f"FROM inspection_report WHERE {ir_predicate} AND active = true)"
            )

        if mode == 'reverse':
            return (
                f"{alias}.{spec['own_col']} IN (SELECT id FROM inspection_report "
                f"WHERE {ir_predicate} AND active = true)"
            )

        if mode == 'reverse2':
            return (
                f"{alias}.{spec['own_col']} IN ("
                f"SELECT _sc_mid.id FROM {spec['mid_table']} _sc_mid "
                f"JOIN inspection_report _sc_ir ON _sc_mid.{spec['mid_fk']} = _sc_ir.id "
                f"WHERE {ir_predicate} AND _sc_mid.active = true AND _sc_ir.active = true)"
            )

        if mode == 'direct':
            pred = scope_filter
            for old_col, new_col in spec['column_map'].items():
                pred = pred.replace(f"ir.{old_col}", f"{alias}.{new_col}")
            return pred

        if mode == 'inspector_only':
            if 'inspector_user_id' not in scope_filter:
                return "1 = 0"
            m = cls._USER_SUBQ_RE.search(scope_filter)
            subq = m.group(0) if m else "(SELECT NULL)"
            return f"{alias}.{spec['own_col']} = {subq}"

        if mode == 'inspector_only_reverse':
            if 'inspector_user_id' not in scope_filter:
                return "1 = 0"
            m = cls._USER_SUBQ_RE.search(scope_filter)
            subq = m.group(0) if m else "(SELECT NULL)"
            return (
                f"{alias}.{spec['own_col']} IN (SELECT id FROM {spec['mid_table']} "
                f"WHERE {spec['mid_col']} = {subq} AND active = true)"
            )

        if mode == 'connected_users':
            return (
                f"{alias}.{spec['own_col']} IN ("
                f"SELECT inspector_user_id FROM inspection_report WHERE {ir_predicate} AND active = true "
                f"UNION "
                f"SELECT inspectee_user_id FROM inspection_report WHERE {ir_predicate} AND active = true)"
            )

        raise ValueError(f"unknown secondary-scope mode {mode!r} for {tbl_name}")


    def enforce_scope(self, sql: str, scope_filter: str) -> str:
        """Hard data-visibility guard — AST-level injection, always applied.

        scope_filter == ''       → unrestricted, return SQL unchanged
        scope_filter == 'BLOCK'  → raise ValueError('SCOPE_BLOCKED')
        anything else            → inject into every Select that directly
                                   FROM/JOINs inspection_report, or a
                                   secondary-scoped table (see
                                   _SECONDARY_SCOPE_TABLES / _build_secondary_predicate)

        Raises ValueError('SCOPE_BLOCKED') — caller converts to HTTP 403.
        """
        if not scope_filter:
            return sql
        if scope_filter == 'BLOCK':
            _log_security.warning("[SCOPE] enforce_scope called with BLOCK — raising SCOPE_BLOCKED")
            raise ValueError('SCOPE_BLOCKED')

        sql_lower = sql.lower()
        has_ir    = 'inspection_report' in sql_lower

        # Strip "ir." alias so the predicate works inside a raw subquery
        # (e.g., dual-role "(ir.inspector_user_id=X OR ir.inspectee_user_id=X)"
        #  → "(inspector_user_id=X OR inspectee_user_id=X)")
        ir_predicate = scope_filter.replace('ir.', '')

        secondary_tables = [
            (tbl, spec) for tbl, spec in self._SECONDARY_SCOPE_TABLES.items()
            if tbl in sql_lower
        ]

        if not has_ir and not secondary_tables:
            # Pure reference tables (frequency_definition, inspection_type, etc.)
            # carry no row-level owner — no scope injection required.
            return sql

        _log_security.info(
            "[SCOPE] injecting scope filter: via=%s filter=%s",
            (["inspection_report"] if has_ir else []) + [t for t, _ in secondary_tables],
            scope_filter[:120],
        )

        # ── Primary path: AST injection via sqlglot ──────────────────────────
        if _HAS_SQLGLOT:
            try:
                tree       = sqlglot.parse_one(sql, dialect="postgres")
                scope_expr = sqlglot.parse_one(scope_filter, dialect="postgres")

                def _alias_for_table(sel: exp.Select, tbl_name: str):
                    """Return alias if sel directly FROM/JOINs tbl_name, else None."""
                    # sqlglot v20+ uses 'from_' key; older versions use 'from'.
                    # Matches the fallback already used by _direct_tables() above —
                    # without it this always returns None on the installed
                    # sqlglot version, silently disabling scope injection.
                    from_clause = sel.args.get("from_") or sel.args.get("from")
                    if from_clause and isinstance(from_clause.this, exp.Table):
                        if from_clause.this.name.lower() == tbl_name:
                            return from_clause.this.alias or tbl_name
                    for join in (sel.args.get("joins") or []):
                        if isinstance(join.this, exp.Table):
                            if join.this.name.lower() == tbl_name:
                                return join.this.alias or tbl_name
                    return None

                # Each SELECT node (including every branch of a UNION, and every
                # subquery) is scoped independently — a query can mix a branch
                # that joins inspection_report with a branch that reaches a
                # secondary-scoped table bare; both must be caught on their
                # own terms, not decided once for the whole statement.
                def _inject(node):
                    if not isinstance(node, exp.Select):
                        return node
                    if _alias_for_table(node, 'inspection_report'):
                        return node.where(scope_expr.copy())
                    for tbl_name, spec in secondary_tables:
                        alias = _alias_for_table(node, tbl_name)
                        if alias:
                            subq_where = self._build_secondary_predicate(
                                tbl_name, spec, alias, scope_filter, ir_predicate
                            )
                            node = node.where(
                                sqlglot.parse_one(subq_where, dialect="postgres")
                            )
                    return node

                rewritten = tree.transform(_inject).sql(dialect="postgres", pretty=False)

                if rewritten and not rewritten.rstrip().endswith(";"):
                    rewritten = rewritten.rstrip() + ";"
                return rewritten

            except Exception:
                logging.warning("enforce_scope: AST injection failed, using string fallback",
                                exc_info=True)

        # ── Fallback: string injection (no sqlglot) ───────────────────────────
        sql_work  = sql.rstrip().rstrip(";").rstrip()
        has_where = bool(re.search(r"\bWHERE\b", sql_work, re.IGNORECASE))
        connector = "AND" if has_where else "WHERE"
        term      = re.search(
            r"\b(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT)\b", sql_work, re.IGNORECASE
        )

        parts = ([scope_filter] if has_ir else []) + [
            self._build_secondary_predicate(tbl_name, spec, tbl_name, scope_filter, ir_predicate)
            for tbl_name, spec in secondary_tables
        ]
        fragment = " AND ".join(f"({p})" for p in parts)

        if term:
            pos = term.start()
            return f"{sql_work[:pos].rstrip()}\n  {connector} {fragment}\n{sql_work[pos:]};"
        return f"{sql_work}\n  {connector} {fragment};"
