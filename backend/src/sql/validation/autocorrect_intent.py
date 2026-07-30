"""
IntentAutocorrectMixin: deterministic SQL fixes gated by the ORIGINAL
QUESTION TEXT, not just SQL shape — turns a "BLOCK + hope the LLM adds the
missing clause on retry" round trip into "add it ourselves", the same
principle cleaner.py's ast_repair() already applies to active=true (never
relying on the LLM to remember a filter it has already been told about
dozens of times in the same session).

Reuses the EXACT compiled question_pattern/sql_pattern regexes already
defined on the corresponding BLOCK rule (audit_rules.py) rather than
duplicating them here, so the two can never silently drift out of sync.
"""

try:
    import sqlglot
    import sqlglot.expressions as exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False


class IntentAutocorrectMixin:

    def inject_ca_adequacy_filter(self, sql: str, question: str) -> tuple:
        """
        Confirmed live 2026-07-30: "show all corrective actions raised for
        the above mentioned audits" needed 2 LLM round trips (an extra ~9s,
        ~14K tokens) purely to add `AND aca.adequacy_status = 'APPROVE'` —
        the exact fix audit_block_ca_draft_not_excluded's BLOCK message
        already states verbatim, every single time this shape recurs
        (confirmed the single most common retry cause in this domain, per
        that rule's own docstring). This makes it unnecessary: parse the SQL
        with sqlglot and append the filter directly, before validate() ever
        gets a chance to block it.

        Only fires when BOTH of that rule's own gates would otherwise have
        blocked this exact (question, sql) pair — reusing its compiled
        _q_re/_sql_re directly, never re-deriving them, so this can never
        drift out of sync with what the BLOCK rule considers "a general
        listing question missing the filter".

        Deliberately only ever APPENDS the filter — never attempts to
        rewrite an existing (wrong) one. Any existing adequacy_status/
        corrective_action_id reference of any shape means the BLOCK rule's
        own sql_pattern wouldn't have matched either, so there'd be nothing
        for this to safely do.
        """
        if not _HAS_SQLGLOT:
            return sql, []

        _rule = next(
            (r for r in self._rules._block_rules
             if r["id"] == "audit_block_ca_draft_not_excluded"),
            None
        )
        if _rule is None or "_q_re" not in _rule or "_sql_re" not in _rule:
            return sql, []
        if not _rule["_q_re"].search(question):
            return sql, []
        if not _rule["_sql_re"].search(sql):
            return sql, []  # already has the filter (or doesn't touch aca) — nothing to do

        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            return sql, []

        # Deliberately restricted to a plain outermost SELECT (no CTE
        # handling) — a general corrective-action LISTING question landing
        # on a CTE-wrapped query is not a shape seen in practice; fail safe
        # rather than risk misplacing the filter.
        if not isinstance(tree, exp.Select):
            return sql, []

        alias = None
        from_clause = tree.args.get("from_") or tree.args.get("from")
        if (from_clause is not None and isinstance(from_clause.this, exp.Table)
                and from_clause.this.name
                and from_clause.this.name.lower() == "audit_corrective_action"):
            alias = (from_clause.this.alias or from_clause.this.name).lower()
        if alias is None:
            for j in (tree.args.get("joins") or []):
                t = j.this
                if isinstance(t, exp.Table) and t.name and t.name.lower() == "audit_corrective_action":
                    alias = (t.alias or t.name).lower()
                    break
        if alias is None:
            return sql, []

        cond = exp.EQ(
            this=exp.Column(
                this=exp.Identifier(this="adequacy_status"),
                table=exp.Identifier(this=alias),
            ),
            expression=exp.Literal.string("APPROVE"),
        )
        tree = tree.where(cond)

        try:
            repaired = tree.sql(dialect="postgres", pretty=False)
            if repaired and not repaired.rstrip().endswith(";"):
                repaired = repaired.rstrip() + ";"
            return repaired, [
                f"Injected {alias}.adequacy_status = 'APPROVE' "
                f"(audit_corrective_action general listing default)"
            ]
        except Exception:
            return sql, []

    def downgrade_ca_multi_axis_joins(self, sql: str, question: str) -> tuple:
        """
        Confirmed live 2026-07-30: "now for the first one show all
        corrective actions" needed 2 full reasoning-model retries (an extra
        ~15s, ~30K tokens) purely because audit_block_ca_multi_axis_inner_join
        kept firing — risk_level/impact/compliance are three independent,
        often-NULL FKs, and INNER-joining two or more at once silently zeros
        out rows missing an FK on the axis the question never asked about.

        The precise trigger wasn't even the reasoning model padding its own
        request with unrequested columns — turn 1's own SQL used safe LEFT
        JOINs for all three from the start. It was this module's OWN
        uuid-guard retry (generation.py — forces a fresh full regeneration
        when a resolved-identity UUID goes missing from the SQL): asking the
        SQL model to regenerate from scratch to fix an unrelated WHERE-clause
        issue silently downgraded those same LEFT JOINs to plain INNER JOINs
        as a side effect, which then tripped this BLOCK on a query that was
        already safe one step earlier.

        Rather than chase reasoning-model retries to preserve join safety
        across regenerations, fix it deterministically here: whenever this
        exact (question, sql) pair would trip
        audit_block_ca_multi_axis_inner_join, downgrade EVERY INNER JOIN to
        risk_level/impact/compliance on audit_corrective_action to LEFT JOIN
        instead of blocking. A LEFT JOIN is always the structurally safe
        choice here — it can only ever ADD potential nulls to display
        columns, never silently drop a legitimate row — so this is a pure
        safety improvement, matching the BLOCK rule's own suggested remedy
        ("...or use LEFT JOIN if you want to display one without filtering
        on it") without needing a retry to get there.

        Reuses audit_block_ca_multi_axis_inner_join's compiled _q_re/_sql_re
        directly (never re-derived) so this can never drift out of sync with
        what that BLOCK rule considers "two or more axes INNER-joined but
        not asked about".
        """
        if not _HAS_SQLGLOT:
            return sql, []

        _rule = next(
            (r for r in self._rules._block_rules
             if r["id"] == "audit_block_ca_multi_axis_inner_join"),
            None
        )
        if _rule is None or "_q_re" not in _rule or "_sql_re" not in _rule:
            return sql, []
        if not _rule["_q_re"].search(question):
            return sql, []
        if not _rule["_sql_re"].search(sql):
            return sql, []  # not simultaneously INNER-joining 2+ axes — nothing to do

        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            return sql, []

        if not isinstance(tree, exp.Select):
            return sql, []

        _axis_tables = {"risk_level", "impact", "compliance"}
        mutations: list = []
        for j in (tree.args.get("joins") or []):
            t = j.this
            if not (isinstance(t, exp.Table) and t.name):
                continue
            if t.name.lower() not in _axis_tables:
                continue
            _side = (j.args.get("side") or "").upper()
            if _side:
                continue  # already LEFT/RIGHT/FULL — nothing to downgrade
            alias = (t.alias or t.name).lower()
            j.set("side", "LEFT")
            mutations.append(
                f"Downgraded JOIN {t.name} {alias} to LEFT JOIN "
                f"(risk_level/impact/compliance are independent, often-NULL "
                f"axes — INNER-joining 2+ at once silently drops rows missing "
                f"an FK on the axis not asked about)"
            )

        if not mutations:
            return sql, []

        try:
            repaired = tree.sql(dialect="postgres", pretty=False)
            if repaired and not repaired.rstrip().endswith(";"):
                repaired = repaired.rstrip() + ";"
            return repaired, mutations
        except Exception:
            return sql, []
