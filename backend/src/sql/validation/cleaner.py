"""
CleanerMixin: clean_sql() (formatting normalization) and ast_repair()
(deterministic sqlglot-AST-level repairs). Extracted verbatim from the
original core/validator.py.
"""

import re

try:
    import sqlglot
    import sqlglot.expressions as exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False


class CleanerMixin:

    # ------------------------------------------------------------------ #
    # clean_sql — formatting only + structural ICA join fix
    # ------------------------------------------------------------------ #

    def clean_sql(self, sql: str) -> str:
        """
        Normalize raw LLM SQL output for execution.

        ALLOWED mutations (formatting / structural type-fixes only):
          ✓ Strip markdown fences and LLM formatting artifacts
          ✓ Normalize whitespace
          ✓ Extract SELECT/WITH from preamble prose
          ✓ Normalize trailing semicolon
          ✓ Fix reserved word aliases (is → isch)
          ✓ Fix ICA→IR join varchar/UUID type mismatch (structural, not semantic)

        NOT ALLOWED (these would change query semantics):
          ✗ Adding/removing WHERE conditions
          ✗ Modifying GROUP BY / HAVING / aggregates
          ✗ Adding/removing JOINs
          ✗ Injecting filters
          ✗ Rewriting ORDER BY

        The ICA→IR join fix is retained because:
          - It is a type-level structural error (VARCHAR vs UUID)
          - The wrong form ALWAYS causes a DB type mismatch error
          - The fix NEVER changes what data is returned — only fixes the join type
          - It has 4 variants covering deepseek's CAST/:: fallback attempts
        """
        # Strip markdown fences and LLM formatting artifacts
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```", "", sql)
        sql = re.sub(r"\[/?[A-Z_]+\]", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^###.*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"-\s*\[.*?\]\(.*?\)", "", sql)
        sql = re.sub(r"^[^\n;]*:\s*$", "", sql, flags=re.MULTILINE)
        sql = sql.strip()

        # Extract SELECT/WITH from any LLM preamble prose
        if "SELECT" in sql.upper():
            select_pos = sql.upper().find("SELECT")
            cte_match = re.search(
                r'\bWITH\s+(?:RECURSIVE\s+)?\w+\s+AS\s*\(',
                sql, re.IGNORECASE,
            )
            if cte_match and cte_match.start() < select_pos:
                sql = sql[cte_match.start():]
            else:
                sql = sql[select_pos:]

        # Truncate at first semicolon (LLM sometimes appends explanation after ;)
        semi = re.search(r";", sql)
        if semi:
            sql = sql[:semi.end()]

        # ICA→IR join structural fix (4 variants: direct, reversed, CAST, ::type)
        # and reserved-alias fix ('is' → 'isch') — both consumed from
        # core/business_rules.py's REGISTRY via _registry_autocorrects().
        # Rationale for the ICA/IR fix: ica.inspection_id is UUID FK;
        # ir.inspection_id is VARCHAR — comparing them is always a type error.
        _sql_before_autocorrect = sql
        sql = self._registry_autocorrects(sql)
        # fix_reserved_aliases (removed) additionally rewrote bare "is."
        # qualifiers anywhere in the query, once "is" was confirmed used as a
        # table alias — the registry's autocorrect_reserved_alias_is rule only
        # fixes the alias *declaration* site itself. No registry equivalent for
        # this broader cleanup; preserved here directly, same trigger condition.
        _is_alias_rule = next(
            (r for r in self._rules._autocorr_rules if r["id"] == "autocorrect_reserved_alias_is"),
            None,
        )
        if _is_alias_rule and _is_alias_rule["_sql_re"].search(_sql_before_autocorrect):
            sql = re.sub(r'\bis\.', 'isch.', sql)

        sql = re.sub(r"\s*\[/?[A-Z_]+\]\s*$", "", sql, flags=re.IGNORECASE).rstrip()

        if sql and not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"

        return sql


    # ------------------------------------------------------------------ #
    # AST repair — deterministic post-generation fixes via sqlglot
    # ------------------------------------------------------------------ #

    # Tables that have an `active` boolean soft-delete column.
    # Audit-domain tables added per knowledge/semantics_registry.py's
    # audit_columns_block(), which already documents every audit_* table as
    # soft-delete-tracked for the LLM's benefit — this set is what actually
    # activates the deterministic auto-repair (the "[ast-repair] Injected
    # X.active = true" fix) for them, which previously only fired for
    # inspection tables despite the fact being known and documented.
    _SOFT_DELETE_TABLES: frozenset = frozenset({
        "inspection_report", "inspection_corrective_action", "facility",
        "users", "inspection_type", "inspection_sub_type",
        "inspector_portfolio", "inspector_portfolio_details",
        "frequency_definition", "po_details", "pr_details", "risk_level",
        "client", "project", "country", "region", "entity", "organisation",
        "inspection_schedule", "impact",
        # Audit domain
        "audit_plan", "audit_schedule", "audit_report", "audit_corrective_action",
        "audit_cycle", "auditor_portfolio", "auditor_portfolio_detail",
        "audit_universe", "audit_category", "audit_sub_category", "audit_type",
        "audit_forms", "compliance",
    })


    # Tables that NEVER have an active column — never inject for these.
    _NO_ACTIVE_TABLES: frozenset = frozenset({
        "ai_answers", "ai_questions", "ai_audit_answers", "role", "user_roles",
        # audit_form_* pure junction tables — bare 2-column mapping tables
        # (audit_form_id + the other FK only, confirmed live) with no
        # soft-delete tracking at all. Caught live in eval_audit.py's
        # FORMS-04 run: the model reflexively added aff.active = true on
        # audit_form_facilities and got a real Postgres "column does not
        # exist" error — this set lets the cleaner strip it automatically
        # instead of relying on the model to remember the exception.
        "audit_form_facilities", "audit_form_countries", "audit_form_organizations",
        "audit_form_regions", "audit_form_projects",
    })


    def ast_repair(self, sql: str) -> tuple:
        """
        Deterministic AST-level repairs using sqlglot. Every mutation is
        logged in the returned mutations list. Falls back to original SQL
        on parse failure — never corrupts.

        Repairs:
          1. Remove aa.active / aq.active  (columns don't exist)
          2. Inject {alias}.active = true  for soft-delete tables missing it
        """
        if not _HAS_SQLGLOT:
            return sql, []

        mutations: list = []

        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            return sql, []

        # ── 1. Drop hallucinated active columns on aliases that can't have one ──
        # Three cases, unioned into one alias->reason map. deepseek reflexively
        # appends `<alias>.active = true` to every table/CTE it references,
        # regardless of whether that alias actually has (or even IS) a
        # soft-delete-tracked table:
        #  (a) alias's table is in _NO_ACTIVE_TABLES — known real tables with
        #      no soft-delete column (ai_answers, ai_questions, role, etc.)
        #  (b) alias is a CTE whose own SELECT never outputs an "active"
        #      column — pre-aggregated derived tables were never soft-delete
        #      tracked in the first place, so `<cte_alias>.active` can never
        #      resolve to a real column no matter what the CTE's base table is
        #  (c) alias isn't introduced anywhere in the statement at all — same
        #      conservative whole-statement alias union as
        #      _check_undefined_table_aliases() below (see its docstring for
        #      why: false negatives are OK, false positives would block valid
        #      SQL). This is typically leftover from copying a WHERE fragment
        #      out of an unrelated few-shot example without its JOIN.
        _bogus_active_aliases: dict = {}

        for node in tree.find_all(exp.Table):
            if node.name and node.name.lower() in self._NO_ACTIVE_TABLES:
                _bogus_active_aliases[(node.alias or node.name).lower()] = (
                    "table has no active column"
                )

        _introduced: set = set()
        for t in tree.find_all(exp.Table):
            if t.alias:
                _introduced.add(t.alias.lower())
            if t.name:
                _introduced.add(t.name.lower())
        for cte in tree.find_all(exp.CTE):
            if cte.alias:
                _introduced.add(cte.alias.lower())
        for sq in tree.find_all(exp.Subquery):
            if sq.alias:
                _introduced.add(sq.alias.lower())

        # cte.alias is the CTE's OWN declared name (e.g. "po_by_ica" in
        # `WITH po_by_ica AS (...)`) — NOT the alias used to reference it in a
        # JOIN (e.g. the "p" in `LEFT JOIN po_by_ica p ON ...`, which is a
        # separate exp.Table node). Column qualifiers use the JOIN alias, so
        # this must resolve CTE-name -> JOIN-alias(es) via exp.Table, exactly
        # like the _NO_ACTIVE_TABLES loop above does for real tables.
        _cte_names_without_active: set = set()
        for cte in tree.find_all(exp.CTE):
            if cte.alias and isinstance(cte.this, exp.Select):
                has_active = any(
                    e.output_name and e.output_name.lower() == "active"
                    for e in cte.this.expressions
                )
                if not has_active:
                    _cte_names_without_active.add(cte.alias.lower())
        for node in tree.find_all(exp.Table):
            if node.name and node.name.lower() in _cte_names_without_active:
                _bogus_active_aliases.setdefault(
                    (node.alias or node.name).lower(), "CTE has no active column"
                )

        for eq in tree.find_all(exp.EQ):
            if (isinstance(eq.left, exp.Column)
                    and eq.left.name.lower() == "active"
                    and eq.left.table
                    and eq.left.table.lower() not in _introduced):
                _bogus_active_aliases.setdefault(
                    eq.left.table.lower(),
                    "alias not joined/defined anywhere in this query",
                )

        _no_active_aliases: set = set(_bogus_active_aliases)

        if _no_active_aliases:
            def _drop_ai_active(node):
                if (isinstance(node, exp.EQ)
                        and isinstance(node.left, exp.Column)
                        and node.left.name.lower() == "active"
                        and node.left.table
                        and node.left.table.lower() in _no_active_aliases):
                    reason = _bogus_active_aliases.get(
                        node.left.table.lower(), "table has no active column"
                    )
                    mutations.append(
                        f"Removed {node.left.table}.active ({reason})"
                    )
                    return exp.true()
                return node

            tree = tree.transform(_drop_ai_active)
            # Clean up AND(TRUE, x) → x and AND(x, TRUE) → x
            def _simplify_and(node):
                if isinstance(node, exp.And):
                    if isinstance(node.left, exp.Boolean) and node.left.this:
                        return node.right
                    if isinstance(node.right, exp.Boolean) and node.right.this:
                        return node.left
                return node
            tree = tree.transform(_simplify_and)

        # ── 2. Inject active = true for soft-delete tables ────────────────
        def _direct_tables(sel: exp.Select) -> dict:
            """
            alias -> table_name for tables that are NEVER NULL for a matched
            row in this SELECT (FROM + INNER-joined tables) — safe to filter
            via WHERE without changing which rows the query returns.

            A LEFT-joined table's own alias is always excluded (it can be
            NULL). A RIGHT/FULL join makes every table accumulated BEFORE it
            nullable too (that's the side RIGHT/FULL JOIN pads with NULLs),
            so those are dropped from the safe set at that point; a RIGHT
            join's own right-hand table is NOT nullable (all its rows are
            kept) and stays in the safe set, while a FULL join's own table
            is nullable like the LEFT-joined case.
            """
            def _alias(t):
                return (t.alias or t.name).lower() if isinstance(t, exp.Table) and t.name else None
            def _tname(t):
                return t.name.lower() if isinstance(t, exp.Table) and t.name else None

            result = {}
            # sqlglot v20+ uses 'from_' key; older versions use 'from'
            from_clause = sel.args.get("from_") or sel.args.get("from")
            if from_clause and _alias(from_clause.this):
                result[_alias(from_clause.this)] = _tname(from_clause.this)

            for j in (sel.args.get("joins") or []):
                _side = (j.args.get("side") or "").upper()
                if _side == "LEFT":
                    continue  # this join's table is nullable — never mark safe
                if _side in ("RIGHT", "FULL"):
                    result.clear()  # everything accumulated so far is now nullable
                    if _side == "FULL":
                        continue  # this join's own table is nullable too
                _alias_ = _alias(j.this)
                if _alias_:
                    result[_alias_] = _tname(j.this)
            return result

        def _active_aliases(where) -> set:
            """Aliases already filtered by active = true/TRUE/1 (qualified or bare)."""
            found = set()
            if where is None:
                return found
            for eq in where.find_all(exp.EQ):
                if (isinstance(eq.left, exp.Column)
                        and eq.left.name.lower() == "active"):
                    right = eq.right
                    is_true = (
                        (isinstance(right, exp.Boolean) and right.this is True)
                        or (isinstance(right, exp.Literal)
                            and str(right.this).lower() in ("1", "true"))
                    )
                    if is_true:
                        if eq.left.table:
                            found.add(eq.left.table.lower())
                        else:
                            # Bare `active = true` without alias covers all tables
                            # in this scope — mark with sentinel so we skip injection
                            found.add("__bare__")
            return found

        def _inject_active(sel: exp.Select) -> exp.Select:
            direct = _direct_tables(sel)
            existing = _active_aliases(sel.args.get("where"))
            # If bare `active = true` present, skip injection — it already filters
            if "__bare__" in existing:
                return sel
            for alias, tname in sorted(direct.items()):
                if (tname in self._SOFT_DELETE_TABLES
                        and alias not in existing):
                    cond = exp.EQ(
                        this=exp.Column(
                            this=exp.Identifier(this="active"),
                            table=exp.Identifier(this=alias),
                        ),
                        expression=exp.true(),
                    )
                    sel = sel.where(cond)
                    mutations.append(f"Injected {alias}.active = true ({tname})")

            # For LEFT/FULL-joined soft-delete tables (excluded from `direct`
            # above because they're nullable), the active filter belongs in
            # the JOIN's own ON clause, not WHERE. This still hides
            # soft-deleted matches but — unlike a WHERE-clause filter — doesn't
            # turn the outer join into an inner join, so rows with no match at
            # all (genuinely NULL on that side) are still kept.
            #
            # The LLM sometimes writes `alias.active = true` in WHERE itself
            # for one of these tables (it dutifully adds the filter it was
            # told to, without realizing that placement breaks outer-join
            # semantics — e.g. `LEFT JOIN organisation org ON ... WHERE
            # org.active = true` silently drops every row whose org is
            # genuinely NULL). `_relocate_aliases` collects those so they can
            # be stripped from WHERE and re-added to the correct ON clause
            # below, instead of just being left in place as "already handled".
            _relocate_aliases: set = set()
            for j in (sel.args.get("joins") or []):
                _join_side = (j.args.get("side") or "").upper()
                # LEFT: the joined table is nullable. FULL: the joined table is
                # also nullable, and ON-clause injection is equally safe there —
                # a false match (soft-deleted row) just falls back to NULL on
                # that side, same as a genuine non-match. (The FROM-side of a
                # FULL JOIN is nullable too, but that side isn't reachable from
                # this loop — FULL JOIN doesn't appear anywhere in this
                # codebase's generated SQL patterns, so it's not worth the
                # extra complexity to handle here.)
                if _join_side not in ("LEFT", "FULL"):
                    continue
                t = j.this
                if not (isinstance(t, exp.Table) and t.name):
                    continue
                alias = (t.alias or t.name).lower()
                tname = t.name.lower()
                if tname not in self._SOFT_DELETE_TABLES:
                    continue
                on_cond = j.args.get("on")
                already_filtered = on_cond is not None and any(
                    isinstance(eq.left, exp.Column)
                    and eq.left.name.lower() == "active"
                    and eq.left.table and eq.left.table.lower() == alias
                    for eq in on_cond.find_all(exp.EQ)
                )
                if already_filtered:
                    continue
                cond = exp.EQ(
                    this=exp.Column(
                        this=exp.Identifier(this="active"),
                        table=exp.Identifier(this=alias),
                    ),
                    expression=exp.true(),
                )
                j.set("on", exp.And(this=on_cond, expression=cond) if on_cond else cond)
                if alias in existing:
                    _relocate_aliases.add(alias)
                    mutations.append(
                        f"Relocated {alias}.active = true from WHERE into {_join_side} "
                        f"JOIN ON clause ({tname}) — WHERE placement silently drops "
                        f"legitimate no-match rows on an outer join"
                    )
                else:
                    mutations.append(
                        f"Injected {alias}.active = true into {_join_side} JOIN ON clause ({tname})"
                    )

            if _relocate_aliases:
                def _strip_relocated(node):
                    if (isinstance(node, exp.EQ)
                            and isinstance(node.left, exp.Column)
                            and node.left.name.lower() == "active"
                            and node.left.table
                            and node.left.table.lower() in _relocate_aliases):
                        return exp.true()
                    return node
                where = sel.args.get("where")
                if where is not None:
                    new_where = where.this.transform(_strip_relocated)
                    def _simplify(n):
                        if isinstance(n, exp.And):
                            if isinstance(n.left, exp.Boolean) and n.left.this:
                                return n.right
                            if isinstance(n.right, exp.Boolean) and n.right.this:
                                return n.left
                        return n
                    new_where = new_where.transform(_simplify)
                    if isinstance(new_where, exp.Boolean) and new_where.this:
                        sel.set("where", None)
                    else:
                        where.set("this", new_where)
            return sel

        # ── 4. Strip date_trunc wrapping bare string literals in comparisons ─
        # deepseek generates: ir.submitted_on >= date_trunc('month', '2026-03-01')
        # PostgreSQL can't infer the type of the bare string inside date_trunc
        # → raises "function date_trunc(unknown, unknown) is ambiguous".
        # Fix: replace with CAST('YYYY-MM-DD' AS DATE) which is unambiguous.
        # sqlglot parses date_trunc('month', 'literal') as TimestampTrunc.
        def _fix_date_trunc_literal(node):
            if not isinstance(node, exp.TimestampTrunc):
                return node
            date_arg = node.args.get("this")
            if isinstance(date_arg, exp.Literal) and date_arg.is_string:
                mutations.append(
                    f"Replaced date_trunc('month', '{date_arg.this}') "
                    f"→ CAST('{date_arg.this}' AS DATE)"
                )
                return exp.Cast(
                    this=exp.Literal.string(date_arg.this),
                    to=exp.DataType(this=exp.DataType.Type.DATE),
                )
            return node

        tree = tree.transform(_fix_date_trunc_literal)

        tree = tree.transform(
            lambda n: _inject_active(n) if isinstance(n, exp.Select) else n
        )

        # ── 5. Fix sched.scheduled_date → sched.schedule_date ────────────
        # deepseek training bias: uses 'scheduled_date' (past participle) but
        # the actual column on inspection_schedule is 'schedule_date'.
        _sched_aliases: set = set()
        for node in tree.find_all(exp.Table):
            if node.name and node.name.lower() == "inspection_schedule":
                _sched_aliases.add((node.alias or node.name).lower())

        if _sched_aliases:
            def _fix_scheduled_date(node):
                if (isinstance(node, exp.Column)
                        and node.name.lower() == "scheduled_date"
                        and node.table
                        and node.table.lower() in _sched_aliases):
                    mutations.append(
                        f"Renamed {node.table}.scheduled_date → {node.table}.schedule_date"
                    )
                    return exp.Column(
                        this=exp.Identifier(this="schedule_date"),
                        table=exp.Identifier(this=node.table),
                    )
                return node
            tree = tree.transform(_fix_scheduled_date)
        # ── 6. Drop ORDER BY / GROUP BY terms referencing an undefined alias ──
        # Weak SQL-gen models (e.g. small local models) frequently copy a seed
        # example's ORDER BY/GROUP BY clause referencing a SELECT alias they
        # themselves dropped from the SELECT list one line earlier — e.g.
        # `SELECT ... FROM ... ORDER BY last_inspection_date` where
        # `last_inspection_date` was never actually selected. Postgres would
        # reject this with "column ... does not exist" / the validator's
        # UNDEFINED_ALIAS check. Removing the dangling sort/group term is
        # always safe: it loses only ordering/grouping, never row content.
        # Deliberately conservative — UNSAFE to touch SELECT columns, WHERE,
        # HAVING, or JOIN ON referencing an undefined alias (those change
        # which rows/values come back), so those are left for the validator
        # to hard-fail and trigger a real LLM retry instead.
        _introduced_all: set = set()
        for t in tree.find_all(exp.Table):
            if t.alias:
                _introduced_all.add(t.alias.lower())
            if t.name:
                _introduced_all.add(t.name.lower())
        for cte in tree.find_all(exp.CTE):
            if cte.alias:
                _introduced_all.add(cte.alias.lower())
        for sq in tree.find_all(exp.Subquery):
            if sq.alias:
                _introduced_all.add(sq.alias.lower())

        if _introduced_all:
            _undefined_all: set = set()
            for col in tree.find_all(exp.Column):
                if col.table and col.table.lower() not in _introduced_all:
                    _undefined_all.add(col.table.lower())

            if _undefined_all:
                def _refs_undefined(node) -> bool:
                    return any(
                        isinstance(c, exp.Column)
                        and c.table and c.table.lower() in _undefined_all
                        for c in node.find_all(exp.Column)
                    )

                for sel in tree.find_all(exp.Select):
                    order = sel.args.get("order")
                    if order is not None:
                        kept = [o for o in order.expressions if not _refs_undefined(o)]
                        if len(kept) < len(order.expressions):
                            for o in order.expressions:
                                if _refs_undefined(o):
                                    mutations.append(
                                        f"Removed ORDER BY term referencing "
                                        f"undefined alias: {o.sql(dialect='postgres')}"
                                    )
                            sel.set("order", exp.Order(expressions=kept) if kept else None)

                    group = sel.args.get("group")
                    if group is not None:
                        kept_g = [g for g in group.expressions if not _refs_undefined(g)]
                        if len(kept_g) < len(group.expressions):
                            for g in group.expressions:
                                if _refs_undefined(g):
                                    mutations.append(
                                        f"Removed GROUP BY term referencing "
                                        f"undefined alias: {g.sql(dialect='postgres')}"
                                    )
                            sel.set("group", exp.Group(expressions=kept_g) if kept_g else None)

        # ── 7. Inject missing ai_questions join for a dangling aq-style alias ──
        # Weak SQL-gen models sometimes compress the well-established 2-table
        # ai_answers/ai_audit_answers + ai_questions pattern (10+ seed examples
        # in both seed_data.py and audit_seed_data.py: "JOIN ai_questions aq ON
        # aa.element_id = aq.element_id") into a single JOIN, folding the
        # ai_questions filter (e.g. `aq.label = 'Findings / Remarks'`) into the
        # ai_answers/ai_audit_answers JOIN's own ON clause while dropping the
        # ai_questions JOIN entirely — leaving `aq` referenced but never
        # introduced. Confirmed live 2026-08-04 (MULTI-03 regression):
        # qwen2.5-coder repeated this exact malformed pattern 6 times in a row
        # despite the [UNDEFINED_ALIAS] validator blocking it every time,
        # exhausting the retry budget and forcing a fallback to a synthesized
        # prose answer instead of real rows. The element_id join key is a
        # single well-known FK used identically everywhere in this codebase's
        # own seed data, so — unlike the general undefined-alias case (left
        # to the validator/retry loop, see UNSAFE note above on JOIN/WHERE
        # repairs) — this exact, narrow shape is safe to repair
        # deterministically instead of relying on the model to self-correct.
        # Gated to fire only when there's exactly one ai_answers/
        # ai_audit_answers table and exactly one dangling `.label` reference
        # in the same SELECT, so it can't misfire on an unrelated undefined
        # alias (e.g. AP_ALIAS_COLLISION-style mistakes elsewhere).
        #
        # First attempt (2026-08-04) just appended the new ai_questions JOIN
        # at the end of the joins list — it passed this validator's
        # order-agnostic alias-existence check, but confirmed live against a
        # real re-run that Postgres itself still rejected it with "missing
        # FROM-clause entry for table aq": JOIN ON-clause resolution is
        # strictly left-to-right, so a join appearing AFTER the one whose ON
        # clause references its alias is not visible there yet. The fix must
        # both (a) extract the dangling predicate out of the
        # ai_answers/ai_audit_answers join's own ON clause and (b) insert the
        # new ai_questions join immediately after it — not merely add a join
        # somewhere valid in isolation.
        _answers_tables = {"ai_answers", "ai_audit_answers"}

        def _inject_missing_ai_questions_join(sel: exp.Select) -> exp.Select:
            joins = sel.args.get("joins") or []
            if not joins:
                return sel

            from_clause = sel.args.get("from_") or sel.args.get("from")
            _introduced_local: set = set()
            if from_clause and isinstance(from_clause.this, exp.Table):
                if from_clause.this.alias:
                    _introduced_local.add(from_clause.this.alias.lower())
                if from_clause.this.name:
                    _introduced_local.add(from_clause.this.name.lower())
            for j in joins:
                t = j.this
                if isinstance(t, exp.Table):
                    if t.alias:
                        _introduced_local.add(t.alias.lower())
                    if t.name:
                        _introduced_local.add(t.name.lower())

            # Locate the single ai_answers/ai_audit_answers JOIN whose own ON
            # clause carries exactly one predicate referencing a dangling
            # `<alias>.label` column — the confirmed shape of this bug.
            _answers_join_idx = None
            _answers_alias = None
            _dangling_alias = None
            _dangling_pred = None

            for idx, j in enumerate(joins):
                t = j.this
                if not (isinstance(t, exp.Table) and t.name
                        and t.name.lower() in _answers_tables):
                    continue
                on_cond = j.args.get("on")
                if on_cond is None:
                    continue
                preds = list(on_cond.flatten()) if isinstance(on_cond, exp.And) else [on_cond]
                _local_dangling = [
                    p for p in preds
                    if any(
                        isinstance(c, exp.Column) and c.table
                        and c.table.lower() not in _introduced_local
                        and c.name.lower() == "label"
                        for c in p.find_all(exp.Column)
                    )
                ]
                if len(_local_dangling) == 1 and _answers_join_idx is None:
                    _answers_join_idx = idx
                    _answers_alias = (t.alias or t.name).lower()
                    _dangling_pred = _local_dangling[0]
                    _dangling_alias = next(
                        c.table.lower() for c in _dangling_pred.find_all(exp.Column)
                        if c.table and c.table.lower() not in _introduced_local
                    )
                elif _local_dangling:
                    # A second candidate join makes this ambiguous — bail out
                    # conservatively rather than guess which one to fix.
                    return sel

            if _answers_join_idx is None:
                return sel

            # Extract the dangling predicate out of the answers-table join's
            # own ON clause, preserving any sibling conditions.
            _answers_join = joins[_answers_join_idx]
            on_cond = _answers_join.args.get("on")
            remaining = [
                p for p in (list(on_cond.flatten()) if isinstance(on_cond, exp.And) else [on_cond])
                if p is not _dangling_pred
            ]
            if not remaining:
                return sel  # the join's own key condition should always survive
            new_on = remaining[0]
            for p in remaining[1:]:
                new_on = exp.And(this=new_on, expression=p)
            _answers_join.set("on", new_on)

            # New ai_questions join, inserted immediately after the
            # answers-table join so its alias is valid at that point in the
            # left-to-right JOIN chain — carries both the element_id key and
            # the extracted label predicate.
            new_join = exp.Join(
                this=exp.Table(
                    this=exp.Identifier(this="ai_questions"),
                    alias=exp.TableAlias(this=exp.Identifier(this=_dangling_alias)),
                ),
                on=exp.And(
                    this=exp.EQ(
                        this=exp.Column(this=exp.Identifier(this="element_id"),
                                         table=exp.Identifier(this=_answers_alias)),
                        expression=exp.Column(this=exp.Identifier(this="element_id"),
                                               table=exp.Identifier(this=_dangling_alias)),
                    ),
                    expression=_dangling_pred,
                ),
                kind="INNER",
            )
            new_joins = joins[:_answers_join_idx + 1] + [new_join] + joins[_answers_join_idx + 1:]
            sel.set("joins", new_joins)
            mutations.append(
                f"Injected missing JOIN ai_questions {_dangling_alias} ON "
                f"{_answers_alias}.element_id = {_dangling_alias}.element_id "
                f"immediately after {_answers_alias}'s JOIN, moving the dangling "
                f"{_dangling_alias}.label filter there too (preserves left-to-right "
                f"JOIN alias visibility)"
            )
            return sel

        tree = tree.transform(
            lambda n: _inject_missing_ai_questions_join(n) if isinstance(n, exp.Select) else n
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
