"""
IdentityMixin: inject_row_identity() — deterministic AST-level enrichment
that captures a "nameable" table's real UUID primary key alongside any
display column (title/name/schedule_id/...) already selected from it, so a
later conversational follow-up can resolve "the third one"/"that plan" to a
real row id instead of re-deriving one from a possibly-duplicated display
string.

Confirmed live 2026-07-30: audit_plan.title has NO uniqueness constraint at
the DB level (only `id` does) — the SAME 7-row "list all plans under this
cycle" result already shown to a user contains 'Delegation testing' TWICE,
as two different plans (different audit_type/category, different UUIDs).
Without this, an ordinal follow-up ("the third one") resolves to the STRING
'Delegation testing', and a subsequent WHERE ap.title = '...' filter would
silently match BOTH rows and merge two different plans' data into one
answer — wrong, with no error.

The injected column is a synthetic, double-underscore-suffixed sibling of
whichever display column it's paired with (e.g. `ap.title AS
audit_plan_title` gets `ap.id AS audit_plan_title__id` appended) — never a
bare `.id` selection, so it can never trip the UUID_IN_SELECT check (see
ast_checks.py's _check_uuid_pk_in_select_ast), which only scans the
OUTERMOST select for exactly that bare shape. This mixin's inject_row_identity()
must only ever be called from generation.py AFTER validate()/validate_shape()
have already passed on the model's own SQL — never before — so the synthetic
column is invisible to every validation pass and never risks being injected
into a draft that's about to be discarded on retry.

The injected column must be stripped before it ever reaches the end user or
the reasoning LLM's own prompt for its NEXT SQL-generation call — see
src/api/response_formatting.py's strip_row_identity_columns() and its call
sites in synthesis.py/chat.py. Nothing in this module is responsible for
that; it only ever adds the column, on the assumption every consumer strips
it before display.

strip_bare_uuid_pk_select() below does the OPPOSITE job: instead of adding a
hidden id, it deterministically REMOVES/replaces a bare `<alias>.id` the
reasoning model selected directly for DISPLAY (not paired with a __id
suffix) — the same mistake the existing UUID_IN_SELECT validation check
catches reactively, but confirmed live 2026-07-30 (5 of 74 eval questions,
reproducible at temperature=0.0 — not sampling noise) to recur often enough
that fixing it deterministically, before validate() ever runs, is strictly
better than costing a retry every time. Investigated first whether this was
caused by conflicting guidance in our own prompts: it isn't — AUDIT_USEFUL_
COLUMNS (schema_introspector.py) already omits "id" from every table's
curated column list, and no seed/worked example anywhere selects a bare id.
This is the model's own generic SQL-writing prior (near-universal "SELECT
id, name, ..." pattern in training data) overriding the specific guidance
it's given, not a self-inflicted contradiction in this codebase's own
prompts.
"""

import re

try:
    import sqlglot
    import sqlglot.expressions as exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False


class IdentityMixin:

    # table_name -> its single-column display value. Deliberately adjacent to
    # (not derived from) engine.py's AUDIT_UUID_PK_DISPLAY_MAP (alias-keyed,
    # audit-only, includes tables with NO display column at all). This map is
    # table-NAME-keyed instead (works regardless of whatever alias a given
    # query happens to use) and only lists tables that have exactly ONE
    # simple display column worth pairing an id with.
    #
    # `users` is deliberately excluded: its display value is a composite
    # (first_name || ' ' || last_name), not a plain column this
    # match-by-column-name algorithm can detect — and people already have a
    # correct, separately-built id-resolution path (disambiguation_candidates
    # + session.py's _ambiguous_duplicate_names STOP-guard), not this bug
    # class.
    ROW_IDENTITY_TABLES: dict = {
        "audit_plan": "title",
        "audit_schedule": "schedule_id",
        "audit_report": "audit_id",
        "audit_corrective_action": "corrective_action_id",
        "inspection_report": "inspection_id",
        "inspection_corrective_action": "corrective_action_id",
        "facility": "name",
        "project": "name",
        "client": "name",
        "organisation": "name",
    }

    # Cycle tables — no name/title column exists at all (confirmed live,
    # documented separately in engine.py's AUDIT_UUID_PK_DISPLAY_MAP and this
    # module's strip_hallucinated_no_substitute_columns). Deliberately
    # separate from ROW_IDENTITY_TABLES: the display value is a synthetic
    # composite of THREE separate SELECT expressions (start_date, end_date,
    # status), not one column or one concat expression like the `users`
    # branch below — requires ALL THREE present as direct SELECT expressions
    # from the SAME aliased table before firing, so a lone `acyc.status`
    # reference (e.g. in an unrelated WHERE-only query) never triggers this.
    _CYCLE_TABLES: dict = {
        "audit_cycle": {"start_date", "end_date", "status"},
        "inspection_cycle": {"start_date", "end_date", "status"},
    }

    def inject_row_identity(self, sql: str) -> tuple:
        """
        For every direct SELECT expression in the OUTERMOST select that is a
        (optionally aliased) column matching a ROW_IDENTITY_TABLES table's
        registered display column, append a sibling `<alias>.id AS
        <output_name>__id`. Runs per-expression, independently — a query
        joining `review` + `audit_plan` and selecting a review column
        alongside `ap.title` gets an id injected for `ap.title` specifically,
        never a single FROM-privileged id for the whole row.

        Skips entirely (returns the SQL unchanged, no mutations) if:
          - sqlglot isn't installed, or the SQL doesn't parse
          - no table/alias in this query is in ROW_IDENTITY_TABLES
          - a matching display column is found but its own `<col>__id`
            output name is already taken (don't guess, don't collide)

        A GROUP BY no longer blocks injection wholesale — only injection
        for a SPECIFIC alias whose `.id` isn't itself already part of the
        GROUP BY clause (adding `<alias>.id` to SELECT would then be a real
        "column must appear in the GROUP BY clause" error). When `<alias>.id`
        IS already grouped (e.g. `GROUP BY u.id, u.first_name, u.last_name`
        — a common shape for "list all X with a COUNT(...)"), injecting a
        second SELECT expression referencing that same already-grouped
        column is always valid, so it proceeds normally for that alias.
        Confirmed live 2026-08-03: "list all auditors" (grouped by
        u.id/first_name/last_name, with a COUNT(schedule) aggregate) got
        NO id-injection at all under the old blanket skip, despite u.id
        being safely groupable — a later "the 6th auditor" ordinal
        follow-up had no id to resolve to, fell back to re-searching by
        name from scratch, and surfaced unrelated same-named people instead
        of the specific auditor already shown.

        Returns (sql, mutations) — mutations is a list of human-readable
        strings for a "[row-identity] ..." debug log line, empty if nothing
        was injected.
        """
        if not _HAS_SQLGLOT:
            return sql, []

        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            return sql, []

        # Navigate to the outermost SELECT only — same restriction as
        # ast_checks.py's _check_uuid_pk_in_select_ast, and for the same
        # reason: a nested subquery's own `id` selection is a legitimate JOIN
        # key, not a display value, and must never be touched here.
        top_select = None
        if isinstance(tree, exp.Select):
            top_select = tree
        elif isinstance(tree, exp.With):
            final = tree.this
            if isinstance(final, exp.Select):
                top_select = final
            elif isinstance(final, exp.Subquery):
                inner = final.this
                if isinstance(inner, exp.Select):
                    top_select = inner

        if top_select is None:
            return sql, []

        # alias -> table_name for EVERY table in FROM/JOIN, regardless of
        # join side. Unlike cleaner.py's _direct_tables (which deliberately
        # excludes LEFT-joined tables, since it's deciding what's safe to
        # WHERE-filter on), a LEFT-joined nameable table's display column is
        # just as valid a row identity as one from an inner-joined table —
        # it's the display column's own presence in SELECT that gates
        # injection below, not join safety.
        alias_to_table: dict = {}
        from_clause = top_select.args.get("from_") or top_select.args.get("from")
        if (from_clause is not None and isinstance(from_clause.this, exp.Table)
                and from_clause.this.name):
            t = from_clause.this
            alias_to_table[(t.alias or t.name).lower()] = t.name.lower()
        for j in (top_select.args.get("joins") or []):
            t = j.this
            if isinstance(t, exp.Table) and t.name:
                alias_to_table[(t.alias or t.name).lower()] = t.name.lower()

        if not any(tname in self.ROW_IDENTITY_TABLES or tname == "users"
                   or tname in self._CYCLE_TABLES
                   for tname in alias_to_table.values()):
            return sql, []

        # Single-table queries often omit an explicit alias entirely (e.g.
        # `SELECT id, start_date, status FROM audit_cycle` — no "acyc"
        # anywhere, every column's .table is empty). When there's exactly
        # one table in the whole FROM/JOIN clause, an unqualified column can
        # only ever mean that one table — safe to treat it as implicitly
        # qualified by it (Postgres itself allows referencing
        # <table_name>.col even with no explicit alias declared). A 2+-table
        # query is NOT covered by this — a bare column there could belong to
        # any of them, so col.table is still required in that case.
        # Confirmed live 2026-08-03: a "list all cycles" query came back
        # with zero table qualifiers at all, so none of the matching below
        # fired, row_titles stayed empty for that turn, and a follow-up
        # ordinal reference ("the 5th one") fell back to the SQL-gen model
        # guessing — first a real Postgres error (`acyc.id = 5`, uuid =
        # integer), then a working-but-fragile ROW_NUMBER() workaround.
        _sole_alias = next(iter(alias_to_table)) if len(alias_to_table) == 1 else None

        # GROUP BY safety: collect "<alias>.<col>" for every plain column
        # reference in the GROUP BY clause (same _sole_alias fallback as
        # SELECT columns above, for an unaliased single-table GROUP BY).
        # `_group_clause is None` means no GROUP BY at all — injection is
        # then always safe for every alias, same as before this fix.
        _group_clause = top_select.args.get("group")
        _grouped_cols: set = set()
        if _group_clause is not None:
            for _g in _group_clause.expressions:
                if isinstance(_g, exp.Column) and _g.name:
                    _g_alias = (_g.table or _sole_alias or "").lower()
                    if _g_alias:
                        _grouped_cols.add(f"{_g_alias}.{_g.name.lower()}")

        def _id_safe_to_inject(alias: str) -> bool:
            return _group_clause is None or f"{alias}.id" in _grouped_cols

        existing_outputs = {
            e.output_name.lower() for e in top_select.expressions if e.output_name
        }

        mutations: list = []
        to_append: list = []

        for col_expr in top_select.expressions:
            col = col_expr.this if isinstance(col_expr, exp.Alias) else col_expr
            if not isinstance(col, exp.Column):
                continue

            source_alias = (col.table or _sole_alias or "").lower()
            if not source_alias:
                continue
            table_name = alias_to_table.get(source_alias)
            if not table_name or table_name not in self.ROW_IDENTITY_TABLES:
                continue
            if col.name.lower() != self.ROW_IDENTITY_TABLES[table_name].lower():
                continue
            if not _id_safe_to_inject(source_alias):
                continue

            output_alias = col_expr.output_name
            if not output_alias:
                continue
            id_output_name = f"{output_alias}__id"
            if id_output_name.lower() in existing_outputs:
                continue  # already present (or would collide) — skip, don't guess

            to_append.append(
                exp.Alias(
                    this=exp.Column(
                        this=exp.Identifier(this="id"),
                        table=exp.Identifier(this=source_alias),
                    ),
                    alias=exp.Identifier(this=id_output_name),
                )
            )
            existing_outputs.add(id_output_name.lower())
            mutations.append(
                f"Injected {source_alias}.id AS {id_output_name} ({table_name})"
            )

        # ── People (users) — composite full-name display, no single column ──
        # `users` is deliberately excluded from ROW_IDENTITY_TABLES above
        # (its display value is first_name || ' ' || last_name, a
        # concatenation, not a plain column the bare-column loop above can
        # match) — but that concatenation still unambiguously identifies one
        # real person per row. Confirmed live 2026-07-31: a follow-up "the
        # 2nd auditor in the list" had no id to resolve to for exactly this
        # reason, so instead of reusing the specific person the prior turn
        # had already pinned down, the model had to re-search auditor_name
        # across the WHOLE users table from scratch — surfacing other people
        # who share that name and were never in the original list at all,
        # then asking the user to pick one. Injecting <alias>.id AS
        # <output>__id here lets a later ordinal resolve straight to that
        # one already-identified person instead of re-searching.
        # Deliberately narrow: exactly 2 columns, both first_name/last_name
        # (either order), both from the same users-aliased table — the one
        # concatenation shape used throughout this codebase's own seed
        # examples — to avoid misfiring on some unrelated 2-column concat.
        for col_expr in top_select.expressions:
            if not isinstance(col_expr, exp.Alias):
                continue
            output_alias = col_expr.output_name
            if not output_alias:
                continue
            id_output_name = f"{output_alias}__id"
            if id_output_name.lower() in existing_outputs:
                continue
            cols_used = list(col_expr.this.find_all(exp.Column))
            if len(cols_used) != 2:
                continue
            aliases_used = {(c.table or _sole_alias or "").lower() for c in cols_used}
            aliases_used.discard("")
            names_used = {c.name.lower() for c in cols_used}
            if len(aliases_used) != 1 or names_used != {"first_name", "last_name"}:
                continue
            src_alias = next(iter(aliases_used))
            if alias_to_table.get(src_alias) != "users":
                continue
            if not _id_safe_to_inject(src_alias):
                continue
            to_append.append(
                exp.Alias(
                    this=exp.Column(this=exp.Identifier(this="id"),
                                     table=exp.Identifier(this=src_alias)),
                    alias=exp.Identifier(this=id_output_name),
                )
            )
            existing_outputs.add(id_output_name.lower())
            mutations.append(
                f"Injected {src_alias}.id AS {id_output_name} (users, composite name)"
            )

        # ── Cycles (audit_cycle/inspection_cycle) — composite date+status
        # display, no single output_name to suffix "__id" onto (the display
        # spans 3 separate SELECT expressions, not one). Emits a FIXED,
        # alias-namespaced sibling instead: <alias>_cycle_row__id.
        # synthesis.py reads it back via this exact suffix, gated on its
        # presence so it can never misfire on some unrelated joined table's
        # own "status" column.
        cols_by_alias: dict = {}
        for col_expr in top_select.expressions:
            col = col_expr.this if isinstance(col_expr, exp.Alias) else col_expr
            if isinstance(col, exp.Column):
                _alias = (col.table or _sole_alias or "").lower()
                if _alias:
                    cols_by_alias.setdefault(_alias, set()).add(col.name.lower())

        for src_alias, table_name in alias_to_table.items():
            required = self._CYCLE_TABLES.get(table_name)
            if not required or not required.issubset(cols_by_alias.get(src_alias, set())):
                continue
            id_output_name = f"{src_alias}_cycle_row__id"
            if id_output_name.lower() in existing_outputs:
                continue
            if not _id_safe_to_inject(src_alias):
                continue
            to_append.append(
                exp.Alias(
                    this=exp.Column(this=exp.Identifier(this="id"),
                                     table=exp.Identifier(this=src_alias)),
                    alias=exp.Identifier(this=id_output_name),
                )
            )
            existing_outputs.add(id_output_name.lower())
            mutations.append(
                f"Injected {src_alias}.id AS {id_output_name} ({table_name}, composite date+status)"
            )

        if not to_append:
            return sql, []

        for node in to_append:
            top_select.expressions.append(node)

        try:
            repaired = tree.sql(dialect="postgres", pretty=False)
            if repaired and not repaired.rstrip().endswith(";"):
                repaired = repaired.rstrip() + ";"
            return repaired, mutations
        except Exception:
            return sql, []

    # Tables where answer_text is a free-text form answer and submission_id
    # is the row's own directly-filterable identifier (not a UUID primary
    # key needing a __id sibling — submission_id IS the real, joinable value,
    # see synthesis.py's row_titles/row_ids "Priority 0" capture).
    _OBSERVATION_ANSWER_TABLES: frozenset = frozenset({"ai_audit_answers", "ai_answers"})

    def inject_observation_submission_id(self, sql: str) -> tuple:
        """
        Ensures submission_id rides along whenever answer_text is selected
        from ai_audit_answers/ai_answers, so a later ordinal follow-up
        ("the first finding") has a real id to resolve to instead of falling
        back to a generic ORDER BY/LIMIT re-derivation.

        Confirmed live 2026-07-31: the SAME question ("observations for audit
        X") generated SQL with submission_id selected on one run and without
        it on another — the reasoning/SQL-gen models don't reliably include
        it on their own even though the worked example shows it, so
        synthesis.py's row_titles/row_ids capture (which depends on
        submission_id being present) only worked opportunistically. This
        makes it deterministic instead, mirroring inject_row_identity()'s
        own rationale but for a column that's already a directly-filterable
        value, not a primary key needing a synthetic __id sibling.

        Same skip conditions as inject_row_identity(): no sqlglot, unparsable
        SQL, or a GROUP BY on the outermost SELECT.
        """
        if not _HAS_SQLGLOT:
            return sql, []
        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            return sql, []

        top_select = None
        if isinstance(tree, exp.Select):
            top_select = tree
        elif isinstance(tree, exp.With):
            final = tree.this
            if isinstance(final, exp.Select):
                top_select = final
            elif isinstance(final, exp.Subquery):
                inner = final.this
                if isinstance(inner, exp.Select):
                    top_select = inner
        if top_select is None:
            return sql, []
        if top_select.args.get("group") is not None:
            return sql, []

        alias_to_table: dict = {}
        from_clause = top_select.args.get("from_") or top_select.args.get("from")
        if (from_clause is not None and isinstance(from_clause.this, exp.Table)
                and from_clause.this.name):
            t = from_clause.this
            alias_to_table[(t.alias or t.name).lower()] = t.name.lower()
        for j in (top_select.args.get("joins") or []):
            t = j.this
            if isinstance(t, exp.Table) and t.name:
                alias_to_table[(t.alias or t.name).lower()] = t.name.lower()

        _obs_aliases = {
            a for a, tname in alias_to_table.items()
            if tname in self._OBSERVATION_ANSWER_TABLES
        }
        if not _obs_aliases:
            return sql, []

        _answer_text_alias = None
        _has_submission_id = False
        for col_expr in top_select.expressions:
            col = col_expr.this if isinstance(col_expr, exp.Alias) else col_expr
            if not isinstance(col, exp.Column) or not col.table:
                continue
            src_alias = col.table.lower()
            if src_alias not in _obs_aliases:
                continue
            if col.name.lower() == "answer_text":
                _answer_text_alias = src_alias
            elif col.name.lower() == "submission_id":
                _has_submission_id = True

        if not _answer_text_alias or _has_submission_id:
            return sql, []

        top_select.expressions.append(
            exp.Column(this=exp.Identifier(this="submission_id"),
                       table=exp.Identifier(this=_answer_text_alias))
        )
        mutations = [f"Injected {_answer_text_alias}.submission_id (observation row identity)"]

        try:
            repaired = tree.sql(dialect="postgres", pretty=False)
            if repaired and not repaired.rstrip().endswith(";"):
                repaired = repaired.rstrip() + ";"
            return repaired, mutations
        except Exception:
            return sql, []

    def strip_bare_uuid_pk_select(self, sql: str) -> tuple:
        """
        Deterministically fix a bare `<alias>.id` selected for DISPLAY on any
        alias in AUDIT_UUID_PK_DISPLAY_MAP (engine.py) — the exact mistake
        the UUID_IN_SELECT validation check exists to catch, done here
        instead so it never costs a retry in the first place. Audit-domain
        only (gated on self.domain) — AUDIT_UUID_PK_DISPLAY_MAP's aliases
        (ap/asch/arep/aca/acyc/po) only mean anything in that domain's own
        seed examples/table conventions.

        Two outcomes, matching what the BLOCK rule's own message already
        recommends:
          - alias has NO display substitute (acyc, po) → the bare id is
            simply removed from the SELECT list.
          - alias HAS one (ap → ap.title, asch → asch.schedule_id, ...) →
            the bare id is replaced with that display column instead
            (or just removed if that column is already selected elsewhere,
            to avoid a duplicate).

        Never touches a `__id`-suffixed output (inject_row_identity()'s own
        legitimate columns, always paired with a display value already) or
        leaves a SELECT with zero expressions (an extremely unlikely edge
        case — e.g. `SELECT ap.id FROM audit_plan ap` — where removing the
        only expression would produce invalid SQL; skipped entirely rather
        than risk that).
        """
        if not _HAS_SQLGLOT or self.domain != "audit":
            return sql, []

        pk_map = getattr(self, "AUDIT_UUID_PK_DISPLAY_MAP", None)
        if not pk_map:
            return sql, []

        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            return sql, []

        if not isinstance(tree, exp.Select):
            return sql, []

        # What's already selected (alias, column) — so a swapped-in display
        # column never gets duplicated if it's already present elsewhere.
        already_selected: set = set()
        for e in tree.expressions:
            col = e.this if isinstance(e, exp.Alias) else e
            if isinstance(col, exp.Column) and col.table:
                already_selected.add((col.table.lower(), col.name.lower()))

        mutations: list = []
        new_expressions: list = []

        for e in tree.expressions:
            col = e.this if isinstance(e, exp.Alias) else e
            if not (isinstance(col, exp.Column) and col.name.lower() == "id" and col.table):
                new_expressions.append(e)
                continue

            alias = col.table.lower()
            if alias not in pk_map:
                new_expressions.append(e)
                continue

            output_name = e.output_name or ""
            if output_name.lower().endswith("__id"):
                new_expressions.append(e)  # inject_row_identity()'s own column — never touch
                continue

            display = pk_map[alias]
            if not display:
                mutations.append(
                    f"Removed bare {alias}.id from SELECT "
                    f"(no display substitute for this table)"
                )
                continue

            # display is like "ap.title" or "asch.schedule_id (the human-
            # readable varchar display code)" — extract just the column ref.
            m = re.match(r"(\w+)\.(\w+)", display)
            if not m:
                new_expressions.append(e)
                continue
            disp_alias, disp_col = m.group(1).lower(), m.group(2).lower()

            if (disp_alias, disp_col) in already_selected:
                mutations.append(
                    f"Removed bare {alias}.id from SELECT "
                    f"({disp_alias}.{disp_col} already selected)"
                )
                continue

            new_alias_name = output_name if (output_name and output_name.lower() != "id") else disp_col
            new_expressions.append(
                exp.Alias(
                    this=exp.Column(
                        this=exp.Identifier(this=disp_col),
                        table=exp.Identifier(this=disp_alias),
                    ),
                    alias=exp.Identifier(this=new_alias_name),
                )
            )
            already_selected.add((disp_alias, disp_col))
            mutations.append(
                f"Replaced bare {alias}.id with {disp_alias}.{disp_col} AS {new_alias_name}"
            )

        if not mutations or not new_expressions:
            return sql, []

        tree.set("expressions", new_expressions)

        try:
            repaired = tree.sql(dialect="postgres", pretty=False)
            if repaired and not repaired.rstrip().endswith(";"):
                repaired = repaired.rstrip() + ";"
            return repaired, mutations
        except Exception:
            return sql, []

    # table_name -> (HALLUCINATED_*_COLUMNS attribute name on SQLValidator, domain gate)
    # These three specific maps (out of engine.py's full HALLUCINATED_* family) are
    # exactly the "column doesn't exist AT ALL, no substitute" case — every entry's
    # own hint says so (e.g. "cycles have no name ... identify them by start_date/
    # end_date/status instead") — unlike e.g. HALLUCINATED_IR_COLUMNS, where the
    # hint is "JOIN facility fac ON ... -> fac.name" (needs a JOIN ADDED, not just a
    # column dropped — too structurally invasive to auto-fix blindly). That's what
    # makes these three safe to mechanically strip from SELECT; the others aren't.
    _NO_SUBSTITUTE_TABLES = (
        ("audit_cycle", "HALLUCINATED_AC_COLUMNS", "audit"),
        ("auditor_portfolio", "HALLUCINATED_PO_COLUMNS", "audit"),
        ("inspection_cycle", "HALLUCINATED_IC_COLUMNS", "inspection"),
    )

    def strip_hallucinated_no_substitute_columns(self, sql: str) -> tuple:
        """
        Deterministically removes a hallucinated display-name column (e.g.
        `acyc.name AS cycle_name`, `po.portfolio_name`) from the outermost
        SELECT when it's one of the columns engine.py's HALLUCINATED_AC_
        COLUMNS/HALLUCINATED_PO_COLUMNS/HALLUCINATED_IC_COLUMNS maps already
        know don't exist at all on that table — same "fix it ourselves
        instead of costing a retry" principle as strip_bare_uuid_pk_select
        above.

        Confirmed live and already documented as high-cost BEFORE this fix
        existed: engine.py's own comments cite eval_audit.py's PLAN-08 run
        burning 7 LLM calls / 21s repeatedly confabulating audit_cycle.name,
        and PORT-03 hallucinating auditor_portfolio.portfolio_name TWICE —
        once before AND once again after a get_schema round-trip confirmed
        the real column list, despite a correct in-context seed example.

        Deliberately scoped to the SELECT list only — NOT to WHERE/GROUP BY/
        ORDER BY, even though the reactive WRONG_COLUMN check (engine.py)
        scans the whole statement for these. Dropping a bad SELECT column is
        always safe (never changes which rows come back). Silently dropping
        a WHERE condition on a hallucinated column could silently WIDEN the
        result set to the wrong rows instead of erroring — a wrong answer
        with no error is worse than a caught error that costs a retry, so
        that case is deliberately left to the existing reactive check +
        retry instead of being auto-"fixed" here.
        """
        if not _HAS_SQLGLOT:
            return sql, []

        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            return sql, []

        if not isinstance(tree, exp.Select):
            return sql, []

        alias_to_table: dict = {}
        from_clause = tree.args.get("from_") or tree.args.get("from")
        if (from_clause is not None and isinstance(from_clause.this, exp.Table)
                and from_clause.this.name):
            t = from_clause.this
            alias_to_table[(t.alias or t.name).lower()] = t.name.lower()
        for j in (tree.args.get("joins") or []):
            t = j.this
            if isinstance(t, exp.Table) and t.name:
                alias_to_table[(t.alias or t.name).lower()] = t.name.lower()

        active_maps: dict = {}  # alias -> {col_name: hint}
        for table_name, attr_name, domain in self._NO_SUBSTITUTE_TABLES:
            if self.domain != domain:
                continue
            col_map = getattr(self, attr_name, None)
            if not col_map:
                continue
            for alias, tname in alias_to_table.items():
                if tname == table_name:
                    active_maps[alias] = col_map

        if not active_maps:
            return sql, []

        def _is_hallucinated(target) -> bool:
            return (isinstance(target, exp.Column) and bool(target.table)
                    and target.table.lower() in active_maps
                    and target.name.lower() in active_maps[target.table.lower()])

        mutations: list = []
        new_expressions: list = []
        select_changed = False
        for e in tree.expressions:
            col = e.this if isinstance(e, exp.Alias) else e
            if _is_hallucinated(col):
                mutations.append(
                    f"Removed hallucinated {col.table.lower()}.{col.name.lower()} from SELECT "
                    f"({active_maps[col.table.lower()][col.name.lower()]})"
                )
                select_changed = True
                continue
            new_expressions.append(e)

        if select_changed:
            if not new_expressions:
                return sql, []  # would empty the SELECT list — bail out entirely
            tree.set("expressions", new_expressions)

        # A hallucinated column is just as invalid in GROUP BY/ORDER BY as in
        # SELECT — checked independently here, not only as a side effect of
        # a SELECT-list removal, since a model can reference one in GROUP
        # BY/ORDER BY without ever selecting it at all. Confirmed live
        # 2026-07-30 (CYCLE-05): removing acyc.name from SELECT alone still
        # left `GROUP BY acyc.id, acyc.name, acyc.status` referencing it, so
        # validation still failed on the orphaned GROUP BY entry even though
        # the SELECT-list fix was itself correct. Safe to clean up here
        # (unlike a WHERE-clause reference — see this method's docstring): a
        # GROUP BY/ORDER BY entry for a column that cannot exist was never
        # contributing valid grouping/ordering semantics to begin with, so
        # dropping it doesn't change which rows qualify.
        for clause_name in ("group", "order"):
            clause = tree.args.get(clause_name)
            if clause is None:
                continue
            kept = []
            for e in clause.expressions:
                target = e.this if isinstance(e, exp.Ordered) else e
                if _is_hallucinated(target):
                    mutations.append(
                        f"Removed hallucinated {target.table.lower()}.{target.name.lower()} "
                        f"from {clause_name.upper()} BY "
                        f"({active_maps[target.table.lower()][target.name.lower()]})"
                    )
                    continue
                kept.append(e)
            if len(kept) != len(clause.expressions):
                if kept:
                    clause.set("expressions", kept)
                else:
                    tree.set(clause_name, None)

        if not mutations:
            return sql, []

        try:
            repaired = tree.sql(dialect="postgres", pretty=False)
            if repaired and not repaired.rstrip().endswith(";"):
                repaired = repaired.rstrip() + ";"
            return repaired, mutations
        except Exception:
            return sql, []
