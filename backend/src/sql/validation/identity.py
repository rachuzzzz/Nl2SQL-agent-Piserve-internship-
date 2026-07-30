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
          - the outermost SELECT has a GROUP BY (can't inject an ungrouped
            column into an aggregate query without breaking it — the
            plan_ids-style DB round-trip fallback in synthesis.py keeps
            covering this specific gap)
          - no table/alias in this query is in ROW_IDENTITY_TABLES
          - a matching display column is found but its own `<col>__id`
            output name is already taken (don't guess, don't collide)

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

        if top_select.args.get("group") is not None:
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

        if not any(tname in self.ROW_IDENTITY_TABLES for tname in alias_to_table.values()):
            return sql, []

        existing_outputs = {
            e.output_name.lower() for e in top_select.expressions if e.output_name
        }

        mutations: list = []
        to_append: list = []

        for col_expr in top_select.expressions:
            col = col_expr.this if isinstance(col_expr, exp.Alias) else col_expr
            if not isinstance(col, exp.Column) or not col.table:
                continue

            source_alias = col.table.lower()
            table_name = alias_to_table.get(source_alias)
            if not table_name or table_name not in self.ROW_IDENTITY_TABLES:
                continue
            if col.name.lower() != self.ROW_IDENTITY_TABLES[table_name].lower():
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
                        table=exp.Identifier(this=col.table),
                    ),
                    alias=exp.Identifier(this=id_output_name),
                )
            )
            existing_outputs.add(id_output_name.lower())
            mutations.append(
                f"Injected {source_alias}.id AS {id_output_name} ({table_name})"
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
