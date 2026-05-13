"""
Schema Introspector
====================
Introspects the database at startup and builds the schema text that gets
injected into prompts.

Covers two domains:
  - INSPECTION WORKFLOW tables (inspection_report, inspection_corrective_action, etc.)
  - AI FLAT TABLES (ai_question, ai_answer)

The fb_* form-builder tables are intentionally excluded — the ai_* tables
are the public interface for all form question/answer data.
"""

from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

ENUM_COLUMNS = {
    "inspection_corrective_action.responsible": [
        "CLIENT", "INTERNAL_OPERATIONS", "SUB_CONTRACTOR",
    ],
    "inspection_corrective_action.status": [
        "OPEN", "CLOSED", "OVERDUE", "CLOSE_WITH_DEFERRED",
    ],
    "inspection_corrective_action.pending_with": [],
    "inspection_report.status": [
        "DRAFT", "SUBMITTED", "CLOSED", "UNDER_REVIEW", "RETURN_FOR_MODIFICATION",
    ],
    "inspection_schedule.status": [
        "PENDING", "ONGOING", "COMPLETED", "OVERDUE", "CANCELLED",
    ],
    "inspection_cycle.status": [],
}

BUSINESS_RULES = [
    "DRAFT inspections usually have NULL scores. Always add WHERE status != 'DRAFT' "
    "when querying inspection_score, unless the user explicitly asks about drafts.",
    "NEVER use SELECT *. Always specify the columns the user needs.",
    "NEVER use SQL reserved words as table aliases (is, as, in, on, by, do, if).",
    "Use ILIKE for text matching, never LIKE.",
    "corrective_action_id is a human-readable ID like '2026/01/ST064/INS001_MA001', not a UUID.",
    "inspection_corrective_action.inspection_id is the PRIMARY FK to inspection_report.id.",
    "Use LEFT JOIN (not INNER JOIN) for optional lookups like inspection_sub_type, "
    "entity, and inspection_report_remark.",
    "VOCABULARY — corrective action status: "
    "'open' = 'OPEN'. 'overdue' = 'OVERDUE'. "
    "'closed'/'resolved' = status IN ('CLOSED','CLOSE_WITH_DEFERRED').",
    "VOCABULARY — inspection report status: "
    "'completed'/'done' = 'CLOSED'. 'pending review' = 'UNDER_REVIEW'. "
    "'returned'/'rejected' = 'RETURN_FOR_MODIFICATION'.",
    "For form answers use ai_answers (plural). For question labels use ai_questions (plural). "
    "JOIN via: ai_answers.element_id = ai_questions.element_id.",
    "JOIN ai_answers to inspection_report via: ai_answers.inspection_report_id = inspection_report.id.",
    "CRITICAL — inspection_report has TWO id columns: "
    "'id' is the UUID primary key — NEVER select this for display (outputs UUID garbage). "
    "'inspection_id' is the human-readable varchar like '2026/04/ST001/INS001' — ALWAYS use "
    "ir.inspection_id (not ir.id) when showing the inspection identifier to the user.",
]

TABLE_RELATIONSHIPS = """\
WHERE CONCEPTS LIVE:
  - Inspection TYPE names → inspection_type.name
  - Inspector names → users.first_name + users.last_name (JOIN via inspector_user_id)
  - Facility names → facility.name (JOIN via facility_id)
  - Project names → project.name (JOIN via project_id)
  - Client names → client.name (JOIN via client_id)
  - Scores, status, dates → inspection_report
  - Causes, costs, overdue → inspection_corrective_action
  - Form question labels → ai_questions.label  (NOT question_label)
  - Form text answers → ai_answers.answer_text
  - Form numeric answers → ai_answers.answer_numeric
  - Form scores → ai_answers.score

HOW TABLES CONNECT (inspection_report is the hub):
  inspection_report.inspection_type_id → inspection_type.id
  inspection_report.facility_id → facility.id
  inspection_report.project_id → project.id
  inspection_report.client_id → client.id
  inspection_report.inspector_user_id → users.id
  inspection_report.cycle_id → inspection_cycle.id
  inspection_report.schedule_id → inspection_schedule.id
  inspection_corrective_action.inspection_id → inspection_report.id
  ai_answers.inspection_report_id → inspection_report.id
  ai_answers.element_id → ai_questions.element_id

CRITICAL ai_* JOIN PATTERN:
  FROM ai_answers aa
  LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id
  JOIN inspection_report ir ON aa.inspection_report_id = ir.id
  WHERE aq.label ILIKE '%keyword%'

MULTI-HOP EXAMPLES:
  "answers at Al Ghadeer":
    FROM ai_answers aa
    LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id
    JOIN inspection_report ir ON aa.inspection_report_id = ir.id
    JOIN facility fac ON ir.facility_id = fac.id
    WHERE fac.name ILIKE '%Al Ghadeer%'

  "corrective actions for safety inspections":
    FROM inspection_corrective_action ica
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    JOIN inspection_type it ON ir.inspection_type_id = it.id
    WHERE it.name ILIKE '%safety%'

RULE: TYPE names (Safety, PPE, Hygiene, etc.) live ONLY in inspection_type.name.
"""

LOOKUP_TABLES = {
    "users":               {"name_expr": "first_name || ' ' || last_name", "name_col": None},
    "facility":            {"name_expr": None, "name_col": "name"},
    "project":             {"name_expr": None, "name_col": "name"},
    "client":              {"name_expr": None, "name_col": "name"},
    "entity":              {"name_expr": None, "name_col": "name"},
    "organisation":        {"name_expr": None, "name_col": "name"},
    "inspection_type":     {"name_expr": None, "name_col": "name"},
    "inspection_sub_type": {"name_expr": None, "name_col": "name"},
    # Frequency/cadence lookup — defines how often inspections happen in a portfolio
    # columns: id, name, repeat_count (int), repeat_interval (int), repeat_unit (DAY/WEEK/MONTH)
    # e.g. 'Quarterly Twice' = repeat_count=2, repeat_interval=3, repeat_unit=MONTH
    # JOIN via: inspector_portfolio_details.frequency_definition_id → frequency_definition.id
    "frequency_definition": {"name_expr": None, "name_col": "name"},
    # Risk/impact lookup tables
    "risk_level":          {"name_expr": None, "name_col": "name"},
    "impact":              {"name_expr": None, "name_col": "name"},
}

FK_RESOLUTION = {
    "inspector_user_id":     ("users", "u",   "u.first_name || ' ' || u.last_name AS inspector_name"),
    "inspectee_user_id":     ("users", "insp","insp.first_name || ' ' || insp.last_name AS inspectee_name"),
    "facility_id":           ("facility", "fac", "fac.name AS facility_name"),
    "project_id":            ("project",  "proj", "proj.name AS project_name"),
    "client_id":             ("client",   "cl",   "cl.name AS client_name"),
    "entity_id":             ("entity",   "ent",  "ent.name AS entity_name"),
    "inspection_type_id":    ("inspection_type",     "it",  "it.name AS inspection_type_name"),
    "inspection_sub_type_id":("inspection_sub_type", "ist", "ist.name AS inspection_subtype_name"),
    # Frequency/cadence FK — inspector_portfolio_details only
    "frequency_definition_id": (
        "frequency_definition", "fd",
        "fd.name AS frequency_name, fd.repeat_count, fd.repeat_interval, fd.repeat_unit"
    ),
    # Risk and impact FKs — inspection_corrective_action only
    "risk_level_id":         ("risk_level", "rl",  "rl.name AS risk_level_name"),
    "impact_id":             ("impact",     "imp", "imp.name AS impact_name"),
    # Schedule → portfolio_details FK (critical: the column is portfolio_details_id, not portfolio_id)
    "portfolio_details_id":  ("inspector_portfolio_details", "ipd", "ipd.id"),
    # inspection_schedule → inspection_cycle FK
    "inspection_cycle_id":   ("inspection_cycle", "ic", "ic.id"),
    # inspector_portfolio → cycle
    "cycle_id":              ("inspection_cycle", "ic", "ic.id"),
}

OPTIONAL_FKS = {
    "inspection_sub_type_id", "entity_id", "close_submission_id",
    "closed_by_user_id", "schedule_id",
}

INSPECTION_TABLES = [
    "inspection_report",
    "inspection_corrective_action",
    "inspection_schedule",
    "inspection_cycle",
    "inspection_type",
    "inspection_sub_type",
    "inspector_portfolio",
    "inspector_portfolio_details",
    "inspectioncaprogress_tracking",
    "inspection_report_remark",
    "accompanying_inspectors",
    "additional_inspectees",
]

AI_TABLES = [
    "ai_questions",
    "ai_answers",
]

# CRITICAL: inspection_report has TWO id-like columns:
#   id             → UUID primary key  — NEVER select this for display, always UUID garbage
#   inspection_id  → varchar like '2026/04/ST001/INS001' — ALWAYS use this for display
# deepseek commonly selects ir.id when it should select ir.inspection_id. Listing
# inspection_id first in USEFUL_COLUMNS reinforces the correct choice.
USEFUL_COLUMNS = {
    "inspection_report": [
        "inspection_id",   # varchar '2026/04/ST001/INS001' — use this NOT ir.id
        "inspection_score", "gp_score", "status",
        "submitted_on", "total_inspection_hours",
    ],
    "inspection_corrective_action": [
        "corrective_action_id", "cause", "correction", "corrective_action",
        "responsible", "status", "progress_stage", "capex", "opex",
        "target_close_out_date", "age",
    ],
    "ai_questions": [
        "element_id", "label", "module_name", "module_id", "entity_type",
    ],
    "ai_answers": [
        "inspection_id", "inspection_report_id", "element_id", "module_name",
        "answer_text", "answer_numeric", "score", "score_type", "max_score",
        "submitted_on",
    ],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TableSchema:
    name: str
    columns: list
    foreign_keys: list
    primary_key: list


@dataclass
class DatabaseSchema:
    tables: dict = field(default_factory=dict)       # inspection_* tables
    ai_tables: dict = field(default_factory=dict)    # ai_question, ai_answer
    lookup_tables: dict = field(default_factory=dict)
    enum_values: dict = field(default_factory=dict)  # table → list of name values (dynamic)
    _system_cache: Optional[str] = field(default=None, repr=False)
    _sql_cache: Optional[str] = field(default=None, repr=False)

    def for_system_prompt(self) -> str:
        if not self._system_cache:
            self._system_cache = self._build_system_block()
        return self._system_cache

    def for_sql_prompt(self) -> str:
        if not self._sql_cache:
            self._sql_cache = self._build_sql_block()
        return self._sql_cache

    def for_schema_hint(self, table_names: list) -> str:
        parts = []
        for tname in table_names:
            ts = self.tables.get(tname) or self.ai_tables.get(tname) or self.lookup_tables.get(tname)
            if ts:
                parts.append(self._fmt_table(ts))
        parts.append(self._fmt_lookup())
        parts.append(TABLE_RELATIONSHIPS)
        return "\n".join(parts)

    def _fmt_table(self, ts: TableSchema) -> str:
        lines = [f"### {ts.name}"]
        col_parts = []
        for col in ts.columns:
            ctype = str(col["type"]).lower()
            for long, short in [
                ("character varying", "varchar"),
                ("timestamp without time zone", "timestamp"),
                ("timestamp with time zone", "timestamptz"),
                ("double precision", "float"),
                ("boolean", "bool"),
            ]:
                ctype = ctype.replace(long, short)

            col_name = col["name"]
            full_key = f"{ts.name}.{col_name}"

            if full_key in ENUM_COLUMNS:
                vals = ENUM_COLUMNS[full_key]
                if vals:
                    col_parts.append(
                        f"  {col_name} ({ctype} — ENUM: {', '.join(repr(v) for v in vals)})")
                else:
                    col_parts.append(f"  {col_name} ({ctype} — plain string)")
                continue

            fk = next((f for f in ts.foreign_keys if f["column"] == col_name), None)
            if fk:
                ref = fk["referred_table"]
                is_opt = col_name in OPTIONAL_FKS
                jtype = "LEFT JOIN" if is_opt else "JOIN"
                if col_name in FK_RESOLUTION:
                    _, alias, sel = FK_RESOLUTION[col_name]
                    col_parts.append(
                        f"  {col_name} (uuid FK → {ref}. "
                        f"{jtype} {ref} {alias} ON ...{col_name} = {alias}.id → {sel})")
                else:
                    note = " — may be NULL, use LEFT JOIN" if is_opt else ""
                    col_parts.append(f"  {col_name} (uuid FK → {ref}{note})")
            else:
                col_parts.append(f"  {col_name} ({ctype})")

        lines.append("Columns:")
        lines.extend(col_parts)
        if ts.name in USEFUL_COLUMNS:
            lines.append(f"Useful output columns: {', '.join(USEFUL_COLUMNS[ts.name])}")
        return "\n".join(lines)

    def _fmt_ai_section(self) -> str:
        lines = ["", "### AI FLAT TABLES — form questions and answers:", ""]
        for tname in AI_TABLES:
            ts = self.ai_tables.get(tname)
            if ts:
                lines.append(self._fmt_table(ts))
                lines.append("")
        lines.append(
            "Standard JOIN:\n"
            "  FROM ai_answers aa\n"
            "  LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
            "  JOIN inspection_report ir ON aa.inspection_id = ir.id"
        )
        return "\n".join(lines)

    def _fmt_lookup(self) -> str:
        lines = [
            "", "### UUID RESOLUTION:",
            "NEVER show raw UUIDs. Always JOIN to lookup tables.",
            "'responsible', 'pending_with', 'status' are string enums — NOT FKs.",
            "", "Lookup tables:",
        ]
        for tname, info in LOOKUP_TABLES.items():
            ts = self.lookup_tables.get(tname)
            if not ts:
                continue
            expr = info["name_expr"] or info["name_col"]
            # Append live enum values if available
            ev = self.enum_values.get(tname)
            if ev and isinstance(ev[0], str):
                vals = ", ".join(repr(v) for v in ev[:20])  # cap at 20
                lines.append(f"  {tname}: {expr}  [values: {vals}]")
            elif ev and isinstance(ev[0], dict):
                # frequency_definition — show names only in the lookup section
                names = ", ".join(repr(d["name"]) for d in ev[:10])
                lines.append(f"  {tname}: {expr}  [definitions: {names}]")
            else:
                lines.append(f"  {tname}: {expr}")

        # Show live status enum values for key tables
        for col_key in ("inspection_report.status", "inspection_cycle.status",
                        "inspection_schedule.status"):
            vals = self.enum_values.get(col_key)
            if vals:
                lines.append(f"  {col_key}: {', '.join(repr(v) for v in vals)}")

        lines.append("")
        lines.append("JOIN patterns:")
        for fk_col, (table, alias, sel) in FK_RESOLUTION.items():
            lines.append(f"  {fk_col} → JOIN {table} {alias} ON ...{fk_col} = {alias}.id → {sel}")
        return "\n".join(lines)

    def _build_system_block(self) -> str:
        lines = ["━━━ INSPECTION WORKFLOW TABLES ━━━", ""]
        for tname in INSPECTION_TABLES:
            ts = self.tables.get(tname)
            if ts:
                lines.append(self._fmt_table(ts))
                lines.append("")
        lines.append(self._fmt_lookup())
        lines.append(self._fmt_ai_section())
        lines.append("")
        lines.append("BUSINESS RULES:")
        for rule in BUSINESS_RULES:
            lines.append(f"  - {rule}")
        lines.append("")
        lines.append(TABLE_RELATIONSHIPS)
        return "\n".join(lines)

    def _build_sql_block(self) -> str:
        lines = ["### INSPECTION WORKFLOW tables:", ""]
        for tname in INSPECTION_TABLES:
            ts = self.tables.get(tname)
            if ts:
                lines.append(self._fmt_table(ts))
                lines.append("")
        lines.append(self._fmt_lookup())
        lines.append(self._fmt_ai_section())
        lines.append("")
        for rule in BUSINESS_RULES:
            lines.append(f"- {rule}")
        lines.append("")
        lines.append(TABLE_RELATIONSHIPS)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Introspector
# ---------------------------------------------------------------------------

def introspect_schema(db_engine: Engine) -> DatabaseSchema:
    print("  Introspecting database schema...")
    inspector = sa_inspect(db_engine)
    schema = DatabaseSchema()
    all_tables = set(inspector.get_table_names())

    def _do(tname) -> Optional[TableSchema]:
        if tname not in all_tables:
            return None
        try:
            columns = inspector.get_columns(tname)
            fks_raw = inspector.get_foreign_keys(tname)
            pk = inspector.get_pk_constraint(tname)
            fks = []
            for fk in fks_raw:
                for i, col in enumerate(fk.get("constrained_columns", [])):
                    ref_cols = fk.get("referred_columns", [])
                    fks.append({
                        "column": col,
                        "referred_table": fk.get("referred_table", ""),
                        "referred_column": ref_cols[i] if i < len(ref_cols) else "id",
                    })
            return TableSchema(
                name=tname,
                columns=[{"name": c["name"], "type": str(c["type"]),
                           "nullable": c.get("nullable", True)} for c in columns],
                foreign_keys=fks,
                primary_key=pk.get("constrained_columns", []) if pk else [],
            )
        except Exception as e:
            print(f"    ⚠ Could not introspect {tname}: {e}")
            return None

    for tname in INSPECTION_TABLES:
        ts = _do(tname)
        if ts:
            schema.tables[tname] = ts

    for tname in AI_TABLES:
        ts = _do(tname)
        if ts:
            schema.ai_tables[tname] = ts
        else:
            print(f"    ⚠ AI table '{tname}' not found in database")

    for tname in LOOKUP_TABLES:
        ts = _do(tname)
        if ts:
            schema.lookup_tables[tname] = ts

    # ── Dynamic enum extraction ───────────────────────────────────────────────
    # Query actual values from lookup tables and key enum columns at startup.
    # This means new risk_level tiers, inspection types, frequency definitions,
    # etc. are automatically visible without any code changes.
    _ENUM_QUERIES = {
        # Lookup tables — query their name column
        "risk_level":           ("SELECT name FROM risk_level ORDER BY name", "name"),
        "impact":               ("SELECT name FROM impact ORDER BY name", "name"),
        "inspection_type":      ("SELECT name FROM inspection_type ORDER BY name", "name"),
        "frequency_definition": (
            "SELECT name, repeat_count, repeat_interval, repeat_unit "
            "FROM frequency_definition ORDER BY name", "name"
        ),
        # Enum columns on main tables — query DISTINCT values
        "inspection_report.status": (
            "SELECT DISTINCT status FROM inspection_report WHERE status IS NOT NULL ORDER BY status",
            "status"
        ),
        "inspection_cycle.status": (
            "SELECT DISTINCT status FROM inspection_cycle WHERE status IS NOT NULL ORDER BY status",
            "status"
        ),
        "inspection_schedule.status": (
            "SELECT DISTINCT status FROM inspection_schedule WHERE status IS NOT NULL ORDER BY status",
            "status"
        ),
    }
    try:
        from sqlalchemy import text as sa_text
        with db_engine.connect() as conn:
            for key, (query, col) in _ENUM_QUERIES.items():
                try:
                    rows = conn.execute(sa_text(query)).fetchall()
                    if key == "frequency_definition":
                        # Store full definition for richer context
                        schema.enum_values[key] = [
                            {"name": r[0], "repeat_count": r[1],
                             "repeat_interval": r[2], "repeat_unit": r[3]}
                            for r in rows
                        ]
                    else:
                        schema.enum_values[key] = [r[0] for r in rows if r[0]]
                except Exception as e:
                    pass  # Non-fatal: hardcoded values in ENUM_COLUMNS still apply
    except Exception:
        pass  # DB not available — degrade gracefully

    print(
        f"  ✓ Schema: {len(schema.tables)} inspection tables, "
        f"{len(schema.ai_tables)} ai tables, "
        f"{len(schema.lookup_tables)} lookup tables"
    )
    if schema.enum_values:
        extracted = ", ".join(
            f"{k}={len(v)}" for k, v in schema.enum_values.items()
            if isinstance(v[0], str) if v
        )
        print(f"  ✓ Dynamic enums: {extracted}")
    return schema