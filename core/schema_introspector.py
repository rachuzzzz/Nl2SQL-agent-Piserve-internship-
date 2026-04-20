"""
Schema Introspector
====================
Fetches database schema at startup via SQLAlchemy inspect().
Replaces all hardcoded schema in prompts with dynamically generated text.

Two kinds of knowledge:
  1. INTROSPECTED (automatic): table names, column names, types, foreign keys
  2. ANNOTATED (config): enum values, business rules, "don't JOIN this column"
     — things the database metadata doesn't tell you.

The annotations are minimal (~30 lines of config) vs the old approach of
maintaining 3 copies of the full schema across prompts.py and orchestrator.py.
"""

from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Annotations — things we CAN'T introspect from the database
# ---------------------------------------------------------------------------

# Columns that look like FKs but are actually plain string enums.
# Format: "table.column" → list of known enum values
ENUM_COLUMNS = {
    "inspection_corrective_action.responsible": [
        "CLIENT", "INTERNAL_OPERATIONS", "SUB_CONTRACTOR",
    ],
    "inspection_corrective_action.status": [
        "OPEN", "CLOSED", "OVERDUE", "CLOSE_WITH_DEFERRED",
    ],
    "inspection_corrective_action.pending_with": [],  # enum but values unknown
    "inspection_report.status": [
        "DRAFT", "SUBMITTED", "CLOSED", "UNDER_REVIEW", "RETURN_FOR_MODIFICATION",
    ],
    "inspection_schedule.status": [
        "PENDING", "ONGOING", "COMPLETED", "OVERDUE", "CANCELLED",
    ],
    "inspection_cycle.status": [],  # enum but values unknown
}

# Business rules that affect SQL generation
BUSINESS_RULES = [
    "DRAFT inspections usually have NULL scores. Always add WHERE status != 'DRAFT' "
    "when querying inspection_score, unless the user explicitly asks about drafts.",
    "NEVER use SELECT *. Always specify the columns the user needs.",
    "NEVER use SQL reserved words as table aliases (is, as, in, on, by, do, if).",
    "Use ILIKE for text matching, never LIKE.",
    "corrective_action_id is a human-readable ID like '2026/01/ST064/INS001_MA001', not a UUID.",
    "inspection_corrective_action.inspection_id is the PRIMARY FK to inspection_report.id. "
    "Use this for all JOINs between corrective actions and reports. "
    "Do NOT use observation_id for this — it links to a different table.",
    "Use LEFT JOIN (not INNER JOIN) for optional lookups like inspection_sub_type, "
    "entity, and inspection_report_remark. Not every inspection has a sub_type or entity. "
    "INNER JOIN would silently drop rows that have NULL FKs.",
    # Vocabulary mappings — how business language maps to database values
    "VOCABULARY — corrective action status synonyms: "
    "'pending'/'outstanding'/'unresolved'/'active' = status NOT IN ('CLOSED','CLOSE_WITH_DEFERRED'). "
    "'open' = status = 'OPEN'. "
    "'overdue'/'late'/'past due' = status = 'OVERDUE'. "
    "'closed'/'resolved'/'completed' = status IN ('CLOSED','CLOSE_WITH_DEFERRED'). "
    "NEVER filter by progress_stage for status questions — use the status column.",
    "VOCABULARY — inspection report status synonyms: "
    "'completed'/'done'/'finished' = status = 'CLOSED'. "
    "'pending review'/'under review'/'awaiting review' = status = 'UNDER_REVIEW'. "
    "'returned'/'sent back'/'rejected' = status = 'RETURN_FOR_MODIFICATION'. "
    "'submitted'/'filed' = status = 'SUBMITTED'.",
]

# ---------------------------------------------------------------------------
# Table Relationships — WHERE concepts live and HOW to reach them
# ---------------------------------------------------------------------------
# The LLM knows what columns each table has (introspected), but not how
# concepts flow between tables. Without this, "safety inspection cycle"
# makes the LLM look for "safety" on inspection_cycle (wrong) instead of
# going through inspection_type → inspection_report → inspection_cycle.

TABLE_RELATIONSHIPS = """\
CRITICAL — WHERE CONCEPTS LIVE:
  - Inspection TYPE names (Safety, Building, PPE, etc.) → inspection_type.name
  - Inspection SUB-TYPE names → inspection_sub_type.name
  - Inspector names → users table (JOIN via inspector_user_id)
  - Facility names → facility table (JOIN via facility_id)
  - Project names → project table (JOIN via project_id)
  - Client names → client table (JOIN via client_id)
  - Scores, status, dates → inspection_report (direct columns)
  - Causes, costs, overdue → inspection_corrective_action (direct columns)
  - Schedule dates → inspection_schedule (direct columns)
  - Cycle date ranges → inspection_cycle (direct columns)

CRITICAL — HOW TABLES CONNECT (the hub is inspection_report):
  inspection_report is the HUB table. It connects to everything:
    inspection_report.inspection_type_id → inspection_type.id
    inspection_report.facility_id → facility.id
    inspection_report.project_id → project.id
    inspection_report.client_id → client.id
    inspection_report.inspector_user_id → users.id
    inspection_report.cycle_id → inspection_cycle.id
    inspection_report.schedule_id → inspection_schedule.id
    inspection_corrective_action.inspection_id → inspection_report.id

  To connect ANY two concepts, go THROUGH inspection_report:
    type → cycle:   inspection_type → inspection_report → inspection_cycle
    type → facility: inspection_type → inspection_report → facility
    facility → corrective_action: facility → inspection_report → inspection_corrective_action
    inspector → facility: users → inspection_report → facility
    corrective_action → type: inspection_corrective_action → inspection_report → inspection_type

EXAMPLES of multi-hop queries:
  "cycles for safety inspections":
    JOIN inspection_report ir ON ir.cycle_id = ic.id
    JOIN inspection_type it ON ir.inspection_type_id = it.id
    WHERE it.name ILIKE '%safety%'

  "corrective actions at Al Ghadeer facility":
    JOIN inspection_report ir ON ica.inspection_id = ir.id
    JOIN facility fac ON ir.facility_id = fac.id
    WHERE fac.name ILIKE '%Al Ghadeer%'

  "which inspector has the most safety inspections":
    JOIN inspection_report ir ON ir.inspector_user_id = u.id
    JOIN inspection_type it ON ir.inspection_type_id = it.id
    WHERE it.name ILIKE '%safety%'

RULE: When the user mentions a TYPE name (safety, building, PPE, hygiene, etc.),
ALWAYS filter via inspection_type.name — never look for the name on other tables.
inspection_cycle, inspection_schedule, and inspection_corrective_action do NOT
have a name or type column — you MUST go through inspection_report to reach
inspection_type.
"""

# Lookup tables used for UUID resolution
LOOKUP_TABLES = {
    "users": {"name_expr": "first_name || ' ' || last_name", "name_col": None},
    "facility": {"name_expr": None, "name_col": "name"},
    "project": {"name_expr": None, "name_col": "name"},
    "client": {"name_expr": None, "name_col": "name"},
    "entity": {"name_expr": None, "name_col": "name"},
    "organisation": {"name_expr": None, "name_col": "name"},
    "inspection_type": {"name_expr": None, "name_col": "name"},
    "inspection_sub_type": {"name_expr": None, "name_col": "name"},
}

# FK columns → which lookup table + alias to use
# "optional" FKs should use LEFT JOIN (not all inspections have these)
FK_RESOLUTION = {
    "inspector_user_id": ("users", "u", "u.first_name || ' ' || u.last_name AS inspector_name"),
    "inspectee_user_id": ("users", "insp", "insp.first_name || ' ' || insp.last_name AS inspectee_name"),
    "facility_id": ("facility", "fac", "fac.name AS facility_name"),
    "project_id": ("project", "proj", "proj.name AS project_name"),
    "client_id": ("client", "cl", "cl.name AS client_name"),
    "entity_id": ("entity", "ent", "ent.name AS entity_name"),
    "inspection_type_id": ("inspection_type", "it", "it.name AS inspection_type_name"),
    "inspection_sub_type_id": ("inspection_sub_type", "ist", "ist.name AS inspection_subtype_name"),
}

# FKs that may be NULL — use LEFT JOIN, not INNER JOIN
OPTIONAL_FKS = {
    "inspection_sub_type_id", "entity_id", "close_submission_id",
    "closed_by_user_id", "schedule_id",
}

# Tables to introspect (inspection domain)
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

# Useful columns to show in output (instead of SELECT *)
USEFUL_COLUMNS = {
    "inspection_report": [
        "inspection_id", "inspection_score", "gp_score", "status",
        "submitted_on", "total_inspection_hours", "start_date_time", "end_date_time",
    ],
    "inspection_corrective_action": [
        "corrective_action_id", "cause", "correction", "corrective_action",
        "responsible", "status", "progress_stage", "capex", "opex",
        "target_close_out_date", "age",
    ],
}


# ---------------------------------------------------------------------------
# Introspector
# ---------------------------------------------------------------------------

@dataclass
class TableSchema:
    """Schema for a single table."""
    name: str
    columns: list  # [{name, type, nullable}, ...]
    foreign_keys: list  # [{column, referred_table, referred_column}, ...]
    primary_key: list  # [column_name, ...]


@dataclass
class DatabaseSchema:
    """Complete introspected schema for the inspection domain."""
    tables: dict = field(default_factory=dict)  # table_name → TableSchema
    lookup_tables: dict = field(default_factory=dict)  # table_name → TableSchema
    _schema_text_cache: Optional[str] = field(default=None, repr=False)
    _sql_prompt_cache: Optional[str] = field(default=None, repr=False)
    _system_prompt_cache: Optional[str] = field(default=None, repr=False)

    def for_system_prompt(self) -> str:
        """Schema block for the reasoning LLM's system prompt."""
        if self._system_prompt_cache:
            return self._system_prompt_cache
        self._system_prompt_cache = self._build_system_prompt_block()
        return self._system_prompt_cache

    def for_sql_prompt(self) -> str:
        """Schema block for the SQL generation prompt."""
        if self._sql_prompt_cache:
            return self._sql_prompt_cache
        self._sql_prompt_cache = self._build_sql_prompt_block()
        return self._sql_prompt_cache

    def for_schema_hint(self, table_names: list) -> str:
        """
        Build a schema_hint string for specific tables.
        Used by the orchestrator's auto-injection.
        """
        parts = []
        for tname in table_names:
            ts = self.tables.get(tname) or self.lookup_tables.get(tname)
            if ts:
                parts.append(self._format_table_compact(ts))
        # Always append lookup tables + resolution rules
        parts.append(self._format_lookup_section())
        # Always append relationship map — critical for multi-hop JOINs
        parts.append(TABLE_RELATIONSHIPS)
        return "\n".join(parts)

    # --- Internal builders ---

    def _format_table_compact(self, ts: TableSchema) -> str:
        """Format one table as compact schema text."""
        lines = [f"### {ts.name}"]
        col_parts = []
        for col in ts.columns:
            ctype = str(col["type"]).lower()
            # Shorten common types
            for long, short in [("character varying", "varchar"),
                                ("timestamp without time zone", "timestamp"),
                                ("timestamp with time zone", "timestamptz"),
                                ("double precision", "float"),
                                ("boolean", "bool")]:
                ctype = ctype.replace(long, short)

            col_name = col["name"]
            full_key = f"{ts.name}.{col_name}"

            # Check if this is an enum column
            if full_key in ENUM_COLUMNS:
                vals = ENUM_COLUMNS[full_key]
                if vals:
                    enum_str = ", ".join(f"'{v}'" for v in vals)
                    col_parts.append(
                        f"  {col_name} ({ctype} — ENUM: {enum_str} — NOT a FK)")
                else:
                    col_parts.append(f"  {col_name} ({ctype} — plain string, NOT a FK)")
                continue

            # Check if this is a FK
            fk_match = next((fk for fk in ts.foreign_keys
                             if fk["column"] == col_name), None)
            if fk_match:
                ref = fk_match["referred_table"]
                is_optional = col_name in OPTIONAL_FKS
                join_type = "LEFT JOIN" if is_optional else "JOIN"
                # Check if we have a resolution pattern
                if col_name in FK_RESOLUTION:
                    _, alias, select = FK_RESOLUTION[col_name]
                    col_parts.append(
                        f"  {col_name} (uuid FK → {ref}. "
                        f"{join_type} {ref} {alias} ON ...{col_name} = {alias}.id → {select})")
                else:
                    optional_note = " — may be NULL, use LEFT JOIN" if is_optional else ""
                    col_parts.append(f"  {col_name} (uuid FK → {ref}{optional_note})")
            else:
                col_parts.append(f"  {col_name} ({ctype})")

        lines.append("Columns:")
        lines.extend(col_parts)

        # Add useful columns note
        if ts.name in USEFUL_COLUMNS:
            useful = ", ".join(USEFUL_COLUMNS[ts.name])
            lines.append(f"Useful output columns: {useful}")

        return "\n".join(lines)

    def _format_lookup_section(self) -> str:
        """Format the lookup tables and UUID resolution rules."""
        lines = [
            "",
            "### UUID RESOLUTION — CRITICAL RULE:",
            "NEVER show raw UUIDs in query results. Always JOIN to lookup tables.",
            "EXCEPTION: 'responsible', 'pending_with', 'status' are plain string enums — NOT FKs.",
            "",
            "Lookup tables:",
        ]
        for tname, info in LOOKUP_TABLES.items():
            ts = self.lookup_tables.get(tname)
            if not ts:
                continue
            if info["name_expr"]:
                lines.append(f"  {tname}: {info['name_expr']}")
            elif info["name_col"]:
                lines.append(f"  {tname}: {info['name_col']}")

        lines.append("")
        lines.append("Common JOIN patterns:")
        for fk_col, (table, alias, select_expr) in FK_RESOLUTION.items():
            lines.append(f"  {fk_col} → JOIN {table} {alias} ON ...{fk_col} = {alias}.id → {select_expr}")

        return "\n".join(lines)

    def _build_system_prompt_block(self) -> str:
        """Build the full schema block for SYSTEM_PROMPT."""
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "INSPECTION WORKFLOW TABLES (auto-introspected)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]
        # Main inspection tables
        for tname in INSPECTION_TABLES:
            ts = self.tables.get(tname)
            if ts:
                lines.append(self._format_table_compact(ts))
                lines.append("")

        # Lookup/resolution
        lines.append(self._format_lookup_section())

        # Business rules
        lines.append("")
        lines.append("BUSINESS RULES:")
        for rule in BUSINESS_RULES:
            lines.append(f"  - {rule}")

        # Table relationships — how concepts connect
        lines.append("")
        lines.append(TABLE_RELATIONSHIPS)

        return "\n".join(lines)

    def _build_sql_prompt_block(self) -> str:
        """Build the schema block for SQL_GENERATION_PROMPT."""
        lines = [
            "### INSPECTION WORKFLOW tables (auto-introspected from database):",
            "",
        ]
        for tname in INSPECTION_TABLES:
            ts = self.tables.get(tname)
            if ts:
                lines.append(self._format_table_compact(ts))
                lines.append("")

        lines.append(self._format_lookup_section())

        lines.append("")
        for rule in BUSINESS_RULES:
            lines.append(f"- {rule}")

        # Table relationships
        lines.append("")
        lines.append(TABLE_RELATIONSHIPS)

        return "\n".join(lines)


def introspect_schema(db_engine: Engine) -> DatabaseSchema:
    """
    Introspect the database at startup.
    Returns a DatabaseSchema with all inspection + lookup table metadata.
    """
    print("  Introspecting database schema...")
    inspector = sa_inspect(db_engine)
    schema = DatabaseSchema()

    all_tables = set(inspector.get_table_names())

    # Introspect inspection tables
    for tname in INSPECTION_TABLES:
        if tname not in all_tables:
            continue
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

            col_info = [
                {"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True)}
                for c in columns
            ]

            schema.tables[tname] = TableSchema(
                name=tname,
                columns=col_info,
                foreign_keys=fks,
                primary_key=pk.get("constrained_columns", []) if pk else [],
            )
        except Exception as e:
            print(f"    ⚠ Could not introspect {tname}: {e}")

    # Introspect lookup tables
    for tname in LOOKUP_TABLES:
        if tname not in all_tables:
            continue
        try:
            columns = inspector.get_columns(tname)
            col_info = [
                {"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True)}
                for c in columns
            ]
            schema.lookup_tables[tname] = TableSchema(
                name=tname, columns=col_info, foreign_keys=[], primary_key=["id"],
            )
        except Exception as e:
            print(f"    ⚠ Could not introspect lookup {tname}: {e}")

    insp_count = len(schema.tables)
    lookup_count = len(schema.lookup_tables)
    total_cols = sum(len(t.columns) for t in schema.tables.values())
    total_fks = sum(len(t.foreign_keys) for t in schema.tables.values())
    print(f"  ✓ Schema introspected: {insp_count} inspection tables "
          f"({total_cols} columns, {total_fks} FKs), {lookup_count} lookup tables")

    return schema