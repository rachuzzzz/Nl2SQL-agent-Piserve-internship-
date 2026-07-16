"""
Standalone module-level helpers: SQL literal escaping, the raw-LLM debug
logger, truncation detection, UUID-leak detection, and constraint-note
building. Extracted verbatim from the original agent/orchestrator.py.
"""

import logging
import re




# ---------------------------------------------------------------------------
# SQL literal escaping — for names pulled from query results (facility,
# inspector, entity candidates) that get spliced into ILIKE '%...%' text
# fed to the SQL-generating LLM. Without this, a name containing an
# apostrophe (e.g. "O'Brien's Yard") breaks the string literal.
# ---------------------------------------------------------------------------

def _ilike_escape(value: str) -> str:
    """Escape a raw value for safe interpolation into a SQL string literal."""
    return value.replace("'", "''")




# ---------------------------------------------------------------------------
# Debug logger — enable with DEBUG_LLM=1 in environment or .env
# Writes to debug_llm_YYYYMMDD_HHMMSS.log in the working directory
# ---------------------------------------------------------------------------

def _make_debug_logger() -> logging.Logger | None:
    """Return a file logger for raw LLM output when DEBUG_LLM=1.

    Reuses the file that setup_logging() already opened (the inspectai
    RotatingFileHandler) so all logs — structured and raw — go to the same
    debug_llm_*.log file rather than creating a second file.
    """
    from src.config.settings import get_settings
    if not get_settings().debug_llm:
        return None

    # Find the file path already opened by setup_logging() on the inspectai logger.
    # Fall back to a fresh timestamped file if setup_logging() hasn't run yet.
    from src.core.logging_config import get_log_file
    log_path = get_log_file()

    logger = logging.getLogger("debug_llm_raw")
    logger.setLevel(logging.DEBUG)
    # Avoid adding duplicate handlers if called more than once (e.g. hot-reload)
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    print(f"  [DEBUG_LLM] Raw LLM output → {log_path}")
    return logger




def _dbg(logger, section: str, content: str, iteration: int = 0):
    """Write a labelled block to the debug log."""
    if logger is None:
        return
    sep = "=" * 70
    it = f" [iter {iteration}]" if iteration else ""
    logger.debug(f"\n{sep}\n>>> {section}{it}\n{sep}\n{content}\n")




def _detect_truncation(raw: str) -> str | None:
    """
    Return a warning string if the raw LLM output looks truncated.
    Common signs: ends without closing braces, ends mid-string.
    """
    stripped = raw.strip()
    if not stripped:
        return "EMPTY output"
    open_b = stripped.count("{") - stripped.count("}")
    open_q = stripped.count('"') % 2
    if open_b > 0:
        return f"TRUNCATED — {open_b} unclosed brace(s)"
    if open_q:
        return "TRUNCATED — odd number of quotes (mid-string cut)"
    if not stripped.endswith(("}", "]")):
        return f"SUSPICIOUS — ends with: {repr(stripped[-30:])}"
    return None






# UUID pattern — 8-4-4-4-12 hex
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE
)


# Columns whose values should NEVER be raw UUIDs — they need a JOIN
_NAME_COLUMNS = {
    "facility_name", "facility", "inspector_name", "inspector",
    "inspectee_name", "inspectee", "client_name", "client",
    "project_name", "project", "type_name", "inspection_type_name",
    "module_name", "subtype_name",
}


def _has_uuid_leak(rows: list, columns: list) -> list[str]:
    """
    Check result rows for UUID values in columns that should be human-readable names.
    Returns list of offending column names.
    """
    leaking = set()
    for col in columns:
        col_lower = col.lower()
        # Only check columns that are supposed to be name-like
        if not any(n in col_lower for n in ("name", "facility", "inspector",
                                             "client", "project", "type")):
            continue
        # Sample first 5 rows
        for row in rows[:5]:
            val = row.get(col)
            if val and isinstance(val, str) and _UUID_RE.match(val.strip()):
                leaking.add(col)
                break
    return sorted(leaking)




# ---------------------------------------------------------------------------
# Retry constraint notes builder
# ---------------------------------------------------------------------------

# Maps validator error codes → focused constraint notes for the retry prompt.
# These notes survive deepseek's seed-example bias because they are placed in
# Qwen's message (not in schema_hint which deepseek may override with seeds).
# Rule: give the exact fix as a constraint, never a full replacement SQL.
#
# STATUS + CAST notes are generated from src.core.semantics_registry — the single
# source of truth. Never hardcode status groupings or cast directions here.
try:
    from src.core.semantics_registry import constraint_note_status, constraint_note_cast
    _NOTE_STATUS = constraint_note_status()
    _NOTE_CAST = constraint_note_cast()
except ImportError:
    # Fallback literals — keep in sync with core/semantics_registry.py
    _NOTE_STATUS = (
        "-- completed = status = 'CLOSED' | conducted = status IN "
        "('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION') | always exclude DRAFT"
    )
    _NOTE_CAST = "-- JOIN ON aa.submission_id = ica.observation_id::varchar"


_CONSTRAINT_NOTE_MAP = {
    "WRONG_CAST_DIR": _NOTE_CAST,
    "IMPOSSIBLE_JOIN": (
        "-- Check the join path carefully. Use get_schema to verify FK columns."
    ),
    "WRONG_COLUMN": (
        "-- ai_answers has NO 'label' column. To filter by question label:\n"
        "--   JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
        "--   WHERE aq.label = 'Observation'  (or whichever label)\n"
        "-- ai_answers also has NO 'form_name' column — it has 'module_name'."
    ),
    "CYCLE_FREQUENCY_MISSING_SUBQUERY": (
        "-- 'list all cycles' / 'cycle types' = frequency_definition directly, NO inspection_cycle join:\n"
        "--   SELECT name AS frequency_name, repeat_count, repeat_interval, repeat_unit\n"
        "--   FROM frequency_definition WHERE active = true ORDER BY name;\n"
        "-- The ic->ip->ipd->fd join chain with status='COMPLETED' is ONLY for\n"
        "--   'the MOST RECENT completed cycle' and REQUIRES:\n"
        "--   WHERE ic.id = (SELECT id FROM inspection_cycle WHERE status='COMPLETED'\n"
        "--                   ORDER BY end_date DESC LIMIT 1)"
    ),
    "UNDEFINED_ALIAS": (
        "-- A table alias was referenced in WHERE/SELECT but never introduced via FROM/JOIN.\n"
        "--   EITHER add the missing JOIN for that alias (if you need its columns),\n"
        "--   OR remove every reference to that alias if the table isn't actually needed.\n"
        "--   Re-check every alias used (ica, ir, fac, u, ...) has a matching\n"
        "--   'FROM <table> <alias>' or 'JOIN <table> <alias>' in the query."
    ),
    "WRONG_JOIN": (
        "-- 'responsible' and 'pending_with' are VARCHAR enums, not UUID FKs.\n"
        "--   Do NOT join to users table on these columns.\n"
        "-- 'responsible party' = ica.responsible directly (CLIENT | INTERNAL_OPERATIONS | SUB_CONTRACTOR)\n"
        "--   SELECT ica.responsible AS responsible_party — NO JOIN to users needed or possible.\n"
        "--   If a previous attempt added LEFT JOIN users ON ica.responsible = '...', REMOVE that join entirely."
    ),
    "UUID_IN_SELECT": (
        "-- ir.id is the UUID primary key — never SELECT it for display.\n"
        "--   Use ir.inspection_id (VARCHAR) for human-readable inspection IDs."
    ),
    "SPLIT_NAME_MISMATCH": (
        "-- (x.first_name ILIKE '%A%' OR x.last_name ILIKE '%B%') with A != B matches\n"
        "--   anyone named A OR anyone surnamed B — NOT people named 'A B'.\n"
        "-- For a two-word full name, use the CONCATENATED form instead:\n"
        "--   WHERE (x.first_name || ' ' || x.last_name) ILIKE '%A B%'\n"
        "-- Only use the OR form when both literals are IDENTICAL (single-word partial name):\n"
        "--   WHERE (x.first_name ILIKE '%george%' OR x.last_name ILIKE '%george%')  -- OK"
    ),
    "TOCHAR_WHERE_FILTER": (
        "-- TO_CHAR(col, 'Month YYYY') = 'January 2026' ALWAYS RETURNS 0 ROWS in WHERE.\n"
        "--   PostgreSQL pads month names with spaces: 'January   2026' != 'January 2026'.\n"
        "--   CORRECT: EXTRACT(MONTH FROM ir.submitted_on) = 1 AND EXTRACT(YEAR ...) = 2026\n"
        "--   TO_CHAR is only valid for SELECT/GROUP BY display labels, never WHERE."
    ),
    "STALE_DATE_LITERAL": (
        "-- NEVER hardcode a past year in date literals like DATE '2023-01-01'.\n"
        "--   The data is from 2025/2026. Use EXTRACT with CURRENT_DATE:\n"
        "--     EXTRACT(MONTH FROM ir.submitted_on) = 1\n"
        "--     AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)\n"
        "--   Only use a specific year if the user explicitly asked for it."
    ),
    "TRUNCATED_INSPECTION_ID": (
        "-- inspection_id is always the FULL string 'YYYY/MM/STXXX/INSXXX'.\n"
        "--   NEVER truncate to 'STXXX/INSXXX' — always use the complete value the user provided.\n"
        "--   Example: WHERE ir.inspection_id = '2026/01/ST074/INS002'"
    ),
    "MISSING_COMPLETED_STATUS": _NOTE_STATUS,
    # Fired when business_rules.py BLOCK rules trigger (error code is "BUSINESS_RULE").
    # The most common cause is block_capex_left_join_multiplication or
    # block_cte_reference_without_definition.  The note gives deepseek the two
    # safe alternatives so it does NOT repeat the same broken pattern.
    "BUSINESS_RULE": (
        "-- CAPEX ROW-MULTIPLICATION FIX: never LEFT JOIN po_details directly when SUM(ica.capex) is present.\n"
        "-- Safe option A — scalar subqueries (no GROUP BY needed):\n"
        "--   SELECT\n"
        "--     (SELECT COALESCE(SUM(capex),0) FROM inspection_corrective_action\n"
        "--      WHERE active=true AND capex>0) AS capex_generated,\n"
        "--     (SELECT COALESCE(SUM(po_amount),0) FROM po_details WHERE active=true) AS capex_converted\n"
        "-- Safe option B — CTE pre-aggregation (required when grouping by region/facility/etc):\n"
        "--   WITH po_by_ica AS (\n"
        "--       SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po\n"
        "--       FROM po_details WHERE active=true GROUP BY 1\n"
        "--   )\n"
        "--   SELECT ... LEFT JOIN po_by_ica p ON p.ica_id = ica.id ...\n"
        "-- NEVER omit the WITH clause when referencing po_by_ica in a JOIN.\n"
        "-- NEVER: LEFT JOIN po_details po ON po.mitigative_action_uuid = ica.id  (direct join inflates SUM)"
    ),
}


def _build_constraint_notes(retry_reason: str) -> str:
    """
    Given a retry_reason string (from ValidationResult.retry_message()),
    return focused constraint notes for the codes present.

    Returns empty string if no matching codes found — avoids adding noise
    for codes with no known constraint fix.
    """
    notes = []
    for code, note in _CONSTRAINT_NOTE_MAP.items():
        if code in retry_reason:
            notes.append(note)
    if notes:
        return "Constraints to follow in your rewrite:\n" + "\n".join(notes) + "\n"
    return ""
