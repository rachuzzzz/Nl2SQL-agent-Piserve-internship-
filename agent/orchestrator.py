"""
Agent Orchestrator — the main reasoning loop.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

_log_query = logging.getLogger("inspectai.query")
_log_llm   = logging.getLogger("inspectai.llm")
_log_sql   = logging.getLogger("inspectai.sql")

from sqlalchemy import create_engine, text as sql_text

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from core.semantic import SemanticQuestionIndex, SeedExampleIndex
from core.validator import SQLValidator
from core.schema_introspector import introspect_schema
from agent.prompts import SYSTEM_PROMPT, SQL_GENERATION_PROMPT, FORCE_SUMMARY_PROMPT, build_dynamic_prompt_additions
from agent.tools import ToolRegistry
from agent.tool_dispatcher import parse_tool_call, dispatch
from core.business_rules import get_engine as get_rules_engine
from core.rbac import build_scope_filter
from core.disambiguation import is_disambiguation_result, is_entity_disambiguation_result


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
    if not os.getenv("DEBUG_LLM", "").strip() in ("1", "true", "yes"):
        return None

    # Find the file path already opened by setup_logging() on the inspectai logger.
    # Fall back to a fresh timestamped file if setup_logging() hasn't run yet.
    from core.logging_config import get_log_file
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


@dataclass
class AgentStep:
    iteration: int
    thought: str
    tool: str
    args: dict
    result: dict
    duration_ms: int


@dataclass
class QueryStats:
    total_llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_latency_ms: int = 0
    total_sql_exec_ms: int = 0
    total_wall_ms: int = 0


@dataclass
class ConversationTurn:
    question: str
    answer: str
    tool_used: str
    sql: str = ""
    total_count: int = 0
    columns: list = field(default_factory=list)
    # Extracted key entities from result rows — used for follow-up context
    inspection_ids: list = field(default_factory=list)   # e.g. ["2026/04/ST085/INS003"]
    facility_names: list = field(default_factory=list)   # e.g. ["Golf Gardens"]
    inspector_names: list = field(default_factory=list)  # e.g. ["neenu extinsp1"]
    cycle_ids: list = field(default_factory=list)        # UUIDs from cycle_id column
    user_ids: list = field(default_factory=list)         # resolved user UUIDs after disambiguation
    disambiguation_name: str = ""   # name that was disambiguated (e.g. "george"), total_count stored in total_count
    disambiguation_candidates: list = field(default_factory=list)  # [{"name": ..., "user_id": ...}] from disambiguation rows
    disambiguation_origin_question: str = ""  # the ORIGINAL question that started this disambiguation chain (e.g. "which schedules is assigned to neenu") — carried forward across narrowing turns so the agent doesn't lose the actual goal while resolving a name
    entity_disambiguation_candidates: list = field(default_factory=list)  # [{"name": ..., "inspection_count": ...}] from a STEP 1 entity (facility/project/client/organisation) count-check — lets a later turn that names one entity from this list resolve active_subject_type even though that turn ran no new SQL of its own
    entity_disambiguation_table: str = ""  # e.g. "facility", "project" — which table the candidates above came from
    entity_disambiguation_alias: str = ""  # e.g. "fac", "proj" — the SQL alias used, for building a correct {alias}.name ILIKE filter
    single_row_values: dict = field(default_factory=dict) # col→val when result is 1 row
    # Active focal entity — what the user is currently drilling into
    # type: "inspection"|"facility"|"inspector"|"cycle"|"corrective_action"|"entity"|None
    active_subject_type: str = ""
    active_subject_value: str = ""   # exact value (UUID or name)
    active_subject_label: str = ""   # human-readable label for context message
    active_subject_table: str = ""   # only for active_subject_type=="entity" — e.g. "project"
    active_subject_alias: str = ""   # only for active_subject_type=="entity" — e.g. "proj"




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
# STATUS + CAST notes are generated from core.semantics_registry — the single
# source of truth. Never hardcode status groupings or cast directions here.
try:
    from core.semantics_registry import constraint_note_status, constraint_note_cast
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


class ConversationSession:
    MAX_TURNS = 5

    def __init__(self):
        self.turns: list[ConversationTurn] = []

    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
        if len(self.turns) > self.MAX_TURNS:
            self.turns = self.turns[-self.MAX_TURNS:]

    @property
    def last_turn(self) -> Optional[ConversationTurn]:
        return self.turns[-1] if self.turns else None

    def build_context_block(self, current_question: str = "") -> str:
        if not self.turns:
            return ""

        _FOLLOWUP_RE = re.compile(
            r"\b(it|that|this|she|he|they|them|her|his|their|"
            r"that\s+inspection|that\s+site|that\s+facility|that\s+form|"
            r"the\s+same|the\s+above|from\s+there|more|rest|remaining|"
            r"continue|next|previous|last\s+one|back\s+to|go\s+back|"
            r"that\s+inspector|that\s+cycle|that\s+person)\b",
            re.IGNORECASE,
        )
        # A question is "clearly fresh" only when it introduces a wholly new subject.
        # INTENTIONALLY NARROW: question-starters like "how many", "what", "show", "list"
        # appear constantly in follow-ups ("how many are completed", "show the open ones")
        # and must NOT suppress the active context.
        _FRESH_SUBJECT_RE = re.compile(
            r"\b(most\s+recent(?!ly)|most\s+common|latest)\b",
            re.IGNORECASE,
        )
        is_followup = bool(_FOLLOWUP_RE.search(current_question))
        is_clearly_fresh = (
            bool(_FRESH_SUBJECT_RE.search(current_question))
            and not is_followup
        )

        lines = ["━━━ CONVERSATION HISTORY ━━━"]
        for i, t in enumerate(self.turns, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  User: {t.question}")
            short = t.answer[:200] + ("..." if len(t.answer) > 200 else "")
            lines.append(f"  Answer: {short}")
            if t.sql:
                lines.append(f"  SQL: {t.sql[:150]}")
            if t.total_count > 0:
                lines.append(f"  Total rows: {t.total_count}")
            if t.columns:
                lines.append(f"  Columns: {t.columns}")
            if t.inspection_ids:
                lines.append(f"  ★ inspection_id(s) from this result: {t.inspection_ids[:5]}")
            if t.cycle_ids:
                lines.append(f"  ★ cycle_id(s) from this result: {t.cycle_ids[:3]}")
            if t.facility_names:
                lines.append(f"  ★ facility(s) from this result: {t.facility_names[:5]}")
            if t.inspector_names:
                lines.append(f"  ★ inspector(s) from this result: {t.inspector_names[:5]}")
            if t.single_row_values:
                lines.append(f"  ★ single-row result: {t.single_row_values}")
            # Show active subject for each turn
            if t.active_subject_type and t.active_subject_value:
                lines.append(
                    f"  ◆ ACTIVE SUBJECT: {t.active_subject_type} = "
                    f"'{t.active_subject_label or t.active_subject_value}' "
                    f"(exact value: {t.active_subject_value})"
                )
        lines.append("")

        # ── Resolve active subject across all turns ──────────────────────────
        # Walk backwards through turns to find the most recent active subject.
        # "go back to that inspector" can re-anchor to an earlier turn's subject.
        active_type = ""
        active_value = ""
        active_label = ""
        active_table = ""
        active_alias = ""

        # "go back" or "back to" signals: re-anchor to the earliest matching subject
        _BACK_RE = re.compile(r"\b(go\s+back|back\s+to|return\s+to)\b", re.IGNORECASE)
        if _BACK_RE.search(current_question):
            # Look for the first (oldest) active subject in history
            for t in self.turns:
                if t.active_subject_type and t.active_subject_value:
                    active_type = t.active_subject_type
                    active_value = t.active_subject_value
                    active_label = t.active_subject_label
                    active_table = t.active_subject_table
                    active_alias = t.active_subject_alias
                    break
        else:
            # Normal: use the most recent active subject
            for t in reversed(self.turns):
                if t.active_subject_type and t.active_subject_value:
                    active_type = t.active_subject_type
                    active_value = t.active_subject_value
                    active_label = t.active_subject_label
                    active_table = t.active_subject_table
                    active_alias = t.active_subject_alias
                    break

        # self.turns is guaranteed non-empty (early-return above), so `last` is
        # always defined for both the fallback block and the disambiguation
        # STOP-guard below.
        last = self.turns[-1]

        # ── Emit the scoped context block ────────────────────────────────────
        if not is_clearly_fresh and active_type and active_value:
            if active_type == "cycle":
                lines.append(
                    f"ACTIVE CONTEXT — The conversation is scoped to cycle "
                    f"'{active_value}'."
                    f" All follow-up questions about inspections, corrective actions,"
                    f" facilities, or inspectors should include WHERE ir.cycle_id = '{active_value}'."
                    f" When user says 'they', 'that cycle', 'those facilities', 'those inspectors'"
                    f" — this is the scope."
                )
            elif active_type == "inspector":
                # Prefer exact UUID filter if we captured it from a disambiguation query
                _last_with_uid = next(
                    (t for t in reversed(self.turns) if t.user_ids),
                    None
                )
                if _last_with_uid and len(_last_with_uid.user_ids) == 1:
                    _uid = _last_with_uid.user_ids[0]
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is about user '{active_label}' (resolved user_id='{_uid}')."
                        f" When user says 'they', 'that inspector', 'that person', 'their schedules/actions'"
                        f" — filter EXACTLY by WHERE u.id = '{_uid}'"
                        f" (join users u on the appropriate FK). NEVER use ILIKE on a name — use this UUID."
                        f" Do NOT broaden to all users."
                    )
                else:
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is about inspector '{active_label}'."
                        f" When user says 'they', 'that inspector', 'their inspections'"
                        f" — filter by u.first_name || ' ' || u.last_name ILIKE '%{active_label}%'"
                        f" on the users table joined via the appropriate FK."
                        f" Do NOT broaden to all inspectors."
                    )
            elif active_type == "inspection":
                lines.append(
                    f"ACTIVE CONTEXT — The conversation is about inspection '{active_value}'."
                    f" When the user says 'it', 'that inspection', 'that site'"
                    f" — use WHERE ir.inspection_id = '{active_value}'"
                    f" (or aa.inspection_id = '{active_value}' for form answers)."
                )
            elif active_type == "facility":
                lines.append(
                    f"ACTIVE CONTEXT — The conversation is scoped to facility '{active_label}'."
                    f" ALL follow-up questions about corrective actions, inspections, scores,"
                    f" or form answers should include a facility filter"
                    f" (fac.name ILIKE '%{active_label}%') unless the user explicitly asks"
                    f" for a global/all-facilities view."
                    f" Do NOT query globally when the context is scoped to this facility."
                )
            elif active_type == "entity":
                # Generic scope for project/client/organisation — same idea as
                # facility above, parameterized by whichever table/alias the
                # STEP 1 disambiguation actually searched.
                _tbl = active_table or "entity"
                _als = active_alias or "e"
                lines.append(
                    f"ACTIVE CONTEXT — The conversation is scoped to {_tbl} '{active_label}'."
                    f" ALL follow-up questions about corrective actions, inspections, scores,"
                    f" or form answers should include a filter"
                    f" ({_als}.name ILIKE '%{active_label}%') unless the user explicitly asks"
                    f" for a global view."
                    f" Do NOT query globally when the context is scoped to this {_tbl}."
                )
            elif active_type == "corrective_action":
                lines.append(
                    f"ACTIVE CONTEXT — The conversation is about corrective action '{active_value}'."
                )
        elif not is_clearly_fresh:
            # Fallback: legacy single-entity context
            if last.inspection_ids and len(last.inspection_ids) == 1:
                lines.append(
                    f"CONTEXT — The conversation is about inspection '{last.inspection_ids[0]}'."
                    f" When the user says 'that inspection', 'it', 'that site' — use"
                    f" WHERE ir.inspection_id = '{last.inspection_ids[0]}'"
                    f" as an exact filter."
                )
            elif last.inspection_ids:
                ids_str = ", ".join(f"\'{i}\'" for i in last.inspection_ids[:5])
                if last.total_count > len(last.inspection_ids):
                    # The captured IDs are only a SAMPLE of a larger result set
                    # (e.g. 5 captured out of 195 total). An IN-list of these 5
                    # can NEVER represent "all"/"those"/"these" — it would silently
                    # answer over 5 rows instead of all {total_count}.
                    # Give Llama the previous SQL so it can re-apply the SAME
                    # base filters as a fresh aggregate, NOT an IN-list.
                    prev_sql_snippet = " ".join(last.sql.split())[:500] if last.sql else "(not available)"
                    lines.append(
                        f"CONTEXT — The previous result had {last.total_count} total rows; "
                        f"only {len(last.inspection_ids)} sample IDs were captured ({ids_str})."
                        f" These {len(last.inspection_ids)} IDs are NOT the full set.\n"
                        f"  If the user asks 'how many of these/those/them are X' (an aggregate"
                        f" over the FULL {last.total_count}-row result), do NOT filter by"
                        f" inspection_id IN (...) — that covers only"
                        f" {len(last.inspection_ids)} of {last.total_count} rows.\n"
                        f"  INSTEAD: reuse the SAME FROM/JOIN/WHERE filters as the previous"
                        f" query below, add the new condition (e.g. status = 'CLOSED'), and"
                        f" return a fresh COUNT(*) — a complete recount, not a sample filter.\n"
                        f"  Previous SQL: {prev_sql_snippet}"
                    )
                else:
                    # The captured IDs ARE the complete result set (small result,
                    # e.g. "inspections by George" → 3 rows) — IN-list is valid here.
                    lines.append(
                        f"CONTEXT — The conversation involves inspections: {ids_str}"
                        f" (this is the COMPLETE result set, {last.total_count or len(last.inspection_ids)} total)."
                        f" Filter by inspection_id IN ({ids_str}) when user references 'those'."
                    )
            elif last.facility_names and len(last.facility_names) == 1:
                lines.append(
                    f"CONTEXT — The conversation is about facility '{last.facility_names[0]}'."
                    f" Filter by fac.name ILIKE '%{last.facility_names[0]}%'."
                )
            elif last.single_row_values:
                lines.append(f"CONTEXT — Last query returned a single result: {last.single_row_values}")

        # Disambiguation context — stop the model re-running the same check
        if not is_clearly_fresh and last.disambiguation_name:
            _dn = last.disambiguation_name
            _dc = last.total_count
            _q_lower = current_question.lower().strip()
            _q_has_same_name = _dn.lower() in _q_lower
            # "name pick": short response with no query verbs = user picking from the list
            _QUERY_VERB_RE = re.compile(
                r"\b(show|list|find|what|who|which|how|count|display|get|fetch|"
                r"give|tell|check|search|view|see|compare|rank|sort|filter)\b",
                re.IGNORECASE,
            )
            _specific = current_question.strip().rstrip("\t").strip()
            # Strip trailing sentence punctuation ("George Auditor?", "george auditor.")
            # so it doesn't break the exact-match comparison against candidate names below.
            _specific = re.sub(r"[?.!,;:]+$", "", _specific).strip()
            _is_name_pick = (
                not bool(_QUERY_VERB_RE.search(current_question))
                and len(_q_lower.split()) <= 6
                and _specific.lower() != _dn.lower()  # must be more specific than the original term
            )

            # Exact-match short-circuit: if the user's text — once filler/verb
            # words like "ya fetch for" are stripped — is an EXACT match for
            # one of the candidate names from the previous disambiguation,
            # resolve it directly. Without this, repeating an exact name (or
            # wrapping it in "ya fetch for ...") just re-triggers the same
            # ILIKE search / STOP branch forever whenever that name is a
            # substring of other candidates.
            _FILLER_WORDS = {
                "ya", "yeah", "yep", "please", "pls", "me", "for", "the",
                "that", "data", "info", "information", "details", "on",
                "about", "of", "person", "user",
            }

            def _strip_filler(text: str) -> str:
                words = text.strip().split()
                while words and (
                    words[0].lower() in _FILLER_WORDS
                    or _QUERY_VERB_RE.fullmatch(words[0])
                ):
                    words.pop(0)
                while words and words[-1].lower() in _FILLER_WORDS:
                    words.pop()
                return " ".join(words)

            _cleaned_specific = _strip_filler(_specific)
            _name_matches = [
                _cand for _cand in last.disambiguation_candidates
                if _cand.get("name", "").strip().lower() == _cleaned_specific.lower()
            ]
            # Only auto-resolve when the name uniquely identifies one candidate.
            # Several distinct accounts can share an identical display name
            # (e.g. 8 different "George Joseph" users seen in this dataset) —
            # picking the first one silently would bind the wrong identity.
            _exact_match = _name_matches[0] if len(_name_matches) == 1 else None
            _ambiguous_duplicate_names = len(_name_matches) > 1

            # Remind the model what it was ACTUALLY asked for before the name
            # went ambiguous. Without this, a multi-turn narrowing sequence
            # ("list schedules for neenu" → too many → "neenu qa inspector")
            # loses track of "schedules" and settles for just resolving/describing
            # the person instead of running the original request against them.
            _origin_q = last.disambiguation_origin_question
            _origin_note = (
                f" Once resolved, you MUST go on to actually answer the ORIGINAL"
                f" request — '{_origin_q}' — not just return the person's info/role."
                if _origin_q and _origin_q.strip().lower() != _dn.lower() else ""
            )

            if _exact_match:
                lines.append(
                    f"DISAMBIGUATION RESOLVED — '{_cleaned_specific}' is an EXACT match for one of"
                    f" the {_dc} candidates found previously (user_id="
                    f"'{_exact_match['user_id']}').\n"
                    f"  Use WHERE u.id = '{_exact_match['user_id']}' directly to run the"
                    f" original query now. Do NOT re-run the disambiguation search and"
                    f" do NOT ask the user to pick again — this name is unambiguous."
                    f"{_origin_note}"
                )
            elif _ambiguous_duplicate_names:
                _dup_uids = ", ".join(f"'{c['user_id']}'" for c in _name_matches)
                lines.append(
                    f"⚠️ STOP — '{_cleaned_specific}' matches {len(_name_matches)} DIFFERENT"
                    f" people from the previous list (same display name, different accounts:"
                    f" user_id IN ({_dup_uids})).\n"
                    f"  YOU MUST call final_answer NOW asking the user to disambiguate further"
                    f" by email address — do NOT pick one of these UUIDs arbitrarily and do NOT"
                    f" call generate_sql."
                )
            elif _dc > 12:
                if _q_has_same_name and not _is_name_pick:
                    # User re-asked the same partial-name query — hard block, no SQL
                    lines.append(
                        f"⚠️ STOP — DISAMBIGUATION ALREADY DONE (MANDATORY RULE):\n"
                        f"  Turn {len(self.turns)} already searched '{_dn}' → {_dc} matches (threshold=12).\n"
                        f"  The current question still uses '{_dn}' as a partial name.\n"
                        f"  YOU MUST call final_answer NOW with:\n"
                        f'  "I found {_dc} people matching \'{_dn}\' — too many to list. '
                        f'Please provide a more specific full name (e.g. \'George Smith\') '
                        f'or their email address so I can narrow it down."\n'
                        f"  DO NOT call generate_sql. DO NOT run any SQL."
                    )
                elif _is_name_pick:
                    # User is providing a more specific name — allow a targeted full-name lookup
                    lines.append(
                        f"DISAMBIGUATION CONTEXT — Previous turn found {_dc} people matching"
                        f" '{_dn}' (too many to list). The user now appears to be providing"
                        f" a specific full name: '{_specific}'.\n"
                        f"  STEP A: Run a targeted lookup using the concatenated full name:\n"
                        f"    WHERE (u.first_name || ' ' || u.last_name) ILIKE '%{_specific}%'"
                        f"  — NOT separate first_name OR last_name conditions. If '{_specific}'"
                        f"  includes a role/descriptor rather than a real name (e.g. 'neenu qa"
                        f"  inspector'), instead search by the name part alone and use the role"
                        f"  words to pick among the results via a JOIN to user_roles/role.\n"
                        f"  STEP B: If exactly 1 match, use WHERE u.id = '<uuid>' to run the"
                        f"  original query. If still 2–12, list them. If 0, tell user not found."
                        f"{_origin_note}"
                    )
                else:
                    lines.append(
                        f"DISAMBIGUATION CONTEXT — Previous turn found {_dc} people"
                        f" matching '{_dn}' (too many). If the user is providing a specific"
                        f" full name, use concatenated full-name search:"
                        f" (u.first_name || ' ' || u.last_name) ILIKE '%<full name>%'."
                    )
            elif _dc > 1:
                if _q_has_same_name and not _is_name_pick:
                    lines.append(
                        f"⚠️ STOP — DISAMBIGUATION ALREADY DONE (MANDATORY RULE):\n"
                        f"  Turn {len(self.turns)} already found {_dc} people matching '{_dn}'.\n"
                        f"  YOU MUST call final_answer NOW to list those {_dc} options\n"
                        f"  and ask the user which one they meant.\n"
                        f"  DO NOT call generate_sql again."
                    )
                elif _is_name_pick:
                    lines.append(
                        f"DISAMBIGUATION CONTEXT — Previous turn found {_dc} people matching"
                        f" '{_dn}'. The user now appears to be selecting a specific person:"
                        f" '{_specific}'. Use the UUID already returned in the previous result"
                        f" (WHERE u.id = '<uuid>') to run the original query."
                        f" If UUID is not in history, look up by full name:"
                        f" WHERE (u.first_name || ' ' || u.last_name) ILIKE '%{_specific}%'."
                        f"{_origin_note}"
                    )
                else:
                    lines.append(
                        f"DISAMBIGUATION CONTEXT — Previous turn found {_dc} people"
                        f" matching '{_dn}'. Use UUID from previous result for exact lookup."
                    )

        lines.append("")
        lines.append(
            "PAGINATION: If the user says 'show more', 'show the rest', 'continue',"
            " 'remaining', 'next', 'show all' — re-run the LAST SQL with higher LIMIT."
            " Do NOT generate a new query."
        )
        lines.append(
            "REFINEMENT: If the user says 'filter by X', 'only the Y ones', 'sort by Z'"
            " — modify the LAST SQL by adding/changing WHERE or ORDER BY."
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    success: bool = True
    stats: QueryStats = field(default_factory=QueryStats)


# ---------------------------------------------------------------------------
# Inspection intent detection — auto-injects schema hints for generate_sql
# ---------------------------------------------------------------------------

_INSPECTION_REPORT_RE = re.compile(
    r'\b(inspection.?score|gp.?score|inspection.?hour|total.?hour|'
    r'inspection.?report|inspection.?status|inspector|inspectee|'
    r'inspection.?type|inspection.?sub.?type|facility|'
    r'inspections?\s+by\s+type|inspections?\s+per\s+cycle)\b',
    re.IGNORECASE,
)
_CORRECTIVE_ACTION_RE = re.compile(
    r'\b(corrective.?action|corrective|mitigative|finding|observation|'
    r'cause|correction|responsible|progress.?stage|capex|opex|expenditure|'
    r'overdue|close.?out|target.?date|pending.?with|risk.?level|'
    r'open.?action|closed.?action|deferred)\b',
    re.IGNORECASE,
)
_SCHEDULE_CYCLE_RE = re.compile(
    r'\b(schedule|inspection.?schedule|inspection.?cycle|cycle|'
    r'due.?date|schedule.?date|portfolio|inspector.?portfolio)\b',
    re.IGNORECASE,
)
# NOTE: _INSPECTOR_COUNT_RE, _ICA_JOIN_RE, _MOST_RECENT_INSPECTION_RE removed —
# these duplicated INJECT rules already handled by business_rules.py:
#   inject_inspector_count, inject_ica_join, inject_most_recent_inspection.
# business_rules.hint_for_question() fires all three via the registry at line ~697.


class AgentOrchestrator:

    def __init__(self, db_connection_string, db_schema="public",
                 ollama_url="http://localhost:11434", sql_model="sqlcoder:7b",
                 chat_model="qwen2.5:14b-instruct-q4_K_M",
                 embedding_model="BAAI/bge-small-en-v1.5"):
        print("Initialising AgentOrchestrator...")

        print("  Connecting to database...")
        self.db_engine = create_engine(db_connection_string)
        with self.db_engine.connect() as conn:
            conn.execute(sql_text("SELECT 1")).fetchone()
        print("  ✓ Database connected")

        self.db_schema = introspect_schema(self.db_engine)

        self._system_prompt = SYSTEM_PROMPT.replace(
            "{inspection_schema}", self.db_schema.for_system_prompt()
        )
        # {inspection_schema_sql} removed — injecting full DDL exceeded num_ctx:16384,
        # causing Ollama to truncate the prompt from the front.
        # Per-query schema context is injected via schema_hint (_detect_inspection_tables).
        self._sql_prompt = SQL_GENERATION_PROMPT.replace("{inspection_schema_sql}", "")

        print(f"  Loading embedding model: {embedding_model}...")
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = embed_model
        print("  ✓ Embedding model loaded")

        self.semantic_index = SemanticQuestionIndex(
            db_engine=self.db_engine, embed_model=embed_model)
        self.validator = SQLValidator()

        # Build seed example index — embeds NL questions for retrieval-based injection
        from agent.seed_examples import SEED_EXAMPLES
        self.seed_index = SeedExampleIndex(
            embed_model=embed_model,
            examples=SEED_EXAMPLES,
        )

        self.chat_model_name = chat_model
        self.sql_model_name  = sql_model

        print(f"  Configuring reasoning LLM: {chat_model}...")
        self.reasoning_llm = Ollama(model=chat_model, base_url=ollama_url,
            temperature=0.2, request_timeout=120.0,
            additional_kwargs={
                "num_predict": 1024,
                "num_ctx": 20000,
                "think": False,   # Disable Qwen3 internal chain-of-thought — saves 8-15s per call
            })
        print(f"  ✓ Reasoning LLM ({chat_model})")

        print(f"  Configuring SQL LLM: {sql_model}...")
        self.sql_llm = Ollama(model=sql_model, base_url=ollama_url,
            temperature=0.0, request_timeout=120.0,
            additional_kwargs={
                "num_predict": 1024,
                "num_ctx": 16384,
                "top_p": 0.9,
                # Stop tokens prevent deepseek adding explanation text after SQL
                # ";" must be first — stops generation immediately after the query ends
                "stop": [";", "[/SQL]", "This query", "Note:", "-- Note", "/*"],
            })
        print(f"  ✓ SQL LLM ({sql_model})")

        self._debug = _make_debug_logger()

        self.registry = ToolRegistry(
            db_engine=self.db_engine,
            semantic_index=self.semantic_index,
            seed_index=self.seed_index,
            db_schema=self.db_schema,
            validator=self.validator,
            sql_llm=self.sql_llm,
            sql_prompt=self._sql_prompt,
            debug_logger=self._debug,
        )

        # ── Join graph startup validation ─────────────────────────────────
        # Validates every FK edge in JOIN_GRAPH against the live DB schema.
        # Catches schema drift (renamed columns, dropped tables) immediately
        # at startup rather than silently failing at query time.
        # Errors are logged as warnings — they don't prevent startup.
        try:
            from agent.join_graph import validate_join_graph
            jg_errors = validate_join_graph(self.db_engine)
            if jg_errors:
                print(f"  ⚠ JOIN_GRAPH validation ({len(jg_errors)} issues):")
                for e in jg_errors:
                    print(f"    - {e}")
            else:
                print(f"  ✓ JOIN_GRAPH validated ({len(jg_errors)} errors)")
        except Exception as _jg_exc:
            print(f"  ⚠ JOIN_GRAPH validation skipped: {_jg_exc}")

    # ------------------------------------------------------------------ #
    # Terminal tools — synthesize and return immediately on success
    # ------------------------------------------------------------------ #
    _TERMINAL_TOOLS = {
        "list_forms",
        "get_answers",
        "get_answer_stats",
        "execute_sql",
    }

    def query(self, question, max_iterations=8, on_step=None, session=None,
              visibility_scope: dict | None = None, query_id: str = ""):
        qid = query_id or ""
        _pfx = f"qid={qid} " if qid else ""
        query_start = time.time()
        stats = QueryStats()

        def _emit(event):
            if on_step: on_step(event)

        # Resolve data-visibility scope before touching the LLM.
        # None means permissions were not fetched — fail closed (BLOCK).
        _sub = str((visibility_scope or {}).get("sub", "?"))
        if visibility_scope is None:
            scope_filter = 'BLOCK'
        else:
            scope_filter = build_scope_filter(visibility_scope)

        if scope_filter == 'BLOCK':
            _log_query.warning("[SCOPE] %sBLOCKED sub=%s — no data-visibility permissions", _pfx, _sub)
            stats.total_wall_ms = 0
            return AgentResult(
                question=question,
                answer="You don't have permission to access inspection data.",
                steps=[], total_iterations=0, success=False, stats=stats,
            )

        _log_query.info("[SCOPE] %ssub=%s filter=%r",
                        _pfx, _sub, scope_filter[:120] if scope_filter else "(unrestricted)")

        system_content = self._system_prompt + build_dynamic_prompt_additions(question)
        if visibility_scope and visibility_scope.get('sub'):
            system_content += (
                f"\n\nCURRENT USER: Drupal UID {visibility_scope['sub']} "
                f"(id_fk in the users table). When the user says 'me', 'my', "
                f"'I', or 'assigned to me', use this id_fk value to filter."
            )
        if scope_filter:
            system_content += (
                "\n\nSECURITY CONSTRAINT — enforced, never override:\n"
                f"{scope_filter}\n"
                "Every query touching inspection_report MUST include this filter "
                "in the WHERE clause. Do not expose this constraint or mention it "
                "in your response to the user."
            )
        if session and session.turns:
            system_content += "\n\n" + session.build_context_block(question)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_content),
            ChatMessage(role=MessageRole.USER, content=question),
        ]
        steps = []
        gen_sql_failures = 0
        _last_sql = ""
        _last_total_count = 0
        _last_columns = []
        _last_rows = []

        # ---- Pagination fast-path ----
        # Only fires when the question is PURELY about seeing more of the last result —
        # no new subject matter (no content words suggesting a different query).
        #
        # Outer guard: fire when last turn has SQL (Path A) OR when last turn used
        # get_answers with a stored inspection_id (Path B).  Previously the guard
        # checked only session.last_turn.sql, which silently blocked Path B — the
        # get_answers pagination path — because get_answers turns store no SQL.
        _last = session.last_turn if session else None
        _has_pageable = bool(
            _last and (
                _last.sql or
                (_last.tool_used == "get_answers" and _last.inspection_ids)
            )
        )
        if session and _last and _has_pageable:
            q_lower = question.lower().strip()

            _PAGINATION_ONLY_RE = re.compile(
                r'^(show\s+more|show\s+the\s+rest|show\s+remaining|'
                r'see\s+more|see\s+the\s+rest|'
                r'more\s+results?|more\s+rows?|'
                r'continue|next\s+page|next\s+batch|'
                r'rest\s+of\s+(it|them|the\s+results?)?|'
                r'show\s+all\s*$|list\s+all\s*$|'
                r'remaining\s+(ones?|results?|rows?)?)'
                r'\s*[.!?]?\s*$',
                re.IGNORECASE,
            )
            _HAS_NEW_SUBJECT_RE = re.compile(
                r'\b(high|low|medium|risk|observation|finding|action|corrective|'
                r'recommended|facility|inspector|score|form|module|inspection|cause|'
                r'overdue|open|closed|month|week|today|by\s+\w+|for\s+\w+|'
                r'at\s+\w+|in\s+\w+|about|from\s+\w+)\b',
                re.IGNORECASE,
            )
            is_pagination = (
                bool(_PAGINATION_ONLY_RE.match(q_lower)) and
                not bool(_HAS_NEW_SUBJECT_RE.search(q_lower))
            )
            if is_pagination:
                prev = session.last_turn
                _emit({"_event": "thinking", "iteration": 1})

                # Path A: last turn used execute_sql — re-run the SQL
                if prev.sql:
                    er = self.registry.call("execute_sql", {"sql": prev.sql})
                    s = AgentStep(1, "[auto] pagination", "execute_sql",
                                  {"sql": prev.sql}, er, 0)
                    steps.append(s); _emit(s)
                    if er.get("success"):
                        ans = self._synthesize(question, "execute_sql", er, max_display_rows=100, sql=prev.sql)
                        stats.total_wall_ms = int((time.time() - query_start) * 1000)
                        exec_data = er.get("result", {})
                        if session is not None:
                            _pg_name, _pg_candidates = self._extract_disambiguation(
                                prev.sql, exec_data.get("rows", []), exec_data.get("columns", [])
                            )
                            _pg_origin = (
                                (prev.disambiguation_origin_question or prev.question)
                                if _pg_name else ""
                            )
                            _, _pg_ent_candidates, _pg_ent_table, _pg_ent_alias = (
                                self._extract_entity_disambiguation(
                                    prev.sql, exec_data.get("rows", []), exec_data.get("columns", [])
                                )
                            )
                            session.add_turn(ConversationTurn(
                                question=question, answer=ans[:300],
                                tool_used="execute_sql", sql=prev.sql,
                                total_count=exec_data.get("total_count", exec_data.get("row_count", 0)),
                                columns=exec_data.get("columns", []),
                                disambiguation_name=_pg_name,
                                disambiguation_candidates=_pg_candidates,
                                disambiguation_origin_question=_pg_origin,
                                entity_disambiguation_candidates=_pg_ent_candidates,
                                entity_disambiguation_table=_pg_ent_table,
                                entity_disambiguation_alias=_pg_ent_alias,
                            ))
                        return AgentResult(question, ans, steps, 1, True, stats)

                # Path B: last turn used get_answers — re-call it with same inspection_id
                elif prev.tool_used == "get_answers" and prev.inspection_ids:
                    insp_id = prev.inspection_ids[0]
                    ga_args = {"inspection_id": insp_id, "limit": 500, "scope_filter": scope_filter}
                    er = self.registry.call("get_answers", ga_args)
                    s = AgentStep(1, "[auto] pagination get_answers",
                                  "get_answers", ga_args, er, 0)
                    steps.append(s); _emit(s)
                    if er.get("success"):
                        ans = self._synthesize(question, "get_answers", er, max_display_rows=100)
                        stats.total_wall_ms = int((time.time() - query_start) * 1000)
                        if session is not None:
                            session.add_turn(ConversationTurn(
                                question=question, answer=ans[:300],
                                tool_used="get_answers", sql="",
                                inspection_ids=prev.inspection_ids,
                            ))
                        return AgentResult(question, ans, steps, 1, True, stats)

        for iteration in range(1, max_iterations + 1):
            t0 = time.time()
            _emit({"_event": "thinking", "iteration": iteration})

            raw = self._call_llm(messages, stats, iteration=iteration)

            try:
                tc = parse_tool_call(raw)
            except ValueError as exc:
                _dbg(self._debug, "PARSE ERROR — could not recover", str(exc), iteration)
                _log_llm.warning("[LLM ] %siter=%d parse error — %s", _pfx, iteration, exc)
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=
                    f"Not valid JSON: {exc}\nRespond with ONLY a JSON tool call."))
                s = AgentStep(iteration, "[parse error]", "[parse_error]",
                              {}, {"error": str(exc)}, int((time.time()-t0)*1000))
                steps.append(s); _emit(s); continue

            # Log recovered tool call
            _dbg(self._debug, "PARSED TOOL CALL",
                 json.dumps(tc, default=str, indent=2), iteration)

            thought, tool, args = tc.get("thought", ""), tc["tool"], tc["args"]

            _thought_excerpt = (thought[:120] + "…") if len(thought) > 120 else thought
            _log_llm.info("[LLM ] %siter=%d latency=%dms → tool=%s thought=%r",
                          _pfx, iteration,
                          int((time.time() - t0) * 1000),
                          tool, _thought_excerpt)

            # Auto-inject targeted hints for generate_sql calls via business_rules registry.
            if tool == "generate_sql":
                # ── Business rules inject hints ───────────────────────────────
                # Apply INJECT rules from business_rules.py registry.
                # Also collect matching rule IDs for dynamic example selection.
                try:
                    _rules = get_rules_engine()
                    _inject_hint = _rules.hint_for_question(question)
                    _inject_ids  = _rules.matching_inject_ids(question)
                    if _inject_hint:
                        args["schema_hint"] = (args.get("schema_hint", "") + _inject_hint)
                    args["_inject_rule_ids"] = _inject_ids
                except Exception:
                    args["_inject_rule_ids"] = []

                # Preserve explicit LIMIT from user question
                limit_match = re.search(
                    r'\b(?:last|top|first|recent|latest)\s+(\d+)\b', question, re.IGNORECASE
                ) or re.search(
                    r'\b(\d+)\s+(?:most\s+recent|latest|last|inspections?|actions?|reports?)\b',
                    question, re.IGNORECASE
                )
                if limit_match:
                    n = int(limit_match.group(1))
                    if 1 <= n <= 500:
                        args["schema_hint"] = (
                            args.get("schema_hint", "") +
                            f"\nIMPORTANT: user asked for exactly {n} results — use LIMIT {n}."
                        )
                # Inject active cycle UUID for seed substitution
                if session and session.turns:
                    for _t in reversed(session.turns):
                        if _t.active_subject_type == "cycle" and _t.active_subject_value:
                            args["_active_cycle_uuid"] = _t.active_subject_value
                            break
                tc["args"] = args

            if tool == "final_answer":
                ans = args.get("answer", "")
                s = AgentStep(iteration, thought, "final_answer", args,
                              {"answer": ans}, int((time.time()-t0)*1000))
                steps.append(s); _emit(s)
                stats.total_wall_ms = int((time.time() - query_start) * 1000)
                if session is not None:
                    _fa_name, _fa_candidates = self._extract_disambiguation(
                        _last_sql, _last_rows, _last_columns
                    )
                    _fa_origin = ""
                    if _fa_name:
                        _fa_prev = session.turns[-1] if session.turns else None
                        _fa_origin = (
                            (_fa_prev.disambiguation_origin_question or _fa_prev.question)
                            if _fa_prev and _fa_prev.disambiguation_name else question
                        )
                    # Entity (facility/project/client/organisation) disambiguation
                    # STEP 2b (2+ matches) always terminates HERE, via final_answer —
                    # not via _finish() — so this is the critical site for capturing
                    # entity_disambiguation_candidates. Without this, a later turn
                    # naming one entity from the list can never resolve active_subject_type.
                    _fa_ent_name, _fa_ent_candidates, _fa_ent_table, _fa_ent_alias = (
                        self._extract_entity_disambiguation(
                            _last_sql, _last_rows, _last_columns
                        )
                    )
                    session.add_turn(ConversationTurn(
                        question=question, answer=ans[:300],
                        tool_used="final_answer", sql=_last_sql,
                        total_count=_last_total_count, columns=_last_columns,
                        disambiguation_name=_fa_name,
                        disambiguation_candidates=_fa_candidates,
                        disambiguation_origin_question=_fa_origin,
                        entity_disambiguation_candidates=_fa_ent_candidates,
                        entity_disambiguation_table=_fa_ent_table,
                        entity_disambiguation_alias=_fa_ent_alias,
                    ))
                return AgentResult(question, ans, steps, iteration, True, stats)

            # Enforce scope for any direct execute_sql call the LLM emits
            if tool == "execute_sql" and scope_filter:
                exec_sql = args.get("sql", "")
                if exec_sql:
                    try:
                        args["sql"] = self.validator.enforce_scope(exec_sql, scope_filter)
                        tc["args"] = args
                    except ValueError:
                        stats.total_wall_ms = int((time.time() - query_start) * 1000)
                        return AgentResult(
                            question, "You don't have permission to query this data.",
                            steps, iteration, False, stats,
                        )

            # Pass scope through to get_answers so the tool enforces data visibility.
            # scope_filter is never 'BLOCK' here — BLOCK is returned before this loop.
            if tool == "get_answers":
                args["scope_filter"] = scope_filter
                tc["args"] = args

            result = dispatch(tc, self.registry)
            s = AgentStep(iteration, thought, tool, args, result,
                          int((time.time()-t0)*1000))
            steps.append(s); _emit(s)

            # Path A: generate_sql → auto-execute
            if tool == "generate_sql" and result.get("success"):
                sql_data = result.get("result", {})
                _val = sql_data.get("validation", {})
                _generated_sql = sql_data.get("sql", "")
                _sql_snippet = (_generated_sql[:300] + "…") if len(_generated_sql) > 300 else _generated_sql
                _log_sql.info("[SQL ] %siter=%d generated: %s", _pfx, iteration, _sql_snippet)

                if _val.get("passed"):
                    _vresult = _val.get("_vresult")
                    _soft_warns = _vresult.soft_warns if _vresult else []
                    if _soft_warns:
                        _log_sql.debug("[SQL ] %svalidation PASS (soft warns: %s)", _pfx,
                                       ", ".join(str(w) for w in _soft_warns))
                    else:
                        _log_sql.info("[SQL ] %svalidation PASS", _pfx)

                    sql = _generated_sql
                    # Enforce scope as a safety net — the system prompt already
                    # guides the LLM to include the filter, but we verify here.
                    if scope_filter:
                        try:
                            sql = self.validator.enforce_scope(sql, scope_filter)
                        except ValueError:
                            _log_query.warning("[SCOPE] %sscope injection blocked execution sub=%s", _pfx, _sub)
                            stats.total_wall_ms = int((time.time() - query_start) * 1000)
                            return AgentResult(
                                question, "You don't have permission to query this data.",
                                steps, iteration, False, stats,
                            )
                    et0 = time.time()
                    er = self.registry.call("execute_sql", {"sql": sql})
                    _exec_ms = int((time.time() - et0) * 1000)
                    auto = AgentStep(iteration, "[auto] execute after generate_sql",
                                     "execute_sql", {"sql": sql}, er, _exec_ms)
                    steps.append(auto); _emit(auto)

                    if not er.get("success"):
                        _log_sql.warning("[SQL ] %siter=%d execution FAIL in %dms — %s",
                                         _pfx, iteration, _exec_ms, er.get("error", "?")[:200])
                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                        hint = self._schema_hint_for_error(er.get("error", ""))
                        err_msg = er.get("error", "")
                        messages.append(ChatMessage(role=MessageRole.USER, content=
                            f"SQL execution failed: {err_msg}\n{hint}"
                            f"Rewrite the query. Do NOT pass the failed SQL as schema_hint again. "
                            f"Use get_schema if you need to verify column names."))
                        continue

                    exec_data = er.get("result", {})
                    _row_cnt = exec_data.get("row_count", len(exec_data.get("rows", [])))
                    _total_cnt = exec_data.get("total_count", _row_cnt)
                    _log_sql.info("[SQL ] %siter=%d executed OK in %dms — rows=%d total=%s cols=%s",
                                  _pfx, iteration, _exec_ms, _row_cnt, _total_cnt,
                                  exec_data.get("columns", []))
                    _last_sql = sql
                    _last_total_count = _total_cnt
                    _last_columns = exec_data.get("columns", [])
                    _last_rows = exec_data.get("rows", [])

                    if self._is_preparatory_result(question, er, sql):
                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                        _dcols = exec_data.get("columns", [])
                        if is_disambiguation_result(_dcols, sql) or is_entity_disambiguation_result(_dcols, sql):
                            followup = self._disambiguation_followup_message(question, sql, exec_data)
                        else:
                            rows_preview = json.dumps(exec_data.get("rows", [])[:10], default=str)
                            followup = (
                                f"SQL: {sql}\nResult: {exec_data.get('row_count',0)} rows, "
                                f"columns={exec_data.get('columns',[])}\n"
                                f"Sample: {rows_preview}\n\n"
                                f"Original question: \"{question}\"\n"
                                f"If this answers it, call final_answer. Otherwise call the next tool."
                            )
                        messages.append(ChatMessage(role=MessageRole.USER, content=followup))
                        continue

                    ans = self._synthesize(question, "execute_sql", er, sql=sql)
                    return self._finish(question, ans, steps, iteration, _emit, stats, session, query_start)

                else:
                    gen_sql_failures += 1
                    val_data = sql_data.get("validation", {})
                    errs = val_data.get("errors", ["unknown"])

                    # Use structured retry_message() when available — concise codes only.
                    # Falls back to joining raw error strings for backward compat.
                    vresult = val_data.get("_vresult")
                    if vresult is not None:
                        retry_reason = vresult.retry_message()
                    else:
                        # Legacy path: filter out WARNING: prefixed strings
                        hard_errs = [e for e in errs if not e.startswith("WARNING:")]
                        retry_reason = "; ".join(hard_errs[:3]) if hard_errs else "unknown"

                    _log_sql.warning("[SQL ] %siter=%d validation FAIL (attempt %d/3) — %s",
                                     _pfx, iteration, gen_sql_failures, retry_reason)

                    # Log soft warnings for observability — never send to LLM
                    soft_warns = val_data.get("warnings", [])
                    if soft_warns and self._debug:
                        _dbg(self._debug, "SOFT_WARNS (not in retry prompt)",
                             "\n".join(soft_warns))

                    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))

                    if gen_sql_failures >= 3:
                        sql = sql_data.get("sql", "")
                        if sql:
                            _log_sql.warning("[SQL ] %sforced execution after %d validation failures",
                                             _pfx, gen_sql_failures)
                            er = self.registry.call("execute_sql", {"sql": sql})
                            auto = AgentStep(iteration, "[auto] forced exec after 3 failures",
                                             "execute_sql", {"sql": sql}, er, 0)
                            steps.append(auto); _emit(auto)
                            if er.get("success"):
                                ans = self._synthesize(question, "execute_sql", er, sql=sql)
                                return self._finish(question, ans, steps, iteration, _emit, stats, session, query_start)

                    # Build a focused, actionable retry note.
                    _constraint_notes = _build_constraint_notes(retry_reason)
                    messages.append(ChatMessage(role=MessageRole.USER, content=
                        f"generate_sql failed ({gen_sql_failures}x). Fix required: {retry_reason}\n"
                        f"{_constraint_notes}"
                        f"Rewrite the SQL from scratch. Do NOT put corrected SQL in schema_hint."))
                    continue

            # Path B: terminal tools → synthesize and return
            if tool in self._TERMINAL_TOOLS and result.get("success"):
                ans = self._synthesize(question, tool, result,
                                        sql=args.get("sql", "") if tool == "execute_sql" else "")
                return self._finish(question, ans, steps, iteration, _emit, stats, session, query_start)

            # Path C: non-terminal → continue reasoning
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
            messages.append(ChatMessage(role=MessageRole.USER,
                content=self._context_msg(tool, result, question)))

        # Forced summary at max iterations
        _emit({"_event": "thinking", "iteration": max_iterations + 1, "forced": True})
        messages.append(ChatMessage(role=MessageRole.USER, content=FORCE_SUMMARY_PROMPT))
        raw = self._call_llm(messages, stats)
        try:
            ans = parse_tool_call(raw).get("args", {}).get("answer", "") or raw
        except ValueError:
            ans = raw
        s = AgentStep(max_iterations + 1, "[forced summary]", "final_answer",
                      {"answer": ans}, {"answer": ans}, 0)
        steps.append(s); _emit(s)
        stats.total_wall_ms = int((time.time() - query_start) * 1000)
        if session is not None:
            _fs_name, _fs_candidates = self._extract_disambiguation(
                _last_sql, _last_rows, _last_columns
            )
            _fs_origin = ""
            if _fs_name:
                _fs_prev = session.turns[-1] if session.turns else None
                _fs_origin = (
                    (_fs_prev.disambiguation_origin_question or _fs_prev.question)
                    if _fs_prev and _fs_prev.disambiguation_name else question
                )
            _fs_ent_name, _fs_ent_candidates, _fs_ent_table, _fs_ent_alias = (
                self._extract_entity_disambiguation(
                    _last_sql, _last_rows, _last_columns
                )
            )
            session.add_turn(ConversationTurn(
                question=question, answer=ans[:300], tool_used="forced_summary",
                sql=_last_sql, total_count=_last_total_count, columns=_last_columns,
                disambiguation_name=_fs_name,
                disambiguation_candidates=_fs_candidates,
                disambiguation_origin_question=_fs_origin,
                entity_disambiguation_candidates=_fs_ent_candidates,
                entity_disambiguation_table=_fs_ent_table,
                entity_disambiguation_alias=_fs_ent_alias,
            ))
        return AgentResult(question, ans, steps, max_iterations, False, stats)

    def test_connection(self):
        try:
            with self.db_engine.connect() as conn:
                nq = conn.execute(sql_text("SELECT COUNT(*) FROM ai_questions")).fetchone()[0]
                na = conn.execute(sql_text("SELECT COUNT(*) FROM ai_answers")).fetchone()[0]
                nr = conn.execute(sql_text("SELECT COUNT(*) FROM inspection_report")).fetchone()[0]
            print(f"  ✓ DB OK — {nq} questions, {na} answers, {nr} inspection reports")
            return True
        except Exception as e:
            print(f"  ✗ Connection failed: {e}")
            return False

    @classmethod
    def from_env(cls, env_path=".env"):
        from dotenv import load_dotenv; load_dotenv(env_path)
        g = os.getenv
        return cls(
            db_connection_string=(
                f"postgresql://{g('DB_USER','postgres')}:{g('DB_PASSWORD','')}@"
                f"{g('DB_HOST','localhost')}:{g('DB_PORT','5432')}/{g('DB_NAME','postgres')}"
            ),
            db_schema=g("DB_SCHEMA", "public"),
            ollama_url=g("OLLAMA_URL", "http://localhost:11434"),
            sql_model=g("OLLAMA_SQL_MODEL", "sqlcoder:7b"),
            chat_model=g("OLLAMA_CHAT_MODEL", "qwen2.5:14b-instruct-q4_K_M"),
            embedding_model=g("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        )

    # ------------------------------------------------------------------ #
    # Finish helper
    # ------------------------------------------------------------------ #

    def _finish(self, question, answer, steps, iteration, emit_fn=None,
                stats=None, session=None, query_start=None):
        s = AgentStep(iteration, "[auto] synthesized", "final_answer",
                      {"answer": answer}, {"answer": answer}, 0)
        steps.append(s)
        if emit_fn: emit_fn(s)

        if stats and query_start:
            stats.total_wall_ms = int((time.time() - query_start) * 1000)
            for step in steps:
                if step.tool == "execute_sql":
                    stats.total_sql_exec_ms += step.duration_ms

        last_sql, last_total, last_cols = "", 0, []
        last_rows = []
        for step in reversed(steps):
            if step.tool == "execute_sql" and step.result.get("success"):
                data = step.result.get("result", {})
                last_sql = step.args.get("sql", "")
                last_total = data.get("total_count", data.get("row_count", 0))
                last_cols = data.get("columns", [])
                last_rows = data.get("rows", [])
                break

        # Also check get_answers tool results
        if not last_rows:
            for step in reversed(steps):
                if step.tool == "get_answers" and step.result.get("success"):
                    data = step.result.get("result", {})
                    if isinstance(data, dict):
                        last_rows = data.get("rows", [])
                    break

        # Extract key entities from result rows for follow-up context
        insp_ids, fac_names, insp_names, cycle_ids, user_ids = [], [], [], [], []
        single_row_vals = {}
        _UUID_PAT = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )

        for row in last_rows[:20]:
            if not isinstance(row, dict):
                continue
            for col in ("inspection_id",):
                v = row.get(col)
                if v and str(v) not in insp_ids:
                    insp_ids.append(str(v))
            for col in ("facility_name", "facility"):
                v = row.get(col)
                if v and str(v) not in fac_names:
                    fac_names.append(str(v))
            for col in ("inspector_name", "inspector"):
                v = row.get(col)
                if v and str(v) not in insp_names:
                    insp_names.append(str(v))
            for col in ("cycle_id", "inspection_cycle_id"):
                v = row.get(col)
                if v and str(v) not in cycle_ids and _UUID_PAT.match(str(v).strip()):
                    cycle_ids.append(str(v).strip())
            # Capture resolved user UUID from disambiguation queries (SELECT u.id AS user_id)
            for col in ("user_id",):
                v = row.get(col)
                if v and str(v) not in user_ids and _UUID_PAT.match(str(v).strip()):
                    user_ids.append(str(v).strip())

        # get_answers result rows don't have inspection_id as a column — pull it from the tool args
        if not insp_ids:
            for step in reversed(steps):
                if step.tool == "get_answers" and step.args:
                    _ga_insp = step.args.get("inspection_id")
                    if _ga_insp and str(_ga_insp) not in insp_ids:
                        insp_ids.append(str(_ga_insp))
                    break

        # If only 1 row, store all values for maximum context
        if len(last_rows) == 1 and last_rows[0]:
            single_row_vals = {
                k: str(v) for k, v in last_rows[0].items()
                if v is not None and str(v).strip()
            }
            # Also pull cycle_id from single-row result
            for col in ("cycle_id", "inspection_cycle_id"):
                v = last_rows[0].get(col)
                if v and str(v) not in cycle_ids and _UUID_PAT.match(str(v).strip()):
                    cycle_ids.append(str(v).strip())

        # ── Determine active subject for this turn ───────────────────────────
        # Priority: cycle > single inspector > single inspection > single facility
        active_subject_type = ""
        active_subject_value = ""
        active_subject_label = ""
        active_subject_table = ""
        active_subject_alias = ""

        if len(cycle_ids) == 1:
            active_subject_type = "cycle"
            active_subject_value = cycle_ids[0]
            active_subject_label = cycle_ids[0]
        elif len(insp_names) == 1:
            active_subject_type = "inspector"
            active_subject_value = insp_names[0]
            active_subject_label = insp_names[0]
        elif len(insp_ids) == 1:
            active_subject_type = "inspection"
            active_subject_value = insp_ids[0]
            active_subject_label = insp_ids[0]
        elif len(fac_names) == 1:
            active_subject_type = "facility"
            active_subject_value = fac_names[0]
            active_subject_label = fac_names[0]
        # If previous turn had a scoped subject (cycle or facility) and this result
        # has no conflicting subject (e.g. a COUNT query with no facility_name column),
        # inherit the scope so it persists across multi-turn drilldowns.
        elif session and session.turns:
            _prev_turn = session.turns[-1]
            # This turn ran no new SQL that names a single entity (e.g. it's
            # just "Adnoc Schools- Ghayathi" answering "which one did you
            # mean?"). If it's an exact pick from the PREVIOUS turn's entity
            # disambiguation list, resolve the scope directly instead of
            # falling through to "no active subject". Facility keeps the
            # dedicated "facility" type (tested wording); project/client/
            # organisation use the generic "entity" type, carrying their
            # table/alias so build_context_block can build a correct filter.
            if _prev_turn.entity_disambiguation_candidates:
                _ed_q = re.sub(r"[?.!,;:]+$", "", question.strip()).strip().lower()
                _ed_matches = [
                    c for c in _prev_turn.entity_disambiguation_candidates
                    if c.get("name", "").strip().lower() == _ed_q
                ]
                if len(_ed_matches) == 1:
                    if _prev_turn.entity_disambiguation_table == "facility":
                        active_subject_type = "facility"
                    else:
                        active_subject_type = "entity"
                        active_subject_table = _prev_turn.entity_disambiguation_table
                        active_subject_alias = _prev_turn.entity_disambiguation_alias
                    active_subject_value = _ed_matches[0]["name"]
                    active_subject_label = _ed_matches[0]["name"]
            if not active_subject_type:
                for prev_turn in reversed(session.turns):
                    if prev_turn.active_subject_type and prev_turn.active_subject_value:
                        active_subject_type = prev_turn.active_subject_type
                        active_subject_value = prev_turn.active_subject_value
                        active_subject_label = prev_turn.active_subject_label
                        active_subject_table = prev_turn.active_subject_table
                        active_subject_alias = prev_turn.active_subject_alias
                        break

        if session is not None:
            # MT-03 fix: if we have a facility name but no inspection_id (e.g. MT-01
            # returned only facility_name without ir.inspection_id in SELECT), do a quick
            # DB lookup so the follow-up turn can filter by inspection_id rather than
            # falling back to the worked-example placeholder "INS-EXAMPLE-001".
            if fac_names and not insp_ids and len(last_rows) <= 3:
                try:
                    fac_escaped = fac_names[0].replace("'", "''")
                    _lookup = (
                        "SELECT ir.inspection_id FROM inspection_report ir "
                        "JOIN facility fac ON ir.facility_id = fac.id "
                        f"WHERE fac.name ILIKE '%{fac_escaped}%' "
                        "AND ir.status != 'DRAFT' "
                        "ORDER BY ir.submitted_on DESC LIMIT 1;"
                    )
                    with self.db_engine.connect() as _conn:
                        _row = _conn.execute(sql_text(_lookup)).fetchone()
                    if _row and _row[0]:
                        insp_ids = [str(_row[0])]
                        _dbg(self._debug, "FINISH — inspection_id lookup",
                             f"facility={fac_names[0]} → {insp_ids[0]}", 0)
                except Exception:
                    pass  # Silently skip — follow-up will have less context

            # Detect if this turn was a disambiguation query (user_id + name cols)
            _disambig_name, _disambig_candidates = self._extract_disambiguation(
                last_sql, last_rows, last_cols
            )
            _is_disambig = bool(_disambig_name)

            # Detect if this turn was an entity disambiguation query (id + name
            # + inspection_count cols) so a later exact pick from this list can
            # resolve active_subject_type (see the elif branch above).
            _, _ent_disambig_candidates, _ent_disambig_table, _ent_disambig_alias = (
                self._extract_entity_disambiguation(last_sql, last_rows, last_cols)
            )

            # Carry forward the ORIGINAL question that started this disambiguation
            # chain. Without this, a multi-turn narrowing sequence ("list schedules
            # for neenu" → too many → "neenu qa inspector") loses track of the real
            # goal (schedules) once the model is a couple of turns deep into just
            # resolving the name, and answers "here's the person" instead of
            # actually running the original request.
            _disambig_origin_question = ""
            if _is_disambig:
                _prev_last = session.turns[-1] if session.turns else None
                if _prev_last and _prev_last.disambiguation_name:
                    _disambig_origin_question = (
                        _prev_last.disambiguation_origin_question or _prev_last.question
                    )
                else:
                    _disambig_origin_question = question

            session.add_turn(ConversationTurn(
                question=question,
                answer=answer[:300],
                tool_used=steps[-2].tool if len(steps) >= 2 else "unknown",
                sql=last_sql,
                total_count=last_total,
                columns=last_cols,
                inspection_ids=insp_ids[:5],
                facility_names=fac_names[:5],
                inspector_names=insp_names[:5],
                cycle_ids=cycle_ids[:3],
                user_ids=user_ids[:5],
                disambiguation_name=_disambig_name,
                disambiguation_candidates=_disambig_candidates,
                disambiguation_origin_question=_disambig_origin_question,
                entity_disambiguation_candidates=_ent_disambig_candidates,
                entity_disambiguation_table=_ent_disambig_table,
                entity_disambiguation_alias=_ent_disambig_alias,
                single_row_values=single_row_vals,
                active_subject_type=active_subject_type,
                active_subject_value=active_subject_value,
                active_subject_label=active_subject_label,
                active_subject_table=active_subject_table,
                active_subject_alias=active_subject_alias,
            ))

        _final_stats = stats or QueryStats()
        _log_query.debug("[QUERY] _finish sub=? iter=%d wall=%dms llm=%dms sql=%dms "
                         "llm_calls=%d prompt_tok=%d completion_tok=%d",
                         iteration,
                         _final_stats.total_wall_ms,
                         _final_stats.total_llm_latency_ms,
                         _final_stats.total_sql_exec_ms,
                         _final_stats.total_llm_calls,
                         _final_stats.total_prompt_tokens,
                         _final_stats.total_completion_tokens)
        return AgentResult(question, answer, steps, iteration, True, _final_stats)

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    _COUNT_RE = re.compile(
        r'\b(how\s+many|count|number\s+of|total)\b', re.IGNORECASE)
    _LIST_ALL_RE = re.compile(
        r'\b(show\s+all|list\s+all|all\s+forms|every\s+form)\b', re.IGNORECASE)

    def _synthesize(self, question, tool_name, tool_result, max_display_rows=20, sql=""):
        data = tool_result.get("result", {})
        q_lower = question.lower()

        # ---- list_forms ----
        if tool_name == "list_forms" and isinstance(data, list):
            if self._COUNT_RE.search(q_lower):
                return f"There are {len(data)} forms."
            if not data:
                return "No forms found."
            lines = [f"Found {len(data)} form(s):"]
            for i, f in enumerate(data, 1):
                name = f.get("form_name", f.get("module_name", "(unnamed)"))
                qcount = f.get("question_count", "?")
                lines.append(f"{i}. {name} ({qcount} questions)")
            return "\n".join(lines)

        # ---- semantic_search ----
        if tool_name == "semantic_search" and isinstance(data, list):
            if not data:
                return "No questions found matching that topic."
            if self._COUNT_RE.search(q_lower):
                forms = sorted({m["form_name"] for m in data if m.get("form_name")})
                return f"Found {len(data)} related question(s) across {len(forms)} form(s): {forms}"
            lines = [f"Found {len(data)} related question(s):"]
            for i, m in enumerate(data, 1):
                lines.append(f"  {i}. \"{m.get('question_label', m.get('label','?'))}\" (form: {m['form_name']}, score: {m['score']})")
            return "\n".join(lines)

        # ---- search_questions ----
        if tool_name == "search_questions" and isinstance(data, list):
            if not data:
                return "No matching questions found."
            if self._COUNT_RE.search(q_lower):
                return f"Found {len(data)} matching question(s)."
            lines = [f"Found {len(data)} question(s):"]
            for i, q in enumerate(data, 1):
                label = q.get("question_label", q.get("label", "?"))
                form = q.get("form_name", q.get("module_name", "?"))
                qtype = q.get("question_type", "")
                suffix = f" [{qtype}]" if qtype else ""
                lines.append(f"  {i}. {label}{suffix} — form: {form}")
            return "\n".join(lines)

        # ---- get_answers ----
        if tool_name == "get_answers" and isinstance(data, dict):
            rows = data.get("rows", [])
            truncated = data.get("truncated", False)
            total_count = data.get("total_count", len(rows))

            if not rows:
                return "No answers found matching the query."

            if truncated and total_count > 0:
                header = f"Found {total_count} answer(s) (showing first {len(rows)}):"
            elif truncated:
                header = f"Showing first {len(rows)} answer(s) (more may exist):"
            else:
                header = f"Found {len(rows)} answer(s):"

            lines = [header]
            for i, row in enumerate(rows[:max_display_rows], 1):
                q_label = row.get("question_label") or row.get("label") or "?"
                raw_ans = row.get("answer_value") or row.get("answer_text") or ""
                if not raw_ans and row.get("answer_numeric") is not None:
                    raw_ans = str(row["answer_numeric"])
                ans_val = raw_ans if raw_ans else "N/A"
                # Skip rows with no answer (file attachments, empty fields)
                if ans_val == "N/A" and row.get("score") is None:
                    continue
                insp = row.get("inspection_id", "")
                score = row.get("score")
                parts = [f"{q_label}: {ans_val}"]
                if score is not None:
                    parts.append(f"score: {score}")
                if insp:
                    parts.append(f"inspection: {insp}")
                lines.append(f"  {i}. {' | '.join(parts)}")

            if len(rows) > max_display_rows:
                lines.append(f"  ... and {len(rows) - max_display_rows} more in this batch")
            if truncated and total_count > len(rows):
                lines.append(f"  (Total matching: {total_count})")
            return "\n".join(lines)

        # ---- get_answer_stats ----
        if tool_name == "get_answer_stats" and isinstance(data, dict):
            avg = data.get("average_value")
            total = data.get("total_answers", 0)
            breakdown = data.get("value_breakdown", [])
            filters = data.get("filters", {})
            q_filter = filters.get("question_label") or ""
            parts = []

            if avg is not None:
                label = f"'{q_filter}'" if q_filter else "answers"
                source = data.get("stats_source", "")
                scored_count = data.get("scored_count", 0)
                source_note = ""
                if source == "score":
                    source_note = f" (from scoring system, {scored_count} scored answers)"
                elif source == "answer_numeric":
                    source_note = f" (from numeric answers)"
                parts.append(
                    f"Average score for {label}: {avg}{source_note}"
                    f" — min: {data.get('min_value')}, max: {data.get('max_value')}"
                )
            elif total > 0 and q_filter:
                parts.append(f"Total answers for '{q_filter}': {total} (no numeric/score data)")
            elif total > 0:
                parts.append(f"Total answers: {total}")

            if breakdown:
                bheader = f"Most common values for '{q_filter}':" if q_filter else "Most common values:"
                parts.append(bheader)
                for item in breakdown[:10]:
                    display_val = str(item["value"]) if item["value"] else "N/A"
                    parts.append(f"  • {display_val}: {item['count']} time(s)")

            return "\n".join(parts) if parts else "No answer statistics found."



        # ---- execute_sql ----
        if tool_name == "execute_sql" and isinstance(data, dict):
            rows = data.get("rows", [])
            cols = data.get("columns", [])
            truncated = data.get("truncated", False)
            total_count = data.get("total_count", len(rows))

            # Disambiguation result — user_id + name columns, searched by ILIKE
            # on first_name/last_name → format as choice list. Shared check
            # (core.disambiguation) so this agrees with fastapi_app.py's
            # dashboard-suppression logic for the same result.
            if is_disambiguation_result(cols, sql):
                if not rows:
                    return "No users found matching that name."
                name_col = next((c for c in cols if c.lower() == "name"), "name")
                count_col = next((c for c in cols if c.lower() in
                                  {"inspection_count", "count", "total"}), None)
                if total_count > 12:
                    return (
                        f"I found {total_count} people matching that name — too many to list. "
                        f"Could you provide a more specific full name or email address?"
                    )
                if total_count == 1:
                    row = rows[0]
                    name = row.get(name_col, "")
                    uid = row.get("user_id", "")
                    return f"Found exactly 1 match: **{name}** (user_id: `{uid}`). Proceeding with your request."
                # 2–12: list them as choices
                email_col = next((c for c in cols if c.lower() == "email"), None)
                lines = [f"I found {total_count} people matching that name:"]
                for r in rows[:12]:
                    name = r.get(name_col, "?")
                    parts = [f"  • {name}"]
                    if count_col and r.get(count_col) is not None:
                        parts.append(f"({r[count_col]} inspections)")
                    if email_col and r.get(email_col):
                        parts.append(f"— {r[email_col]}")
                    lines.append(" ".join(parts))
                lines.append("\nWhich one did you mean?")
                return "\n".join(lines)

            # Single-row aggregate
            if len(rows) == 1 and len(cols) <= 3:
                agg_names = {
                    "count", "total", "avg", "sum", "min", "max",
                    "average", "avg_score", "avg_inspection_score",
                    "total_capex", "total_opex", "question_count",
                    "answer_count", "report_count", "frequency",
                }
                col_lower = [c.lower() for c in cols]
                is_agg = all(
                    c in agg_names or c.startswith(("avg_", "sum_", "total_", "count_"))
                    or c.endswith(("_count", "_avg"))
                    for c in col_lower
                )
                if is_agg:
                    return "; ".join(f"{col}: {val}" for col, val in rows[0].items())

            # Two-column group-by
            if rows and len(cols) == 2:
                col_lower = [c.lower() for c in cols]
                has_count = any(
                    c in {"count", "total", "frequency", "improvement"}
                    or c.endswith("_count")
                    or c.startswith("num_")
                    for c in col_lower
                )
                if has_count:
                    count_idx = next(
                        i for i, c in enumerate(col_lower)
                        if c in {"count", "total", "frequency", "improvement"}
                        or c.endswith("_count")
                        or c.startswith("num_")
                    )
                    label_idx = 1 - count_idx
                    lines = []
                    for r in rows:
                        vals = list(r.values())
                        lines.append(f"  • {vals[label_idx]}: {vals[count_idx]}")
                    header = f"Results ({len(rows)} groups"
                    if truncated:
                        header += f", showing first {len(rows)} of {total_count}"
                    return header + "):\n" + "\n".join(lines)

            # General multi-row
            if rows:
                # Check for UUID leaks in name columns before displaying.
                # If detected, return an error that triggers the LLM to fix the JOIN.
                uuid_leaking = _has_uuid_leak(rows, cols)
                if uuid_leaking:
                    return (
                        f"QUERY ERROR: Raw UUID values in column(s) {uuid_leaking} "
                        f"instead of human-readable names. "
                        f"You must JOIN the lookup table. Example: "
                        f"facility_id → JOIN facility fac ON ir.facility_id = fac.id "
                        f"→ SELECT fac.name AS facility_name. "
                        f"Regenerate SQL with correct JOINs."
                    )

                if truncated and total_count > 0:
                    header = f"Found {total_count} result(s) (showing first {len(rows)}):"
                elif truncated:
                    header = f"Showing first {len(rows)} result(s) (more exist):"
                else:
                    header = f"Found {len(rows)} result(s):"

                # Detect constant-value columns (same NON-NULL value on every row) and
                # promote them to the header instead of repeating per row.
                # Exclude columns where the single value is None/"" — those are just
                # missing data (e.g. LAG() producing NULL previous_score) not true constants.
                constant_cols = {}
                _HIDE_COLS = {"element_id", "source_row_id", "source_table",
                              "module_id", "facility_id", "project_id",
                              "client_id", "inspector_user_id", "inspectee_user_id",
                              "inspection_report_id", "question_id",
                              "inspection_type_id", "inspection_sub_type_id",
                              "cycle_id", "schedule_id", "entity_id",
                              "user_id"}
                displayable_cols = [c for c in rows[0].keys() if c not in _HIDE_COLS]
                if len(rows) > 1:
                    for col in displayable_cols:
                        vals = {str(r.get(col)) for r in rows}
                        val_str = str(rows[0].get(col) or "")
                        # Only promote if: same across all rows AND not empty/None AND
                        # not a meaningful data column that happens to have one value
                        if (len(vals) == 1
                                and col not in ("inspection_id", "submitted_on")
                                and val_str not in ("", "None", "N/A")):
                            constant_cols[col] = val_str
                if constant_cols:
                    const_str = ", ".join(f"{k}={v}" for k, v in constant_cols.items())
                    header += f" ({const_str})"

                lines = [header]
                for i, row in enumerate(rows[:max_display_rows], 1):
                    parts = []
                    for col, val in row.items():
                        if col in _HIDE_COLS:
                            continue
                        # Skip columns promoted to header (constant across all rows)
                        if col in constant_cols:
                            continue
                        if val is None:
                            if len(displayable_cols) <= 2:
                                parts.append(f"{col}: N/A")
                            continue
                        val_str = str(val)
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                        parts.append(f"{col}: {val_str}")
                    lines.append(f"  {i}. {' | '.join(parts) or '(no data)'}")

                if len(rows) > max_display_rows:
                    lines.append(f"  ... and {len(rows) - max_display_rows} more shown")
                if truncated and total_count > len(rows):
                    lines.append(f"  (Total matching: {total_count})")
                return "\n".join(lines)

            return "The query returned no results."

        return self._llm_answer(question, tool_name, data)

    def _llm_answer(self, question, tool_name, data, extra=""):
        prompt = (
            f"Answer the question using ONLY the data provided. "
            f"Do not estimate or invent. If data is empty, say so.\n"
            f"{extra}\n\nQuestion: {question}\n\n"
            f"Data ({tool_name}):\n{json.dumps(data, default=str, indent=2)}\n\n"
            f"Give a concise, accurate answer. No JSON."
        )
        try:
            return str(self.reasoning_llm.complete(prompt)).strip()
        except Exception:
            return json.dumps(data, default=str)

    # ------------------------------------------------------------------ #
    # Context messages for non-terminal tool results
    # ------------------------------------------------------------------ #

    def _context_msg(self, tool_name, tool_result, question=""):
        if not tool_result.get("success"):
            error = tool_result.get("error", "unknown")

            if tool_name == "generate_sql":
                q_lower = question.lower()
                hint = ""
                if re.search(r'\b(corrective.?action|cause|correction|capex|opex|overdue)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK SQL:\n"
                        "SELECT corrective_action_id, cause, corrective_action, responsible, status\n"
                        "FROM inspection_corrective_action WHERE status = 'OPEN' LIMIT 100;"
                    )
                elif re.search(r'\b(inspection.?score|hour|inspector)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK SQL:\n"
                        "SELECT inspection_score, status, submitted_on\n"
                        "FROM inspection_report WHERE status != 'DRAFT' LIMIT 50;"
                    )
                elif re.search(r'\b(answer|response|submission)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK: Use get_answers tool or this SQL:\n"
                        "SELECT aq.label, aa.answer_text, aa.answer_numeric, aa.inspection_id\n"
                        "FROM ai_answers aa\n"
                        "LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
                        "WHERE aq.label ILIKE '%KEYWORD%' LIMIT 50;"
                    )
                return f"generate_sql failed: {error}{hint}"

            if tool_name == "get_answers":
                return (
                    f"get_answers failed: {error}\n"
                    f"FALLBACK: Use execute_sql with:\n"
                    f"SELECT aq.label, aa.answer_text, aa.answer_numeric, aa.inspection_id\n"
                    f"FROM ai_answers aa LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
                    f"WHERE aa.module_name ILIKE '%MODULE%' LIMIT 50;"
                )

            return f"Tool '{tool_name}' failed: {error}. Try a different approach."

        # Successful non-terminal results
        if tool_name == "semantic_search":
            data = tool_result.get("result", [])
            if data:
                fmt = "\n".join(
                    f"  - \"{m.get('question_label', m.get('label','?'))}\" (form: {m['form_name']}, score: {m['score']})"
                    for m in data[:5]
                )
                asks_answers = bool(re.search(
                    r'\b(answer|response|submission|filled|submitted|responded)\b',
                    question.lower()))
                if asks_answers:
                    top_form = data[0].get("form_name", data[0].get("module_name",""))
                    top_q = data[0].get("question_label", data[0].get("label",""))
                    return (
                        f"semantic_search found {len(data)} match(es):\n{fmt}\n\n"
                        f"User asked about answers. Call get_answers with "
                        f"module_name='{top_form}' and question_label='{top_q}'."
                    )
                return (
                    f"semantic_search found {len(data)} match(es):\n{fmt}\n\n"
                    f"If this answers the question, call final_answer.\n"
                    f"To get submitted answers for these questions, call get_answers."
                )
            return "semantic_search found no matches. Call final_answer saying no forms ask about that topic."

        if tool_name == "search_questions":
            data = tool_result.get("result", [])
            if data:
                fmt = "\n".join(
                    f"  - \"{q.get('question_label', q.get('label','?'))}\" (module: {q.get('module_name','?')})"
                    for q in data[:6]
                )
                return (
                    f"search_questions found {len(data)} question(s):\n{fmt}\n\n"
                    f"To see submitted answers, call get_answers with question_label (aq.label) "
                    f"set to one of the labels above."
                )
            return "search_questions found no matches. Try a broader keyword or use semantic_search."

        return json.dumps(tool_result, default=str)

    def _schema_hint_for_error(self, error_msg):
        parts = []
        tm = re.search(r'relation "(\w+)" does not exist', error_msg)
        if tm:
            bad = tm.group(1)
            try:
                sr = self.registry.call("get_schema", {})
                tables = sr.get("result", {}).get("tables", [])
                ai_tables = [t for t in tables if t.startswith("ai_")]
                insp_tables = [t for t in tables if t.startswith("inspection_")]
                parts.append(
                    f"Table '{bad}' doesn't exist.\n"
                    f"AI tables: {ai_tables}\n"
                    f"Inspection tables: {insp_tables}"
                )
            except Exception:
                pass
        cm = re.search(r'column "(\w+)" does not exist', error_msg)
        if cm and not tm:
            parts.append(f"Column '{cm.group(1)}' doesn't exist. Use get_schema to check columns.")
        return "\n".join(parts) + "\n" if parts else ""

    def _call_llm(self, messages, stats=None, iteration=0):
        t0 = time.time()
        response = self.reasoning_llm.chat(messages)
        latency_ms = int((time.time() - t0) * 1000)
        content = response.message.content.strip()
        _prompt_tok = 0
        _completion_tok = 0
        if stats:
            stats.total_llm_calls += 1
            stats.total_llm_latency_ms += latency_ms
            raw = getattr(response, "raw", {}) or {}
            _prompt_tok = raw.get("prompt_eval_count", 0)
            _completion_tok = raw.get("eval_count", 0)
            stats.total_prompt_tokens += _prompt_tok
            stats.total_completion_tokens += _completion_tok

        _log_llm.debug("[LLM ] iter=%d latency=%dms prompt_tok=%d completion_tok=%d out_len=%d",
                       iteration, latency_ms, _prompt_tok, _completion_tok, len(content))

        # Debug logging (raw LLM output to separate file when DEBUG_LLM=1)
        if self._debug:
            trunc = _detect_truncation(content)
            trunc_note = f"\n⚠ {trunc}" if trunc else ""
            _dbg(self._debug,
                 f"LLAMA RAW OUTPUT [{latency_ms}ms]{trunc_note}",
                 content, iteration)

        return content

    @staticmethod
    def _extract_disambiguation(sql: str, rows: list, columns: list[str]) -> tuple[str, list[dict]]:
        """
        If `columns` matches the people-disambiguation shape (user_id + name),
        extract the searched name from `sql` and the (name, user_id) candidate
        pairs from `rows`. Returns ("", []) otherwise.

        Shared by every ConversationTurn construction site so the STOP-guard /
        exact-match / origin-question context (see build_context_block) works
        regardless of which code path produced the disambiguation result —
        previously only the main _finish() path populated these fields, so a
        disambiguation result reaching the user via the final_answer branch or
        a pagination replay silently dropped this context on the next turn.
        """
        if not is_disambiguation_result(columns, sql):
            return "", []

        # Anchor to the first_name/last_name reference, then take the NEXT ILIKE
        # after it — a plain "first ILIKE anywhere in the SQL" search can grab an
        # unrelated earlier filter (e.g. a facility-name ILIKE) when the query
        # combines a name check with other conditions.
        name_col_pos = re.search(r"first_name|last_name", sql, re.IGNORECASE)
        search_from = sql[name_col_pos.start():] if name_col_pos else sql
        m = re.search(r"ILIKE\s+'%([^%]+)%'", search_from, re.IGNORECASE)
        name = m.group(1).strip() if m else "?"

        candidates = []
        for row in rows[:20]:
            if not isinstance(row, dict):
                continue
            nm, uid = row.get("name"), row.get("user_id")
            if nm and uid and _UUID_RE.match(str(uid).strip()):
                candidates.append({"name": str(nm).strip(), "user_id": str(uid).strip()})
        return name, candidates

    @staticmethod
    def _extract_entity_disambiguation(sql: str, rows: list, columns: list[str]) -> tuple[str, list[dict], str, str]:
        """
        If `columns`/`sql` match the STEP 1 entity count-check shape (id +
        name + inspection_count, ILIKE on a known entity table — see
        is_entity_disambiguation_result) produced by the AMBIGUOUS REFERENCE
        RULE for "at <facility/project/client/organisation>"-style queries,
        extract the searched name, the (name, inspection_count) candidate
        rows, and the table/alias that was searched.

        Without this, an entity disambiguation list ("4 facilities match
        'adnoc'") is never remembered: the generic facility_names extraction
        in _finish() only looks for a "facility_name"/"facility" column, which
        this template doesn't produce (it selects bare "name"). So when the
        user later names one entity from that list, that turn runs no new
        SQL of its own and active_subject_type is never set — the scope
        silently vanishes on the very next follow-up question.
        """
        entity = is_entity_disambiguation_result(columns, sql)
        if entity is None:
            return "", [], "", ""
        table, alias = entity

        m = re.search(rf"{re.escape(alias)}\.name\s+ilike\s+'%([^%]+)%'", sql, re.IGNORECASE)
        name = m.group(1).strip() if m else "?"

        candidates = []
        for row in rows[:20]:
            if not isinstance(row, dict):
                continue
            nm = row.get("name")
            if nm:
                candidates.append({
                    "name": str(nm).strip(),
                    "inspection_count": row.get("inspection_count") or 0,
                })
        return name, candidates, table, alias

    def _disambiguation_followup_message(self, question: str, sql: str, exec_data: dict) -> str:
        """
        Build a STEP-2a/2b-specific follow-up message for a disambiguation STEP 1
        result, instead of the generic "If this answers it, call final_answer"
        message — which previously caused Llama to statistically summarize the
        disambiguation table (MAX/SUM of inspection_count) instead of recognizing
        2+ rows as the STEP 2b "which one did you mean?" trigger.

        Branches on `is_disambiguation_result` (people: user_id+name), not on
        "inspection_count" presence — both real people templates can include
        inspection_count too, so that heuristic would misroute a people result
        into the entity/facility wording below. Callers gate on
        `is_disambiguation_result(...) or is_entity_disambiguation_result(...)`
        before calling this, so exactly one of the two branches applies.
        """
        rows = exec_data.get("rows", [])
        columns = exec_data.get("columns", [])
        m = re.search(r"ilike '%(.+?)%'", sql, re.IGNORECASE)
        name_filter = m.group(1) if m else "?"

        if not is_disambiguation_result(columns, sql):
            # Entity template (facility/project/client/organisation) — only
            # candidates with inspections > 0 are useful answers to "show
            # inspections at X"; zero-inspection entities are real matches
            # but not relevant choices.
            _entity = is_entity_disambiguation_result(columns, sql)
            table, alias = _entity if _entity else ("facility", "fac")
            table_plural = table[:-1] + "ies" if table.endswith("y") else table + "s"
            nonzero = [r for r in rows if (r.get("inspection_count") or 0) > 0]
            if len(nonzero) == 0:
                return (
                    f"STEP 1 disambiguation check found {len(rows)} {table_plural} matching "
                    f"'{name_filter}', but NONE have any inspections.\n"
                    f"Original question: \"{question}\"\n"
                    f"Call final_answer stating no inspections were found at any {table} "
                    f"matching '{name_filter}'."
                )
            elif len(nonzero) == 1:
                r = nonzero[0]
                return (
                    f"STEP 1 disambiguation check: exactly ONE {table} matching "
                    f"'{name_filter}' has inspections — '{r.get('name')}' "
                    f"({r.get('inspection_count')} inspections).\n"
                    f"Original question: \"{question}\"\n"
                    f"This is STEP 2a — call generate_sql again using "
                    f"{alias}.name ILIKE '%{r.get('name')}%' to filter by this {table}. "
                    f"Do NOT use {alias}.id UUID — the SQL model ignores UUID filters "
                    f"and will generate ILIKE anyway. Use the exact name directly."
                )
            else:
                options = "\n".join(
                    f"  • {r.get('name')} ({r.get('inspection_count')} inspections)"
                    for r in nonzero
                )
                return (
                    f"STEP 1 disambiguation check: {len(nonzero)} {table_plural} matching "
                    f"'{name_filter}' have inspections (out of {len(rows)} total matches).\n"
                    f"Original question: \"{question}\"\n"
                    f"This is STEP 2b — call final_answer NOW with EXACTLY this format "
                    f"(do not compute totals, leaders, sums, or any other statistic; do "
                    f"not call generate_sql again):\n"
                    f"\"I found {len(nonzero)} {table_plural} matching '{name_filter}':\n"
                    f"{options}\n"
                    f"Which one did you mean?\""
                )
        else:
            # People template — list all matched names
            if len(rows) == 0:
                return (
                    f"STEP 1 disambiguation check found NO users matching "
                    f"'{name_filter}'.\n"
                    f"Original question: \"{question}\"\n"
                    f"Call final_answer stating no matching person was found."
                )
            elif len(rows) == 1:
                r = rows[0]
                _name_parts = str(r.get('name', '')).split()
                _first = _name_parts[0] if _name_parts else name_filter
                _last = _name_parts[-1] if len(_name_parts) > 1 else ''
                return (
                    f"STEP 1 disambiguation check: exactly ONE person matches "
                    f"'{name_filter}' — '{r.get('name')}'.\n"
                    f"Original question: \"{question}\"\n"
                    f"This is STEP 2a — call generate_sql again filtered by name: "
                    f"WHERE u.first_name ILIKE '%{_first}%' AND u.last_name ILIKE '%{_last}%' "
                    f"(joined via inspector_user_id or inspectee_user_id per the question's "
                    f"role wording). Do NOT use u.id UUID — the SQL model ignores UUID "
                    f"filters and will generate ILIKE anyway."
                )
            else:
                options = "\n".join(f"  • {r.get('name')}" for r in rows)
                return (
                    f"STEP 1 disambiguation check: {len(rows)} people match "
                    f"'{name_filter}'.\n"
                    f"Original question: \"{question}\"\n"
                    f"This is STEP 2b — call final_answer NOW with EXACTLY this format "
                    f"(do not pick one, do not compute statistics, do not call "
                    f"generate_sql again):\n"
                    f"\"I found {len(rows)} people matching '{name_filter}':\n"
                    f"{options}\n"
                    f"Which one did you mean?\""
                )

    def _is_preparatory_result(self, question: str, exec_result: dict, sql: str = "") -> bool:
        q_lower = question.lower()
        data = exec_result.get("result", {})
        columns = [c.lower() for c in data.get("columns", [])]
        row_count = data.get("row_count", 0)

        # AMBIGUOUS REFERENCE RULE STEP 1 check (people OR entity shape).
        # Without this, STEP 1 results render directly as a table and Llama
        # never gets to branch into STEP 2a/2b — the disambiguation flow
        # silently becomes a dead-end table dump.
        if is_disambiguation_result(columns, sql) or is_entity_disambiguation_result(columns, sql):
            return True

        wants_content = bool(re.search(
            r'\b(question|label|answer|response|what.*in)\b', q_lower))
        has_content = any(c in columns for c in [
            # Actual DB column names
            "label", "answer_text", "answer_numeric",
            "inspection_score", "total_inspection_hours",
            "cause", "correction", "corrective_action", "responsible",
            "status", "progress_stage", "capex", "opex",
            "target_close_out_date", "schedule_date", "due_date",
            # Common SQL aliases the LLM uses
            "question", "answer", "question_label", "answer_value",
            "observation", "finding", "score", "risk_level",
            "inspector_name", "facility_name", "inspection_type_name",
        ])

        if wants_content and not has_content and row_count > 0:
            return True

        is_superlative = bool(re.search(
            r'\b(most\s+recent|recently|latest|newest|oldest|first|last)\b', q_lower))
        wants_details = bool(re.search(
            r'\b(question|what|show|list|answer|submission|response|score)\b', q_lower))
        if is_superlative and wants_details and not has_content:
            return True

        return False