"""
Agent Orchestrator — the main reasoning loop.

Changes vs original:
  Problem 4 — SeedExampleIndex initialised from SEED_EXAMPLES and passed to
              ToolRegistry so generate_sql() receives it for dynamic retrieval.

  Problem 5 — _DEFERRED_CHAIN_RE added; when matched, the correct
              CLOSE_WITH_DEFERRED SQL pattern is injected as schema_hint so
              deepseek generates the right temporal chain query.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, text as sql_text

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from core.business_rules import get_engine as _get_rules_engine
from core.semantic import SemanticQuestionIndex, SeedExampleIndex
from core.validator import SQLValidator
from core.schema_introspector import introspect_schema
from agent.prompts import (
    SYSTEM_PROMPT, SQL_GENERATION_PROMPT, FORCE_SUMMARY_PROMPT, SEED_EXAMPLES
)
try:
    from agent.prompts import SQL_GENERATION_PROMPT_SQLCODER
except ImportError:
    SQL_GENERATION_PROMPT_SQLCODER = SQL_GENERATION_PROMPT
from agent.tools import ToolRegistry
from agent.tool_dispatcher import parse_tool_call, dispatch


# ---------------------------------------------------------------------------
# Debug logger — enable with DEBUG_LLM=1 in environment or .env
# Writes to debug_llm_YYYYMMDD_HHMMSS.log in the working directory
# ---------------------------------------------------------------------------

def _make_debug_logger() -> logging.Logger | None:
    """Return a file logger if DEBUG_LLM=1, else None."""
    if not os.getenv("DEBUG_LLM", "").strip() in ("1", "true", "yes"):
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"debug_llm_{ts}.log"
    logger = logging.getLogger(f"debug_llm_{ts}")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    print(f"  [DEBUG_LLM] Logging to {log_path}")
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
    if not stripped.endswith(("}", "}")):
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
    single_row_values: dict = field(default_factory=dict) # col→val when result is 1 row




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
            r"continue|next|previous|last\s+one)\b",
            re.IGNORECASE,
        )
        _FRESH_SUBJECT_RE = re.compile(
            r"\b(what|which|show|list|give|find|how\s+many|average|total|"
            r"all\s+\w+|most\s+recent(?!ly)|most\s+common|latest)\b",
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
            if t.facility_names:
                lines.append(f"  ★ facility(s) from this result: {t.facility_names[:5]}")
            if t.inspector_names:
                lines.append(f"  ★ inspector(s) from this result: {t.inspector_names[:5]}")
            if t.single_row_values:
                lines.append(f"  ★ single-row result: {t.single_row_values}")
        lines.append("")

        if not is_clearly_fresh:
            last = self.turns[-1]
            # Look back through all recent turns — not just the last one.
            # MT-01 may have stored the inspection_id, but MT-02 only returned inspector_name,
            # which clears the per-turn inspection_id. Without this look-back, MT-03 loses context.
            _insp_turn = next(
                (t for t in reversed(self.turns) if t.inspection_ids),
                None
            )
            _fac_turn = next(
                (t for t in reversed(self.turns) if t.facility_names),
                None
            )

            if _insp_turn and len(_insp_turn.inspection_ids) == 1:
                lines.append(
                    f"CONTEXT — The conversation is about inspection '{_insp_turn.inspection_ids[0]}'."
                    f" When the user says 'that inspection', 'it', 'that site', 'she filled',"
                    f" 'he filled', 'they filled', 'that form' — use"
                    f" WHERE aa.inspection_id = '{_insp_turn.inspection_ids[0]}'"
                    f" (or ir.inspection_id = '{_insp_turn.inspection_ids[0]}') as an exact filter."
                    f" Do NOT filter by facility name or inspector name — use the inspection_id."
                )
            elif _insp_turn and _insp_turn.inspection_ids:
                ids_str = ", ".join(f"'{i}'" for i in _insp_turn.inspection_ids[:5])
                lines.append(
                    f"CONTEXT — The conversation involves inspections: {ids_str}."
                    f" When the user references 'those inspections' or 'that data',"
                    f" filter by inspection_id IN ({ids_str})."
                )
            elif _fac_turn and len(_fac_turn.facility_names) == 1:
                lines.append(
                    f"CONTEXT — The conversation is about facility '{_fac_turn.facility_names[0]}'."
                    f" If the user asks about 'that facility' or 'that site',"
                    f" filter by fac.name ILIKE '%{_fac_turn.facility_names[0]}%'."
                )
            elif last.single_row_values:
                lines.append(f"CONTEXT — Last query returned a single result: {last.single_row_values}")

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

# Problem 5: deferred / recurring action chain queries
_DEFERRED_CHAIN_RE = re.compile(
    r'\b(deferred|recurring|recur|carried.?forward|repeated.?finding|'
    r'same.?issue|chronic|persistent|repeat.?observation|'
    r'close.?with.?deferred|multiple.?cycles?)\b',
    re.IGNORECASE,
)

# Inspector count query — deepseek reliably generates hardcoded years for this pattern.
_INSPECTOR_COUNT_RE = re.compile(
    r'\b(which|who|top|most|count|how\s+many|list).{0,30}'
    r'(inspector|inspectors?).{0,30}'
    r'(most|conducted|done|most\s+inspections?|count|number|year)\b',
    re.IGNORECASE,
)

# Percentage of inspections with corrective action — DeepSeek always fails this.
# Inject exact working SQL as schema_hint.
_PCT_WITH_CA_RE = re.compile(
    r'\b(percentage|percent|pct|proportion|ratio).{0,40}'
    r'(corrective.?action|mitigative|finding)\b'
    r'|\b(corrective.?action|mitigative|finding).{0,40}'
    r'(percentage|percent|pct|proportion|ratio)\b',
    re.IGNORECASE,
)

# Corrective action + facility/grouping queries
_ICA_JOIN_RE = re.compile(
    r'\b(corrective.?action|capex|opex).{0,50}'
    r'(facility|client|inspector|type|count|by|group|breakdown)\b'
    r'|\b(facility|client|inspector|type|breakdown).{0,50}'
    r'(corrective.?action|capex|opex)\b',
    re.IGNORECASE,
)

# Most-recent inspection query
_MOST_RECENT_INSPECTION_RE = re.compile(
    r'\b(most\s+recent|latest|last)\s+(inspection|site|facility|form)\b'
    r'|\b(which|what)\s+(facility|site).{0,30}(most\s+recent|latest|last)\b',
    re.IGNORECASE,
)


def _detect_inspection_tables(question: str) -> list:
    tables = []
    q = question.lower()
    if _INSPECTION_REPORT_RE.search(q):
        tables.extend(["inspection_report", "inspection_type", "inspection_sub_type"])
    if _CORRECTIVE_ACTION_RE.search(q):
        tables.append("inspection_corrective_action")
    if _SCHEDULE_CYCLE_RE.search(q):
        tables.extend(["inspection_schedule", "inspection_cycle", "inspector_portfolio"])
    return tables


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
        # Pick SQL prompt based on model — SQLCoder uses native DDL schema format,
        # generic deepseek uses the instruction-heavy prose format.
        _is_sqlcoder = "sqlcoder" in sql_model.lower()
        base_sql_prompt = SQL_GENERATION_PROMPT_SQLCODER if _is_sqlcoder else SQL_GENERATION_PROMPT
        self._sql_prompt = base_sql_prompt.replace("{inspection_schema_sql}", "")

        print(f"  Loading embedding model: {embedding_model}...")
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = embed_model
        print("  ✓ Embedding model loaded")

        self.semantic_index = SemanticQuestionIndex(
            db_engine=self.db_engine, embed_model=embed_model)
        self.validator = SQLValidator()

        # Problem 4: initialise seed example index from SEED_EXAMPLES in prompts.py
        print(f"  Initialising seed example index ({len(SEED_EXAMPLES)} examples)...")
        self.seed_index = SeedExampleIndex(embed_model=embed_model, examples=SEED_EXAMPLES)

        print(f"  Configuring reasoning LLM: {chat_model}...")
        self.reasoning_llm = Ollama(model=chat_model, base_url=ollama_url,
            temperature=0.2, request_timeout=120.0,
            additional_kwargs={"num_predict": 768, "num_ctx": 16384})
        print(f"  ✓ Reasoning LLM ({chat_model})")

        print(f"  Configuring SQL LLM: {sql_model}...")
        self.sql_llm = Ollama(model=sql_model, base_url=ollama_url,
            temperature=0.0, request_timeout=120.0,
            additional_kwargs={
                "num_predict": 350,
                "num_ctx": 16384,
                "top_p": 0.9,
                "stop": [";", "[/SQL]", "This query", "Note:", "-- Note", "/*"],
            })
        print(f"  ✓ SQL LLM ({sql_model})")

        self._debug = _make_debug_logger()

        self.registry = ToolRegistry(
            db_engine=self.db_engine,
            semantic_index=self.semantic_index,
            validator=self.validator,
            sql_llm=self.sql_llm,
            sql_prompt=self._sql_prompt,
            seed_index=self.seed_index,   # Problem 4: pass seed index
            debug_logger=self._debug,
        )

    # ------------------------------------------------------------------ #
    # Terminal tools — synthesize and return immediately on success
    # ------------------------------------------------------------------ #
    _TERMINAL_TOOLS = {
        "list_forms",
        "get_answers",
        "get_answer_stats",
        "execute_sql",
    }

    def query(self, question, max_iterations=8, on_step=None, session=None):
        query_start = time.time()
        stats = QueryStats()

        def _emit(event):
            if on_step: on_step(event)

        system_content = self._system_prompt
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

        # ---- Pagination fast-path ----
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
                        ans = self._synthesize(question, "execute_sql", er, max_display_rows=100)
                        stats.total_wall_ms = int((time.time() - query_start) * 1000)
                        exec_data = er.get("result", {})
                        if session is not None:
                            session.add_turn(ConversationTurn(
                                question=question, answer=ans[:300],
                                tool_used="execute_sql", sql=prev.sql,
                                total_count=exec_data.get("total_count", exec_data.get("row_count", 0)),
                                columns=exec_data.get("columns", []),
                            ))
                        return AgentResult(question, ans, steps, 1, True, stats)

                # Path B: last turn used get_answers — re-call it with same inspection_id
                elif prev.tool_used == "get_answers" and prev.inspection_ids:
                    insp_id = prev.inspection_ids[0]
                    ga_args = {"inspection_id": insp_id, "limit": 500}
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
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=
                    f"Not valid JSON: {exc}\nRespond with ONLY a JSON tool call."))
                s = AgentStep(iteration, "[parse error]", "[parse_error]",
                              {}, {"error": str(exc)}, int((time.time()-t0)*1000))
                steps.append(s); _emit(s); continue

            _dbg(self._debug, "PARSED TOOL CALL",
                 json.dumps(tc, default=str, indent=2), iteration)

            thought, tool, args = tc.get("thought", ""), tc["tool"], tc["args"]

            # Auto-inject targeted hints for generate_sql calls.
            # All domain-specific patterns now live in business_rules.REGISTRY (INJECT rules).
            # The engine matches the question and returns combined hints in priority order.
            if tool == "generate_sql":
                registry_hint = _get_rules_engine().hint_for_question(question)
                if registry_hint:
                    args["schema_hint"] = registry_hint + args.get("schema_hint", "")

                # Preserve explicit LIMIT from user question — kept here because it
                # requires parsing a number from the question, not just pattern matching.
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
                tc["args"] = args

            if tool == "final_answer":
                ans = args.get("answer", "")
                s = AgentStep(iteration, thought, "final_answer", args,
                              {"answer": ans}, int((time.time()-t0)*1000))
                steps.append(s); _emit(s)
                stats.total_wall_ms = int((time.time() - query_start) * 1000)
                if session is not None:
                    session.add_turn(ConversationTurn(
                        question=question, answer=ans[:300],
                        tool_used="final_answer", sql=_last_sql,
                        total_count=_last_total_count, columns=_last_columns,
                    ))
                return AgentResult(question, ans, steps, iteration, True, stats)

            result = dispatch(tc, self.registry)
            s = AgentStep(iteration, thought, tool, args, result,
                          int((time.time()-t0)*1000))
            steps.append(s); _emit(s)

            # Path A: generate_sql → auto-execute
            if tool == "generate_sql" and result.get("success"):
                sql_data = result.get("result", {})
                if sql_data.get("validation", {}).get("passed"):
                    sql = sql_data.get("sql", "")
                    et0 = time.time()
                    er = self.registry.call("execute_sql", {"sql": sql})
                    auto = AgentStep(iteration, "[auto] execute after generate_sql",
                                     "execute_sql", {"sql": sql}, er,
                                     int((time.time()-et0)*1000))
                    steps.append(auto); _emit(auto)

                    if not er.get("success"):
                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                        hint = self._schema_hint_for_error(er.get("error", ""))
                        messages.append(ChatMessage(role=MessageRole.USER, content=
                            f"SQL failed: {er.get('error','')}\nSQL: {sql}\n{hint}"
                            f"Fix the query or use get_schema to check column names."))
                        continue

                    exec_data = er.get("result", {})
                    _last_sql = sql
                    _last_total_count = exec_data.get("total_count", exec_data.get("row_count", 0))
                    _last_columns = exec_data.get("columns", [])

                    if self._is_preparatory_result(question, er):
                        rows_preview = json.dumps(exec_data.get("rows", [])[:10], default=str)
                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                        messages.append(ChatMessage(role=MessageRole.USER, content=
                            f"SQL: {sql}\nResult: {exec_data.get('row_count',0)} rows, "
                            f"columns={exec_data.get('columns',[])}\n"
                            f"Sample: {rows_preview}\n\n"
                            f"Original question: \"{question}\"\n"
                            f"If this answers it, call final_answer. Otherwise call the next tool."))
                        continue

                    ans = self._synthesize(question, "execute_sql", er)
                    return self._finish(question, ans, steps, iteration, _emit, stats, session, query_start)

                else:
                    gen_sql_failures += 1
                    errs = sql_data.get("validation", {}).get("errors", ["unknown"])
                    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))

                    if gen_sql_failures >= 3:
                        sql = sql_data.get("sql", "")
                        if sql:
                            er = self.registry.call("execute_sql", {"sql": sql})
                            auto = AgentStep(iteration, "[auto] forced exec after 3 failures",
                                             "execute_sql", {"sql": sql}, er, 0)
                            steps.append(auto); _emit(auto)
                            if er.get("success"):
                                ans = self._synthesize(question, "execute_sql", er)
                                return self._finish(question, ans, steps, iteration, _emit, stats, session, query_start)

                    messages.append(ChatMessage(role=MessageRole.USER, content=
                        f"generate_sql failed ({gen_sql_failures}x). Errors: {'; '.join(errs)}\n"
                        f"Try a different approach or use get_schema to verify columns."))
                    continue

            # Path B: terminal tools → synthesize and return
            if tool in self._TERMINAL_TOOLS and result.get("success"):
                ans = self._synthesize(question, tool, result)
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
            session.add_turn(ConversationTurn(
                question=question, answer=ans[:300], tool_used="forced_summary",
                sql=_last_sql, total_count=_last_total_count, columns=_last_columns,
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

        if not last_rows:
            for step in reversed(steps):
                if step.tool == "get_answers" and step.result.get("success"):
                    data = step.result.get("result", {})
                    if isinstance(data, dict):
                        last_rows = data.get("rows", [])
                    break

        insp_ids, fac_names, insp_names = [], [], []
        single_row_vals = {}

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

        if len(last_rows) == 1 and last_rows[0]:
            single_row_vals = {
                k: str(v) for k, v in last_rows[0].items()
                if v is not None and str(v).strip()
            }

        if session is not None:
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
                    pass

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
                single_row_values=single_row_vals,
            ))

        return AgentResult(question, answer, steps, iteration, True, stats or QueryStats())

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    _COUNT_RE = re.compile(
        r'\b(how\s+many|count|number\s+of|total)\b', re.IGNORECASE)
    _LIST_ALL_RE = re.compile(
        r'\b(show\s+all|list\s+all|all\s+forms|every\s+form)\b', re.IGNORECASE)

    def _synthesize(self, question, tool_name, tool_result, max_display_rows=20):
        data = tool_result.get("result", {})
        q_lower = question.lower()

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

        if tool_name == "execute_sql" and isinstance(data, dict):
            rows = data.get("rows", [])
            cols = data.get("columns", [])
            truncated = data.get("truncated", False)
            total_count = data.get("total_count", len(rows))

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

            if rows:
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

                constant_cols = {}
                _HIDE_COLS = {"element_id", "source_row_id", "source_table",
                              "module_id", "facility_id", "project_id",
                              "client_id", "inspector_user_id", "inspectee_user_id",
                              "inspection_report_id", "question_id",
                              "inspection_type_id", "inspection_sub_type_id",
                              "cycle_id", "schedule_id", "entity_id"}
                displayable_cols = [c for c in rows[0].keys() if c not in _HIDE_COLS]
                if len(rows) > 1:
                    for col in displayable_cols:
                        vals = {str(r.get(col)) for r in rows}
                        val_str = str(rows[0].get(col) or "")
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
                elif re.search(r'\b(inspection.?score|gp.?score|hour|inspector)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK SQL:\n"
                        "SELECT inspection_score, gp_score, status, submitted_on\n"
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
        if stats:
            stats.total_llm_calls += 1
            stats.total_llm_latency_ms += latency_ms
            raw = getattr(response, "raw", {}) or {}
            stats.total_prompt_tokens += raw.get("prompt_eval_count", 0)
            stats.total_completion_tokens += raw.get("eval_count", 0)

        if self._debug:
            trunc = _detect_truncation(content)
            trunc_note = f"\n⚠ {trunc}" if trunc else ""
            _dbg(self._debug,
                 f"LLAMA RAW OUTPUT [{latency_ms}ms]{trunc_note}",
                 content, iteration)

        return content

    def _is_preparatory_result(self, question: str, exec_result: dict) -> bool:
        q_lower = question.lower()
        data = exec_result.get("result", {})
        columns = [c.lower() for c in data.get("columns", [])]
        row_count = data.get("row_count", 0)

        wants_content = bool(re.search(
            r'\b(question|label|answer|response|what.*in)\b', q_lower))
        has_content = any(c in columns for c in [
            "label", "answer_text", "answer_numeric",
            "inspection_score", "gp_score", "total_inspection_hours",
            "cause", "correction", "corrective_action", "responsible",
            "status", "progress_stage", "capex", "opex",
            "target_close_out_date", "schedule_date", "due_date",
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