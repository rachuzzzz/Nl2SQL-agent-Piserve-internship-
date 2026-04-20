"""
Agent Orchestrator — the main reasoning loop.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import create_engine, text as sql_text

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from core.semantic import SemanticQuestionIndex
from core.validator import SQLValidator
from core.schema_introspector import introspect_schema
from agent.prompts import SYSTEM_PROMPT, SQL_GENERATION_PROMPT, FORCE_SUMMARY_PROMPT
from agent.tools import ToolRegistry
from agent.tool_dispatcher import parse_tool_call, dispatch


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
    """Observability metrics for a single query."""
    total_llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_latency_ms: int = 0
    total_sql_exec_ms: int = 0
    total_wall_ms: int = 0


@dataclass
class ConversationTurn:
    """Compact record of one Q&A turn for conversation memory."""
    question: str
    answer: str
    tool_used: str           # primary tool that produced the answer
    sql: str = ""            # if SQL was executed, the query
    total_count: int = 0     # if truncated, the real total
    columns: list = field(default_factory=list)


class ConversationSession:
    """
    Maintains conversation state across multiple query() calls.
    Stores the last N turns as compact summaries (not raw data).
    """
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

    def build_context_block(self) -> str:
        """Build a compact conversation history for the system prompt."""
        if not self.turns:
            return ""

        lines = ["━━━ CONVERSATION HISTORY (last turns) ━━━"]
        for i, t in enumerate(self.turns, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  User: {t.question}")
            # Keep answer compact — first 200 chars
            short_answer = t.answer[:200] + ("..." if len(t.answer) > 200 else "")
            lines.append(f"  Answer: {short_answer}")
            if t.sql:
                lines.append(f"  SQL used: {t.sql[:150]}")
            if t.total_count > 0:
                lines.append(f"  Total rows: {t.total_count} (showed first 50)")
            if t.columns:
                lines.append(f"  Columns: {t.columns}")
        lines.append("")
        lines.append(
            "If the user says 'show more', 'show the rest', 'continue', 'remaining', "
            "'next', or 'show all' — they want more rows from the LAST query above. "
            "Re-use the same SQL with OFFSET to skip already-shown rows, "
            "or remove the LIMIT to show all. Do NOT generate a new query."
        )
        lines.append(
            "If the user says 'filter by X', 'only the Y ones', 'sort by Z' — "
            "modify the LAST SQL by adding/changing WHERE or ORDER BY clauses."
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
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
# Inspection-intent detector — auto-injects schema hints when the agent
# calls generate_sql for inspection-domain queries so the SQL model knows
# which tables and columns to use.
# ---------------------------------------------------------------------------

_INSPECTION_REPORT_RE = re.compile(
    r'\b(inspection.?score|gp.?score|inspection.?hour|total.?hour|'
    r'inspection.?report|inspection.?status|inspector|inspectee|'
    r'inspection.?type|inspection.?sub.?type|facility|'
    r'inspection.?per|inspections?\s+by\s+type|inspections?\s+per\s+cycle)\b',
    re.IGNORECASE,
)

_CORRECTIVE_ACTION_RE = re.compile(
    r'\b(corrective.?action|corrective|mitigative|finding|observation|'
    r'cause|correction|responsible|progress.?stage|adequacy|'
    r'capex|opex|expenditure|overdue|close.?out|target.?date|'
    r'pending.?with|implementation.?status|risk.?level|'
    r'open.?action|closed.?action|deferred)\b',
    re.IGNORECASE,
)

_SCHEDULE_CYCLE_RE = re.compile(
    r'\b(schedule|inspection.?schedule|inspection.?cycle|cycle|'
    r'due.?date|schedule.?date|portfolio|inspector.?portfolio|'
    r'cancellation)\b',
    re.IGNORECASE,
)

# Intent detection regexes — used to decide which tables to inject into schema_hint
# (These stay as regex since they're about question intent, not schema content)

def _detect_inspection_tables(question: str) -> list:
    """
    Detect which inspection tables the question is about.
    Returns a list of table names to include in the schema hint.
    """
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

        # Introspect database schema — replaces all hardcoded schema in prompts
        self.db_schema = introspect_schema(self.db_engine)

        # Build the actual prompts with introspected schema injected.
        # Use .replace() not .format() — prompts contain JSON examples
        # with curly braces that would crash .format().
        self._system_prompt = SYSTEM_PROMPT.replace(
            "{inspection_schema}",
            self.db_schema.for_system_prompt()
        )
        self._sql_prompt = SQL_GENERATION_PROMPT.replace(
            "{inspection_schema_sql}",
            self.db_schema.for_sql_prompt()
        )

        print(f"  Loading embedding model: {embedding_model}...")
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = embed_model
        print("  ✓ Embedding model loaded")

        self.semantic_index = SemanticQuestionIndex(
            db_engine=self.db_engine, embed_model=embed_model)
        self.validator = SQLValidator()

        print(f"  Configuring reasoning LLM: {chat_model}...")
        self.reasoning_llm = Ollama(model=chat_model, base_url=ollama_url,
            temperature=0.1, request_timeout=120.0,
            additional_kwargs={"num_predict": 1024})
        print(f"  ✓ Reasoning LLM ready ({chat_model})")

        print(f"  Configuring SQL LLM: {sql_model}...")
        self.sql_llm = Ollama(model=sql_model, base_url=ollama_url,
            temperature=0.0, request_timeout=120.0,
            additional_kwargs={"num_predict": 512, "top_p": 0.9})
        print(f"  ✓ SQL LLM ready ({sql_model})")

        self.registry = ToolRegistry(db_engine=self.db_engine,
            semantic_index=self.semantic_index, validator=self.validator,
            sql_llm=self.sql_llm, sql_prompt=self._sql_prompt)

        print(f"\n{'='*50}")
        print(f"  AgentOrchestrator ready")
        print(f"  Reasoning: {chat_model}")
        print(f"  SQL gen:   {sql_model}")
        print(f"  Embedding: {embedding_model}")
        print(f"{'='*50}\n")

    # ------------------------------------------------------------------ #

    def query(self, question, max_iterations=8, on_step=None, session=None):
        """
        Main query method.

        Args:
            question:       Natural language question from the user.
            max_iterations: Max LLM reasoning steps before forced summary.
            on_step:        Callback for live trace output.
            session:        ConversationSession for multi-turn context.
                            Pass the same session object across calls to enable
                            "show more", "filter by X", and other follow-ups.
        """
        import time as _time
        query_start = _time.time()
        stats = QueryStats()

        def _emit(event):
            if on_step: on_step(event)

        # Build system prompt — inject conversation history if available
        system_content = self._system_prompt
        if session and session.turns:
            system_content += "\n\n" + session.build_context_block()

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_content),
            ChatMessage(role=MessageRole.USER, content=question),
        ]
        steps = []
        gen_sql_failures = 0
        # Track the last executed SQL and result for session storage
        _last_sql = ""
        _last_total_count = 0
        _last_columns = []
        _last_tool = ""

        # ============================================================
        # Deterministic fast-path: pagination ("show the rest", "show
        # more", "show all", "continue", "remaining", "next").
        # If the session has a previous SQL query with more rows than
        # were displayed, re-run it WITHOUT the LLM. Instant, reliable.
        # ============================================================
        if session and session.last_turn and session.last_turn.sql:
            q_lower = question.lower().strip()
            is_pagination = bool(re.search(
                r'^(show|see|display|give|list)?\s*(the\s+)?(rest|more|remaining|all'
                r'|next|continue|full\s+list|other|everything)',
                q_lower
            ))
            if is_pagination:
                prev = session.last_turn
                _emit({"_event": "thinking", "iteration": 1})

                # Re-run the previous SQL — it already returns up to 100 rows
                er = self.registry.call("execute_sql", {"sql": prev.sql})
                s = AgentStep(1, "[auto] pagination — re-running previous query",
                    "execute_sql", {"sql": prev.sql}, er, 0)
                steps.append(s); _emit(s)

                if er.get("success"):
                    ans = self._synthesize(question, "execute_sql", er,
                                           max_display_rows=100)
                    stats.total_wall_ms = int((time.time() - query_start) * 1000)
                    # Update session with the full result
                    exec_data = er.get("result", {})
                    if session is not None:
                        session.add_turn(ConversationTurn(
                            question=question, answer=ans[:300],
                            tool_used="execute_sql", sql=prev.sql,
                            total_count=exec_data.get("total_count",
                                                       exec_data.get("row_count", 0)),
                            columns=exec_data.get("columns", []),
                        ))
                    return AgentResult(question, ans, steps, 1, True, stats)

        for iteration in range(1, max_iterations + 1):
            t0 = time.time()
            _emit({"_event": "thinking", "iteration": iteration})

            raw = self._call_llm(messages, stats)

            try:
                tc = parse_tool_call(raw)
            except ValueError as exc:
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=
                    f"Not valid JSON: {exc}\nRespond with ONLY a JSON tool call."))
                s = AgentStep(iteration, "[parse error]", "[parse_error]",
                    {}, {"error": str(exc)}, int((time.time()-t0)*1000))
                steps.append(s); _emit(s); continue

            thought, tool, args = tc.get("thought",""), tc["tool"], tc["args"]

            # --- Auto-inject inspection schema into generate_sql ---
            if tool == "generate_sql":
                existing = args.get("schema_hint", "")
                # Detect which inspection tables the question is about
                insp_tables = _detect_inspection_tables(
                    args.get("question", question)
                )
                if insp_tables:
                    schema_hint = self.db_schema.for_schema_hint(insp_tables)
                    if existing:
                        args["schema_hint"] = existing + "\n" + schema_hint
                    else:
                        args["schema_hint"] = schema_hint

                # Extract numeric LIMIT from the ORIGINAL user question.
                # "last 10 inspections" / "top 5 corrective actions" / "first 20"
                # The LLM often drops the number when rephrasing, so we inject it.
                limit_match = re.search(
                    r'\b(?:last|top|first|recent|latest)\s+(\d+)\b', question, re.IGNORECASE
                )
                if not limit_match:
                    limit_match = re.search(
                        r'\b(\d+)\s+(?:most\s+recent|latest|last|inspections?|actions?|reports?)\b',
                        question, re.IGNORECASE
                    )
                if limit_match:
                    n = int(limit_match.group(1))
                    if 1 <= n <= 500:
                        args["schema_hint"] = (
                            args.get("schema_hint", "") +
                            f"\nIMPORTANT: The user asked for exactly {n} results. "
                            f"Use LIMIT {n} in the query."
                        )

                tc["args"] = args

            if tool == "final_answer":
                ans = args.get("answer", "")
                s = AgentStep(iteration, thought, "final_answer", args,
                    {"answer": ans}, int((time.time()-t0)*1000))
                steps.append(s); _emit(s)
                # Finalize stats
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

            # --- Path A: generate_sql → auto-execute ---
            if tool == "generate_sql" and result.get("success"):
                sql_data = result.get("result", {})
                if sql_data.get("validation", {}).get("passed"):
                    sql = sql_data.get("sql", "")
                    et0 = time.time()
                    er = self.registry.call("execute_sql", {"sql": sql})
                    auto = AgentStep(iteration,
                        "[auto] generate_sql validated → executing",
                        "execute_sql", {"sql": sql}, er,
                        int((time.time()-et0)*1000))
                    steps.append(auto); _emit(auto)

                    if not er.get("success"):
                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                        hint = self._schema_hint_for_error(er.get("error",""))
                        messages.append(ChatMessage(role=MessageRole.USER, content=
                            f"SQL failed execution.\nError: {er.get('error','')}\n"
                            f"SQL: {sql}\n{hint}"
                            f"Fix the query. Use get_schema if unsure about columns."))
                        continue

                    # Track for session memory
                    exec_data = er.get("result", {})
                    _last_sql = sql
                    _last_total_count = exec_data.get("total_count",
                                                       exec_data.get("row_count", 0))
                    _last_columns = exec_data.get("columns", [])

                    if self._is_preparatory_result(question, er):
                        exec_data = er.get("result", {})
                        rows_preview = json.dumps(exec_data.get("rows", [])[:10], default=str)
                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                        messages.append(ChatMessage(role=MessageRole.USER, content=
                            f"Tool results:\nSQL: {sql}\n"
                            f"Result: {exec_data.get('row_count', 0)} rows, "
                            f"columns={exec_data.get('columns', [])}\n"
                            f"Data (first 10): {rows_preview}\n\n"
                            f"This looks like a preparatory step — the user's original "
                            f"question was: \"{question}\"\n"
                            f"If you have enough data to answer, call final_answer.\n"
                            f"If you need more data, call the next tool."))
                        continue

                    ans = self._synthesize(question, "execute_sql", er)
                    return self._finish(question, ans, steps, iteration, _emit, stats, session, query_start)
                else:
                    gen_sql_failures += 1
                    errs = sql_data.get("validation",{}).get("errors",["unknown"])
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
                        f"generate_sql failed validation ({gen_sql_failures}x).\n"
                        f"Errors: {'; '.join(errs)}\n"
                        f"Try a different approach or use get_schema first."))
                    continue

            # --- Path B: terminal tools → synthesize ---
            if tool in self._TERMINAL_TOOLS and result.get("success"):
                ans = self._synthesize(question, tool, result)
                return self._finish(question, ans, steps, iteration, _emit, stats, session, query_start)

            # --- Path C: non-terminal → continue ---
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
            messages.append(ChatMessage(role=MessageRole.USER,
                content=self._context_msg(tool, result, question)))

        # Max iterations
        _emit({"_event": "thinking", "iteration": max_iterations+1, "forced": True})
        messages.append(ChatMessage(role=MessageRole.USER, content=FORCE_SUMMARY_PROMPT))
        raw = self._call_llm(messages, stats)
        try:
            ans = parse_tool_call(raw).get("args",{}).get("answer","") or raw
        except ValueError:
            ans = raw
        s = AgentStep(max_iterations+1, "[forced summary]", "final_answer",
            {"answer": ans}, {"answer": ans}, 0)
        steps.append(s); _emit(s)
        stats.total_wall_ms = int((time.time() - query_start) * 1000)
        # Save to session
        if session is not None:
            session.add_turn(ConversationTurn(
                question=question, answer=ans[:300],
                tool_used="forced_summary", sql=_last_sql,
                total_count=_last_total_count, columns=_last_columns,
            ))
        return AgentResult(question, ans, steps, max_iterations, False, stats)

    def test_connection(self):
        try:
            with self.db_engine.connect() as conn:
                n = conn.execute(sql_text("SELECT COUNT(*) FROM fb_forms")).fetchone()[0]
            print(f"  ✓ Connection OK — {n} forms"); return True
        except Exception as e:
            print(f"  ✗ Connection failed: {e}"); return False

    @classmethod
    def from_env(cls, env_path=".env"):
        from dotenv import load_dotenv; load_dotenv(env_path)
        g = os.getenv
        return cls(
            db_connection_string=f"postgresql://{g('DB_USER','postgres')}:{g('DB_PASSWORD','')}@{g('DB_HOST','localhost')}:{g('DB_PORT','5432')}/{g('DB_NAME','postgres')}",
            db_schema=g("DB_SCHEMA","public"),
            ollama_url=g("OLLAMA_URL","http://localhost:11434"),
            sql_model=g("OLLAMA_SQL_MODEL","sqlcoder:7b"),
            chat_model=g("OLLAMA_CHAT_MODEL","qwen2.5:14b-instruct-q4_K_M"),
            embedding_model=g("EMBEDDING_MODEL","BAAI/bge-small-en-v1.5"),
        )

    # ------------------------------------------------------------------ #
    _TERMINAL_TOOLS = {"list_forms", "execute_sql", "query_answers",
                       "get_answer_summary"}

    def _finish(self, question, answer, steps, iteration, emit_fn=None,
               stats=None, session=None, query_start=None):
        s = AgentStep(iteration, "[auto] synthesized", "final_answer",
            {"answer": answer}, {"answer": answer}, 0)
        steps.append(s)
        if emit_fn: emit_fn(s)

        # Finalize stats
        if stats and query_start:
            stats.total_wall_ms = int((time.time() - query_start) * 1000)
            # Extract SQL exec time from steps
            for step in steps:
                if step.tool == "execute_sql":
                    stats.total_sql_exec_ms += step.duration_ms

        # Extract last SQL metadata from steps for session storage
        last_sql = ""
        last_total = 0
        last_cols = []
        for step in reversed(steps):
            if step.tool == "execute_sql" and step.result.get("success"):
                data = step.result.get("result", {})
                last_sql = step.args.get("sql", "")
                last_total = data.get("total_count", data.get("row_count", 0))
                last_cols = data.get("columns", [])
                break

        # Save turn to session
        if session is not None:
            session.add_turn(ConversationTurn(
                question=question,
                answer=answer[:300],
                tool_used=steps[-2].tool if len(steps) >= 2 else "unknown",
                sql=last_sql,
                total_count=last_total,
                columns=last_cols,
            ))

        return AgentResult(question, answer, steps, iteration, True,
                           stats or QueryStats())

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    _COUNT_INTENT_RE = re.compile(
        r'\b(how\s+many|count|number\s+of|total\s+(?:number\s+of\s+)?)\b',
        re.IGNORECASE,
    )
    _ENUMERATE_INTENT_RE = re.compile(
        r'\b(show\s+all|list\s+all|all\s+(?:the\s+)?forms|every\s+form|which\s+forms)\b',
        re.IGNORECASE,
    )

    def _synthesize(self, question, tool_name, tool_result, max_display_rows=20):
        data = tool_result.get("result", {})
        q_lower = question.lower()

        # ---- list_forms fast-paths ----
        if tool_name == "list_forms" and isinstance(data, list):
            if self._COUNT_INTENT_RE.search(q_lower):
                return f"There are {len(data)} forms."
            if self._ENUMERATE_INTENT_RE.search(q_lower) or not q_lower.strip():
                if not data:
                    return "No forms found."
                lines = [f"Found {len(data)} form(s):"]
                for i, f in enumerate(data, 1):
                    name = f.get("name") or "(unnamed)"
                    status = f.get("status", "?")
                    active = "active" if f.get("active") else "inactive"
                    lines.append(f"{i}. {name} — {status}, {active}")
                return "\n".join(lines)

        # ---- get_answer_summary fast-path ----
        if tool_name == "get_answer_summary" and isinstance(data, dict):
            total = data.get("total_rows", 0)
            distinct = data.get("distinct_forms", 0)
            earliest = data.get("earliest", "?")
            latest = data.get("latest", "?")
            statuses = data.get("status_breakdown", {})
            parts = [f"There are {total} submission(s)."]
            if distinct and distinct > 1:
                parts.append(f"Across {distinct} distinct forms.")
            if statuses:
                status_str = ", ".join(f"{k}: {v}" for k, v in statuses.items())
                parts.append(f"Status breakdown: {status_str}.")
            if earliest and earliest != "?" and latest and latest != "?":
                parts.append(f"Date range: {earliest} to {latest}.")
            return " ".join(parts)

        # ---- get_score_stats fast-path ----
        if tool_name == "get_score_stats" and isinstance(data, dict):
            avg = data.get("average_score")
            total = data.get("total_scores", 0)
            min_s = data.get("min_score")
            max_s = data.get("max_score")
            by_type = data.get("by_score_type", [])
            form_filter = data.get("form_name_filter") or "all forms"
            status_filter = data.get("status_filter") or "all statuses"
            if avg is None or total == 0:
                return f"No numeric scores found (filter: {status_filter}, {form_filter})."
            parts = [f"Average score: {avg}"]
            parts.append(f"(from {total} score entries, range {min_s} – {max_s}).")
            if by_type:
                type_lines = [
                    f"  • {t['score_type'] or 'untyped'}: avg {t['average']}, count {t['count']}"
                    for t in by_type[:5]
                ]
                parts.append("By score type:\n" + "\n".join(type_lines))
            return "\n".join(parts)

        # ---- execute_sql deterministic fast-paths ----
        if tool_name == "execute_sql" and isinstance(data, dict):
            rows = data.get("rows", [])
            cols = data.get("columns", [])
            truncated = data.get("truncated", False)
            total_count = data.get("total_count", len(rows))

            # --- Single-row aggregate result (COUNT, AVG, SUM, MIN, MAX) ---
            # Render deterministically — the LLM misreads these too often.
            if len(rows) == 1 and len(cols) <= 3:
                row = rows[0]
                # Check if all columns look like aggregates
                agg_names = {"count", "total", "avg", "sum", "min", "max",
                             "average", "avg_score", "avg_inspection_score",
                             "total_capex", "total_opex", "question_count",
                             "page_count", "module_count", "form_count",
                             "action_count", "report_count", "inspection_count"}
                col_lower = [c.lower() for c in cols]
                is_aggregate = all(
                    c in agg_names
                    or c.startswith("avg_") or c.startswith("sum_")
                    or c.startswith("total_") or c.startswith("count_")
                    or c.endswith("_count") or c.endswith("_avg")
                    for c in col_lower
                )
                if is_aggregate:
                    parts = []
                    for col_name, value in row.items():
                        if value is None:
                            parts.append(f"{col_name}: no data")
                        else:
                            # Round floats for readability
                            try:
                                fval = float(value)
                                display = f"{fval:,.4f}".rstrip('0').rstrip('.')
                            except (TypeError, ValueError):
                                display = str(value)
                            parts.append(f"{col_name}: {display}")
                    return "  |  ".join(parts)

            # --- Multi-row grouped results (e.g. status breakdown) ---
            # When we have 2 columns and one looks like a name/category and
            # the other is a count, render as a deterministic list.
            if 2 <= len(rows) <= 30 and len(cols) == 2:
                col_lower = [c.lower() for c in cols]
                count_col = None
                label_col = None
                for i, c in enumerate(col_lower):
                    if c in ("count", "action_count", "inspection_count",
                             "report_count", "cnt") or c.endswith("_count"):
                        count_col = i
                    else:
                        label_col = i
                if count_col is not None and label_col is not None:
                    lines = []
                    for r in rows:
                        vals = list(r.values())
                        lines.append(f"  • {vals[label_col]}: {vals[count_col]}")
                    header = f"Results ({len(rows)} groups"
                    if truncated:
                        header += f", showing first {len(rows)} of {total_count}"
                    header += "):"
                    return header + "\n" + "\n".join(lines)

            # --- Deterministic multi-row result ---
            # The LLM consistently misreads multi-row results (e.g. says
            # "none are overdue" while looking at 50 overdue rows because
            # a null column confused it). Build the answer directly.
            if rows:
                # Header with truncation info
                if truncated and total_count > 0:
                    header = f"Found {total_count} result(s) (showing first {len(rows)}):"
                elif truncated:
                    header = f"Showing first {len(rows)} result(s) (more exist):"
                else:
                    header = f"Found {len(rows)} result(s):"

                # Format rows as a readable list
                display_rows = rows[:max_display_rows]
                num_cols = len(cols)
                lines = [header]
                for i, row in enumerate(display_rows, 1):
                    # Build a compact one-line representation
                    parts = []
                    for col, val in row.items():
                        if val is None:
                            # For narrow results (1-2 cols), show nulls
                            # explicitly — the null IS the answer (e.g.
                            # "inspection_score: N/A" not a blank line).
                            # For wide results (3+), skip nulls for readability.
                            if num_cols <= 2:
                                parts.append(f"{col}: N/A")
                            continue
                        # Truncate long values
                        val_str = str(val)
                        if len(val_str) > 60:
                            val_str = val_str[:57] + "..."
                        parts.append(f"{col}: {val_str}")
                    if parts:
                        lines.append(f"  {i}. {' | '.join(parts)}")
                    else:
                        lines.append(f"  {i}. (no data)")

                if len(rows) > max_display_rows:
                    lines.append(f"  ... and {len(rows) - max_display_rows} more rows shown")
                if truncated and total_count > len(rows):
                    lines.append(f"  (Total matching: {total_count})")

                return "\n".join(lines)

            # Empty result
            return "The query returned no results."

        # ---- general path ----
        return self._llm_answer(question, tool_name, data)

    def _llm_answer(self, question, tool_name, data, extra_instruction=""):
        prompt = (
            f"Answer the question using ONLY the exact values from the data. "
            f"Do not estimate, round, or invent numbers. If asked to list "
            f"items, enumerate ALL of them — do not truncate or sample. "
            f"If the data is empty, say so clearly.\n"
            f"{extra_instruction}\n\n"
            f"Question: {question}\n\n"
            f"Data ({tool_name}):\n{json.dumps(data, default=str, indent=2)}\n\n"
            f"Give a concise, accurate answer. No JSON or tool references."
        )
        try:
            return str(self.reasoning_llm.complete(prompt)).strip()
        except Exception:
            return json.dumps(data, default=str)

    # ------------------------------------------------------------------ #
    # Context messages
    # ------------------------------------------------------------------ #

    def _context_msg(self, tool_name, tool_result, question=""):
        if not tool_result.get("success"):
            error = tool_result.get('error', 'unknown')

            if tool_name == "generate_sql":
                q_lower = question.lower()
                hint = ""
                # Provide correct fallback patterns based on question type
                if re.search(r'\b(corrective.?action|cause|correction|capex|opex|overdue)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK: Call execute_sql directly. Table is "
                        "inspection_corrective_action with plain columns: "
                        "cause, correction, corrective_action, responsible, "
                        "status, progress_stage, capex, opex, target_close_out_date, "
                        "completed_on. Use ILIKE not LIKE."
                    )
                elif re.search(r'\b(inspection.?score|gp.?score|inspection.?hour|inspector)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK: Call execute_sql directly. Table is "
                        "inspection_report with: inspection_score (numeric), "
                        "gp_score (numeric), total_inspection_hours (numeric), "
                        "status, inspector_user_id, inspection_type_id. "
                        "JOIN inspection_type for type names."
                    )
                elif re.search(r'\b(question|page)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK: Call execute_sql with JSONB pattern:\n"
                        "  SELECT COUNT(*) FROM fb_forms f\n"
                        "  JOIN fb_modules m ON f.module_id = m.id\n"
                        "  JOIN fb_translation_json tj ON f.translations_id = tj.id,\n"
                        "       jsonb_array_elements(tj.translations) AS elem\n"
                        "  WHERE m.name ILIKE '%MODULE_NAME%'\n"
                        "    AND elem->>'language' = 'eng'\n"
                        "    AND elem->>'attribute' = 'NAME'\n"
                        "    AND elem->>'entityType' = 'QUESTION';"
                    )
                elif re.search(r'\b(module)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK: Call execute_sql:\n"
                        "  SELECT name FROM fb_modules WHERE name ILIKE '%KEYWORD%' "
                        "ORDER BY name LIMIT 100;"
                    )

                return (
                    f"Tool 'generate_sql' failed: {error}\n"
                    f"The SQL model may be overloaded or crashed.{hint}"
                )

            return f"Tool '{tool_name}' failed: {error}\nTry a different approach."

        if tool_name == "semantic_search":
            data = tool_result.get("result", [])
            if data:
                fmt = "\n".join(
                    f"  - \"{m['text']}\" (form: {m['form_name']}, score: {m['score']})"
                    for m in data[:5]
                )
                distinct_forms = sorted({m["form_name"] for m in data if m.get("form_name")})

                needs_meta = bool(re.search(
                    r'\b(when|which day|what date|created|modified|timestamp|who created|who made)\b',
                    question.lower()))
                asks_submissions = bool(re.search(
                    r'\b(submission|submitted|response|responded|answer|answered|filled|scored?)\b',
                    question.lower()))

                if needs_meta:
                    top = data[0]
                    return (f"semantic_search found {len(data)} match(es):\n{fmt}\n\n"
                        f"User needs metadata. Top match in form '{top['form_name']}'.\n"
                        f"Call generate_sql with schema_hint mentioning form name "
                        f"'{top['form_name']}' and the metadata needed.")

                if asks_submissions:
                    return (f"semantic_search found {len(data)} match(es) in "
                        f"{len(distinct_forms)} form(s): {distinct_forms}\n{fmt}\n\n"
                        f"User asked about submissions. For EACH form, call "
                        f"resolve_answer_table then query_answers, then aggregate.")

                return (f"semantic_search found {len(data)} match(es):\n{fmt}\n\n"
                    f"If this answers the question, call final_answer.\n"
                    f"If user needs counts/metadata, use generate_sql.")
            return ("semantic_search found no matches above threshold. "
                    "Call final_answer saying no forms ask about that topic.")

        if tool_name == "lookup_form":
            names = tool_result.get("result", [])
            source = tool_result.get("_source", "ilike")
            if names:
                source_label = {
                    "ilike": "exact-substring",
                    "semantic_form": "semantic form-name",
                    "semantic_question": "semantic question-label (indirect)",
                }.get(source, source)
                return (f"lookup_form found {len(names)} candidate(s) via {source_label}:\n"
                        f"  {', '.join(repr(n) for n in names)}\n"
                        f"Use ONE of these exact names. Drop filler words (form/the) from ILIKE.")
            return ("lookup_form found no matches. "
                    "Consider list_forms() to show what's available.")

        if tool_name == "resolve_answer_table":
            info = tool_result.get("result", {})
            if not info:
                return "resolve_answer_table found no match. Try lookup_form first."

            ambiguous = info.get("ambiguous", False)
            missing_cols = info.get("missing_required_columns", [])
            resolved_via = info.get("resolved_via", "form_name")
            module_candidates = info.get("all_module_candidates", [])
            forms_in_module = info.get("forms_in_module", [])

            candidates_block = ""
            if ambiguous and module_candidates:
                lines = [f"  - {c['module_name']} ({'standard' if c.get('has_answer_data') else 'workflow'})"
                         for c in module_candidates[:10]]
                candidates_block = "\nAll matching modules:\n" + "\n".join(lines)

            forms_block = ""
            if forms_in_module:
                fnames = [f["form_name"] for f in forms_in_module[:10]]
                forms_block = f"\n  Forms in module: {fnames}"

            if info.get("table_exists") and not missing_cols:
                via = {"module_uuid": "UUID", "module_name": "module name"}.get(resolved_via, "form name")
                return (
                    f"resolve_answer_table resolved (via {via}):\n"
                    f"  Module: '{info['module_name']}'\n"
                    f"  Table: {info['answer_table']} (standard shape)"
                    f"{forms_block}{candidates_block}\n\n"
                    f"Call get_score_stats or query_answers with "
                    f"answer_table=\"{info['answer_table']}\"."
                )

            if info.get("table_exists") and missing_cols:
                return (
                    f"Table {info['answer_table']} exists but missing: {missing_cols}.\n"
                    f"Columns: {info.get('columns', [])}\n"
                    f"This is a WORKFLOW table — use generate_sql/execute_sql."
                    f"{candidates_block}"
                )

            return (
                f"Table {info.get('answer_table')} does NOT exist.\n"
                f"{candidates_block}\n"
                f"No submissions yet."
            )

        if tool_name == "get_answer_summary":
            data = tool_result.get("result", {})
            return (
                f"Summary: {data.get('total_rows', 0)} rows, "
                f"{data.get('distinct_forms', 0)} forms, "
                f"statuses={data.get('status_breakdown', {})}.\n"
                f"Call final_answer or query_answers for details."
            )

        if tool_name == "get_score_stats":
            data = tool_result.get("result", {})
            avg = data.get("average_score")
            total = data.get("total_scores", 0)
            if avg is None or total == 0:
                stats = "No numeric scores found."
            else:
                stats = (f"Average: {avg}, Min: {data.get('min_score')}, "
                         f"Max: {data.get('max_score')}, Count: {total}")
                by_type = data.get("by_score_type", [])
                if by_type:
                    stats += "\nBy type: " + ", ".join(
                        f"{t.get('score_type','?')}={t.get('average')}" for t in by_type[:5])

            return (
                f"get_score_stats for {data.get('table', '?')}:\n  {stats}\n\n"
                f"If comparing with another module, call resolve_answer_table "
                f"for the next one, then get_score_stats again.\n"
                f"If this answers the question, call final_answer."
            )

        return json.dumps(tool_result, default=str)

    def _schema_hint_for_error(self, error_msg):
        parts = []
        tm = re.search(r'relation "(\w+)" does not exist', error_msg)
        if tm:
            bad = tm.group(1)
            sr = self.registry.call("get_schema", {})
            if sr.get("success"):
                tables = sr["result"].get("tables", [])
                similar = [t for t in tables if bad.replace("fb_","").split("_")[0] in t]
                if similar:
                    parts.append(f"Table '{bad}' doesn't exist. Similar: {similar}")
                    for t in similar[:2]:
                        cr = self.registry.call("get_schema", {"table_name": t})
                        if cr.get("success"):
                            parts.append(f"  {t} columns: {[c['name'] for c in cr['result'].get('columns',[])]}")
                else:
                    parts.append(f"Table '{bad}' doesn't exist. Available: {[t for t in tables if t.startswith('fb_')][:15]}")
        cm = re.search(r'column "(\w+)" does not exist', error_msg)
        if cm and not tm:
            parts.append(f"Column '{cm.group(1)}' doesn't exist. Use get_schema to check.")
        return "\n".join(parts) + "\n" if parts else ""

    def _call_llm(self, messages, stats=None):
        """
        Call the reasoning LLM and return the response text.
        If stats is provided, accumulates token counts and latency.
        """
        t0 = time.time()
        response = self.reasoning_llm.chat(messages)
        latency_ms = int((time.time() - t0) * 1000)

        content = response.message.content.strip()

        if stats:
            stats.total_llm_calls += 1
            stats.total_llm_latency_ms += latency_ms
            # Ollama returns token counts in the raw response
            raw = getattr(response, 'raw', {}) or {}
            stats.total_prompt_tokens += raw.get('prompt_eval_count', 0)
            stats.total_completion_tokens += raw.get('eval_count', 0)

        return content

    def _is_preparatory_result(self, question: str, exec_result: dict) -> bool:
        q_lower = question.lower()
        data = exec_result.get("result", {})
        columns = [c.lower() for c in data.get("columns", [])]
        row_count = data.get("row_count", 0)

        wants_content = bool(re.search(
            r'\b(question|label|title|page|element|content|what.*in)\b', q_lower))
        # "Content" means the result contains actual data the user asked for,
        # not just IDs/names for a follow-up step. Includes both form-builder
        # labels AND inspection-domain data columns.
        has_content = any(c in columns for c in [
            # Form builder
            "label", "title", "translatedtext", "text",
            "question_label", "question_text", "answer",
            # Inspection domain
            "inspection_score", "gp_score", "total_inspection_hours",
            "cause", "correction", "corrective_action", "responsible",
            "status", "progress_stage", "adequacy_status",
            "capex", "opex", "target_close_out_date",
            "schedule_date", "due_date",
        ])

        if wants_content and not has_content and row_count > 0:
            return True

        is_superlative = bool(re.search(
            r'\b(most\s+recent|recently|latest|newest|oldest|first|last)\b', q_lower))
        wants_details = bool(re.search(
            r'\b(question|page|element|what|show|list|answer|submission|response|score)\b',
            q_lower))

        # Only treat as preparatory if we asked for details about a form
        # (labels, questions) and got back metadata (just a name/id).
        # If we got inspection data back (scores, causes, etc.), it IS the answer.
        wants_submissions = bool(re.search(
            r'\b(submission|submitted|response|answer|score|filled)\b', q_lower))
        if wants_submissions and columns == ["name"] and row_count >= 1:
            return True

        if is_superlative and wants_details and not has_content:
            return True

        return False