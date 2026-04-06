"""
Agent Orchestrator — the main reasoning loop.

Implements a synchronous tool-calling agent loop:
  1. Send system prompt + user question to qwen2.5:14b-instruct-q4_K_M.
  2. Parse the JSON tool call from the response.
  3. If tool == "final_answer", return the result.
  4. Otherwise dispatch the tool, append the result as a user turn, repeat.
  5. If max_iterations is reached, force a summary call.

No LangChain, no LangGraph, no LlamaIndex agent abstractions.
All LLM calls go via Ollama using llama_index's Ollama wrapper for
consistency with the rest of the codebase.
"""

import json
import os
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
from agent.prompts import SYSTEM_PROMPT, FORCE_SUMMARY_PROMPT
from agent.tools import ToolRegistry
from agent.tool_dispatcher import parse_tool_call, dispatch


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """A single iteration of the agent loop."""
    iteration: int
    thought: str
    tool: str
    args: dict
    result: dict
    duration_ms: int


@dataclass
class AgentResult:
    """Complete result returned by AgentOrchestrator.query()."""
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    success: bool = True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Connects all components and runs the agent loop.

    Usage:
        orch = AgentOrchestrator.from_env()
        result = orch.query("Which forms ask about age?")
        print(result.answer)
        for step in result.steps:
            print(step.tool, step.duration_ms, "ms")
    """

    def __init__(
        self,
        db_connection_string: str,
        db_schema: str = "public",
        ollama_url: str = "http://localhost:11434",
        sql_model: str = "sqlcoder:7b",
        chat_model: str = "qwen2.5:14b-instruct-q4_K_M",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        print("Initialising AgentOrchestrator...")

        # --- Database ---
        print("  Connecting to database...")
        self.db_engine = create_engine(db_connection_string)
        with self.db_engine.connect() as conn:
            conn.execute(sql_text("SELECT 1")).fetchone()
        print("  ✓ Database connected")

        # --- Embedding model (shared by semantic index and nothing else here) ---
        print(f"  Loading embedding model: {embedding_model}...")
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = embed_model
        print("  ✓ Embedding model loaded")

        # --- Semantic question index ---
        self.semantic_index = SemanticQuestionIndex(
            db_engine=self.db_engine,
            embed_model=embed_model,
        )

        # --- Validator ---
        self.validator = SQLValidator()

        # --- LLM clients ---
        print(f"  Configuring reasoning LLM: {chat_model}...")
        self.reasoning_llm = Ollama(
            model=chat_model,
            base_url=ollama_url,
            temperature=0.1,       # low temp for consistent JSON output
            request_timeout=120.0,
            additional_kwargs={"num_predict": 1024},
        )
        print(f"  ✓ Reasoning LLM ready ({chat_model})")

        print(f"  Configuring SQL LLM: {sql_model}...")
        self.sql_llm = Ollama(
            model=sql_model,
            base_url=ollama_url,
            temperature=0.0,
            request_timeout=120.0,
            additional_kwargs={"num_predict": 512, "top_p": 0.9},
        )
        print(f"  ✓ SQL LLM ready ({sql_model})")

        # --- Tool registry ---
        self.registry = ToolRegistry(
            db_engine=self.db_engine,
            semantic_index=self.semantic_index,
            validator=self.validator,
            sql_llm=self.sql_llm,
        )

        # --- Session memory ---
        self.session_history: list[ChatMessage] = []

        print()
        print("=" * 50)
        print("  AgentOrchestrator ready")
        print(f"  Reasoning: {chat_model}")
        print(f"  SQL gen:   {sql_model}")
        print(f"  Embedding: {embedding_model}")
        print("=" * 50)
        print()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        max_iterations: int = 8,
        on_step=None,
        use_session: bool = True,
    ) -> AgentResult:
        """
        Run the agent loop for the given user question.

        on_step — optional callable invoked immediately on every event:
          on_step({"_event": "thinking", "iteration": N})  — LLM call starting
          on_step(AgentStep)                               — step completed

        Builds an initial message list [system, user], then iterates:
          - Call reasoning_llm.chat(messages)
          - Parse tool call JSON
          - If final_answer → return
          - Dispatch tool → format observation → append as next user turn
          - Repeat

        If max_iterations is exhausted, force a summary response.
        """
        def _emit(event):
            if on_step is not None:
                on_step(event)

        if use_session and self.session_history:
            messages: list[ChatMessage] = (
                [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]
                + self.session_history
                + [ChatMessage(role=MessageRole.USER, content=question)]
            )
        else:
            messages: list[ChatMessage] = [
                ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER, content=question),
            ]

        steps: list[AgentStep] = []

        for iteration in range(1, max_iterations + 1):
            step_start = time.time()

            # Signal that the LLM is being called — lets the UI print
            # a "thinking..." indicator immediately
            _emit({"_event": "thinking", "iteration": iteration})

            # 1. Call reasoning LLM
            raw_output = self._call_reasoning_llm(messages)

            # 2. Parse tool call
            try:
                tool_call = parse_tool_call(raw_output)
            except ValueError as exc:
                # Model produced non-JSON — treat as a failed step and retry
                # by injecting an error nudge
                error_msg = (
                    f"Your previous response was not valid JSON. Error: {exc}\n"
                    "Please respond with ONLY a valid JSON tool call object."
                )
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw_output))
                messages.append(ChatMessage(role=MessageRole.USER, content=error_msg))
                # Record as a parse-error step
                _step = AgentStep(
                    iteration=iteration,
                    thought="[parse error]",
                    tool="[parse_error]",
                    args={},
                    result={"error": str(exc)},
                    duration_ms=int((time.time() - step_start) * 1000),
                )
                steps.append(_step)
                _emit(_step)
                continue

            thought = tool_call.get("thought", "")
            tool_name = tool_call["tool"]
            args = tool_call["args"]

            # 3. Final answer?
            if tool_name == "final_answer":
                answer = args.get("answer", "")
                duration_ms = int((time.time() - step_start) * 1000)
                _step = AgentStep(
                    iteration=iteration,
                    thought=thought,
                    tool="final_answer",
                    args=args,
                    result={"answer": answer},
                    duration_ms=duration_ms,
                )
                steps.append(_step)
                _emit(_step)

                if use_session:
                    new_turns = messages[1 + len(self.session_history):]
                    self.session_history.extend(new_turns)
                    while len(self.session_history) > 20:
                        self.session_history = self.session_history[2:]

                return AgentResult(
                    question=question,
                    answer=answer,
                    steps=steps,
                    total_iterations=iteration,
                    success=True,
                )

            # 4. Dispatch tool
            tool_result = dispatch(tool_call, self.registry)
            duration_ms = int((time.time() - step_start) * 1000)

            _step = AgentStep(
                iteration=iteration,
                thought=thought,
                tool=tool_name,
                args=args,
                result=tool_result,
                duration_ms=duration_ms,
            )
            steps.append(_step)
            _emit(_step)

            # 5. Format observation and return to qwen — always, no branching
            observation = self._format_observation(tool_name, tool_result, args)
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw_output))
            messages.append(ChatMessage(role=MessageRole.USER, content=observation))
            # loop continues — qwen decides what to do next

        # --- max_iterations reached — force a summary ---
        force_start = time.time()
        _emit({"_event": "thinking", "iteration": max_iterations + 1, "forced": True})
        messages.append(
            ChatMessage(role=MessageRole.USER, content=FORCE_SUMMARY_PROMPT)
        )
        raw_summary = self._call_reasoning_llm(messages)

        try:
            summary_call = parse_tool_call(raw_summary)
            answer = summary_call.get("args", {}).get("answer", "").strip()
            if not answer or answer in ("<text>", "..."):
                answer = raw_summary
        except ValueError:
            answer = raw_summary

        _summary_step = AgentStep(
            iteration=max_iterations + 1,
            thought="[forced summary after max_iterations]",
            tool="final_answer",
            args={"answer": answer},
            result={"answer": answer},
            duration_ms=int((time.time() - force_start) * 1000),
        )
        steps.append(_summary_step)
        _emit(_summary_step)

        if use_session:
            new_turns = messages[1 + len(self.session_history):]
            self.session_history.extend(new_turns)
            while len(self.session_history) > 20:
                self.session_history = self.session_history[2:]

        return AgentResult(
            question=question,
            answer=answer,
            steps=steps,
            total_iterations=max_iterations,
            success=False,  # didn't reach final_answer naturally
        )

    def clear_session(self) -> None:
        """Reset session memory. Call between unrelated conversations."""
        self.session_history = []
        print("  [session] History cleared.")

    def test_connection(self) -> bool:
        """Quick connection check (mirrors NL2SQLEngine.test_connection)."""
        try:
            with self.db_engine.connect() as conn:
                count = conn.execute(
                    sql_text("SELECT COUNT(*) FROM fb_forms")
                ).fetchone()[0]
            print(f"  ✓ Connection OK — {count} forms in database")
            return True
        except Exception as exc:
            print(f"  ✗ Connection failed: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "AgentOrchestrator":
        """Create an orchestrator from a .env file."""
        from dotenv import load_dotenv
        load_dotenv(env_path)

        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "postgres")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "")
        db_schema = os.getenv("DB_SCHEMA", "public")
        connection_string = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        return cls(
            db_connection_string=connection_string,
            db_schema=db_schema,
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            sql_model=os.getenv("OLLAMA_SQL_MODEL", "sqlcoder:7b"),
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:14b-instruct-q4_K_M"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _format_observation(self, tool_name: str, tool_result: dict, args: dict = None) -> str:
        args = args or {}
        success = tool_result.get("success", False)
        data = tool_result.get("result")
        error = tool_result.get("error", "unknown error")

        if tool_name == "list_forms":
            if not success:
                return f"list_forms failed: {error}"
            if data:
                lines = [f"list_forms returned {len(data)} form(s):\n"]
                for form in data:
                    lines.append(f"  \u2022 {form['name']}  |  status: {form['status']}  |  active: {form['active']}\n")
                lines.append("\n")
                return "".join(lines)
            else:
                return (
                    "list_forms returned 0 forms. There may be no forms with that status, "
                    "or the status value may be incorrect. Valid values are DRAFT, PUBLISHED, DELETED."
                )

        if tool_name == "lookup_form":
            if not success:
                return f"lookup_form failed: {error}"
            names = data if data else []
            if names:
                lines = [f"lookup_form found {len(names)} matching form name(s) in the database:\n"]
                for name in names:
                    lines.append(f"  \u2022 {name}\n")
                lines.append("\n")
                lines.append("These are the exact names stored in the database. Use one of them verbatim in your next generate_sql or semantic_search call.")
                return "".join(lines)
            else:
                return (
                    "lookup_form found no forms matching that name. The form may not exist, "
                    "or the name may be spelled differently. Call list_forms() with no filter to see all available form names."
                )

        if tool_name == "semantic_search":
            if not success:
                return f"semantic_search failed: {error}"
            matches = data if data else []
            query_str = args.get("query", "")
            if matches:
                scores = [m["score"] for m in matches]
                max_score = max(scores)
                lines = [f"semantic_search found {len(matches)} match(es) for '{query_str}':\n"]
                for m in matches:
                    lines.append(f"  \u2022 [{m['score']:.0%}] {m['entity_type']} \u2014 \"{m['text']}\" (form: {m['form_name']})\n")
                lines.append("\n")
                if max_score >= 0.75:
                    lines.append("Scores are strong. These results likely answer the question.")
                elif max_score >= 0.55:
                    lines.append(
                        "Scores are moderate. Results are probably relevant but consider "
                        "whether a SQL keyword search might be more precise."
                    )
                else:
                    lines.append(
                        "Scores are weak (all below 55%). These may not be reliable matches. "
                        "Consider using generate_sql with an ILIKE keyword search as an alternative approach."
                    )
                return "".join(lines)
            else:
                return (
                    "semantic_search found no matches above the 0.4 threshold. The concept may not appear "
                    "in any form labels. Consider using generate_sql with an ILIKE keyword search instead."
                )

        if tool_name == "generate_sql":
            if not success:
                return f"generate_sql failed entirely: {error}"
            sql_data = data or {}
            sql = sql_data.get("sql", "")
            validation = sql_data.get("validation", {})
            passed = validation.get("passed", False)
            errors = validation.get("errors", [])
            real_errors = [e for e in errors if not e.startswith("WARNING:")]
            warnings = [e for e in errors if e.startswith("WARNING:")]
            if passed:
                lines = [f"sqlcoder generated the following SQL:\n\n{sql}\n\nValidation passed with no blocking errors."]
                if warnings:
                    lines.append("\nValidator warnings (non-blocking):\n")
                    for w in warnings:
                        lines.append(f"  \u26a0 {w}\n")
                lines.append(
                    "\nReview this SQL carefully. If it looks correct for the question, "
                    "call execute_sql with it. If something looks wrong \u2014 incorrect table, "
                    "missing filter, wrong entityType \u2014 call generate_sql again with a corrected "
                    "schema_hint describing the exact pattern needed."
                )
                return "".join(lines)
            else:
                lines = [f"sqlcoder generated the following SQL:\n\n{sql}\n\nValidation found blocking errors:\n"]
                for e in real_errors:
                    lines.append(f"  \u2717 {e}\n")
                lines.append(
                    "\nDo not call execute_sql with this SQL. Reason about the errors above. Common fixes:\n"
                    "  \u2014 If the error mentions a hallucinated column (e.g. fb_question.label),\n"
                    "    the SQL must use jsonb_array_elements() instead.\n"
                    "  \u2014 If the error mentions a wrong entityType, valid values are:\n"
                    "    QUESTION, PAGE, FORM only.\n"
                    "  \u2014 Provide a corrected schema_hint to generate_sql describing\n"
                    "    exactly which SQL pattern to use."
                )
                return "".join(lines)

        if tool_name == "execute_sql":
            if not success:
                return (
                    f"execute_sql failed: {error}\n"
                    "This is a database-level error, not a validator warning. The SQL\n"
                    "may have a syntax error or reference a table/column that does not\n"
                    "exist. Review the SQL and either correct it or call generate_sql\n"
                    "again with a more explicit schema_hint."
                )
            row_count = data.get("row_count", 0) if data else 0
            columns = data.get("columns", []) if data else []
            rows = data.get("rows", []) if data else []
            truncated = data.get("truncated", False) if data else False
            validator_warnings = data.get("validator_warnings", []) if data else []
            if row_count > 0:
                lines = [f"execute_sql returned {row_count} row(s)."]
                if truncated:
                    lines.append(" Results were truncated at 50 rows \u2014 there may be more data.\n"
                                 "Consider adding a more specific WHERE clause if you need all rows.\n")
                else:
                    lines.append("\n")
                lines.append(f"Columns: {columns}\n\n")
                # Format rows as readable table
                col_widths = {col: len(str(col)) for col in columns}
                for row in rows:
                    for col in columns:
                        col_widths[col] = max(col_widths[col], len(str(row.get(col, ""))))
                header = "  ".join(str(col).ljust(col_widths[col]) for col in columns)
                separator = "  ".join("-" * col_widths[col] for col in columns)
                lines.append(header + "\n")
                lines.append(separator + "\n")
                for row in rows:
                    lines.append("  ".join(str(row.get(col, "")).ljust(col_widths[col]) for col in columns) + "\n")
                if validator_warnings:
                    lines.append("\nValidator noted (informational):\n")
                    for w in validator_warnings:
                        lines.append(f"  \u26a0 {w}\n")
                return "".join(lines)
            else:
                lines = [
                    "execute_sql ran successfully but returned 0 rows.\n"
                    "Possible reasons:\n"
                    "  \u2014 The form name filter did not match any records (ILIKE is\n"
                    "    case-insensitive but the pattern must appear in the name).\n"
                    "  \u2014 The entityType filter excluded all records.\n"
                    "  \u2014 There genuinely is no data matching this query.\n"
                    "Consider calling lookup_form to verify the exact form name, or\n"
                    "broadening the WHERE clause."
                ]
                if validator_warnings:
                    lines.append("\nValidator also noted:\n")
                    for w in validator_warnings:
                        lines.append(f"  \u26a0 {w}\n")
                return "".join(lines)

        if tool_name == "validate_sql":
            if not success:
                return f"validate_sql raised an exception: {error}"
            passed = data.get("passed", False) if data else False
            real_errors = data.get("real_errors", []) if data else []
            warnings = data.get("warnings", []) if data else []
            if passed:
                lines = ["Validation passed. No blocking errors found.\n"]
                if warnings:
                    lines.append("\nNon-blocking warnings:\n")
                    for w in warnings:
                        lines.append(f"  \u26a0 {w}\n")
                lines.append("This SQL is safe to execute with execute_sql.")
                return "".join(lines)
            else:
                lines = [f"Validation found {len(real_errors)} blocking error(s):\n"]
                for e in real_errors:
                    lines.append(f"  \u2717 {e}\n")
                if warnings:
                    lines.append("\nAdditional warnings:\n")
                    for w in warnings:
                        lines.append(f"  \u26a0 {w}\n")
                lines.append("Do not execute this SQL. Review the errors and regenerate.")
                return "".join(lines)

        if tool_name == "get_schema":
            if not success:
                return f"get_schema failed: {error}"
            if "tables" in data:
                lines = [f"Database contains {len(data['tables'])} table(s):\n"]
                for t in data["tables"]:
                    lines.append(f"  \u2022 {t}\n")
                return "".join(lines)
            elif "columns" in data:
                table_name = args.get("table_name", "unknown")
                lines = [f"Table '{table_name}' has {len(data['columns'])} column(s):\n"]
                for col in data["columns"]:
                    lines.append(f"  \u2022 {col['name']}  ({col['type']})\n")
                return "".join(lines)
            return str(data)

        # Unknown tool
        from agent.tools import ToolRegistry
        return (
            f"Unknown tool '{tool_name}' was called. Available tools are: "
            f"{sorted(ToolRegistry.KNOWN_TOOLS)}. Correct your tool name."
        )

    def _call_reasoning_llm(
        self,
        messages: list[ChatMessage],
        retries: int = 3,
        backoff: float = 2.0,
    ) -> str:
        last_exc = None
        for attempt in range(retries):
            try:
                response = self.reasoning_llm.chat(messages)
                return response.message.content.strip()
            except Exception as exc:
                last_exc = exc
                if attempt < retries - 1:
                    wait = backoff * (attempt + 1)
                    print(
                        f"  [llm] Attempt {attempt + 1} failed: {exc}. "
                        f"Retrying in {wait:.0f}s...",
                        flush=True,
                    )
                    time.sleep(wait)
        raise RuntimeError(
            f"Reasoning LLM failed after {retries} attempts. "
            f"Last error: {last_exc}"
        )
