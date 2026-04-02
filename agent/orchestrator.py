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
          - Dispatch tool → append result as next user turn
          - Repeat

        If max_iterations is exhausted, force a summary response.
        """
        def _emit(event):
            if on_step is not None:
                on_step(event)

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

            # 5. Handle result — three paths:
            #
            # A) generate_sql with valid SQL → auto-execute, auto-synthesize, return.
            # B) terminal tools (list_forms, semantic_search, execute_sql) → auto-synthesize, return.
            # C) non-terminal tools (lookup_form, get_schema) or failures → append context, loop.
            #
            # Paths A and B never loop back; the answer is produced by a single
            # focused synthesis call instead of asking the model to call final_answer.

            # --- Path A: generate_sql ---
            if tool_name == "generate_sql" and tool_result.get("success"):
                sql_data = tool_result.get("result", {})
                if sql_data.get("validation", {}).get("passed"):
                    sql = sql_data.get("sql", "")

                    exec_start = time.time()
                    exec_result = self.registry.call("execute_sql", {"sql": sql})
                    exec_ms = int((time.time() - exec_start) * 1000)
                    _auto_step = AgentStep(
                        iteration=iteration,
                        thought="[auto] generate_sql validated → executing immediately",
                        tool="execute_sql",
                        args={"sql": sql},
                        result=exec_result,
                        duration_ms=exec_ms,
                    )
                    steps.append(_auto_step)
                    _emit(_auto_step)

                    answer = self._auto_synthesize(question, "execute_sql", exec_result)
                    return self._make_result(question, answer, steps, iteration, _emit)

                else:
                    # Validation failed — loop back so model can retry
                    errors = (tool_result.get("result", {})
                              .get("validation", {})
                              .get("errors", [str(tool_result.get("error", "unknown"))]))
                    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw_output))
                    messages.append(ChatMessage(
                        role=MessageRole.USER,
                        content=(
                            f"generate_sql failed validation.\n"
                            f"Errors: {'; '.join(errors)}\n"
                            f"Try again with a corrected question or schema_hint."
                        ),
                    ))
                    continue

            # --- Path B: terminal tools with data → synthesize and return ---
            if tool_name in self._TERMINAL_TOOLS and tool_result.get("success"):
                answer = self._auto_synthesize(question, tool_name, tool_result)
                return self._make_result(question, answer, steps, iteration, _emit)

            # --- Path C: non-terminal tools or failures → continue loop ---
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw_output))
            messages.append(ChatMessage(
                role=MessageRole.USER,
                content=self._build_context_message(tool_name, tool_result),
            ))

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

        return AgentResult(
            question=question,
            answer=answer,
            steps=steps,
            total_iterations=max_iterations,
            success=False,  # didn't reach final_answer naturally
        )

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

    # Tools whose result is always sufficient to answer — no further agent
    # loop iteration needed; the orchestrator synthesizes the answer directly.
    _TERMINAL_TOOLS = {"list_forms", "semantic_search", "execute_sql"}

    def _make_result(
        self,
        question: str,
        answer: str,
        steps: list,
        iteration: int,
        emit_fn=None,
    ) -> AgentResult:
        """Wrap a completed answer in AgentResult and emit the final step."""
        final_step = AgentStep(
            iteration=iteration,
            thought="[auto] synthesized from tool result",
            tool="final_answer",
            args={"answer": answer},
            result={"answer": answer},
            duration_ms=0,
        )
        steps.append(final_step)
        if emit_fn is not None:
            emit_fn(final_step)
        return AgentResult(
            question=question,
            answer=answer,
            steps=steps,
            total_iterations=iteration,
            success=True,
        )

    def _auto_synthesize(
        self, question: str, tool_name: str, tool_result: dict
    ) -> str:
        """
        Produce a natural-language answer from a terminal tool result without
        running another agent loop iteration.

        Uses a single focused completion call (no JSON format required) so the
        model just writes a plain answer.
        """
        data = tool_result.get("result", {})
        prompt = (
            f"Answer the following question based on the data provided.\n\n"
            f"Question: {question}\n\n"
            f"Data ({tool_name}):\n"
            f"{json.dumps(data, default=str, indent=2)}\n\n"
            f"Give a concise, direct answer. Do not mention JSON, tool names, "
            f"or data formats — just answer the question naturally."
        )
        try:
            response = self.reasoning_llm.complete(prompt)
            return str(response).strip()
        except Exception:
            return json.dumps(data, default=str)

    def _build_context_message(self, tool_name: str, tool_result: dict) -> str:
        """
        Build the user-turn message for Path C tools (lookup_form, get_schema)
        and tool failures.  Terminal tools and generate_sql are handled in query().
        """
        if not tool_result.get("success"):
            return (
                f"Tool '{tool_name}' failed: {tool_result.get('error', 'unknown error')}\n"
                f"Consider a different approach."
            )

        if tool_name == "lookup_form":
            names = tool_result.get("result", [])
            if names:
                names_fmt = ", ".join(f"'{n}'" for n in names)
                return (
                    f"lookup_form found these exact names in the database: {names_fmt}\n\n"
                    f"Now call generate_sql using one of these exact names in schema_hint.\n"
                    f"Example: schema_hint='The exact name is: {names[0]}'"
                )
            return (
                f"lookup_form found no matches.\n"
                f"Call generate_sql directly, or call list_forms() to see all names."
            )

        # get_schema and anything else — plain JSON
        return json.dumps(tool_result, default=str)

    def _call_reasoning_llm(self, messages: list[ChatMessage]) -> str:
        """
        Call mistral:latest with the current message list.
        Returns the raw string content of the response.
        """
        response = self.reasoning_llm.chat(messages)
        return response.message.content.strip()
