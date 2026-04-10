"""
Agent Orchestrator — the main reasoning loop.

Clean version: no review agent, no hardcoded SQL builders.
Trusts sqlcoder (15B) + auto-schema injection + validator guardrails.
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
from agent.prompts import SYSTEM_PROMPT, FORCE_SUMMARY_PROMPT
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
class AgentResult:
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    success: bool = True


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
            sql_llm=self.sql_llm)

        print(f"\n{'='*50}")
        print(f"  AgentOrchestrator ready")
        print(f"  Reasoning: {chat_model}")
        print(f"  SQL gen:   {sql_model}")
        print(f"  Embedding: {embedding_model}")
        print(f"{'='*50}\n")

    # ------------------------------------------------------------------ #

    def query(self, question, max_iterations=8, on_step=None):
        def _emit(event):
            if on_step: on_step(event)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=question),
        ]
        steps = []
        gen_sql_failures = 0

        for iteration in range(1, max_iterations + 1):
            t0 = time.time()
            _emit({"_event": "thinking", "iteration": iteration})

            raw = self._call_llm(messages)

            # Parse
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

            # Final answer
            if tool == "final_answer":
                ans = args.get("answer", "")
                s = AgentStep(iteration, thought, "final_answer", args,
                    {"answer": ans}, int((time.time()-t0)*1000))
                steps.append(s); _emit(s)
                return AgentResult(question, ans, steps, iteration, True)

            # Dispatch
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

                    ans = self._synthesize(question, "execute_sql", er)
                    return self._finish(question, ans, steps, iteration, _emit)
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
                                return self._finish(question, ans, steps, iteration, _emit)

                    messages.append(ChatMessage(role=MessageRole.USER, content=
                        f"generate_sql failed validation ({gen_sql_failures}x).\n"
                        f"Errors: {'; '.join(errs)}\n"
                        f"Try a different approach or use get_schema first."))
                    continue

            # --- Path B: terminal tools → synthesize ---
            if tool in self._TERMINAL_TOOLS and result.get("success"):
                ans = self._synthesize(question, tool, result)
                return self._finish(question, ans, steps, iteration, _emit)

            # --- Path C: non-terminal → continue ---
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
            messages.append(ChatMessage(role=MessageRole.USER,
                content=self._context_msg(tool, result, question)))

        # Max iterations
        _emit({"_event": "thinking", "iteration": max_iterations+1, "forced": True})
        messages.append(ChatMessage(role=MessageRole.USER, content=FORCE_SUMMARY_PROMPT))
        raw = self._call_llm(messages)
        try:
            ans = parse_tool_call(raw).get("args",{}).get("answer","") or raw
        except ValueError:
            ans = raw
        s = AgentStep(max_iterations+1, "[forced summary]", "final_answer",
            {"answer": ans}, {"answer": ans}, 0)
        steps.append(s); _emit(s)
        return AgentResult(question, ans, steps, max_iterations, False)

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
    # Private
    # ------------------------------------------------------------------ #

    _TERMINAL_TOOLS = {"list_forms", "execute_sql"}

    def _finish(self, question, answer, steps, iteration, emit_fn=None):
        s = AgentStep(iteration, "[auto] synthesized", "final_answer",
            {"answer": answer}, {"answer": answer}, 0)
        steps.append(s)
        if emit_fn: emit_fn(s)
        return AgentResult(question, answer, steps, iteration, True)

    def _synthesize(self, question, tool_name, tool_result):
        data = tool_result.get("result", {})
        prompt = (f"Answer this question based on the data.\n\n"
            f"Question: {question}\n\nData ({tool_name}):\n"
            f"{json.dumps(data, default=str, indent=2)}\n\n"
            f"Give a concise, direct answer. No JSON or tool references.")
        try: return str(self.reasoning_llm.complete(prompt)).strip()
        except: return json.dumps(data, default=str)

    def _context_msg(self, tool_name, tool_result, question=""):
        if not tool_result.get("success"):
            return f"Tool '{tool_name}' failed: {tool_result.get('error','unknown')}\nTry a different approach."

        if tool_name == "semantic_search":
            data = tool_result.get("result", [])
            if data:
                fmt = "\n".join(f"  - \"{m['text']}\" (form: {m['form_name']}, score: {m['score']})" for m in data[:5])
                needs_meta = bool(re.search(
                    r'\b(when|which day|what date|created|modified|timestamp|who created|who made)\b',
                    question.lower()))
                if needs_meta:
                    top = data[0]
                    return (f"semantic_search found {len(data)} match(es):\n{fmt}\n\n"
                        f"The user needs metadata semantic_search cannot provide.\n"
                        f"Question found in form '{top['form_name']}'.\n"
                        f"Call generate_sql with schema_hint mentioning form name "
                        f"'{top['form_name']}' and the metadata needed.\n"
                        f"Do NOT call final_answer until you have the metadata.")
                return (f"semantic_search found {len(data)} match(es):\n{fmt}\n\n"
                    f"If this answers the question, call final_answer.\n"
                    f"If user needs dates/counts/metadata, use generate_sql.")
            return "semantic_search found no matches. Try generate_sql."

        if tool_name == "lookup_form":
            names = tool_result.get("result", [])
            if names:
                return f"lookup_form found: {', '.join(repr(n) for n in names)}\nNow use generate_sql with one of these exact names."
            return "lookup_form found no matches. Try list_forms()."

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

    def _call_llm(self, messages):
        return self.reasoning_llm.chat(messages).message.content.strip()