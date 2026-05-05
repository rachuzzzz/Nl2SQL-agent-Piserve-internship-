#!/usr/bin/env python3
"""
Interactive CLI for the NL2SQL agent.
"""

import sys
import textwrap
from agent.orchestrator import AgentOrchestrator, ConversationSession


def _format_result_summary(tool, result):
    """Compact one-line summary for the trace display."""
    if not result.get("success"):
        err = result.get("error", "unknown error")
        return f"ERROR: {err[:80]}"

    data = result.get("result", {})

    if tool == "list_forms":
        forms = data if isinstance(data, list) else data.get("forms", [])
        return f"{len(forms)} forms returned"

    if tool == "execute_sql":
        if data:
            trunc = ""
            if data.get("truncated"):
                total = data.get("total_count", "?")
                trunc = f" (of {total} total)"
            return f"{data['row_count']} rows{trunc}  cols={data['columns']}"
        return "no data"

    if tool == "generate_sql":
        sql_data = data
        valid = sql_data.get("validation", {}).get("passed", False)
        if valid:
            return f"✓ valid"
        errs = sql_data.get("validation", {}).get("errors", [])
        return f"✗ invalid — {'; '.join(errs)[:80]}"

    if tool in ("get_answer_summary", "get_score_stats"):
        return str(data)[:120]

    if tool == "resolve_answer_table":
        tbl = data.get("table_name", "?")
        mod = data.get("module_name", "?")
        return f"✓ table={tbl}  module='{mod}'"

    if tool == "lookup_form":
        name = data.get("form_name") or data.get("name", "?")
        return f"found: {name}"

    if tool == "semantic_search":
        matches = data.get("matches", [])
        return f"{len(matches)} matches"

    if tool == "get_schema":
        cols = data.get("columns", [])
        return f"{len(cols)} columns"

    if tool == "query_answers":
        sub_count = data.get("submission_count", 0)
        ans_count = data.get("total_answers", 0)
        return f"{sub_count} submissions, {ans_count} answers"

    return str(data)[:100]


def on_step(event):
    """Live trace callback — prints each step as it happens."""
    # Thinking events are plain dicts; AgentSteps are dataclass instances
    if isinstance(event, dict):
        if event.get("_event") == "thinking":
            it = event["iteration"]
            forced = " (forced)" if event.get("forced") else ""
            print(f"[{it}] ⟳ thinking...{forced}")
        return

    # It's an AgentStep (dataclass, not dict) — use attribute access
    step = event
    tool = step.tool
    args = step.args
    thought = step.thought
    result = step.result
    duration = step.duration_ms

    # Thought
    if thought and thought not in ("[parse error]", "[auto] synthesized"):
        # Wrap long thoughts
        lines = textwrap.wrap(thought, width=95)
        print(f"      thought: {lines[0]}")
        for line in lines[1:]:
            print(f"         {line}")

    # Tool call
    if tool == "final_answer":
        return
    if tool == "execute_sql" and "[auto]" in thought:
        sql = args.get("sql", "")
        summary = _format_result_summary(tool, result)
        # Show SQL indented
        print(f"      auto:  execute_sql")
        for line in sql.strip().split("\n"):
            print(f"              {line}")
        print(f"              → {summary}  [{duration} ms]")
    else:
        # Format tool(arg=val, ...)
        arg_parts = []
        for k, v in args.items():
            sv = str(v)
            if len(sv) > 60:
                sv = sv[:57] + "…"
            arg_parts.append(f"{k}={sv!r}")
        arg_str = ", ".join(arg_parts)
        summary = _format_result_summary(tool, result)
        print(f"      tool:   {tool}({arg_str})")
        print(f"      result: {summary}  [{duration} ms]")


def print_stats(stats):
    """Print query performance stats."""
    parts = []
    if stats.total_wall_ms:
        parts.append(f"⏱ {stats.total_wall_ms}ms total")
    if stats.total_llm_calls:
        parts.append(f"{stats.total_llm_calls} LLM calls")
    if stats.total_prompt_tokens:
        parts.append(f"{stats.total_prompt_tokens} prompt tokens")
    if stats.total_completion_tokens:
        parts.append(f"{stats.total_completion_tokens} completion tokens")
    if stats.total_sql_exec_ms:
        parts.append(f"{stats.total_sql_exec_ms}ms SQL")
    if stats.total_llm_latency_ms:
        parts.append(f"{stats.total_llm_latency_ms}ms LLM")
    if parts:
        print(f"  [{' | '.join(parts)}]")


def main():
    orch = AgentOrchestrator.from_env()
    session = ConversationSession()

    print("Mode: full agent")
    print("Type a question  ('quit' to exit)")
    print("-" * 55)

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = orch.query(q, on_step=on_step, session=session)

        print(f"\n--- Answer ---")
        print(result.answer)

        # Show stats
        print_stats(result.stats)

        if not result.success:
            print("⚠  Agent reached max iterations without a definitive answer.")


if __name__ == "__main__":
    main()