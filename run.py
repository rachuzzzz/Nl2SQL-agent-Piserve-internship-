"""
NL2SQL Agent — entry point
===========================
Run with: python run.py

Options:
  python run.py              → interactive agent mode (full loop)
  python run.py --sql-only   → dry-run: print tool calls but do not execute SQL
  python run.py --test       → test database connection only
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Live event handler — called by the orchestrator immediately on each event
# ---------------------------------------------------------------------------

def _on_event(event) -> None:
    """
    Called by AgentOrchestrator immediately on every event.
    Prints to stdout with flush=True so output appears in real time.

    event is either:
      {"_event": "thinking", "iteration": N}  — LLM call starting
      AgentStep                                — step completed
    """
    if isinstance(event, dict) and event.get("_event") == "thinking":
        forced = event.get("forced", False)
        label = "forced summary" if forced else "thinking..."
        print(f"\n[{event['iteration']}] ⟳ {label}", flush=True)
        return

    step = event  # it's an AgentStep

    if step.tool == "[parse_error]":
        print(f"      ⚠  Parse error — model returned non-JSON", flush=True)
        if step.result.get("error"):
            print(f"         {step.result['error']}", flush=True)
        return

    if step.tool == "final_answer":
        # Answer is printed after query() returns — skip here
        return

    if step.thought.startswith("[auto]") and step.tool == "execute_sql":
        # Auto-chained execute_sql — print full SQL for debugging
        sql_full = step.args.get("sql", "")
        result_summary = _format_result_summary("execute_sql", step.result)
        label = "auto"
        if "[review]" in step.thought:
            label = "auto(reviewed)"
        print(f"      {label}:  execute_sql", flush=True)
        # Print full SQL, indented
        for line in sql_full.strip().split("\n"):
            print(f"              {line}", flush=True)
        print(f"              → {result_summary}  [{step.duration_ms} ms]", flush=True)
        return

    # --- Normal step: thought + tool + result ---

    # Full thought, word-wrapped at 100 chars
    if step.thought:
        wrapped = _wrap(step.thought, width=100, indent="         ")
        print(f"      thought: {wrapped}", flush=True)

    args_str = _format_args(step.args)
    print(f"      tool:   {step.tool}({args_str})", flush=True)

    result_summary = _format_result_summary(step.tool, step.result)
    print(f"      result: {result_summary}  [{step.duration_ms} ms]", flush=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 100, indent: str = "") -> str:
    """
    Wrap text at word boundaries.  First line has no indent (caller provides
    the label prefix); subsequent lines get `indent`.
    """
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).lstrip()
    if current:
        lines.append(current)
    return ("\n" + indent).join(lines)


def _format_args(args: dict) -> str:
    """Compact one-line representation of tool args."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 72:
            v = v[:69] + "…"
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _format_result_summary(tool: str, result: dict) -> str:
    """Human-readable one-line summary of a tool result."""
    if not result.get("success"):
        err = result.get("error", "unknown")
        return f"ERROR: {err[:100]}"

    data = result.get("result")

    if tool == "list_forms":
        return f"{len(data)} forms returned"

    if tool == "lookup_form":
        return f"found: {', '.join(data[:5])}" if data else "no matches"

    if tool == "semantic_search":
        if data:
            top = data[0]
            return (
                f"{len(data)} matches — top: \"{top['text'][:50]}\" "
                f"({top['score']:.0%}, {top['form_name']})"
            )
        return "0 matches"

    if tool == "generate_sql":
        v = data.get("validation", {})
        status = "✓ valid" if v.get("passed") else "✗ invalid"
        real_errs = [e for e in v.get("errors", []) if not e.startswith("WARNING:")]
        suffix = f" — {real_errs[0][:70]}" if real_errs else ""
        return f"{status}{suffix}"

    if tool == "execute_sql":
        if data:
            return f"{data['row_count']} rows  cols={data['columns']}"
        return "no data"

    if tool == "get_schema":
        if "tables" in data:
            return f"{len(data['tables'])} tables"
        return f"{len(data.get('columns', []))} columns"

    return str(data)[:100] if data else "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sql_only = "--sql-only" in sys.argv
    test_only = "--test" in sys.argv

    if not os.path.exists(".env"):
        print("=" * 55)
        print("  .env file not found!")
        print("=" * 55)
        print()
        print("  1. Copy the template:  copy .env.template .env")
        print("  2. Edit .env with your database credentials:")
        print("     DB_HOST=your-server.example.com")
        print("     DB_PORT=5432")
        print("     DB_NAME=your_database")
        print("     DB_USER=your_username")
        print("     DB_PASSWORD=your_password")
        print()
        sys.exit(1)

    try:
        from agent.orchestrator import AgentOrchestrator
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)

    try:
        orch = AgentOrchestrator.from_env()
    except Exception as exc:
        print(f"\nFailed to initialise agent: {exc}")
        print("\nCheck your .env and make sure:")
        print("  - PostgreSQL server is reachable")
        print("  - Credentials are correct")
        print("  - Ollama is running (ollama serve)")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if test_only:
        orch.test_connection()
        sys.exit(0)

    mode = "sql-only (no execution)" if sql_only else "full agent"
    print(f"Mode: {mode}", flush=True)
    print("Type a question  ('quit' to exit)", flush=True)
    print("-" * 55, flush=True)

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        try:
            if sql_only:
                original_call = orch.registry.call

                def _no_execute(name, args, _orig=original_call):
                    if name == "execute_sql":
                        return {
                            "success": True,
                            "result": {"rows": [], "row_count": 0,
                                       "columns": [], "_note": "Skipped — sql-only mode"},
                            "error": None,
                        }
                    return _orig(name, args)

                orch.registry.call = _no_execute

            # Steps are printed live via _on_event as they happen
            result = orch.query(question, on_step=_on_event)

            if sql_only:
                orch.registry.call = original_call

            # Final answer — printed once after query() returns
            print(f"\n--- Answer ---", flush=True)
            print(result.answer, flush=True)

            if not result.success:
                print("\n⚠  Agent reached max iterations without a definitive answer.",
                      flush=True)

        except Exception as exc:
            print(f"\nError: {exc}", flush=True)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()