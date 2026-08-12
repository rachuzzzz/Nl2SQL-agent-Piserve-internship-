"""
Regression test for loop.py's proactive context trim — fires BEFORE the
accumulated `messages` history grows large enough to overflow num_ctx
(unlike the pre-existing reactive trim, which only fires AFTER 2
consecutive EMPTY completions have already been wasted). A long-running,
many-iteration query must have its history bounded well before
max_iterations, not just react to a failure that already happened.
"""

import json
import logging

from src.agent.orchestrator.loop import AgentOrchestrator
from src.agent.orchestrator.session import ConversationSession


class _SchemaLoopRegistry:
    """A non-terminal tool (get_schema) that always "succeeds" with a
    sizeable canned result — used to drive the reasoning loop through many
    iterations without ever calling generate_sql/execute_sql (which would
    trip the unrelated refuse/clarify-after-3-failures path instead)."""

    def __init__(self):
        self.call_count = 0

    def call(self, tool_name, args):
        assert tool_name == "get_schema"
        self.call_count += 1
        return {"success": True, "result": {"tables": [f"table_{i}" for i in range(50)]}}


def _fake_orchestrator(registry) -> AgentOrchestrator:
    """Minimal AgentOrchestrator bypassing __init__ (same pattern used
    across this test package) — _call_llm always requests get_schema, a
    non-terminal tool, so the loop keeps iterating instead of returning
    after the first successful call."""
    fake = object.__new__(AgentOrchestrator)
    fake.domain = "inspection"
    fake._system_prompt = "SYSTEM PROMPT"
    fake._builder_module = type(
        "B", (), {"build_dynamic_prompt_additions": staticmethod(lambda q, score_fn=None: "")}
    )()
    fake.embed_model = None
    fake._scope_primary_table = "inspection_report"
    fake._scope_primary_alias = "ir"
    fake._debug = None
    fake.registry = registry
    # Reached only because this test's get_schema loop never terminates
    # early — it runs the full max_iterations budget and falls into the
    # (pre-existing, unrelated to this fix) forced-summary-at-max-iterations
    # path, which needs this.
    fake._domain_cfg = type("D", (), {"force_summary_prompt": "Summarize what you found so far."})()
    _raw = json.dumps({
        "thought": "checking schema again",
        "tool": "get_schema",
        "args": {},
    })
    fake._call_llm = lambda messages, stats, iteration=None: _raw
    return fake


def test_proactive_trim_fires_before_max_iterations(caplog):
    registry = _SchemaLoopRegistry()
    orch = _fake_orchestrator(registry)
    session = ConversationSession()

    with caplog.at_level(logging.WARNING, logger="inspectai.llm"):
        result = orch.query(
            "what columns does the inspection table have",
            session=session, visibility_scope={}, max_iterations=8,
        )

    assert result.total_iterations == 8  # ran the full budget — never terminated early
    assert registry.call_count == 8
    trim_logs = [r.message for r in caplog.records if "proactive context-trim" in r.message]
    assert trim_logs, "expected at least one proactive context-trim log line"


def test_proactive_trim_does_not_fire_on_short_queries(caplog):
    """A short (well under the iteration/char thresholds) query must never
    trigger the trim — it should be a no-op in the common case."""
    registry = _SchemaLoopRegistry()
    orch = _fake_orchestrator(registry)
    session = ConversationSession()

    with caplog.at_level(logging.WARNING, logger="inspectai.llm"):
        orch.query(
            "what columns does the inspection table have",
            session=session, visibility_scope={}, max_iterations=3,
        )

    trim_logs = [r.message for r in caplog.records if "proactive context-trim" in r.message]
    assert trim_logs == []
