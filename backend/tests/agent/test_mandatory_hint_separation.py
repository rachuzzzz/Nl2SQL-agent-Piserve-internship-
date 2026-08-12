"""
Regression test for loop.py's _mandatory_hint separation — Item 2 of the
2026-08-07 generation-quality plan (qid=e63f03af). INJECT-rule hints (and
the LIMIT-N hint) must land in args["_mandatory_hint"], not get
concatenated onto args["schema_hint"] — otherwise generation.py's "Planner
draft — editable, may be wrong" framing would apply to genuinely mandatory
content too.
"""

import json

from src.agent.orchestrator.loop import AgentOrchestrator
from src.agent.orchestrator.session import ConversationSession

from tests.agent.test_refuse_or_clarify_after_failures import _gen_sql_validation_fail


class _RecordingRegistry:
    """Like test_refuse_or_clarify_after_failures.py's _FakeRegistry, but
    also records every generate_sql call's args — needed here to inspect
    _mandatory_hint vs. schema_hint, which _FakeRegistry doesn't expose.

    Deliberately always fails validation (see the test below) rather than
    succeeding — a successful generate_sql call falls through to loop.py's
    auto-execute-then-synthesize path, which needs far more orchestrator
    fixture setup (_domain_cfg, SynthesisMixin deps) than this minimal fake
    provides. The already-proven-safe refuse-after-3-failures path (see
    test_refuse_or_clarify_after_failures.py) needs none of that, and is
    all this test needs to inspect the FIRST call's recorded args."""

    def __init__(self, gen_sql_results):
        self._gen_sql_results = list(gen_sql_results)
        self.generate_sql_calls: list[dict] = []

    def call(self, tool_name, args):
        if tool_name == "generate_sql":
            self.generate_sql_calls.append(args)
            return (self._gen_sql_results.pop(0)
                    if len(self._gen_sql_results) > 1 else self._gen_sql_results[0])
        raise AssertionError(f"unexpected tool call in test: {tool_name}")


class _FakeRulesEngine:
    """Deterministic stand-in for RulesEngine — always fires with a fixed,
    recognizable hint, regardless of question text."""

    HINT = "USE THIS EXACT SQL FOR CAPEX/OPEX (do not deviate):\nSELECT 1;"

    def hint_for_question(self, question):
        return self.HINT

    def matching_inject_ids(self, question):
        return ["fake_capex_rule"]


def _fake_orchestrator(registry: _RecordingRegistry) -> AgentOrchestrator:
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
    fake._rules_engine = _FakeRulesEngine()
    fake.registry = registry
    _raw = json.dumps({
        "thought": "generating SQL",
        "tool": "generate_sql",
        "args": {"question": "placeholder", "schema_hint": "my own draft SELECT 1"},
    })
    fake._call_llm = lambda messages, stats, iteration=None: _raw
    return fake


def test_inject_hint_goes_to_mandatory_hint_not_schema_hint():
    registry = _RecordingRegistry(gen_sql_results=[
        _gen_sql_validation_fail("UNDEFINED_ALIAS"),
        _gen_sql_validation_fail("UNDEFINED_ALIAS"),
        _gen_sql_validation_fail("UNDEFINED_ALIAS"),
    ])
    orch = _fake_orchestrator(registry)
    session = ConversationSession()
    orch.query("capex and opex breakdown for mitigative actions", session=session, visibility_scope={})

    assert registry.generate_sql_calls, "generate_sql was never called"
    recorded_args = registry.generate_sql_calls[0]
    assert recorded_args.get("_mandatory_hint", "") == _FakeRulesEngine.HINT
    assert recorded_args.get("schema_hint") == "my own draft SELECT 1"
    assert "USE THIS EXACT SQL" not in recorded_args.get("schema_hint", "")
