"""
generate_sql() (src/sql/generation.py) can make 1-4 real LLM calls per
invocation: the initial completion plus up to three guard retries
(offset-guard, table-guard, uuid-guard). Before this, none of those calls
fed token counts back to the caller at all — QueryStats (loop.py) only ever
counted the reasoning role's tokens, so tokens=X/Y in [QUERY] END
undercounted real spend on every query that generated SQL. These tests
cover the accounting: single-call totals for both provider response shapes,
and summing across a forced guard retry — including cache_read_tokens/
cache_creation_tokens (Anthropic-only; 0 on Ollama-shaped responses), which
must sum across multiple calls the same way prompt/completion tokens do.
"""

from types import SimpleNamespace

from src.sql.generation import generate_sql
from src.sql.validation import SQLValidator


class _FakeResponse:
    def __init__(self, text, raw):
        self._text = text
        self.raw = raw

    def __str__(self):
        return self._text


class _FakeSqlLlm:
    def __init__(self, responses):
        self.model = "fake-model"
        self._responses = list(responses)
        self.calls = 0

    def complete(self, prompt):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _validator():
    return SQLValidator(domain="audit")


def test_single_call_anthropic_shape_reports_token_usage():
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT ap.title FROM audit_plan ap WHERE ap.active = true;",
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=20)},
        ),
    ])
    result = generate_sql(sql_llm, _validator(), question="list audit plans")
    assert result["success"] is True
    usage = result["result"]["_token_usage"]
    assert usage == {
        "prompt_tokens": 100, "completion_tokens": 20,
        "cache_read_tokens": 0, "cache_creation_tokens": 0,
    }
    assert sql_llm.calls == 1


def test_single_call_anthropic_shape_with_cache_tokens():
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT ap.title FROM audit_plan ap WHERE ap.active = true;",
            {"usage": SimpleNamespace(
                input_tokens=100, output_tokens=20,
                cache_read_input_tokens=3000, cache_creation_input_tokens=0,
            )},
        ),
    ])
    result = generate_sql(sql_llm, _validator(), question="list audit plans")
    assert result["success"] is True
    usage = result["result"]["_token_usage"]
    assert usage == {
        "prompt_tokens": 100, "completion_tokens": 20,
        "cache_read_tokens": 3000, "cache_creation_tokens": 0,
    }


def test_single_call_ollama_shape_reports_token_usage():
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT ap.title FROM audit_plan ap WHERE ap.active = true;",
            {"prompt_eval_count": 250, "eval_count": 40},
        ),
    ])
    result = generate_sql(sql_llm, _validator(), question="list audit plans")
    assert result["success"] is True
    usage = result["result"]["_token_usage"]
    assert usage == {
        "prompt_tokens": 250, "completion_tokens": 40,
        "cache_read_tokens": 0, "cache_creation_tokens": 0,
    }
    assert sql_llm.calls == 1


def test_offset_guard_retry_sums_tokens_across_both_calls():
    """schema_hint carries a LIMIT 1 OFFSET clause (the ordinal-resolution
    fallback) but the model's first completion drops it — forces the
    offset-guard retry (generation.py ~line 322). _token_usage must be the
    SUM across both real API calls, not just the retry's count — including
    cache_read_tokens, which must sum the same way prompt/completion do."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT ac.status FROM audit_cycle ac;",  # no OFFSET -> triggers retry
            {"usage": SimpleNamespace(
                input_tokens=100, output_tokens=10,
                cache_read_input_tokens=500, cache_creation_input_tokens=0,
            )},
        ),
        _FakeResponse(
            "SELECT ac.status FROM audit_cycle ac LIMIT 1 OFFSET 3;",
            {"usage": SimpleNamespace(
                input_tokens=150, output_tokens=15,
                cache_read_input_tokens=700, cache_creation_input_tokens=0,
            )},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="the 4th cycle's status",
        schema_hint="SELECT ac.id FROM audit_cycle ac ORDER BY ac.start_date LIMIT 1 OFFSET 3",
    )
    assert sql_llm.calls == 2
    usage = result["result"]["_token_usage"]
    assert usage == {
        "prompt_tokens": 250, "completion_tokens": 25,
        "cache_read_tokens": 1200, "cache_creation_tokens": 0,
    }


def test_last_attempt_skips_offset_guard_retry():
    """_last_attempt=True means loop.py already knows this call, if it also
    fails, trips the outer 3-attempt cap — so the offset-guard's own retry
    (which would otherwise fire, per test_offset_guard_retry_sums_tokens_
    across_both_calls above using the identical schema_hint/question) must
    be skipped: only the initial sql_llm.complete() call happens, not a
    second one."""
    sql_llm = _FakeSqlLlm([
        _FakeResponse(
            "SELECT ac.status FROM audit_cycle ac;",  # no OFFSET -> would trigger retry
            {"usage": SimpleNamespace(input_tokens=100, output_tokens=10)},
        ),
        _FakeResponse(
            "SELECT ac.status FROM audit_cycle ac LIMIT 1 OFFSET 3;",
            {"usage": SimpleNamespace(input_tokens=150, output_tokens=15)},
        ),
    ])
    result = generate_sql(
        sql_llm, _validator(), question="the 4th cycle's status",
        schema_hint="SELECT ac.id FROM audit_cycle ac ORDER BY ac.start_date LIMIT 1 OFFSET 3",
        _last_attempt=True,
    )
    assert sql_llm.calls == 1
    usage = result["result"]["_token_usage"]
    assert usage == {
        "prompt_tokens": 100, "completion_tokens": 10,
        "cache_read_tokens": 0, "cache_creation_tokens": 0,
    }


def test_exception_path_still_reports_zero_usage_not_a_crash():
    """A sql_llm whose .complete() raises must still hit the except branch's
    _token_usage key (defends against the locals()-guard regressing to a
    KeyError for callers that assume the key always exists)."""
    class _BoomLlm:
        model = "fake-model"

        def complete(self, prompt):
            raise RuntimeError("boom")

    result = generate_sql(_BoomLlm(), _validator(), question="anything")
    assert result["success"] is False
    assert result["result"]["_token_usage"] == {
        "prompt_tokens": 0, "completion_tokens": 0,
        "cache_read_tokens": 0, "cache_creation_tokens": 0,
    }
