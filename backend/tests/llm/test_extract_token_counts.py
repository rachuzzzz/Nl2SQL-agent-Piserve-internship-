"""
extract_token_counts (src/llm/base.py) handles three incompatible response
shapes: Ollama's flat prompt_eval_count/eval_count dict, Anthropic's nested
usage.input_tokens/output_tokens object, and vLLM's (OpenAI-compatible)
nested usage.prompt_tokens/completion_tokens object. Regression coverage for
the provider-blind bug where the Ollama-only lookup silently returned 0/0 for
every Claude call (see synthesis.py's _call_llm history) — the same bug class
vLLM's distinct usage-attribute names would otherwise reintroduce.

Also covers cache_read_tokens/cache_creation_tokens (Anthropic-only —
cache_read_input_tokens/cache_creation_input_tokens on its Usage object,
present only once prompt caching is enabled on a request; 0 everywhere else,
including on Anthropic responses from a request that didn't use caching).
"""

from types import SimpleNamespace

from src.llm.base import extract_token_counts, TokenUsage


def test_ollama_flat_shape():
    raw = {"prompt_eval_count": 123, "eval_count": 45}
    assert extract_token_counts(raw) == TokenUsage(prompt_tokens=123, completion_tokens=45)


def test_anthropic_nested_usage_shape():
    raw = {"usage": SimpleNamespace(input_tokens=19, output_tokens=4)}
    assert extract_token_counts(raw) == TokenUsage(prompt_tokens=19, completion_tokens=4)


def test_anthropic_usage_with_cache_fields():
    """Once prompt caching is enabled on a request, Anthropic's Usage object
    also carries cache_read_input_tokens/cache_creation_input_tokens — these
    must be surfaced, not silently dropped like before this field existed."""
    raw = {
        "usage": SimpleNamespace(
            input_tokens=50, output_tokens=12,
            cache_read_input_tokens=800, cache_creation_input_tokens=0,
        ),
    }
    assert extract_token_counts(raw) == TokenUsage(
        prompt_tokens=50, completion_tokens=12,
        cache_read_tokens=800, cache_creation_tokens=0,
    )


def test_anthropic_usage_without_cache_fields_degrades_to_zero():
    """Older anthropic SDK versions (or a non-caching response) may not set
    cache_read_input_tokens/cache_creation_input_tokens on the Usage object
    at all — getattr's default must apply, not raise."""
    raw = {"usage": SimpleNamespace(input_tokens=50, output_tokens=12)}
    assert extract_token_counts(raw) == TokenUsage(
        prompt_tokens=50, completion_tokens=12,
        cache_read_tokens=0, cache_creation_tokens=0,
    )


def test_none_input_degrades_to_zero():
    assert extract_token_counts(None) == TokenUsage()


def test_empty_dict_degrades_to_zero():
    assert extract_token_counts({}) == TokenUsage()


def test_usage_present_but_not_anthropic_shaped_falls_back_to_ollama_keys():
    """A `usage` key that isn't Anthropic's object (e.g. some other dict)
    must not be mistaken for the Anthropic shape — only presence of the
    input_tokens attribute qualifies."""
    raw = {"usage": {"something": "else"}, "prompt_eval_count": 7, "eval_count": 2}
    assert extract_token_counts(raw) == TokenUsage(prompt_tokens=7, completion_tokens=2)


def test_vllm_openai_style_usage_shape():
    """vLLM's OpenAI-compatible server nests counts under `usage` too, but
    with prompt_tokens/completion_tokens — OpenAI's naming, not Anthropic's
    input_tokens/output_tokens. Must be distinguished from both other
    shapes, not silently fall through to the Ollama flat-key fallback. No
    cache concept exists on this shape — cache fields stay 0."""
    raw = {"usage": SimpleNamespace(prompt_tokens=31, completion_tokens=8)}
    assert extract_token_counts(raw) == TokenUsage(prompt_tokens=31, completion_tokens=8)


def test_vllm_raw_is_an_object_not_a_dict():
    """Regression: unlike Ollama (genuine dict) and Anthropic (explicitly
    dict()-converted by llama_index), the OpenAI integration underlying
    vLLM's OpenAILike client sets `.raw` to the openai SDK's ChatCompletion
    object as-is. `raw.get("usage")` crashes on it with "'ChatCompletion'
    object has no attribute 'get'" — confirmed live. `raw` itself must be
    read via attribute access, not assumed to always be dict-like."""
    raw = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=14, completion_tokens=2))
    assert extract_token_counts(raw) == TokenUsage(prompt_tokens=14, completion_tokens=2)


def test_object_raw_without_usage_degrades_to_zero():
    raw = SimpleNamespace(usage=None)
    assert extract_token_counts(raw) == TokenUsage()
