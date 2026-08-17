"""
extract_token_counts (src/llm/base.py) handles three incompatible response
shapes: Ollama's flat prompt_eval_count/eval_count dict, Anthropic's nested
usage.input_tokens/output_tokens object, and vLLM's (OpenAI-compatible)
nested usage.prompt_tokens/completion_tokens object. Regression coverage for
the provider-blind bug where the Ollama-only lookup silently returned 0/0 for
every Claude call (see synthesis.py's _call_llm history) — the same bug class
vLLM's distinct usage-attribute names would otherwise reintroduce.
"""

from types import SimpleNamespace

from src.llm.base import extract_token_counts


def test_ollama_flat_shape():
    raw = {"prompt_eval_count": 123, "eval_count": 45}
    assert extract_token_counts(raw) == (123, 45)


def test_anthropic_nested_usage_shape():
    raw = {"usage": SimpleNamespace(input_tokens=19, output_tokens=4)}
    assert extract_token_counts(raw) == (19, 4)


def test_none_input_degrades_to_zero():
    assert extract_token_counts(None) == (0, 0)


def test_empty_dict_degrades_to_zero():
    assert extract_token_counts({}) == (0, 0)


def test_usage_present_but_not_anthropic_shaped_falls_back_to_ollama_keys():
    """A `usage` key that isn't Anthropic's object (e.g. some other dict)
    must not be mistaken for the Anthropic shape — only presence of the
    input_tokens attribute qualifies."""
    raw = {"usage": {"something": "else"}, "prompt_eval_count": 7, "eval_count": 2}
    assert extract_token_counts(raw) == (7, 2)


def test_vllm_openai_style_usage_shape():
    """vLLM's OpenAI-compatible server nests counts under `usage` too, but
    with prompt_tokens/completion_tokens — OpenAI's naming, not Anthropic's
    input_tokens/output_tokens. Must be distinguished from both other
    shapes, not silently fall through to the Ollama flat-key fallback."""
    raw = {"usage": SimpleNamespace(prompt_tokens=31, completion_tokens=8)}
    assert extract_token_counts(raw) == (31, 8)


def test_vllm_raw_is_an_object_not_a_dict():
    """Regression: unlike Ollama (genuine dict) and Anthropic (explicitly
    dict()-converted by llama_index), the OpenAI integration underlying
    vLLM's OpenAILike client sets `.raw` to the openai SDK's ChatCompletion
    object as-is. `raw.get("usage")` crashes on it with "'ChatCompletion'
    object has no attribute 'get'" — confirmed live. `raw` itself must be
    read via attribute access, not assumed to always be dict-like."""
    raw = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=14, completion_tokens=2))
    assert extract_token_counts(raw) == (14, 2)


def test_object_raw_without_usage_degrades_to_zero():
    raw = SimpleNamespace(usage=None)
    assert extract_token_counts(raw) == (0, 0)
