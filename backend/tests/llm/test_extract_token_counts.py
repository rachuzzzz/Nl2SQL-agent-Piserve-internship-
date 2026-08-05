"""
extract_token_counts (src/llm/base.py) handles two incompatible response
shapes: Ollama's flat prompt_eval_count/eval_count dict vs Anthropic's
nested usage.input_tokens/output_tokens object. Regression coverage for the
provider-blind bug where the Ollama-only lookup silently returned 0/0 for
every Claude call (see synthesis.py's _call_llm history).
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
