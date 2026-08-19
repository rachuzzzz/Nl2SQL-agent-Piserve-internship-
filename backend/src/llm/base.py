"""
Minimal LLM client interface.

Every call site in this codebase does `str(client.complete(prompt))` — this
Protocol documents that contract rather than inventing a richer interface
nothing currently needs. Any future provider (OpenAI, Anthropic, Azure) only
has to satisfy `.complete(prompt) -> Any` to be usable everywhere an Ollama
client is used today.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str) -> Any:
        ...


@dataclass
class LlamaCppTimings:
    """
    llama.cpp's own per-call prefill/decode split, read from the `timings`
    object llama-server attaches to every /v1/chat/completions response
    (confirmed live: survives through the openai SDK's ChatCompletion model
    as an extra field, and through llama_index's OpenAILike wrapper's
    response.raw, unmodified — see Phase 3.2 profiling). Zero-valued (not
    None) on any provider/response without it (Ollama, Anthropic, vLLM
    proper, or an error response) so callers can sum unconditionally.

    prompt_ms:    prefill — time processing the input context. Scales with
                  prompt SIZE (roughly, minus whatever `cache_n` covers).
    predicted_ms: decode — time generating output tokens. Scales with
                  OUTPUT LENGTH and raw per-token model speed.
    cache_n:      how many of this call's prompt tokens llama.cpp's own
                  slot cache served from a prior call's KV cache instead of
                  recomputing — i.e. prefill work this call skipped. Distinct
                  from TokenUsage.cache_read_tokens (Anthropic-only).
    """
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0
    cache_n: int = 0


def extract_llamacpp_timings(raw) -> LlamaCppTimings:
    """See LlamaCppTimings' docstring. `raw` is a response's `.raw` value —
    same input extract_token_counts takes."""
    if raw is None:
        return LlamaCppTimings()
    t = raw.get("timings") if isinstance(raw, dict) else getattr(raw, "timings", None)
    if not t:
        return LlamaCppTimings()
    _get = t.get if isinstance(t, dict) else (lambda k, d=None: getattr(t, k, d))
    return LlamaCppTimings(
        prompt_ms=_get("prompt_ms", 0.0) or 0.0,
        predicted_ms=_get("predicted_ms", 0.0) or 0.0,
        cache_n=_get("cache_n", 0) or 0,
    )


@dataclass
class TokenUsage:
    """
    Every field this codebase can currently observe about one LLM call's
    token spend. cache_read_tokens/cache_creation_tokens are Anthropic-only
    today (its Usage object carries cache_read_input_tokens/
    cache_creation_input_tokens once prompt caching is enabled on a
    request) — 0 on Ollama/vLLM, which have no such concept, not None, so
    every caller can sum/log these fields unconditionally without a
    provider check of its own.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


def extract_token_counts(raw) -> TokenUsage:
    """
    Returns a TokenUsage from a completion/chat response's `.raw`. Provider
    shapes differ at TWO levels, not just one:

    1. `raw` itself: Ollama's llama_index integration sets raw to a genuine
       dict; Anthropic's explicitly does `dict(response)`; but the OpenAI
       integration (which vLLM's OpenAILike client subclasses) sets raw to
       the openai SDK's `ChatCompletion` object as-is — NOT a dict. Confirmed
       live: `raw.get("usage")` raised "'ChatCompletion' object has no
       attribute 'get'" for a vLLM response. So `usage` must be looked up via
       dict-or-attribute access on `raw`, not assumed to always be a dict.

    2. `usage` (once found): Anthropic's nests counts under input_tokens/
       output_tokens (plus cache_read_input_tokens/cache_creation_input_tokens
       when caching is enabled on the request — absent/0 otherwise, read via
       getattr so older anthropic SDK versions without the fields still
       degrade to 0 rather than raising); vLLM's (OpenAI-style) nests them
       under prompt_tokens/completion_tokens instead, with no cache concept
       at all — checked separately, not lumped together. Ollama's raw dict
       also has its own `usage` dict with prompt_tokens/completion_tokens
       keys (not attributes) as of recent ollama-python versions, but that's
       a plain dict, so neither hasattr check below matches it — falls
       through correctly to Ollama's real source of truth, the top-level
       prompt_eval_count/eval_count keys (also with no cache concept).
    """
    if raw is None:
        return TokenUsage()
    usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
    if usage is not None and hasattr(usage, "input_tokens"):
        return TokenUsage(
            prompt_tokens=usage.input_tokens or 0,
            completion_tokens=usage.output_tokens or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
    if usage is not None and hasattr(usage, "prompt_tokens"):
        return TokenUsage(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
        )
    if isinstance(raw, dict):
        return TokenUsage(
            prompt_tokens=raw.get("prompt_eval_count", 0),
            completion_tokens=raw.get("eval_count", 0),
        )
    return TokenUsage()
