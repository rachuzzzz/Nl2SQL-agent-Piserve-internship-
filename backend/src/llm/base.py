"""
Minimal LLM client interface.

Every call site in this codebase does `str(client.complete(prompt))` — this
Protocol documents that contract rather than inventing a richer interface
nothing currently needs. Any future provider (OpenAI, Anthropic, Azure) only
has to satisfy `.complete(prompt) -> Any` to be usable everywhere an Ollama
client is used today.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str) -> Any:
        ...


def extract_token_counts(raw) -> tuple[int, int]:
    """
    Returns (prompt_tokens, completion_tokens) from a completion/chat
    response's `.raw`. Provider shapes differ at TWO levels, not just one:

    1. `raw` itself: Ollama's llama_index integration sets raw to a genuine
       dict; Anthropic's explicitly does `dict(response)`; but the OpenAI
       integration (which vLLM's OpenAILike client subclasses) sets raw to
       the openai SDK's `ChatCompletion` object as-is — NOT a dict. Confirmed
       live: `raw.get("usage")` raised "'ChatCompletion' object has no
       attribute 'get'" for a vLLM response. So `usage` must be looked up via
       dict-or-attribute access on `raw`, not assumed to always be a dict.

    2. `usage` (once found): Anthropic's nests counts under input_tokens/
       output_tokens; vLLM's (OpenAI-style) nests them under prompt_tokens/
       completion_tokens instead — checked separately, not lumped together.
       Ollama's raw dict also has its own `usage` dict with prompt_tokens/
       completion_tokens keys (not attributes) as of recent ollama-python
       versions, but that's a plain dict, so neither hasattr check below
       matches it — falls through correctly to Ollama's real source of
       truth, the top-level prompt_eval_count/eval_count keys.
    """
    if raw is None:
        return 0, 0
    usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
    if usage is not None and hasattr(usage, "input_tokens"):
        return usage.input_tokens or 0, usage.output_tokens or 0
    if usage is not None and hasattr(usage, "prompt_tokens"):
        return usage.prompt_tokens or 0, usage.completion_tokens or 0
    if isinstance(raw, dict):
        return raw.get("prompt_eval_count", 0), raw.get("eval_count", 0)
    return 0, 0
