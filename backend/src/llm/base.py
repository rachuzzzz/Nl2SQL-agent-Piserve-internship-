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


def extract_token_counts(raw: dict | None) -> tuple[int, int]:
    """
    Returns (prompt_tokens, completion_tokens) from a completion/chat
    response's `.raw` dict. Ollama and Anthropic shapes are incompatible:
    Ollama's raw dict has flat prompt_eval_count/eval_count keys; Anthropic's
    nests counts under a `usage` object with input_tokens/output_tokens
    instead. Check for Anthropic's shape first since `raw` is always a dict
    either way — only Ollama's ever has prompt_eval_count as a top-level key.
    """
    usage = raw.get("usage") if raw else None
    if usage is not None and hasattr(usage, "input_tokens"):
        return usage.input_tokens or 0, usage.output_tokens or 0
    raw = raw or {}
    return raw.get("prompt_eval_count", 0), raw.get("eval_count", 0)
