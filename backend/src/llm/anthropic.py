"""
Anthropic (Claude) client construction — the only module in this codebase
that imports llama_index.llms.anthropic directly. Everything else goes
through llm/factory.py, mirroring ollama.py's shape so both providers satisfy
the same LLMClient protocol (llm/base.py) — a Claude client can drop into
any call site that currently does `str(client.complete(prompt))`.
"""

from __future__ import annotations

from llama_index.llms.anthropic import Anthropic


def build_anthropic_client(
    model: str,
    api_key: str,
    *,
    temperature: float | None = None,
    max_tokens: int = 4096,
    additional_kwargs: dict | None = None,
    thinking: bool | None = None,
) -> Anthropic:
    """
    thinking=False disables extended/adaptive thinking (llama_index's
    `thinking_dict` constructor field, mapped to the API's `thinking:
    {"type": "disabled"}`) — deliberately mirrors ollama.py's `thinking`
    param and the exact bug it guards against: adaptive thinking is on by
    default on Claude Sonnet 5 just like it was on Qwen, and `max_tokens` is
    a hard cap on thinking + response combined. A short, tight max_tokens
    tuned for a compact tool-call JSON or single SQL statement (no reasoning
    prose expected) can otherwise be entirely consumed by thinking, leaving
    nothing for the actual answer — the same failure this codebase already
    hit and fixed for Ollama's qwen3.5:9b (see ollama.py's docstring).
    """
    kwargs: dict = {"model": model, "api_key": api_key, "max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if additional_kwargs:
        kwargs["additional_kwargs"] = additional_kwargs
    if thinking is False:
        kwargs["thinking_dict"] = {"type": "disabled"}
    return Anthropic(**kwargs)
