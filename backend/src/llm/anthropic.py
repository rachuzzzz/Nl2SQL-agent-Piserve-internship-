"""
Anthropic (Claude) client construction — the only module in this codebase
that imports llama_index.llms.anthropic directly. Everything else goes
through llm/factory.py, mirroring ollama.py's shape so both providers satisfy
the same LLMClient protocol (llm/base.py) — a Claude client can drop into
any call site that currently does `str(client.complete(prompt))`.
"""

from __future__ import annotations

from llama_index.core.llms import (
    CacheControl,
    CachePoint,
    ChatMessage,
    MessageRole,
    TextBlock,
)
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


def build_cached_system_message(llm_client, static_prefix: str, dynamic_tail: str) -> ChatMessage:
    """
    Build the SYSTEM ChatMessage the reasoning loop sends every call
    (loop.py) with an Anthropic prompt-cache breakpoint placed after
    static_prefix — the part of the system prompt that's byte-identical
    across every turn for a given domain (self._system_prompt in loop.py) —
    and before dynamic_tail (per-question/per-turn content: conditional
    prompt blocks, the current user, the security scope filter, session
    history). No-op passthrough (today's exact single flat-text message) for
    every non-Anthropic client — Ollama's and vLLM's llama_index
    integrations have no concept of a CachePoint content block and must
    never receive one.

    Deliberately bypasses llama_index's own cache_idx mechanism rather than
    threading a cache_idx kwarg through build_anthropic_client(): cache_idx
    is gated by is_anthropic_prompt_caching_supported_model(), and that
    model allowlist (llama-index-llms-anthropic 0.11.9, the version this
    project pins >=0.6.0 to) does not include claude-sonnet-5 or
    claude-sonnet-4-6 — so cache_idx would silently do nothing on this
    deployment's configured ANTHROPIC_MODEL, with no error. Setting
    cache_control via an explicit CachePoint content block instead bypasses
    that allowlist check entirely: llama_index's anthropic/utils.py reads
    cache_control off the message/blocks unconditionally, with no
    model-support gate at that point. A single CachePoint (not cache_idx=-1
    or one per message) also keeps this comfortably under Anthropic's
    4-breakpoint-per-request cap regardless of how long the conversation
    history in `messages` grows.
    """
    if not isinstance(llm_client, Anthropic):
        return ChatMessage(role=MessageRole.SYSTEM, content=static_prefix + dynamic_tail)
    return ChatMessage(
        role=MessageRole.SYSTEM,
        blocks=[
            TextBlock(text=static_prefix),
            CachePoint(cache_control=CacheControl(type="ephemeral")),
            TextBlock(text=dynamic_tail),
        ],
    )
