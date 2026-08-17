"""
Ollama client construction — the only module in this codebase that imports
llama_index.llms.ollama directly. Everything else goes through llm/factory.py.
"""

from __future__ import annotations

from llama_index.llms.ollama import Ollama


def build_ollama_client(
    model: str,
    base_url: str,
    *,
    temperature: float | None = None,
    request_timeout: float = 120.0,
    additional_kwargs: dict | None = None,
    thinking: bool | None = None,
    json_mode: bool = False,
    keep_alive: str | None = None,
) -> Ollama:
    """
    thinking: llama_index's Ollama integration reads this as a genuine
    top-level constructor field (Ollama.chat()'s `think = kwargs.pop("think",
    None) or self.thinking`) — it is NOT part of `additional_kwargs`, which
    only ever becomes the Ollama API's `options` dict. A caller that put
    "think": False inside additional_kwargs (as this codebase did until this
    was found) silently does nothing — Ollama ignores an unrecognized
    options.think field and the model's own default thinking behavior wins.
    Confirmed live: qwen3.5:9b burned its entire num_predict budget on
    `message.thinking` with zero tokens left for the actual answer content
    (done_reason='length', content=''), for a prompt that additional_kwargs'
    "think": False was supposed to prevent.

    json_mode: another genuine top-level constructor field (base.py's
    `format = kwargs.pop("format", "json" if self.json_mode else None)`) —
    same "not part of additional_kwargs/options" shape as thinking above.
    Forces Ollama's grammar-constrained JSON decoding, so the model is
    incapable of emitting free-form prose instead of the tool-call JSON —
    a stronger guarantee than the system prompt's "respond ONLY with JSON"
    instruction alone. Confirmed live 2026-08-13: ibm/granite4.1:8b (unlike
    qwen3, which already complied via the text instruction) wrote a full
    prose reasoning essay straight into `content` on iter 1 of a capex
    query, failing parse_tool_call and burning an ~11s retry before
    succeeding on iter 2 — json_mode prevents that failure mode outright
    rather than just recovering from it after the fact.

    keep_alive: how long Ollama keeps the model (and its loaded slots' KV
    cache) resident after the last request before unloading it. Ollama's
    server default is 5 minutes; a quiet period longer than that silently
    evicts every cached prefix, independent of anything PrefixCacheManager
    does (see llm/prefix_cache.py). Passed straight through as a genuine
    top-level constructor field, same shape as thinking/json_mode above.
    """
    kwargs: dict = {"model": model, "base_url": base_url, "request_timeout": request_timeout}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if additional_kwargs:
        # Strip any accidental "think"/"thinking" from additional_kwargs —
        # it belongs in the dedicated `thinking` param above, not `options`.
        additional_kwargs = {k: v for k, v in additional_kwargs.items() if k not in ("think", "thinking")}
        if additional_kwargs:
            kwargs["additional_kwargs"] = additional_kwargs
    if thinking is not None:
        kwargs["thinking"] = thinking
    if json_mode:
        kwargs["json_mode"] = json_mode
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    return Ollama(**kwargs)
