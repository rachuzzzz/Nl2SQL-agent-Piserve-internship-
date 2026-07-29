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
    return Ollama(**kwargs)
