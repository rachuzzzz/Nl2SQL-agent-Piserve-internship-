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
) -> Ollama:
    kwargs: dict = {"model": model, "base_url": base_url, "request_timeout": request_timeout}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if additional_kwargs:
        kwargs["additional_kwargs"] = additional_kwargs
    return Ollama(**kwargs)
