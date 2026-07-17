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
