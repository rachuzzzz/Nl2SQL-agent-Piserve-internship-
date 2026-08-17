"""
vLLM client construction — the only module in this codebase that imports
llama_index.llms.openai_like directly. Everything else goes through
llm/factory.py, mirroring ollama.py's/anthropic.py's shape so a vLLM client
satisfies the same LLMClient protocol (llm/base.py).

vLLM's server exposes an OpenAI-compatible REST API (not Ollama's native
/api/generate protocol and not Anthropic's Messages API), so this uses the
generic OpenAILike wrapper with api_base pointed at the vLLM server, rather
than a vLLM-specific integration (llama-index-llms-vllm is for embedding a
vLLM engine in-process/offline, not talking to a remote hosted server).

Unlike ollama.py/anthropic.py, there is no dedicated `thinking` param here —
vLLM's OpenAI-compatible endpoint has no equivalent to Ollama's `think`
field or Anthropic's `thinking_dict`. A served model with its own thinking
mode (confirmed live: qwen3.5-9b-awq) is controlled per-call instead via
additional_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking":
False}}} — see llm/factory.py's get_reasoning_llm_vllm docstring for why
extra_body specifically (not a plain top-level kwarg) is required to reach
the server.
"""

from __future__ import annotations

from llama_index.llms.openai_like import OpenAILike


def build_vllm_client(
    model: str,
    base_url: str,
    *,
    api_key: str = "dummy",
    temperature: float | None = None,
    max_tokens: int | None = None,
    context_window: int | None = None,
    additional_kwargs: dict | None = None,
    is_chat_model: bool = True,
) -> OpenAILike:
    """
    base_url is the vLLM server's OpenAI-compatible base, e.g.
    "http://host:8000/v1" (passed through as OpenAILike's `api_base`).
    api_key defaults to a dummy value — most self-hosted vLLM servers accept
    any key unless auth is explicitly configured on the server.
    """
    kwargs: dict = {
        "model": model,
        "api_base": base_url,
        "api_key": api_key,
        "is_chat_model": is_chat_model,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if context_window is not None:
        kwargs["context_window"] = context_window
    if additional_kwargs:
        kwargs["additional_kwargs"] = additional_kwargs
    return OpenAILike(**kwargs)
