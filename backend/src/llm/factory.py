"""
Role-based LLM client construction.

Each role's tuning (context window, stop tokens, timeout, temperature) is
load-bearing — see inline comments — so this factory keeps per-role presets
rather than exposing one generic client. Consumers (agent/orchestrator) ask
for a role, not a provider; swapping providers means changing the functions
below, not every call site. Provider selection is independent per role (see
get_llm_client below) — e.g. reasoning via vLLM and sql via Ollama at the
same time is a supported combination, not a special case.
"""

from __future__ import annotations

from src.llm.ollama import build_ollama_client
from src.llm.anthropic import build_anthropic_client
from src.llm.vllm import build_vllm_client


def get_reasoning_llm(model: str, base_url: str):
    """
    Reasoning/planning LLM. num_ctx=20000 fits schema + business rules +
    Tier-2 worked examples; thinking disabled via the dedicated `thinking`
    param (NOT additional_kwargs — see llm/ollama.py's build_ollama_client
    docstring for why that silently didn't work before) to save ~8-15s/call
    on Qwen3's internal chain-of-thought (see agent/orchestrator/loop.py
    history) and to avoid it burning the whole num_predict budget on
    <thinking> with zero tokens left for the actual answer — confirmed live
    via agent/domain_router.py's Tier-3 classifier prompt.

    temperature=0.0 (was 0.2) — confirmed live 2026-07-30 via 3 back-to-back
    eval_audit.py runs on the IDENTICAL question set: at 0.2, the same
    question's llm_calls varied run-to-run (e.g. one run's schema_hint
    included a bare `ap.id`/`acyc.id` and needed a validation retry, the next
    run's didn't select it at all — same prompt, different sampled
    completion). The SQL-generation LLM (get_sql_llm below) already runs at
    0.0 for exactly this reason; this brings the reasoning stage in line with
    it. This also makes eval runs directly comparable again — before this,
    re-running the same eval twice could show a different call count on the
    same question from sampling alone, making it hard to attribute a change
    in the numbers to an actual prompt/rule fix versus noise.
    """
    from src.config.settings import get_settings
    return build_ollama_client(
        model=model,
        base_url=base_url,
        temperature=0.0,
        request_timeout=120.0,
        thinking=False,
        json_mode=True,
        keep_alive=get_settings().ollama_keep_alive,
        additional_kwargs={
            "num_predict": 1024,
            "num_ctx": 20000,
        },
    )


def get_sql_llm(model: str, base_url: str):
    """
    SQL-generation LLM. temperature=0 for determinism; stop tokens prevent
    the model from appending explanation text after the query ends.
    """
    from src.config.settings import get_settings
    return build_ollama_client(
        model=model,
        base_url=base_url,
        temperature=0.0,
        request_timeout=120.0,
        keep_alive=get_settings().ollama_keep_alive,
        additional_kwargs={
            "num_predict": 1024,
            "num_ctx": 16384,
            "top_p": 0.9,
            "stop": [";", "[/SQL]", "This query", "Note:", "-- Note", "/*"],
        },
    )


def get_reasoning_llm_anthropic(model: str, api_key: str):
    """
    Claude equivalent of get_reasoning_llm above — same role tuning intent
    (deterministic, thinking disabled so a tight max_tokens can't be
    silently consumed by it), same model as get_sql_llm_anthropic per
    current setup (Sonnet for both roles) rather than Ollama's split
    between a small SQL-specialized model and a general chat model —
    Claude has no equivalent specialized-SQL-model distinction to mirror.
    """
    return build_anthropic_client(
        model=model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=1024,
        thinking=False,
    )


def get_sql_llm_anthropic(model: str, api_key: str):
    """
    Claude equivalent of get_sql_llm above. stop_sequences is the Anthropic
    Messages API's name for what Ollama calls `stop` — same values, same
    purpose (cut off trailing chatter after the query ends).

    No top_p — confirmed live: Claude Sonnet 5 (and the rest of the
    4.6+/5 family) rejects it outright with a 400 ("top_p is deprecated for
    this model"), unlike Ollama where top_p is a normal sampling knob.
    temperature=0.0 is accepted fine; it's specifically top_p/top_k that
    4.6+ models reject.
    """
    return build_anthropic_client(
        model=model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=1024,
        thinking=False,
        additional_kwargs={
            "stop_sequences": [";", "[/SQL]", "This query", "Note:", "-- Note", "/*"],
        },
    )


def get_reasoning_llm_vllm(model: str, base_url: str):
    """
    vLLM equivalent of get_reasoning_llm above. context_window/max_tokens
    mirror Ollama's num_ctx/num_predict values for this role.

    extra_body.chat_template_kwargs.enable_thinking=False disables Qwen3.5's
    thinking mode — confirmed live: without this, qwen3.5-9b-awq on vLLM
    returns content=null / finish_reason="length" with the entire max_tokens
    budget consumed by a `reasoning` field instead, the same failure class
    ollama.py's `thinking` param and anthropic.py's `thinking_dict` guard
    against for their providers. vLLM's OpenAI-compatible endpoint has no
    dedicated `thinking` field like those — passing extra_body is required
    because the openai SDK's create() only accepts its fixed named params
    directly; unknown keys must go through extra_body to reach the server.
    """
    return build_vllm_client(
        model=model,
        base_url=base_url,
        temperature=0.0,
        max_tokens=1024,
        context_window=20000,
        additional_kwargs={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )


def get_sql_llm_vllm(model: str, base_url: str):
    """
    vLLM equivalent of get_sql_llm above. Same stop tokens/top_p as the
    Ollama version, passed via additional_kwargs (OpenAI-compatible chat
    completion params) instead of Ollama's `options` dict. Also disables
    thinking mode — see get_reasoning_llm_vllm's docstring for why this is
    needed for Qwen3.5-family models served via vLLM.
    """
    return build_vllm_client(
        model=model,
        base_url=base_url,
        temperature=0.0,
        max_tokens=1024,
        context_window=16384,
        additional_kwargs={
            "top_p": 0.9,
            "stop": [";", "[/SQL]", "This query", "Note:", "-- Note", "/*"],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )


# {provider: {role: factory}} — every (provider, role) combination this app
# supports. Adding a new provider means adding one entry here, not touching
# every call site (see llm/base.py's LLMClient protocol).
_PROVIDER_ROLE_FACTORIES = {
    "ollama": {
        "reasoning": get_reasoning_llm,
        "sql": get_sql_llm,
    },
    "anthropic": {
        "reasoning": get_reasoning_llm_anthropic,
        "sql": get_sql_llm_anthropic,
    },
    "vllm": {
        "reasoning": get_reasoning_llm_vllm,
        "sql": get_sql_llm_vllm,
    },
}

# Providers whose factory takes an api_key as its second positional arg
# instead of a base_url (currently just Anthropic's hosted API).
_API_KEY_PROVIDERS = {"anthropic"}


def get_llm_client(provider: str, role: str, model: str, *, base_url: str = "", api_key: str = ""):
    """
    Resolve a single (provider, role) pair to a constructed LLM client.
    Reasoning and sql are resolved independently by callers — this function
    doesn't assume both roles share a provider.
    """
    try:
        roles = _PROVIDER_ROLE_FACTORIES[provider]
    except KeyError:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Expected one of {sorted(_PROVIDER_ROLE_FACTORIES)}."
        )
    try:
        factory = roles[role]
    except KeyError:
        raise ValueError(f"Unknown LLM role: {role!r}. Expected one of {sorted(roles)}.")
    if provider in _API_KEY_PROVIDERS:
        return factory(model, api_key)
    return factory(model, base_url)
