"""
Role-based LLM client construction.

Each role's tuning (context window, stop tokens, timeout, temperature) is
load-bearing — see inline comments — so this factory keeps per-role presets
rather than exposing one generic client. Consumers (agent/orchestrator) ask
for a role, not a provider; swapping Ollama for another provider later means
changing the functions below, not every call site.
"""

from __future__ import annotations

from src.llm.ollama import build_ollama_client


def get_reasoning_llm(model: str, base_url: str):
    """
    Reasoning/planning LLM. num_ctx=20000 fits schema + business rules +
    Tier-2 worked examples; "think" disabled to save ~8-15s/call on Qwen3's
    internal chain-of-thought (see agent/orchestrator/loop.py history).
    """
    return build_ollama_client(
        model=model,
        base_url=base_url,
        temperature=0.2,
        request_timeout=120.0,
        additional_kwargs={
            "num_predict": 1024,
            "num_ctx": 20000,
            "think": False,
        },
    )


def get_sql_llm(model: str, base_url: str):
    """
    SQL-generation LLM. temperature=0 for determinism; stop tokens prevent
    the model from appending explanation text after the query ends.
    """
    return build_ollama_client(
        model=model,
        base_url=base_url,
        temperature=0.0,
        request_timeout=120.0,
        additional_kwargs={
            "num_predict": 1024,
            "num_ctx": 16384,
            "top_p": 0.9,
            "stop": [";", "[/SQL]", "This query", "Note:", "-- Note", "/*"],
        },
    )


_ROLE_FACTORIES = {
    "reasoning": get_reasoning_llm,
    "sql": get_sql_llm,
}


def get_llm_client(role: str, model: str, base_url: str):
    try:
        factory = _ROLE_FACTORIES[role]
    except KeyError:
        raise ValueError(f"Unknown LLM role: {role!r}. Expected one of {sorted(_ROLE_FACTORIES)}.")
    return factory(model, base_url)
