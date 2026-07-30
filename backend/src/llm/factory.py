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
    return build_ollama_client(
        model=model,
        base_url=base_url,
        temperature=0.0,
        request_timeout=120.0,
        thinking=False,
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
