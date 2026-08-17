"""
Application-level cooperator for Ollama's own slot-based prefix caching.

NOT a KV-cache store — Ollama exposes no API to externally save/restore a
prefix's KV state (see the PrefixCache MVP plan). What Ollama's llama.cpp
server already does, automatically, is match the longest common prefix of a
new request against whatever it still has cached from a recent one on the
same model. This module's job is just to (a) warm that cache before real
traffic needs it, and (b) make cache hits/misses visible in logs, so the app
cooperates with behavior Ollama already provides rather than reimplementing
it.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger("inspectai.llm")


class PrefixCacheManager:
    def __init__(self, llm_client: Any, domain: str, role: str):
        self._llm_client = llm_client
        self._domain = domain
        self._role = role

    def warmup(self, static_prefix: str, sample_question: str = "warmup") -> None:
        """
        Fire one cheap real call carrying only the static prefix, so the
        first genuine user request isn't the one paying the cold-miss
        prefill cost. Best-effort: a failed warmup just means the first
        real request pays cold-miss cost, same as if warmup were never
        called — it must never block orchestrator startup.

        Mirrors the real call shape for this role: sql role's real call
        site (sql/generation.py) always uses .complete(prompt) on a single
        string, never .chat() — Ollama's prefix match is against the actual
        request bytes it receives, so warming via the wrong call shape
        would prime a prefix the real traffic never sends. Every other role
        (reasoning) is called via .chat() (see synthesis.py/turn_classifier.py).
        """
        try:
            if self._role == "sql":
                self._llm_client.complete(static_prefix)
            else:
                from llama_index.core.llms import ChatMessage, MessageRole
                self._llm_client.chat([
                    ChatMessage(role=MessageRole.SYSTEM, content=static_prefix),
                    ChatMessage(role=MessageRole.USER, content=sample_question),
                ])
            _log.debug("[PREFIX-CACHE] warmup complete domain=%s role=%s",
                       self._domain, self._role)
        except Exception as e:
            _log.debug("[PREFIX-CACHE] warmup failed domain=%s role=%s: %s",
                       self._domain, self._role, e)

    def record(self, raw_response: Any) -> None:
        """
        Log a derived prefill-speed metric from an already-made call's raw
        response, so cache hits (fast prefill) vs. misses (slow prefill) are
        visible directly in debug_llm_*.log without needing the benchmark
        harness. Silently no-ops for non-Ollama raw shapes (Anthropic/vLLM
        raw objects don't carry these keys) so it's safe to call from any
        provider's call site — see llm/base.py's extract_token_counts for
        the same cross-provider raw-shape caveat.
        """
        if not isinstance(raw_response, dict):
            return
        prompt_eval_count = raw_response.get("prompt_eval_count")
        prompt_eval_duration = raw_response.get("prompt_eval_duration")
        if not prompt_eval_count or not prompt_eval_duration:
            return
        prefill_ms = prompt_eval_duration / 1e6
        tok_per_sec = prompt_eval_count / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
        load_ms = (raw_response.get("load_duration") or 0) / 1e6
        _log.debug(
            "[PREFIX-CACHE] domain=%s role=%s prefill_tokens=%d prefill_ms=%.0f "
            "prefill_tok_per_sec=%.0f load_ms=%.0f",
            self._domain, self._role, prompt_eval_count, prefill_ms, tok_per_sec, load_ms,
        )
