"""
Real per-string token counts via llama.cpp's native /tokenize endpoint.

Phase 3.8 profiling need: 3.7 established that static prompt SIZE doesn't
predict prefill COST (llama.cpp's slot/prefix cache absorbs repeated static
content almost for free) — see generation.py's _profile-gated prompt
composition breakdown. To find the next real optimization target we need
real tokenizer counts for the DYNAMIC prompt components (question,
schema_hint, retrieved examples), not chars/4 estimates, and not just the
model's own reported prompt_tokens (which is a single total, not a
breakdown). llama.cpp's OpenAI-compatible /v1 route has no per-string
tokenize endpoint, but llama-server itself exposes one at /tokenize (server
root, NOT under /v1) — this hits that directly.

Deliberately NOT used on the hot path: only sql/generation.py's _profile
flag (off by default, opt-in via eval scripts' --profile-tokens) calls this,
since it adds one extra HTTP round-trip per component per SQL-gen call.
"""

import httpx


def count_tokens_llamacpp(text: str, base_url: str, timeout: float = 10.0) -> int:
    """
    Returns the real token count of `text` per llama-server's own tokenizer.

    base_url is expected in the same shape as LLAMACPP_SQL_URL/
    LLAMACPP_REASONING_URL (e.g. "http://host:port/v1", the OpenAI-compatible
    chat route) — /tokenize lives at the server root, so a trailing "/v1" is
    stripped before appending it.

    Returns 0 (never raises) on empty input, a missing base_url, or any
    request failure (timeout, connection error, non-2xx) — a flaky profiling
    call should degrade one component's count to 0, not abort the eval run
    that's using it.
    """
    if not text or not base_url:
        return 0
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        resp = httpx.post(f"{root}/tokenize", json={"content": text}, timeout=timeout)
        resp.raise_for_status()
        return len(resp.json().get("tokens", []))
    except Exception:
        return 0
