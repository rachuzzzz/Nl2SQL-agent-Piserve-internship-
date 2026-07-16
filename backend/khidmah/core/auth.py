"""
Authentication — bearer tokens and session state
===================================================
Everything needed to answer "is this request authenticated, and as whom":
JWT payload decoding (unverified — the OAuth server is the source of truth,
this just reads the sub claim), the in-memory token store, and the
`require_auth` FastAPI dependency every protected route depends on.

Not concerned with *what* an authenticated user is allowed to see — that's
core/rbac.py.
"""

import time as _time

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer as _HTTPBearer, HTTPAuthorizationCredentials as _HTTPCreds

# Maps access_token → {expiry, sub, permissions[]}. Populated on /api/login
# (sub + expiry only) and hydrated by /api/me/permissions. Cleared on restart.
_valid_tokens: dict[str, dict] = {}
_bearer_scheme = _HTTPBearer(auto_error=False)


def _decode_jwt_payload(token: str) -> dict:
    """Decode the JWT payload segment (base64url) without signature verification."""
    import base64, json as _json
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return {}
        payload = parts[1]
        payload += '=' * (4 - len(payload) % 4)
        return _json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _register_token(token: str, sub: str, permissions: list, expires_in: int = 3600) -> None:
    _valid_tokens[token] = {
        'expiry': _time.time() + expires_in,
        'sub': sub,
        'permissions': permissions,
    }
    now = _time.time()
    expired = [t for t, data in list(_valid_tokens.items()) if data['expiry'] < now]
    for t in expired:
        _valid_tokens.pop(t, None)


async def require_auth(
    credentials: _HTTPCreds | None = Depends(_bearer_scheme),
) -> dict:
    token = credentials.credentials if credentials else None
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    data = _valid_tokens.get(token)
    if not data or data['expiry'] < _time.time():
        raise HTTPException(status_code=401, detail="Authentication required")
    return data
