"""
Auth routes — /api/login, /api/logout, /api/me/permissions
=============================================================
Everything a client needs to establish and manage a session lives here:
proxying the OAuth password grant (client credentials never reach the
browser), invalidating a token server-side, and fetching/caching the
authenticated user's permission list.

Uses core.auth for token/JWT mechanics and core.rbac for the
data-visibility permission whitelist — this file is just the HTTP-endpoint
wiring around them. Included into the main app via
`app.include_router(auth_router)` in api/app.py.
"""

import logging
import re

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials as _HTTPCreds
from pydantic import BaseModel

from src.api.deps import get_orch
from src.config.settings import get_settings
from src.core.auth import _bearer_scheme, _decode_jwt_payload, _register_token, _valid_tokens, require_auth
from src.core.rbac import DATA_VISIBILITY_PERMS

log_auth = logging.getLogger("inspectai.auth")

auth_router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


@auth_router.post("/api/login")
async def login(req: LoginRequest, request: Request):
    """Proxy OAuth password-grant to the identity server.
    Client credentials never leave the backend. Returns the OAuth response
    plus the decoded Drupal sub so Angular can display user info."""
    settings = get_settings()
    oauth_url     = settings.oauth_url
    client_id     = settings.client_id
    client_secret = settings.client_secret
    ca_cert = settings.oauth_ca_cert
    skip_tls = settings.oauth_skip_tls_verify
    verify: str | bool = False if skip_tls else (ca_cert if ca_cert else True)

    payload = {
        "grant_type":    "password",
        "client_id":     client_id,
        "client_secret": client_secret,
        "username":      req.username,
        "password":      req.password,
    }
    log_auth.info("[AUTH] Login attempt user=%r", req.username)
    try:
        async with httpx.AsyncClient(verify=verify, timeout=15) as client:
            resp = await client.post(oauth_url, data=payload)
        if resp.status_code == 200:
            data = resp.json()
            access_token = data["access_token"]
            token_hint = access_token[:8] + "…"

            # Decode JWT to extract Drupal UID (sub claim).
            # Validate at decode time — a malformed sub must never enter the token store.
            claims = _decode_jwt_payload(access_token)
            sub = str(claims.get("sub", "")).strip()
            log_auth.debug("[AUTH] JWT decoded sub=%r claims_keys=%s", sub, list(claims.keys()))
            if not re.match(r'^\d+$', sub) or sub == "0":
                log_auth.warning("[AUTH] Login FAIL user=%r — invalid sub=%r", req.username, sub)
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token: sub must be a positive integer"
                )

            # Verify user is synced in the local DB (OAuth already validated credentials)
            login_orch = get_orch(request)
            from sqlalchemy import text as _sql_text
            try:
                with login_orch.db_engine.connect() as conn:
                    row = conn.execute(
                        _sql_text(
                            "SELECT id_fk FROM users WHERE id_fk = :sub LIMIT 1"
                        ),
                        {"sub": int(sub)},
                    ).fetchone()
                    log_auth.debug("[AUTH] DB lookup id_fk=%r → found=%s", sub, row is not None)
            except (ValueError, TypeError):
                log_auth.exception("[AUTH] sub=%r could not be cast to int", sub)
                raise HTTPException(status_code=401, detail="Invalid token: sub is not a valid user ID")
            except Exception:
                log_auth.exception("[AUTH] DB user sync check failed for sub=%r", sub)
                raise HTTPException(status_code=502, detail="Database lookup failed")

            if not row:
                log_auth.warning("[AUTH] Login FAIL user=%r sub=%r — not in local DB", req.username, sub)
                raise HTTPException(status_code=403, detail="User not found in system")
            if row.id_fk == -1:
                log_auth.warning("[AUTH] Login FAIL user=%r sub=%r — account not synced", req.username, sub)
                raise HTTPException(status_code=403, detail="User account not synchronized")

            # Register token with empty permissions — client fetches via /api/me/permissions
            _register_token(access_token, sub, [], int(data.get("expires_in", 3600)))
            log_auth.info("[AUTH] Login OK user=%r sub=%s token=%s expires_in=%ss",
                          req.username, sub, token_hint, data.get("expires_in", 3600))
            return {**data, "sub": sub}

        try:
            detail = resp.json().get("error_description") or resp.json().get("message") or "Invalid credentials"
        except Exception:
            detail = "Invalid credentials"
        log_auth.warning("[AUTH] Login FAIL user=%r — OAuth status=%d detail=%r",
                         req.username, resp.status_code, detail)
        # Never forward a 5xx from the OAuth server — it looks like our fault.
        fwd_status = resp.status_code if resp.status_code < 500 else 502
        raise HTTPException(status_code=fwd_status, detail=detail)
    except HTTPException:
        raise
    except Exception as _exc:
        import traceback
        _tb = traceback.format_exc()
        log_auth.error("[AUTH] Login ERROR user=%r exc=%s\n%s", req.username, _exc, _tb)
        raise HTTPException(status_code=502, detail=f"Authentication service unavailable: {_exc}")


@auth_router.post("/api/logout")
async def logout(
    credentials: _HTTPCreds | None = Depends(_bearer_scheme),
    token_data: dict = Depends(require_auth),
):
    """Invalidate the caller's access token server-side."""
    token = credentials.credentials  # type: ignore[union-attr]
    _valid_tokens.pop(token, None)
    log_auth.info("[AUTH] Logout sub=%s", token_data.get("sub"))
    return {"status": "logged_out"}


@auth_router.get("/api/me/permissions")
async def get_my_permissions(
    credentials: _HTTPCreds | None = Depends(_bearer_scheme),
    token_data: dict = Depends(require_auth),
    orch=Depends(get_orch),
):
    """Fetch (and cache) the permission list for the authenticated user.
    The full list is stored in the token store but only DATA_VISIBILITY_PERMS
    is ever forwarded to the orchestrator — nothing else reaches the SQL layer."""
    sub = token_data["sub"]  # Drupal integer UID from JWT

    # Return cached list if already populated (avoids hammering the Java API)
    if token_data.get("permissions"):
        return {"sub": sub, "permissions": token_data["permissions"],
                "first_name": token_data.get("first_name", "")}

    # The permissions API expects the internal UUID, not the Drupal integer.
    # Resolve: JWT sub (int) → users.id_fk → users.id (UUID)
    from sqlalchemy import text as _sql_text
    try:
        with orch.db_engine.connect() as conn:
            row = conn.execute(
                _sql_text("SELECT id, first_name FROM users WHERE id_fk = :sub LIMIT 1"),
                {"sub": int(sub)},
            ).fetchone()
    except Exception:
        logging.exception("UUID lookup failed for sub=%s", sub)
        raise HTTPException(status_code=502, detail="User UUID lookup failed")

    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    user_uuid  = str(row.id)
    first_name = (row.first_name or "").strip()

    perms_base = get_settings().permissions_api_url
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{perms_base}/api/users/{user_uuid}/permissions")
        if resp.status_code == 404:
            log_auth.warning("[AUTH] Permissions API 404 sub=%s uuid=%s — no permissions assigned", sub, user_uuid)
            raise HTTPException(status_code=403, detail="You have no permission to access this feature")
        if resp.status_code != 200:
            log_auth.warning("[AUTH] Permissions API returned %s for sub=%s uuid=%s",
                             resp.status_code, sub, user_uuid)
            raise HTTPException(status_code=502, detail="Permissions service unavailable")
        result = resp.json()
        raw = result.get("result", [])
        # Deduplicate while preserving order (API sometimes repeats entries)
        permissions = list(dict.fromkeys(raw))

        if not permissions:
            log_auth.warning("[AUTH] No permissions assigned sub=%s uuid=%s", sub, user_uuid)
            raise HTTPException(status_code=403, detail="You have no permission to access this feature")

        data_vis = [p for p in permissions if p in DATA_VISIBILITY_PERMS]

        if not data_vis:
            log_auth.warning(
                "[AUTH] Access DENIED sub=%s uuid=%s — total=%d permissions but 0 data-visibility grants",
                sub, user_uuid, len(permissions),
            )
            raise HTTPException(status_code=403, detail="You have no permission to access this feature")

        # Hydrate the token store so future require_auth calls carry permissions
        token_str = credentials.credentials if credentials else None
        if token_str and token_str in _valid_tokens:
            _valid_tokens[token_str]["permissions"] = permissions
            _valid_tokens[token_str]["first_name"]  = first_name

        log_auth.info("[AUTH] Permissions loaded sub=%s uuid=%s total=%d data_visibility=%d: %s",
                      sub, user_uuid, len(permissions), len(data_vis), data_vis or "(none)")
        return {"sub": sub, "permissions": permissions, "first_name": first_name}
    except HTTPException:
        raise
    except Exception:
        log_auth.exception("[AUTH] Permissions fetch error for sub=%s uuid=%s", sub, user_uuid)
        raise HTTPException(status_code=502, detail="Permissions service unavailable")
