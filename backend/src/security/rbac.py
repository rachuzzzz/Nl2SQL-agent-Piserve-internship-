"""
RBAC data-visibility scope
===========================
Turns a user's permission list into the SQL predicate that scopes their
queries — the boundary between "what permissions the identity API returned"
and "which rows the SQL layer is allowed to touch."

Role names must never appear here — only permission strings drive scope
decisions, matching how the identity/permissions API actually describes
users. Consumed by agent/orchestrator.py (query()) and fastapi_app.py /
auth_routes.py (pagination re-fetch, permissions endpoint).

Covers two domains (inspection and audit) — build_scope_filter(domain=...)
dispatches to the right branch. Audit permission STRINGS below are
placeholders following the inspection domain's naming convention
('view list of own audit schedules' etc.) pending confirmation against the
real identity-API values — see the domain-router plan's §5. Structurally
this mirrors inspection's inspector/inspectee split with auditor/auditee.
"""

import logging

_log_security = logging.getLogger("inspectai.security")

# Inspection data-visibility permissions — the only ones the orchestrator is
# allowed to know about. All other permissions are stripped before the
# visibility_scope dict reaches the SQL layer. This is the boundary wall.
_INSPECTION_DATA_VISIBILITY_PERMS: frozenset = frozenset({
    'view list of all inspections',
    'view list of own inspections',
    'view own inspection dashboard',
    'view own inspection portfolio list',
    'view list of own inspection schedules',
    'view list of own inspection mitigative actions',
    'view list of all inspection mitigative actions',
    'conduct an inspection',
    'can be an inspectee',
    # Grants scoped access to PO details linked to the user's own MAs
    'view po details in any mitigative action',
    # Scoped cycle and project dashboard access via the inspection chain
    'view inspection cycle list',
    'view project inspection dashboard',
})

# Audit data-visibility permissions — same boundary-wall role as the
# inspection set above, added to the flat allowlist so chat.py's single
# pre-filter step (`[p for p in perms if p in DATA_VISIBILITY_PERMS]`)
# doesn't need to know which domain a request will resolve to before the
# router has even run. ⚠ PLACEHOLDER STRINGS — confirm against the real
# identity-API permission names before this goes live (see module docstring).
_AUDIT_DATA_VISIBILITY_PERMS: frozenset = frozenset({
    'view list of all audits',
    'view list of own audits',
    'view own audit dashboard',
    'view own auditor portfolio list',
    'view list of own audit schedules',
    'view list of own audit corrective actions',
    'view list of all audit corrective actions',
    'conduct an audit',
    'can be an auditee',
})

# Combined allowlist — chat.py filters the raw identity-API permission list
# against this ONE frozenset before the domain is even known (routing happens
# after). build_scope_filter(domain=...) below is what actually branches.
DATA_VISIBILITY_PERMS: frozenset = _INSPECTION_DATA_VISIBILITY_PERMS | _AUDIT_DATA_VISIBILITY_PERMS

# Permissions that exclusively identify the inspector role (drive scope to
# ir.inspector_user_id).  Shared permissions — view list of own inspections,
# view own inspection dashboard, view list of own inspection mitigative
# actions, view inspection cycle list, view project inspection dashboard —
# are intentionally excluded because FM roles (AD FM, FM Manager, FM
# Supervisor, etc.) also hold them despite being inspectee-only roles.
# Including shared perms here would give FM users ir.inspector_user_id scope
# instead of the correct ir.inspectee_user_id scope.
_INSPECTOR_PERMS: frozenset = frozenset({
    'conduct an inspection',
    'view own inspection portfolio list',
    'view list of own inspection schedules',
})

# Audit equivalent of _INSPECTOR_PERMS — mirrors the same "exclude shared
# perms" reasoning for the auditor/auditee split.
_AUDITOR_PERMS: frozenset = frozenset({
    'conduct an audit',
    'view own auditor portfolio list',
    'view list of own audit schedules',
})


def _build_inspection_scope_filter(perms: set, user_subq: str) -> str:
    # Unrestricted — can see every inspection / every MA (and thus every PO)
    if 'view list of all inspections' in perms:
        return ''

    is_inspector = bool(perms & _INSPECTOR_PERMS)
    is_inspectee = 'can be an inspectee' in perms

    # Dual-role: union both columns so neither view is suppressed
    if is_inspector and is_inspectee:
        return (
            f"(ir.inspector_user_id = {user_subq}"
            f" OR ir.inspectee_user_id = {user_subq})"
        )
    if is_inspector:
        return f"ir.inspector_user_id = {user_subq}"
    if is_inspectee:
        return f"ir.inspectee_user_id = {user_subq}"
    return 'BLOCK'


def _build_audit_scope_filter(perms: set, user_subq: str) -> str:
    # Unrestricted — can see every audit
    if 'view list of all audits' in perms:
        return ''

    is_auditor = bool(perms & _AUDITOR_PERMS)
    is_auditee = 'can be an auditee' in perms

    if is_auditor and is_auditee:
        return (
            f"(asch.auditor_id = {user_subq}"
            f" OR asch.auditee_id = {user_subq})"
        )
    if is_auditor:
        return f"asch.auditor_id = {user_subq}"
    if is_auditee:
        return f"asch.auditee_id = {user_subq}"
    return 'BLOCK'


def build_scope_filter(visibility_scope: dict, domain: str = "inspection", qid: str = "") -> str:
    """Return a SQL WHERE fragment that enforces data-visibility scope.

    Returns '' for unrestricted access, 'BLOCK' when the user has no
    permissions for the given domain, or a concrete SQL predicate otherwise.
    Role names must never appear here — only permission strings drive this.

    domain: "inspection" (default, unchanged behavior) or "audit" — selects
    which permission set/predicate shape applies. See agent/domains.py's
    DOMAIN_REGISTRY and AgentOrchestrator, which passes its own domain
    through automatically.

    Dual-role users (both inspector/auditor and inspectee/auditee) get a
    union predicate so they can see rows where they appear in either column.

    ⚠ FAIL-OPEN TRAP: ANY falsy `visibility_scope` — including `{}`, not just
    `None` — is treated as UNRESTRICTED access (returns ''), matching eval.py's
    intentional use of `visibility_scope={}` for its offline full-access test
    harness. This means "permissions not yet fetched" must NEVER be
    represented as `{}` by a caller — it must be `None`, checked with
    `is None` BEFORE calling this function (see agent/orchestrator.py's
    `query()`, which converts `None` to a hard 'BLOCK' without ever reaching
    here). A caller that computes `{}` instead of `None` for an "unavailable
    permissions" case will silently grant unrestricted DB access instead of
    blocking.
    """
    if visibility_scope is None or visibility_scope == {}:
        _log_security.warning(
            "[SCOPE] qid=%s build_scope_filter received empty/None visibility_scope — "
            "treating as UNRESTRICTED (fail-open); confirm caller meant this, "
            "not an unfetched-permissions bug",
            qid,
        )
        return ''
    perms = set(visibility_scope.get('permissions', []))
    sub   = str(visibility_scope.get('sub', '')).strip()

    # Empty perms inside a non-None visibility_scope means the Java API
    # returned a list with zero data-visibility entries — deny access.
    # (None visibility_scope is already converted to 'BLOCK' before this
    # function is called, so we never reach here with an unfetched state.)
    if not perms:
        return 'BLOCK'
    if not sub:
        return 'BLOCK'

    # Sub-query to map Drupal UID → internal UUID
    user_subq = (
        f"(SELECT id FROM users WHERE id_fk = {int(sub)} "
        f"AND active = true AND is_synced = true)"
    )

    if domain == "audit":
        return _build_audit_scope_filter(perms, user_subq)
    return _build_inspection_scope_filter(perms, user_subq)
