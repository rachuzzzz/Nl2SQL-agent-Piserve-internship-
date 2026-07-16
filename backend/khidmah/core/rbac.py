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
"""

# Inspection data-visibility permissions — the only ones the orchestrator is
# allowed to know about. All other permissions are stripped before the
# visibility_scope dict reaches the SQL layer. This is the boundary wall.
DATA_VISIBILITY_PERMS: frozenset = frozenset({
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


def build_scope_filter(visibility_scope: dict) -> str:
    """Return a SQL WHERE fragment that enforces data-visibility scope.

    Returns '' for unrestricted access, 'BLOCK' when the user has no
    inspection permissions at all, or a concrete SQL predicate otherwise.
    Role names must never appear here — only permission strings drive this.

    Dual-role users (both inspector and inspectee) get a union predicate so
    they can see rows where they appear in either column.

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

    # Unrestricted — can see every inspection / every MA (and thus every PO)
    if 'view list of all inspections' in perms:
        return ''

    # Sub-query to map Drupal UID → internal UUID
    user_subq = (
        f"(SELECT id FROM users WHERE id_fk = {int(sub)} "
        f"AND active = true AND is_synced = true)"
    )

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
