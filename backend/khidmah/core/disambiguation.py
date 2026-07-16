"""
Disambiguation result-shape detection
========================================
One canonical answer to "is this SQL result a name-disambiguation pick-list
(the AMBIGUOUS REFERENCE RULE STEP 1 people-search shape), or just a
legitimate multi-row report that happens to alias its columns the same way."

Requires the SQL text, not just column names — a report like "list all users
with email" can legitimately select `u.id AS user_id, name, email` without
being a name search. Column-only detection can't tell those apart; this can.

Shared by agent/orchestrator.py's _synthesize()/_extract_disambiguation() and
fastapi_app.py's dashboard-suppression check, so the chat text and the
dashboard/chart shown for the same result never disagree about what kind of
result it is. Deliberately dependency-free (no llama_index/torch/sqlalchemy)
so importing it doesn't force fastapi_app.py to eagerly load the heavy
agent.orchestrator module it otherwise only imports lazily at startup.

Also provides `is_entity_disambiguation_result`, the sibling check for
non-people (facility/project/client/organisation) STEP 1 count-checks.
"""

import re


def is_disambiguation_result(columns: list, sql: str = "") -> bool:
    """
    True if `columns` has the people-disambiguation shape (user_id + name)
    AND, when `sql` is available, the query actually searches by name via
    ILIKE on first_name/last_name — the hallmark of a STEP 1 disambiguation
    count-check, as opposed to some other report that reuses the same column
    aliasing convention.

    If `sql` isn't available at the call site, falls back to the column-only
    signal rather than never firing.
    """
    col_set = {c.lower() for c in columns}
    if not ("user_id" in col_set and "name" in col_set):
        return False
    if not sql:
        return True
    sql_lower = sql.lower()
    return "ilike" in sql_lower and ("first_name" in sql_lower or "last_name" in sql_lower)


# Tables the AMBIGUOUS REFERENCE RULE documents as sharing the entity STEP 1
# shape (agent/prompts.py: "same shape for client/project/organisation —
# swap table and the FK"). Restricting to this known list (rather than
# accepting any table name) avoids false-positives on unrelated 3-column
# reports that happen to alias their columns id/name/inspection_count.
_ENTITY_DISAMBIGUATION_TABLES = {"facility", "project", "client", "organisation"}

_FROM_TABLE_ALIAS_RE = re.compile(
    r"\bfrom\s+(\w+)\s+(?:as\s+)?(\w+)\b", re.IGNORECASE
)


def is_entity_disambiguation_result(columns: list, sql: str = "") -> tuple[str, str] | None:
    """
    True-equivalent for the entity/facility-disambiguation shape (id + name +
    inspection_count) — the sibling check to `is_disambiguation_result` for
    non-people STEP 1 count-checks ("4 facilities match 'adnoc'", "2 projects
    match 'tower'", etc).

    Requires the exact 3-column shape plus an ILIKE filter, and — when `sql`
    is available — parses the first `FROM <table> <alias>` in the query and
    confirms `<table>` is one of the entities the AMBIGUOUS REFERENCE RULE
    documents as using this template. Returns `(table, alias)` on match
    (e.g. `("facility", "fac")`), `None` otherwise — the table/alias let
    callers build a correct `{alias}.name ILIKE '%...%'` filter and correct
    wording for whichever entity was actually searched, instead of assuming
    facility.
    """
    col_set = {c.lower() for c in columns}
    if col_set != {"id", "name", "inspection_count"}:
        return None
    if not sql or "ilike" not in sql.lower():
        return None
    m = _FROM_TABLE_ALIAS_RE.search(sql)
    if not m:
        return None
    table, alias = m.group(1).lower(), m.group(2)
    if table not in _ENTITY_DISAMBIGUATION_TABLES:
        return None
    return table, alias


_SEARCH_TERM_RE = re.compile(r"ILIKE\s+'%([^%]+)%'", re.IGNORECASE)


def extract_search_term(sql: str) -> str | None:
    """
    Pull the literal search text out of a disambiguation STEP 1 query's ILIKE
    clause (e.g. `fac.name ILIKE '%ADNOc%'` -> "ADNOc"). Callers must only
    invoke this once `is_disambiguation_result`/`is_entity_disambiguation_result`
    has already confirmed the SQL is a STEP 1 count-check — which by
    construction has exactly one name/entity ILIKE filter — so taking the
    first match unqualified (unlike the more carefully-anchored extraction in
    agent/orchestrator.py) is safe here.
    """
    if not sql:
        return None
    m = _SEARCH_TERM_RE.search(sql)
    return m.group(1).strip() if m and m.group(1).strip() else None
