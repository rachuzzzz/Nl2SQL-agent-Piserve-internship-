"""
AUTOCORRECT_RULES: extracted verbatim from the original
core/business_rules.py REGISTRY list (all "type": AUTOCORRECT entries,
in original relative order). See core.business_rules.engine for how
these are aggregated and consumed.
"""

from src.sql.rules._types import AUTOCORRECT

AUTOCORRECT_RULES: list[dict] = [

    # ── AUTOCORRECT: strip the hallucinated frequency_definition join ─────────
    {
        "id":          "autocorrect_isched_frequency_definition_join",
        "type":        AUTOCORRECT,
        "description": "Remove hallucinated LEFT JOIN frequency_definition fd ON isched.frequency_definition_id = fd.id.",
        "sql_pattern": r"LEFT\s+JOIN\s+frequency_definition\s+\w+\s+ON\s+\w+\.frequency_definition_id\s*=\s*\w+\.id\s*",
        "replacement": "",
        "message": "[autocorrect] Removed hallucinated JOIN frequency_definition via non-existent isched.frequency_definition_id",
    },

    # ── AUTOCORRECT: remove fac.name ILIKE from inside CTE bodies ─────
    {
        "id":          "autocorrect_fac_filter_in_cte",
        "type":        AUTOCORRECT,
        "description": "Remove fac.name ILIKE filter placed inside a CTE WHERE clause where fac is not in scope. Facility filter must be in the outer query.",
        "sql_pattern": r"\s+AND\s+fac\.name\s+ILIKE\s+'%[^']*%'(?=\s*(?:GROUP BY|ORDER BY|LIMIT|\)))",
        "replacement": "",
        "message": "[autocorrect] Removed fac.name ILIKE from CTE body where fac is not in scope; facility filter must be in outer WHERE clause",
    },


    # ══════════════════════════════════════════════════════════════════════════
    # AUTOCORRECT — applied automatically in clean_sql() before validation
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id":          "autocorrect_ica_ir_join_varchar_uuid",
        "type":        AUTOCORRECT,
        "description": "Fix ICA→IR join using varchar column instead of UUID PK.",
        "sql_pattern": r"\bir\.inspection_id\s*=\s*ica\.inspection_id\b",
        "replacement": "ica.inspection_id = ir.id",
        "message":     "[autocorrect] Fixed ICA→IR join direction (varchar=UUID → UUID=UUID)",
    },

    {
        "id":          "autocorrect_ica_ir_join_reversed",
        "type":        AUTOCORRECT,
        "description": "Fix reversed ICA→IR join.",
        "sql_pattern": r"\bica\.inspection_id\s*=\s*ir\.inspection_id\b",
        "replacement": "ica.inspection_id = ir.id",
        "message":     "[autocorrect] Fixed reversed ICA→IR join",
    },

    {
        "id":          "autocorrect_strip_responsible_users_join",
        "type":        AUTOCORRECT,
        "description": "Strip invalid LEFT JOIN users on ica.responsible enum comparison.",
        # Matches: LEFT JOIN users <alias> ON ica.responsible = '<ENUM>' AND <anything up to u.id>
        "sql_pattern": (
            r"\s*LEFT\s+JOIN\s+users\s+\w+\s+ON\s+ica\.responsible\s*=\s*'[A-Z_]+'"
            r"\s+AND\s+[\w.]+\s*=\s*\w+\.id"
        ),
        "replacement": "",
        "message":     "[autocorrect] Removed invalid users join on ica.responsible enum",
    },

    {
        "id":          "autocorrect_responsible_party_select",
        "type":        AUTOCORRECT,
        "description": "Replace u.first_name||' '||u.last_name AS responsible_party with ica.responsible.",
        "sql_pattern": (
            r"\w+\.first_name\s*\|\|\s*'\s+'\s*\|\|\s*\w+\.last_name\s+AS\s+responsible_party"
        ),
        "replacement": "ica.responsible AS responsible_party",
        "message":     "[autocorrect] responsible_party now selects ica.responsible enum, not a user name",
    },

    {
        "id":          "autocorrect_ica_ir_join_cast",
        "type":        AUTOCORRECT,
        "description": "Fix CAST(ir.inspection_id AS ...) = ica.inspection_id type mismatch.",
        "sql_pattern": r"CAST\s*\(\s*ir\.inspection_id\s+AS\s+\w+\s*\)\s*=\s*ica\.inspection_id\b",
        "replacement": "ica.inspection_id = ir.id",
        "message":     "[autocorrect] Fixed CAST ICA→IR join",
    },

    {
        "id":          "autocorrect_ica_ir_join_cast_typecast",
        "type":        AUTOCORRECT,
        "description": "Fix ir.inspection_id::type = ica.inspection_id type mismatch.",
        "sql_pattern": r"\bir\.inspection_id\s*::\s*\w+(?:\(\d+\))?\s*=\s*ica\.inspection_id\b",
        "replacement": "ica.inspection_id = ir.id",
        "message":     "[autocorrect] Fixed typecast ICA→IR join",
    },

    {
        "id":          "autocorrect_reserved_alias_is",
        "type":        AUTOCORRECT,
        "description": "Replace 'is' as table alias (SQL reserved word) with 'isch'.",
        "sql_pattern": r"(\b\w+\s+)is\b(?=\s*\.|\s+ON\b|\s+WHERE\b|\s+GROUP\b|\s+ORDER\b|\s+LEFT\b|\s+JOIN\b|\s+INNER\b|\s*\n|\s*,)",
        "replacement": r"\1isch",
        "message":     "[autocorrect] Replaced reserved alias 'is' with 'isch'",
    },


    {
        "id":          "autocorrect_impact_ilike",
        "type":        AUTOCORRECT,
        "description": "aq.label ILIKE '%impact%' matches wrong labels — use exact match",
        "sql_pattern": r"aq\.label\s+ILIKE\s+'%[Ii]mpact%'",
        "replacement": "aq.label = 'Impact'",
        "message":     "[autocorrect] Replaced ILIKE Impact with exact aq.label = 'Impact'",
    },

]
