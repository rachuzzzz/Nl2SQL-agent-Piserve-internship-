"""
DisambiguationMixin: orchestrator-specific extraction/messaging glue
built on top of sql.result_shape's generic shape-detection utilities
-- tied to ConversationTurn and LLM prompt strings, so it stays separate
from that dependency-free module. Extracted verbatim from the original
agent/orchestrator.py.
"""

import re

from src.sql.result_shape import is_disambiguation_result, is_entity_disambiguation_result

from src.agent.orchestrator.debug import _ilike_escape, _UUID_RE


# Module-level (not a method) so session.py's ordinal-noun-type lookup
# (entity_type_for_ordinal_noun below) can reuse the EXACT SAME map
# _is_preparatory_result consumes for its table-membership check, instead
# of hand-rolling a parallel noun list that could drift out of sync.
#
# 3-tuples: (regex pattern, table name or None, entity-type label).
#   - `table` is consumed ONLY by _is_preparatory_result below (is the
#     table this noun implies actually present in the executed SQL?).
#   - `entity_type` is consumed ONLY by entity_type_for_ordinal_noun below,
#     and must match the row_entity_type strings synthesis.py's _finish()
#     assigns (see that file's row_titles-capture loop).
#   - `table=None` marks an entry that exists PURELY for the entity-type
#     lookup, deliberately excluded from the table-membership check — a
#     bare "auditor"/"inspector" -> "users" table check would misfire on a
#     legitimate answer already scoped by a resolved id (e.g. WHERE
#     po.auditor_id = '<uuid>', no JOIN users needed at all).
def _entity_noun_tables(domain):
    _people = [
        (r"\bauditors?\b", None, "person"),
        (r"\binspectors?\b", None, "person"),
    ]
    if domain == "audit":
        return [
            (r"\baudit\s+plans?\b", "audit_plan", "plan"),
            (r"\b(?:audit\s+)?schedules?\b", "audit_schedule", "schedule"),
            (r"\bportfolios?\b", "auditor_portfolio", "portfolio"),
            (r"\bcorrective\s+actions?\b", "audit_corrective_action", "corrective_action"),
            (r"\baudit\s+reports?\b", "audit_report", "audit"),
            (r"\baudit\s+cycles?\b", "audit_cycle", "cycle"),
            (r"\baudit\s+forms?\b", "audit_forms", "form"),
        ] + _people
    return [
        (r"\binspection\s+reports?\b", "inspection_report", "inspection"),
        (r"\bmitigative\s+actions?\b", "inspection_corrective_action", "corrective_action"),
        (r"\bschedules?\b", "inspection_schedule", "schedule"),
        (r"\bportfolios?\b", "inspector_portfolio", "portfolio"),
    ] + _people


def entity_type_for_ordinal_noun(noun_snippet: str, domain: str) -> str:
    """Entity-type label for the first recognized noun in `noun_snippet`.

    Callers (session.py's ordinal resolver) must pass only the few words
    immediately following an ordinal phrase ("the 6th ___"), NOT the whole
    question — so an unrelated entity noun elsewhere in the sentence ("list
    all audit PLANS under the 6th AUDITOR") can't be mistaken for the noun
    THIS ordinal is actually about. Returns "" when nothing recognized —
    e.g. a bare ordinal ("the 4th one") — callers must preserve today's
    untyped-match behavior in that case, not silently skip resolution.
    """
    for pat, _table, etype in _entity_noun_tables(domain):
        if re.search(pat, noun_snippet, re.IGNORECASE):
            return etype
    if not domain:
        # domain wasn't passed/known — also try "audit" rather than
        # silently defaulting to only the inspection-side map (which
        # _entity_noun_tables' domain=="audit"-else branching would
        # otherwise do for any falsy domain).
        for pat, _table, etype in _entity_noun_tables("audit"):
            if re.search(pat, noun_snippet, re.IGNORECASE):
                return etype
    return ""


class DisambiguationMixin:

    @staticmethod
    def _extract_disambiguation(sql: str, rows: list, columns: list[str]) -> tuple[str, list[dict]]:
        """
        If `columns` matches the people-disambiguation shape (user_id + name),
        extract the searched name from `sql` and the (name, user_id) candidate
        pairs from `rows`. Returns ("", []) otherwise.

        Shared by every ConversationTurn construction site so the STOP-guard /
        exact-match / origin-question context (see build_context_block) works
        regardless of which code path produced the disambiguation result —
        previously only the main _finish() path populated these fields, so a
        disambiguation result reaching the user via the final_answer branch or
        a pagination replay silently dropped this context on the next turn.
        """
        if not is_disambiguation_result(columns, sql):
            return "", []

        # Anchor to the first_name/last_name reference, then take the NEXT ILIKE
        # after it — a plain "first ILIKE anywhere in the SQL" search can grab an
        # unrelated earlier filter (e.g. a facility-name ILIKE) when the query
        # combines a name check with other conditions.
        name_col_pos = re.search(r"first_name|last_name", sql, re.IGNORECASE)
        search_from = sql[name_col_pos.start():] if name_col_pos else sql
        m = re.search(r"ILIKE\s+'%([^%]+)%'", search_from, re.IGNORECASE)
        name = m.group(1).strip() if m else "?"

        candidates = []
        for row in rows[:20]:
            if not isinstance(row, dict):
                continue
            nm, uid = row.get("name"), row.get("user_id")
            if nm and uid and _UUID_RE.match(str(uid).strip()):
                candidates.append({"name": str(nm).strip(), "user_id": str(uid).strip()})
        return name, candidates


    @staticmethod
    def _extract_entity_disambiguation(sql: str, rows: list, columns: list[str]) -> tuple[str, list[dict], str, str]:
        """
        If `columns`/`sql` match the STEP 1 entity count-check shape (id +
        name + a count column, ILIKE on a known entity table — see
        is_entity_disambiguation_result) produced by the AMBIGUOUS REFERENCE
        RULE for "at <facility/project/client/organisation>"-style queries,
        extract the searched name, the (name, count) candidate rows, and the
        table/alias that was searched.

        Domain-neutral: is_entity_disambiguation_result() returns whichever
        count column the query actually used (inspection_count for
        inspection, schedule_count/audit_count for audit — see that
        function's docstring), stored here under the neutral key "count" so
        this works for either domain rather than assuming inspection's name.

        Without this, an entity disambiguation list ("4 facilities match
        'adnoc'") is never remembered: the generic facility_names extraction
        in _finish() only looks for a "facility_name"/"facility" column, which
        this template doesn't produce (it selects bare "name"). So when the
        user later names one entity from that list, that turn runs no new
        SQL of its own and active_subject_type is never set — the scope
        silently vanishes on the very next follow-up question.
        """
        entity = is_entity_disambiguation_result(columns, sql)
        if entity is None:
            return "", [], "", ""
        table, alias, count_col = entity

        m = re.search(rf"{re.escape(alias)}\.name\s+ilike\s+'%([^%]+)%'", sql, re.IGNORECASE)
        name = m.group(1).strip() if m else "?"

        candidates = []
        for row in rows[:20]:
            if not isinstance(row, dict):
                continue
            nm = row.get("name")
            if nm:
                candidates.append({
                    "name": str(nm).strip(),
                    "count": row.get(count_col) or 0,
                    "id": str(row.get("id")).strip() if row.get("id") else "",
                })
        return name, candidates, table, alias


    def _disambiguation_followup_message(self, question: str, sql: str, exec_data: dict) -> str:
        """
        Build a STEP-2a/2b-specific follow-up message for a disambiguation STEP 1
        result, instead of the generic "If this answers it, call final_answer"
        message — which previously caused Llama to statistically summarize the
        disambiguation table (MAX/SUM of inspection_count) instead of recognizing
        2+ rows as the STEP 2b "which one did you mean?" trigger.

        Branches on `is_disambiguation_result` (people: user_id+name), not on
        "inspection_count" presence — both real people templates can include
        inspection_count too, so that heuristic would misroute a people result
        into the entity/facility wording below. Callers gate on
        `is_disambiguation_result(...) or is_entity_disambiguation_result(...)`
        before calling this, so exactly one of the two branches applies.
        """
        rows = exec_data.get("rows", [])
        columns = exec_data.get("columns", [])
        m = re.search(r"ilike '%(.+?)%'", sql, re.IGNORECASE)
        name_filter = m.group(1) if m else "?"

        if not is_disambiguation_result(columns, sql):
            # Entity template (facility/project/client/organisation) — only
            # candidates with a non-zero count are useful answers to "show
            # <records> at X"; zero-count entities are real name matches but
            # not relevant choices. Domain-neutral wording ("records") since
            # what's being counted varies (inspections for the inspection
            # domain, schedules/audits for the audit domain) — see
            # _extract_entity_disambiguation's docstring for the count-column
            # generalization this relies on.
            # IMPORTANT: `rows` here are the RAW execute_sql result rows, keyed
            # by whatever column name the SQL itself used (schedule_count,
            # audit_count, inspection_count, ...) — NOT the "count" key that
            # only exists in the already-normalized entity_disambiguation_candidates
            # dicts _extract_entity_disambiguation builds. Must read via the
            # actual count_col name, not a hardcoded literal.
            _entity = is_entity_disambiguation_result(columns, sql)
            if _entity:
                table, alias, count_col = _entity
            else:
                table, alias, count_col = "facility", "fac", "count"
            table_plural = table[:-1] + "ies" if table.endswith("y") else table + "s"
            nonzero = [r for r in rows if (r.get(count_col) or 0) > 0]
            if len(nonzero) == 0:
                return (
                    f"STEP 1 disambiguation check found {len(rows)} {table_plural} matching "
                    f"'{name_filter}', but NONE have any related records.\n"
                    f"Original question: \"{question}\"\n"
                    f"Call final_answer stating no matching records were found at any {table} "
                    f"matching '{name_filter}'."
                )
            elif len(nonzero) == 1:
                r = nonzero[0]
                return (
                    f"STEP 1 disambiguation check: exactly ONE {table} matching "
                    f"'{name_filter}' has related records — '{r.get('name')}' "
                    f"({r.get(count_col)} records).\n"
                    f"Original question: \"{question}\"\n"
                    f"This is STEP 2a — call generate_sql again using "
                    f"{alias}.name ILIKE '%{_ilike_escape(str(r.get('name') or ''))}%' to filter by this {table}. "
                    f"Do NOT use {alias}.id UUID — the SQL model ignores UUID filters "
                    f"and will generate ILIKE anyway. Use the exact name directly."
                )
            else:
                options = "\n".join(
                    f"  • {r.get('name')} ({r.get(count_col)} records)"
                    for r in nonzero
                )
                return (
                    f"STEP 1 disambiguation check: {len(nonzero)} {table_plural} matching "
                    f"'{name_filter}' have related records (out of {len(rows)} total matches).\n"
                    f"Original question: \"{question}\"\n"
                    f"This is STEP 2b — call final_answer NOW with EXACTLY this format "
                    f"(do not compute totals, leaders, sums, or any other statistic; do "
                    f"not call generate_sql again):\n"
                    f"\"I found {len(nonzero)} {table_plural} matching '{name_filter}':\n"
                    f"{options}\n"
                    f"Which one did you mean?\""
                )
        else:
            # People template — list all matched names
            if len(rows) == 0:
                return (
                    f"STEP 1 disambiguation check found NO users matching "
                    f"'{name_filter}'.\n"
                    f"Original question: \"{question}\"\n"
                    f"Call final_answer stating no matching person was found."
                )
            elif len(rows) == 1:
                r = rows[0]
                _name_parts = str(r.get('name', '')).split()
                _first = _name_parts[0] if _name_parts else name_filter
                _last = _name_parts[-1] if len(_name_parts) > 1 else ''
                return (
                    f"STEP 1 disambiguation check: exactly ONE person matches "
                    f"'{name_filter}' — '{r.get('name')}'.\n"
                    f"Original question: \"{question}\"\n"
                    f"This is STEP 2a — call generate_sql again filtered by name: "
                    f"WHERE u.first_name ILIKE '%{_ilike_escape(_first)}%' AND u.last_name ILIKE '%{_ilike_escape(_last)}%' "
                    f"(joined via inspector_user_id or inspectee_user_id per the question's "
                    f"role wording). Do NOT use u.id UUID — the SQL model ignores UUID "
                    f"filters and will generate ILIKE anyway."
                )
            else:
                options = "\n".join(f"  • {r.get('name')}" for r in rows)
                return (
                    f"STEP 1 disambiguation check: {len(rows)} people match "
                    f"'{name_filter}'.\n"
                    f"Original question: \"{question}\"\n"
                    f"This is STEP 2b — call final_answer NOW with EXACTLY this format "
                    f"(do not pick one, do not compute statistics, do not call "
                    f"generate_sql again):\n"
                    f"\"I found {len(rows)} people matching '{name_filter}':\n"
                    f"{options}\n"
                    f"Which one did you mean?\""
                )


    def _is_preparatory_result(self, question: str, exec_result: dict, sql: str = "") -> bool:
        q_lower = question.lower()
        data = exec_result.get("result", {})
        columns = [c.lower() for c in data.get("columns", [])]
        row_count = data.get("row_count", 0)

        # AMBIGUOUS REFERENCE RULE STEP 1 check (people OR entity shape).
        # Without this, STEP 1 results render directly as a table and Llama
        # never gets to branch into STEP 2a/2b — the disambiguation flow
        # silently becomes a dead-end table dump.
        if is_disambiguation_result(columns, sql) or is_entity_disambiguation_result(columns, sql):
            return True

        # A result whose only columns are bare identifier lookups (e.g.
        # "cycle_id", "id", "portfolio_id") is virtually always an
        # intermediate ordinal/name-resolution step — the reasoning model
        # asking "what's this row's actual ID" before it can filter a
        # follow-up query — never a genuine answer to a question asking to
        # list/show entity data. This has to be checked before the
        # wants_content/is_superlative heuristics below, which are both
        # keyword-gated on the ORIGINAL question text and can miss this
        # entirely when the question names an entity (e.g. "portfolios")
        # that isn't one of their trigger words.
        #
        # Confirmed live 2026-08-03 (qid=f13ab827): "list all portfolios
        # submitted for the 4th one in the list" — after two failed attempts
        # at a combined query (hallucinated acyc.cycle_number), the reasoning
        # model fell back to just resolving the ordinal (`SELECT id AS
        # cycle_id FROM audit_cycle ... LIMIT 1 OFFSET 3`, its own thought
        # said "I need to first get the 4th cycle's ID... then use it in a
        # separate query"). Neither wants_content ("question/label/answer/
        # response/what...in" — none appear) nor is_superlative ("first/
        # last/latest/..." — none appear either) fired, so the loop treated
        # this single-column, single-row cycle ID as the final answer and
        # never issued the follow-up query the model itself said it needed.
        _id_only_columns = bool(columns) and all(
            c == "id" or c.endswith("_id") or c.endswith("_uuid") for c in columns
        )
        if _id_only_columns and row_count <= 5:
            return True

        # The question names a specific entity table ("audit plans",
        # "portfolios", ...) but the executed SQL never touches that table
        # at all — a strong signal the query only resolved an ORDINAL
        # REFERENCE ("the 6th auditor") and stopped there, never taking the
        # second step the reasoning model's own plan called for. Unlike the
        # bare-id-only check above, this catches the case where the
        # preparatory result ALSO includes a real display column (e.g. a
        # person's name) — not just raw ids — so it doesn't look
        # column-empty enough for that check to fire.
        #
        # Confirmed live 2026-08-03 (qid=081c7a81): "list all audit plans
        # under the 6th auditor from the list" — the model's own thought
        # said "I need to first identify who the 6th auditor is... Then
        # I'll get the audit plans for that auditor", but only ever issued
        # the first half (`SELECT u.id, u.first_name || ' ' || u.last_name
        # AS name FROM users u ... LIMIT 1 OFFSET 5` — no audit_plan
        # anywhere). Result: columns=['id','name','name__id'], 1 row — not
        # id-only (has a real 'name' column), so the check above missed it,
        # and the loop returned "the 6th auditor's name" as if it were "the
        # audit plans under them".
        sql_lower = sql.lower()
        for _pat, _table, _etype in _entity_noun_tables(getattr(self, "domain", None)):
            if _table is None:
                continue  # type-only entry (e.g. auditor/inspector) — see _entity_noun_tables
            if re.search(_pat, q_lower) and _table not in sql_lower:
                return True

        wants_content = bool(re.search(
            r'\b(question|label|answer|response|what.*in)\b', q_lower))
        has_content = any(c in columns for c in [
            # Actual DB column names — cause/correction/status/progress_stage/
            # responsible/target_close_out_date are shared literally between
            # inspection_corrective_action and audit_corrective_action, so
            # this list already covers most audit CA results without a
            # separate audit branch.
            "label", "answer_text", "answer_numeric",
            "inspection_score", "total_inspection_hours",
            "cause", "correction", "corrective_action", "responsible",
            "status", "progress_stage", "capex", "opex",
            "target_close_out_date", "schedule_date", "due_date",
            # Common SQL aliases the LLM uses
            "question", "answer", "question_label", "answer_value",
            "observation", "finding", "score", "risk_level",
            "inspector_name", "facility_name", "inspection_type_name",
            # Audit-domain equivalents without an inspection-side namesake
            "audit_score", "schedule_status", "auditor_name", "auditee_name",
            "compliance_name", "impact_name", "audit_type_name",
        ])

        if wants_content and not has_content and row_count > 0:
            return True

        is_superlative = bool(re.search(
            r'\b(most\s+recent|recently|latest|newest|oldest|first|last)\b', q_lower))
        wants_details = bool(re.search(
            r'\b(question|what|show|list|answer|submission|response|score)\b', q_lower))
        if is_superlative and wants_details and not has_content:
            return True

        return False
