"""
DisambiguationMixin: orchestrator-specific extraction/messaging glue
built on top of core.disambiguation's generic shape-detection utilities
-- tied to ConversationTurn and LLM prompt strings, so it stays separate
from that dependency-free module. Extracted verbatim from the original
agent/orchestrator.py.
"""

import re

from src.core.disambiguation import is_disambiguation_result, is_entity_disambiguation_result

from src.agent.orchestrator.debug import _ilike_escape, _UUID_RE


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
        name + inspection_count, ILIKE on a known entity table — see
        is_entity_disambiguation_result) produced by the AMBIGUOUS REFERENCE
        RULE for "at <facility/project/client/organisation>"-style queries,
        extract the searched name, the (name, inspection_count) candidate
        rows, and the table/alias that was searched.

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
        table, alias = entity

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
                    "inspection_count": row.get("inspection_count") or 0,
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
            # candidates with inspections > 0 are useful answers to "show
            # inspections at X"; zero-inspection entities are real matches
            # but not relevant choices.
            _entity = is_entity_disambiguation_result(columns, sql)
            table, alias = _entity if _entity else ("facility", "fac")
            table_plural = table[:-1] + "ies" if table.endswith("y") else table + "s"
            nonzero = [r for r in rows if (r.get("inspection_count") or 0) > 0]
            if len(nonzero) == 0:
                return (
                    f"STEP 1 disambiguation check found {len(rows)} {table_plural} matching "
                    f"'{name_filter}', but NONE have any inspections.\n"
                    f"Original question: \"{question}\"\n"
                    f"Call final_answer stating no inspections were found at any {table} "
                    f"matching '{name_filter}'."
                )
            elif len(nonzero) == 1:
                r = nonzero[0]
                return (
                    f"STEP 1 disambiguation check: exactly ONE {table} matching "
                    f"'{name_filter}' has inspections — '{r.get('name')}' "
                    f"({r.get('inspection_count')} inspections).\n"
                    f"Original question: \"{question}\"\n"
                    f"This is STEP 2a — call generate_sql again using "
                    f"{alias}.name ILIKE '%{_ilike_escape(str(r.get('name') or ''))}%' to filter by this {table}. "
                    f"Do NOT use {alias}.id UUID — the SQL model ignores UUID filters "
                    f"and will generate ILIKE anyway. Use the exact name directly."
                )
            else:
                options = "\n".join(
                    f"  • {r.get('name')} ({r.get('inspection_count')} inspections)"
                    for r in nonzero
                )
                return (
                    f"STEP 1 disambiguation check: {len(nonzero)} {table_plural} matching "
                    f"'{name_filter}' have inspections (out of {len(rows)} total matches).\n"
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

        wants_content = bool(re.search(
            r'\b(question|label|answer|response|what.*in)\b', q_lower))
        has_content = any(c in columns for c in [
            # Actual DB column names
            "label", "answer_text", "answer_numeric",
            "inspection_score", "total_inspection_hours",
            "cause", "correction", "corrective_action", "responsible",
            "status", "progress_stage", "capex", "opex",
            "target_close_out_date", "schedule_date", "due_date",
            # Common SQL aliases the LLM uses
            "question", "answer", "question_label", "answer_value",
            "observation", "finding", "score", "risk_level",
            "inspector_name", "facility_name", "inspection_type_name",
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
