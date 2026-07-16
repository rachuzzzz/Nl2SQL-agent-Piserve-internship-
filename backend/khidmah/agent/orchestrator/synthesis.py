"""
SynthesisMixin: turns a completed tool-call trace into a final natural-
language answer. Extracted verbatim from the original
agent/orchestrator.py.
"""

import json
import logging
import re
import time

_log_query = logging.getLogger("inspectai.query")
_log_llm   = logging.getLogger("inspectai.llm")

from sqlalchemy import text as sql_text
from llama_index.core.llms import ChatMessage, MessageRole

from khidmah.agent.prompts import FORCE_SUMMARY_PROMPT
from khidmah.core.disambiguation import is_disambiguation_result

from khidmah.agent.orchestrator.models import AgentStep, ConversationTurn, AgentResult
from khidmah.agent.orchestrator.debug import _ilike_escape, _has_uuid_leak, _dbg, _detect_truncation


class SynthesisMixin:

    # ------------------------------------------------------------------ #
    # Finish helper
    # ------------------------------------------------------------------ #

    def _finish(self, question, answer, steps, iteration, emit_fn=None,
                stats=None, session=None, query_start=None):
        s = AgentStep(iteration, "[auto] synthesized", "final_answer",
                      {"answer": answer}, {"answer": answer}, 0)
        steps.append(s)
        if emit_fn: emit_fn(s)

        if stats and query_start:
            stats.total_wall_ms = int((time.time() - query_start) * 1000)
            for step in steps:
                if step.tool == "execute_sql":
                    stats.total_sql_exec_ms += step.duration_ms

        last_sql, last_total, last_cols = "", 0, []
        last_rows = []
        for step in reversed(steps):
            if step.tool == "execute_sql" and step.result.get("success"):
                data = step.result.get("result", {})
                last_sql = step.args.get("sql", "")
                last_total = data.get("total_count", data.get("row_count", 0))
                last_cols = data.get("columns", [])
                last_rows = data.get("rows", [])
                break

        # Also check get_answers tool results
        if not last_rows:
            for step in reversed(steps):
                if step.tool == "get_answers" and step.result.get("success"):
                    data = step.result.get("result", {})
                    if isinstance(data, dict):
                        last_rows = data.get("rows", [])
                    break

        # Extract key entities from result rows for follow-up context
        insp_ids, fac_names, insp_names, cycle_ids, user_ids = [], [], [], [], []
        facility_ids = []
        single_row_vals = {}
        _UUID_PAT = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )

        for row in last_rows[:20]:
            if not isinstance(row, dict):
                continue
            for col in ("inspection_id",):
                v = row.get(col)
                if v and str(v) not in insp_ids:
                    insp_ids.append(str(v))
            for col in ("facility_name", "facility"):
                v = row.get(col)
                if v and str(v) not in fac_names:
                    fac_names.append(str(v))
            # Capture resolved facility UUID when the query happens to select
            # it (mirrors the user_id capture below) — lets build_context_block
            # prefer an exact fac.id filter over ILIKE on the name.
            for col in ("facility_id", "fac_id"):
                v = row.get(col)
                if v and str(v) not in facility_ids and _UUID_PAT.match(str(v).strip()):
                    facility_ids.append(str(v).strip())
            for col in ("inspector_name", "inspector"):
                v = row.get(col)
                if v and str(v) not in insp_names:
                    insp_names.append(str(v))
            for col in ("cycle_id", "inspection_cycle_id"):
                v = row.get(col)
                if v and str(v) not in cycle_ids and _UUID_PAT.match(str(v).strip()):
                    cycle_ids.append(str(v).strip())
            # Capture resolved user UUID from disambiguation queries (SELECT u.id AS user_id),
            # also matching the inspector_user_id FK alias used elsewhere in the schema.
            for col in ("user_id", "inspector_user_id"):
                v = row.get(col)
                if v and str(v) not in user_ids and _UUID_PAT.match(str(v).strip()):
                    user_ids.append(str(v).strip())

        # get_answers result rows don't have inspection_id as a column — pull it from the tool args
        if not insp_ids:
            for step in reversed(steps):
                if step.tool == "get_answers" and step.args:
                    _ga_insp = step.args.get("inspection_id")
                    if _ga_insp and str(_ga_insp) not in insp_ids:
                        insp_ids.append(str(_ga_insp))
                    break

        # If only 1 row, store all values for maximum context
        if len(last_rows) == 1 and last_rows[0]:
            single_row_vals = {
                k: str(v) for k, v in last_rows[0].items()
                if v is not None and str(v).strip()
            }
            # Also pull cycle_id from single-row result
            for col in ("cycle_id", "inspection_cycle_id"):
                v = last_rows[0].get(col)
                if v and str(v) not in cycle_ids and _UUID_PAT.match(str(v).strip()):
                    cycle_ids.append(str(v).strip())

        # ── Determine active subject for this turn ───────────────────────────
        # Priority: cycle > single inspector > single inspection > single facility
        active_subject_type = ""
        active_subject_value = ""
        active_subject_label = ""
        active_subject_table = ""
        active_subject_alias = ""

        if len(cycle_ids) == 1:
            active_subject_type = "cycle"
            active_subject_value = cycle_ids[0]
            active_subject_label = cycle_ids[0]
        elif len(insp_names) == 1:
            active_subject_type = "inspector"
            active_subject_value = insp_names[0]
            active_subject_label = insp_names[0]
        elif len(insp_ids) == 1:
            active_subject_type = "inspection"
            active_subject_value = insp_ids[0]
            active_subject_label = insp_ids[0]
        elif len(fac_names) == 1:
            active_subject_type = "facility"
            active_subject_value = fac_names[0]
            active_subject_label = fac_names[0]
        # If previous turn had a scoped subject (cycle or facility) and this result
        # has no conflicting subject (e.g. a COUNT query with no facility_name column),
        # inherit the scope so it persists across multi-turn drilldowns.
        elif session and session.turns:
            _prev_turn = session.turns[-1]
            # This turn ran no new SQL that names a single entity (e.g. it's
            # just "Adnoc Schools- Ghayathi" answering "which one did you
            # mean?"). If it's an exact pick from the PREVIOUS turn's entity
            # disambiguation list, resolve the scope directly instead of
            # falling through to "no active subject". Facility keeps the
            # dedicated "facility" type (tested wording); project/client/
            # organisation use the generic "entity" type, carrying their
            # table/alias so build_context_block can build a correct filter.
            if _prev_turn.entity_disambiguation_candidates:
                _ed_q = re.sub(r"[?.!,;:]+$", "", question.strip()).strip().lower()
                _ed_matches = [
                    c for c in _prev_turn.entity_disambiguation_candidates
                    if c.get("name", "").strip().lower() == _ed_q
                ]
                if len(_ed_matches) == 1:
                    if _prev_turn.entity_disambiguation_table == "facility":
                        active_subject_type = "facility"
                    else:
                        active_subject_type = "entity"
                        active_subject_table = _prev_turn.entity_disambiguation_table
                        active_subject_alias = _prev_turn.entity_disambiguation_alias
                    active_subject_value = _ed_matches[0]["name"]
                    active_subject_label = _ed_matches[0]["name"]
            if not active_subject_type:
                for prev_turn in reversed(session.turns):
                    if prev_turn.active_subject_type and prev_turn.active_subject_value:
                        active_subject_type = prev_turn.active_subject_type
                        active_subject_value = prev_turn.active_subject_value
                        active_subject_label = prev_turn.active_subject_label
                        active_subject_table = prev_turn.active_subject_table
                        active_subject_alias = prev_turn.active_subject_alias
                        break

        if session is not None:
            # MT-03 fix: if we have a facility name but no inspection_id (e.g. MT-01
            # returned only facility_name without ir.inspection_id in SELECT), do a quick
            # DB lookup so the follow-up turn can filter by inspection_id rather than
            # falling back to the worked-example placeholder "INS-EXAMPLE-001".
            if fac_names and not insp_ids and len(last_rows) <= 3:
                try:
                    fac_escaped = _ilike_escape(fac_names[0])
                    _lookup = (
                        "SELECT ir.inspection_id FROM inspection_report ir "
                        "JOIN facility fac ON ir.facility_id = fac.id "
                        f"WHERE fac.name ILIKE '%{fac_escaped}%' "
                        "AND ir.status != 'DRAFT' "
                        "ORDER BY ir.submitted_on DESC LIMIT 1;"
                    )
                    with self.db_engine.connect() as _conn:
                        _row = _conn.execute(sql_text(_lookup)).fetchone()
                    if _row and _row[0]:
                        insp_ids = [str(_row[0])]
                        _dbg(self._debug, "FINISH — inspection_id lookup",
                             f"facility={fac_names[0]} → {insp_ids[0]}", 0)
                except Exception:
                    pass  # Silently skip — follow-up will have less context

            # Detect if this turn was a disambiguation query (user_id + name cols)
            _disambig_name, _disambig_candidates = self._extract_disambiguation(
                last_sql, last_rows, last_cols
            )
            _is_disambig = bool(_disambig_name)

            # Detect if this turn was an entity disambiguation query (id + name
            # + inspection_count cols) so a later exact pick from this list can
            # resolve active_subject_type (see the elif branch above).
            _, _ent_disambig_candidates, _ent_disambig_table, _ent_disambig_alias = (
                self._extract_entity_disambiguation(last_sql, last_rows, last_cols)
            )

            # Carry forward the ORIGINAL question that started this disambiguation
            # chain. Without this, a multi-turn narrowing sequence ("list schedules
            # for neenu" → too many → "neenu qa inspector") loses track of the real
            # goal (schedules) once the model is a couple of turns deep into just
            # resolving the name, and answers "here's the person" instead of
            # actually running the original request.
            _disambig_origin_question = ""
            if _is_disambig:
                _prev_last = session.turns[-1] if session.turns else None
                if _prev_last and _prev_last.disambiguation_name:
                    _disambig_origin_question = (
                        _prev_last.disambiguation_origin_question or _prev_last.question
                    )
                else:
                    _disambig_origin_question = question

            session.add_turn(ConversationTurn(
                question=question,
                answer=answer[:300],
                tool_used=steps[-2].tool if len(steps) >= 2 else "unknown",
                sql=last_sql,
                total_count=last_total,
                columns=last_cols,
                inspection_ids=insp_ids[:5],
                facility_names=fac_names[:5],
                inspector_names=insp_names[:5],
                cycle_ids=cycle_ids[:3],
                user_ids=user_ids[:5],
                facility_ids=facility_ids[:5],
                disambiguation_name=_disambig_name,
                disambiguation_candidates=_disambig_candidates,
                disambiguation_origin_question=_disambig_origin_question,
                entity_disambiguation_candidates=_ent_disambig_candidates,
                entity_disambiguation_table=_ent_disambig_table,
                entity_disambiguation_alias=_ent_disambig_alias,
                single_row_values=single_row_vals,
                active_subject_type=active_subject_type,
                active_subject_value=active_subject_value,
                active_subject_label=active_subject_label,
                active_subject_table=active_subject_table,
                active_subject_alias=active_subject_alias,
            ))

        _final_stats = stats or QueryStats()
        _log_query.debug("[QUERY] _finish sub=? iter=%d wall=%dms llm=%dms sql=%dms "
                         "llm_calls=%d prompt_tok=%d completion_tok=%d",
                         iteration,
                         _final_stats.total_wall_ms,
                         _final_stats.total_llm_latency_ms,
                         _final_stats.total_sql_exec_ms,
                         _final_stats.total_llm_calls,
                         _final_stats.total_prompt_tokens,
                         _final_stats.total_completion_tokens)
        return AgentResult(question, answer, steps, iteration, True, _final_stats)


    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    _COUNT_RE = re.compile(
        r'\b(how\s+many|count|number\s+of|total)\b', re.IGNORECASE)


    def _synthesize(self, question, tool_name, tool_result, max_display_rows=20, sql=""):
        data = tool_result.get("result", {})
        q_lower = question.lower()

        # ---- list_forms ----
        if tool_name == "list_forms" and isinstance(data, list):
            if self._COUNT_RE.search(q_lower):
                return f"There are {len(data)} forms."
            if not data:
                return "No forms found."
            lines = [f"Found {len(data)} form(s):"]
            for i, f in enumerate(data, 1):
                name = f.get("form_name", f.get("module_name", "(unnamed)"))
                qcount = f.get("question_count", "?")
                lines.append(f"{i}. {name} ({qcount} questions)")
            return "\n".join(lines)

        # ---- semantic_search ----
        if tool_name == "semantic_search" and isinstance(data, list):
            if not data:
                return "No questions found matching that topic."
            if self._COUNT_RE.search(q_lower):
                forms = sorted({m["form_name"] for m in data if m.get("form_name")})
                return f"Found {len(data)} related question(s) across {len(forms)} form(s): {forms}"
            lines = [f"Found {len(data)} related question(s):"]
            for i, m in enumerate(data, 1):
                lines.append(f"  {i}. \"{m.get('question_label', m.get('label','?'))}\" (form: {m['form_name']}, score: {m['score']})")
            return "\n".join(lines)

        # ---- search_questions ----
        if tool_name == "search_questions" and isinstance(data, list):
            if not data:
                return "No matching questions found."
            if self._COUNT_RE.search(q_lower):
                return f"Found {len(data)} matching question(s)."
            lines = [f"Found {len(data)} question(s):"]
            for i, q in enumerate(data, 1):
                label = q.get("question_label", q.get("label", "?"))
                form = q.get("form_name", q.get("module_name", "?"))
                qtype = q.get("question_type", "")
                suffix = f" [{qtype}]" if qtype else ""
                lines.append(f"  {i}. {label}{suffix} — form: {form}")
            return "\n".join(lines)

        # ---- get_answers ----
        if tool_name == "get_answers" and isinstance(data, dict):
            rows = data.get("rows", [])
            truncated = data.get("truncated", False)
            total_count = data.get("total_count", len(rows))

            if not rows:
                return "No answers found matching the query."

            if truncated and total_count > 0:
                header = f"Found {total_count} answer(s) (showing first {len(rows)}):"
            elif truncated:
                header = f"Showing first {len(rows)} answer(s) (more may exist):"
            else:
                header = f"Found {len(rows)} answer(s):"

            lines = [header]
            for i, row in enumerate(rows[:max_display_rows], 1):
                q_label = row.get("question_label") or row.get("label") or "?"
                raw_ans = row.get("answer_value") or row.get("answer_text") or ""
                if not raw_ans and row.get("answer_numeric") is not None:
                    raw_ans = str(row["answer_numeric"])
                ans_val = raw_ans if raw_ans else "N/A"
                # Skip rows with no answer (file attachments, empty fields)
                if ans_val == "N/A" and row.get("score") is None:
                    continue
                insp = row.get("inspection_id", "")
                score = row.get("score")
                parts = [f"{q_label}: {ans_val}"]
                if score is not None:
                    parts.append(f"score: {score}")
                if insp:
                    parts.append(f"inspection: {insp}")
                lines.append(f"  {i}. {' | '.join(parts)}")

            if len(rows) > max_display_rows:
                lines.append(f"  ... and {len(rows) - max_display_rows} more in this batch")
            if truncated and total_count > len(rows):
                lines.append(f"  (Total matching: {total_count})")
            return "\n".join(lines)

        # ---- get_answer_stats ----
        if tool_name == "get_answer_stats" and isinstance(data, dict):
            avg = data.get("average_value")
            total = data.get("total_answers", 0)
            breakdown = data.get("value_breakdown", [])
            filters = data.get("filters", {})
            q_filter = filters.get("question_label") or ""
            parts = []

            if avg is not None:
                label = f"'{q_filter}'" if q_filter else "answers"
                source = data.get("stats_source", "")
                scored_count = data.get("scored_count", 0)
                source_note = ""
                if source == "score":
                    source_note = f" (from scoring system, {scored_count} scored answers)"
                elif source == "answer_numeric":
                    source_note = f" (from numeric answers)"
                parts.append(
                    f"Average score for {label}: {avg}{source_note}"
                    f" — min: {data.get('min_value')}, max: {data.get('max_value')}"
                )
            elif total > 0 and q_filter:
                parts.append(f"Total answers for '{q_filter}': {total} (no numeric/score data)")
            elif total > 0:
                parts.append(f"Total answers: {total}")

            if breakdown:
                bheader = f"Most common values for '{q_filter}':" if q_filter else "Most common values:"
                parts.append(bheader)
                for item in breakdown[:10]:
                    display_val = str(item["value"]) if item["value"] else "N/A"
                    parts.append(f"  • {display_val}: {item['count']} time(s)")

            return "\n".join(parts) if parts else "No answer statistics found."



        # ---- execute_sql ----
        if tool_name == "execute_sql" and isinstance(data, dict):
            rows = data.get("rows", [])
            cols = data.get("columns", [])
            truncated = data.get("truncated", False)
            total_count = data.get("total_count", len(rows))

            # Disambiguation result — user_id + name columns, searched by ILIKE
            # on first_name/last_name → format as choice list. Shared check
            # (core.disambiguation) so this agrees with fastapi_app.py's
            # dashboard-suppression logic for the same result.
            if is_disambiguation_result(cols, sql):
                if not rows:
                    return "No users found matching that name."
                name_col = next((c for c in cols if c.lower() == "name"), "name")
                count_col = next((c for c in cols if c.lower() in
                                  {"inspection_count", "count", "total"}), None)
                if total_count > 12:
                    return (
                        f"I found {total_count} people matching that name — too many to list. "
                        f"Could you provide a more specific full name or email address?"
                    )
                if total_count == 1:
                    row = rows[0]
                    name = row.get(name_col, "")
                    uid = row.get("user_id", "")
                    return f"Found exactly 1 match: **{name}** (user_id: `{uid}`). Proceeding with your request."
                # 2–12: list them as choices
                email_col = next((c for c in cols if c.lower() == "email"), None)
                lines = [f"I found {total_count} people matching that name:"]
                for r in rows[:12]:
                    name = r.get(name_col, "?")
                    parts = [f"  • {name}"]
                    if count_col and r.get(count_col) is not None:
                        parts.append(f"({r[count_col]} inspections)")
                    if email_col and r.get(email_col):
                        parts.append(f"— {r[email_col]}")
                    lines.append(" ".join(parts))
                lines.append("\nWhich one did you mean?")
                return "\n".join(lines)

            # Single-row aggregate
            if len(rows) == 1 and len(cols) <= 3:
                agg_names = {
                    "count", "total", "avg", "sum", "min", "max",
                    "average", "avg_score", "avg_inspection_score",
                    "total_capex", "total_opex", "question_count",
                    "answer_count", "report_count", "frequency",
                }
                col_lower = [c.lower() for c in cols]
                is_agg = all(
                    c in agg_names or c.startswith(("avg_", "sum_", "total_", "count_"))
                    or c.endswith(("_count", "_avg"))
                    for c in col_lower
                )
                if is_agg:
                    return "; ".join(f"{col}: {val}" for col, val in rows[0].items())

            # Two-column group-by
            if rows and len(cols) == 2:
                col_lower = [c.lower() for c in cols]
                has_count = any(
                    c in {"count", "total", "frequency", "improvement"}
                    or c.endswith("_count")
                    or c.startswith("num_")
                    for c in col_lower
                )
                if has_count:
                    count_idx = next(
                        i for i, c in enumerate(col_lower)
                        if c in {"count", "total", "frequency", "improvement"}
                        or c.endswith("_count")
                        or c.startswith("num_")
                    )
                    label_idx = 1 - count_idx
                    lines = []
                    for r in rows:
                        vals = list(r.values())
                        lines.append(f"  • {vals[label_idx]}: {vals[count_idx]}")
                    header = f"Results ({len(rows)} groups"
                    if truncated:
                        header += f", showing first {len(rows)} of {total_count}"
                    return header + "):\n" + "\n".join(lines)

            # General multi-row
            if rows:
                # Check for UUID leaks in name columns before displaying.
                # If detected, return an error that triggers the LLM to fix the JOIN.
                uuid_leaking = _has_uuid_leak(rows, cols)
                if uuid_leaking:
                    return (
                        f"QUERY ERROR: Raw UUID values in column(s) {uuid_leaking} "
                        f"instead of human-readable names. "
                        f"You must JOIN the lookup table. Example: "
                        f"facility_id → JOIN facility fac ON ir.facility_id = fac.id "
                        f"→ SELECT fac.name AS facility_name. "
                        f"Regenerate SQL with correct JOINs."
                    )

                if truncated and total_count > 0:
                    header = f"Found {total_count} result(s) (showing first {len(rows)}):"
                elif truncated:
                    header = f"Showing first {len(rows)} result(s) (more exist):"
                else:
                    header = f"Found {len(rows)} result(s):"

                # Detect constant-value columns (same NON-NULL value on every row) and
                # promote them to the header instead of repeating per row.
                # Exclude columns where the single value is None/"" — those are just
                # missing data (e.g. LAG() producing NULL previous_score) not true constants.
                constant_cols = {}
                _HIDE_COLS = {"element_id", "source_row_id", "source_table",
                              "module_id", "facility_id", "project_id",
                              "client_id", "inspector_user_id", "inspectee_user_id",
                              "inspection_report_id", "question_id",
                              "inspection_type_id", "inspection_sub_type_id",
                              "cycle_id", "schedule_id", "entity_id",
                              "user_id"}
                displayable_cols = [c for c in rows[0].keys() if c not in _HIDE_COLS]
                if len(rows) > 1:
                    for col in displayable_cols:
                        vals = {str(r.get(col)) for r in rows}
                        val_str = str(rows[0].get(col) or "")
                        # Only promote if: same across all rows AND not empty/None AND
                        # not a meaningful data column that happens to have one value
                        if (len(vals) == 1
                                and col not in ("inspection_id", "submitted_on")
                                and val_str not in ("", "None", "N/A")):
                            constant_cols[col] = val_str
                if constant_cols:
                    const_str = ", ".join(f"{k}={v}" for k, v in constant_cols.items())
                    header += f" ({const_str})"

                lines = [header]
                for i, row in enumerate(rows[:max_display_rows], 1):
                    parts = []
                    for col, val in row.items():
                        if col in _HIDE_COLS:
                            continue
                        # Skip columns promoted to header (constant across all rows)
                        if col in constant_cols:
                            continue
                        if val is None:
                            if len(displayable_cols) <= 2:
                                parts.append(f"{col}: N/A")
                            continue
                        val_str = str(val)
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                        parts.append(f"{col}: {val_str}")
                    lines.append(f"  {i}. {' | '.join(parts) or '(no data)'}")

                if len(rows) > max_display_rows:
                    lines.append(f"  ... and {len(rows) - max_display_rows} more shown")
                if truncated and total_count > len(rows):
                    lines.append(f"  (Total matching: {total_count})")
                return "\n".join(lines)

            return "The query returned no results."

        return self._llm_answer(question, tool_name, data)


    def _llm_answer(self, question, tool_name, data, extra=""):
        prompt = (
            f"Answer the question using ONLY the data provided. "
            f"Do not estimate or invent. If data is empty, say so.\n"
            f"{extra}\n\nQuestion: {question}\n\n"
            f"Data ({tool_name}):\n{json.dumps(data, default=str, indent=2)}\n\n"
            f"Give a concise, accurate answer. No JSON."
        )
        try:
            return str(self.reasoning_llm.complete(prompt)).strip()
        except Exception:
            return json.dumps(data, default=str)


    # ------------------------------------------------------------------ #
    # Context messages for non-terminal tool results
    # ------------------------------------------------------------------ #

    def _context_msg(self, tool_name, tool_result, question=""):
        if not tool_result.get("success"):
            error = tool_result.get("error", "unknown")

            if tool_name == "generate_sql":
                q_lower = question.lower()
                hint = ""
                if re.search(r'\b(mitigative.?action|cause|correction|capex|opex|overdue)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK SQL:\n"
                        "SELECT corrective_action_id, cause, corrective_action, responsible, status\n"
                        "FROM inspection_corrective_action WHERE status = 'OPEN' LIMIT 100;"
                    )
                elif re.search(r'\b(inspection.?score|hour|inspector)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK SQL:\n"
                        "SELECT inspection_score, status, submitted_on\n"
                        "FROM inspection_report WHERE status != 'DRAFT' LIMIT 50;"
                    )
                elif re.search(r'\b(answer|response|submission)\b', q_lower):
                    hint = (
                        "\n\nFALLBACK: Use get_answers tool or this SQL:\n"
                        "SELECT aq.label, aa.answer_text, aa.answer_numeric, aa.inspection_id\n"
                        "FROM ai_answers aa\n"
                        "LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
                        "WHERE aq.label ILIKE '%KEYWORD%' LIMIT 50;"
                    )
                return f"generate_sql failed: {error}{hint}"

            if tool_name == "get_answers":
                return (
                    f"get_answers failed: {error}\n"
                    f"FALLBACK: Use execute_sql with:\n"
                    f"SELECT aq.label, aa.answer_text, aa.answer_numeric, aa.inspection_id\n"
                    f"FROM ai_answers aa LEFT JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
                    f"WHERE aa.module_name ILIKE '%MODULE%' LIMIT 50;"
                )

            return f"Tool '{tool_name}' failed: {error}. Try a different approach."

        # Successful non-terminal results
        if tool_name == "semantic_search":
            data = tool_result.get("result", [])
            if data:
                fmt = "\n".join(
                    f"  - \"{m.get('question_label', m.get('label','?'))}\" (form: {m['form_name']}, score: {m['score']})"
                    for m in data[:5]
                )
                asks_answers = bool(re.search(
                    r'\b(answer|response|submission|filled|submitted|responded)\b',
                    question.lower()))
                if asks_answers:
                    top_form = data[0].get("form_name", data[0].get("module_name",""))
                    top_q = data[0].get("question_label", data[0].get("label",""))
                    return (
                        f"semantic_search found {len(data)} match(es):\n{fmt}\n\n"
                        f"User asked about answers. Call get_answers with "
                        f"module_name='{top_form}' and question_label='{top_q}'."
                    )
                return (
                    f"semantic_search found {len(data)} match(es):\n{fmt}\n\n"
                    f"If this answers the question, call final_answer.\n"
                    f"To get submitted answers for these questions, call get_answers."
                )
            return "semantic_search found no matches. Call final_answer saying no forms ask about that topic."

        if tool_name == "search_questions":
            data = tool_result.get("result", [])
            if data:
                fmt = "\n".join(
                    f"  - \"{q.get('question_label', q.get('label','?'))}\" (module: {q.get('module_name','?')})"
                    for q in data[:6]
                )
                return (
                    f"search_questions found {len(data)} question(s):\n{fmt}\n\n"
                    f"To see submitted answers, call get_answers with question_label (aq.label) "
                    f"set to one of the labels above."
                )
            return "search_questions found no matches. Try a broader keyword or use semantic_search."

        return json.dumps(tool_result, default=str)


    def _schema_hint_for_error(self, error_msg):
        parts = []
        tm = re.search(r'relation "(\w+)" does not exist', error_msg)
        if tm:
            bad = tm.group(1)
            try:
                sr = self.registry.call("get_schema", {})
                tables = sr.get("result", {}).get("tables", [])
                ai_tables = [t for t in tables if t.startswith("ai_")]
                insp_tables = [t for t in tables if t.startswith("inspection_")]
                parts.append(
                    f"Table '{bad}' doesn't exist.\n"
                    f"AI tables: {ai_tables}\n"
                    f"Inspection tables: {insp_tables}"
                )
            except Exception:
                pass
        cm = re.search(r'column "(\w+)" does not exist', error_msg)
        if cm and not tm:
            parts.append(f"Column '{cm.group(1)}' doesn't exist. Use get_schema to check columns.")
        return "\n".join(parts) + "\n" if parts else ""


    def _call_llm(self, messages, stats=None, iteration=0):
        t0 = time.time()
        response = self.reasoning_llm.chat(messages)
        latency_ms = int((time.time() - t0) * 1000)
        content = response.message.content.strip()
        _prompt_tok = 0
        _completion_tok = 0
        if stats:
            stats.total_llm_calls += 1
            stats.total_llm_latency_ms += latency_ms
            raw = getattr(response, "raw", {}) or {}
            _prompt_tok = raw.get("prompt_eval_count", 0)
            _completion_tok = raw.get("eval_count", 0)
            stats.total_prompt_tokens += _prompt_tok
            stats.total_completion_tokens += _completion_tok

        _log_llm.debug("[LLM ] iter=%d latency=%dms prompt_tok=%d completion_tok=%d out_len=%d",
                       iteration, latency_ms, _prompt_tok, _completion_tok, len(content))

        # Debug logging (raw LLM output to separate file when DEBUG_LLM=1)
        if self._debug:
            trunc = _detect_truncation(content)
            trunc_note = f"\n⚠ {trunc}" if trunc else ""
            _dbg(self._debug,
                 f"LLAMA RAW OUTPUT [{latency_ms}ms]{trunc_note}",
                 content, iteration)

        return content

