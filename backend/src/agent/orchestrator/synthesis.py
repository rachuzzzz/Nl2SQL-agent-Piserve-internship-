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

from src.sql.result_shape import is_disambiguation_result
from src.api.response_formatting import strip_row_identity_columns

from src.agent.orchestrator.models import AgentStep, QueryStats, ConversationTurn, AgentResult
from src.agent.orchestrator.debug import _ilike_escape, _has_uuid_leak, _dbg, _detect_truncation
from src.llm.base import extract_token_counts


# Maps each ACTIVE CONTEXT scope type to the SQL keyword(s) that indicate
# THIS turn's own query still relates to it — used below to stop a stale
# scope from being silently re-affirmed onto an unrelated turn. Confirmed
# live: a facility scope set by one turn's incidental facility_name column
# (2 rows sharing a facility, neither about the facility itself) got
# silently re-stamped onto 5 unrelated intervening turns (observations,
# corrective actions, capex rankings — none referencing a facility at all)
# via the inheritance branch below, then misapplied to a 6th turn's global
# FM-manager ranking ("show top 5 of the same"), filtering it down to 1 row
# with no error. Deliberately scoped to the types with one fixed,
# unambiguous table name — "inspector"/"entity" aren't included, not
# confirmed to need the same gate and harder to pin to one keyword (person
# joins vary: inspector_user_id/auditor_id/auditee_id/...).
_SCOPE_RELEVANCE_KEYWORDS = {
    "facility": ("facility",),
    "cycle": ("cycle",),
    "audit": ("audit_report", "audit_plan", "audit_schedule"),
    "inspection": ("inspection_report",),
    "plan": ("audit_plan",),
}


def _sql_still_relevant_to_scope(sql: str, scope_type: str) -> bool:
    """True when it's safe to keep carrying this scope forward — either the
    type isn't gated (see _SCOPE_RELEVANCE_KEYWORDS) or this turn's own SQL
    shows some sign of relating to it. Empty SQL (no generate_sql this
    turn — e.g. a get_answers pagination turn) is treated as "can't tell,
    don't disturb it" — the existing pagination-continuation mechanism
    already owns that case elsewhere."""
    if not sql:
        return True
    keywords = _SCOPE_RELEVANCE_KEYWORDS.get(scope_type, ())
    if not keywords:
        return True
    sql_lower = sql.lower()
    return any(kw in sql_lower for kw in keywords)


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
        audit_ids = []
        facility_ids = []
        # audit_plan.title, under whatever alias the SQL happened to give it
        # (bare "title" for a plans-only SELECT, "audit_plan_title"/"plan_title"
        # etc. once it's joined to another table — e.g. review, for an
        # approval-status question). Confirmed live 2026-07-30: a 2-row
        # "approval status for the second one, who approved each level"
        # result (SELECT ap.title AS audit_plan_title, rev.level, ...) had NO
        # column literally named "title", so the OLD exact-match-only capture
        # missed it entirely — active_subject_type stayed unset for that turn,
        # a later non-ordinal follow-up ("who is the auditee for the audit
        # mentioned above") had no structured pointer to the plan actually in
        # focus, asked an unnecessary clarifying question, and its own retry
        # then drifted to a DIFFERENT, older plan from several turns back
        # instead of the one just discussed. Matched by suffix, not just
        # exact name, and independent of row_titles below (which mixes in
        # facility/person names too, not just the plan's own identifier).
        plan_titles = []
        # Real row ids paired with insp_ids/audit_ids/plan_titles/fac_names
        # above, captured via the SAME __id-suffixed sibling columns
        # inject_row_identity() (src/sql/validation/identity.py) attaches —
        # e.g. a column named "audit_id" (inspection_report/audit_report's
        # registered display column) gets a sibling "audit_id__id" holding
        # the real UUID. Lets active_subject_id below resolve to WHERE
        # x.id = '<uuid>' directly instead of re-deriving one from a
        # display value (which may not be unique — see plan_titles' own
        # docstring above for the confirmed live audit_plan.title case).
        insp_row_ids: list = []
        audit_row_ids: list = []
        plan_row_ids: list = []
        fac_row_ids: list = []
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
                    insp_row_ids.append(str(row.get(f"{col}__id", "") or ""))
            for col in ("audit_id",):
                v = row.get(col)
                if v and str(v) not in audit_ids:
                    audit_ids.append(str(v))
                    audit_row_ids.append(str(row.get(f"{col}__id", "") or ""))
            for col in row:
                if col == "title" or (isinstance(col, str) and col.endswith("_title")):
                    v = row.get(col)
                    if v and str(v) not in plan_titles:
                        plan_titles.append(str(v))
                        plan_row_ids.append(str(row.get(f"{col}__id", "") or ""))
                    break
            for col in ("facility_name", "facility"):
                v = row.get(col)
                if v and str(v) not in fac_names:
                    fac_names.append(str(v))
                    fac_row_ids.append(str(row.get(f"{col}__id", "") or ""))
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

        # Capture an ordered display-value list for MULTI-row results (e.g.
        # audit_plan.title) so a later "the first one"/"the second one"
        # follow-up can resolve positionally against what was JUST listed,
        # independent of active_subject_type below (which a multi-row list
        # like this often leaves unset — see the stale-scope-inheritance
        # comment further down). Preference order tries the row's own
        # IDENTIFIER columns first (audit_id, schedule_id, ...) before
        # descriptive/joined-in ATTRIBUTE columns (facility_name,
        # auditor_name, ...). Confirmed live 2026-07-30: "show all audits
        # part of this plan" selected both audit_id AND facility_name on the
        # same row (SELECT ap.title, asch.schedule_id, ..., fac.name AS
        # facility_name, ..., arep.audit_id, ...) — with facility_name
        # checked first, row_titles captured ['ADGM Square', 'Al Oyoun
        # Village', ...] instead of the audits themselves. The next turn,
        # "for the first one show all corrective actions", then silently
        # resolved "the first one" to "the first FACILITY" instead of "the
        # first AUDIT" — losing the plan-scoping filter entirely (a facility
        # can host schedules from OTHER plans too) rather than narrowing to
        # one specific audit under THIS plan. The grammatical subject of a
        # list question ("show all AUDITS", "list all PLANS") is what a bare
        # ordinal should bind to; a facility/auditor/inspector name is
        # usually a joined-in attribute describing that subject, not the
        # subject itself, so it must lose this race whenever both appear on
        # the same row.
        row_titles = []
        # Paired 1:1 with row_titles (same index = same row) via the __id
        # sibling inject_row_identity() attaches to whichever column matched
        # below — empty string when that query shape wasn't eligible for
        # injection (e.g. a GROUP BY). Lets an ordinal follow-up ("the third
        # one") resolve straight to WHERE x.id = '<uuid>' instead of
        # re-deriving a row from row_titles[idx]'s display string, which may
        # not be unique (see plan_titles' docstring above).
        row_ids = []
        # Paired 1:1 with row_ids — the COLUMN NAME the id actually filters
        # on, e.g. "id" for the usual __id-sibling-injected primary key, or
        # "submission_id" for an ai_audit_answers/ai_answers observation row
        # (see Priority 0 below). session.py's ORDINAL RESOLVED hint reads
        # this to phrase the right WHERE clause (<alias>.id vs
        # <alias>.submission_id) instead of always assuming a primary key.
        row_id_columns = []
        # One value for the WHOLE turn (not per row) — which entity type the
        # list actually is, set once from whichever priority branch below
        # fires first. Lets session.py's ordinal resolver require a named
        # noun ("the 6th AUDITOR") to match the type of list being resolved
        # against, instead of matching any list regardless of type — see
        # models.py's row_entity_type field docstring for the confirmed-live
        # bug this closes.
        row_entity_type = ""
        _ROW_ID_COL_TYPES = {
            "audit_id": "audit", "schedule_id": "schedule",
            "inspection_id": "inspection", "corrective_action_id": "corrective_action",
        }
        _ROW_NAME_COL_TYPES = {
            "facility_name": "facility", "auditor_name": "person", "inspector_name": "person",
        }
        # Capped at 20, matching max_display_rows' default — the same number
        # of rows the deterministic answer formatter (_synthesize_body) actually
        # shows the user. Was capped at 5 until 2026-07-31: live-confirmed
        # "list all audits" displayed 17 rows to the user, but "show
        # observations for the 10th one" had nothing to resolve against since
        # row_titles/row_ids only ever captured the first 5 — combined with
        # session.py's old 1st-5th-only ordinal word list, "the 10th one" had
        # no chance of resolving either way. Rows beyond what's actually
        # displayed have no visible position for an ordinal to refer to, so
        # this cap should always track max_display_rows, not exceed it.
        if len(last_rows) > 1:
            for row in last_rows[:20]:
                if not isinstance(row, dict):
                    continue
                _matched = False
                # Priority 0: an observation/answer row (ai_audit_answers or
                # ai_answers) — submission_id IS the real, directly-filterable
                # identifier already (no __id sibling needed, unlike entity
                # tables), and answer_text is the meaningful display value.
                # Confirmed live 2026-07-31: "what were the corrective actions
                # for the first finding in the above list" — the model's OWN
                # reasoning correctly named the exact submission_id from the
                # prior turn's listing, but nothing structured carried it
                # forward, so the generated SQL fell back to a generic
                # ORDER BY/LIMIT 1 instead of filtering on that exact id. Only
                # happened to return the right (empty) answer because that
                # particular audit had zero corrective actions at all —
                # a different audit with multiple findings and CAs would have
                # silently picked the wrong one.
                if row.get("submission_id") and row.get("answer_text"):
                    row_titles.append(str(row["answer_text"]))
                    row_ids.append(str(row["submission_id"]))
                    row_id_columns.append("submission_id")
                    if not row_entity_type:
                        row_entity_type = "observation"
                    continue
                # Priority 1: the row's own identifier columns — these are
                # the grammatical subject of a list question ("show all
                # AUDITS" -> audit_id) even when a joined PARENT's title
                # (e.g. "plan_title" on an audits-under-a-plan listing) is
                # also present on the same row. Must be checked before the
                # title-suffix match below, or the parent's title wins by
                # accident whenever it isn't literally named "title" (see
                # test_row_ids_paired_with_row_titles_multi_row_list).
                for col in ("audit_id", "schedule_id", "inspection_id",
                            "corrective_action_id"):
                    v = row.get(col)
                    if v and str(v).strip():
                        row_titles.append(str(v))
                        row_ids.append(str(row.get(f"{col}__id", "") or ""))
                        row_id_columns.append("id")
                        if not row_entity_type:
                            row_entity_type = _ROW_ID_COL_TYPES[col]
                        _matched = True
                        break
                if _matched:
                    continue
                # Priority 1.5: audit_cycle/inspection_cycle composite display
                # (date range + status) — cycles have NO name/title column at
                # all. Gated on the "<alias>_cycle_row__id" sibling actually
                # being present (i.e. inject_row_identity()'s cycle branch —
                # see validation/identity.py's _CYCLE_TABLES — fired for this
                # row) rather than on a bare "status" column, so this can
                # never misfire on some other joined table's own status
                # column.
                _cycle_id_key = next(
                    (c for c in row if isinstance(c, str) and c.endswith("_cycle_row__id")),
                    None,
                )
                if _cycle_id_key:
                    _sd, _ed, _st = row.get("start_date"), row.get("end_date"), row.get("status")
                    if _sd and _ed and _st:
                        row_titles.append(f"{_sd} to {_ed} ({_st})")
                        row_ids.append(str(row.get(_cycle_id_key) or ""))
                        row_id_columns.append("id")
                        if not row_entity_type:
                            row_entity_type = "cycle"
                        _matched = True
                if _matched:
                    continue
                # Priority 2: bare "title" or *_title (e.g. "audit_plan_title"
                # once aliased) — matches the plan_titles fix above via the
                # SAME suffix logic. Only reached when no identifier column
                # from priority 1 exists on the row at all, which is exactly
                # the case for entities like audit_plan that have no separate
                # business-id display column. Confirmed live 2026-07-30: a
                # plans listing selected ap.title AS audit_plan_title with no
                # audit_id/schedule_id/etc present; the old exact-match-only
                # whitelist never matched "audit_plan_title", so row_titles
                # stayed empty for that turn, the ordinal-resolution block in
                # session.py had nothing to point at, and a later "the third
                # one in the list" follow-up fell back to the LLM guessing its
                # own ORDER BY/OFFSET from scratch — which it got wrong (ASC
                # instead of matching the list's actual DESC display order),
                # silently returning the wrong row.
                for col in row:
                    if isinstance(col, str) and (col == "title" or col.endswith("_title")):
                        v = row.get(col)
                        if v and str(v).strip():
                            row_titles.append(str(v))
                            row_ids.append(str(row.get(f"{col}__id", "") or ""))
                            row_id_columns.append("id")
                            if not row_entity_type:
                                row_entity_type = "plan"
                            _matched = True
                        break
                if _matched:
                    continue
                for col in ("name", "facility_name", "auditor_name", "inspector_name"):
                    v = row.get(col)
                    if v and str(v).strip():
                        row_titles.append(str(v))
                        row_ids.append(str(row.get(f"{col}__id", "") or ""))
                        row_id_columns.append("id")
                        if not row_entity_type and col in _ROW_NAME_COL_TYPES:
                            row_entity_type = _ROW_NAME_COL_TYPES[col]
                        break

        # If only 1 row, store all values for maximum context
        if len(last_rows) == 1 and last_rows[0]:
            # Excludes __id-injected columns (see validation/identity.py) —
            # this dict gets rendered verbatim as "★ single-row result: {...}"
            # in build_context_block(), read directly by the reasoning LLM;
            # unlike the deliberate "resolved id='...'" ACTIVE CONTEXT
            # phrasing (which the generation.py UUID-filter-guard separately
            # re-verifies survives into the next SQL), a raw UUID leaking in
            # here has no such downstream check and risks being hallucinated
            # into a new query verbatim.
            single_row_vals = {
                k: str(v) for k, v in last_rows[0].items()
                if v is not None and str(v).strip() and not k.endswith("__id")
            }
            # Also pull cycle_id from single-row result
            for col in ("cycle_id", "inspection_cycle_id"):
                v = last_rows[0].get(col)
                if v and str(v) not in cycle_ids and _UUID_PAT.match(str(v).strip()):
                    cycle_ids.append(str(v).strip())

        # ── Determine active subject for this turn ───────────────────────────
        # Priority: cycle > single inspector > single inspection > single audit > single facility
        active_subject_type = ""
        active_subject_value = ""
        active_subject_label = ""
        active_subject_table = ""
        active_subject_alias = ""
        active_subject_domain = ""
        # Real row id, paired with active_subject_value above wherever one's
        # available (see the insp_row_ids/audit_row_ids/plan_row_ids/
        # fac_row_ids capture above, or an already-UUID value like cycle_ids
        # itself). Empty when no id could be resolved for this turn's query
        # shape (e.g. a GROUP BY, which inject_row_identity() skips) — the
        # ACTIVE CONTEXT/ordinal resolution in session.py falls back to
        # title/ILIKE-based resolution in that case, same as before this
        # field existed.
        active_subject_id = ""

        if len(cycle_ids) == 1:
            active_subject_type = "cycle"
            active_subject_value = cycle_ids[0]
            active_subject_label = cycle_ids[0]
            active_subject_id = cycle_ids[0]  # already the real UUID, not a display string
            active_subject_domain = self.domain
        elif len(insp_names) == 1:
            active_subject_type = "inspector"
            active_subject_value = insp_names[0]
            active_subject_label = insp_names[0]
            # People have their own established id-resolution path
            # (user_ids, captured from a literal user_id/inspector_user_id
            # column) — not identity.py's injection, since `users` is
            # deliberately excluded from ROW_IDENTITY_TABLES (composite
            # display value; see identity.py's docstring).
            active_subject_id = user_ids[0] if len(user_ids) == 1 else ""
            active_subject_domain = self.domain
        elif len(insp_ids) == 1:
            active_subject_type = "inspection"
            active_subject_value = insp_ids[0]
            active_subject_label = insp_ids[0]
            active_subject_id = insp_row_ids[0] if insp_row_ids and insp_row_ids[0] else ""
            active_subject_domain = self.domain
        elif len(audit_ids) == 1:
            active_subject_type = "audit"
            active_subject_value = audit_ids[0]
            active_subject_label = audit_ids[0]
            active_subject_id = audit_row_ids[0] if audit_row_ids and audit_row_ids[0] else ""
            active_subject_domain = self.domain
        # Distinct plan TITLE count, not row count — a multi-row result that's
        # really about ONE plan (e.g. "approval status ... each review level"
        # joining audit_plan to review, one row per level) still counts as a
        # single resolved plan for follow-up purposes. See plan_titles' capture
        # comment above for the live case this fixes.
        elif len(set(plan_titles)) == 1:
            active_subject_type = "plan"
            active_subject_value = plan_titles[0]
            active_subject_label = plan_titles[0]
            # Any non-empty entry works — every row here is the SAME plan
            # (that's what len(set(plan_titles)) == 1 means), so they all
            # pair with the same real id when injection ran at all.
            active_subject_id = next((pid for pid in plan_row_ids if pid), "")
            active_subject_domain = self.domain
        elif len(fac_names) == 1:
            active_subject_type = "facility"
            active_subject_value = fac_names[0]
            active_subject_label = fac_names[0]
            # Prefer the pre-existing facility_ids capture (a literal
            # facility_id/fac_id UUID column the query happened to select
            # directly) over the new identity-injection pairing — same
            # preference order session.py's facility ACTIVE CONTEXT branch
            # already applies between its own two sources.
            if len(facility_ids) == 1:
                active_subject_id = facility_ids[0]
            elif fac_row_ids and fac_row_ids[0]:
                active_subject_id = fac_row_ids[0]
            active_subject_domain = self.domain
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
                    active_subject_id = _ed_matches[0].get("id", "") or ""
                    active_subject_domain = self.domain
            # Read the persistent scope pointer directly instead of walking
            # session.turns — this is what makes scope survive a turn aging
            # out of MAX_TURNS. Copy active_subject_domain VERBATIM from
            # where the scope actually originated; never recompute it as
            # self.domain here — this turn may be answering in a different
            # domain than the one that first resolved the scope, and a
            # domain-exclusive type (inspection/audit/cycle) must remember
            # its true origin through any number of these inheritance hops,
            # not get silently relabeled as belonging to whichever domain
            # last happened to pass it along.
            if not active_subject_type and session.current_scope:
                _cs = session.current_scope
                if _sql_still_relevant_to_scope(last_sql, _cs["type"]):
                    active_subject_type = _cs["type"]
                    active_subject_value = _cs["value"]
                    active_subject_label = _cs["label"]
                    active_subject_table = _cs["table"]
                    active_subject_alias = _cs["alias"]
                    active_subject_domain = _cs["domain"]
                    active_subject_id = _cs.get("id", "") or ""
                else:
                    # This turn's own SQL shows no sign of relating to the
                    # stale scope — a real topic change, not a drilldown.
                    # Clear it instead of letting it perpetuate onto the
                    # NEXT unrelated turn too (add_turn() alone won't clear
                    # a pre-existing current_scope just because this turn
                    # sets no subject of its own). Doesn't lose the ability
                    # to return to it later — entity_memory already exists
                    # for recalling something named explicitly again.
                    session.current_scope = None

        if session is not None:
            # MT-03 fix: if we have a facility name but no inspection_id (e.g. MT-01
            # returned only facility_name without ir.inspection_id in SELECT), do a quick
            # DB lookup so the follow-up turn can filter by inspection_id rather than
            # falling back to the worked-example placeholder "INS-EXAMPLE-001".
            #
            # Gated to the inspection domain — confirmed live 2026-07-29 (qid=2a519e8d):
            # this fired unconditionally regardless of self.domain, so an AUDIT-domain
            # turn that happened to return a facility name (very common — "Aldar HQ"
            # appears on nearly every audit result) got a totally unrelated inspection_id
            # attached to it anyway. session.py's build_context_block() then surfaces
            # that as a "★ inspection_id(s) from this result" active-context clue, and
            # the reasoning model — reasonably, given how prominently it's shown — used
            # it as if it were the current AUDIT's identifier ("count corrective actions
            # for this audit... the inspection_id associated with this is
            # 2026/07/ST041/INS006"). That value doesn't match any audit_report.audit_id,
            # so the query silently returned 0 instead of the real answer (4) — a wrong
            # answer with no error, not a retryable failure, because the SQL itself was
            # syntactically and structurally correct. This lookup only ever benefits an
            # inspection-domain follow-up in the first place, so it must not run at all
            # while answering in the audit domain.
            if self.domain == "inspection" and fac_names and not insp_ids and len(last_rows) <= 3:
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

            # Resolve the plan's real UUID, not just its title, so a follow-up
            # can filter WHERE ap.id = '<uuid>' instead of ap.title = '<title>'
            # or ILIKE. Confirmed live 2026-07-30: audit_plan.title has NO
            # uniqueness constraint at the DB level — only `id` does — and the
            # live data already has duplicates ('Delegation plan -1' and
            # 'Delegation testing' each have 2 active rows). A follow-up
            # scoped by title alone would silently operate on whichever
            # duplicate the DB happens to return first, with no error and no
            # sign anything's wrong. When the lookup itself finds more than
            # one match, deliberately leave plan_ids empty and flag
            # plan_title_ambiguous instead of guessing — session.py's ACTIVE
            # CONTEXT hint then warns rather than presenting false precision.
            #
            # M4: this round trip is now the FALLBACK, superseded by
            # active_subject_id whenever inject_row_identity() (identity.py)
            # already resolved one directly from the row itself — cheaper,
            # and reflects the EXACT row this turn's query returned rather
            # than a fresh lookup that could theoretically disagree (e.g. if
            # a plan was renamed between the query and this synthesis step).
            # Only reachable when that query shape wasn't eligible for
            # injection (e.g. a GROUP BY).
            plan_ids: list = []
            plan_title_ambiguous = False
            if active_subject_type == "plan" and active_subject_id:
                plan_ids = [active_subject_id]
            elif self.domain == "audit" and active_subject_type == "plan":
                try:
                    with self.db_engine.connect() as _conn:
                        _prows = _conn.execute(
                            sql_text(
                                "SELECT id FROM audit_plan "
                                "WHERE title = :t AND active = true"
                            ),
                            {"t": active_subject_value},
                        ).fetchall()
                    if len(_prows) == 1:
                        plan_ids = [str(_prows[0][0])]
                    elif len(_prows) > 1:
                        plan_title_ambiguous = True
                except Exception:
                    pass  # Silently skip — follow-up falls back to title/ILIKE

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
                domain=self.domain,
                tool_used=steps[-2].tool if len(steps) >= 2 else "unknown",
                sql=last_sql,
                total_count=last_total,
                page_offset=len(last_rows),
                # Excludes __id-injected columns (see validation/identity.py)
                # — this list is rendered verbatim as "Columns: [...]" in
                # build_context_block(), read directly by the reasoning LLM.
                columns=[c for c in last_cols if not c.endswith("__id")],
                inspection_ids=insp_ids[:5],
                audit_ids=audit_ids[:5],
                facility_names=fac_names[:5],
                inspector_names=insp_names[:5],
                cycle_ids=cycle_ids[:3],
                user_ids=user_ids[:5],
                facility_ids=facility_ids[:5],
                plan_ids=plan_ids,
                plan_title_ambiguous=plan_title_ambiguous,
                disambiguation_name=_disambig_name,
                disambiguation_candidates=_disambig_candidates,
                disambiguation_origin_question=_disambig_origin_question,
                entity_disambiguation_candidates=_ent_disambig_candidates,
                entity_disambiguation_table=_ent_disambig_table,
                entity_disambiguation_alias=_ent_disambig_alias,
                single_row_values=single_row_vals,
                row_titles=row_titles,
                row_ids=row_ids,
                row_id_columns=row_id_columns,
                row_entity_type=row_entity_type,
                active_subject_type=active_subject_type,
                active_subject_value=active_subject_value,
                active_subject_label=active_subject_label,
                active_subject_table=active_subject_table,
                active_subject_alias=active_subject_alias,
                active_subject_domain=active_subject_domain,
                active_subject_id=active_subject_id,
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


    def _synthesize(self, question, tool_name, tool_result, max_display_rows=20, sql="", session=None):
        """Thin wrapper around _synthesize_body(): prepends
        session.pending_resolved_disclosure (set by
        ConversationSession.build_context_block() when this turn's question
        resolved an ordinal/back-reference — see session.py) to whatever
        _synthesize_body() returns.

        This has to live here, wrapping the return value, rather than as a
        prompt instruction to the LLM, because most answers below are built
        by plain Python string formatting ("Found N result(s): ...", "The
        query returned no results.") — never composed by the LLM at all.
        Confirmed live 2026-07-31: telling the LLM via the system prompt to
        "state what you resolved" had zero effect on these answers, even
        though the LLM's OWN internal thought correctly named the resolved
        row — that reasoning never reaches this deterministic formatter, so
        it can't appear in what the user sees unless injected here instead.
        """
        answer = self._synthesize_body(question, tool_name, tool_result,
                                        max_display_rows=max_display_rows, sql=sql)
        disclosure = getattr(session, "pending_resolved_disclosure", "") if session else ""
        if disclosure:
            return f"{disclosure}\n\n{answer}"
        return answer

    def _synthesize_body(self, question, tool_name, tool_result, max_display_rows=20, sql=""):
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
            # Strip __id columns (inject_row_identity(), see validation/identity.py)
            # HERE, once, before any downstream branch below can see them —
            # confirmed live 2026-07-30 (eval run, 18/74 questions): the
            # deterministic "Found N result(s): 1. title: ... | title__id:
            # <uuid>" formatter further below built its own row-by-row text
            # directly from these raw rows, via a hide-list (_HIDE_COLS) that
            # had no way to know about this new column shape — a real UUID
            # leaked straight into the actual answer shown to the user. The
            # _llm_answer() fallback further down also json.dumps()s this
            # same `data` into the reasoning LLM's own prompt, risking it
            # echoing a raw UUID into free text too. Stripping once here,
            # before either path runs, closes both at the source instead of
            # patching each rendering branch separately.
            cols, rows = strip_row_identity_columns(cols, rows)
            data = {**data, "rows": rows, "columns": cols}
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
                                  {"inspection_count", "audit_count", "schedule_count", "count", "total"}), None)
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
                    "average", "avg_score", "avg_inspection_score", "avg_audit_score",
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
                              "user_id",
                              # audit-domain raw UUID FKs (see schema_introspector.py's
                              # AUDIT_FK_RESOLUTION) that would otherwise leak as UUIDs
                              "auditor_id", "auditee_id", "audit_cycle_id",
                              "internal_stakeholder_id",
                              # deterministically injected by
                              # inject_observation_submission_id() (see
                              # validation/identity.py) whenever answer_text is
                              # selected — a raw correlation key, not something
                              # a human needs to read; still available in the
                              # underlying row dict for row_titles/row_ids
                              # capture in _finish(), which runs separately.
                              "submission_id"}
                # Confirmed live 2026-07-30 (eval run, 18/74 questions):
                # inject_row_identity()'s __id-suffixed columns (see
                # validation/identity.py) leaked a raw UUID straight into this
                # deterministic answer text — e.g. "1. title: del 01 | ... |
                # title__id: 54327f53-...". This code path builds the actual
                # user-facing answer directly from raw rows and has its own
                # separate hide-list from the columns/single_row_values fields
                # this module already quarantines earlier in _finish() — those
                # only ever covered the fields fed back into a LATER turn's
                # prompt, not this same-turn answer text. Any _HIDE_COLS
                # membership check below must also exclude __id columns.
                def _is_hidden_col(c: str) -> bool:
                    return c in _HIDE_COLS or c.endswith("__id")
                displayable_cols = [c for c in rows[0].keys() if not _is_hidden_col(c)]
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

                # The 100-char cutoff below exists to stop a wall of text when
                # there are MANY rows, not to shorten the one thing a small
                # result is actually about. Confirmed live 2026-07-29
                # (qid=25cbdee7): "show the observation raised for which the
                # above action was proposed" returned exactly 1 row whose whole
                # point was aca.observation_text (a long free-text finding) —
                # the chat summary cut it to 97 chars + "..." even though the
                # separate rows/columns payload the frontend also renders (see
                # api/routes/chat.py) had the full text the whole time, so the
                # chat bubble and the table visibly disagreed. Skip truncation
                # entirely when the result is small enough that there's no
                # wall-of-text risk.
                _truncate_values = len(rows) > 3
                lines = [header]
                for i, row in enumerate(rows[:max_display_rows], 1):
                    parts = []
                    for col, val in row.items():
                        if _is_hidden_col(col):
                            continue
                        # Skip columns promoted to header (constant across all rows)
                        if col in constant_cols:
                            continue
                        if val is None:
                            if len(displayable_cols) <= 2:
                                parts.append(f"{col}: N/A")
                            continue
                        val_str = str(val)
                        if _truncate_values and len(val_str) > 100:
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
            _prompt_tok, _completion_tok = extract_token_counts(raw)
            stats.total_prompt_tokens += _prompt_tok
            stats.total_completion_tokens += _completion_tok

        _log_llm.debug("[LLM ] iter=%d latency=%dms prompt_tok=%d completion_tok=%d out_len=%d",
                       iteration, latency_ms, _prompt_tok, _completion_tok, len(content))

        # Debug logging (raw LLM output to separate file when DEBUG_LLM=1)
        if self._debug:
            trunc = _detect_truncation(content)
            trunc_note = f"\n⚠ {trunc}" if trunc else ""
            # Real configured model name, not a hardcoded "LLAMA" label — this
            # codebase moved from Llama to Qwen (Ollama) and now optionally to
            # Claude (see llm_provider) long ago without anyone updating this
            # string, so the label had stopped meaning anything about what
            # actually produced the output. self.reasoning_llm.model is a
            # real field on both llm/ollama.py's and llm/anthropic.py's client
            # objects, so this stays accurate across any future provider swap
            # too, with no further edit needed here.
            _model_name = getattr(self.reasoning_llm, "model", "?")
            _dbg(self._debug,
                 f"REASONING LLM RAW OUTPUT ({_model_name}) [{latency_ms}ms]{trunc_note}",
                 content, iteration)

        return content

