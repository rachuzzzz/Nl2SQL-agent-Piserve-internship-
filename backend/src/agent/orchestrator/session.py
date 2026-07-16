"""
ConversationSession -- per-user-session conversation state, independent
of AgentOrchestrator itself (constructed by callers: api/routes/chat.py,
scripts/run.py). Extracted verbatim from the original
agent/orchestrator.py.
"""

import re
from typing import Optional

from src.agent.orchestrator.models import ConversationTurn
from src.agent.orchestrator.debug import _ilike_escape




class ConversationSession:
    MAX_TURNS = 5

    def __init__(self):
        self.turns: list[ConversationTurn] = []

    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
        if len(self.turns) > self.MAX_TURNS:
            self.turns = self.turns[-self.MAX_TURNS:]

    @property
    def last_turn(self) -> Optional[ConversationTurn]:
        return self.turns[-1] if self.turns else None

    def build_context_block(self, current_question: str = "") -> str:
        if not self.turns:
            return ""

        # NOTE: bare "that"/"this" are deliberately excluded — they double as
        # relative pronouns/determiners in ordinary questions ("mitigative actions
        # THAT are overdue", "show me THIS month's data") and caused the sticky
        # active-context gate below to false-positive-match plain global questions.
        # Demonstrative reference to a specific entity is still caught via the
        # explicit "that/this <entity>" phrases.
        _FOLLOWUP_RE = re.compile(
            r"\b(it|its|them|they|she|he|her|his|their|"
            r"(?:that|this)\s+(?:inspection|site|facility|form|inspector|cycle|"
            r"person|project|client|action|schedule)|"
            r"the\s+same|the\s+above|from\s+there|more|rest|remaining|"
            r"continue|next|previous|last\s+one|back\s+to|go\s+back)\b",
            re.IGNORECASE,
        )
        # A question is "clearly fresh" only when it introduces a wholly new subject.
        # INTENTIONALLY NARROW: question-starters like "how many", "what", "show", "list"
        # appear constantly in follow-ups ("how many are completed", "show the open ones")
        # and must NOT suppress the active context.
        _FRESH_SUBJECT_RE = re.compile(
            r"\b(most\s+recent(?!ly)|most\s+common|latest)\b",
            re.IGNORECASE,
        )
        is_followup = bool(_FOLLOWUP_RE.search(current_question))
        is_clearly_fresh = (
            bool(_FRESH_SUBJECT_RE.search(current_question))
            and not is_followup
        )

        lines = ["━━━ CONVERSATION HISTORY ━━━"]
        for i, t in enumerate(self.turns, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  User: {t.question}")
            short = t.answer[:200] + ("..." if len(t.answer) > 200 else "")
            lines.append(f"  Answer: {short}")
            if t.sql:
                lines.append(f"  SQL: {t.sql[:150]}")
            if t.total_count > 0:
                lines.append(f"  Total rows: {t.total_count}")
            if t.columns:
                lines.append(f"  Columns: {t.columns}")
            if t.inspection_ids:
                lines.append(f"  ★ inspection_id(s) from this result: {t.inspection_ids[:5]}")
            if t.cycle_ids:
                lines.append(f"  ★ cycle_id(s) from this result: {t.cycle_ids[:3]}")
            if t.facility_names:
                lines.append(f"  ★ facility(s) from this result: {t.facility_names[:5]}")
            if t.inspector_names:
                lines.append(f"  ★ inspector(s) from this result: {t.inspector_names[:5]}")
            if t.single_row_values:
                lines.append(f"  ★ single-row result: {t.single_row_values}")
            # Show active subject for each turn
            if t.active_subject_type and t.active_subject_value:
                lines.append(
                    f"  ◆ ACTIVE SUBJECT: {t.active_subject_type} = "
                    f"'{t.active_subject_label or t.active_subject_value}' "
                    f"(exact value: {t.active_subject_value})"
                )
        lines.append("")

        # ── Resolve active subject across all turns ──────────────────────────
        # Walk backwards through turns to find the most recent active subject.
        # "go back to that inspector" can re-anchor to an earlier turn's subject.
        active_type = ""
        active_value = ""
        active_label = ""
        active_table = ""
        active_alias = ""

        # "go back" or "back to" signals: re-anchor to the earliest matching subject
        _BACK_RE = re.compile(r"\b(go\s+back|back\s+to|return\s+to)\b", re.IGNORECASE)
        if _BACK_RE.search(current_question):
            # Look for the first (oldest) active subject in history
            for t in self.turns:
                if t.active_subject_type and t.active_subject_value:
                    active_type = t.active_subject_type
                    active_value = t.active_subject_value
                    active_label = t.active_subject_label
                    active_table = t.active_subject_table
                    active_alias = t.active_subject_alias
                    break
        else:
            # Normal: use the most recent active subject
            for t in reversed(self.turns):
                if t.active_subject_type and t.active_subject_value:
                    active_type = t.active_subject_type
                    active_value = t.active_subject_value
                    active_label = t.active_subject_label
                    active_table = t.active_subject_table
                    active_alias = t.active_subject_alias
                    break

        # self.turns is guaranteed non-empty (early-return above), so `last` is
        # always defined for both the fallback block and the disambiguation
        # STOP-guard below.
        last = self.turns[-1]

        # ── Emit the scoped context block ────────────────────────────────────
        def _lookup_entity_id(label: str) -> str:
            """
            Find the id captured for `label` in the most recent turn whose
            entity_disambiguation_candidates contains an exact (case-insensitive)
            name match — mirrors the `_last_with_uid` lookup the `inspector`
            branch below already uses for people, so facility/project/client/
            organisation follow-ups can filter by exact id too instead of
            re-searching by ILIKE name every time.
            """
            for t in reversed(self.turns):
                match = next(
                    (c for c in t.entity_disambiguation_candidates
                     if c.get("name", "").strip().lower() == label.lower() and c.get("id")),
                    None
                )
                if match:
                    return match["id"]
            return ""

        # Scope only carries forward when the question is an actual detected
        # follow-up (pronoun, "that <entity>", "back to", etc.) — NOT merely
        # whenever the question fails to match a narrow "fresh subject" allowlist.
        # The previous default (carry forward unless clearly fresh) let stale
        # entity scope silently leak into unrelated global questions that happened
        # not to contain "most recent"/"most common"/"latest" (e.g. "which cycle
        # is currently active" inheriting a facility resolved 2 turns earlier).
        if is_followup and not is_clearly_fresh and active_type and active_value:
            if active_type == "cycle":
                lines.append(
                    f"ACTIVE CONTEXT — The conversation is scoped to cycle "
                    f"'{active_value}'."
                    f" All follow-up questions about inspections, mitigative actions,"
                    f" facilities, or inspectors should include WHERE ir.cycle_id = '{active_value}'."
                    f" When user says 'they', 'that cycle', 'those facilities', 'those inspectors'"
                    f" — this is the scope."
                )
            elif active_type == "inspector":
                # Prefer exact UUID filter if we captured it from a disambiguation query
                _last_with_uid = next(
                    (t for t in reversed(self.turns) if t.user_ids),
                    None
                )
                if _last_with_uid and len(_last_with_uid.user_ids) == 1:
                    _uid = _last_with_uid.user_ids[0]
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is about user '{active_label}' (resolved user_id='{_uid}')."
                        f" When user says 'they', 'that inspector', 'that person', 'their schedules/actions'"
                        f" — filter EXACTLY by WHERE u.id = '{_uid}'"
                        f" (join users u on the appropriate FK). NEVER use ILIKE on a name — use this UUID."
                        f" Do NOT broaden to all users."
                    )
                else:
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is about inspector '{active_label}'."
                        f" When user says 'they', 'that inspector', 'their inspections'"
                        f" — filter by u.first_name || ' ' || u.last_name ILIKE '%{_ilike_escape(active_label)}%'"
                        f" on the users table joined via the appropriate FK."
                        f" Do NOT broaden to all inspectors."
                    )
            elif active_type == "inspection":
                lines.append(
                    f"ACTIVE CONTEXT — The conversation is about inspection '{active_value}'."
                    f" When the user says 'it', 'that inspection', 'that site'"
                    f" — use WHERE ir.inspection_id = '{active_value}'"
                    f" (or aa.inspection_id = '{active_value}' for form answers)."
                )
            elif active_type == "facility":
                # Prefer an exact id filter if one is available. Two sources,
                # checked in order: (1) a facility_id/fac_id column captured
                # directly from an ordinary query result (_last_with_fid —
                # mirrors _last_with_uid for inspectors above), (2) a name
                # resolved via the STEP 1 disambiguation flow (_lookup_entity_id,
                # unchanged). Only falls back to ILIKE if neither has an id.
                _last_with_fid = next(
                    (t for t in reversed(self.turns) if t.facility_ids),
                    None
                )
                _fid = ""
                if _last_with_fid and len(_last_with_fid.facility_ids) == 1:
                    _fid = _last_with_fid.facility_ids[0]
                if not _fid:
                    _fid = _lookup_entity_id(active_label)
                if _fid:
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is scoped to facility '{active_label}'"
                        f" (resolved facility_id='{_fid}')."
                        f" ALL follow-up questions about mitigative actions, inspections, scores,"
                        f" or form answers should filter EXACTLY by WHERE fac.id = '{_fid}'"
                        f" — NOT ILIKE — unless the user explicitly asks for a global/all-facilities view."
                        f" Do NOT query globally when the context is scoped to this facility."
                    )
                else:
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is scoped to facility '{active_label}'."
                        f" ALL follow-up questions about mitigative actions, inspections, scores,"
                        f" or form answers should include a facility filter"
                        f" (fac.name ILIKE '%{_ilike_escape(active_label)}%') unless the user explicitly asks"
                        f" for a global/all-facilities view."
                        f" Do NOT query globally when the context is scoped to this facility."
                    )
            elif active_type == "entity":
                # Generic scope for project/client/organisation — same idea as
                # facility above, parameterized by whichever table/alias the
                # STEP 1 disambiguation actually searched.
                _tbl = active_table or "entity"
                _als = active_alias or "e"
                _eid = _lookup_entity_id(active_label)
                if _eid:
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is scoped to {_tbl} '{active_label}'"
                        f" (resolved {_tbl}_id='{_eid}')."
                        f" ALL follow-up questions about mitigative actions, inspections, scores,"
                        f" or form answers should filter EXACTLY by WHERE {_als}.id = '{_eid}'"
                        f" — NOT ILIKE — unless the user explicitly asks for a global view."
                        f" Do NOT query globally when the context is scoped to this {_tbl}."
                    )
                else:
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is scoped to {_tbl} '{active_label}'."
                        f" ALL follow-up questions about mitigative actions, inspections, scores,"
                        f" or form answers should include a filter"
                        f" ({_als}.name ILIKE '%{_ilike_escape(active_label)}%') unless the user explicitly asks"
                        f" for a global view."
                        f" Do NOT query globally when the context is scoped to this {_tbl}."
                    )
            elif active_type == "corrective_action":
                lines.append(
                    f"ACTIVE CONTEXT — The conversation is about mitigative action '{active_value}'."
                )
        elif is_followup and not is_clearly_fresh:
            # Fallback: legacy single-entity context. Same fix as the primary
            # gate above — only carry this forward on an actual detected
            # follow-up, not merely whenever the question isn't "clearly fresh".
            if last.inspection_ids and len(last.inspection_ids) == 1:
                lines.append(
                    f"CONTEXT — The conversation is about inspection '{last.inspection_ids[0]}'."
                    f" When the user says 'that inspection', 'it', 'that site' — use"
                    f" WHERE ir.inspection_id = '{last.inspection_ids[0]}'"
                    f" as an exact filter."
                )
            elif last.inspection_ids:
                ids_str = ", ".join(f"\'{i}\'" for i in last.inspection_ids[:5])
                if last.total_count > len(last.inspection_ids):
                    # The captured IDs are only a SAMPLE of a larger result set
                    # (e.g. 5 captured out of 195 total). An IN-list of these 5
                    # can NEVER represent "all"/"those"/"these" — it would silently
                    # answer over 5 rows instead of all {total_count}.
                    # Give Llama the previous SQL so it can re-apply the SAME
                    # base filters as a fresh aggregate, NOT an IN-list.
                    prev_sql_snippet = " ".join(last.sql.split())[:500] if last.sql else "(not available)"
                    lines.append(
                        f"CONTEXT — The previous result had {last.total_count} total rows; "
                        f"only {len(last.inspection_ids)} sample IDs were captured ({ids_str})."
                        f" These {len(last.inspection_ids)} IDs are NOT the full set.\n"
                        f"  If the user asks 'how many of these/those/them are X' (an aggregate"
                        f" over the FULL {last.total_count}-row result), do NOT filter by"
                        f" inspection_id IN (...) — that covers only"
                        f" {len(last.inspection_ids)} of {last.total_count} rows.\n"
                        f"  INSTEAD: reuse the SAME FROM/JOIN/WHERE filters as the previous"
                        f" query below, add the new condition (e.g. status = 'CLOSED'), and"
                        f" return a fresh COUNT(*) — a complete recount, not a sample filter.\n"
                        f"  Previous SQL: {prev_sql_snippet}"
                    )
                else:
                    # The captured IDs ARE the complete result set (small result,
                    # e.g. "inspections by George" → 3 rows) — IN-list is valid here.
                    lines.append(
                        f"CONTEXT — The conversation involves inspections: {ids_str}"
                        f" (this is the COMPLETE result set, {last.total_count or len(last.inspection_ids)} total)."
                        f" Filter by inspection_id IN ({ids_str}) when user references 'those'."
                    )
            elif last.facility_names and len(last.facility_names) == 1:
                lines.append(
                    f"CONTEXT — The conversation is about facility '{last.facility_names[0]}'."
                    f" Filter by fac.name ILIKE '%{_ilike_escape(last.facility_names[0])}%'."
                )
            elif last.single_row_values:
                lines.append(f"CONTEXT — Last query returned a single result: {last.single_row_values}")

        # Disambiguation context — stop the model re-running the same check
        if not is_clearly_fresh and last.disambiguation_name:
            _dn = last.disambiguation_name
            _dc = last.total_count
            _q_lower = current_question.lower().strip()
            _q_has_same_name = _dn.lower() in _q_lower
            # "name pick": short response with no query verbs = user picking from the list
            _QUERY_VERB_RE = re.compile(
                r"\b(show|list|find|what|who|which|how|count|display|get|fetch|"
                r"give|tell|check|search|view|see|compare|rank|sort|filter)\b",
                re.IGNORECASE,
            )
            _specific = current_question.strip().rstrip("\t").strip()
            # Strip trailing sentence punctuation ("George Auditor?", "george auditor.")
            # so it doesn't break the exact-match comparison against candidate names below.
            _specific = re.sub(r"[?.!,;:]+$", "", _specific).strip()
            _is_name_pick = (
                not bool(_QUERY_VERB_RE.search(current_question))
                and len(_q_lower.split()) <= 6
                and _specific.lower() != _dn.lower()  # must be more specific than the original term
            )

            # Exact-match short-circuit: if the user's text — once filler/verb
            # words like "ya fetch for" are stripped — is an EXACT match for
            # one of the candidate names from the previous disambiguation,
            # resolve it directly. Without this, repeating an exact name (or
            # wrapping it in "ya fetch for ...") just re-triggers the same
            # ILIKE search / STOP branch forever whenever that name is a
            # substring of other candidates.
            _FILLER_WORDS = {
                "ya", "yeah", "yep", "please", "pls", "me", "for", "the",
                "that", "data", "info", "information", "details", "on",
                "about", "of", "person", "user",
            }

            def _strip_filler(text: str) -> str:
                words = text.strip().split()
                while words and (
                    words[0].lower() in _FILLER_WORDS
                    or _QUERY_VERB_RE.fullmatch(words[0])
                ):
                    words.pop(0)
                while words and words[-1].lower() in _FILLER_WORDS:
                    words.pop()
                return " ".join(words)

            _cleaned_specific = _strip_filler(_specific)
            _name_matches = [
                _cand for _cand in last.disambiguation_candidates
                if _cand.get("name", "").strip().lower() == _cleaned_specific.lower()
            ]
            if not _name_matches:
                # Full-sentence follow-up (e.g. a drilldown-click "list all
                # schedules assigned to kavya insp") embeds the resolved
                # candidate name inside a longer question instead of replying
                # with just the bare name — the whole-message equality check
                # above can never match that shape. Fall back to substring
                # containment against the raw question, preferring the LONGEST
                # candidate name so a more specific match ("kavya insp") wins
                # over a shorter coincidental one ("kavya"). Without this, the
                # question still contains the ORIGINAL search term as a
                # substring, so it gets misclassified by the _q_has_same_name
                # STOP-guard below as "same vague query repeated" instead of
                # "already resolved" — the model then re-runs the broad search
                # and never reaches the user's actual request.
                _q_full_lower = current_question.lower()
                _substr_hits = [
                    _cand for _cand in last.disambiguation_candidates
                    if _cand.get("name") and _cand["name"].strip().lower() in _q_full_lower
                ]
                if _substr_hits:
                    _max_len = max(len(_c["name"]) for _c in _substr_hits)
                    _name_matches = [_c for _c in _substr_hits if len(_c["name"]) == _max_len]
            # Only auto-resolve when the name uniquely identifies one candidate.
            # Several distinct accounts can share an identical display name
            # (e.g. 8 different "George Joseph" users seen in this dataset) —
            # picking the first one silently would bind the wrong identity.
            _exact_match = _name_matches[0] if len(_name_matches) == 1 else None
            _ambiguous_duplicate_names = len(_name_matches) > 1

            # Ordinal short-circuit: "the first one", "option 2", "#3", "the 2nd"
            # — resolve directly to that position in the previously-shown list.
            # Without this, an ordinal pick falls through to the generic
            # _is_name_pick branch below, which can only tell the model to
            # "use the UUID from previous result" with no way to say WHICH
            # uuid that is — the model then re-runs the disambiguation SQL
            # over and over trying to re-derive it from conversation text and
            # never converges within the iteration budget (observed: 8 wasted
            # generate_sql calls / ~70s / final_answer giving up).
            # Gated on _is_name_pick (same short/no-verb/not-the-original-term
            # shape used for exact-name picks) so it can't misfire on a
            # legitimate analytical question that happens to contain an
            # ordinal word (e.g. "what's the second highest score" has a
            # query verb and fails _is_name_pick).
            _ordinal_n = None
            if _is_name_pick:
                _ORDINAL_WORD_TO_N = {
                    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
                }
                _q_for_ordinal = _cleaned_specific.lower()
                for _word, _n in _ORDINAL_WORD_TO_N.items():
                    if re.search(rf"\b{_word}\b", _q_for_ordinal):
                        _ordinal_n = _n
                        break
                if _ordinal_n is None:
                    _om = re.search(
                        r"\b(?:option|choice|number|#)\s*0*(\d{1,2})\b|\b(\d{1,2})(?:st|nd|rd|th)\b",
                        _q_for_ordinal,
                    )
                    if _om:
                        _ordinal_n = int(_om.group(1) or _om.group(2))
            _ordinal_match = (
                last.disambiguation_candidates[_ordinal_n - 1]
                if _ordinal_n and 1 <= _ordinal_n <= len(last.disambiguation_candidates)
                else None
            )

            # Remind the model what it was ACTUALLY asked for before the name
            # went ambiguous. Without this, a multi-turn narrowing sequence
            # ("list schedules for neenu" → too many → "neenu qa inspector")
            # loses track of "schedules" and settles for just resolving/describing
            # the person instead of running the original request against them.
            _origin_q = last.disambiguation_origin_question
            _origin_note = (
                f" Once resolved, you MUST go on to actually answer the ORIGINAL"
                f" request — '{_origin_q}' — not just return the person's info/role."
                if _origin_q and _origin_q.strip().lower() != _dn.lower() else ""
            )

            if _exact_match:
                lines.append(
                    f"DISAMBIGUATION RESOLVED — '{_cleaned_specific}' is an EXACT match for one of"
                    f" the {_dc} candidates found previously (user_id="
                    f"'{_exact_match['user_id']}').\n"
                    f"  Use WHERE u.id = '{_exact_match['user_id']}' directly to run the"
                    f" original query now. Do NOT re-run the disambiguation search and"
                    f" do NOT ask the user to pick again — this name is unambiguous."
                    f"{_origin_note}"
                )
            elif _ordinal_match:
                lines.append(
                    f"DISAMBIGUATION RESOLVED — user picked option #{_ordinal_n}"
                    f" ('{_ordinal_match['name']}') from the {_dc} candidates found"
                    f" previously (user_id='{_ordinal_match['user_id']}').\n"
                    f"  Use WHERE u.id = '{_ordinal_match['user_id']}' directly to run the"
                    f" original query now. Do NOT re-run the disambiguation search and"
                    f" do NOT ask the user to pick again."
                    f"{_origin_note}"
                )
            elif _ambiguous_duplicate_names:
                _dup_uids = ", ".join(f"'{c['user_id']}'" for c in _name_matches)
                lines.append(
                    f"⚠️ STOP — '{_cleaned_specific}' matches {len(_name_matches)} DIFFERENT"
                    f" people from the previous list (same display name, different accounts:"
                    f" user_id IN ({_dup_uids})).\n"
                    f"  YOU MUST call final_answer NOW asking the user to disambiguate further"
                    f" by email address — do NOT pick one of these UUIDs arbitrarily and do NOT"
                    f" call generate_sql."
                )
            elif _dc > 12:
                if _q_has_same_name and not _is_name_pick:
                    # User re-asked the same partial-name query — hard block, no SQL
                    lines.append(
                        f"⚠️ STOP — DISAMBIGUATION ALREADY DONE (MANDATORY RULE):\n"
                        f"  Turn {len(self.turns)} already searched '{_dn}' → {_dc} matches (threshold=12).\n"
                        f"  The current question still uses '{_dn}' as a partial name.\n"
                        f"  YOU MUST call final_answer NOW with:\n"
                        f'  "I found {_dc} people matching \'{_dn}\' — too many to list. '
                        f'Please provide a more specific full name (e.g. \'George Smith\') '
                        f'or their email address so I can narrow it down."\n'
                        f"  DO NOT call generate_sql. DO NOT run any SQL."
                    )
                elif _is_name_pick:
                    # User is providing a more specific name — allow a targeted full-name lookup
                    lines.append(
                        f"DISAMBIGUATION CONTEXT — Previous turn found {_dc} people matching"
                        f" '{_dn}' (too many to list). The user now appears to be providing"
                        f" a specific full name: '{_specific}'.\n"
                        f"  STEP A: Run a targeted lookup using the concatenated full name:\n"
                        f"    WHERE (u.first_name || ' ' || u.last_name) ILIKE '%{_ilike_escape(_specific)}%'"
                        f"  — NOT separate first_name OR last_name conditions. If '{_specific}'"
                        f"  includes a role/descriptor rather than a real name (e.g. 'neenu qa"
                        f"  inspector'), instead search by the name part alone and use the role"
                        f"  words to pick among the results via a JOIN to user_roles/role.\n"
                        f"  STEP B: If exactly 1 match, use WHERE u.id = '<uuid>' to run the"
                        f"  original query. If still 2–12, list them. If 0, tell user not found."
                        f"{_origin_note}"
                    )
                else:
                    lines.append(
                        f"DISAMBIGUATION CONTEXT — Previous turn found {_dc} people"
                        f" matching '{_dn}' (too many). If the user is providing a specific"
                        f" full name, use concatenated full-name search:"
                        f" (u.first_name || ' ' || u.last_name) ILIKE '%<full name>%'."
                    )
            elif _dc > 1:
                if _q_has_same_name and not _is_name_pick:
                    lines.append(
                        f"⚠️ STOP — DISAMBIGUATION ALREADY DONE (MANDATORY RULE):\n"
                        f"  Turn {len(self.turns)} already found {_dc} people matching '{_dn}'.\n"
                        f"  YOU MUST call final_answer NOW to list those {_dc} options\n"
                        f"  and ask the user which one they meant.\n"
                        f"  DO NOT call generate_sql again."
                    )
                elif _is_name_pick:
                    lines.append(
                        f"DISAMBIGUATION CONTEXT — Previous turn found {_dc} people matching"
                        f" '{_dn}'. The user now appears to be selecting a specific person:"
                        f" '{_specific}'. Use the UUID already returned in the previous result"
                        f" (WHERE u.id = '<uuid>') to run the original query."
                        f" If UUID is not in history, look up by full name:"
                        f" WHERE (u.first_name || ' ' || u.last_name) ILIKE '%{_ilike_escape(_specific)}%'."
                        f"{_origin_note}"
                    )
                else:
                    lines.append(
                        f"DISAMBIGUATION CONTEXT — Previous turn found {_dc} people"
                        f" matching '{_dn}'. Use UUID from previous result for exact lookup."
                    )

        # Entity disambiguation context (facility/project/client/organisation) —
        # the entity-equivalent of the people exact-match block above. Gated on
        # the IMMEDIATELY PRECEDING turn's entity_disambiguation_candidates
        # directly (not on active_subject_type — the STEP 2b final_answer
        # termination path, which is how every 2+-match entity disambiguation
        # list actually ends, never sets active_subject_type, so the ACTIVE
        # CONTEXT facility/entity branches above never fire for a same-turn
        # drilldown-click follow-up). Unlike the people block, the typical
        # follow-up here is a full sentence with the resolved name embedded
        # (e.g. drilldown-click text "show mitigative actions at Adnoc
        # Schools- Ghayathi"), not a bare name reply — so this matches by
        # substring containment against the whole question, not equality.
        if not is_clearly_fresh and last.entity_disambiguation_candidates:
            _ent_match = next(
                (c for c in last.entity_disambiguation_candidates
                 if c.get("id") and c.get("name")
                 and c["name"].strip().lower() in current_question.lower()),
                None
            )
            if _ent_match:
                _ent_tbl = last.entity_disambiguation_table or "entity"
                _ent_als = last.entity_disambiguation_alias or "e"
                lines.append(
                    f"DISAMBIGUATION RESOLVED — '{_ent_match['name']}' is an EXACT match for one"
                    f" of the {_ent_tbl} candidates found previously"
                    f" ({_ent_tbl}_id='{_ent_match['id']}').\n"
                    f"  Use WHERE {_ent_als}.id = '{_ent_match['id']}' directly to run the"
                    f" original query now — do NOT use ILIKE on the name."
                )

        lines.append("")
        lines.append(
            "PAGINATION: If the user says 'show more', 'show the rest', 'continue',"
            " 'remaining', 'next', 'show all' — re-run the LAST SQL with higher LIMIT."
            " Do NOT generate a new query."
        )
        lines.append(
            "REFINEMENT: If the user says 'filter by X', 'only the Y ones', 'sort by Z'"
            " — modify the LAST SQL by adding/changing WHERE or ORDER BY."
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)
