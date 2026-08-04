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
from src.agent.orchestrator.disambiguation import entity_type_for_ordinal_noun




# Domain-exclusive active_subject_type values — the underlying ID only means
# something in ONE domain's schema (inspection_id/audit_id/inspection_cycle vs
# audit_cycle are different tables entirely, not shared). "facility"/
# "inspector"/"entity" are deliberately NOT in this set — both domains query
# the same facility/users/project/client/entity/organisation tables, so those
# carry across a domain switch safely. Used by build_context_block()'s domain
# gate and by the entity_memory recall check below.
_DOMAIN_EXCLUSIVE = frozenset({"inspection", "audit", "cycle", "plan"})

# entity_memory cap — deliberately much larger than MAX_TURNS since these are
# small labeled records (no raw text), not full turns. Bounds a very long
# session without limiting name-based recall to only the last few messages.
_ENTITY_MEMORY_CAP = 30


class ConversationSession:
    MAX_TURNS = 8

    def __init__(self):
        self.turns: list[ConversationTurn] = []
        # Persists the MOST RECENT active_subject_type/value regardless of
        # MAX_TURNS eviction — decouples "how much raw history to show" from
        # "what are we currently focused on". Never explicitly cleared on a
        # domain switch; build_context_block() simply won't use a
        # domain-exclusive entry when it doesn't match the current domain.
        self.current_scope: Optional[dict] = None
        # Append-only record of every distinct entity resolved this session
        # (same shape as current_scope), capped at _ENTITY_MEMORY_CAP. Lets a
        # later turn that explicitly re-names something from several turns
        # back ("what about Facility A again") get it back even though it's
        # no longer the current_scope and long since evicted from self.turns.
        self.entity_memory: list[dict] = []
        # Plain-language "here's what I resolved your reference to" string,
        # computed fresh by build_context_block() below whenever THIS turn's
        # question contains an ordinal/back-reference — e.g. "Showing results
        # for 'del 01' (3rd item in your previous list)". Read by
        # synthesis.py's _synthesize() and prepended to the deterministic
        # row-dump answer text. Exists because that answer text is built by a
        # plain Python formatter, not composed by the LLM — a prompt-only
        # instruction telling the LLM to "state what you resolved" can never
        # surface through that path, since the formatter never reads the
        # LLM's own final_answer/thought text. Reset to "" at the top of
        # every build_context_block() call so a resolution from an earlier
        # turn never leaks into a later turn's answer that didn't need one.
        self.pending_resolved_disclosure: str = ""

    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
        if len(self.turns) > self.MAX_TURNS:
            self.turns = self.turns[-self.MAX_TURNS:]
        if turn.active_subject_type and turn.active_subject_value:
            scope = {
                "type": turn.active_subject_type,
                "value": turn.active_subject_value,
                "label": turn.active_subject_label,
                "table": turn.active_subject_table,
                "alias": turn.active_subject_alias,
                "domain": turn.active_subject_domain,
                "id": turn.active_subject_id,
            }
            self.current_scope = scope
            _dup = (
                self.entity_memory
                and self.entity_memory[-1]["type"] == scope["type"]
                and self.entity_memory[-1]["value"] == scope["value"]
            )
            if not _dup:
                self.entity_memory.append(scope)
                if len(self.entity_memory) > _ENTITY_MEMORY_CAP:
                    self.entity_memory = self.entity_memory[-_ENTITY_MEMORY_CAP:]

    @property
    def last_turn(self) -> Optional[ConversationTurn]:
        return self.turns[-1] if self.turns else None

    @property
    def last_domain(self) -> str:
        """Most recent turn's resolved domain, walking backwards — "" if no
        turn has one yet (fresh session, or every turn so far was a parse
        error). This is what DomainRouter's Tier 2 (stickiness) reads: a
        follow-up with no domain keyword stays in whatever domain the last
        resolved turn used."""
        for t in reversed(self.turns):
            if t.domain:
                return t.domain
        return ""

    def build_context_block(self, current_question: str = "", current_domain: str = "") -> str:
        self.pending_resolved_disclosure = ""
        if not self.turns:
            return ""

        # NOTE: bare "that"/"this" are deliberately excluded — they double as
        # relative pronouns/determiners in ordinary questions ("mitigative actions
        # THAT are overdue", "show me THIS month's data") and caused the sticky
        # active-context gate below to false-positive-match plain global questions.
        # Demonstrative reference to a specific entity is still caught via the
        # explicit "that/this <entity>" phrases.
        # "here" IS included unconditionally (unlike bare "there", left out below) —
        # "here" is almost never anything but a location follow-up in this app's
        # question shapes ("was any audit conducted here last month").
        _FOLLOWUP_RE = re.compile(
            r"\b(it|its|them|they|she|he|her|his|their|here|"
            r"(?:that|this)\s+(?:inspection|site|facility|form|inspector|cycle|"
            r"person|project|client|action|schedule)|"
            r"the\s+same|the\s+above|one\s+above|that\s+one|this\s+one|"
            r"from\s+there|more|rest|remaining|"
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
            if t.domain:
                lines.append(f"  Domain: {t.domain}")
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
            if t.audit_ids:
                lines.append(f"  ★ audit_id(s) from this result: {t.audit_ids[:5]}")
            if t.cycle_ids:
                lines.append(f"  ★ cycle_id(s) from this result: {t.cycle_ids[:3]}")
            if t.facility_names:
                lines.append(f"  ★ facility(s) from this result: {t.facility_names[:5]}")
            if t.inspector_names:
                lines.append(f"  ★ inspector(s) from this result: {t.inspector_names[:5]}")
            if t.single_row_values:
                lines.append(f"  ★ single-row result: {t.single_row_values}")
            if t.row_titles:
                # Capped preview, unlike the sibling ★ lists above this was
                # historically rendered in full — harmless for the short
                # id-code lists synthesis.py usually captures, but one of its
                # three capture paths (the observation/submission_id path)
                # stores full free-text answer_text values, and a turn
                # listing many rows (LIMIT 100 is this codebase's standard
                # pattern) echoed the ENTIRE list into every subsequent
                # turn's context for up to MAX_TURNS turns — confirmed live
                # as a real contributor to a reasoning-LLM prompt hitting
                # num_ctx and truncating its own tool-call JSON mid-object.
                # Render-only: ordinal resolution below indexes the full
                # stored row_titles/row_ids on the ConversationTurn, never
                # this preview, so "the 47th one" still works unchanged.
                _preview = [str(x)[:80] for x in t.row_titles[:10]]
                _more = f" (+{len(t.row_titles) - 10} more)" if len(t.row_titles) > 10 else ""
                lines.append(f"  ★ items from this result, IN ORDER: {_preview}{_more}")
            # Show active subject for each turn
            if t.active_subject_type and t.active_subject_value:
                lines.append(
                    f"  ◆ ACTIVE SUBJECT: {t.active_subject_type} = "
                    f"'{t.active_subject_label or t.active_subject_value}' "
                    f"(exact value: {t.active_subject_value})"
                )
        lines.append("")

        # ── Ordinal follow-up ("the first one", "the 2nd one") ────────────────
        # Confirmed live 2026-07-30: "for the first one show the approval
        # status" (right after a 7-row audit_plan listing) resolved against a
        # ◆ ACTIVE SUBJECT pointer that had been silently INHERITED from an
        # unrelated audit_cycle list 5 turns earlier (that plan-listing turn
        # set no active_subject_type of its own, since it returned multiple
        # rows rather than a single entity — see synthesis.py's inheritance
        # branch). The model picked the first row of that stale cycle list
        # instead of the first row of the plan list it had just been shown,
        # answering a different question entirely (portfolio approvals for a
        # 2030 UPCOMING cycle) than the one asked. An ordinal word almost
        # always refers to the MOST RECENT listing shown to the user, not to
        # ACTIVE SUBJECT (which can persist across many unrelated turns) and
        # not to an older turn's list — regardless of what ACTIVE SUBJECT
        # says below.
        # Word-form ordinals up to twentieth — matches the raised row_titles/
        # row_ids capture cap in synthesis.py (up to 20, the same limit the
        # deterministic answer formatter actually displays to the user via
        # max_display_rows). Numeric-suffix ordinals (6th, 10th, 23rd, ...)
        # are handled generically below via regex, not hardcoded — a fixed
        # 1st-5th word list was confirmed live 2026-07-31 to silently fail
        # "show observations for the 10th one" (no match at all, since "10th"
        # wasn't in the list): the model got no ORDINAL context, fell back to
        # re-listing the same rows on its own, and the turn ended there
        # without ever actually answering what was asked.
        _ORDINAL_WORDS = (
            "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
            "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
            "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
            "nineteenth", "twentieth",
        )
        _ORDINAL_WORD_TO_IDX = {w: i for i, w in enumerate(_ORDINAL_WORDS)}

        def _ordinal_to_idx(word: str):
            # \s* between digit and suffix tolerates "2 nd" as well as "2nd"
            # — confirmed live 2026-07-31: "list all audit plans by the 2 nd
            # person in the list" had this exact space (a plausible typo, not
            # contrived), never matched, so NO structured ORDINAL context got
            # injected at all. Unlike other ordinal misses this session where
            # the model's own conversation-history fallback still landed on
            # a reasonable answer, this one sent it into a genuine stuck
            # reasoning loop — 9 LLM calls, ~165K tokens, 122 seconds,
            # success=False. Worth the tolerance given a real failure this
            # severe, not just a theoretical gap.
            w = re.sub(r"\s+", "", word.lower())
            m = re.match(r"^(\d+)(?:st|nd|rd|th)$", w)
            if m:
                n = int(m.group(1))
                return n - 1 if n >= 1 else None
            return _ORDINAL_WORD_TO_IDX.get(w)

        def _ord_suffix(idx0: int) -> str:
            pos = idx0 + 1
            if 10 <= pos % 100 <= 20:
                return "th"
            return {1: "st", 2: "nd", 3: "rd"}.get(pos % 10, "th")

        # \s* between digit and suffix tolerates "2 nd" as well as "2nd" — see
        # _ordinal_to_idx's matching comment for why this matters live.
        #
        # Negative lookahead excludes "First/Second/Third(-)party" — confirmed
        # live 2026-08-04: "list all schedules under Second party audit"
        # matched "Second" as an ordinal, resolving against an unrelated
        # prior turn's list (e.g. the 2nd row of a plans listing) even
        # though the question is a plain name filter — 'First Party',
        # 'Second Party', and 'Third party / ISO audit' are real audit_type
        # values, and 'Third-party inspection' a real inspection_type value
        # (confirmed against the live DB), so this collision hits BOTH
        # domains. Only "party"/"parties" is excluded, not the ordinal words
        # generally — "party" never appears as a genuine ordinal-referenced
        # noun in this schema, so this can't suppress a real ordinal follow-up.
        _ORDINAL_RE = re.compile(
            r"\b(the\s+)?(\d+\s*(?:st|nd|rd|th)|" + "|".join(_ORDINAL_WORDS) + r")"
            r"(?!-?\s*part(?:y|ies)\b)"
            r"(\s+(one|row|result|plan|entry|item))?\b",
            re.IGNORECASE,
        )
        _ordinal_match = _ORDINAL_RE.search(current_question)
        if _ordinal_match:
            # Noun bound to THIS ordinal — only the few words immediately
            # AFTER the match ("the 6th AUDITOR"), never the whole question,
            # so an unrelated noun earlier in the sentence ("list all audit
            # PLANS under the 6th auditor") isn't mistaken for it. Confirmed
            # live (qid=081c7a81): without this type check, "the 6th
            # auditor" silently resolved against the most recent list
            # turn regardless of type (a CYCLES list, in that case), using
            # one cycle's UUID as if it were an auditor's id — a
            # well-formed but empty result, wrong with no error.
            _ord_tail = current_question[_ordinal_match.end():].strip()
            _ord_noun_type = entity_type_for_ordinal_noun(
                " ".join(_ord_tail.split()[:6]), current_domain
            )

            if _ord_noun_type:
                _last_with_items = next(
                    (t for t in reversed(self.turns)
                     if t.row_titles and t.row_entity_type == _ord_noun_type),
                    None,
                )
                if _last_with_items is None:
                    # A noun was named but no matching-type list exists
                    # anywhere in history — ask, never silently resolve
                    # against a differently-typed list.
                    lines.append(
                        f"ORDINAL REFERENCE NEEDS CLARIFICATION "
                        f"('{_ordinal_match.group(0).strip()}') — the question "
                        f"refers to an ordinal position among '{_ord_noun_type}'-"
                        f"type items, but no earlier turn in this conversation "
                        f"actually listed items of that type. Do NOT resolve "
                        f"this against any OTHER list shown above (even if one "
                        f"exists) — it is a different kind of item, and "
                        f"guessing would silently answer the wrong question. "
                        f"YOU MUST call final_answer NOW asking the user to "
                        f"first show/list the '{_ord_noun_type}' items they "
                        f"mean — do NOT call generate_sql.\n"
                    )
            else:
                # No recognized noun — bare ordinal ("the 4th one"). Exact
                # prior behavior, unchanged: most recent turn with any
                # row_titles, regardless of type.
                _last_with_items = next(
                    (t for t in reversed(self.turns) if t.row_titles), None
                )

            if _last_with_items:
                _ord_word = _ordinal_match.group(2).lower()
                _idx = _ordinal_to_idx(_ord_word)
                _target_id = ""
                _target_col = "id"
                if (_idx is not None
                        and _idx < len(_last_with_items.row_titles)
                        and _idx < len(_last_with_items.row_ids)):
                    _target_id = _last_with_items.row_ids[_idx]
                    if _idx < len(_last_with_items.row_id_columns):
                        _target_col = _last_with_items.row_id_columns[_idx] or "id"

                if _target_id:
                    # Deterministic resolution — inject_row_identity() (see
                    # validation/identity.py) paired this exact row with its
                    # real id at query time, so there's no ambiguity window
                    # left to warn about even if the display value below
                    # happens to be shared by another row (confirmed live:
                    # 'Delegation testing' appears twice in the same 7-row
                    # plan listing — two different plans, same title).
                    self.pending_resolved_disclosure = (
                        f"Showing results for {_last_with_items.row_titles[_idx]!r} "
                        f"({_idx + 1}{_ord_suffix(_idx)} item in your previous list):"
                    )
                    _non_pk_note = (
                        # Confirmed live: the reasoning model correctly received
                        # this exact UUID and correctly resolved "the 1st one",
                        # but then wrote WHERE arep.audit_id = '<uuid>' instead
                        # of WHERE arep.id = '<uuid>' — because "id" here is a
                        # bare, ungrounded token; nowhere ELSE in this same
                        # prompt does the model ever see a column literally
                        # named "id" for this row, whereas it has repeatedly
                        # seen audit_id/schedule_id/corrective_action_id/
                        # inspection_id as ITS identifying column (★ items
                        # list, ACTIVE CONTEXT, prior SQL). The query executed
                        # without error and silently returned 0 rows. Naming
                        # and ruling out those display columns explicitly
                        # (rather than just saying "id") is the fix — a
                        # deterministic AST guardrail
                        # (DISPLAY_CODE_UUID_COMPARE in ast_checks.py) also
                        # blocks this shape as a backstop either way.
                        f" NOTE: '{_target_col}' here means the row's actual "
                        f"PRIMARY KEY column (<alias>.id) — it is a different "
                        f"column from audit_id/schedule_id/corrective_action_id/"
                        f"inspection_id, the human-readable VARCHAR display "
                        f"codes you may have seen for this same row elsewhere "
                        f"in this context. A UUID like '{_target_id}' will "
                        f"never match one of those display columns — it "
                        f"silently returns 0 rows instead of erroring.\n"
                        if _target_col == "id" else
                        f" NOTE: {_target_col} is NOT a table's primary key — it's a "
                        f"correlation value shared across tables (e.g. "
                        f"ai_audit_answers.submission_id / ai_answers.submission_id "
                        f"also matches a corrective action's observation_id, cast as "
                        f"::varchar). Filter directly on the exact column this value "
                        f"came from, do not treat it as some table's own .id.\n"
                    )
                    lines.append(
                        f"ORDINAL RESOLVED ('{_ordinal_match.group(0).strip()}') — "
                        f"row {_idx + 1} of the list from the MOST RECENT turn that "
                        f"listed multiple rows = {_target_col} '{_target_id}' (display "
                        f"value {_last_with_items.row_titles[_idx]!r}).\n"
                        f"  Use WHERE <alias>.{_target_col} = '{_target_id}' — NOT "
                        f"title/ILIKE, even if this display value appears more than "
                        f"once in that list. Do NOT resolve against any ◆ ACTIVE "
                        f"SUBJECT below — that pointer may be stale, carried over "
                        f"from an unrelated earlier turn.\n"
                        f"{_non_pk_note}"
                        f"  (Your answer will automatically be prefixed with a "
                        f"disclosure naming this resolved item — see "
                        f"pending_resolved_disclosure — you do not need to restate "
                        f"it yourself.)\n"
                    )
                else:
                    if _idx is not None and _idx < len(_last_with_items.row_titles):
                        self.pending_resolved_disclosure = (
                            f"Showing results for {_last_with_items.row_titles[_idx]!r} "
                            f"({_idx + 1}{_ord_suffix(_idx)} item in your previous list):"
                        )
                    lines.append(
                        f"ORDINAL REFERENCE DETECTED ('{_ordinal_match.group(0).strip()}') — "
                        f"resolve it against the ★ items list from the MOST RECENT turn that "
                        f"listed multiple rows ({_last_with_items.row_titles}), NOT against any "
                        f"◆ ACTIVE SUBJECT below — that pointer may be stale, carried over from "
                        f"an unrelated earlier turn. 'the first one' = "
                        f"{_last_with_items.row_titles[0]!r}.\n"
                        f"  (Your answer will automatically be prefixed with a disclosure "
                        f"naming the resolved item, if one could be determined — see "
                        f"pending_resolved_disclosure — you do not need to restate it "
                        f"yourself.)\n"
                    )
                    # DUPLICATE-WITHIN-LIST TRAP — fallback path only, for
                    # query shapes inject_row_identity() skips (e.g. GROUP
                    # BY — see identity.py). Confirmed live 2026-07-30: the
                    # SAME 7-row plan listing (a single ONGOING cycle)
                    # contains 'Delegation testing' TWICE, at positions 3
                    # and 6 — two genuinely different audit_plan rows
                    # (different audit_type/category) that happen to share
                    # an identical title. Resolving an ordinal to a title
                    # string and then filtering follow-up SQL by WHERE
                    # ap.title = '<that title>' (or ILIKE) would silently
                    # match BOTH rows and merge two different plans' data
                    # into one answer — wrong, with no error.
                    if _idx is not None and _idx < len(_last_with_items.row_titles):
                        _target = _last_with_items.row_titles[_idx]
                        if _last_with_items.row_titles.count(_target) > 1:
                            lines.append(
                                f"WARNING — '{_target}' appears MORE THAN ONCE in that same "
                                f"list (not a data error — confirmed live: distinct rows can "
                                f"share an identical title/name). Filtering by "
                                f"WHERE ... = '{_target}' alone would match ALL of them and "
                                f"silently merge multiple different records into one answer. "
                                f"Do NOT filter by this value alone — either add another "
                                f"filter from a column also shown in that same row (e.g. "
                                f"audit_type/category) to pin down this ONE row specifically, "
                                f"or re-run the LAST SQL with the SAME ORDER BY and "
                                f"OFFSET {_idx} LIMIT 1 to reselect this exact row by "
                                f"position instead of by name.\n"
                            )

        # ── Pending clarifying question this turn's answer may be resolving ──
        # Confirmed live 2026-07-29 (qid=c3004b3d → qid=7d5d343d): "top 5 capex
        # converters" got a clarifying question back ("if you mean FM Managers...
        # just say which group and metric"); the reply "FM manafers" should have
        # resolved it and continued the ORIGINAL request, but instead got treated
        # as a brand-new, unrelated ambiguous-person-name search — the original
        # request was silently dropped and never answered. Distinct from the
        # DISAMBIGUATION RESOLVED/CONTEXT blocks below, which only cover the
        # SQL-driven "N candidates found" flow — this covers a clarifying
        # question the reasoning model invented on its own via final_answer.
        _last_for_clarification = self.turns[-1]
        if _last_for_clarification.clarification_origin_question:
            lines.append(
                f"PENDING CLARIFICATION — your previous turn asked a clarifying "
                f"question in response to "
                f"'{_last_for_clarification.clarification_origin_question}': "
                f"\"{_last_for_clarification.answer}\"\n"
                f"  The current message is very likely answering THAT question, not a "
                f"brand-new unrelated request. Resolve it using the current message, "
                f"then continue answering the ORIGINAL request "
                f"('{_last_for_clarification.clarification_origin_question}') — do NOT "
                f"treat a short reply like a role/group name as an ambiguous-person-name "
                f"search unless the current message clearly asks about a specific person.\n"
            )

        # ── Resolve active subject across all turns ──────────────────────────
        # "go back to that inspector" can re-anchor to an earlier turn's subject.
        # Domain-exclusive types (inspection/audit/cycle — see _DOMAIN_EXCLUSIVE)
        # are only usable when they originated in the domain we're currently
        # answering in; "facility"/"inspector"/"entity" are always usable.
        # Confirmed live: without this gate, an inspection-scoped subject got
        # handed to the audit LLM as "use WHERE ir.inspection_id = ..." —
        # ir/inspection_report doesn't exist in the audit schema at all.
        active_type = ""
        active_value = ""
        active_label = ""
        active_table = ""
        active_alias = ""
        active_id = ""

        def _domain_ok(scope: dict) -> bool:
            return (
                scope["type"] not in _DOMAIN_EXCLUSIVE
                or not current_domain
                or scope["domain"] == current_domain
            )

        # "go back" or "back to" signals: re-anchor to the earliest matching subject
        _BACK_RE = re.compile(r"\b(go\s+back|back\s+to|return\s+to)\b", re.IGNORECASE)
        if _BACK_RE.search(current_question):
            # Look for the first (oldest) DOMAIN-COMPATIBLE active subject in
            # history — skip past an incompatible one rather than stopping dead,
            # so "go back" can still find an earlier, still-usable subject.
            for t in self.turns:
                if t.active_subject_type and t.active_subject_value:
                    _cand = {
                        "type": t.active_subject_type, "domain": t.active_subject_domain,
                    }
                    if not _domain_ok(_cand):
                        continue
                    active_type = t.active_subject_type
                    active_value = t.active_subject_value
                    active_label = t.active_subject_label
                    active_table = t.active_subject_table
                    active_alias = t.active_subject_alias
                    active_id = t.active_subject_id
                    break
        else:
            # Normal: use the persistent current_scope pointer (survives turns
            # aging out of MAX_TURNS — see ConversationSession.add_turn), not a
            # walk over self.turns.
            if self.current_scope and _domain_ok(self.current_scope):
                active_type = self.current_scope["type"]
                active_value = self.current_scope["value"]
                active_label = self.current_scope["label"]
                active_table = self.current_scope["table"]
                active_alias = self.current_scope["alias"]
                active_id = self.current_scope.get("id", "") or ""

        # self.turns is guaranteed non-empty (early-return above), so `last` is
        # always defined for both the fallback block and the disambiguation
        # STOP-guard below.
        last = self.turns[-1]

        # ── Emit the scoped context block ────────────────────────────────────
        def _lookup_entity_id(label: str) -> tuple[str, list[str]]:
            """
            Find the id captured for `label` in the most recent turn whose
            entity_disambiguation_candidates contains an exact (case-insensitive)
            name match — mirrors the `_last_with_uid` lookup the `inspector`
            branch below already uses for people, so facility/project/client/
            organisation follow-ups can filter by exact id too instead of
            re-searching by ILIKE name every time.

            Returns (id, all_matching_ids). Confirmed live 2026-07-30: this
            used to return the FIRST match silently — a name collision (two
            different facility/project/client/organisation rows sharing a
            display name, same class of bug as audit_plan.title duplicates)
            would resolve to an arbitrary one of them with no warning, unlike
            the equivalent people-search flow below, which already correctly
            detects `_ambiguous_duplicate_names` and hard-STOPs instead of
            guessing. Collect ALL matches so the facility/entity branches can
            apply the same STOP guard rather than trusting `next()`'s pick.
            """
            for t in reversed(self.turns):
                matches = [
                    c for c in t.entity_disambiguation_candidates
                    if c.get("name", "").strip().lower() == label.lower() and c.get("id")
                ]
                if matches:
                    _ids = [m["id"] for m in matches]
                    return _ids[0], _ids
            return "", []

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
            elif active_type == "audit":
                lines.append(
                    f"ACTIVE CONTEXT — The conversation is about audit '{active_value}'."
                    f" When the user says 'it', 'that audit', 'this audit'"
                    f" — use WHERE arep.audit_id = '{active_value}'"
                    f" (join audit_report arep as needed for the target table)."
                )
            elif active_type == "plan":
                # Prefer an exact id filter — audit_plan.title has NO uniqueness
                # constraint at the DB level (only `id` does), and the live data
                # already has duplicate active titles (confirmed 2026-07-30:
                # 'Delegation plan -1', 'Delegation testing' — 2 active rows
                # each). Filtering by title/ILIKE risks silently scoping to
                # whichever duplicate the DB happens to return first, with no
                # error. See synthesis.py's plan_ids lookup for how this gets
                # resolved (or flagged ambiguous) at turn-finish time.
                # Prefer active_id (M2: captured directly via the __id
                # sibling column inject_row_identity() attaches, or already
                # resolved by the ordinal-resolution block above) — only
                # fall back to the plan_ids DB round-trip below when
                # injection didn't run for that turn's query shape (e.g. a
                # GROUP BY).
                _last_with_pid = next(
                    (t for t in reversed(self.turns) if t.plan_ids),
                    None
                )
                _pid = active_id
                if not _pid and _last_with_pid and len(_last_with_pid.plan_ids) == 1:
                    _pid = _last_with_pid.plan_ids[0]
                if _pid:
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is about audit plan "
                        f"'{active_value}' (resolved id='{_pid}'). When the user "
                        f"says 'it', 'that plan', 'this plan', or asks an "
                        f"ATTRIBUTE question about 'the audit' (auditee, auditor, "
                        f"facility, schedule date — NOT a status/count/lifecycle "
                        f"question) with no other plan/report named — filter "
                        f"EXACTLY by WHERE ap.id = '{_pid}' — NOT title/ILIKE "
                        f"(join audit_schedule for auditee/auditor/facility, via "
                        f"asch.audit_plan_id = ap.id). Do NOT ask the user to "
                        f"re-specify which audit/plan they mean."
                    )
                else:
                    _ambiguous_note = ""
                    if _last_with_pid is None:
                        _prev_plan_turn = next(
                            (t for t in reversed(self.turns)
                             if t.active_subject_type == "plan"
                             and t.active_subject_value == active_value),
                            None
                        )
                        if _prev_plan_turn and _prev_plan_turn.plan_title_ambiguous:
                            _ambiguous_note = (
                                f" WARNING: more than one ACTIVE audit_plan "
                                f"currently shares the title '{active_value}' — "
                                f"a title/ILIKE filter is NOT guaranteed to hit "
                                f"the one the user actually means. If precision "
                                f"matters, ask the user to disambiguate (by "
                                f"cycle, status, or created date) rather than "
                                f"silently picking one."
                            )
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is about audit plan "
                        f"'{active_value}'. When the user says 'it', 'that plan', "
                        f"'this plan', or asks an ATTRIBUTE question about 'the "
                        f"audit' (auditee, auditor, facility, schedule date — NOT "
                        f"a status/count/lifecycle question) with no other plan/"
                        f"report named — use WHERE ap.title = "
                        f"'{_ilike_escape(active_value)}' (join audit_plan ap as "
                        f"needed; join audit_schedule for auditee/auditor/"
                        f"facility, via asch.audit_plan_id = ap.id). Do NOT ask "
                        f"the user to re-specify which audit/plan they mean."
                        f"{_ambiguous_note}"
                    )
            elif active_type == "facility":
                # Prefer an exact id filter if one is available. Two sources,
                # checked in order: (1) a facility_id/fac_id column captured
                # directly from an ordinary query result (_last_with_fid —
                # mirrors _last_with_uid for inspectors above), (2) a name
                # resolved via the STEP 1 disambiguation flow (_lookup_entity_id,
                # unchanged). Only falls back to ILIKE if neither has an id.
                # M3: active_id (M2's direct capture — either an existing
                # facility_id/fac_id column or the __id sibling identity
                # injection attached to facility.name) takes priority over
                # both fallbacks below.
                _last_with_fid = next(
                    (t for t in reversed(self.turns) if t.facility_ids),
                    None
                )
                _fid = active_id
                _fid_matches: list[str] = []
                if not _fid and _last_with_fid and len(_last_with_fid.facility_ids) == 1:
                    _fid = _last_with_fid.facility_ids[0]
                if not _fid:
                    _fid, _fid_matches = _lookup_entity_id(active_label)
                if len(_fid_matches) > 1:
                    _dup_fids = ", ".join(f"'{f}'" for f in _fid_matches)
                    lines.append(
                        f"⚠️ STOP — '{active_label}' matches {len(_fid_matches)} DIFFERENT facility"
                        f" records from the previous list (same name, different rows:"
                        f" facility id IN ({_dup_fids})).\n"
                        f"  YOU MUST call final_answer NOW asking the user to disambiguate further"
                        f" (e.g. by project, client, or code) — do NOT pick one of these ids"
                        f" arbitrarily and do NOT call generate_sql."
                    )
                elif _fid:
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is scoped to facility '{active_label}'"
                        f" (resolved facility_id='{_fid}')."
                        f" ALL follow-up questions about mitigative actions, inspections, scores,"
                        f" or form answers should filter EXACTLY by WHERE fac.id = '{_fid}'"
                        f" — NOT ILIKE — unless the user explicitly asks for a global/all-facilities view."
                        f" Do NOT query globally when the context is scoped to this facility."
                        f" If this turn's own question isn't actually about mitigative actions/"
                        f"inspections/scores/form answers for a facility at all — e.g. it's asking"
                        f" for a global ranking, count, or comparison — IGNORE this scope and answer"
                        f" globally instead; it may be stale, carried over from an unrelated earlier turn."
                    )
                else:
                    lines.append(
                        f"ACTIVE CONTEXT — The conversation is scoped to facility '{active_label}'."
                        f" ALL follow-up questions about mitigative actions, inspections, scores,"
                        f" or form answers should include a facility filter"
                        f" (fac.name ILIKE '%{_ilike_escape(active_label)}%') unless the user explicitly asks"
                        f" for a global/all-facilities view."
                        f" Do NOT query globally when the context is scoped to this facility."
                        f" If this turn's own question isn't actually about mitigative actions/"
                        f"inspections/scores/form answers for a facility at all — e.g. it's asking"
                        f" for a global ranking, count, or comparison — IGNORE this scope and answer"
                        f" globally instead; it may be stale, carried over from an unrelated earlier turn."
                    )
            elif active_type == "entity":
                # Generic scope for project/client/organisation — same idea as
                # facility above, parameterized by whichever table/alias the
                # STEP 1 disambiguation actually searched.
                _tbl = active_table or "entity"
                _als = active_alias or "e"
                _eid = active_id
                _eid_matches: list[str] = []
                if not _eid:
                    _eid, _eid_matches = _lookup_entity_id(active_label)
                if len(_eid_matches) > 1:
                    _dup_eids = ", ".join(f"'{e}'" for e in _eid_matches)
                    lines.append(
                        f"⚠️ STOP — '{active_label}' matches {len(_eid_matches)} DIFFERENT {_tbl}"
                        f" records from the previous list (same name, different rows:"
                        f" {_tbl} id IN ({_dup_eids})).\n"
                        f"  YOU MUST call final_answer NOW asking the user to disambiguate further"
                        f" — do NOT pick one of these ids arbitrarily and do NOT call generate_sql."
                    )
                elif _eid:
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
            # inspection_ids/audit_ids are domain-exclusive (same reasoning as
            # _DOMAIN_EXCLUSIVE above — only gate THESE, not facility_names/
            # single_row_values below, which are safe regardless of domain).
            # Confirmed live: this fallback reads last.inspection_ids/audit_ids
            # off self.turns[-1] unconditionally — a second, independent leak
            # path from the primary gate above, since `last` isn't necessarily
            # in the domain we're currently answering in.
            _domain_compat = (not current_domain) or (last.domain == current_domain)
            if _domain_compat and last.inspection_ids and len(last.inspection_ids) == 1:
                lines.append(
                    f"CONTEXT — The conversation is about inspection '{last.inspection_ids[0]}'."
                    f" When the user says 'that inspection', 'it', 'that site' — use"
                    f" WHERE ir.inspection_id = '{last.inspection_ids[0]}'"
                    f" as an exact filter."
                )
            elif _domain_compat and last.inspection_ids:
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
            elif _domain_compat and last.audit_ids and len(last.audit_ids) == 1:
                lines.append(
                    f"CONTEXT — The conversation is about audit '{last.audit_ids[0]}'."
                    f" When the user says 'that audit', 'it', 'this audit' — use"
                    f" WHERE arep.audit_id = '{last.audit_ids[0]}'"
                    f" as an exact filter."
                )
            elif _domain_compat and last.audit_ids:
                ids_str = ", ".join(f"\'{i}\'" for i in last.audit_ids[:5])
                if last.total_count > len(last.audit_ids):
                    # Same "sample vs complete set" distinction as inspection_ids
                    # above — the captured IDs may only be a slice of a larger
                    # result, so an IN-list can't stand in for "all"/"those".
                    prev_sql_snippet = " ".join(last.sql.split())[:500] if last.sql else "(not available)"
                    lines.append(
                        f"CONTEXT — The previous result had {last.total_count} total rows; "
                        f"only {len(last.audit_ids)} sample IDs were captured ({ids_str})."
                        f" These {len(last.audit_ids)} IDs are NOT the full set.\n"
                        f"  If the user asks 'how many of these/those/them are X' (an aggregate"
                        f" over the FULL {last.total_count}-row result), do NOT filter by"
                        f" audit_id IN (...) — that covers only"
                        f" {len(last.audit_ids)} of {last.total_count} rows.\n"
                        f"  INSTEAD: reuse the SAME FROM/JOIN/WHERE filters as the previous"
                        f" query below, add the new condition, and"
                        f" return a fresh COUNT(*) — a complete recount, not a sample filter.\n"
                        f"  Previous SQL: {prev_sql_snippet}"
                    )
                else:
                    # The captured IDs ARE the complete result set — IN-list is
                    # valid here. This is the exact scenario that broke: 2
                    # audit_ids captured, question says "across the above two
                    # audits" — the model must reuse THESE real IDs, not invent
                    # a plausible-looking one.
                    lines.append(
                        f"CONTEXT — The conversation involves audits: {ids_str}"
                        f" (this is the COMPLETE result set, {last.total_count or len(last.audit_ids)} total)."
                        f" Filter by arep.audit_id IN ({ids_str}) when user references 'those'/'the above'."
                    )
            elif last.facility_names and len(last.facility_names) == 1:
                lines.append(
                    f"CONTEXT — The conversation is about facility '{last.facility_names[0]}'."
                    f" Filter by fac.name ILIKE '%{_ilike_escape(last.facility_names[0])}%'."
                )
            elif last.single_row_values:
                # A single-row result is EITHER a genuine specific-record lookup OR a
                # scalar aggregate (COUNT/SUM/AVG/...) summarizing many underlying rows —
                # these need completely different follow-up handling, and the generic
                # message below used to conflate them. Confirmed live: "how many
                # corrective actions have not got adequacy approval" (a COUNT) followed
                # by "show more details regarding those" — with no distinguishing hint,
                # the model ignored the count entirely and drilled into a DIFFERENT,
                # older turn's named corrective action instead, because a bare number
                # has nothing to "show details about" but an earlier specific ID does.
                _is_aggregate = bool(
                    last.sql
                    and re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', last.sql, re.IGNORECASE)
                    and not re.search(r'\bGROUP\s+BY\b', last.sql, re.IGNORECASE)
                )
                if _is_aggregate and last.sql:
                    _agg_sql_snippet = " ".join(last.sql.split())[:500]
                    lines.append(
                        f"CONTEXT — The last query was a SCALAR AGGREGATE (a count/sum/average "
                        f"computed over many underlying rows, NOT a single specific record): "
                        f"{last.single_row_values}.\n"
                        f"  Previous SQL: {_agg_sql_snippet}\n"
                        f"  If the user now says 'show more details', 'show them', 'list those', "
                        f"'which ones' — they mean the INDIVIDUAL ROWS behind THIS aggregate. "
                        f"Reuse the SAME FROM/JOIN/WHERE filters as the previous SQL above, but "
                        f"SELECT the actual row-level columns instead of the aggregate function. "
                        f"Do NOT go back to an earlier, unrelated turn's specific entity just "
                        f"because it had a nameable ID — this aggregate is the most recent turn "
                        f"and is what 'those'/'them' refers to."
                    )
                else:
                    lines.append(f"CONTEXT — Last query returned a single result: {last.single_row_values}")

        # ── Explicit-name recall — reaches further back than current_scope ──
        # current_scope (above) only ever holds the MOST RECENT focus, so it
        # can't answer "wait, what about Facility A again" once the
        # conversation has since moved on to Facility B and Facility C — this
        # is a genuinely different case from the pronoun-based ACTIVE CONTEXT
        # above: the user is explicitly re-NAMING something from earlier, not
        # implicitly referring to "whatever we're currently on". Search the
        # full entity_memory (much larger than MAX_TURNS — see
        # ConversationSession.add_turn) for any label the user's question
        # names directly, same domain-compatibility rule as everywhere else.
        _recalled = []
        for _ent in self.entity_memory:
            if not _ent["label"] or len(_ent["label"]) < 3:
                continue  # too short — avoid noisy substring-style false matches
            if _ent["type"] == active_type and _ent["value"] == active_value:
                continue  # already surfaced via ACTIVE CONTEXT above — don't double-print
            if not _domain_ok(_ent):
                continue
            if re.search(rf"\b{re.escape(_ent['label'])}\b", current_question, re.IGNORECASE):
                _recalled.append(_ent)
        if _recalled:
            lines.append("RECALLED FROM EARLIER IN THIS CONVERSATION:")
            for _ent in _recalled:
                lines.append(
                    f"  • {_ent['type']} '{_ent['label']}' (exact value: {_ent['value']})"
                    f" — matches a name in your current question, even though it's not"
                    f" the most recent topic. Use it if relevant to this question."
                )
            lines.append("")

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
        lines.append(
            "AUGMENTATION: If the user says 'show along with X', 'also show Y', 'include Z',"
            " 'add the Y column' — reuse the LAST SQL's FROM/JOIN/WHERE EXACTLY as-is (the"
            " same filters, the same join directions — a LEFT JOIN stays a LEFT JOIN, do not"
            " make it an INNER JOIN) and only ADD the newly-requested column/join to the"
            " SELECT list. Confirmed live 2026-07-29: 'show along with auditees' regenerated"
            " the query from scratch instead of extending it — it silently dropped the"
            " previous query's schedule_status != 'COMPLETED' filter and switched the"
            " auditor/auditee/facility joins from LEFT to INNER, changing the row count from"
            " 10 to 17 for a question that only asked to add one column. Do NOT change or"
            " drop any filter, and do NOT change any join's direction, unless the user"
            " explicitly asked for that change."
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)
