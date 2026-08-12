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




# NOTE: bare "that"/"this" are deliberately excluded — they double as
# relative pronouns/determiners in ordinary questions ("mitigative actions
# THAT are overdue", "show me THIS month's data") and caused the sticky
# active-context gate below to false-positive-match plain global questions.
# Demonstrative reference to a specific entity is still caught via the
# explicit "that/this <entity>" phrases.
# "here" IS included unconditionally (unlike bare "there", left out below) —
# "here" is almost never anything but a location follow-up in this app's
# question shapes ("was any audit conducted here last month").
#
# Module-level (not local to build_context_block) — confirmed live
# 2026-08-07 (debug_llm_20260807_153128): needed at module scope so
# loop.py's final_answer backstop (see _COUNT_FOLLOWUP_RE below) can import
# and reuse it, same as _ORDINAL_BACKSTOP_RE already lives at module level
# next to its own consumer.
#
# Broadened same date: "these"/"those" added — the "that"/"this" exclusion
# above does NOT apply to them (no relative-pronoun/determiner reading in
# English), yet they were previously unmatchable despite being named in this
# module's own re-aggregate warning text and in the "how many of these"
# worked example title. "ones" (plural) and "listed/shown/mentioned/
# displayed/returned above" added for the same reason — "the ones listed
# above" (a real live question) matched neither "the\s+above" (needs "the"
# immediately before "above") nor "one\s+above" (needs singular "one").
_FOLLOWUP_CUES = (
    r"it|its|them|they|she|he|her|his|their|here|these|those|"
    r"(?:that|this)\s+(?:inspection|site|facility|form|inspector|cycle|"
    r"person|project|client|action|schedule)|"
    r"the\s+same|the\s+above|ones?\s+above|"
    r"(?:listed|shown|mentioned|displayed|returned)\s+above|"
    r"that\s+one|this\s+one|\bones\b|"
    r"from\s+there|more|rest|remaining|"
    r"continue|next|previous|last\s+one|back\s+to|go\s+back"
)
_FOLLOWUP_RE = re.compile(rf"\b(?:{_FOLLOWUP_CUES})\b", re.IGNORECASE)

# SEPARATE from _FOLLOWUP_RE, same reason _COUNT_FOLLOWUP_RE below stays
# separate rather than folding in: _FOLLOWUP_RE deliberately excludes bare
# "this"/"that" (see note above) because they double as ordinary determiners
# ("this month's data") and would reintroduce the exact false-positive the
# ACTIVE CONTEXT gate was already burned by. This variant is ONLY consumed
# by the SELECTED-ROWS backstops in loop.py, which are already gated on
# session.pending_selected_positions being non-empty — in that narrower
# context a bare "this"/"that" NOT immediately followed by another word
# ("...list audits under this", not "...this month") safely reads as a
# pronoun standing in for the checked row(s), so it's fine to catch here
# even though it isn't safe for the general-purpose _FOLLOWUP_RE. Confirmed
# live 2026-08-10: "list all audits under this" (1 row checked) matched
# neither the "this <noun>" phrases nor "this one" above, so neither the
# SELECTED ROWS prose nor its backstop fired, and the model asked a
# needless clarifying question instead of resolving the checked row.
_SELECTION_FOLLOWUP_RE = re.compile(
    rf"\b(?:{_FOLLOWUP_CUES}|(?:this|that)(?!\s+\w))\b",
    re.IGNORECASE,
)

# Dedicated, deliberately CONJUNCTIVE count/aggregate-follow-up detector —
# added 2026-08-07 (debug_llm_20260807_153128) alongside _FOLLOWUP_RE above,
# but kept SEPARATE rather than folded into it: _FOLLOWUP_RE also gates
# ACTIVE CONTEXT scope injection (see build_context_block below), and
# broadening it further risks the exact stale-scope-leak regression that
# module already guards against. Requiring BOTH an aggregate cue AND a
# backward-reference cue (either order) keeps false positives near zero — a
# fresh "how many audits were conducted last month" has no backref cue, a
# bare "show me those" has no aggregate cue — which is what makes it safe
# to attach a forceful instruction to (see loop.py's final_answer backstop).
_AGG_CUE = (
    r"how\s+many|how\s+much|number\s+of|count\s+of|what\s+(?:number|count)|"
    r"total\s+(?:number|count)|percentage\s+of|what\s+(?:proportion|share|fraction)"
)
_BACKREF_CUE = (
    r"these|those|them|the\s+above|"
    r"(?:listed|shown|mentioned|displayed|returned)\s+above|ones?\s+above|"
    r"the\s+ones|that\s+list|previous(?:\s+(?:list|result|query))?"
)
_COUNT_FOLLOWUP_RE = re.compile(
    rf"\b(?:{_AGG_CUE})\b.{{0,80}}?\b(?:{_BACKREF_CUE})\b"
    rf"|\b(?:{_BACKREF_CUE})\b.{{0,80}}?\b(?:{_AGG_CUE})\b",
    re.IGNORECASE,
)


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


def _dup_stop_line(active_label: str, kind_label: str, dup_ids: list[str],
                    extra_hint: str = "") -> str:
    """Shared wording for the ACTIVE CONTEXT block's ambiguous-duplicate-name
    hard STOP — facility and entity (project/client/organisation) branches
    both hit this when _lookup_entity_id() finds MORE THAN ONE row sharing
    the active label's display name (the same class of bug as audit_plan.title
    duplicates). Extracted from two previously near-identical, independently
    hand-written copies to remove the drift risk of editing one and not the
    other — zero behavior/wording change from what each branch already had."""
    dup_ids_str = ", ".join(f"'{i}'" for i in dup_ids)
    return (
        f"⚠️ STOP — '{active_label}' matches {len(dup_ids)} DIFFERENT {kind_label}"
        f" records from the previous list (same name, different rows:"
        f" {kind_label} id IN ({dup_ids_str})).\n"
        f"  YOU MUST call final_answer NOW asking the user to disambiguate further"
        f"{extra_hint} — do NOT pick one of these ids arbitrarily and do NOT"
        f" call generate_sql."
    )


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
        # 1-based row positions the user explicitly checked in the UI on the
        # table from the LAST turn, set fresh by routes/chat.py from the
        # incoming request on EVERY call (defaults to [] when nothing was
        # selected) — NOT reset inside build_context_block() like
        # pending_resolved_disclosure above, because it must still be
        # readable by resolve_selected_rows' fn later in this same turn's
        # tool-calling loop. See agent/orchestrator/row_selection.py.
        self.pending_selected_positions: list[int] = []

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
        # Deliberately independent of is_followup/is_clearly_fresh above —
        # see the standalone re-aggregate block further down (module
        # docstring on _COUNT_FOLLOWUP_RE) for why this needs its own,
        # unconditional check rather than living inside the is_followup gate.
        is_count_followup = bool(_COUNT_FOLLOWUP_RE.search(current_question))

        # Full per-turn detail (SQL, columns, id lists, row_titles preview) is
        # only rendered for the most recent _RECENT_FULL_DETAIL_TURNS turns —
        # older turns get a single compact summary line instead. Same
        # principle as the row_titles render cap below (see
        # test_row_titles_context_cap.py): this only shrinks what's
        # TEXTUALLY RENDERED into the prompt, never the underlying
        # ConversationTurn data — ordinal/disambiguation/"go back to"
        # resolution all read structured fields (row_titles/row_ids,
        # active_subject_*, current_scope) directly off self.turns, never
        # this rendered text, so a reference to something from an "older"
        # (compacted) turn still resolves correctly.
        #
        # Confirmed live (token-usage audit, 2026-08-06): system_content
        # reached 83,456 chars (~20.9K tokens) by turn ~18 of one real
        # session — already past num_ctx=20000 before a single tool-call
        # iteration even ran — with this per-turn block (repeated for up to
        # MAX_TURNS=8 turns, each with an uncapped question and uncapped
        # columns list) the largest single contributor.
        _RECENT_FULL_DETAIL_TURNS = 3
        _QUESTION_RENDER_CAP = 300
        _COLUMNS_RENDER_CAP = 12

        lines = ["━━━ CONVERSATION HISTORY ━━━"]
        for i, t in enumerate(self.turns, 1):
            if (len(self.turns) - i) >= _RECENT_FULL_DETAIL_TURNS:
                _q_short = t.question[:150] + ("…" if len(t.question) > 150 else "")
                _a_short = t.answer[:120] + ("…" if len(t.answer) > 120 else "")
                _subj = (
                    f" (scoped to {t.active_subject_type}: "
                    f"{t.active_subject_label or t.active_subject_value})"
                    if t.active_subject_type and t.active_subject_value else ""
                )
                lines.append(f"Turn {i}: User: {_q_short!r} → {_a_short}{_subj}")
                continue

            lines.append(f"Turn {i}:")
            if t.domain:
                lines.append(f"  Domain: {t.domain}")
            _q = t.question[:_QUESTION_RENDER_CAP] + (
                "…" if len(t.question) > _QUESTION_RENDER_CAP else "")
            lines.append(f"  User: {_q}")
            short = t.answer[:200] + ("..." if len(t.answer) > 200 else "")
            lines.append(f"  Answer: {short}")
            if t.sql:
                lines.append(f"  SQL: {t.sql[:150]}")
            if t.total_count > 0:
                lines.append(f"  Total rows: {t.total_count}")
            if t.columns:
                _cols = t.columns[:_COLUMNS_RENDER_CAP]
                _cols_more = (
                    f" (+{len(t.columns) - _COLUMNS_RENDER_CAP} more)"
                    if len(t.columns) > _COLUMNS_RENDER_CAP else ""
                )
                lines.append(f"  Columns: {_cols}{_cols_more}")
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
                # Render-only: the resolve_ordinal_reference tool (agent/
                # orchestrator/ordinal.py) indexes the full stored
                # row_titles/row_ids on the ConversationTurn directly, never
                # this preview, so "the 47th one" still resolves correctly.
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

        # Row-selection priming — the user checked specific rows in the UI
        # (routes/chat.py sets this fresh from the request before the loop
        # starts) rather than referring to a position in words. Tells the
        # LLM to call the dedicated resolver tool instead of treating this
        # like an ordinal reference or a fresh name search. See
        # agent/orchestrator/row_selection.py / agent/tool_defs/row_selection.py.
        #
        # Deliberately rendered FIRST — before the ACTIVE CONTEXT gate and
        # its fallback chain further below — and with an explicit priority
        # statement. Confirmed live 2026-08-07: a user checked specific
        # boxes, then asked "internal stakeholders for these" — the model
        # ignored the selection and answered over the FULL previous result
        # set instead, because a later "CONTEXT — ... this is the COMPLETE
        # result set ... Filter by X IN (...)" block (same turn's audit_ids,
        # rendered ~15 lines further down) gave it a ready-made filter that
        # needed no extra tool call, and nothing told it that block loses to
        # an actual pending selection. This block now renders first and says
        # so explicitly — but see loop.py's matching deterministic backstop
        # (mirroring resolve_ordinal_reference's) for the layer that
        # actually GUARANTEES this, independent of whether the model reads
        # prose ordering correctly; this text alone is not sufficient on its
        # own (the exact lesson every other backstop in this codebase — the
        # ordinal/disambiguation-reply/count-followup ones — was already
        # built to address).
        if self.pending_selected_positions:
            _n_sel = len(self.pending_selected_positions)
            lines.append(
                f"SELECTED ROWS — the user picked {_n_sel} specific row(s) via checkboxes "
                f"in the UI before asking this question. If this question refers to "
                f"'these'/'them'/'the selected ones'/'those' — OR a bare 'this'/'that' "
                f"standing in for them (e.g. 'list all audits under this') — rather than "
                f"naming something new, call resolve_selected_rows FIRST — do NOT use "
                f"resolve_ordinal_reference or re-search by name for this. Do NOT ask the "
                f"user to clarify what 'this'/'that' means before trying this tool — the "
                f"checked row(s) are almost always what it refers to.\n"
                f"  THIS TAKES PRIORITY over any 'Filter by X IN (...)' or 'the conversation "
                f"is about Y' instruction that appears LATER in this context (e.g. under "
                f"ACTIVE CONTEXT or CONTEXT below) — those describe the general conversation "
                f"topic, not what was specifically checked. If the question could mean either "
                f"the selection or the general context, use the selection."
            )
            lines.append("")

        # ── Pending clarifying question this turn's answer may be resolving ──
        # Evaluated for tool conversion in the same effort that produced
        # resolve_ordinal_reference/resolve_disambiguation_reply — rejected,
        # even more clearly than the ACTIVE CONTEXT area above (session.py
        # ~line 235's comment): there is no deterministic resolution
        # algorithm here AT ALL. Ordinal has position math; disambiguation-
        # reply has an exact-name/candidate-list lookup; this block has
        # neither — it's a bare reminder of the LLM's own earlier question,
        # unconditional on the current question's text, with the actual
        # "does this reply answer that question" judgment left entirely to
        # the LLM's own reasoning. Connecting "FM managers" back to "top 5
        # capex converters" isn't reducible to an int/string a tool could
        # return — nothing here to convert. See the tool-conversion plan
        # history for the fuller reasoning; this comment exists so the
        # question isn't silently reopened from scratch.
        #
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
        # Evaluated for tool conversion (agent/tool_defs) in the same effort
        # that produced resolve_ordinal_reference/resolve_disambiguation_reply
        # — deliberately kept as prose instead. current_scope is a PERSISTENT
        # pointer that legitimately stays active for many consecutive turns
        # (multi-turn drilldowns are the normal case here, not an edge case,
        # confirmed live surviving 3+ turns unchanged); converting this to a
        # tool would mean calling it on nearly every follow-up turn in a long
        # conversation — a real per-turn round-trip added to the COMMON case,
        # not the rare one ordinal/disambiguation-reply targeted. There is
        # also nothing here for the LLM to *supply* that this code doesn't
        # already know (contrast resolve_ordinal_reference's `position` or
        # resolve_disambiguation_reply's `picked_name`, which only the LLM
        # can extract from freeform text) — the resolved column/value below
        # is 100% deterministic from session state already in hand. See the
        # tool-conversion plan history for the full reasoning; this comment
        # exists so that hypothesis isn't silently reopened from scratch.
        #
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
            the equivalent people-search flow (see the
            resolve_disambiguation_reply tool, agent/orchestrator/
            disambiguation_reply.py), which already correctly hard-STOPs on
            an ambiguous duplicate name instead of guessing. Collect ALL
            matches so the facility/entity branches can
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
                # sibling column inject_row_identity() attaches) — only
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
                    lines.append(_dup_stop_line(
                        active_label, "facility", _fid_matches,
                        extra_hint=" (e.g. by project, client, or code)",
                    ))
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
                    lines.append(_dup_stop_line(active_label, _tbl, _eid_matches))
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

        # ── Standalone count/aggregate-follow-up re-aggregate warning ──────
        # Added 2026-08-07 (debug_llm_20260807_153128, qid=3dbcdaf3): "how
        # many of the ones listed above are completed" was answered via
        # final_answer with ZERO tool calls — the model echoed the bare
        # "Total rows: N" line below as if it were a real count, and
        # fabricated its own (wrong) recollection of what the previous SQL
        # filtered on.
        #
        # Deliberately NOT nested inside the `elif is_followup and not
        # is_clearly_fresh:` chain above — that block's own re-aggregate
        # guidance (the "NOT the full set" messages) needs SIX conditions to
        # fire (is_followup AND not is_clearly_fresh AND domain-compat AND
        # non-empty inspection_ids/audit_ids AND len>1 AND total_count >
        # len(ids)). A previous turn that listed by title only (no id column
        # selected) captures no audit_ids/inspection_ids at all and gets no
        # warning today even with a perfect regex. This block is keyed only
        # on is_count_followup + last.sql + last.total_count>0 — no
        # dependency on which id lists got captured.
        if is_count_followup and last.sql and last.total_count > 0:
            _sample_n = max(len(last.row_titles), len(last.audit_ids), len(last.inspection_ids))
            _prev_sql_snippet = " ".join(last.sql.split())[:500]
            lines.append(
                f"CONTEXT — The previous turn returned {last.total_count} total row(s), but only "
                f"a CAPPED SAMPLE (~{_sample_n} row(s)) was rendered into this conversation "
                f"history above — not the full result. You have NOT seen the individual rows.\n"
                f"  This question asks to count/aggregate over that previous result "
                f"('how many of these/those/them/the ones listed above are X'). You CANNOT "
                f"answer this from the 'Total rows' number, from the sampled ids/titles above, "
                f"or from your own memory of what the previous SQL filtered on — that memory is "
                f"unreliable and has been wrong before.\n"
                f"  INSTEAD: call generate_sql, reusing the SAME FROM/JOIN/WHERE filters as the "
                f"previous query below, add the new condition, and return a fresh COUNT(*). Do "
                f"NOT call final_answer without running a query first.\n"
                f"  Previous SQL: {_prev_sql_snippet}"
            )

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

        # Disambiguation context — display-only pointer to the pending
        # pick. Actual resolution happens via the resolve_disambiguation_reply
        # tool (agent/orchestrator/disambiguation_reply.py) — retired the old
        # regex matching here (filler-word stripping, exact/substring tiers,
        # a separate ordinal-pick regex) since an LLM-supplied picked_name/
        # picked_position needs none of that; there's no leftover sentence
        # text to fuzzy-match once the LLM has already extracted the pick
        # itself. The ambiguous-duplicate-name STOP and the >12-vs-2..12
        # wording distinction both still exist, just inside the tool/its
        # format_result instead of resolved here by regex.
        if not is_clearly_fresh and last.disambiguation_name:
            _dn = last.disambiguation_name
            _dc = last.total_count
            _names_preview = [c.get("name", "") for c in last.disambiguation_candidates[:12]]
            _too_many_note = (
                f" ({_dc} was too many to list in full — if picking by name doesn't "
                f"resolve, ask for a fuller name or email address.)"
                if _dc > 12 else ""
            )
            lines.append(
                f"PENDING DISAMBIGUATION — the previous turn found {_dc} people matching "
                f"'{_dn}' and asked which one was meant: {_names_preview}.{_too_many_note}\n"
                f"  If THIS message is the user picking one (by name or position), call "
                f"resolve_disambiguation_reply with picked_name or picked_position — do NOT "
                f"guess the id yourself via ILIKE/substring matching.\n"
                f"  If the user is instead providing a MORE SPECIFIC name/email (not simply "
                f"repeating '{_dn}'), run a targeted full-name search via generate_sql instead: "
                f"WHERE (u.first_name || ' ' || u.last_name) ILIKE '%<full name>%'.\n"
                f"  If the message just repeats '{_dn}' with no new information, call "
                f"final_answer and re-list these options rather than re-running the same "
                f"broad search."
            )

        # Entity disambiguation context (facility/project/client/organisation) —
        # the entity-equivalent of the people block above, same tool resolves
        # both (it auto-detects which candidate list is populated).
        if not is_clearly_fresh and last.entity_disambiguation_candidates:
            _ent_tbl = last.entity_disambiguation_table or "entity"
            _ent_names_preview = [
                c.get("name", "") for c in last.entity_disambiguation_candidates[:12]
            ]
            lines.append(
                f"PENDING DISAMBIGUATION — the previous turn found multiple {_ent_tbl} "
                f"matches and asked which one was meant: {_ent_names_preview}.\n"
                f"  If THIS message picks one (by name or position), call "
                f"resolve_disambiguation_reply with picked_name or picked_position — do NOT "
                f"guess the id yourself via ILIKE on the name.\n"
                f"  If the message is a fresh, more specific search instead, run generate_sql "
                f"normally."
            )

        # Gated on the current question actually looking like a "modify the
        # last query" follow-up (token-usage audit, 2026-08-06) — this block
        # was previously unconditional on every turn from turn 2 onward,
        # regardless of whether the question was remotely pagination/
        # refinement/augmentation-shaped. Deliberately broad/inclusive
        # (matches on a short keyword, not a strict phrase) — a false
        # positive here just means the instruction shows up when not
        # strictly needed (cheap); a false negative means a real "show
        # more"/"also show X" follow-up silently regenerates the query from
        # scratch instead of extending it, the exact live-caught bug the
        # AUGMENTATION block documents below.
        if re.search(
            r"(?i)\b(show\s+more|show\s+the\s+rest|show\s+all|continue|remaining|next\b"
            r"|filter\s+by|only\s+the\b|sort\s+by"
            r"|along\s+with|also\s+show|include\b|add\s+the\b)",
            current_question,
        ):
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
