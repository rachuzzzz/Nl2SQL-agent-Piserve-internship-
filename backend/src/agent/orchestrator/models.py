"""
Plain dataclasses shared across the orchestrator package. Extracted
verbatim from the original agent/orchestrator.py.
"""

from dataclasses import dataclass, field
from typing import Optional




@dataclass
class AgentStep:
    iteration: int
    thought: str
    tool: str
    args: dict
    result: dict
    duration_ms: int




@dataclass
class QueryStats:
    total_llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_latency_ms: int = 0
    total_sql_exec_ms: int = 0
    total_wall_ms: int = 0




@dataclass
class ConversationTurn:
    question: str
    answer: str
    tool_used: str
    sql: str = ""
    total_count: int = 0
    # The OFFSET to use for the NEXT execute_sql fetch against this turn's sql
    # (i.e. how many rows of it have already been shown) — lets a "show more"
    # follow-up (see loop.py's pagination Path A) fetch the next window
    # instead of silently re-fetching the same first N rows. Set to the row
    # count of whatever was just fetched at every ConversationTurn
    # construction site; Path A itself advances it cumulatively.
    page_offset: int = 0
    columns: list = field(default_factory=list)
    # Extracted key entities from result rows — used for follow-up context
    inspection_ids: list = field(default_factory=list)   # e.g. ["2026/04/ST085/INS003"]
    audit_ids: list = field(default_factory=list)         # e.g. ["2026/07/ST033/AUD001"] — audit domain's equivalent of inspection_ids
    facility_names: list = field(default_factory=list)   # e.g. ["Golf Gardens"]
    inspector_names: list = field(default_factory=list)  # e.g. ["neenu extinsp1"]
    cycle_ids: list = field(default_factory=list)        # UUIDs from cycle_id column
    user_ids: list = field(default_factory=list)         # resolved user UUIDs after disambiguation
    facility_ids: list = field(default_factory=list)     # resolved facility UUIDs, captured opportunistically from facility_id/fac_id columns
    plan_ids: list = field(default_factory=list)         # resolved audit_plan.id UUID, via a title->id lookup (see synthesis.py) — audit_plan.title has NO unique constraint, so a follow-up should prefer this over filtering by title/ILIKE whenever available
    plan_title_ambiguous: bool = False                    # True when the title->id lookup above found MORE THAN ONE active plan sharing this exact title — confirmed live 2026-07-30 (e.g. 'Delegation plan -1' has 2 active rows); build_context_block should warn rather than silently scope to title
    disambiguation_name: str = ""   # name that was disambiguated (e.g. "george"), total_count stored in total_count
    disambiguation_candidates: list = field(default_factory=list)  # [{"name": ..., "user_id": ...}] from disambiguation rows
    disambiguation_origin_question: str = ""  # the ORIGINAL question that started this disambiguation chain (e.g. "which schedules is assigned to neenu") — carried forward across narrowing turns so the agent doesn't lose the actual goal while resolving a name
    entity_disambiguation_candidates: list = field(default_factory=list)  # [{"name": ..., "count": ...}] from a STEP 1 entity (facility/project/client/organisation) count-check — "count" is domain-neutral (inspection_count/schedule_count/audit_count all normalize to this key, see sql/result_shape.py's is_entity_disambiguation_result) — lets a later turn that names one entity from this list resolve active_subject_type even though that turn ran no new SQL of its own
    entity_disambiguation_table: str = ""  # e.g. "facility", "project" — which table the candidates above came from
    entity_disambiguation_alias: str = ""  # e.g. "fac", "proj" — the SQL alias used, for building a correct {alias}.name ILIKE filter
    single_row_values: dict = field(default_factory=dict) # col→val when result is 1 row
    # Which domain answered this turn — "inspection" | "audit" | "" (unset, e.g.
    # a parse-error step). Read by ConversationSession.last_domain for Tier-2
    # router stickiness (see agent/domain_router.py) — a follow-up with no
    # domain keyword stays in whatever domain the last resolved turn used.
    domain: str = ""
    # Active focal entity — what the user is currently drilling into
    # type: "inspection"|"audit"|"facility"|"inspector"|"cycle"|"plan"|"corrective_action"|"entity"|None
    active_subject_type: str = ""
    active_subject_value: str = ""   # exact value (UUID or name)
    active_subject_label: str = ""   # human-readable label for context message
    active_subject_table: str = ""   # only for active_subject_type=="entity" — e.g. "project"
    active_subject_alias: str = ""   # only for active_subject_type=="entity" — e.g. "proj"
    # Which domain this active_subject_type/value actually BELONGS to — NOT
    # necessarily the same as this turn's own `domain` above. When scope is
    # INHERITED from an earlier turn (see synthesis.py's inheritance branch),
    # the turn answering now may be in a different domain than the one that
    # originally resolved the subject; this field tracks the origin so
    # domain-exclusive types ("inspection"/"audit"/"cycle") never get reused
    # in the wrong domain's prompt regardless of how many inheritance hops
    # they've passed through. "facility"/"inspector"/"entity" ignore this
    # (safe cross-domain — same underlying tables in both schemas).
    active_subject_domain: str = ""
    # Real row id paired with active_subject_value, when one could be
    # resolved for this turn's query shape — via the __id sibling column
    # inject_row_identity() (src/sql/validation/identity.py) attaches
    # alongside a display column, or directly for types whose value already
    # IS a UUID (cycle) or that have their own established id-resolution
    # path (inspector, via user_ids/disambiguation_candidates). Empty when no
    # id could be resolved (e.g. the query had a GROUP BY, which injection
    # skips) — ACTIVE CONTEXT/ordinal resolution in session.py falls back to
    # title/ILIKE-based resolution in that case, same as before this field
    # existed. Lets a follow-up filter WHERE x.id = '<uuid>' directly instead
    # of re-deriving a row from a display value that may not be unique
    # (confirmed live: audit_plan.title has no uniqueness constraint).
    active_subject_id: str = ""
    # Set when THIS turn's own answer was itself a clarifying question (e.g.
    # "Could you clarify who and what 'top 5 capex converters' means? If you
    # mean FM Managers: ... Just say which group and metric") — holds the
    # ORIGINAL request that triggered it ("top 5 capex converters"). Confirmed
    # live 2026-07-29: without this, a short reply that answers the clarifying
    # question ("FM manafers") gets treated as a brand-new, unrelated request
    # instead of a resolution of the pending one — it fell through to the
    # generic ambiguous-person-name search (43 unrelated user matches) and the
    # original "top 5 capex converters" request was silently dropped, never
    # answered. Distinct from disambiguation_origin_question, which only
    # covers the SQL-driven "N candidates found, pick one" flow — this covers
    # a clarifying question the reasoning model invented on its own.
    clarification_origin_question: str = ""
    # Ordered display values (e.g. audit_plan.title) for a MULTI-row result,
    # in the exact order the SQL returned them — lets a same-turn-or-later
    # ordinal follow-up ("the first one", "the second one") resolve against
    # THIS list. Deliberately separate from active_subject_* (which only ever
    # tracks a single resolved entity and — via ConversationSession.current_scope
    # — can persist across many turns): a multi-row list turn often sets no
    # active_subject_type at all, which previously let an ordinal follow-up
    # silently fall through to whatever unrelated single-entity scope was
    # last set several turns back. Capped at 5, matching the other per-turn
    # capture lists (inspection_ids, facility_names, etc).
    row_titles: list = field(default_factory=list)
    # Paired 1:1 with row_titles (same index = same row) — the real row id
    # for that same row, via the __id sibling column inject_row_identity()
    # attaches. Empty string at an index means no id could be resolved for
    # that particular row (not necessarily the whole turn — each row is
    # captured independently in the same loop that builds row_titles).
    row_ids: list = field(default_factory=list)
    # Paired 1:1 with row_ids — the COLUMN NAME the id actually filters on
    # (e.g. "id" for the usual primary key, "submission_id" for an
    # ai_audit_answers/ai_answers observation row). Lets session.py's
    # ORDINAL RESOLVED hint phrase the right WHERE clause instead of always
    # assuming a primary key — see synthesis.py's row_titles/row_ids capture.
    row_id_columns: list = field(default_factory=list)
    # One value per TURN (not per row) — the entity type of the list this
    # turn displayed: "audit", "schedule", "inspection", "corrective_action",
    # "cycle", "plan", "facility", "person" (auditor/inspector — unified,
    # both draw from `users`), "observation", or "" when ambiguous/unmatched.
    # Lets session.py's ordinal resolver require the noun in an ordinal
    # phrase ("the 6th AUDITOR") to match the type of list being resolved
    # against, instead of blindly grabbing the most recent turn with ANY
    # row_titles regardless of what kind of list it was — confirmed live
    # (qid=081c7a81): "list all audit plans under the 6th auditor from the
    # list" resolved "the 6th" against a CYCLES list (the only row_titles
    # this session had), used one cycle's UUID as if it were an auditor's
    # id, and produced a well-formed but empty result — wrong, no error.
    # Set once per turn in synthesis.py's row_titles capture loop, from
    # whichever priority branch fires first for that turn.
    row_entity_type: str = ""




@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    success: bool = True
    stats: QueryStats = field(default_factory=QueryStats)
