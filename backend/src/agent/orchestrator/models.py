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
    # type: "inspection"|"audit"|"facility"|"inspector"|"cycle"|"corrective_action"|"entity"|None
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




@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    success: bool = True
    stats: QueryStats = field(default_factory=QueryStats)
