"""
Agent Orchestrator — the main reasoning loop.
"""

import json
import logging
import re
import time
from typing import Optional

_log_query    = logging.getLogger("inspectai.query")
_log_llm      = logging.getLogger("inspectai.llm")
_log_sql      = logging.getLogger("inspectai.sql")
_log_security = logging.getLogger("inspectai.security")

from sqlalchemy import create_engine, text as sql_text

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.llm.factory import get_llm_client
from src.llm.prefix_cache import PrefixCacheManager
from src.llm.anthropic import build_cached_system_message
from src.retrieval import SemanticQuestionIndex, SeedExampleIndex, text_similarity
from src.sql.validation import SQLValidator
from src.agent.domains import get_domain_config
from src.agent.tools import ToolRegistry
from src.agent.tool_dispatcher import parse_tool_call, dispatch
from src.agent.tool_defs import get_declared, declared_tools
from src.security.rbac import build_scope_filter
from src.sql.result_shape import is_disambiguation_result, is_entity_disambiguation_result
from src.knowledge.fuzzy_vocab import fuzzy_augment

from src.agent.orchestrator.models import AgentStep, QueryStats, ConversationTurn, AgentResult
from src.agent.orchestrator.debug import _make_debug_logger, _dbg, _build_constraint_notes, _CONSTRAINT_NOTE_MAP, _cap_text
from src.agent.orchestrator.synthesis import SynthesisMixin
from src.agent.orchestrator.disambiguation import DisambiguationMixin
from src.agent.orchestrator.turn_classifier import TurnClassifier


_PG_ERROR_CLASS_RE = re.compile(r"psycopg2\.errors\.(\w+)")


def _failure_signature(error_text: str) -> frozenset:
    """
    Normalize a raw execute_sql error string into a stable signature for
    same-failure-repeated detection (see _sql_retry_signal's comment in
    query()). Postgres errors from SQLAlchemy/psycopg2 are prefixed like
    "(psycopg2.errors.UndefinedColumn) column ip.inspector_id does not
    exist\nLINE 2: ..." — the exception class name (UndefinedColumn,
    UndefinedTable, DatatypeMismatch, ...) is a far more stable signature
    than the full message, which often varies in the specific column/table
    name even across two attempts hitting the conceptually same wall.
    Falls back to a truncated prefix of the raw text when that shape isn't
    present (e.g. a plain Python exception, not a psycopg2 DB error).
    """
    m = _PG_ERROR_CLASS_RE.search(error_text or "")
    return frozenset([m.group(1) if m else (error_text or "")[:80]])


# Extracts the first single-quoted identifier out of a ValidationIssue.message
# (e.g. "Table alias 'proj' is used..." -> "proj"). See _validation_issue_signature.
_QUOTED_IDENT_RE = re.compile(r"'([^']{1,100})'")


def _validation_issue_signature(issue) -> str:
    """
    Refines a validator HARD_FAIL's bare .code into a signature that also
    captures WHICH identifier it complained about, for the _sql_retry_signal
    "stuck on the same wall" comparison below (NOT for anything else — see
    the note below).

    Confirmed live 2026-08-07 (qid=e63f03af): a capex/opex question failed
    UNDEFINED_ALIAS on attempt 2 (alias 'ir' missing) and again on attempt 3
    (alias 'proj' missing) — two unrelated bugs the coder model introduced
    fresh each time, not the same wall twice. Comparing by bare .code alone
    treated both as identical, escalated _sql_retry_signal as if the model
    were stuck, and that escalation (see retrieval/examples.py's skip_top_n)
    dropped the #1-ranked worked example — which was actually the correct
    CTE pattern for the question — from the next attempt's prompt. Folding
    the quoted identifier into the signature means two different aliases no
    longer collide; a genuinely repeated identical alias failure still does.

    Deliberately does NOT touch issue.code or .message — only builds a
    separate string for this comparison. ValidatorStats.record() (aggregates
    by bare code), retry_message() (the LLM-facing text), and
    orchestrator/debug.py's _build_constraint_notes() (substring-matches
    .code against _CONSTRAINT_NOTE_MAP keys) all still see the unmodified
    code/message and are unaffected.
    """
    m = _QUOTED_IDENT_RE.search(issue.message or "")
    return f"{issue.code}:{m.group(1)}" if m else issue.code


_TOOL_NAME_RE = re.compile(r'"tool"\s*:\s*"(\w+)"')


def _summarize_dropped_messages(dropped: list[ChatMessage]) -> str:
    """
    Mechanical, purely-textual summary of round-trips being proactively
    trimmed out of `messages` (see the proactive context trim in query()) —
    a breadcrumb so the model retains SOME memory of what it already tried
    instead of silently losing it and potentially repeating an already-
    completed tool call. Deliberately conservative: extracts just the tool
    name (regex over the raw ASSISTANT tool-call JSON) and the first ~150
    chars of its paired USER-role result, rather than a per-tool-shape
    summarizer — a real one can be built later from actual debug_llm_*.log
    traces if this proves too thin in practice.
    """
    parts: list[str] = []
    for m in dropped:
        if m.role == MessageRole.ASSISTANT:
            tm = _TOOL_NAME_RE.search(m.content or "")
            if tm:
                parts.append(tm.group(1))
        elif m.role == MessageRole.USER and parts:
            snippet = (m.content or "")[:150].replace("\n", " ")
            parts[-1] = f"{parts[-1]} → {snippet}"
    return "; ".join(parts)


class AgentOrchestrator(SynthesisMixin, DisambiguationMixin):
    def __init__(self, db_connection_string, db_schema="public",
                 ollama_url="http://localhost:11434", sql_model="sqlcoder:7b",
                 chat_model="qwen2.5:14b-instruct-q4_K_M",
                 embedding_model="BAAI/bge-small-en-v1.5",
                 domain: str = "inspection",
                 db_engine=None, embed_model=None, reasoning_llm=None,
                 reasoning_llm_provider: str = "ollama", sql_llm_provider: str = "ollama",
                 anthropic_api_key: str = "", vllm_url: str = "",
                 llamacpp_reasoning_url: str = "", llamacpp_sql_url: str = ""):
        print(f"Initialising AgentOrchestrator (domain={domain})...")
        self.domain = domain
        self._domain_cfg = get_domain_config(domain)
        self._scope_primary_table = self._domain_cfg.scope_primary_table
        self._scope_primary_alias = self._domain_cfg.scope_primary_alias

        # db_engine/embed_model/reasoning_llm may be shared across domain
        # instances (see api/app.py's lifespan — one DB connection pool, one
        # embedding model, one reasoning LLM client for both "inspection" and
        # "audit" orchestrators, per the domain-router plan's §3) rather than
        # each building its own.
        if db_engine is not None:
            self.db_engine = db_engine
            print("  ✓ Database connection reused from another domain instance")
        else:
            print("  Connecting to database...")
            self.db_engine = create_engine(db_connection_string)
            with self.db_engine.connect() as conn:
                conn.execute(sql_text("SELECT 1")).fetchone()
            print("  ✓ Database connected")

        self.db_schema = self._domain_cfg.introspect_fn(self.db_engine)

        # Reasoning LLM gets the curated/slim schema block (for_reasoning_prompt),
        # not the full per-column dump (for_system_prompt) — the full dump pushed
        # this prompt to 19953/20000 num_ctx tokens, causing truncated tool-call
        # JSON (see debug_llm_*.log "TRUNCATED — unclosed brace(s)" warnings).
        #
        # business_rules/table_relationships here use the CORE-only subset
        # (DatabaseSchema._business_rules_core/_table_relationships_core), not
        # the full BUSINESS_RULES/TABLE_RELATIONSHIPS — 2026-08 token audit:
        # the full audit-domain content alone measured ~34K chars (~8.5K
        # tokens), unconditionally, on every question regardless of topic.
        # The narrow-topic remainder (observation/MA paths, risk/impact,
        # approval workflow, user roles, communication groups, ...) is
        # surfaced per-question instead via build_dynamic_prompt_additions'
        # CONDITIONAL_BLOCKS below — same question_pattern-gate mechanism
        # already used for MA_WORKFLOW_BLOCK. See knowledge/schema_introspector.py's
        # "Topic-gated core/topical split" section for what stayed core vs why.
        _rules_block = "BUSINESS_RULES:\n" + "\n".join(
            f"  - {r}" for r in self.db_schema._business_rules_core
        )
        self._system_prompt = self._domain_cfg.system_prompt.replace(
            self._domain_cfg.schema_placeholder,
            self.db_schema.for_reasoning_prompt() + "\n\n" + _rules_block
            + "\n\n" + self.db_schema._table_relationships_core
        )
        # {inspection_schema_sql}/{audit_schema_sql} removed — injecting full DDL
        # exceeded num_ctx:16384, causing Ollama to truncate the prompt from the
        # front. Per-call SQL-model context is built in sql/generation.py's
        # generate_sql() (via _select_schema_block): DatabaseSchema.for_sql_prompt_slim()
        # (a fixed curated subset) by default, or DatabaseSchema.for_schema_hint()
        # for just the tables named in the reasoning LLM's optional
        # relevant_tables tool-call argument — real per-question targeting,
        # not deterministic table detection.
        self._sql_prompt = (
            self._domain_cfg.sql_generation_prompt
            .replace("{inspection_schema_sql}", "")
            .replace("{audit_schema_sql}", "")
        )
        # The template puts {dynamic_examples} before {schema_hint} (see
        # sql/examples/templates/sql_generation_prompt.txt — deliberately
        # reordered so the guaranteed-static content comes first; it used to
        # be {schema_hint} first, which meant even the STATIC CORE_CONTEXT
        # block landed at a different byte offset on every call, since it's
        # substituted into {dynamic_examples} and {schema_hint} is fully
        # variable per question — see sql/generation.py's dynamic_examples =
        # core_context + build_core_context_additions(question) + retrieved).
        # {question} still comes last, after {schema_hint}.
        #
        # core_context (CORE_CONTEXT / CORE_CONTEXT_AUDIT, per domain — see
        # domains.py) is itself fully static, built once at import, but it's
        # SUBSTITUTED content, not template text — splitting the raw template
        # string on {dynamic_examples} only gets the literal skeleton text
        # before that placeholder, not what actually replaces it. Appending
        # core_context here is what makes the warmed prefix match the real
        # bytes every call sends: skeleton + core_context is byte-identical
        # across every question in this domain (retrieved examples and any
        # per-question conditional block are appended after core_context
        # inside {dynamic_examples}'s own substituted value, so they don't
        # participate in this guaranteed-stable prefix).
        self._sql_static_prefix = (
            self._sql_prompt.split("{dynamic_examples}")[0] + self._domain_cfg.core_context
        )

        if embed_model is not None:
            self.embed_model = embed_model
            print("  ✓ Embedding model reused from another domain instance")
        else:
            print(f"  Loading embedding model: {embedding_model}...")
            embed_model = HuggingFaceEmbedding(model_name=embedding_model)
            Settings.embed_model = embed_model
            self.embed_model = embed_model  # reused to rank/budget-cap Tier-2 prompt blocks
            print("  ✓ Embedding model loaded")

        self.semantic_index = SemanticQuestionIndex(
            db_engine=self.db_engine, embed_model=self.embed_model)
        self.validator = SQLValidator(domain=self.domain)

        # Build seed example index — embeds NL questions for retrieval-based injection.
        # query_domain_tags/query_domain_fallback: None for inspection (keeps
        # SeedExampleIndex's built-in inspection-tuned keyword map); audit
        # passes explicit neutral overrides via DomainConfig — see that
        # dataclass's field comment and SeedExampleIndex.__init__'s docstring.
        self.seed_index = SeedExampleIndex(
            embed_model=self.embed_model,
            examples=self._domain_cfg.seed_examples,
            query_domain_tags=self._domain_cfg.seed_query_domain_tags,
            query_domain_fallback=self._domain_cfg.seed_query_domain_fallback,
        )

        self.chat_model_name = chat_model
        self.sql_model_name  = sql_model

        # reasoning_llm_provider/sql_llm_provider are resolved independently —
        # e.g. reasoning via vLLM and sql via Ollama at the same time is a
        # supported combination, not a special case. Every call site
        # downstream (generate_sql, the reasoning loop, etc.) only ever calls
        # .complete(prompt) per the LLMClient protocol (llm/base.py), so
        # neither knows or cares which provider built it.
        _base_urls = {"ollama": ollama_url, "vllm": vllm_url}
        # llamacpp gets a per-role base_url instead of one shared URL: unlike
        # Ollama (one server, many models) or this app's vLLM usage,
        # reasoning and sql are two separate llama-server processes/ports
        # (one model per process — see llm/factory.py's
        # get_reasoning_llm_llamacpp docstring).
        _llamacpp_role_urls = {"reasoning": llamacpp_reasoning_url, "sql": llamacpp_sql_url}

        def _build(role: str, provider: str, model: str):
            base_url = _llamacpp_role_urls[role] if provider == "llamacpp" else _base_urls.get(provider, "")
            return get_llm_client(provider, role, model, base_url=base_url, api_key=anthropic_api_key)

        if reasoning_llm is not None:
            self.reasoning_llm = reasoning_llm
            print("  ✓ Reasoning LLM reused from another domain instance")
        else:
            print(f"  Configuring reasoning LLM ({reasoning_llm_provider}): {chat_model}...")
            self.reasoning_llm = _build("reasoning", reasoning_llm_provider, chat_model)
            print(f"  ✓ Reasoning LLM ({chat_model})")

        print(f"  Configuring SQL LLM ({sql_llm_provider}): {sql_model}...")
        self.sql_llm = _build("sql", sql_llm_provider, sql_model)
        print(f"  ✓ SQL LLM ({sql_model})")

        # Warm this domain's static system-prompt prefix into Ollama's own
        # slot cache before real user traffic hits it — see llm/prefix_cache.py.
        # Best-effort (never blocks startup on a network hiccup); harmless
        # extra call for non-Ollama reasoning providers, which have no slot
        # cache to warm but answer the call fine regardless.
        self._prefix_cache = PrefixCacheManager(self.reasoning_llm, domain=self.domain, role="reasoning")
        self._prefix_cache.warmup(self._system_prompt)

        # Same warmup for the SQL role's static skeleton — without this the
        # first SQL-gen call of every session still pays near-cold prefill
        # (confirmed live: 954ms vs ~390ms on warm calls in the same
        # session, see debug_llm_20260814_115608.log) because generate_sql()
        # builds its own PrefixCacheManager per-call with no warmup of its
        # own (sql/generation.py).
        self._sql_prefix_cache = PrefixCacheManager(self.sql_llm, domain=self.domain, role="sql")
        self._sql_prefix_cache.warmup(self._sql_static_prefix)

        self._debug = _make_debug_logger()

        # Classifies each turn (fresh question vs. some kind of follow-up)
        # and, for follow-ups that add/change something, extracts what
        # changed — see turn_classifier.py's module docstring for the regex
        # bank this replaces. Reuses self.reasoning_llm, same as DomainRouter
        # does for its own Tier-3 (one extra short completion, no model swap).
        self._turn_classifier = TurnClassifier(self.reasoning_llm, self._debug, domain=self.domain)

        # Domain-scoped pieces threaded through to generate_sql() (see
        # sql/generation.py) via ToolRegistry — the rules engine is also
        # cached on self for query()'s INJECT-hint step.
        self._rules_engine = self._domain_cfg.rules_engine_getter()
        self._builder_module = self._domain_cfg.builder_module

        self.registry = ToolRegistry(
            db_engine=self.db_engine,
            semantic_index=self.semantic_index,
            seed_index=self.seed_index,
            db_schema=self.db_schema,
            validator=self.validator,
            sql_llm=self.sql_llm,
            sql_prompt=self._sql_prompt,
            debug_logger=self._debug,
            core_context=self._domain_cfg.core_context,
            build_core_context_fn=self._domain_cfg.build_core_context_additions_fn,
            join_graph_module=self._domain_cfg.join_graph_module,
            rules_engine=self._rules_engine,
            domain=self.domain,
        )

        # Instance attribute shadows the class-level _TERMINAL_TOOLS set,
        # unioned with any declared tool (agent/tool_defs) that opts into
        # terminal=True — avoids a manual edit to the class-level set for
        # every future declared tool.
        self._TERMINAL_TOOLS = self._TERMINAL_TOOLS | {
            t.name for t in declared_tools() if t.terminal
        }

        # ── Join graph startup validation ─────────────────────────────────
        # Validates every FK edge in the domain's join graph against the live
        # DB schema. Catches schema drift (renamed columns, dropped tables)
        # immediately at startup rather than silently failing at query time.
        # Errors are logged as warnings — they don't prevent startup.
        try:
            jg_errors = self._domain_cfg.join_graph_module.validate_join_graph(self.db_engine)
            if jg_errors:
                print(f"  ⚠ JOIN_GRAPH validation ({len(jg_errors)} issues):")
                for e in jg_errors:
                    print(f"    - {e}")
            else:
                print(f"  ✓ JOIN_GRAPH validated ({len(jg_errors)} errors)")
        except Exception as _jg_exc:
            print(f"  ⚠ JOIN_GRAPH validation skipped: {_jg_exc}")


    # ------------------------------------------------------------------ #
    # Terminal tools — synthesize and return immediately on success
    # ------------------------------------------------------------------ #
    _TERMINAL_TOOLS = {
        "list_forms",
        "get_answers",
        "get_answer_stats",
        "execute_sql",
    }


    def query(self, question, max_iterations=8, on_step=None, session=None,
              visibility_scope: dict | None = None, query_id: str = "",
              router_locked_domain: bool = False):
        # router_locked_domain: True when agent/domain_router.py resolved THIS
        # message's domain via a Tier-1 keyword hit (confidence=1.0) — an
        # unambiguous domain word actually present in the user's own wording,
        # not session stickiness or an LLM guess. When True, the in-prompt
        # DOMAIN-MISMATCH CHECK's redirect_domain tool is suppressed below:
        # confirmed live (debug_llm_20260813_123024, 20260814_102645) that the
        # reasoning LLM calls redirect_domain anyway — contradicting its own
        # system prompt's "an unambiguous inspection word... is NOT a
        # mismatch" rule — apparently swayed by stale conversation history
        # from a prior domain still sitting in context. The router's
        # deterministic regex match on the CURRENT message outranks that.
        qid = query_id or ""
        _pfx = f"qid={qid} " if qid else ""
        query_start = time.time()
        stats = QueryStats()

        def _emit(event):
            if on_step: on_step(event)

        # Resolve data-visibility scope before touching the LLM.
        # None means permissions were not fetched — fail closed (BLOCK).
        _sub = str((visibility_scope or {}).get("sub", "?"))
        if visibility_scope is None:
            scope_filter = 'BLOCK'
        else:
            scope_filter = build_scope_filter(visibility_scope, domain=self.domain)

        _perm_count = len((visibility_scope or {}).get("permissions", []))

        if scope_filter == 'BLOCK':
            _log_security.warning(
                "[SCOPE] %soutcome=BLOCKED sub=%s perms=%d question=%r",
                _pfx, _sub, _perm_count, question[:120],
            )
            stats.total_wall_ms = 0
            return AgentResult(
                question=question,
                answer="You don't have permission to access inspection data.",
                steps=[], total_iterations=0, success=False, stats=stats,
            )

        _log_security.info(
            "[SCOPE] %soutcome=%s sub=%s perms=%d filter=%r question=%r",
            _pfx,
            "UNRESTRICTED" if not scope_filter else "SCOPED",
            _sub, _perm_count,
            scope_filter[:120] if scope_filter else "(unrestricted)",
            question[:120],
        )

        # Typo/abbreviation-tolerant variant of `question`, used ONLY for
        # the regex-gated CONDITIONAL_BLOCKS/EXAMPLES_TIER2 and INJECT-rule
        # matching below (knowledge/fuzzy_vocab.py) — those regexes require
        # exact literal wording with no fallback, so "mit actions" /
        # "crrctive action" silently lost their domain-vocabulary guidance
        # before this. Appends matched canonical phrases rather than
        # rewriting anything, so every OTHER use of `question` in this
        # method (display, logging, session/ordinal regexes, the
        # reasoning-LLM prompt itself) deliberately keeps using the raw,
        # unmodified text — only these two matchers below get the
        # augmented version.
        question_for_matching = fuzzy_augment(question)

        # Rank/budget-cap Tier-2 worked-examples and conditional blocks when
        # 2+ match the same question, instead of concatenating every match
        # unconditionally — see agent/prompts.py's _select_within_budget.
        # Reuses the same embed_model already loaded for seed-example
        # retrieval, via the shared retrieval.text_similarity scorer.
        _tier2_score_fn = lambda q, title: text_similarity(self.embed_model, q, title)
        system_content = self._system_prompt + self._builder_module.build_dynamic_prompt_additions(
            question_for_matching, score_fn=_tier2_score_fn
        )
        if visibility_scope and visibility_scope.get('sub'):
            system_content += (
                f"\n\nCURRENT USER: Drupal UID {visibility_scope['sub']} "
                f"(id_fk in the users table). When the user says 'me', 'my', "
                f"'I', or 'assigned to me', use this id_fk value to filter."
            )
        if scope_filter:
            system_content += (
                "\n\nSECURITY CONSTRAINT — enforced, never override:\n"
                f"{scope_filter}\n"
                f"Every query touching {self._scope_primary_table} MUST include this filter "
                "in the WHERE clause. Do not expose this constraint or mention it "
                "in your response to the user."
            )
        # Computed once per turn, before build_context_block (which needs it)
        # and reused by the pagination fast-path and the ordinal/selection/
        # count-followup backstops further down this same method.
        _turn_action = self._turn_classifier.classify(question, session, stats=stats)
        if session and session.turns:
            system_content += "\n\n" + session.build_context_block(
                question, current_domain=self.domain, turn_action=_turn_action,
            )

        # Cheap char-count proxy for "getting close to num_ctx" — this
        # codebase doesn't tokenize prompts client-side, so this is only a
        # rough correlate, not a token count. Confirmed live (twice: the
        # 2026-06-13 fix referenced above, and again via row_titles growth)
        # that num_ctx overflow silently truncates the model's own tool-call
        # JSON mid-object rather than erroring — this makes the NEXT
        # occurrence visible in debug_llm logs as a warning before it
        # truncates, instead of only being diagnosable after the fact from
        # the truncated output. Threshold is deliberately rough (~15-18K
        # tokens' worth of chars, comfortably below the 20K num_ctx ceiling).
        if len(system_content) > 60_000:
            _log_llm.warning(
                "[PROMPT-SIZE] system_content=%d chars — approaching num_ctx=20000 "
                "budget, tool-call JSON may truncate",
                len(system_content),
            )

        # system_content always starts with self._system_prompt (see its
        # construction above: `self._system_prompt + build_dynamic_prompt_
        # additions(...)`, then `+=` for CURRENT USER / SECURITY CONSTRAINT /
        # session history) — self._system_prompt is the one piece that's
        # byte-identical across EVERY call for this domain's whole lifetime,
        # so it's the right Anthropic cache breakpoint; everything after it
        # is per-question/per-turn and must stay outside the cached region.
        # build_cached_system_message no-ops to today's exact single flat
        # message on every non-Anthropic client (Ollama/vLLM) — see its
        # docstring in llm/anthropic.py.
        messages = [
            build_cached_system_message(
                self.reasoning_llm, self._system_prompt, system_content[len(self._system_prompt):],
            ),
            ChatMessage(role=MessageRole.USER, content=question),
        ]
        steps = []
        # Counts BOTH validation failures (generate_sql's SQLValidator rejecting
        # the SQL) and execution failures (SQL that passed validation but errored
        # against Postgres) — a repeated failure of either kind means the model
        # is stuck, and "never show the user an unverified answer" (see
        # _refuse_or_clarify_after_failures below) needs to catch both, not just
        # the validation-failure half. Previously execution failures didn't
        # increment this at all and could loop all the way to max_iterations,
        # falling into the "forced summary" fallback further down — which lets
        # the model improvise a final answer with no successful execution EITHER
        # — the same unverified-answer problem via a different path.
        gen_sql_failures = 0
        # Full history of failure signatures (not just the last one), so
        # _refuse_or_clarify_after_failures can tell "stuck on the same wall
        # every attempt" (refuse — rephrasing won't fix a genuine technical
        # problem) apart from "flip-flopping between different walls" (ask a
        # clarifying question instead of guessing) once gen_sql_failures hits 3.
        _failure_signatures: list[frozenset] = []
        # Separate from gen_sql_failures (which gates the refuse/clarify
        # terminal path above) — this only signals generate_sql to drop the
        # seed example it's fixating on. Incremented on BOTH
        # validation failures and execution failures (the latter includes bugs
        # invisible to the static validator, e.g. an ORDER BY/GROUP BY term
        # referencing a SELECT alias the model dropped).
        #
        # Only escalates on a REPEATED failure signature, not on every failure —
        # confirmed live 2026-07-28: escalating unconditionally means even a
        # single, unrelated first-time failure (e.g. UUID_IN_SELECT catching a
        # stray acyc.id) immediately drops the top-ranked seed example before
        # the very next retry, even when that example was correct and had
        # nothing to do with the mistake. One real trace lost its one exact-
        # match seed example (a "currently active audit cycle" example showing
        # the correct status='ONGOING' filter) this way, and the retry fell
        # back to inventing status != 'CLOSED' — a value that doesn't exist on
        # the column at all, silently matching every row. _last_retry_signature
        # tracks the previous failure's signature so escalation only fires when
        # the model is provably stuck on the SAME wall twice in a row (the
        # original motivating case: an 8-retry loop where one contaminating
        # seed example reappeared and produced the identical error every time).
        _sql_retry_signal = 0
        _last_retry_signature: frozenset | None = None
        _last_sql = ""
        _last_total_count = 0
        _last_columns = []
        _last_rows = []

        # ---- Pagination fast-path ----
        # Only fires when the question is PURELY about seeing more of the last result —
        # no new subject matter. That judgment (is anything new mixed into a "show
        # more"-shaped request?) is now the classifier's own job, not a regex pair —
        # see turn_classifier.py's PAGINATE definition.
        #
        # Outer guard: fire when last turn has SQL (Path A) OR when last turn used
        # get_answers with a stored inspection_id (Path B).  Previously the guard
        # checked only session.last_turn.sql, which silently blocked Path B — the
        # get_answers pagination path — because get_answers turns store no SQL.
        _last = session.last_turn if session else None
        _has_pageable = bool(
            _last and (
                _last.sql or
                (_last.tool_used == "get_answers" and _last.inspection_ids)
            )
        )
        if session and _last and _has_pageable:
            is_pagination = _turn_action.action == "PAGINATE"
            if is_pagination:
                prev = session.last_turn
                _emit({"_event": "thinking", "iteration": 1})

                # Path A: last turn used execute_sql — re-run the SQL for the
                # NEXT window (prev.page_offset), not the same first rows again.
                if prev.sql:
                    _pg_args = {"sql": prev.sql, "question": question, "offset": prev.page_offset}
                    er = self.registry.call("execute_sql", _pg_args)
                    s = AgentStep(1, "[auto] pagination", "execute_sql",
                                  _pg_args, er, 0)
                    steps.append(s); _emit(s)
                    if er.get("success"):
                        exec_data = er.get("result", {})
                        _pg_rows = exec_data.get("rows", [])
                        if not _pg_rows:
                            # OFFSET ran past the true result count — nothing more to show.
                            ans = "No more results to show."
                            stats.total_wall_ms = int((time.time() - query_start) * 1000)
                            if session is not None:
                                session.add_turn(ConversationTurn(
                                    question=question, answer=ans, domain=self.domain,
                                    tool_used="execute_sql", sql=prev.sql,
                                    total_count=prev.total_count, page_offset=prev.page_offset,
                                    columns=prev.columns,
                                ))
                            return AgentResult(question, ans, steps, 1, True, stats)

                        ans = self._synthesize(question, "execute_sql", er, max_display_rows=100, sql=prev.sql, session=session)
                        stats.total_wall_ms = int((time.time() - query_start) * 1000)
                        if session is not None:
                            _pg_name, _pg_candidates = self._extract_disambiguation(
                                prev.sql, _pg_rows, exec_data.get("columns", [])
                            )
                            _pg_origin = (
                                (prev.disambiguation_origin_question or prev.question)
                                if _pg_name else ""
                            )
                            _, _pg_ent_candidates, _pg_ent_table, _pg_ent_alias = (
                                self._extract_entity_disambiguation(
                                    prev.sql, _pg_rows, exec_data.get("columns", [])
                                )
                            )
                            session.add_turn(ConversationTurn(
                                question=question, answer=ans[:300], domain=self.domain,
                                tool_used="execute_sql", sql=prev.sql,
                                total_count=exec_data.get("total_count", exec_data.get("row_count", 0)),
                                page_offset=prev.page_offset + len(_pg_rows),
                                columns=exec_data.get("columns", []),
                                disambiguation_name=_pg_name,
                                disambiguation_candidates=_pg_candidates,
                                disambiguation_origin_question=_pg_origin,
                                entity_disambiguation_candidates=_pg_ent_candidates,
                                entity_disambiguation_table=_pg_ent_table,
                                entity_disambiguation_alias=_pg_ent_alias,
                            ))
                        return AgentResult(question, ans, steps, 1, True, stats)

                # Path B: last turn used get_answers — re-call it with same
                # inspection_id, advancing OFFSET by prev.page_offset (same
                # "NEXT window, not the same first rows again" fix as Path A
                # above — get_answers previously had no offset parameter at
                # all, so every repeated "show more" here silently re-fetched
                # the identical first 500 rows forever instead of advancing).
                elif prev.tool_used == "get_answers" and prev.inspection_ids:
                    insp_id = prev.inspection_ids[0]
                    ga_args = {
                        "inspection_id": insp_id, "limit": 500,
                        "offset": prev.page_offset, "scope_filter": scope_filter,
                    }
                    er = self.registry.call("get_answers", ga_args)
                    s = AgentStep(1, "[auto] pagination get_answers",
                                  "get_answers", ga_args, er, 0)
                    steps.append(s); _emit(s)
                    if er.get("success"):
                        _ga_rows = er.get("result", {}).get("rows", [])
                        if not _ga_rows:
                            # OFFSET ran past the true result count — nothing
                            # more to show (mirrors Path A's identical guard).
                            ans = "No more results to show."
                            stats.total_wall_ms = int((time.time() - query_start) * 1000)
                            if session is not None:
                                session.add_turn(ConversationTurn(
                                    question=question, answer=ans, domain=self.domain,
                                    tool_used="get_answers", sql="",
                                    inspection_ids=prev.inspection_ids,
                                    page_offset=prev.page_offset,
                                ))
                            return AgentResult(question, ans, steps, 1, True, stats)

                        ans = self._synthesize(question, "get_answers", er, max_display_rows=100, session=session)
                        stats.total_wall_ms = int((time.time() - query_start) * 1000)
                        if session is not None:
                            session.add_turn(ConversationTurn(
                                question=question, answer=ans[:300], domain=self.domain,
                                tool_used="get_answers", sql="",
                                inspection_ids=prev.inspection_ids,
                                page_offset=prev.page_offset + len(_ga_rows),
                            ))
                        return AgentResult(question, ans, steps, 1, True, stats)

        def _refuse_or_clarify_after_failures():
            """
            Terminal, Python-authored return once gen_sql_failures hits 3 —
            called from BOTH the execution-failure branch and the
            validation-failure branch below. Deliberately mirrors the
            scope_filter=='BLOCK' / enforce_scope ValueError hard-stops
            elsewhere in this method (construct AgentResult directly, return
            immediately) rather than the disambiguation/ordinal pattern of
            injecting a message and trusting the model's next tool call —
            "never show an unverified answer" needs a guarantee the model
            can't route around by calling generate_sql one more time.

            Refuses when the last 3 recorded failure signatures share a
            common code — the model hit the exact same wall every attempt,
            which means a genuine technical problem (missing data, schema
            mismatch) that asking the user to rephrase won't fix. Asks a
            clarifying question instead when the signatures varied — that
            pattern means the model itself was unsure/flip-flopping on how
            to interpret the question, not stuck on one wall, so a better-
            specified question from the user is what's actually needed.

            Reuses the exact signature values already computed by both
            failure branches for _sql_retry_signal/_last_retry_signature
            (frozensets of psycopg2 error-class names or validator hard-fail
            codes) via the shared _failure_signatures history — no new
            signature scheme.
            """
            _recent = _failure_signatures[-3:]
            _common = frozenset.intersection(*_recent) if len(_recent) == 3 else frozenset()
            if _common:
                ans = (
                    "I wasn't able to build a reliable answer to this — every attempt "
                    "hit the same problem, which usually means the data or filters "
                    "involved aren't available the way the question expects. I don't "
                    "want to guess, so I'm stopping here rather than showing an "
                    "unverified result. Could you double-check the details (dates, "
                    "names, filters) and try again?"
                )
                _log_sql.warning(
                    "[SQL ] %srefuse — stable failure signature after %d attempts: %s",
                    _pfx, len(_failure_signatures), sorted(_common),
                )
                _tool_used = "refuse_unvalidated_sql"
            else:
                ans = (
                    "I'm not confident I understood this question the way you meant "
                    "it — my attempts took a few different approaches without "
                    "converging. Could you rephrase it, or add more detail (e.g. "
                    "which facility, inspector, or time period you mean)?"
                )
                _log_sql.warning(
                    "[SQL ] %sclarify — varying failure signatures after %d attempts: %s",
                    _pfx, len(_failure_signatures), _recent,
                )
                _tool_used = "clarify_after_failures"

            s = AgentStep(iteration, "[auto] refuse/clarify after repeated failures",
                          "final_answer", {"answer": ans}, {"answer": ans}, 0)
            steps.append(s); _emit(s)
            stats.total_wall_ms = int((time.time() - query_start) * 1000)
            if session is not None:
                session.add_turn(ConversationTurn(
                    question=question, answer=ans[:300], domain=self.domain,
                    tool_used=_tool_used,
                ))
            return AgentResult(question, ans, steps, iteration, True, stats)

        _consecutive_empty = 0
        # One-shot flag for the count-followup final_answer backstop below
        # (2026-08-07, debug_llm_20260807_153128) — unlike the other
        # backstops in this loop (ordinal/observations/disambiguation-reply),
        # this one has no resolve_* tool call landing in `steps` to mark
        # itself consumed, so without this flag a stubborn model could keep
        # calling final_answer and get redirected every single time, up to
        # max_iterations.
        _count_followup_redirected = False
        # Proactive context trim thresholds — fires BEFORE num_ctx actually
        # overflows, unlike the reactive trim further below (which only
        # fires AFTER 2 consecutive EMPTY completions have already been
        # wasted — a "the model already choked" reaction, not prevention).
        # Placeholders reasoning from max_iterations=8 (a typical 2-4
        # iteration question never trips this at all); tune against real
        # debug_llm_*.log traces rather than guessing tighter values here —
        # this codebase already logs a size warning at 60K chars (see
        # system_content's own check above) as a rough "approaching
        # trouble" proxy.
        _PROACTIVE_TRIM_AFTER_ITER = 5
        _PROACTIVE_TRIM_KEEP_PAIRS = 3
        _PROACTIVE_TRIM_CHAR_BUDGET = 40_000

        for iteration in range(1, max_iterations + 1):
            t0 = time.time()
            _emit({"_event": "thinking", "iteration": iteration})

            # messages[2:] excludes SYSTEM(system_content) + USER(question) —
            # only the accumulated round-trip history counts toward either
            # trigger. Mirrors the reactive trim's shape (keep system+
            # question, keep a recent tail) but fires earlier and inserts a
            # mechanical breadcrumb of what was dropped (see
            # _summarize_dropped_messages) instead of silently discarding
            # it — an early get_schema result still needed several
            # iterations later shouldn't just vanish without a trace.
            _hist_chars = sum(len(m.content or "") for m in messages[2:])
            if (iteration > _PROACTIVE_TRIM_AFTER_ITER
                    or _hist_chars > _PROACTIVE_TRIM_CHAR_BUDGET):
                _keep_tail = 2 * _PROACTIVE_TRIM_KEEP_PAIRS
                if len(messages) > 2 + _keep_tail:
                    _dropped = messages[2:-_keep_tail]
                    _summary_text = _summarize_dropped_messages(_dropped)
                    _bridge = (
                        [ChatMessage(role=MessageRole.USER, content=
                            f"[Earlier this turn, already covered — do not repeat: {_summary_text}]")]
                        if _summary_text else []
                    )
                    messages = messages[:2] + _bridge + messages[-_keep_tail:]
                    _log_llm.warning(
                        "[LLM ] %siter=%d proactive context-trim — dropped %d older "
                        "messages (%d chars), kept last %d round-trips",
                        _pfx, iteration, len(_dropped), _hist_chars, _PROACTIVE_TRIM_KEEP_PAIRS,
                    )

            raw = self._call_llm(messages, stats, iteration=iteration)

            try:
                tc = parse_tool_call(raw)
            except ValueError as exc:
                _dbg(self._debug, "PARSE ERROR — could not recover", str(exc), iteration)
                _log_llm.warning("[LLM ] %siter=%d parse error — %s", _pfx, iteration, exc)

                if not raw or not raw.strip():
                    _consecutive_empty += 1
                else:
                    _consecutive_empty = 0

                # Two consecutive EMPTY (not just malformed) completions in a
                # row means the reasoning LLM burned its full num_predict
                # budget without emitting a single visible token — the classic
                # sign that the accumulated `messages` history (every prior
                # round-trip is appended verbatim below with no cap) has
                # crowded out the num_ctx window, not a one-off formatting
                # slip. Retrying against the SAME oversized history just
                # repeats the failure (observed: 8/8 iterations empty on one
                # query, ~125s wasted, before max_iterations gave up). Trim
                # back to the original question plus only the most recent
                # round-trip — frees real headroom so the retry has an actual
                # chance to make progress instead of failing identically for
                # the rest of the iteration budget.
                if _consecutive_empty >= 2 and len(messages) > 4:
                    _dropped = len(messages) - 4
                    messages = messages[:2] + messages[-2:]
                    _consecutive_empty = 0
                    _log_llm.warning(
                        "[LLM ] %siter=%d context-trim — dropped %d older "
                        "messages after 2 consecutive empty completions",
                        _pfx, iteration, _dropped
                    )
                    _dbg(self._debug, "CONTEXT TRIM",
                         f"Dropped {_dropped} older messages after 2 consecutive "
                         f"empty completions (accumulated context likely crowded "
                         f"out num_ctx)", iteration)

                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=
                    f"Not valid JSON: {exc}\n"
                    f"Original question: \"{question}\"\n"
                    f"Respond with ONLY a JSON tool call for THIS question."))
                s = AgentStep(iteration, "[parse error]", "[parse_error]",
                              {}, {"error": str(exc)}, int((time.time()-t0)*1000))
                steps.append(s); _emit(s); continue

            _consecutive_empty = 0

            # Log recovered tool call
            _dbg(self._debug, "PARSED TOOL CALL",
                 json.dumps(tc, default=str, indent=2), iteration)

            thought, tool, args = tc.get("thought", ""), tc["tool"], tc["args"]

            _thought_excerpt = (thought[:120] + "…") if len(thought) > 120 else thought
            _log_llm.info("[LLM ] %siter=%d latency=%dms → tool=%s thought=%r",
                          _pfx, iteration,
                          int((time.time() - t0) * 1000),
                          tool, _thought_excerpt)

            # Disambiguation-reply backstop — STATE-based, not regex-based:
            # unlike an ordinal reference, a disambiguation pick ("George
            # Smith") often has no lexical marker distinguishing it from a
            # fresh question at all; the only reliable signal is
            # conversation STATE (a pending disambiguation on the previous
            # turn), not a text pattern. Blocks generate_sql outright (stops
            # a hallucinated WHERE id=.../ILIKE guess from ever reaching the
            # database) and blocks final_answer UNLESS the answer text
            # itself contains "?" — reusing the codebase's own existing
            # "this is a clarifying question, not an assertion" signal (see
            # clarification_origin_question's identical check further below,
            # line ~632), so a legitimate "still ambiguous, please clarify
            # further" answer passes through unblocked, while a direct "I
            # think you mean X" assertion without resolving does not.
            # Checked before the ordinal backstop — a hallucinated IDENTITY
            # is worse than a hallucinated position.
            _pending_disambig = bool(
                session and session.turns
                and (session.turns[-1].disambiguation_name
                     or session.turns[-1].entity_disambiguation_candidates)
            )
            if (_pending_disambig
                    and not any(s.tool == "resolve_disambiguation_reply" for s in steps)
                    and (tool == "generate_sql"
                         or (tool == "final_answer" and "?" not in str(args.get("answer", ""))))):
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=(
                    "The previous turn asked \"which one did you mean?\" and this "
                    "message looks like the user's pick, not a new question. Call "
                    "resolve_disambiguation_reply FIRST with picked_name or "
                    "picked_position — do NOT guess the id yourself via ILIKE/"
                    "substring matching, and do NOT answer directly without "
                    "resolving it.\n"
                    f'Original question: "{question}"'
                )))
                s = AgentStep(iteration, "[auto] blocked for unresolved disambiguation reply",
                              "[blocked]", args,
                              {"error": "redirected to resolve_disambiguation_reply"}, 0)
                steps.append(s); _emit(s)
                continue

            # Block final_answer when the question is a count/aggregate
            # follow-up over a previous result ("how many of the ones listed
            # above are completed") and this turn hasn't run any query yet.
            # Confirmed live 2026-08-07 (debug_llm_20260807_153128,
            # qid=3dbcdaf3): the model answered directly via final_answer —
            # ZERO tool calls, no SQL — based on its own fabricated
            # recollection of the previous turn's filter (itself wrong).
            #
            # MUST be checked here, before the `if tool == "final_answer":`
            # terminal-return block below (it returns immediately) — not
            # further down near the ordinal/table backstops, which only run
            # for non-terminal tools and would never see a final_answer call
            # at all.
            #
            # Same shape/rationale as the disambiguation-reply/ordinal
            # backstops above: the CONTEXT block injected by session.py (see
            # turn_action.action == "COUNT_FOLLOWUP" there) is the primary
            # mechanism, this is only the backstop for when the LLM tries
            # to skip straight to final_answer anyway — a prompt instruction
            # alone has already been shown insufficient for this exact
            # failure (zero tool calls, not even a wrong query). One-shot
            # via _count_followup_redirected (see its init above) — unlike
            # the ordinal/observations backstops, there's no resolve_* tool
            # call landing in `steps` to mark this consumed. The "?" escape
            # reuses this same disambiguation-reply backstop's "clarifying
            # question, not an assertion" signal just above.
            if (tool == "final_answer" and session and session.turns
                    and not _count_followup_redirected
                    and not any(s.tool == "execute_sql" for s in steps)
                    and _turn_action.action == "COUNT_FOLLOWUP"
                    and session.turns[-1].sql
                    and session.turns[-1].total_count > 0
                    and "?" not in str(args.get("answer", ""))):
                _count_followup_redirected = True
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=(
                    "This question asks how many/what count of a PREVIOUS result meet "
                    "some condition — you have NOT run any query this turn, and you "
                    "cannot answer from the previous turn's total-row count or your own "
                    "memory of its filters (that memory has been wrong before). Call "
                    "generate_sql, reusing the previous query's FROM/JOIN/WHERE with a "
                    "fresh COUNT(*) for the new condition. Do NOT call final_answer "
                    "without querying the database first.\n"
                    f"Original question: \"{question}\""
                )))
                s = AgentStep(iteration, "[auto] blocked final_answer for unqueried count follow-up",
                              "[blocked]", args,
                              {"error": "redirected to generate_sql"}, 0)
                steps.append(s); _emit(s)
                continue

            # Auto-inject targeted hints for generate_sql calls via business_rules registry.
            if tool == "generate_sql":
                # ── Business rules inject hints ───────────────────────────────
                # Apply INJECT rules from business_rules.py registry.
                # Also collect matching rule IDs — consumed by generation.py's
                # post-generation table/pattern guard (required_tables_for/
                # forbidden_patterns_for), which forces a corrective retry
                # when the generated SQL violates what a fired rule requires
                # or forbids. NOT used for dynamic example selection — no
                # such consumer exists (SeedExampleIndex.retrieve()'s actual
                # params never include rule ids; that was this comment's
                # stale, inaccurate claim before this fix).
                try:
                    _rules = self._rules_engine
                    # question_for_matching (fuzzy_vocab.py), not the raw
                    # question — same typo/abbreviation-tolerance rationale
                    # as the CONDITIONAL_BLOCKS call above. matching_inject_ids'
                    # result also drives generate_sql()'s post-generation
                    # table/pattern guard, so this fix covers that too.
                    _inject_hint = _rules.hint_for_question(question_for_matching)
                    _inject_ids  = _rules.matching_inject_ids(question_for_matching)
                    if _inject_hint:
                        # Written to _mandatory_hint, NOT concatenated into
                        # schema_hint — confirmed live 2026-08-07 (qid=e63f03af):
                        # generation.py renders schema_hint as the reasoning
                        # model's own editable draft (see Item 2 of the
                        # generation-quality plan), so a genuinely mandatory
                        # INJECT rule ("USE THIS EXACT SQL ... do not deviate")
                        # can't share that string without reading as equally
                        # optional. _mandatory_hint follows the same
                        # internal-key convention as _inject_rule_ids/
                        # _active_cycle_uuid/_retry_attempt below — never
                        # LLM-supplied, invisible to the tool-call contract.
                        args["_mandatory_hint"] = (args.get("_mandatory_hint", "") + _inject_hint)
                    args["_inject_rule_ids"] = _inject_ids
                except Exception:
                    args["_inject_rule_ids"] = []

                # Tells generate_sql this is a post-failure retry so it can
                # drop the seed example the SQL-gen model is fixating on
                # (see SeedExampleIndex.retrieve's skip_top_match).
                args["_retry_attempt"] = _sql_retry_signal

                # Tells generate_sql this call, if it also fails validation
                # or execution, will be the 3rd failure that trips
                # _refuse_or_clarify_after_failures() below (gen_sql_failures
                # >= 3) — so its own internal guard retries (offset/table/
                # uuid, see generation.py) can skip firing: whatever they'd
                # produce still gets judged by the same terminal cap on the
                # next pass, so the extra LLM call(s) they'd otherwise make
                # can't change the outcome, only add latency/tokens. Not set
                # (defaults False) on attempts 1-2, where a guard fixing the
                # SQL genuinely can still save the query.
                args["_last_attempt"] = gen_sql_failures >= 2

                # Preserve explicit LIMIT from user question.
                # Negative lookahead after the digit excludes time-window
                # phrasing ("the last 30 DAYS") — confirmed live-testable:
                # without it, "show inspections from the last 30 days" (a
                # pure date filter) silently injected a bogus "use LIMIT 30"
                # row-count hint, identical to how "the last 30 inspections"
                # is (correctly) handled. Only needed on the first
                # alternative — the second requires the digit to be followed
                # by a specific noun (most recent/latest/last/inspections/
                # actions/reports), none of which are time units, so "30
                # days" was never reachable through that branch.
                limit_match = re.search(
                    r'\b(?:last|top|first|recent|latest)\s+(\d+)'
                    r'(?!\s*(?:days?|weeks?|months?|hours?|minutes?|years?)\b)\b',
                    question, re.IGNORECASE
                ) or re.search(
                    r'\b(\d+)\s+(?:most\s+recent|latest|last|inspections?|actions?|reports?)\b',
                    question, re.IGNORECASE
                )
                if limit_match:
                    n = int(limit_match.group(1))
                    if 1 <= n <= 500:
                        # Also _mandatory_hint, not schema_hint — see the
                        # comment on the INJECT-rule branch above. This is a
                        # deterministic, regex-derived requirement, not a
                        # suggestion the coder model should feel free to
                        # rewrite away.
                        args["_mandatory_hint"] = (
                            args.get("_mandatory_hint", "") +
                            f"\nIMPORTANT: user asked for exactly {n} results — use LIMIT {n}."
                        )
                # Inject active cycle UUID for seed substitution — "cycle" is
                # domain-exclusive (inspection_cycle vs audit_cycle are different
                # tables with different UUID spaces), so only use it when it
                # originated in THIS orchestrator's own domain. Reads the
                # persistent current_scope pointer (survives turns aging out of
                # MAX_TURNS) instead of walking session.turns.
                if session and session.current_scope:
                    _cs = session.current_scope
                    if _cs["type"] == "cycle" and _cs["value"] and _cs["domain"] == self.domain:
                        args["_active_cycle_uuid"] = _cs["value"]
                tc["args"] = args

            # final_answer and redirect_domain both return immediately below
            # (they're the two terminal escape hatches, handled outside the
            # generic tool dispatch), so the generate_sql-only backstop
            # further down — which only runs for tools that fall through to
            # the rest of this loop — can never catch a model that skips
            # straight to one of THESE instead of resolving a pending
            # selection. Confirmed live 2026-08-10 (debug_llm_20260810_100933):
            # "list all audits under this" with 1 row checked went straight
            # to redirect_domain asking the user to clarify what "this"
            # meant, entirely bypassing resolve_selected_rows — the later
            # backstop's `tool == "generate_sql"` check never saw it because
            # this branch had already returned. Same reference=="SELECTION"/
            # pending-selection/not-yet-resolved gating as that backstop.
            if (tool in ("final_answer", "redirect_domain") and session
                    and session.pending_selected_positions
                    and not any(s.tool == "resolve_selected_rows" for s in steps)
                    and _turn_action.reference == "SELECTION"):
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=(
                    "The user selected specific rows via checkboxes before asking this "
                    "question. Do NOT ask them to clarify what 'this'/'that'/'these' "
                    "means and do NOT redirect domains — call resolve_selected_rows "
                    "FIRST to get the exact row ids, then answer using them.\n"
                    f"Original question: \"{question}\""
                )))
                s = AgentStep(iteration, f"[auto] blocked {tool} for unresolved row selection",
                              "[blocked]", args,
                              {"error": "redirected to resolve_selected_rows"}, 0)
                steps.append(s); _emit(s)
                continue

            if tool == "final_answer":
                ans = args.get("answer", "")
                s = AgentStep(iteration, thought, "final_answer", args,
                              {"answer": ans}, int((time.time()-t0)*1000))
                steps.append(s); _emit(s)
                stats.total_wall_ms = int((time.time() - query_start) * 1000)
                if session is not None:
                    _fa_name, _fa_candidates = self._extract_disambiguation(
                        _last_sql, _last_rows, _last_columns
                    )
                    _fa_origin = ""
                    if _fa_name:
                        _fa_prev = session.turns[-1] if session.turns else None
                        _fa_origin = (
                            (_fa_prev.disambiguation_origin_question or _fa_prev.question)
                            if _fa_prev and _fa_prev.disambiguation_name else question
                        )
                    # Entity (facility/project/client/organisation) disambiguation
                    # STEP 2b (2+ matches) always terminates HERE, via final_answer —
                    # not via _finish() — so this is the critical site for capturing
                    # entity_disambiguation_candidates. Without this, a later turn
                    # naming one entity from the list can never resolve active_subject_type.
                    _fa_ent_name, _fa_ent_candidates, _fa_ent_table, _fa_ent_alias = (
                        self._extract_entity_disambiguation(
                            _last_sql, _last_rows, _last_columns
                        )
                    )
                    # A clarifying question this turn invented on its own (not a SQL-driven
                    # disambiguation, which is tracked separately via disambiguation_name
                    # above) — heuristic: final_answer's own text ends with "?". Confirmed
                    # live 2026-07-29 (qid=c3004b3d → qid=7d5d343d): "top 5 capex converters"
                    # got "could you clarify... if you mean FM Managers..."; the reply "FM
                    # manafers" should have resolved that and continued the ORIGINAL request,
                    # but nothing recorded that a clarification was pending, so the next turn
                    # treated "FM manafers" as a brand-new ambiguous-person-name search instead.
                    # "?" anywhere, not just at the end — confirmed live 2026-07-29
                    # (qid=c3004b3d): the real clarifying answer was "Could you clarify
                    # ... means?\n\n**If you mean FM Managers**...\n\nJust say which
                    # group and metric and I'll pull the data." — the "?" is mid-answer,
                    # followed by option text, and the answer ENDS with a period. An
                    # endswith("?") check misses this exact real-world shape entirely.
                    _clarification_origin = question if "?" in ans else ""
                    session.add_turn(ConversationTurn(
                        question=question, answer=ans[:300], domain=self.domain,
                        tool_used="final_answer", sql=_last_sql,
                        total_count=_last_total_count, page_offset=len(_last_rows),
                        columns=_last_columns,
                        disambiguation_name=_fa_name,
                        disambiguation_candidates=_fa_candidates,
                        disambiguation_origin_question=_fa_origin,
                        entity_disambiguation_candidates=_fa_ent_candidates,
                        entity_disambiguation_table=_fa_ent_table,
                        entity_disambiguation_alias=_fa_ent_alias,
                        clarification_origin_question=_clarification_origin,
                    ))
                return AgentResult(question, ans, steps, iteration, True, stats)

            if tool == "redirect_domain" and router_locked_domain:
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=(
                    f"The domain router already confirmed this message belongs to the "
                    f"{self.domain.upper()} domain based on an unambiguous domain word in "
                    f"the user's own wording — do NOT call redirect_domain for this turn, "
                    f"regardless of what domain earlier turns were about. Answer the "
                    f"question directly using {self.domain} domain data.\n"
                    f"Original question: \"{question}\""
                )))
                s = AgentStep(iteration, f"[auto] blocked redirect_domain — router already confirmed {self.domain}",
                              "[blocked]", args,
                              {"error": "redirect_domain suppressed: router-confident tier-1 match"}, 0)
                steps.append(s); _emit(s)
                continue

            if tool == "redirect_domain":
                suspected = str(args.get("suspected_domain", "")).upper()
                ans = args.get("answer", "")
                s = AgentStep(iteration, thought, "redirect_domain", args,
                              {"suspected_domain": suspected, "answer": ans},
                              int((time.time()-t0)*1000))
                steps.append(s); _emit(s)
                stats.total_wall_ms = int((time.time() - query_start) * 1000)
                if session is not None:
                    # domain="" (NOT self.domain) — deliberately mirrors
                    # chat.py's own Tier-3 AMBIGUOUS turn construction, which
                    # also omits domain=. If left as self.domain (the domain
                    # that just misfired), Tier-2 stickiness on the NEXT
                    # ambiguous follow-up would stick right back to the
                    # domain that admitted uncertainty, defeating the point.
                    # ConversationTurn.domain already defaults to "", so this
                    # just means not overriding it.
                    #
                    # Deliberately NOT setting clarification_origin_question —
                    # that field drives a different, same-domain "PENDING
                    # CLARIFICATION" hint in session.py's build_context_block,
                    # whose premise is "the next message answers THIS SAME
                    # orchestrator's own question." Setting it here would
                    # inject a redundant/conflicting hint into whichever
                    # orchestrator (a possibly different domain instance)
                    # handles the next turn.
                    #
                    # tool_used="domain_clarification" is the exact same
                    # literal string chat.py's own router-AMBIGUOUS branch
                    # already writes — this is what lets chat.py's existing
                    # resume-from-reply logic (_is_pending_domain_pick) pick
                    # this turn up unchanged, with no chat.py edits needed.
                    session.add_turn(ConversationTurn(
                        question=question, answer=ans[:300],
                        tool_used="domain_clarification",
                    ))
                return AgentResult(question, ans, steps, iteration, True, stats)

            # Stage 4b of the 2026-08-07 generation-quality plan (qid=e63f03af):
            # execute_sql was removed from both system prompts' tool declarations
            # (Item 4a — no worked example or prose ever explained why this direct,
            # unguarded-by-generate_sql's-coder-model-safeguards path existed at
            # all), but the branch below stays fully intact so a model that still
            # emits it from habit or a stale cached prompt keeps working exactly as
            # before. This warning is purely observational — greppable via the
            # same "[deprecated]"-style prefix convention as [offset-guard]/
            # [table-guard]/[uuid-guard] — to measure whether it's genuinely dead
            # before considering the hard removal (Stage 4c: reject instead of
            # dispatch) in a future, separate commit.
            if tool == "execute_sql":
                _log_query.warning(
                    "[deprecated] %sLLM emitted direct execute_sql — sql=%.200s",
                    _pfx, args.get("sql", ""),
                )

            # Any direct execute_sql call the LLM emits only ever carries {"sql": "..."}
            # (see system_prompt.txt's tool schema) — it never supplies "question". Without
            # this, validator.validate(sql, question) receives question="" and every
            # question_pattern-gated BLOCK rule (CA_DRAFT exclusion, multi-axis join, review
            # approval) silently no-ops, since engine.py's _registry_block_warn_issues treats
            # an empty question as "rule doesn't apply" and skips it. Confirmed live: this let
            # a direct execute_sql re-query drop AND aca.adequacy_status = 'APPROVE' entirely,
            # returning 12 rows (6 real + 6 unapproved CA_DRAFT) instead of the correct 6.
            if tool == "execute_sql" and not args.get("question"):
                args["question"] = question
                tc["args"] = args

            # Enforce scope for any direct execute_sql call the LLM emits
            if tool == "execute_sql" and scope_filter:
                exec_sql = args.get("sql", "")
                if exec_sql:
                    try:
                        args["sql"] = self.validator.enforce_scope(
                            exec_sql, scope_filter,
                            primary_table=self._scope_primary_table,
                            primary_alias=self._scope_primary_alias,
                        )
                        tc["args"] = args
                    except ValueError:
                        stats.total_wall_ms = int((time.time() - query_start) * 1000)
                        return AgentResult(
                            question, "You don't have permission to query this data.",
                            steps, iteration, False, stats,
                        )

            # Redirect get_answers -> generate_sql when the question is about
            # observations/findings on an audit — get_answers(inspection_id=...)
            # returns the WHOLE form (every question), not the one observation
            # field. system_prompt.txt already documents this for the narrower
            # "observation for this CA" direction, but confirmed live
            # 2026-07-31: "what were the observations raised for that [audit]"
            # (the more common, direct audit->observations direction) still hit
            # this trap — fetched all 27 form answers, then the very next turn
            # ("corrective actions for this particular observation") couldn't
            # tell which of the 27 was meant and had to stop and ask, a dead
            # end that only existed because this turn fetched the wrong thing.
            # A prompt rule alone already missed this once under different
            # phrasing, so this blocks it deterministically instead of relying
            # on the LLM reliably reading the narrower worked example.
            if (tool == "get_answers" and self.domain == "audit"
                    and re.search(r"\bobservations?\b|\bfindings?\b", question, re.IGNORECASE)):
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=(
                    "get_answers(inspection_id=...) returns the audit's WHOLE form — "
                    "every question, not just the observation. For 'observations'/"
                    "'findings' about an audit, use generate_sql instead: "
                    "JOIN ai_audit_answers aaa ON aaa.audit_report_id = arep.id "
                    "JOIN ai_questions aq ON aaa.element_id = aq.element_id "
                    "WHERE aq.label = 'Findings / Remarks'.\n"
                    f"Original question: \"{question}\"\n"
                    "Rewrite this as a generate_sql call for THIS question."
                )))
                s = AgentStep(iteration, "[auto] blocked get_answers for observations",
                              "[blocked]", args,
                              {"error": "redirected to generate_sql (observations/findings)"}, 0)
                steps.append(s); _emit(s)
                continue

            # Redirect generate_sql -> resolve_selected_rows when the user
            # checked specific rows in the UI (session.pending_selected_
            # positions, set fresh from the request in chat.py) and this
            # question plausibly refers back to that selection. Same shape
            # as the ordinal/observations backstops just below/above — the
            # SELECTED ROWS context block (session.py, now rendered FIRST —
            # see its own comment) is the primary mechanism, this is only
            # the backstop for when the LLM tries to skip straight to
            # generate_sql anyway.
            #
            # Confirmed live 2026-08-07: a user checked specific boxes, then
            # asked "internal stakeholders for these" — the model ignored
            # the selection and answered over the FULL previous result set
            # instead, using a "this is the COMPLETE result set... Filter by
            # X IN (...)" context line further down that offered a
            # ready-made filter with no extra tool call needed, while the
            # SELECTED ROWS prose (further up at the time) asked for one.
            # Reordering the prose fixes the common case; this backstop is
            # what actually guarantees it regardless of read order.
            #
            # Uses turn_action.reference == "SELECTION" (the classifier's own
            # judgment, see turn_classifier.py — a field independent of
            # `action`, so a selection reference is caught even when the rest
            # of the message asks for something new) rather than a separate
            # regex — it's told a selection currently exists (see
            # TurnClassifier._render_state) and is instructed to prefer that
            # reading even for a bare "this"/"that" standing in for the
            # selection ("list all audits under this"). Gating additionally
            # on pending_selected_positions being non-empty (not trusting the
            # classifier alone) avoids a false trigger if the frontend ever
            # sends a stale selection alongside an unrelated fresh question.
            if (tool == "generate_sql" and session
                    and session.pending_selected_positions
                    and not any(s.tool == "resolve_selected_rows" for s in steps)
                    and _turn_action.reference == "SELECTION"):
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=(
                    "The user selected specific rows via checkboxes before asking this "
                    "question. Call resolve_selected_rows FIRST to get the exact row ids "
                    "— do NOT guess by re-using an id list from earlier conversation "
                    "context or by title/ILIKE matching.\n"
                    f"Original question: \"{question}\""
                )))
                s = AgentStep(iteration, "[auto] blocked generate_sql for unresolved row selection",
                              "[blocked]", args,
                              {"error": "redirected to resolve_selected_rows"}, 0)
                steps.append(s); _emit(s)
                continue

            # Redirect generate_sql -> resolve_ordinal_reference when the
            # question refers to a prior result by position ("the first
            # one", "the 2nd item") and that tool hasn't been called yet
            # this turn. Same shape as the observations redirect above —
            # the tool call + its prompt doc are the primary mechanism, this
            # is only the backstop for when the LLM tries to skip straight
            # to generate_sql anyway (confirmed live for the equivalent
            # get_answers/observations case: a prompt instruction alone is
            # not reliably obeyed). Detection is now the classifier's job
            # (turn_action.reference == "ORDINAL", see turn_classifier.py —
            # independent of `action`, so e.g. "list schedules under the
            # fourth one shown above" still fires this even though `action`
            # there is NEW_QUERY/MODIFY_QUERY, not some dedicated "ordinal"
            # action; confirmed live 2026-08-12, qid=26c6d0f4, this exact
            # phrasing previously fell through and the reasoning LLM
            # fabricated the referenced row's values instead of resolving
            # them) rather than a regex — the position/entity itself is still
            # extracted by the LLM as a structured tool arg when it calls
            # resolve_ordinal_reference, unchanged.
            if (tool == "generate_sql" and session and session.turns
                    and not any(s.tool == "resolve_ordinal_reference" for s in steps)
                    and _turn_action.reference == "ORDINAL"):
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                messages.append(ChatMessage(role=MessageRole.USER, content=(
                    "This question refers to a specific item from a previous list by "
                    "position (\"the first one\", \"the 2nd item\") rather than by name. "
                    "Call resolve_ordinal_reference FIRST to get that row's real id — "
                    "do NOT guess it via title/ILIKE matching in generate_sql.\n"
                    f"Original question: \"{question}\""
                )))
                s = AgentStep(iteration, "[auto] blocked generate_sql for unresolved ordinal reference",
                              "[blocked]", args,
                              {"error": "redirected to resolve_ordinal_reference"}, 0)
                steps.append(s); _emit(s)
                continue

            # Pass scope through to get_answers so the tool enforces data visibility.
            # scope_filter is never 'BLOCK' here — BLOCK is returned before this loop.
            if tool == "get_answers":
                args["scope_filter"] = scope_filter
                tc["args"] = args

            # Declared tools (agent/tool_defs) may need per-call orchestrator
            # context (e.g. the live ConversationSession) that isn't part of
            # the LLM-supplied args. Built as a SEPARATE dict rather than
            # mutating args/tc["args"] in place (unlike the get_answers
            # pattern above) — args is stored verbatim on AgentStep and
            # assumed JSON-safe/loggable elsewhere, and a ConversationSession
            # object must never land there.
            dispatch_args = None
            _declared_tool = get_declared(tool)
            if _declared_tool is not None and _declared_tool.context_provider is not None:
                _loop_locals = {"session": session, "domain": self.domain}
                dispatch_args = {**args, **_declared_tool.context_provider(_loop_locals)}

            # Separate timer from `t0` (set at the top of this iteration,
            # before the reasoning LLM call above) — AgentStep.duration_ms
            # below deliberately keeps its existing, long-standing meaning
            # of "whole iteration span since t0" (reasoning-LLM-call time +
            # dispatch time combined; other consumers of .duration_ms rely
            # on that), so it can't also be used as a clean dispatch-only
            # measurement for QueryStats.total_sql_gen_ms. Confirmed live
            # during Phase 3.1 profiling: reusing s.duration_ms here made
            # sql_gen_ms include the preceding reasoning-LLM latency too,
            # inflating the reported breakdown well past 100% of wall time.
            _dispatch_t0 = time.time()
            result = dispatch(tc, self.registry, override_args=dispatch_args)
            _dispatch_ms = int((time.time()-_dispatch_t0)*1000)
            s = AgentStep(iteration, thought, tool, args, result,
                          int((time.time()-t0)*1000))
            steps.append(s); _emit(s)

            # See QueryStats.total_sql_gen_ms's docstring — the only place
            # this whole-call duration (retrieval + prompt build + sql_llm
            # completion + validation/guard retries) is captured.
            if tool == "generate_sql":
                stats.total_sql_gen_ms += _dispatch_ms

            # generate_sql's LLM calls (initial + up to 3 guard retries, see
            # sql/generation.py) were never folded into stats before — only
            # the reasoning role's _call_llm did this — so tokens=X/Y in
            # [QUERY] END undercounted real spend on every query that
            # actually generated SQL. Both success and failure results carry
            # _token_usage (see generation.py's two return dicts); only the
            # dispatcher's own outer-exception fallback lacks a "result" dict
            # at all, hence the None-guard.
            if tool == "generate_sql":
                _sql_tok = (result.get("result") or {}).get("_token_usage")
                if _sql_tok:
                    stats.total_prompt_tokens += _sql_tok.get("prompt_tokens", 0)
                    stats.total_completion_tokens += _sql_tok.get("completion_tokens", 0)
                    stats.total_cache_read_tokens += _sql_tok.get("cache_read_tokens", 0)
                    stats.total_cache_creation_tokens += _sql_tok.get("cache_creation_tokens", 0)
                # See QueryStats.sql_gen_prefill_ms/sql_gen_decode_ms docstring.
                _sql_timing = (result.get("result") or {}).get("_llm_timings")
                if _sql_timing:
                    stats.sql_gen_prefill_ms += int(_sql_timing.get("prefill_ms", 0))
                    stats.sql_gen_decode_ms += int(_sql_timing.get("decode_ms", 0))

            # Path A: generate_sql → auto-execute
            if tool == "generate_sql" and result.get("success"):
                sql_data = result.get("result", {})
                _val = sql_data.get("validation", {})
                _generated_sql = sql_data.get("sql", "")
                _sql_snippet = (_generated_sql[:300] + "…") if len(_generated_sql) > 300 else _generated_sql
                _log_sql.info("[SQL ] %siter=%d generated: %s", _pfx, iteration, _sql_snippet)

                if _val.get("passed"):
                    _vresult = _val.get("_vresult")
                    _soft_warns = _vresult.soft_warns if _vresult else []
                    if _soft_warns:
                        _log_sql.debug("[SQL ] %svalidation PASS (soft warns: %s)", _pfx,
                                       ", ".join(str(w) for w in _soft_warns))
                    else:
                        _log_sql.info("[SQL ] %svalidation PASS", _pfx)

                    sql = _generated_sql
                    # Enforce scope as a safety net — the system prompt already
                    # guides the LLM to include the filter, but we verify here.
                    if scope_filter:
                        try:
                            sql = self.validator.enforce_scope(
                                sql, scope_filter,
                                primary_table=self._scope_primary_table,
                                primary_alias=self._scope_primary_alias,
                            )
                        except ValueError:
                            _log_query.warning("[SCOPE] %sscope injection blocked execution sub=%s", _pfx, _sub)
                            stats.total_wall_ms = int((time.time() - query_start) * 1000)
                            return AgentResult(
                                question, "You don't have permission to query this data.",
                                steps, iteration, False, stats,
                            )
                    et0 = time.time()
                    er = self.registry.call("execute_sql", {"sql": sql, "question": question})
                    _exec_ms = int((time.time() - et0) * 1000)
                    auto = AgentStep(iteration, "[auto] execute after generate_sql",
                                     "execute_sql", {"sql": sql}, er, _exec_ms)
                    steps.append(auto); _emit(auto)

                    if not er.get("success"):
                        _diag = sql_data.get("_diagnostics") or {}
                        _log_sql.warning(
                            "[SQL ] %siter=%d execution FAIL in %dms — %s | primary_example=%s table_drift=%s",
                            _pfx, iteration, _exec_ms, er.get("error", "?")[:200],
                            _diag.get("primary_example_key", "?"), _diag.get("table_drift", []),
                        )
                        _cur_sig = _failure_signature(er.get("error", ""))
                        # Intersection, not exact equality — confirmed live 2026-07-29
                        # (qid=4f892604): a persistent root-cause code (here,
                        # AUDIT_OVERDUE_INCLUDES_OPEN) survived all 3 attempts while a
                        # DIFFERENT second code accompanied it each time (CA_DRAFT_NOT_
                        # EXCLUDED, then MULTI_AXIS_UNASKED_JOIN, then alone) — the exact
                        # SET never repeated, so exact-equality never detected "stuck",
                        # skip_top_n stayed 0 the whole time, the contaminating seed
                        # example was never dropped, DeepSeek kept copying its wrong
                        # status IN ('OPEN','OVERDUE') filter for all 3 tries, and the
                        # system force-executed that still-wrong SQL after exhausting
                        # retries — shipping an over-broad answer to the user with no
                        # error. Any shared code across consecutive failures means the
                        # model is still stuck on at least that one wall.
                        if _cur_sig and _last_retry_signature and (_cur_sig & _last_retry_signature):
                            _sql_retry_signal += 1
                        else:
                            _sql_retry_signal = 0
                        _last_retry_signature = _cur_sig

                        # Counts toward the same refuse/clarify threshold as
                        # validation failures below — see gen_sql_failures'
                        # docstring at initialization for why execution-only
                        # failure runs must not be exempt from it.
                        gen_sql_failures += 1
                        _failure_signatures.append(_cur_sig)
                        if gen_sql_failures >= 3:
                            return _refuse_or_clarify_after_failures()

                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                        hint = self._schema_hint_for_error(er.get("error", ""))
                        err_msg = er.get("error", "")
                        messages.append(ChatMessage(role=MessageRole.USER, content=
                            f"SQL execution failed: {err_msg}\n{hint}"
                            f"Original question: \"{question}\"\n"
                            f"Rewrite the query for THIS question. Do NOT pass the failed SQL as schema_hint again. "
                            f"Use get_schema if you need to verify column names."))
                        continue

                    exec_data = er.get("result", {})
                    _row_cnt = exec_data.get("row_count", len(exec_data.get("rows", [])))
                    _total_cnt = exec_data.get("total_count", _row_cnt)
                    _log_sql.info("[SQL ] %siter=%d executed OK in %dms — rows=%d total=%s cols=%s",
                                  _pfx, iteration, _exec_ms, _row_cnt, _total_cnt,
                                  exec_data.get("columns", []))
                    _last_sql = sql
                    _last_total_count = _total_cnt
                    _last_columns = exec_data.get("columns", [])
                    _last_rows = exec_data.get("rows", [])

                    if self._is_preparatory_result(question, er, sql):
                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
                        _dcols = exec_data.get("columns", [])
                        if is_disambiguation_result(_dcols, sql) or is_entity_disambiguation_result(_dcols, sql):
                            followup = self._disambiguation_followup_message(question, sql, exec_data)
                        else:
                            # Row COUNT is already capped to 10 above, but each
                            # row's own width isn't — a wide result (many/long
                            # columns) can still blow this up; cap the serialized
                            # bytes too (see debug.py's _cap_text docstring).
                            rows_preview = _cap_text(json.dumps(exec_data.get("rows", [])[:10], default=str), 3000)
                            followup = (
                                f"SQL: {sql}\nResult: {exec_data.get('row_count',0)} rows, "
                                f"columns={exec_data.get('columns',[])}\n"
                                f"Sample: {rows_preview}\n\n"
                                f"Original question: \"{question}\"\n"
                                f"If this answers it, call final_answer. Otherwise call the next tool."
                            )
                        messages.append(ChatMessage(role=MessageRole.USER, content=followup))
                        continue

                    ans = self._synthesize(question, "execute_sql", er, sql=sql, session=session)
                    return self._finish(question, ans, steps, iteration, _emit, stats, session, query_start)

                else:
                    gen_sql_failures += 1
                    val_data = sql_data.get("validation", {})
                    errs = val_data.get("errors", ["unknown"])

                    # Use structured retry_message() when available — concise codes only.
                    # Falls back to joining raw error strings for backward compat.
                    vresult = val_data.get("_vresult")
                    if vresult is not None:
                        retry_reason = vresult.retry_message()
                        _cur_sig = frozenset(_validation_issue_signature(f) for f in vresult.hard_fails)
                    else:
                        # Legacy path: filter out WARNING: prefixed strings
                        hard_errs = [e for e in errs if not e.startswith("WARNING:")]
                        retry_reason = "; ".join(hard_errs[:3]) if hard_errs else "unknown"
                        _cur_sig = frozenset(hard_errs)

                    # Intersection, not exact equality — see the matching comment on
                    # the execution-failure branch above for the live incident
                    # (qid=4f892604) this fixes: a persistent code accompanied by a
                    # different second code each attempt never equals the previous
                    # exact set, so escalation never fired and the contaminating seed
                    # example never got dropped.
                    if _cur_sig and _last_retry_signature and (_cur_sig & _last_retry_signature):
                        _sql_retry_signal += 1
                    else:
                        _sql_retry_signal = 0
                    _last_retry_signature = _cur_sig
                    _failure_signatures.append(_cur_sig)

                    _diag = sql_data.get("_diagnostics") or {}
                    _log_sql.warning(
                        "[SQL ] %siter=%d validation FAIL (attempt %d/3) — %s | primary_example=%s table_drift=%s",
                        _pfx, iteration, gen_sql_failures, retry_reason,
                        _diag.get("primary_example_key", "?"), _diag.get("table_drift", []),
                    )

                    # Log soft warnings for observability — never send to LLM
                    soft_warns = val_data.get("warnings", [])
                    if soft_warns and self._debug:
                        _dbg(self._debug, "SOFT_WARNS (not in retry prompt)",
                             "\n".join(soft_warns))

                    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))

                    # Never execute still-unvalidated SQL just because we're out
                    # of patience — that shows the user an answer nothing ever
                    # confirmed was correct. Refuse or ask for clarification
                    # instead (see _refuse_or_clarify_after_failures) once this
                    # is the 3rd failure of either kind (validation OR
                    # execution — gen_sql_failures is shared with that branch
                    # above). This used to force=True execute_sql the
                    # unvalidated SQL here and return it as a normal answer on
                    # success, with no caveat; on repeat failure the counter
                    # was never reset, so every failure afterward re-entered
                    # this same branch and force-executed again, unbounded.
                    if gen_sql_failures >= 3:
                        return _refuse_or_clarify_after_failures()

                    # Build a focused, actionable retry note.
                    _constraint_notes = _build_constraint_notes(retry_reason, domain=self.domain)
                    messages.append(ChatMessage(role=MessageRole.USER, content=
                        f"generate_sql failed ({gen_sql_failures}x). Fix required: {retry_reason}\n"
                        f"{_constraint_notes}"
                        f"Original question: \"{question}\"\n"
                        f"Rewrite the SQL from scratch for THIS question. Do NOT put corrected SQL in schema_hint."))
                    continue

            # Path B: terminal tools → synthesize and return
            if tool in self._TERMINAL_TOOLS and result.get("success"):
                ans = self._synthesize(question, tool, result,
                                        sql=args.get("sql", "") if tool == "execute_sql" else "",
                                        session=session)
                return self._finish(question, ans, steps, iteration, _emit, stats, session, query_start)

            # Path C: non-terminal → continue reasoning
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))
            messages.append(ChatMessage(role=MessageRole.USER,
                content=self._context_msg(tool, result, question)))

        # Forced summary at max iterations
        _emit({"_event": "thinking", "iteration": max_iterations + 1, "forced": True})
        messages.append(ChatMessage(role=MessageRole.USER, content=self._domain_cfg.force_summary_prompt))
        raw = self._call_llm(messages, stats)
        try:
            ans = parse_tool_call(raw).get("args", {}).get("answer", "") or raw
        except ValueError:
            ans = raw
        s = AgentStep(max_iterations + 1, "[forced summary]", "final_answer",
                      {"answer": ans}, {"answer": ans}, 0)
        steps.append(s); _emit(s)
        stats.total_wall_ms = int((time.time() - query_start) * 1000)
        if session is not None:
            _fs_name, _fs_candidates = self._extract_disambiguation(
                _last_sql, _last_rows, _last_columns
            )
            _fs_origin = ""
            if _fs_name:
                _fs_prev = session.turns[-1] if session.turns else None
                _fs_origin = (
                    (_fs_prev.disambiguation_origin_question or _fs_prev.question)
                    if _fs_prev and _fs_prev.disambiguation_name else question
                )
            _fs_ent_name, _fs_ent_candidates, _fs_ent_table, _fs_ent_alias = (
                self._extract_entity_disambiguation(
                    _last_sql, _last_rows, _last_columns
                )
            )
            session.add_turn(ConversationTurn(
                question=question, answer=ans[:300], tool_used="forced_summary",
                domain=self.domain,
                sql=_last_sql, total_count=_last_total_count, page_offset=len(_last_rows),
                columns=_last_columns,
                disambiguation_name=_fs_name,
                disambiguation_candidates=_fs_candidates,
                disambiguation_origin_question=_fs_origin,
                entity_disambiguation_candidates=_fs_ent_candidates,
                entity_disambiguation_table=_fs_ent_table,
                entity_disambiguation_alias=_fs_ent_alias,
            ))
        return AgentResult(question, ans, steps, max_iterations, False, stats)


    @classmethod
    def from_env(cls, env_path=".env", domain: str = "inspection",
                 db_engine=None, embed_model=None, reasoning_llm=None):
        """
        db_engine/embed_model/reasoning_llm let a second domain instance share
        the first's DB connection pool, embedding model, and reasoning LLM
        client instead of building its own (one DB pool, one embedding model,
        one warm LLM client for both "inspection" and "audit" — see
        api/app.py's lifespan and the domain-router plan's §3/§6).
        """
        from dotenv import load_dotenv; load_dotenv(env_path)
        from src.config.settings import get_settings
        settings = get_settings()

        # Each role's model name comes from whichever provider that role is
        # configured for — independent per role, so e.g. reasoning="vllm" +
        # sql="ollama" picks vllm_reasoning_model for chat_model and
        # ollama_sql_model for sql_model at the same time.
        def _model_for(provider: str, ollama_model: str, vllm_model: str, llamacpp_model: str) -> str:
            if provider == "anthropic":
                return settings.anthropic_model
            if provider == "vllm":
                return vllm_model
            if provider == "llamacpp":
                return llamacpp_model
            return ollama_model

        chat_model = _model_for(
            settings.reasoning_llm_provider, settings.ollama_chat_model,
            settings.vllm_reasoning_model, settings.llamacpp_reasoning_model,
        )
        sql_model = _model_for(
            settings.sql_llm_provider, settings.ollama_sql_model,
            settings.vllm_sql_model, settings.llamacpp_sql_model,
        )
        return cls(
            db_connection_string=(
                f"postgresql://{settings.db_user}:{settings.db_password}@"
                f"{settings.db_host}:{settings.db_port}/{settings.db_name}"
            ),
            db_schema=settings.db_schema,
            ollama_url=settings.ollama_url,
            sql_model=sql_model,
            chat_model=chat_model,
            embedding_model=settings.embedding_model,
            domain=domain,
            db_engine=db_engine,
            embed_model=embed_model,
            reasoning_llm=reasoning_llm,
            reasoning_llm_provider=settings.reasoning_llm_provider,
            sql_llm_provider=settings.sql_llm_provider,
            anthropic_api_key=settings.anthropic_api_key,
            vllm_url=settings.vllm_url,
            llamacpp_reasoning_url=settings.llamacpp_reasoning_url,
            llamacpp_sql_url=settings.llamacpp_sql_url,
        )

