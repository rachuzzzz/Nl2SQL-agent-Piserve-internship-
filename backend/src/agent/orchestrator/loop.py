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

from src.llm.factory import get_reasoning_llm, get_sql_llm
from src.retrieval import SemanticQuestionIndex, SeedExampleIndex, text_similarity
from src.sql.validation import SQLValidator
from src.agent.domains import get_domain_config
from src.agent.tools import ToolRegistry
from src.agent.tool_dispatcher import parse_tool_call, dispatch
from src.security.rbac import build_scope_filter
from src.sql.result_shape import is_disambiguation_result, is_entity_disambiguation_result

from src.agent.orchestrator.models import AgentStep, QueryStats, ConversationTurn, AgentResult
from src.agent.orchestrator.debug import _make_debug_logger, _dbg, _build_constraint_notes, _CONSTRAINT_NOTE_MAP
from src.agent.orchestrator.synthesis import SynthesisMixin
from src.agent.orchestrator.disambiguation import DisambiguationMixin


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


class AgentOrchestrator(SynthesisMixin, DisambiguationMixin):
    def __init__(self, db_connection_string, db_schema="public",
                 ollama_url="http://localhost:11434", sql_model="sqlcoder:7b",
                 chat_model="qwen2.5:14b-instruct-q4_K_M",
                 embedding_model="BAAI/bge-small-en-v1.5",
                 domain: str = "inspection",
                 db_engine=None, embed_model=None, reasoning_llm=None):
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
        # The domain's business rules/table relationships (already embedded in
        # for_reasoning_prompt() via DatabaseSchema's injected constant set —
        # see knowledge/schema_introspector.py's DomainConfig fields) are
        # re-appended verbatim since routing logic still relies on them.
        _rules_block = "BUSINESS_RULES:\n" + "\n".join(
            f"  - {r}" for r in self.db_schema._business_rules
        )
        self._system_prompt = self._domain_cfg.system_prompt.replace(
            self._domain_cfg.schema_placeholder,
            self.db_schema.for_reasoning_prompt() + "\n\n" + _rules_block
            + "\n\n" + self.db_schema._table_relationships
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

        # Build seed example index — embeds NL questions for retrieval-based injection
        self.seed_index = SeedExampleIndex(
            embed_model=self.embed_model,
            examples=self._domain_cfg.seed_examples,
        )

        self.chat_model_name = chat_model
        self.sql_model_name  = sql_model

        if reasoning_llm is not None:
            self.reasoning_llm = reasoning_llm
            print("  ✓ Reasoning LLM reused from another domain instance")
        else:
            print(f"  Configuring reasoning LLM: {chat_model}...")
            self.reasoning_llm = get_reasoning_llm(chat_model, ollama_url)
            print(f"  ✓ Reasoning LLM ({chat_model})")

        print(f"  Configuring SQL LLM: {sql_model}...")
        self.sql_llm = get_sql_llm(sql_model, ollama_url)
        print(f"  ✓ SQL LLM ({sql_model})")

        self._debug = _make_debug_logger()

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
            domain=self.domain,
        )

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
              visibility_scope: dict | None = None, query_id: str = ""):
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

        # Rank/budget-cap Tier-2 worked-examples and conditional blocks when
        # 2+ match the same question, instead of concatenating every match
        # unconditionally — see agent/prompts.py's _select_within_budget.
        # Reuses the same embed_model already loaded for seed-example
        # retrieval, via the shared retrieval.text_similarity scorer.
        _tier2_score_fn = lambda q, title: text_similarity(self.embed_model, q, title)
        system_content = self._system_prompt + self._builder_module.build_dynamic_prompt_additions(
            question, score_fn=_tier2_score_fn
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
        if session and session.turns:
            system_content += "\n\n" + session.build_context_block(question, current_domain=self.domain)

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

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_content),
            ChatMessage(role=MessageRole.USER, content=question),
        ]
        steps = []
        gen_sql_failures = 0
        # Separate from gen_sql_failures (which gates the "force-execute after
        # 3 validation failures" safety valve) — this only signals generate_sql
        # to drop the seed example it's fixating on. Incremented on BOTH
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
        # no new subject matter (no content words suggesting a different query).
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
            q_lower = question.lower().strip()

            _PAGINATION_ONLY_RE = re.compile(
                r'^(show\s+more|show\s+the\s+rest|show\s+remaining|'
                r'see\s+more|see\s+the\s+rest|'
                r'more\s+results?|more\s+rows?|'
                r'continue|next\s+page|next\s+batch|'
                r'rest\s+of\s+(it|them|the\s+results?)?|'
                r'show\s+all\s*$|list\s+all\s*$|'
                r'remaining\s+(ones?|results?|rows?)?)'
                r'\s*[.!?]?\s*$',
                re.IGNORECASE,
            )
            _HAS_NEW_SUBJECT_RE = re.compile(
                r'\b(high|low|medium|risk|observation|finding|action|mitigative|'
                r'recommended|facility|inspector|score|form|module|inspection|cause|'
                r'overdue|open|closed|month|week|today|by\s+\w+|for\s+\w+|'
                r'at\s+\w+|in\s+\w+|about|from\s+\w+)\b',
                re.IGNORECASE,
            )
            is_pagination = (
                bool(_PAGINATION_ONLY_RE.match(q_lower)) and
                not bool(_HAS_NEW_SUBJECT_RE.search(q_lower))
            )
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

                # Path B: last turn used get_answers — re-call it with same inspection_id
                elif prev.tool_used == "get_answers" and prev.inspection_ids:
                    insp_id = prev.inspection_ids[0]
                    ga_args = {"inspection_id": insp_id, "limit": 500, "scope_filter": scope_filter}
                    er = self.registry.call("get_answers", ga_args)
                    s = AgentStep(1, "[auto] pagination get_answers",
                                  "get_answers", ga_args, er, 0)
                    steps.append(s); _emit(s)
                    if er.get("success"):
                        ans = self._synthesize(question, "get_answers", er, max_display_rows=100, session=session)
                        stats.total_wall_ms = int((time.time() - query_start) * 1000)
                        if session is not None:
                            session.add_turn(ConversationTurn(
                                question=question, answer=ans[:300], domain=self.domain,
                                tool_used="get_answers", sql="",
                                inspection_ids=prev.inspection_ids,
                            ))
                        return AgentResult(question, ans, steps, 1, True, stats)

        _consecutive_empty = 0
        for iteration in range(1, max_iterations + 1):
            t0 = time.time()
            _emit({"_event": "thinking", "iteration": iteration})

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

            # Auto-inject targeted hints for generate_sql calls via business_rules registry.
            if tool == "generate_sql":
                # ── Business rules inject hints ───────────────────────────────
                # Apply INJECT rules from business_rules.py registry.
                # Also collect matching rule IDs for dynamic example selection.
                try:
                    _rules = self._rules_engine
                    _inject_hint = _rules.hint_for_question(question)
                    _inject_ids  = _rules.matching_inject_ids(question)
                    if _inject_hint:
                        args["schema_hint"] = (args.get("schema_hint", "") + _inject_hint)
                    args["_inject_rule_ids"] = _inject_ids
                except Exception:
                    args["_inject_rule_ids"] = []

                # Tells generate_sql this is a post-failure retry so it can
                # drop the seed example the SQL-gen model is fixating on
                # (see SeedExampleIndex.retrieve's skip_top_match).
                args["_retry_attempt"] = _sql_retry_signal

                # Preserve explicit LIMIT from user question
                limit_match = re.search(
                    r'\b(?:last|top|first|recent|latest)\s+(\d+)\b', question, re.IGNORECASE
                ) or re.search(
                    r'\b(\d+)\s+(?:most\s+recent|latest|last|inspections?|actions?|reports?)\b',
                    question, re.IGNORECASE
                )
                if limit_match:
                    n = int(limit_match.group(1))
                    if 1 <= n <= 500:
                        args["schema_hint"] = (
                            args.get("schema_hint", "") +
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

            # Pass scope through to get_answers so the tool enforces data visibility.
            # scope_filter is never 'BLOCK' here — BLOCK is returned before this loop.
            if tool == "get_answers":
                args["scope_filter"] = scope_filter
                tc["args"] = args

            result = dispatch(tc, self.registry)
            s = AgentStep(iteration, thought, tool, args, result,
                          int((time.time()-t0)*1000))
            steps.append(s); _emit(s)

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
                        _log_sql.warning("[SQL ] %siter=%d execution FAIL in %dms — %s",
                                         _pfx, iteration, _exec_ms, er.get("error", "?")[:200])
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
                            rows_preview = json.dumps(exec_data.get("rows", [])[:10], default=str)
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
                        _cur_sig = frozenset(f.code for f in vresult.hard_fails)
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

                    _log_sql.warning("[SQL ] %siter=%d validation FAIL (attempt %d/3) — %s",
                                     _pfx, iteration, gen_sql_failures, retry_reason)

                    # Log soft warnings for observability — never send to LLM
                    soft_warns = val_data.get("warnings", [])
                    if soft_warns and self._debug:
                        _dbg(self._debug, "SOFT_WARNS (not in retry prompt)",
                             "\n".join(soft_warns))

                    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw))

                    if gen_sql_failures >= 3:
                        sql = sql_data.get("sql", "")
                        if sql:
                            _log_sql.warning("[SQL ] %sforced execution after %d validation failures",
                                             _pfx, gen_sql_failures)
                            # force=True: let Postgres itself be the final arbiter for any
                            # RETRYABLE validation issue (logical-correctness guards like
                            # UNDEFINED_ALIAS/CAPEX_LEFT_JOIN_MULT) instead of looping
                            # forever — execute_sql() still hard-blocks non-retryable
                            # issues (DML_BLOCKED, EMPTY_SQL) regardless of force=True.
                            # Confirmed live: without force=True this call re-validates
                            # the SAME sql, fails identically every time, and "forced
                            # execution" never actually forced anything — one real trace
                            # looped 8 iterations / 76s / ~172K tokens past this exact
                            # point because of it.
                            er = self.registry.call("execute_sql",
                                                     {"sql": sql, "question": question, "force": True})
                            auto = AgentStep(iteration, "[auto] forced exec after 3 failures",
                                             "execute_sql", {"sql": sql}, er, 0)
                            steps.append(auto); _emit(auto)
                            if er.get("success"):
                                ans = self._synthesize(question, "execute_sql", er, sql=sql, session=session)
                                return self._finish(question, ans, steps, iteration, _emit, stats, session, query_start)

                            # Forced execution still failed — either a genuine Postgres
                            # error (the real DB confirming the SQL is actually broken,
                            # not a validator false-positive — much stronger signal than
                            # the same validator message the model already saw 3x) or a
                            # non-retryable safety block. Surface whichever it was.
                            _forced_err = er.get("error", "") or retry_reason
                            messages.append(ChatMessage(role=MessageRole.USER, content=
                                f"Forced execution after {gen_sql_failures} failures also failed: {_forced_err}\n"
                                f"Original question: \"{question}\"\n"
                                f"This came from the real database, not just the validator — rewrite the SQL "
                                f"from scratch to actually fix this, don't assume it's a false positive."))
                            continue

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
        return cls(
            db_connection_string=(
                f"postgresql://{settings.db_user}:{settings.db_password}@"
                f"{settings.db_host}:{settings.db_port}/{settings.db_name}"
            ),
            db_schema=settings.db_schema,
            ollama_url=settings.ollama_url,
            sql_model=settings.ollama_sql_model,
            chat_model=settings.ollama_chat_model,
            embedding_model=settings.embedding_model,
            domain=domain,
            db_engine=db_engine,
            embed_model=embed_model,
            reasoning_llm=reasoning_llm,
        )

