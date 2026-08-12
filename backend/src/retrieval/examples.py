"""
SeedExampleIndex — retrieval-reranking index for NL->SQL seed examples.
"""

import numpy as np
from typing import Optional


class SeedExampleIndex:
    """
    Retrieval-reranking index for NL→SQL seed examples.

    Architecture:
      Stage 1 — Embedding retrieval (bge-small-en-v1.5):
        Embed NL questions at startup. At query time, cosine-similarity
        finds top_k * RERANK_FACTOR candidate examples.

      Stage 2 — Reranking (one of two modes):
        MODE A — CrossEncoder (when RERANKER_MODEL env var is set):
          Uses BAAI/bge-reranker-base (or any sentence-transformers CrossEncoder).
          Scores each (query, candidate_nl) pair jointly — true pairwise scoring.
          Much better at distinguishing semantically similar but table-different queries.
        MODE B — Augmented embedding fallback (when no CrossEncoder available):
          Appends SQL table tokens to candidate NL before re-embedding.
          Score = 0.6 * original_similarity + 0.4 * augmented_similarity.

      Stage 3 — NL-prefix deduplication:
        Keep only the highest-scoring example per 4-word NL fingerprint.
        Prevents near-duplicate examples from contradicting each other.

    Environment variables:
      RERANKER_MODEL  — path or HuggingFace model name for CrossEncoder
                        e.g. RERANKER_MODEL=/models/bge-reranker-base
                             RERANKER_MODEL=BAAI/bge-reranker-base
      RERANKER_DEVICE — 'cpu' (default) or 'cuda'
    """

    RERANK_FACTOR = 5      # fetch top_k * RERANK_FACTOR candidates, rerank → top_k
    ALPHA         = 0.40   # fallback mode: weight of augmented score vs original

    _TABLE_TOKENS = [
        "po_details", "pr_details", "risk_level", "inspection_schedule",
        "inspection_cycle", "inspector_portfolio",
        "inspectee_user_id", "inspector_user_id", "frequency_definition",
        "user_roles",
    ]

    def __init__(self, embed_model, examples: list,
                 query_domain_tags: Optional[dict] = None,
                 query_domain_fallback: Optional[tuple] = None):
        """
        query_domain_tags / query_domain_fallback override the built-in
        _QUERY_DOMAIN keyword→tag map and the '[inspection]' fallback word
        list used by _query_tag() to boost/penalize candidates by domain tag
        (see retrieve()'s Stage 1). Both default to None, which keeps this
        class's original inspection-tuned behavior (self._QUERY_DOMAIN, the
        '[inspection]' fallback) — existing inspection callers are
        unaffected either way.

        Pass an explicit override (e.g. {} / ([], '')) for an index built
        from a DIFFERENT domain's seed examples. Confirmed live 2026-08-11:
        the inspection-only _QUERY_DOMAIN/fallback were being applied
        uniformly to the audit domain's index too — an audit query
        containing any inspection-vocabulary word (e.g. "schedule", which
        is also completely ordinary audit phrasing) got tagged '[schedule]'
        or '[inspection]', and every audit seed example (tagged by its own
        'domain' field, e.g. '[audit_report]') silently took the 0.88
        cross-domain-mismatch penalty on every such query — including
        exactly the query this was caught on ("list all audits — plans,
        schedules, conducted, and completed combined"), which suppressed a
        seed example carrying a required NOT EXISTS dedup guard from ever
        being retrieved. Pass query_domain_tags={} for any non-inspection
        index until that domain has its own tuned keyword map.
        """
        self.embed_model = embed_model
        self.examples = list(examples)
        self.embeddings: Optional[np.ndarray] = None
        self._augmented_embeddings: Optional[np.ndarray] = None
        self._cross_encoder = None  # set if RERANKER_MODEL env var is present
        self._query_domain_tags = (
            query_domain_tags if query_domain_tags is not None else self._QUERY_DOMAIN
        )
        self._query_domain_fallback_words, self._query_domain_fallback_tag = (
            query_domain_fallback if query_domain_fallback is not None
            else (['score', 'inspection', 'facility', 'inspector',
                   'completed', 'submitted', 'average'], '[inspection]')
        )

        if self.examples:
            print(f"  Building seed example index ({len(self.examples)} examples)...")

            # Use metadata from dict if available, otherwise derive from SQL
            # Supports both legacy tuple format and new dict format
            self._seed_domains = [
                ex.get('domain', '') if isinstance(ex, dict)
                else self._domain_tag(ex[1]).strip('[]')
                for ex in self.examples
            ]
            self._seed_intents = [
                ex.get('intent', 'general') if isinstance(ex, dict)
                else 'general'
                for ex in self.examples
            ]

            # Stage 1 embeddings: domain-tagged NL questions
            nl_texts = []
            for i, ex in enumerate(self.examples):
                nl   = ex['nl']  if isinstance(ex, dict) else ex[0]
                dom  = self._seed_domains[i]
                tag  = f'[{dom}]' if dom else ''
                nl_texts.append(f"{tag} {nl}" if tag else nl)

            self.embeddings = np.array(
                embed_model.get_text_embedding_batch(nl_texts)
            )

            # Stage 2 fallback embeddings: NL + SQL table tokens
            aug_texts = [
                self._augment(
                    ex['nl']  if isinstance(ex, dict) else ex[0],
                    ex['sql'] if isinstance(ex, dict) else ex[1],
                )
                for ex in self.examples
            ]
            self._augmented_embeddings = np.array(
                embed_model.get_text_embedding_batch(aug_texts)
            )

            # Keep legacy _seed_tags for domain boost in retrieve()
            self._seed_tags = [f'[{d}]' if d else '' for d in self._seed_domains]

            # Try loading CrossEncoder if RERANKER_MODEL is configured
            from src.config.settings import get_settings
            reranker_model = get_settings().reranker_model
            if reranker_model:
                try:
                    from sentence_transformers import CrossEncoder
                    device = get_settings().reranker_device
                    print(f"  Loading CrossEncoder reranker: {reranker_model} ({device})...")
                    self._cross_encoder = CrossEncoder(reranker_model, device=device)
                    print(f"  ✓ Seed example index ready (CrossEncoder reranker)")
                except Exception as e:
                    print(f"  ⚠ CrossEncoder load failed ({e}), falling back to embedding reranker")
                    self._cross_encoder = None
            else:
                print(f"  ✓ Seed example index ready (embedding reranker — set RERANKER_MODEL for CrossEncoder)")

    # Domain tags — prepended to seed NL and query before embedding.
    # This gives bge-small a hard categorical signal so financial seeds
    # don't retrieve for count/score queries and vice versa.
    _DOMAIN_RULES = [
        ('[financial]',        ['po_details', 'pr_details']),
        ('[form_answers]',     ['ai_answers', 'ai_questions']),
        ('[schedule]',         ['inspection_schedule', 'inspector_portfolio',
                                'frequency_definition', 'inspection_cycle']),
        ('[corrective_action]',['inspection_corrective_action']),
        ('[inspection]',       ['inspection_report']),
    ]
    # Query-side keyword signals for each domain tag
    _QUERY_DOMAIN = {
        '[financial]':         ['capex','opex','po ','purchase order','pr ',
                                'purchase requisition','converted','generated',
                                'conversion rate','financial'],
        '[form_answers]':      ['observation','risk level','answer','question',
                                'form','what did inspector','write','fill','answered',
                                'list questions','show questions'],
        '[schedule]':          ['schedule','portfolio','frequency','daily',
                                'cycle','assigned to','pending inspection'],
        '[corrective_action]': ['mitigative action','overdue action',
                                'cause','deferred','responsible party'],
    }

    @classmethod
    def _domain_tag(cls, sql: str) -> str:
        """Return the domain tag for a seed based on its SQL tables."""
        sql_up = sql.upper()
        # Check financial first (po/pr tables are unambiguous)
        if 'PO_DETAILS' in sql_up or 'PR_DETAILS' in sql_up:
            return '[financial]'
        for tag, tables in cls._DOMAIN_RULES:
            if any(t.upper() in sql_up for t in tables):
                return tag
        return ''

    def _query_tag(self, query: str) -> str:
        """Classify the user query into a domain tag using this index's
        keyword signals (self._query_domain_tags / self._query_domain_fallback_*
        — see __init__ for why these are instance-level, not the class
        constants directly)."""
        ql = query.lower()
        for tag, keywords in self._query_domain_tags.items():
            if any(kw in ql for kw in keywords):
                return tag
        if any(w in ql for w in self._query_domain_fallback_words):
            return self._query_domain_fallback_tag
        return ''

    @classmethod
    def _augment(cls, nl: str, sql: str) -> str:
        """Build an augmented representation: NL + detected table tokens from SQL."""
        sql_upper = sql.upper()
        found = [t for t in cls._TABLE_TOKENS if t.upper() in sql_upper]
        if found:
            return f"{nl} | tables: {' '.join(found)}"
        return nl

    @classmethod
    def _nl_fingerprint(cls, nl: str) -> str:
        """4-word content-word fingerprint for deduplication."""
        import re as _re
        words = _re.findall(r'\b[a-z]{3,}\b', nl.lower())
        stop = {'the','and','for','are','was','from','with','that',
                'this','all','how','what','which','show','list',
                'give','get','most','each','have','been','does'}
        content = [w for w in words if w not in stop]
        return ' '.join(content[:4])

    def retrieve(self, query: str, top_k: int = 4, skip_top_match: bool = False,
                 skip_top_n: int = 0) -> list:
        """
        Three-stage retrieval + reranking with intent pre-filtering:
          Stage 0: intent pre-filter — 'which single X' queries only see
                   single_value seeds; 'top N X' only sees ranked_list seeds.
                   Prevents FM-02-style loops where the top-5 seed beats LIMIT 1.
          Stage 1: domain-tagged cosine similarity → top_k * RERANK_FACTOR candidates
          Stage 2: rerank — CrossEncoder if available, else augmented embedding fallback
          Stage 3: deduplicate by 4-word NL fingerprint → return top_k

        skip_top_match: drop the single best-scoring example before returning
          (equivalent to skip_top_n=1 — kept for backward compatibility).
        skip_top_n: drop the top N best-scoring examples before returning. Use
          an ESCALATING count across retries, not just 1 — confirmed live: a
          persistently-contaminating example that happens to rank #2/#3/#4
          (never #1) survives skip_top_match=True on every single retry,
          since that only ever drops whichever example is #1 THAT round. One
          real trace burned 8 retries / 76s / ~172K tokens because the exact
          same seed example (containing an inspection_report 'ir' JOIN the
          model kept dropping while keeping a WHERE-clause fragment that
          referenced it) reappeared in the prompt on every single attempt.
          Escalating the drop count (e.g. skip_top_n = min(retry_attempt, 3))
          guarantees a stubborn example eventually falls out of the window
          instead of contaminating every retry indefinitely.

          INVARIANT: rank #1 (the single best-scoring example) is NEVER
          dropped, regardless of how large skip_top_n grows — only ranks 2+
          escalate. Confirmed live 2026-08-07 (qid=e63f03af): a capex/opex
          question's #1-ranked example was the objectively correct po_agg/
          pr_agg CTE pattern for the question, shown on attempt 1 and ignored
          by the coder model — then ACTIVELY REMOVED by this same drop logic
          on attempts 2 and 3 (skip_top_n escalating to 1, then 2), forcing
          the model toward progressively worse alternatives instead of the
          one worked example most likely to fix the next attempt. The
          escalation mechanism's original motivating case above (a
          contaminating example at rank #2+) is unaffected by this
          invariant; it only stops the mechanism from discarding a #1 result
          that was correct all along.
        """
        skip_top_n = max(skip_top_n, 1 if skip_top_match else 0)
        if self.embeddings is None or not self.examples:
            return []

        # ── Stage 0: intent pre-filter ────────────────────────────────────────
        query_intent = self._classify_intent(query)
        if query_intent in ('single_value', 'ranked_list'):
            # Hard filter: only allow same-intent seeds plus aggregate/filter
            # which are neutral and useful for both
            eligible = np.array([
                i for i, ex in enumerate(self.examples)
                if self._seed_intents[i] in (query_intent, 'aggregate', 'filter', 'general')
            ])
        else:
            eligible = np.arange(len(self.examples))

        if len(eligible) == 0:
            eligible = np.arange(len(self.examples))

        # ── Stage 1: domain-tagged embedding retrieval ────────────────────────
        query_tag = self._query_tag(query)
        tagged_query = f"{query_tag} {query}" if query_tag else query
        q_emb = np.array(self.embed_model.get_text_embedding(tagged_query))
        q_norm = np.linalg.norm(q_emb)

        sub_emb   = self.embeddings[eligible]
        sub_norms = np.linalg.norm(sub_emb, axis=1) * q_norm
        sub_norms = np.where(sub_norms == 0, 1, sub_norms)
        sub_scores = np.dot(sub_emb, q_emb) / sub_norms

        # Domain boost/penalty within the eligible subset
        if query_tag:
            for j, orig_i in enumerate(eligible):
                if self._seed_tags[orig_i] == query_tag:
                    sub_scores[j] = min(sub_scores[j] * 1.08, 1.0)
                elif self._seed_tags[orig_i] and self._seed_tags[orig_i] != query_tag:
                    sub_scores[j] = sub_scores[j] * 0.88

        wide_k = min(top_k * self.RERANK_FACTOR, len(eligible))
        top_j   = np.argsort(sub_scores)[::-1][:wide_k]
        candidates = [
            (int(eligible[j]), float(sub_scores[j]))
            for j in top_j if sub_scores[j] >= 0.25
        ]

        if not candidates:
            return []

        # ── Stage 2: reranking ────────────────────────────────────────────────
        # Skip reranking entirely when stage 1 already has a high-confidence
        # top candidate — applies to BOTH the CrossEncoder and the augmented-
        # embedding fallback (not just the fallback, as this used to read).
        # Confirmed live 2026-08-11: BAAI/bge-reranker-base's scores cluster
        # in a narrow ~0.94-0.997 band for this topically-narrow audit seed
        # pool, and within that band it is NOT a reliable relevance signal —
        # it ranked an unrelated "how many audit categories" example above a
        # near-verbatim phrasing match (stage-1 cosine 0.9069, the highest
        # score in the whole index for that query) for "list all audits —
        # plans, schedules, conducted, and completed combined", which
        # suppressed the one seed example carrying a required NOT EXISTS
        # dedup guard from ever reaching the SQL-gen model. Trusting a
        # stage-1 score this high (same 0.85 bar already used below) avoids
        # a second, noisier scoring pass overriding an already-clear match.
        top_score = candidates[0][1] if candidates else 0.0
        if top_score >= 0.85:
            reranked = candidates
        elif self._cross_encoder is not None:
            pairs = [
                (query, self.examples[i]['nl'] if isinstance(self.examples[i], dict) else self.examples[i][0])
                for i, _ in candidates
            ]
            ce_scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
            reranked = sorted(
                zip([i for i, _ in candidates], ce_scores),
                key=lambda x: x[1], reverse=True
            )
        else:
            aug_query = f"{query} | context: inspection management facility mitigative action capex opex"
            q_aug_emb = np.array(self.embed_model.get_text_embedding(aug_query))
            q_aug_norm = np.linalg.norm(q_aug_emb)

            reranked = []
            for i, s1 in candidates:
                aug_emb  = self._augmented_embeddings[i]
                aug_norm = np.linalg.norm(aug_emb) * q_aug_norm
                s2       = float(np.dot(aug_emb, q_aug_emb) / max(aug_norm, 1e-9))
                combined = (1.0 - self.ALPHA) * s1 + self.ALPHA * s2
                reranked.append((i, combined))
            reranked.sort(key=lambda x: x[1], reverse=True)

        # ── Stage 3: NL-prefix deduplication ─────────────────────────────────
        # Fetch skip_top_n extra so dropping them below still leaves top_k.
        _fetch_k = top_k + skip_top_n
        seen: set = set()
        deduped   = []
        for i, score in reranked:
            ex = self.examples[i]
            nl = ex['nl'] if isinstance(ex, dict) else ex[0]
            fp = self._nl_fingerprint(nl)
            if fp not in seen:
                seen.add(fp)
                deduped.append(ex)
            if len(deduped) >= _fetch_k:
                break
        if skip_top_n and len(deduped) > 1:
            # Keep rank #1 fixed, escalate through ranks 2+ only — see the
            # INVARIANT note on skip_top_n above. Same final length as the
            # old deduped[skip_top_n:] slice for ranks 2+, just never
            # sacrifices the top-ranked example to do it.
            deduped = [deduped[0]] + deduped[1 + skip_top_n:]
        return deduped

    @classmethod
    def _classify_intent(cls, query: str) -> str:
        """Classify query intent for pre-filtering seed candidates."""
        import re as _re
        ql = query.lower()
        # single_value: "which <1-3 word subject> has/is/generates..."
        # Handles "which fm manager has", "which facility is", "which inspector has"
        if _re.search(r'\bwhich\s+\w+(?:\s+\w+){0,2}\s+(has|is|generates|incurred|costs|performs|performed)\b', ql):
            return 'single_value'
        if _re.search(r'\bwho\s+(is|has)\s+(the\s+)?(top\s+1\b|best\b|highest\b)', ql):
            return 'single_value'
        # ranked_list: "top N", "best N"
        if _re.search(r'\btop\s+\d+\b|\bbest\s+\d+\b', ql):
            return 'ranked_list'
        # aggregate: how many, what is the total/average
        if _re.search(r'\bhow\s+(many|much)\b|\bwhat\s+is\s+the\s+(total|average|avg|percentage)\b', ql):
            return 'aggregate'
        # trend: month by month, quarterly, over time
        if _re.search(r'\bmonth\s+by\s+month\b|\bquarter\b|\btrend\b|\bover\s+time\b', ql):
            return 'trend'
        return 'general'

    _PRIMARY_HEADING = (
        "#### PRIMARY EXAMPLE — closest matching pattern for this question\n"
        "Reuse this example's join path, CTE structure, and filter shape. "
        "Change only what the question actually requires (columns, "
        "literals, grouping, LIMIT)."
    )
    _SECONDARY_HEADING = (
        "#### OTHER REFERENCE EXAMPLES — style/pattern reference only"
    )

    @staticmethod
    def _render_example(ex) -> str:
        if isinstance(ex, dict):
            nl, sql = ex['nl'], ex['sql']
        else:
            nl, sql = ex[0], ex[1]
        return f"-- {nl}\n{sql}"

    @classmethod
    def _example_key(cls, ex) -> str:
        """
        Stable identifier for a seed example, derived (not stored) so no
        seed_data.py edit is required — f"{domain}:{sha1(nl)[:8]}". Used to
        report WHICH example was anchored as PRIMARY for a given
        generate_sql call (see generation.py's SQL-GEN PLANNER CONTRACT
        debug block / _diagnostics["primary_example_key"]), so a failure
        can be attributed to "generator ignored a correct example" vs.
        "generator followed a wrong one" without needing the reasoning
        model to choose an example id itself (see the generation-quality
        plan's Item 3 for why a planner-chosen id doesn't work: retrieval
        runs inside generate_sql, after the planner has already committed
        to its tool call — it can't see the pool).
        """
        import hashlib
        if isinstance(ex, dict):
            nl, domain = ex['nl'], ex.get('domain', '')
        else:
            nl, domain = ex[0], ''
        digest = hashlib.sha1(nl.encode('utf-8')).hexdigest()[:8]
        return f"{domain or '?'}:{digest}"

    def _format_examples(self, examples: list) -> str:
        """Shared rendering logic behind format_for_prompt() and
        retrieve_with_primary_key() — one PRIMARY heading for rank #1, one
        OTHER REFERENCE heading for the rest, kept in sync by construction
        rather than by two independent implementations."""
        if not examples:
            return "-- No specific examples matched this query."
        parts = [self._PRIMARY_HEADING, self._render_example(examples[0])]
        if len(examples) > 1:
            parts.append(self._SECONDARY_HEADING)
            parts.extend(self._render_example(ex) for ex in examples[1:])
        return "\n\n".join(parts)

    def format_for_prompt(self, query: str, top_k: int = 4, skip_top_match: bool = False,
                          skip_top_n: int = 0) -> str:
        """
        Retrieve top examples and format them as a SQL comment block
        suitable for injection into the SQL generation prompt.
        Handles both legacy tuple format and new dict format.

        The first (rank #1) example is marked PRIMARY under its own
        heading — confirmed live 2026-08-07 (qid=e63f03af): a correct
        rank-#1 example rendered identically to ranks #2-4 (all uniform
        "-- <nl>\\n<sql>" blocks, no rank markup at all) was ignored by the
        coder model, which reverted to a direct po_details/pr_details join
        instead. retrieve()'s skip_top_n fix already guarantees index 0 is
        always the protected top-ranked example, on the first attempt and
        every retry — this just makes that fact visible in the rendered
        prompt instead of leaving it an internal-only property.
        """
        examples = self.retrieve(query, top_k=top_k, skip_top_match=skip_top_match,
                                  skip_top_n=skip_top_n)
        return self._format_examples(examples)

    def retrieve_with_primary_key(self, query: str, top_k: int = 4,
                                   skip_top_match: bool = False,
                                   skip_top_n: int = 0) -> tuple[str, str]:
        """
        Same rendering as format_for_prompt(), plus the anchored PRIMARY
        example's stable key — a new method rather than changing
        format_for_prompt()'s return type, so every existing caller/test
        of format_for_prompt() is untouched. primary_key is "" when
        nothing matched (text still carries the usual "no match" comment).
        """
        examples = self.retrieve(query, top_k=top_k, skip_top_match=skip_top_match,
                                  skip_top_n=skip_top_n)
        text = self._format_examples(examples)
        primary_key = self._example_key(examples[0]) if examples else ""
        return text, primary_key
