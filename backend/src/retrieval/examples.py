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

    def __init__(self, embed_model, examples: list):
        self.embed_model = embed_model
        self.examples = list(examples)
        self.embeddings: Optional[np.ndarray] = None
        self._augmented_embeddings: Optional[np.ndarray] = None
        self._cross_encoder = None  # set if RERANKER_MODEL env var is present

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

    @classmethod
    def _query_tag(cls, query: str) -> str:
        """Classify the user query into a domain tag using keyword signals."""
        ql = query.lower()
        for tag, keywords in cls._QUERY_DOMAIN.items():
            if any(kw in ql for kw in keywords):
                return tag
        # Fallback: if mentions scores/facilities/inspectors → inspection domain
        if any(w in ql for w in ['score','inspection','facility','inspector',
                                  'completed','submitted','average']):
            return '[inspection]'
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
        # top candidate — avoids a second embedding call for clear matches.
        top_score = candidates[0][1] if candidates else 0.0
        if self._cross_encoder is not None:
            pairs = [
                (query, self.examples[i]['nl'] if isinstance(self.examples[i], dict) else self.examples[i][0])
                for i, _ in candidates
            ]
            ce_scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
            reranked = sorted(
                zip([i for i, _ in candidates], ce_scores),
                key=lambda x: x[1], reverse=True
            )
        elif top_score >= 0.85:
            # High-confidence stage 1 result — skip aug embedding call
            reranked = candidates
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
        if skip_top_n and deduped:
            deduped = deduped[skip_top_n:]
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

    def format_for_prompt(self, query: str, top_k: int = 4, skip_top_match: bool = False,
                          skip_top_n: int = 0) -> str:
        """
        Retrieve top examples and format them as a SQL comment block
        suitable for injection into the SQL generation prompt.
        Handles both legacy tuple format and new dict format.
        """
        examples = self.retrieve(query, top_k=top_k, skip_top_match=skip_top_match,
                                  skip_top_n=skip_top_n)
        if not examples:
            return "-- No specific examples matched this query."
        parts = []
        for ex in examples:
            if isinstance(ex, dict):
                nl, sql = ex['nl'], ex['sql']
            else:
                nl, sql = ex[0], ex[1]
            parts.append(f"-- {nl}\n{sql}")
        return "\n\n".join(parts)
