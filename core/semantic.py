"""
Semantic Question Search
=========================
Pulls all question labels from ai_questions + form/module names from
fb_forms/fb_modules at startup, embeds them, and enables meaning-based search.

"forms about age"  → finds "How old are you?" (no keyword match needed)
"safety audit"     → finds a form named "Safety Audit 2024"

Problem 1 fix — domain-aware index:
  Labels are embedded as contextually enriched strings rather than bare text.
  "Risk Level" in "Inspection Form" → "Risk Level [Inspection Form]"
  "Risk Level" in "Statutory FLS Audit" → "Risk Level [Statutory FLS Audit]"
  This gives bge-small-en-v1.5 enough signal to distinguish same-label questions
  across different workflow contexts.

  Form-weight normalization is applied at search time: over-represented forms
  (e.g. "Statutory FLS Audit" with 738 questions = 31% of index) receive a
  dampened inverse-frequency penalty so they cannot flood every result set.

Problem 4 fix — SeedExampleIndex:
  Embeds NL question strings from seed SQL examples at startup.
  generate_sql() calls retrieve() to inject the 3-4 most relevant examples
  instead of all 26 static patterns unconditionally.
"""

import math
import os
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from sqlalchemy import text as sql_text


@dataclass
class SemanticMatch:
    text: str
    entity_type: str   # "QUESTION" | "FORM" | "MODULE"
    element_id: str
    form_name: str
    score: float


class SemanticQuestionIndex:
    """
    At startup: pulls question labels from ai_questions + form/module names,
    embeds them with contextual enrichment. At query time: embeds the search
    term, finds closest matches, applies form-weight normalization.
    """

    def __init__(self, db_engine, embed_model, top_k: int = 10):
        self.embed_model = embed_model
        self.top_k = top_k
        self.labels = []
        self.embeddings = None
        self._db_engine = db_engine
        self._built_at: Optional[datetime] = None
        # form_name → question count, used for weight normalization
        self._form_question_counts: dict[str, int] = {}
        self._build_index(db_engine)

    def _build_index(self, db_engine):
        print("  Building semantic index from ai_questions...")

        # Modules that exist in ai_questions but have ZERO rows in ai_answers.
        # Including them pollutes every semantic search — they account for ~65% of
        # ai_questions but are never answered in this system (audit domain, test data).
        # Only index modules that actually have answer data.
        EXCLUDED_MODULES = (
            # Audit domain — out of scope, zero ai_answers rows
            "Statutory FLS Audit",
            "QHSE Internal audit",
            "Statutory - Specialist",
            "External Specialist",
            "QHSE Internal audit FTZ...",
            "external-stores",
            "FM Audit",
            "Internal Accomodation",
            "Internal-Subcontractors",
            "Quality Audit",
            "Second Party - Sustainability",
            "Audit Corrective Action Entry",
            "Audit Corrective Action Close With Deferred",
            "Audit Corrective Action Completion",
            "Audit Corrective Action Progress Tracking",
            "Audit Corrective Action Adequacy Approval",
            "Audit Corrective Action Adequacy Rejection",
            "Audit Corrective Action Implementation Approval",
            "Audit Corrective Action Implementation Rejection",
            "Audit Plan Approval",
            "Audit Plan Rejection",
            "Audit Plan Remarks",
            "Audit Portfolio Approval",
            "Audit Portfolio Rejection",
            "Audit Report Closeout",
            "Audit Report Remarks Entry",
            "Audit Reschedule Request",
            "Audit Schedule Approval",
            "Audit Schedule Rejection",
            # Test data
            "Test cat test type",
        )
        excluded_placeholders = ",".join(f"'{m}'" for m in EXCLUDED_MODULES)

        # Question labels — the main source.
        # module_name is fetched so we can construct enriched label text.
        questions_sql = f"""
            SELECT
                COALESCE(aq.module_name, '')       AS form_name,
                aq.label                           AS label_text,
                'QUESTION'                         AS entity_type,
                COALESCE(aq.element_id::text, '')  AS element_id
            FROM ai_questions aq
            WHERE aq.label IS NOT NULL
              AND aq.label != ''
              AND (aq.module_name IS NULL OR aq.module_name NOT IN ({excluded_placeholders}))
        """

        # Form names for semantic form lookup
        forms_sql = """
            SELECT f.name   AS form_name,
                   f.name   AS label_text,
                   'FORM'   AS entity_type,
                   f.id::text AS element_id
            FROM fb_forms f
            WHERE f.name IS NOT NULL AND f.name != ''
        """

        # Module names
        modules_sql = """
            SELECT ''       AS form_name,
                   m.name   AS label_text,
                   'MODULE' AS entity_type,
                   m.id::text AS element_id
            FROM fb_modules m
            WHERE m.name IS NOT NULL AND m.name != ''
        """

        rows_all = []
        for label, query in [
            ("ai_questions labels", questions_sql),
            ("form names",          forms_sql),
            ("module names",        modules_sql),
        ]:
            try:
                with db_engine.connect() as conn:
                    rows = conn.execute(sql_text(query)).fetchall()
                rows_all.extend(rows)
                print(f"    ✓ {len(rows)} {label}")
            except Exception as e:
                print(f"  ⚠ Could not pull {label}: {e}")

        if not rows_all:
            print("  ⚠ No labels found — semantic search unavailable")
            return

        # ------------------------------------------------------------------ #
        # Build per-form question counts for weight normalization (Problem 1) #
        # ------------------------------------------------------------------ #
        self._form_question_counts = {}
        for row in rows_all:
            form_name, label_text, entity_type, element_id = row
            if entity_type == "QUESTION" and form_name:
                self._form_question_counts[form_name] = (
                    self._form_question_counts.get(form_name, 0) + 1
                )

        # ------------------------------------------------------------------ #
        # Build labels list and enriched text strings (Problem 1)            #
        # ------------------------------------------------------------------ #
        texts = []
        for row in rows_all:
            form_name, label_text, entity_type, element_id = row
            self.labels.append({
                "text": label_text,
                "entity_type": entity_type,
                "element_id": element_id or "",
                "form_name": form_name or "",
            })

            # Contextual enrichment: embed "[label] [form_name]" instead of bare label.
            # This gives the embedding model enough signal to separate:
            #   "Risk Level [Inspection Form]" vs "Risk Level [Statutory FLS Audit]"
            # Forms/modules are embedded as-is (no enrichment needed).
            if entity_type == "QUESTION" and form_name:
                enriched = f"{label_text} [{form_name}]"
            else:
                enriched = label_text
            texts.append(enriched)

        self.embeddings = np.array([
            self.embed_model.get_text_embedding(t) for t in texts
        ])
        self._built_at = datetime.now()

        type_counts = {}
        for lbl in self.labels:
            type_counts[lbl["entity_type"]] = type_counts.get(lbl["entity_type"], 0) + 1

        # Report largest forms so over-representation is visible in logs
        top_forms = sorted(
            self._form_question_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        top_form_str = ", ".join(f"{n}={c}" for n, c in top_forms)
        print(
            f"  ✓ Semantic index: {len(self.labels)} labels "
            f"({', '.join(f'{k}={v}' for k, v in sorted(type_counts.items()))})"
        )
        if top_forms:
            print(f"    Top forms by question count: {top_form_str}")

    def refresh_if_stale(self, max_age_minutes: int = 30) -> None:
        if self._built_at is None:
            self._build_index(self._db_engine)
            return
        age = (datetime.now() - self._built_at).total_seconds() / 60
        if age < max_age_minutes:
            return
        prev = len(self.labels)
        self.labels = []
        self.embeddings = None
        self._form_question_counts = {}
        self._build_index(self._db_engine)
        delta = len(self.labels) - prev
        sign = "+" if delta >= 0 else ""
        print(f"  [semantic] Refreshed: {len(self.labels)} labels ({sign}{delta})")

    def search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        form_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[SemanticMatch]:
        if self.embeddings is None or not self.labels:
            return []

        top_k = top_k or self.top_k
        q_emb = np.array(self.embed_model.get_text_embedding(query))
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
        norms = np.where(norms == 0, 1, norms)
        scores = np.dot(self.embeddings, q_emb) / norms

        # ------------------------------------------------------------------ #
        # Form-weight normalization (Problem 1)                               #
        # ------------------------------------------------------------------ #
        # Apply a dampened inverse-frequency penalty to questions from
        # over-represented forms.  A form with 738 questions should not occupy
        # 7 of the top 10 slots for an unrelated query.
        #
        # Penalty formula:
        #   if form_count > median_count:
        #     weight = 1 / (1 + log(form_count / median_count))
        #   else:
        #     weight = 1.0  (no penalty for average-or-below-size forms)
        #
        # This dampens large forms without eliminating them — a genuinely
        # relevant result from a large form will still rank highly.
        if self._form_question_counts:
            counts = sorted(self._form_question_counts.values())
            median_count = counts[len(counts) // 2] if counts else 1
            for i, lbl in enumerate(self.labels):
                if lbl["entity_type"] == "QUESTION" and lbl["form_name"]:
                    form_count = self._form_question_counts.get(lbl["form_name"], 1)
                    if form_count > median_count and median_count > 0:
                        penalty = 1.0 / (1.0 + math.log(form_count / median_count))
                        scores[i] *= penalty

        mask = np.ones(len(self.labels), dtype=bool)
        if entity_type:
            mask &= np.array([l["entity_type"] == entity_type for l in self.labels])
        if form_name and entity_type not in ("FORM", "MODULE"):
            mask &= np.array([form_name.lower() in l["form_name"].lower() for l in self.labels])

        masked = scores * mask
        top_idx = np.argsort(masked)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if masked[idx] <= 0:
                break
            lbl = self.labels[idx]
            results.append(SemanticMatch(
                text=lbl["text"],
                entity_type=lbl["entity_type"],
                element_id=lbl["element_id"],
                form_name=lbl["form_name"],
                score=float(scores[idx]),
            ))
        return results


# ============================================================================
# SeedExampleIndex — Problem 4: retrieved example store
# ============================================================================

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

            self.embeddings = np.array([
                embed_model.get_text_embedding(t) for t in nl_texts
            ])

            # Stage 2 fallback embeddings: NL + SQL table tokens
            aug_texts = [
                self._augment(
                    ex['nl']  if isinstance(ex, dict) else ex[0],
                    ex['sql'] if isinstance(ex, dict) else ex[1],
                )
                for ex in self.examples
            ]
            self._augmented_embeddings = np.array([
                embed_model.get_text_embedding(t) for t in aug_texts
            ])

            # Keep legacy _seed_tags for domain boost in retrieve()
            self._seed_tags = [f'[{d}]' if d else '' for d in self._seed_domains]

            # Try loading CrossEncoder if RERANKER_MODEL is configured
            reranker_model = os.environ.get("RERANKER_MODEL", "").strip()
            if reranker_model:
                try:
                    from sentence_transformers import CrossEncoder
                    device = os.environ.get("RERANKER_DEVICE", "cpu")
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
        '[corrective_action]': ['corrective action','overdue action','cause',
                                'deferred','responsible party'],
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

    def retrieve(self, query: str, top_k: int = 4) -> list:
        """
        Three-stage retrieval + reranking with intent pre-filtering:
          Stage 0: intent pre-filter — 'which single X' queries only see
                   single_value seeds; 'top N X' only sees ranked_list seeds.
                   Prevents FM-02-style loops where the top-5 seed beats LIMIT 1.
          Stage 1: domain-tagged cosine similarity → top_k * RERANK_FACTOR candidates
          Stage 2: rerank — CrossEncoder if available, else augmented embedding fallback
          Stage 3: deduplicate by 4-word NL fingerprint → return top_k
        """
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
        else:
            aug_query = f"{query} | context: inspection management facility corrective action capex opex"
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
        seen: set = set()
        deduped   = []
        for i, score in reranked:
            ex = self.examples[i]
            nl = ex['nl'] if isinstance(ex, dict) else ex[0]
            fp = self._nl_fingerprint(nl)
            if fp not in seen:
                seen.add(fp)
                deduped.append(ex)
            if len(deduped) >= top_k:
                break
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

    def format_for_prompt(self, query: str, top_k: int = 4) -> str:
        """
        Retrieve top examples and format them as a SQL comment block
        suitable for injection into the SQL generation prompt.
        Handles both legacy tuple format and new dict format.
        """
        examples = self.retrieve(query, top_k=top_k)
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