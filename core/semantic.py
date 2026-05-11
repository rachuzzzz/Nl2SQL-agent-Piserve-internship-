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
    Embeds the NL question strings from seed SQL examples at startup.
    At generate_sql call time, retrieve() finds the 3-4 most similar examples
    by cosine similarity and returns them for injection into the SQL prompt.

    This replaces the previous approach of injecting all 26 static patterns
    unconditionally into every SQL generation call (~500 tokens wasted per call).
    With retrieval, each call gets ~150 tokens of precisely relevant examples.

    The 26 existing static patterns are the starting seed set (see prompts.py:
    SEED_EXAMPLES).  New patterns can be added as the system encounters them.
    """

    def __init__(self, embed_model, examples: list[tuple[str, str]]):
        """
        Parameters
        ----------
        embed_model : any model with .get_text_embedding(text) -> list[float]
        examples    : list of (nl_question, sql_snippet) pairs
        """
        self.embed_model = embed_model
        self.examples = list(examples)
        self.embeddings: Optional[np.ndarray] = None

        if self.examples:
            print(f"  Building seed example index ({len(self.examples)} examples)...")
            texts = [ex[0] for ex in self.examples]
            self.embeddings = np.array([
                embed_model.get_text_embedding(t) for t in texts
            ])
            print(f"  ✓ Seed example index ready")

    def retrieve(self, query: str, top_k: int = 4) -> list[tuple[str, str]]:
        """
        Return the top_k (nl_question, sql_snippet) pairs most similar to query.
        Returns fewer than top_k if similarity scores are low (threshold: 0.30).
        """
        if self.embeddings is None or not self.examples:
            return []

        q_emb = np.array(self.embed_model.get_text_embedding(query))
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
        norms = np.where(norms == 0, 1, norms)
        scores = np.dot(self.embeddings, q_emb) / norms

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            self.examples[i]
            for i in top_idx
            if scores[i] >= 0.30
        ]

    def format_for_prompt(self, query: str, top_k: int = 4) -> str:
        """
        Retrieve top examples and format them as a SQL comment block
        suitable for injection into the SQL generation prompt.
        """
        examples = self.retrieve(query, top_k=top_k)
        if not examples:
            return "-- No specific examples matched this query."
        parts = []
        for nl, sql_snippet in examples:
            parts.append(f"-- {nl}\n{sql_snippet}")
        return "\n\n".join(parts)