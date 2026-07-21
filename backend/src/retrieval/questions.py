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
        self._penalty_weights: Optional[np.ndarray] = None
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

        # fb_forms and fb_modules intentionally excluded:
        # FORM and MODULE entity types are never retrieved at query time —
        # the LLM only uses semantic_search to find QUESTION matches.
        # Including them added startup latency, RAM overhead, and
        # diluted cosine similarity scores with irrelevant vectors.

        rows_all = []
        try:
            with db_engine.connect() as conn:
                rows = conn.execute(sql_text(questions_sql)).fetchall()
            rows_all.extend(rows)
            print(f"    ✓ {len(rows)} ai_questions labels")
        except Exception as e:
            print(f"  ⚠ Could not pull ai_questions labels: {e}")

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
            # Gives the embedding model enough signal to separate same-label questions
            # across different forms: "Risk Level [Inspection Form]" vs
            # "Risk Level [Mitigative Action Entry]".
            enriched = f"{label_text} [{form_name}]" if form_name else label_text
            texts.append(enriched)

        self.embeddings = np.array(
            self.embed_model.get_text_embedding_batch(texts)
        )
        self._built_at = datetime.now()

        # Pre-compute per-label penalty weights so search() can apply them
        # via a single numpy multiply instead of a Python for loop.
        if self._form_question_counts:
            counts = sorted(self._form_question_counts.values())
            median_count = counts[len(counts) // 2] if counts else 1
            weights = []
            for lbl in self.labels:
                if (lbl["entity_type"] == "QUESTION" and lbl["form_name"]
                        and median_count > 0):
                    fc = self._form_question_counts.get(lbl["form_name"], 1)
                    w = (1.0 / (1.0 + math.log(fc / median_count))
                         if fc > median_count else 1.0)
                else:
                    w = 1.0
                weights.append(w)
            self._penalty_weights = np.array(weights, dtype=np.float32)
        else:
            self._penalty_weights = np.ones(len(self.labels), dtype=np.float32)

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
        self._penalty_weights = None
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

        # Form-weight normalization: pre-computed in _build_index(), applied in one op.
        if self._penalty_weights is not None:
            scores *= self._penalty_weights

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
