"""
Semantic Question Search
=========================
Pulls all question labels from ai_question + form/module names from
fb_forms/fb_modules at startup, embeds them, and enables meaning-based search.

"forms about age"  → finds "How old are you?" (no keyword match needed)
"safety audit"     → finds a form named "Safety Audit 2024"
"""

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
    At startup: pulls question labels from ai_question + form/module names,
    embeds them all. At query time: embeds the search term, finds closest matches.
    """

    def __init__(self, db_engine, embed_model, top_k: int = 10):
        self.embed_model = embed_model
        self.top_k = top_k
        self.labels = []
        self.embeddings = None
        self._db_engine = db_engine
        self._built_at: Optional[datetime] = None
        self._build_index(db_engine)

    def _build_index(self, db_engine):
        print("  Building semantic index from ai_questions...")

        # Question labels — the main source now
        questions_sql = """
            SELECT
                COALESCE(aq.module_name, '')       AS form_name,
                aq.label                           AS label_text,
                'QUESTION'                         AS entity_type,
                COALESCE(aq.element_id::text, '')  AS element_id
            FROM ai_questions aq
            WHERE aq.label IS NOT NULL
              AND aq.label != ''
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

        texts = []
        for row in rows_all:
            form_name, label_text, entity_type, element_id = row
            self.labels.append({
                "text": label_text,
                "entity_type": entity_type,
                "element_id": element_id or "",
                "form_name": form_name or "",
            })
            texts.append(label_text)

        self.embeddings = np.array([
            self.embed_model.get_text_embedding(t) for t in texts
        ])
        self._built_at = datetime.now()

        type_counts = {}
        for lbl in self.labels:
            type_counts[lbl["entity_type"]] = type_counts.get(lbl["entity_type"], 0) + 1
        print(
            f"  ✓ Semantic index: {len(self.labels)} labels "
            f"({', '.join(f'{k}={v}' for k, v in sorted(type_counts.items()))})"
        )

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
