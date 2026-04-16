"""
Semantic Question Search
=========================
Pulls all question/page/section labels — plus form and module names — from
the database at startup, embeds them, and enables meaning-based search.

"forms about age" → finds "How old are you?" (no keyword match needed)
"safety audit"    → finds a form named "Safety Audit 2024"
"HR"              → finds a module named "Human Resources"

Uses the same bge-small embedding model already loaded for table retrieval.
Runs on CPU, ~50MB extra RAM for a few thousand labels.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, text as sql_text


@dataclass
class SemanticMatch:
    """A question/element that semantically matches the search."""
    text: str
    entity_type: str
    element_id: str
    form_name: str
    score: float


class SemanticQuestionIndex:
    """
    At startup: pulls all translatedText from JSONB + form names + module names,
    embeds them. At query time: embeds the search term, finds closest matches.

    Indexed entity_type values:
      - QUESTION  (from fb_translation_json — a question label)
      - PAGE      (from fb_translation_json — a page label)
      - FORM      (from fb_forms.name — a form's own name)
      - MODULE    (from fb_modules.name — a module's own name)
    """

    def __init__(self, db_engine, embed_model, top_k: int = 10):
        self.embed_model = embed_model
        self.top_k = top_k
        self.labels = []       # list of dicts: {text, entity_type, element_id, form_name}
        self.embeddings = None  # numpy array of shape (n, embed_dim)
        self._db_engine = db_engine  # kept for refresh
        self._built_at: Optional[datetime] = None

        self._build_index(db_engine)

    def _build_index(self, db_engine):
        """Pull all labels from JSONB + form/module names, then embed them."""
        print("  Building semantic question index...")

        # --- 1. QUESTION and PAGE labels from JSONB translations ---
        jsonb_query = """
        SELECT
            COALESCE(
                (SELECT elem2->>'translatedText'
                 FROM jsonb_array_elements(tj.translations) AS elem2
                 WHERE elem2->>'entityType' = 'FORM'
                   AND elem2->>'language' = 'eng'
                   AND elem2->>'attribute' = 'NAME'
                 LIMIT 1),
                (SELECT f.name FROM fb_forms f
                 WHERE f.translations_id = tj.id LIMIT 1),
                ''
            ) AS form_name,
            elem->>'translatedText' AS label_text,
            elem->>'entityType'     AS entity_type,
            elem->>'elementId'      AS element_id
        FROM fb_translation_json tj,
             jsonb_array_elements(tj.translations) AS elem
        WHERE elem->>'language'       = 'eng'
          AND elem->>'attribute'      = 'NAME'
          AND elem->>'entityType'     IN ('QUESTION', 'PAGE')
          AND elem->>'translatedText' IS NOT NULL
          AND elem->>'translatedText' != ''
        """

        # --- 2. FORM names from fb_forms ---
        forms_query = """
        SELECT f.name AS form_name,
               f.name AS label_text,
               'FORM' AS entity_type,
               f.id::text AS element_id
        FROM fb_forms f
        WHERE f.name IS NOT NULL AND f.name != ''
        """

        # --- 3. MODULE names from fb_modules ---
        modules_query = """
        SELECT '' AS form_name,
               m.name AS label_text,
               'MODULE' AS entity_type,
               m.id::text AS element_id
        FROM fb_modules m
        WHERE m.name IS NOT NULL AND m.name != ''
        """

        rows_all = []
        try:
            with db_engine.connect() as conn:
                rows_all.extend(conn.execute(sql_text(jsonb_query)).fetchall())
        except Exception as e:
            print(f"  ⚠ Could not pull JSONB labels: {e}")

        try:
            with db_engine.connect() as conn:
                rows_all.extend(conn.execute(sql_text(forms_query)).fetchall())
        except Exception as e:
            print(f"  ⚠ Could not pull form names: {e}")

        try:
            with db_engine.connect() as conn:
                rows_all.extend(conn.execute(sql_text(modules_query)).fetchall())
        except Exception as e:
            print(f"  ⚠ Could not pull module names: {e}")

        if not rows_all:
            print("  ⚠ No labels found in database")
            return

        # Store label metadata
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

        # Embed all labels in one batch
        self.embeddings = np.array([
            self.embed_model.get_text_embedding(t) for t in texts
        ])

        self._built_at = datetime.now()
        type_counts = {}
        for lbl in self.labels:
            type_counts[lbl["entity_type"]] = type_counts.get(lbl["entity_type"], 0) + 1
        print(
            f"  ✓ Semantic index built: {len(self.labels)} labels "
            f"({', '.join(f'{k}={v}' for k, v in sorted(type_counts.items()))})"
        )

    def refresh_if_stale(self, max_age_minutes: int = 30) -> None:
        """
        Rebuild the embedding index if older than max_age_minutes.
        No-op if recent.
        """
        if self._built_at is None:
            print("  [semantic] Index not yet built — building now.")
            self._build_index(self._db_engine)
            return

        age_minutes = (datetime.now() - self._built_at).total_seconds() / 60
        if age_minutes < max_age_minutes:
            print(
                f"  [semantic] Index is fresh ({age_minutes:.1f} min old, "
                f"threshold {max_age_minutes} min) — skipping refresh."
            )
            return

        prev_count = len(self.labels)
        print(
            f"  [semantic] Index is stale ({age_minutes:.1f} min old) — refreshing..."
        )
        # Reset state before rebuild to avoid appending duplicates
        self.labels = []
        self.embeddings = None
        self._build_index(self._db_engine)
        new_count = len(self.labels)
        delta = new_count - prev_count
        sign = "+" if delta >= 0 else ""
        print(
            f"  [semantic] Refresh complete: {new_count} labels "
            f"({sign}{delta} vs previous {prev_count})."
        )

    def search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        form_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[SemanticMatch]:
        """
        Find labels semantically similar to the query.

        Args:
            query:       search text.
            entity_type: filter by type ("QUESTION", "PAGE", "FORM", "MODULE")
                         or None for all.
            form_name:   filter by form name containing this substring, or None.
                         (Ignored when entity_type in {"FORM","MODULE"} because
                         those entries have no enclosing form.)
            top_k:       number of results (default self.top_k).
        """
        if self.embeddings is None or len(self.labels) == 0:
            return []

        top_k = top_k or self.top_k

        # Embed the query
        query_embedding = np.array(self.embed_model.get_text_embedding(query))

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        norms = np.where(norms == 0, 1, norms)
        similarities = np.dot(self.embeddings, query_embedding) / norms

        # Filter
        mask = np.ones(len(self.labels), dtype=bool)
        if entity_type:
            mask &= np.array([l["entity_type"] == entity_type for l in self.labels])
        if form_name and entity_type not in ("FORM", "MODULE"):
            mask &= np.array([form_name.lower() in l["form_name"].lower() for l in self.labels])

        masked_scores = similarities * mask

        top_indices = np.argsort(masked_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if masked_scores[idx] <= 0:
                break
            label = self.labels[idx]
            results.append(SemanticMatch(
                text=label["text"],
                entity_type=label["entity_type"],
                element_id=label["element_id"],
                form_name=label["form_name"],
                score=float(similarities[idx]),
            ))

        return results

    def search_and_build_sql(
        self,
        search_term: str,
        entity_type: str = "QUESTION",
        form_name: Optional[str] = None,
        threshold: float = 0.5,
    ) -> Optional[str]:
        """
        Search for labels matching the term and build a SQL query
        that uses exact elementId matches instead of ILIKE.

        Returns SQL or None if no good matches found.
        """
        matches = self.search(
            search_term, entity_type=entity_type, form_name=form_name, top_k=10
        )
        good_matches = [m for m in matches if m.score >= threshold]

        if not good_matches:
            return None

        element_ids = [m.element_id for m in good_matches if m.element_id]
        if not element_ids:
            return None

        id_list = ", ".join(f"'{eid}'" for eid in element_ids)

        where_parts = [
            f"elem->>'language' = 'eng'",
            f"elem->>'attribute' = 'NAME'",
            f"elem->>'entityType' = '{entity_type}'",
            f"elem->>'elementId' IN ({id_list})",
        ]
        if form_name:
            where_parts.insert(0, f"f.name ILIKE '%{form_name}%'")

        sql = (
            f"SELECT f.name AS form_name,\n"
            f"       elem->>'translatedText' AS label,\n"
            f"       elem->>'elementId' AS element_id\n"
            f"FROM fb_forms f\n"
            f"JOIN fb_translation_json tj ON f.translations_id = tj.id,\n"
            f"     jsonb_array_elements(tj.translations) AS elem\n"
            f"WHERE {chr(10) + '  AND '.join(where_parts)}\n"
            f"LIMIT 100;"
        )
        return sql