"""
Semantic Question Search
=========================
Pulls all question/page/section labels from the database at startup,
embeds them, and enables meaning-based search.

"forms about age" → finds "How old are you?" (no keyword match needed)

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
    At startup: pulls all translatedText from JSONB, embeds them.
    At query time: embeds the search term, finds closest matches.
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
        """Pull all labels from JSONB and embed them."""
        print("  Building semantic question index...")

        # Pull all NAME entries for QUESTION and PAGE entities.
        # Form name is extracted from the FORM entityType entry in the same
        # translations array (no fb_forms JOIN required), with a fallback
        # to the fb_forms.name JOIN for databases that use translations_id.
        query = """
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

        try:
            with db_engine.connect() as conn:
                result = conn.execute(sql_text(query))
                rows = result.fetchall()
        except Exception as e:
            print(f"  ⚠ Could not build semantic index: {e}")
            return

        if not rows:
            print("  ⚠ No labels found in database")
            return

        # Store label metadata
        texts = []
        for row in rows:
            form_name, label_text, entity_type, element_id = row
            self.labels.append({
                "text": label_text,
                "entity_type": entity_type,
                "element_id": element_id or "",
                "form_name": form_name or "",
            })
            texts.append(label_text)

        # Embed all labels
        self.embeddings = np.array([
            self.embed_model.get_text_embedding(t) for t in texts
        ])

        self._built_at = datetime.now()
        print(f"  ✓ Semantic index built: {len(self.labels)} labels from {len(set(l['form_name'] for l in self.labels))} forms")

    def refresh_if_stale(self, max_age_minutes: int = 30) -> None:
        """
        Rebuild the embedding index if it is older than max_age_minutes.
        Logs how many labels were found vs the previous count.
        No-op if the index was built less than max_age_minutes ago.
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
        self._build_index(self._db_engine)
        new_count = len(self.labels)
        delta = new_count - prev_count
        sign = "+" if delta >= 0 else ""
        print(
            f"  [semantic] Refresh complete: {new_count} labels "
            f"({sign}{delta} vs previous {prev_count})."
        )

    def search(self, query: str, entity_type: Optional[str] = None, form_name: Optional[str] = None, top_k: Optional[int] = None) -> list[SemanticMatch]:
        """
        Find labels semantically similar to the query.
        
        Args:
            query: search text ("age", "employee details", "safety checklist")
            entity_type: filter by type ("QUESTION", "PAGE", "SECTION") or None for all
            form_name: filter by form name or None for all
            top_k: number of results (default: self.top_k)
        """
        if self.embeddings is None or len(self.labels) == 0:
            return []

        top_k = top_k or self.top_k

        # Embed the query
        query_embedding = np.array(self.embed_model.get_text_embedding(query))

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        similarities = np.dot(self.embeddings, query_embedding) / norms

        # Filter by entity_type and form_name if specified
        mask = np.ones(len(self.labels), dtype=bool)
        if entity_type:
            mask &= np.array([l["entity_type"] == entity_type for l in self.labels])
        if form_name:
            mask &= np.array([form_name.lower() in l["form_name"].lower() for l in self.labels])

        # Apply mask
        masked_scores = similarities * mask

        # Get top-k indices
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

    def search_and_build_sql(self, search_term: str, entity_type: str = "QUESTION", form_name: Optional[str] = None, threshold: float = 0.5) -> Optional[str]:
        """
        Search for labels matching the term and build a SQL query
        that uses exact elementId matches instead of ILIKE.
        
        Returns SQL or None if no good matches found.
        """
        matches = self.search(search_term, entity_type=entity_type, form_name=form_name, top_k=10)

        # Filter by threshold
        good_matches = [m for m in matches if m.score >= threshold]

        if not good_matches:
            return None

        # Build SQL using elementId IN (...) for precise matching
        element_ids = [m.element_id for m in good_matches if m.element_id]

        if not element_ids:
            return None

        id_list = ", ".join(f"'{eid}'" for eid in element_ids)

        # Build WHERE clause
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
