"""
text_similarity — one-off cosine scoring for small candidate sets.
"""

import numpy as np


def text_similarity(embed_model, query: str, text: str) -> float:
    """
    Plain cosine similarity between `query` and `text` using `embed_model`.

    SeedExampleIndex (retrieval/examples.py) batches and caches embeddings for
    its ~170-entry corpus because it's queried on every generate_sql() call.
    This standalone version is for one-off ranking of a handful of already
    topic-gated candidates (e.g. agent/prompts.py's EXAMPLES_TIER2/
    CONDITIONAL_BLOCKS, ranked by title) where building a real index would be
    overkill — a few embed calls on short strings, only when more than one
    candidate matched.
    """
    q = np.array(embed_model.get_text_embedding(query))
    t = np.array(embed_model.get_text_embedding(text))
    denom = float(np.linalg.norm(q) * np.linalg.norm(t))
    return float(np.dot(q, t) / denom) if denom else 0.0
