"""
Regression tests for SeedExampleIndex.retrieve()'s skip_top_n drop logic —
the Bug 2 fix from the 2026-08-07 capex/opex audit (qid=e63f03af). See
examples.py's retrieve() docstring ("INVARIANT: rank #1 ... NEVER dropped")
for the full incident writeup.

No test coverage existed for SeedExampleIndex before this file.
"""

import numpy as np

from src.retrieval.examples import SeedExampleIndex


class _FakeEmbedModel:
    """Deterministic stand-in for the real bge-small embedder — returns a
    fixed, hand-picked unit vector per known input string so retrieval
    ranking is fully controllable without loading a real model."""

    def __init__(self, vectors: dict):
        self._vectors = vectors

    def get_text_embedding_batch(self, texts):
        return [self._vectors[t] for t in texts]

    def get_text_embedding(self, text):
        return self._vectors[text]


# Neutral query — deliberately avoids every keyword in
# SeedExampleIndex._QUERY_DOMAIN / the domain-fallback list, so
# _query_tag() returns '' and no domain boost/penalty or query-tag prefix
# perturbs the hand-picked cosine scores below.
_QUERY = "please summarize these five rows of numbers for me"

# Unit vectors chosen so cosine similarity to _QUERY (mapped to [1, 0]) is
# strictly descending: ex0=1.0, ex1=0.9, ex2=0.7, ex3=0.5, ex4=0.35 — all
# clear the retrieve() candidate threshold (>= 0.25).
_QV = [1.0, 0.0]
_EX_VECTORS = [
    [1.0, 0.0],
    [0.9, 0.4358898944],
    [0.7, 0.7141428429],
    [0.5, 0.8660254038],
    [0.35, 0.9367497617],
]


def _make_index(n: int) -> SeedExampleIndex:
    examples = [{"nl": f"example {w} nl text", "sql": f"SELECT {i}"}
                for i, w in enumerate(["zero", "one", "two", "three", "four"][:n])]
    vectors = {_QUERY: _QV}
    for ex, v in zip(examples, _EX_VECTORS[:n]):
        vectors[ex["nl"]] = v
    index = SeedExampleIndex(_FakeEmbedModel(vectors), examples)
    # Force the embedding-fallback rerank path deterministically, regardless
    # of whatever RERANKER_MODEL happens to be configured in this environment.
    index._cross_encoder = None
    return index


def test_top_ranked_example_never_dropped_as_skip_top_n_escalates():
    index = _make_index(5)
    for skip_top_n in (0, 1, 2, 3):
        results = index.retrieve(_QUERY, top_k=3, skip_top_n=skip_top_n)
        nls = [r["nl"] for r in results]
        assert "example zero nl text" in nls, (
            f"skip_top_n={skip_top_n} dropped the #1-ranked example: {nls}"
        )
        assert nls[0] == "example zero nl text"  # still ranked first, not just present


def test_lower_ranked_examples_still_escalate_as_skip_top_n_increases():
    """The original escalating-drop behavior (for ranks 2+) must survive —
    only rank #1 is now protected, not the whole mechanism disabled."""
    index = _make_index(5)
    at_1 = {r["nl"] for r in index.retrieve(_QUERY, top_k=3, skip_top_n=1)}
    at_2 = {r["nl"] for r in index.retrieve(_QUERY, top_k=3, skip_top_n=2)}
    assert at_1 != at_2, "lower-ranked example set should change as skip_top_n escalates"
    # Rank #1 is the one constant across both.
    assert "example zero nl text" in at_1 and "example zero nl text" in at_2


def test_few_eligible_examples_with_large_skip_top_n_does_not_crash():
    index = _make_index(2)
    results = index.retrieve(_QUERY, top_k=4, skip_top_n=3)
    nls = [r["nl"] for r in results]
    assert nls  # non-empty — did not crash into an empty/broken result
    assert nls[0] == "example zero nl text"
    assert len(results) <= 4
