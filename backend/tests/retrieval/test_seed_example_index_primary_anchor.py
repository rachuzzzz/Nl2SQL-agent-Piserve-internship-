"""
Regression tests for SeedExampleIndex.format_for_prompt()'s rank-aware
PRIMARY/OTHER sectioning — Item 1 of the 2026-08-07 generation-quality plan
(qid=e63f03af). See examples.py's format_for_prompt() docstring for the
full incident writeup.
"""

from src.retrieval.examples import SeedExampleIndex

from tests.retrieval.test_seed_example_index_skip_top_n import (
    _QUERY,
    _make_index,
)


def test_primary_heading_marks_rank_one():
    index = _make_index(5)
    text = index.format_for_prompt(_QUERY, top_k=3)
    assert text.count("PRIMARY EXAMPLE") == 1
    primary_idx = text.index("PRIMARY EXAMPLE")
    example_idx = text.index("example zero nl text")
    other_idx = text.index("OTHER REFERENCE EXAMPLES")
    # The rank-0 example sits between the PRIMARY heading and the OTHER heading.
    assert primary_idx < example_idx < other_idx


def test_primary_example_is_stable_across_skip_top_n():
    index = _make_index(5)
    for skip_top_n in (0, 1, 2, 3):
        text = index.format_for_prompt(_QUERY, top_k=3, skip_top_n=skip_top_n)
        primary_idx = text.index("PRIMARY EXAMPLE")
        other_idx = text.index("OTHER REFERENCE EXAMPLES") if "OTHER REFERENCE EXAMPLES" in text else len(text)
        primary_block = text[primary_idx:other_idx]
        assert "example zero nl text" in primary_block, (
            f"skip_top_n={skip_top_n}: primary example changed: {primary_block!r}"
        )


def test_secondary_heading_omitted_when_only_one_example():
    index = _make_index(1)
    text = index.format_for_prompt(_QUERY, top_k=4)
    assert "PRIMARY EXAMPLE" in text
    assert "OTHER REFERENCE EXAMPLES" not in text


def test_no_headings_when_nothing_matched():
    index = SeedExampleIndex(embed_model=None, examples=[])
    text = index.format_for_prompt(_QUERY)
    assert text == "-- No specific examples matched this query."
    assert "PRIMARY" not in text


def test_all_example_sql_still_present():
    index = _make_index(5)
    text = index.format_for_prompt(_QUERY, top_k=5)
    for i in range(5):
        assert f"SELECT {i}" in text
