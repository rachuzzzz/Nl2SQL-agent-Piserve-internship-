"""
Unit tests for generation.py's _build_additional_context() — Item 2 of the
2026-08-07 generation-quality plan (qid=e63f03af). Separates the three
registers that make up "{schema_hint}" (factual schema, mandatory
requirements, editable planner draft) into labelled sub-sections instead of
one undifferentiated concatenated blob.
"""

from src.sql.generation import _build_additional_context


def test_all_three_parts_present_in_order():
    text = _build_additional_context(
        "TABLE inspection_report (...)",
        "SELECT * FROM inspection_report",
        "USE THIS EXACT SQL ... do not deviate",
    )
    assert "#### Schema reference" in text
    assert "#### Required" in text
    assert "#### Planner draft" in text
    schema_idx = text.index("#### Schema reference")
    required_idx = text.index("#### Required")
    draft_idx = text.index("#### Planner draft")
    assert schema_idx < required_idx < draft_idx


def test_empty_mandatory_hint_omits_required_heading():
    text = _build_additional_context("TABLE x", "SELECT * FROM x", "")
    assert "#### Required" not in text
    assert "#### Schema reference" in text
    assert "#### Planner draft" in text


def test_all_empty_falls_back_to_no_context_message():
    text = _build_additional_context("", "", "")
    assert text == "No additional context provided."


def test_inject_text_lands_under_required_llm_hint_under_draft():
    text = _build_additional_context(
        "", "my own guess SELECT 1", "USE THIS EXACT SQL FOR CAPEX (do not deviate)",
    )
    required_idx = text.index("#### Required")
    draft_idx = text.index("#### Planner draft")
    inject_idx = text.index("USE THIS EXACT SQL FOR CAPEX")
    hint_idx = text.index("my own guess SELECT 1")
    assert required_idx < inject_idx < draft_idx
    assert draft_idx < hint_idx


def test_planner_draft_framing_mentions_editable_and_resolved_literals():
    text = _build_additional_context("", "SELECT 1", "")
    assert "STARTING POINT" in text
    assert "resolved" in text.lower()
