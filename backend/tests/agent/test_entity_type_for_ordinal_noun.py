"""
Unit tests for disambiguation.py's entity_type_for_ordinal_noun() — the
helper session.py's ordinal resolver uses to check whether the noun in an
ordinal phrase ("the 6th AUDITOR") matches the entity type of the list it's
about to resolve against.
"""

from src.agent.orchestrator.disambiguation import entity_type_for_ordinal_noun


def test_auditor_noun_maps_to_person():
    assert entity_type_for_ordinal_noun("auditor from the list", "audit") == "person"


def test_inspector_noun_maps_to_person_in_inspection_domain():
    assert entity_type_for_ordinal_noun("inspector in the list", "inspection") == "person"


def test_audit_plan_noun_maps_to_plan():
    assert entity_type_for_ordinal_noun("audit plan shown above", "audit") == "plan"


def test_cycle_noun_maps_to_cycle():
    assert entity_type_for_ordinal_noun("audit cycle", "audit") == "cycle"


def test_portfolio_noun_maps_to_portfolio():
    assert entity_type_for_ordinal_noun("portfolio in the list", "audit") == "portfolio"


def test_unrecognized_noun_returns_empty():
    assert entity_type_for_ordinal_noun("one in the list", "audit") == ""


def test_empty_domain_still_finds_audit_only_noun():
    assert entity_type_for_ordinal_noun("audit plan", "") == "plan"
