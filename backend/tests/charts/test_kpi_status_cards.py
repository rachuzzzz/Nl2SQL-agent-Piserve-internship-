"""
Tests for the status-breakdown KPI card path (charts/advisor/kpi.py's
_status_breakdown_cards, wired into build_kpi_cards). Confirms a "corrective
actions overdue vs open vs closed"-shaped GROUP BY result gets one card per
status (severity-ordered, most-urgent-first) instead of one meaningless
"Total X" sum card, for BOTH domains' own status vocabulary — and that
every other existing card shape (score/pct avg+range, plain sum, distinct
count) is completely unaffected.
"""

from src.charts.advisor.kpi import build_kpi_cards


def test_audit_corrective_action_status_breakdown_gets_per_status_cards():
    rows = [
        {"status": "OVERDUE", "cnt": 5},
        {"status": "OPEN", "cnt": 3},
        {"status": "CLOSED", "cnt": 12},
    ]
    cards = build_kpi_cards(["status", "cnt"], rows)
    assert len(cards) == 3
    labels = [c["label"] for c in cards]
    assert labels == ["Overdue", "Open", "Closed"]  # severity order: critical, warning, ok
    assert cards[0]["severity"] == "critical"
    assert cards[0]["value"] == 5
    assert cards[0]["filter_value"] == "OVERDUE"
    assert cards[1]["severity"] == "warning"
    assert cards[2]["severity"] == "ok"


def test_inspection_mitigative_action_status_breakdown_uses_its_own_vocab():
    """Inspection domain's own workflow enum (RETURN_FOR_MODIFICATION /
    PENDING_FOR_REVIEW / APPROVE — see knowledge/semantics_registry.py) is
    differently spelled from the audit domain's but must classify the same
    way via keyword substrings, not an exact per-domain enum whitelist."""
    rows = [
        {"implementation_status": "RETURN_FOR_MODIFICATION", "cnt": 2},
        {"implementation_status": "PENDING_FOR_REVIEW", "cnt": 7},
        {"implementation_status": "APPROVE", "cnt": 20},
    ]
    cards = build_kpi_cards(["implementation_status", "cnt"], rows)
    assert len(cards) == 3
    assert cards[0]["label"] == "Return For Modification"
    assert cards[0]["severity"] == "critical"
    assert cards[1]["label"] == "Pending For Review"
    assert cards[1]["severity"] == "warning"
    assert cards[2]["label"] == "Approve"
    assert cards[2]["severity"] == "ok"


def test_status_breakdown_does_not_fire_below_two_distinct_values():
    rows = [
        {"status": "OVERDUE", "cnt": 5},
        {"status": "OVERDUE", "cnt": 3},
    ]
    cards = build_kpi_cards(["status", "cnt"], rows)
    # Falls through to the generic path (a "Total cnt" sum + "Unique status" card)
    assert not any(c.get("kind") == "status" for c in cards)


def test_status_breakdown_does_not_fire_above_six_distinct_values():
    rows = [{"status": f"STATE_{i}", "cnt": i} for i in range(8)]
    cards = build_kpi_cards(["status", "cnt"], rows)
    assert not any(c.get("kind") == "status" for c in cards)


def test_unrelated_score_and_sum_cards_are_unaffected():
    """No status/workflow-named column at all — completely untouched by
    this feature, existing avg/range/sum/distinct behavior unchanged."""
    rows = [
        {"facility_name": "A", "inspection_score": 80},
        {"facility_name": "B", "inspection_score": 90},
        {"facility_name": "C", "inspection_score": 70},
    ]
    cards = build_kpi_cards(["facility_name", "inspection_score"], rows)
    kinds = {c["kind"] for c in cards}
    assert "status" not in kinds
    assert "avg" in kinds
    assert "range" in kinds


def test_generic_sum_card_still_used_for_non_status_categorical_axis():
    """A categorical column that ISN'T status/workflow-named (e.g. facility)
    paired with a plain count must still get the old generic 'Total X' sum
    card, not per-value cards — this feature is scoped to status columns
    only, not every low-cardinality category."""
    rows = [
        {"facility_name": "A", "cnt": 5},
        {"facility_name": "B", "cnt": 3},
        {"facility_name": "C", "cnt": 12},
    ]
    cards = build_kpi_cards(["facility_name", "cnt"], rows)
    assert not any(c.get("kind") == "status" for c in cards)
    assert any(c["kind"] == "sum" and c["label"] == "Total Cnt" for c in cards)
