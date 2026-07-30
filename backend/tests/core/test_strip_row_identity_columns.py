"""
Regression tests for src/api/response_formatting.py's
strip_row_identity_columns() — the quarantine boundary for columns injected
by src/sql/validation/identity.py's inject_row_identity().
"""

from src.api.response_formatting import strip_row_identity_columns


def test_strips_id_suffixed_columns():
    columns = ["title", "status", "title__id"]
    rows = [
        {"title": "Delegation retirement plan", "status": "APPROVED",
         "title__id": "c58b5540-6cac-41ea-a308-2020ee9497b2"},
    ]
    kept_cols, kept_rows = strip_row_identity_columns(columns, rows)
    assert kept_cols == ["title", "status"]
    assert kept_rows == [{"title": "Delegation retirement plan", "status": "APPROVED"}]


def test_no_op_when_no_id_columns_present():
    columns = ["title", "status"]
    rows = [{"title": "x", "status": "y"}]
    kept_cols, kept_rows = strip_row_identity_columns(columns, rows)
    assert kept_cols == columns
    assert kept_rows == rows


def test_handles_empty_input():
    assert strip_row_identity_columns([], []) == ([], [])


def test_strips_multiple_id_columns():
    columns = ["title", "title__id", "facility_name", "facility_name__id"]
    rows = [{"title": "x", "title__id": "u1", "facility_name": "y", "facility_name__id": "u2"}]
    kept_cols, kept_rows = strip_row_identity_columns(columns, rows)
    assert kept_cols == ["title", "facility_name"]
    assert kept_rows == [{"title": "x", "facility_name": "y"}]
