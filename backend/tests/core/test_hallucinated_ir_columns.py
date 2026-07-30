"""
Regression test for engine.py's HALLUCINATED_IR_COLUMNS — confirmed live
2026-07-29 (qid=fe1426e8): "was there any inspection at aldar hQ" hallucinated
ir.inspection_type_name directly on inspection_report, got a real Postgres
"column does not exist" error, and self-corrected on the next retry. The
existing "inspection_type" entry didn't catch it because word-boundary
matching requires the exact column name and "_name" continues the word.
"""

from src.sql.validation import SQLValidator

validator = SQLValidator(domain="inspection")


def test_block_hallucinated_inspection_type_name():
    sql = (
        "SELECT ir.inspection_id, ir.inspection_score, ir.status, ir.submitted_on, "
        "ir.inspection_type_name, fac.name AS facility_name "
        "FROM inspection_report ir JOIN facility fac ON ir.facility_id = fac.id "
        "WHERE fac.name ILIKE '%Aldar HQ%' AND ir.active = true AND ir.status != 'DRAFT';"
    )
    vresult = validator.validate(sql)
    hard = {i.code for i in vresult.hard_fails}
    assert "WRONG_COLUMN" in hard
    assert not vresult.passed


def test_hallucinated_inspection_type_name_stays_quiet_with_correct_join():
    sql = (
        "SELECT ir.inspection_id, it.name AS inspection_type_name, fac.name AS facility_name "
        "FROM inspection_report ir "
        "JOIN facility fac ON ir.facility_id = fac.id "
        "JOIN inspection_type it ON ir.inspection_type_id = it.id "
        "WHERE fac.name ILIKE '%Aldar HQ%' AND ir.active = true AND ir.status != 'DRAFT' "
        "AND fac.active = true AND it.active = true LIMIT 100;"
    )
    vresult = validator.validate(sql)
    hard = {i.code for i in vresult.hard_fails}
    assert "WRONG_COLUMN" not in hard
