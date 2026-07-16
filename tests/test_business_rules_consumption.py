"""
Deterministic regression tests for the validator <-> business_rules registry
consumption refactor (see core/validator.py::_CONSUMED_RULE_IDS).

No LLM, no live DB — pure regex/string matching against fixture SQL. Run
directly (`python tests/test_business_rules_consumption.py`) or via pytest
if it's ever added to this project (test_*() functions are pytest-discoverable
as-is).

Each fixture below is deliberately the smallest SQL string that triggers
exactly one allowlisted rule, so a failure here points at one specific rule.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.validator import SQLValidator

validator = SQLValidator()

FAILURES = []


def _check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def _hard_fail_codes(sql, question=""):
    vresult = validator.validate(sql, question)
    return {i.code for i in vresult.hard_fails}, {i.code for i in vresult.soft_warns}, vresult.passed


# ---------------------------------------------------------------------------
# Group A — pure consolidation (formerly hand-written duplicates)
# ---------------------------------------------------------------------------

def test_block_dml():
    hard, _, passed = _hard_fail_codes("DELETE FROM inspection_report;")
    _check("block_dml -> DML_BLOCKED", "DML_BLOCKED" in hard and not passed)


def test_block_table_ai_question_singular():
    hard, _, _ = _hard_fail_codes("SELECT * FROM ai_question;")
    _check("block_table_ai_question_singular -> UNKNOWN_TABLE", "UNKNOWN_TABLE" in hard)


def test_block_table_ai_answer_singular():
    hard, _, _ = _hard_fail_codes("SELECT * FROM ai_answer;")
    _check("block_table_ai_answer_singular -> UNKNOWN_TABLE", "UNKNOWN_TABLE" in hard)


def test_block_table_fb_users():
    hard, _, _ = _hard_fail_codes("SELECT * FROM fb_users;")
    _check("block_table_fb_users -> UNKNOWN_TABLE", "UNKNOWN_TABLE" in hard)


def test_block_table_fb_submissions():
    hard, _, _ = _hard_fail_codes("SELECT * FROM fb_submissions;")
    _check("block_table_fb_submissions -> UNKNOWN_TABLE", "UNKNOWN_TABLE" in hard)


def test_block_table_fb_answers():
    hard, _, _ = _hard_fail_codes("SELECT * FROM fb_answers;")
    _check("block_table_fb_answers -> UNKNOWN_TABLE", "UNKNOWN_TABLE" in hard)


def test_block_table_fb_responses():
    hard, _, _ = _hard_fail_codes("SELECT * FROM fb_responses;")
    _check("block_table_fb_responses -> UNKNOWN_TABLE", "UNKNOWN_TABLE" in hard)


def test_block_table_dynamic_fb():
    hard, _, _ = _hard_fail_codes(
        "SELECT * FROM fb_12345678_1234_1234_1234_123456789012;"
    )
    _check("block_table_dynamic_fb -> DYNAMIC_TABLE", "DYNAMIC_TABLE" in hard)


def test_block_responsible_fk_join():
    hard, _, _ = _hard_fail_codes(
        "SELECT * FROM inspection_corrective_action ica "
        "JOIN users u ON ica.responsible = u.id;"
    )
    _check("block_responsible_fk_join -> WRONG_JOIN", "WRONG_JOIN" in hard)


def test_block_close_on_arithmetic():
    hard, _, _ = _hard_fail_codes(
        "SELECT * FROM inspection_corrective_action WHERE close_on > CURRENT_DATE;"
    )
    _check("block_close_on_arithmetic -> DEAD_COLUMN", "DEAD_COLUMN" in hard)


def test_warn_no_limit():
    _, soft, passed = _hard_fail_codes("SELECT id, name FROM facility;")
    _check("warn_no_limit -> MISSING_LIMIT (soft, still passes)", "MISSING_LIMIT" in soft and passed)


def test_block_hardcoded_year_is_soft_warn_not_hard_fail():
    """
    Negative fixture — this is the regression this refactor must never
    reintroduce. The registry rule was BLOCK; validator.py deliberately
    demoted it to WARN in production (blocking broke legitimate historical-
    year queries). Confirm it stays WARN after the registry fix.
    """
    hard, soft, passed = _hard_fail_codes(
        "SELECT * FROM inspection_report WHERE EXTRACT(YEAR FROM submitted_on) = 2025;"
    )
    _check(
        "block_hardcoded_year -> SOFT_WARN only, query still passes",
        "HARDCODED_YEAR" in soft and "HARDCODED_YEAR" not in hard and passed,
    )


# ---------------------------------------------------------------------------
# Group A — AUTOCORRECT (clean_sql())
# ---------------------------------------------------------------------------

def test_autocorrect_ica_ir_join_varchar_uuid():
    out = validator.clean_sql(
        "SELECT * FROM inspection_report ir "
        "JOIN inspection_corrective_action ica ON ir.inspection_id = ica.inspection_id;"
    )
    _check(
        "autocorrect_ica_ir_join_varchar_uuid",
        "ica.inspection_id = ir.id" in out,
        detail=out,
    )


def test_autocorrect_ica_ir_join_reversed():
    out = validator.clean_sql(
        "SELECT * FROM inspection_report ir "
        "JOIN inspection_corrective_action ica ON ica.inspection_id = ir.inspection_id;"
    )
    _check(
        "autocorrect_ica_ir_join_reversed",
        "ica.inspection_id = ir.id" in out,
        detail=out,
    )


def test_autocorrect_ica_ir_join_cast():
    out = validator.clean_sql(
        "SELECT * FROM inspection_report ir "
        "JOIN inspection_corrective_action ica ON CAST(ir.inspection_id AS TEXT) = ica.inspection_id;"
    )
    _check(
        "autocorrect_ica_ir_join_cast",
        "ica.inspection_id = ir.id" in out,
        detail=out,
    )


def test_autocorrect_ica_ir_join_cast_typecast():
    out = validator.clean_sql(
        "SELECT * FROM inspection_report ir "
        "JOIN inspection_corrective_action ica ON ir.inspection_id::text = ica.inspection_id;"
    )
    _check(
        "autocorrect_ica_ir_join_cast_typecast",
        "ica.inspection_id = ir.id" in out,
        detail=out,
    )


def test_autocorrect_reserved_alias_is():
    out = validator.clean_sql(
        "SELECT is.id FROM inspection_schedule is WHERE is.active = true;"
    )
    _check(
        "autocorrect_reserved_alias_is (declaration site)",
        "inspection_schedule isch" in out,
        detail=out,
    )
    _check(
        "autocorrect_reserved_alias_is (residual is. -> isch. cleanup)",
        "isch.id" in out and "isch.active" in out and "is." not in out,
        detail=out,
    )


# ---------------------------------------------------------------------------
# Group B — new enforcement (previously-dormant registry-only rules)
# ---------------------------------------------------------------------------

def test_block_po_pr_double_direct_join():
    hard, _, _ = _hard_fail_codes(
        "SELECT ica.id, SUM(po.po_amount), SUM(pr.pr_amount) "
        "FROM inspection_corrective_action ica "
        "LEFT JOIN po_details po ON po.mitigative_action_uuid = ica.id "
        "LEFT JOIN pr_details pr ON pr.mitigative_action_uuid = ica.id "
        "GROUP BY ica.id;"
    )
    _check("block_po_pr_double_direct_join -> PO_PR_DOUBLE_JOIN", "PO_PR_DOUBLE_JOIN" in hard)


def test_block_capex_left_join_multiplication():
    hard, _, _ = _hard_fail_codes(
        "SELECT SUM(ica.capex) FROM inspection_corrective_action ica "
        "LEFT JOIN po_details po ON po.mitigative_action_uuid = ica.id;"
    )
    _check("block_capex_left_join_multiplication -> CAPEX_LEFT_JOIN_MULT", "CAPEX_LEFT_JOIN_MULT" in hard)


def test_block_po_pr_join_to_ir():
    hard, _, _ = _hard_fail_codes(
        "SELECT * FROM po_details po "
        "JOIN inspection_report ir ON po.mitigative_action_uuid = ir.id;"
    )
    _check("block_po_pr_join_to_ir -> PO_PR_JOIN_TO_IR", "PO_PR_JOIN_TO_IR" in hard)


def test_block_mitigative_uuid_wrong_target():
    hard, _, _ = _hard_fail_codes(
        "SELECT * FROM po_details po "
        "JOIN facility fac ON po.mitigative_action_uuid = fac.id;"
    )
    _check("block_mitigative_uuid_wrong_target -> MITIGATIVE_UUID_WRONG_TARGET", "MITIGATIVE_UUID_WRONG_TARGET" in hard)


def test_block_expenditure_as_conversion():
    hard, _, _ = _hard_fail_codes(
        "SELECT CASE WHEN expenditure IN ('CAPEX') THEN capex END AS converted "
        "FROM inspection_corrective_action;"
    )
    _check("block_expenditure_as_conversion -> EXPENDITURE_AS_CONVERSION", "EXPENDITURE_AS_CONVERSION" in hard)


def test_block_inspector_not_in_pattern():
    hard, _, _ = _hard_fail_codes(
        "SELECT * FROM users u WHERE u.id NOT IN "
        "(SELECT inspector_user_id FROM inspection_report);",
        question="which inspectors have not had any inspection activity",
    )
    _check("block_inspector_not_in_pattern -> INSPECTOR_NOT_IN_PATTERN", "INSPECTOR_NOT_IN_PATTERN" in hard)


# ---------------------------------------------------------------------------
# Regression guard: a rule NOT in the allowlist must never fire, even if a
# question/SQL combination would match its pattern — proves the allowlist
# gate itself works, not just that the 17 consumed rules work.
# ---------------------------------------------------------------------------

def test_non_allowlisted_rule_never_fires():
    # block_organisation_join is a real, valid BLOCK rule in the registry
    # (sql_pattern: r"\bJOIN\s+organisation\b") but deliberately NOT in
    # _CONSUMED_RULE_IDS for this pass — confirm it stays inert even though
    # this SQL would trip it if the allowlist gate were broken.
    hard, _, _ = _hard_fail_codes(
        "SELECT * FROM facility fac JOIN organisation o ON fac.organization_id = o.id;"
    )
    _check(
        "non-allowlisted registry rule (block_organisation_join) does not fire",
        len(hard) == 0,
        detail=f"hard_fails={hard}",
    )


if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in test_fns:
        try:
            fn()
        except Exception as e:
            print(f"[ERROR] {fn.__name__} raised {e!r}")
            FAILURES.append(fn.__name__)

    print()
    if FAILURES:
        print(f"{len(FAILURES)}/{len(test_fns)} FAILED: {FAILURES}")
        sys.exit(1)
    else:
        print(f"ALL {len(test_fns)} CHECKS PASSED")
