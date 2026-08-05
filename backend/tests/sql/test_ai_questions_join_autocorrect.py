"""
Regression tests for cleaner.py's ast_repair() step 7 — deterministic
injection of a missing `JOIN ai_questions aq ON ...element_id = aq.element_id`
when the SQL-gen model compresses the ai_answers/ai_audit_answers +
ai_questions pattern into a single JOIN, folding an ai_questions filter (e.g.
`aq.label = 'Findings / Remarks'`) into the ai_answers/ai_audit_answers JOIN's
own ON clause while dropping the ai_questions JOIN entirely.

Confirmed live 2026-08-04: this exact malformed shape was the root cause of
the audit eval's MULTI-03 regression — qwen2.5-coder repeated it 6 times in a
row despite the [UNDEFINED_ALIAS] validator blocking it every time, exhausting
the retry budget and falling back to a synthesized prose answer.

The fix moves the dangling predicate (e.g. `aq.label = 'Findings / Remarks'`)
out of the answers-table JOIN's own ON clause and into a newly-inserted
ai_questions JOIN placed immediately after it — a first attempt that merely
appended the new JOIN at the end passed this validator's order-agnostic
alias-existence check, but re-confirmed live via EXPLAIN against the real
staging DB that Postgres itself still rejected it ("missing FROM-clause entry
for table aq"), since real JOIN ON-clause resolution is strictly left-to-right.
"""

from src.sql.validation.engine import SQLValidator


def _validator(domain="audit"):
    return SQLValidator(domain=domain)


def test_injects_missing_join_for_ai_audit_answers_dangling_label():
    v = _validator("audit")
    sql = (
        "SELECT aca.corrective_action_id, aaa.answer_text AS finding_text "
        "FROM audit_corrective_action AS aca "
        "JOIN review AS rev ON rev.entity_id = aca.id "
        "LEFT JOIN ai_audit_answers AS aaa ON aaa.submission_id = CAST(aca.observation_id AS VARCHAR) "
        "AND aq.label = 'Findings / Remarks' "
        "WHERE aca.active = TRUE;"
    )
    fixed, mutations = v.ast_repair(sql)
    assert len(mutations) == 1
    assert "ai_questions" in fixed
    assert "aaa.element_id = aq.element_id" in fixed
    result = v.validate(fixed, "show details")
    assert result.passed, [str(f) for f in result.hard_fails]


def test_injects_missing_join_for_ai_answers_dangling_label_inspection_domain():
    """Same malformed shape, inspection domain's ai_answers table (not
    ai_audit_answers) — the repair covers both tables since the same
    ai_answers/ai_questions element_id pairing exists in both domains."""
    v = _validator("inspection")
    sql = (
        "SELECT ir.inspection_id, aa.answer_text AS observation "
        "FROM inspection_report AS ir "
        "LEFT JOIN ai_answers AS aa ON aa.inspection_report_id = ir.id "
        "AND aq.label = 'Observation' "
        "WHERE ir.active = TRUE;"
    )
    fixed, mutations = v.ast_repair(sql)
    assert len(mutations) == 1
    assert "ai_questions" in fixed
    assert "aa.element_id = aq.element_id" in fixed


def test_no_op_when_join_already_present():
    """The correct 2-join form (matching every seed example) must not be
    touched — no double-injection."""
    v = _validator("audit")
    sql = (
        "SELECT aca.corrective_action_id, aaa.answer_text AS finding_text "
        "FROM audit_corrective_action AS aca "
        "JOIN ai_audit_answers AS aaa ON aaa.submission_id = aca.observation_id::varchar "
        "JOIN ai_questions AS aq ON aaa.element_id = aq.element_id "
        "WHERE aca.active = TRUE AND aq.label = 'Findings / Remarks';"
    )
    fixed, mutations = v.ast_repair(sql)
    assert mutations == []
    assert fixed == sql


def test_no_op_when_no_answers_table_present():
    """An unrelated undefined alias (e.g. a genuinely different bug) must
    not trigger this narrow, specific repair."""
    v = _validator("audit")
    sql = (
        "SELECT ap.title FROM audit_plan ap WHERE ap.active = TRUE "
        "AND aq.label = 'something unrelated';"
    )
    fixed, mutations = v.ast_repair(sql)
    assert mutations == []
    assert fixed == sql


def test_no_op_when_dangling_label_is_ambiguous():
    """Two distinct dangling `.label` aliases inside the SAME answers-table
    JOIN's ON clause is outside this repair's narrow, deterministic scope —
    leave it for the validator/retry loop rather than guessing which one is
    ai_questions."""
    v = _validator("audit")
    sql = (
        "SELECT aca.corrective_action_id, aaa.answer_text AS finding_text "
        "FROM audit_corrective_action AS aca "
        "LEFT JOIN ai_audit_answers AS aaa ON aaa.submission_id = aca.observation_id::varchar "
        "AND aq.label = 'Findings / Remarks' AND other_aq.label = 'Something else' "
        "WHERE aca.active = TRUE;"
    )
    fixed, mutations = v.ast_repair(sql)
    assert mutations == []
    assert fixed == sql


def test_repairs_the_answers_join_even_when_an_unrelated_dangling_alias_exists_elsewhere():
    """A separate, unrelated dangling alias elsewhere in the query (e.g. in
    WHERE, nothing to do with the ai_answers/ai_questions pattern) must not
    block this narrowly-scoped repair — the validator will still separately
    catch the unrelated alias afterward."""
    v = _validator("audit")
    sql = (
        "SELECT aca.corrective_action_id, aaa.answer_text AS finding_text "
        "FROM audit_corrective_action AS aca "
        "LEFT JOIN ai_audit_answers AS aaa ON aaa.submission_id = aca.observation_id::varchar "
        "AND aq.label = 'Findings / Remarks' "
        "WHERE aca.active = TRUE AND other_alias.status = 'X';"
    )
    fixed, mutations = v.ast_repair(sql)
    assert len(mutations) == 1
    assert "aaa.element_id = aq.element_id" in fixed
    # the unrelated dangling alias is untouched — still there for the
    # validator to catch as its own separate [UNDEFINED_ALIAS] failure
    assert "other_alias.status" in fixed
