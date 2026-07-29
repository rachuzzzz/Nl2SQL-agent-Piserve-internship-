"""
AUDIT_BLOCK_RULES / AUDIT_WARN_RULES / AUDIT_AUTOCORRECT_RULES / AUDIT_INJECT_RULES
====================================================================================
Audit-domain equivalent of block_rules.py / warn_rules.py / autocorrect_rules.py /
inject_rules.py — same dict shape (see _types.py), consumed by a separate
RulesEngine instance (get_audit_engine() in engine.py), not merged into the
inspection REGISTRY.

Every trap encoded here was confirmed live against the DB — see
docs/Audit_Domain_Schema_Reference.docx, knowledge/schema_introspector.py's
AUDIT_BUSINESS_RULES (prose form of most of these), and the domain-router
plan's §1b for the review/review_level findings.
"""

from src.sql.rules._types import BLOCK, WARN, INJECT, AUTOCORRECT


AUDIT_BLOCK_RULES: list[dict] = [
    {
        "id":          "audit_block_dead_schedule_status",
        "type":        BLOCK,
        "code":        "AUDIT_DEAD_SCHEDULE_STATUS",
        "description": "audit_schedule.audit_status is always NULL (confirmed live) — the real status column is schedule_status.",
        "sql_pattern": r"\.audit_status\b",
        "message": (
            "BLOCKED: audit_status on audit_schedule is a dead column (always NULL, confirmed "
            "live). Use schedule_status instead — values: DRAFT, PENDING_REVIEW, APPROVED, "
            "ONGOING, COMPLETED, RETURN_MODIFICATION."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_dead_report_audit_type",
        "type":        BLOCK,
        "code":        "AUDIT_DEAD_REPORT_AUDIT_TYPE",
        "description": "audit_report.audit_type_id is always NULL (confirmed live) — resolve audit type via audit_plan instead.",
        "sql_pattern": r"(?:audit_report|arep)\.audit_type_id\b",
        "message": (
            "BLOCKED: audit_report.audit_type_id is a dead column (always NULL, confirmed live). "
            "The real audit-type path is THREE hops: audit_report.schedule_id → "
            "audit_schedule.id → audit_schedule.audit_plan_id → audit_plan.id → "
            "audit_plan.audit_type_id → audit_type.id."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_report_missing_facility_project",
        "type":        BLOCK,
        "code":        "AUDIT_REPORT_NO_FACILITY_PROJECT",
        "description": "audit_report has no facility_id/project_id column of its own — must go via audit_schedule.",
        "sql_pattern": r"(?:audit_report|arep)\.(?:facility_id|project_id)\b",
        "message": (
            "BLOCKED: audit_report has NO facility_id or project_id column (confirmed live) — "
            "unlike inspection_report where facility_id is direct. Two-hop via "
            "audit_report.schedule_id → audit_schedule.id → audit_schedule.facility_id/project_id."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_responsible_join",
        "type":        BLOCK,
        "code":        "AUDIT_RESPONSIBLE_NOT_FK",
        "description": "audit_corrective_action.responsible is a plain string enum (CLIENT/INTERNAL_OPERATIONS/SUB_CONTRACTOR), not a users FK — confirmed live (a real generation failure caught in end-to-end testing: 'operator does not exist: character varying = integer').",
        "sql_pattern": r"\.responsible\s*=\s*\w+\.id\b",
        "message": (
            "BLOCKED: audit_corrective_action.responsible is a plain string enum "
            "('CLIENT', 'INTERNAL_OPERATIONS', 'SUB_CONTRACTOR'), NOT a foreign key to users. "
            "Never JOIN users ON ...responsible = u.id — filter/select it directly as a string."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_ca_report_join_wrong_column",
        "type":        BLOCK,
        "code":        "AUDIT_CA_REPORT_JOIN_WRONG_COL",
        "description": (
            "audit_corrective_action.audit_id (UUID FK) must join to audit_report.id, NEVER "
            "audit_report.audit_id (the VARCHAR display code) — confirmed live: a real generation "
            "failure in end-to-end testing produced 'operator does not exist: uuid = character "
            "varying', and the model's own retry then made it WORSE by skipping audit_report "
            "entirely (see audit_block_ca_schedule_direct_join below)."
        ),
        "sql_pattern": r"\baudit_id\s*=\s*\w+\.audit_id\b",
        "message": (
            "BLOCKED: comparing two columns both literally named audit_id is always wrong here — "
            "audit_corrective_action.audit_id is a UUID FK, audit_report.audit_id is the VARCHAR "
            "display code (a completely different column despite the shared name). "
            "Correct: JOIN audit_report arep ON aca.audit_id = arep.id"
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_review_entity_id_corrective_action_id",
        "type":        BLOCK,
        "code":        "AUDIT_REVIEW_ENTITY_ID_WRONG_COL",
        "description": (
            "review.entity_id (UUID, polymorphic) must join to audit_corrective_action.id, "
            "NEVER .corrective_action_id (the VARCHAR display code, e.g. "
            "'2026/06/ST041/AUD003_CA004' — not UUID-shaped at all). Same trap family as "
            "audit_block_ca_report_join_wrong_column above (two differently-typed columns "
            "compared where the wrong one 'sounds like' the right ID). Confirmed live: this "
            "burned 3 wasted iterations (uuid=varchar error, then a bad ::uuid cast producing "
            "'invalid input syntax for type uuid', then the identical uncast error again) plus "
            "2 get_schema calls before the model found aca.id on its own — 6 LLM calls / 27s / "
            "64K tokens for a query an adjacent, nearly-identical question answered correctly "
            "in 1 call."
        ),
        "sql_pattern": (
            r"\bentity_id\s*=\s*\w+\.corrective_action_id\b(?:\s*::\s*\w+)?"
            r"|\w+\.corrective_action_id\b(?:\s*::\s*\w+)?\s*=\s*\w+\.entity_id\b"
        ),
        "message": (
            "BLOCKED: review.entity_id must join to audit_corrective_action.id (the UUID "
            "primary key), NOT .corrective_action_id (the VARCHAR display code — never "
            "UUID-shaped, casting it to uuid will fail). Correct: "
            "JOIN review rev ON rev.entity_id = aca.id AND rev.entity_type = 'AUDIT_CORRECTIVE_ACTION'"
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_ca_schedule_direct_join",
        "type":        BLOCK,
        "code":        "AUDIT_CA_SCHEDULE_DIRECT_JOIN",
        "description": (
            "audit_corrective_action must NEVER join directly to audit_schedule — audit_report is "
            "a mandatory intermediate hop. Confirmed live: caught as the model's own (wrong) retry "
            "fix after audit_block_ca_report_join_wrong_column fired — it silently executes (both "
            "sides are UUID-typed) but returns 0 rows because the join target is unrelated."
        ),
        "sql_pattern": r"\baca\.audit_id\s*=\s*asch\.id\b|\basch\.id\s*=\s*aca\.audit_id\b",
        "message": (
            "BLOCKED: audit_corrective_action.audit_id points to audit_report.id, NOT "
            "audit_schedule.id — joining it directly to audit_schedule silently returns 0 rows "
            "(no error, since both sides are UUID-typed, but the join target is wrong table entirely). "
            "Correct 2-hop path: JOIN audit_report arep ON aca.audit_id = arep.id "
            "JOIN audit_schedule asch ON arep.schedule_id = asch.id"
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_portfolio_ap_alias",
        "type":        BLOCK,
        "code":        "PORTFOLIO_AP_ALIAS_COLLISION",
        "description": (
            "auditor_portfolio aliased as 'ap' collides with audit_plan's established alias — "
            "'ap' legitimately has .title/.status on audit_plan, so a query that (mis)aliases "
            "auditor_portfolio as 'ap' hallucinates those same column names onto the wrong "
            "table. Confirmed live: the reasoning LLM's own schema_hint did exactly this for "
            "'which auditor portfolios were approved' (FROM auditor_portfolio ap ... ap.title), "
            "burning 9 LLM calls / 36s before giving up — the business-rule prose note alone "
            "(see schema_introspector.py's ALIAS COLLISION rule) wasn't consistently followed, "
            "same as every other HALLUCINATED_* column trap in this codebase needing a hard "
            "enforcement layer, not just documentation."
        ),
        # Unconditional — aliasing auditor_portfolio as 'ap' is ALWAYS wrong, not
        # question-dependent, so no question_pattern gate is needed.
        "sql_pattern": r"\bauditor_portfolio\s+(?:AS\s+)?ap\b",
        "message": (
            "BLOCKED: auditor_portfolio aliased as 'ap' collides with audit_plan's established "
            "alias convention. Use 'po' for auditor_portfolio instead (e.g. "
            "FROM auditor_portfolio po) — matching the seed examples. auditor_portfolio has NO "
            "'title' or 'portfolio_id' column; those belong to audit_plan."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_dead_portfolio_junctions",
        "type":        BLOCK,
        "code":        "AUDIT_DEAD_PORTFOLIO_JUNCTION",
        "description": "audit_category_portfolio / audit_sub_category_portfolio / audit_type_portfolio / audit_cycle_auditor_portfolios are always 0 rows (confirmed live) — dead tables.",
        "sql_pattern": r"\b(?:audit_category_portfolio|audit_sub_category_portfolio|audit_type_portfolio|audit_cycle_auditor_portfolios)\b",
        "message": (
            "BLOCKED: this table is permanently empty (0 rows, confirmed live) despite naming "
            "that reads as equally valid to the real tables. Use "
            "portfolio_detail_audit_category / portfolio_detail_audit_subcategory / "
            "portfolio_detail_audit_type instead (joined via auditor_portfolio_detail.id → "
            "detail_id), and auditor_portfolio.audit_cycle_id directly for cycle coverage."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_ca_draft_not_excluded",
        "type":        BLOCK,
        "code":        "AUDIT_CA_DRAFT_NOT_EXCLUDED",
        "description": (
            "General audit_corrective_action listings must exclude rows still "
            "pending adequacy approval by default (adequacy_status != 'APPROVE', "
            "displayed as corrective_action_id = 'CA_DRAFT') unless the question "
            "explicitly asks for pending/draft/unapproved actions or adequacy "
            "review status."
        ),
        # Gated by question_pattern, not SQL-shape alone — a valid answer to an
        # explicit pending-approval ask can reasonably go through the review
        # table (review_type='ADEQUACY' AND status='PENDING') and reference
        # neither adequacy_status nor corrective_action_id at all. Only fire
        # this rule for questions that look like a GENERAL listing.
        "question_pattern": (
            r"(?i)(?=.*\bcorrective\s+actions?\b)"
            r"(?!.*\b(pending|draft|unapprov\w*|adequac\w*|approv\w*|review)\b)"
        ),
        "sql_pattern": (
            r"(?=.*\bFROM\s+audit_corrective_action\b)"
            r"(?!.*\badequacy_status\s*(?:!=|=|IN\s*\(|NOT\s+IN|"
            r"IS(?:\s+NOT)?\s+NULL|IS\s+DISTINCT\s+FROM))"
            r"(?!.*\bcorrective_action_id\s*(?:!=|=|NOT\s+IN|IN\s*\())"
        ),
        "message": (
            "BLOCKED: general corrective-action listing queries must exclude "
            "rows still pending adequacy approval by default — add "
            "AND aca.adequacy_status = 'APPROVE'. Only omit this when the "
            "question explicitly asks for pending/draft/unapproved corrective "
            "actions or adequacy review/approval status."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_ca_existence_check_adequacy_filtered",
        "type":        BLOCK,
        "code":        "AUDIT_CA_EXISTENCE_CHECK_ADEQUACY_FILTERED",
        "description": (
            "An existence check against audit_corrective_action (NOT EXISTS / "
            "NOT IN / aca.id IS NULL after LEFT JOIN) must never also filter by "
            "adequacy_status or corrective_action_id — that silently narrows "
            "'no corrective action raised' into 'no APPROVED corrective action', "
            "wrongly counting a CA_DRAFT (raised, pending approval) row as if "
            "it doesn't exist. Confirmed live: this reintroduces the CA_DRAFT/"
            "existence-check conflation bug via a different SQL shape than the "
            "LISTING case audit_block_ca_draft_not_excluded already guards —"
            "unlike that rule, this one is unconditional (no question_pattern),"
            " because the follow-up question that triggered it didn't even"
            " contain the words 'corrective action'."
        ),
        # Deliberately unconditional (no question_pattern) — a genuine NOT
        # EXISTS existence check should never need an adequacy_status/
        # corrective_action_id filter inside it regardless of how the
        # question was phrased, so gating on question text isn't needed and
        # would just reopen the same gap this rule exists to close.
        # Broadened after a second live failure: caught NOT EXISTS + adequacy_status,
        # but missed "aca.id IS NULL OR aca.corrective_action_id = 'CA_DRAFT'" — the
        # exact original bug shape, reintroduced a third time. Any direct comparison
        # against adequacy_status='APPROVE' or corrective_action_id vs 'CA_DRAFT'
        # (either direction, = or !=) is the same conflation once an existence-check
        # shape is also present — and the existence-check detector itself needed
        # broadening from aca.id specifically to any aca.<col> IS NULL, since
        # LEFT JOIN ... IS NULL commonly lands on whichever column was SELECTed,
        # not always the PK.
        "sql_pattern": (
            r"(?=.*\baudit_corrective_action\b)"
            r"(?=.*(?:\baca\.\w+\s+IS\s+NULL\b|\bNOT\s+EXISTS\s*\(|\bNOT\s+IN\s*\())"
            r"(?=.*(?:adequacy_status\s*(?:=|!=|<>)\s*'APPROVE'|"
            r"corrective_action_id\s*(?:=|!=|<>)\s*'CA_DRAFT'))"
        ),
        "message": (
            "BLOCKED: this is an existence check (NOT EXISTS / NOT IN / "
            "aca.id IS NULL) against audit_corrective_action, but it also "
            "filters by adequacy_status or corrective_action_id — remove that "
            "filter entirely from the existence check. A 'CA_DRAFT' row is a "
            "REAL corrective action that WAS raised, just not yet approved; it "
            "must count as EXISTS/raised = true for this kind of question."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_review_completed_not_approved",
        "type":        BLOCK,
        "code":        "REVIEW_COMPLETED_NOT_APPROVED",
        "description": (
            "review.status = 'COMPLETED' does NOT mean approved — a completed review can "
            "have action = 'RETURN_FOR_MODIFICATION' (rejected/sent back) instead of "
            "'APPROVE'. Confirmed live: 73 of 261 completed reviews (28%) are rejections, "
            "not approvals. 'who/which approved' questions must filter "
            "status = 'COMPLETED' AND action = 'APPROVE', not status alone — same shape of "
            "trap as audit_block_ca_draft_not_excluded's adequacy_status filter."
        ),
        # Gated on the question actually asking about approval (not rejection/pending) —
        # same style as audit_block_ca_draft_not_excluded's question_pattern, so a genuine
        # "list all reviews" or "which are pending" question that legitimately has no
        # action filter doesn't get blocked.
        # MUST be anchored with ^ — re.search() tries every starting position, and
        # without the anchor a question like "percentage of completed reviews that
        # were rejections vs approvals" (exclude-word BEFORE the approve-word) finds
        # a later start position past "rejections" where the negative lookahead's
        # forward-only scan no longer sees it, while "approvals" is still ahead —
        # a false match. Confirmed live: eval_audit.py's APPR-08 blocked a
        # correct, already-safe COUNT(CASE WHEN action='APPROVE'...) query for 9
        # LLM calls / 49s before giving up. ^ forces the lookaheads to evaluate
        # the WHOLE string exactly once, from the start.
        # need\w*/requir\w* excluded too — confirmed live: "which corrective actions
        # NEEDED a second level of adequacy approval" asks about the PROCESS
        # (which ones went through/required level-2 review, both APPROVE and
        # RETURN_FOR_MODIFICATION outcomes legitimately belong), not the OUTCOME
        # ("who approved" = APPROVE only). Without this exclusion, this rule blocked
        # a correct query that already correctly included both outcomes, then the
        # retry dropped RETURN_FOR_MODIFICATION entirely and silently returned 0 rows
        # for a question whose one true match was a RETURN_FOR_MODIFICATION row.
        "question_pattern": (
            r"(?i)^(?=.*\bapprov\w*\b)"
            r"(?!.*\b(reject\w*|return\w*|declin\w*|pending|need\w*|requir\w*)\b)"
        ),
        "sql_pattern": (
            r"(?=.*\b(?:FROM|JOIN)\s+review\b)"
            r"(?=.*\bstatus\s*=\s*'COMPLETED')"
            r"(?!.*\baction\s*=\s*'APPROVE')"
        ),
        "message": (
            "BLOCKED: review.status = 'COMPLETED' includes REJECTED reviews "
            "(action = 'RETURN_FOR_MODIFICATION'), not just approved ones. For "
            "'who/which approved' questions add AND action = 'APPROVE' alongside "
            "status = 'COMPLETED'."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_ca_multi_axis_inner_join",
        "type":        BLOCK,
        "code":        "AUDIT_CA_MULTI_AXIS_UNASKED_JOIN",
        "description": (
            "risk_level, impact, and compliance are three independent, often-NULL FKs on "
            "audit_corrective_action — INNER JOINing two or more of them at once silently drops "
            "any row missing an FK on the axis that wasn't actually asked about. Confirmed live: "
            "a 'show details' follow-up on 6 already-known high-risk corrective actions "
            "INNER-joined risk_level + impact + compliance together and silently returned 0 rows "
            "(success=True, rows=0) — instead of the 6 rows with their correction/cause text — "
            "because impact_id/compliance_id were NULL on those rows. The prompt-level guidance "
            "(audit_inject_impact_compliance_risk) alone did not reliably stop this, so it needs "
            "a hard enforcement layer, same as every other over-join trap in this file."
        ),
        # Only block when the question does NOT explicitly name two or more of the three
        # axis words itself — a genuine "corrective actions by risk level and compliance
        # gap" question legitimately wants both joined and must not be blocked. Six
        # lookaheads (both orders per pair) since ".*X.*Y" alone misses "Y ... X" ordering.
        "question_pattern": (
            r"(?i)(?!.*\bimpact\b.*\bcompliance\b)(?!.*\bcompliance\b.*\bimpact\b)"
            r"(?!.*\bimpact\b.*\brisk\b)(?!.*\brisk\b.*\bimpact\b)"
            r"(?!.*\bcompliance\b.*\brisk\b)(?!.*\brisk\b.*\bcompliance\b)"
        ),
        "sql_pattern": (
            r"(?=.*\baudit_corrective_action\b)"
            r"(?:"
            r"(?=.*(?<!LEFT )\bJOIN\s+risk_level\b)(?=.*(?<!LEFT )\bJOIN\s+impact\b)"
            r"|(?=.*(?<!LEFT )\bJOIN\s+risk_level\b)(?=.*(?<!LEFT )\bJOIN\s+compliance\b)"
            r"|(?=.*(?<!LEFT )\bJOIN\s+impact\b)(?=.*(?<!LEFT )\bJOIN\s+compliance\b)"
            r")"
        ),
        "message": (
            "BLOCKED: this query INNER JOINs two or more of risk_level/impact/compliance on "
            "audit_corrective_action, but the question only asks about one (or none) of those "
            "axes — each is an independent, sometimes-NULL FK, so joining ones the question "
            "didn't ask about can silently zero out the whole result. Keep only the axis "
            "already established by the question/prior filter (e.g. risk_level); drop the "
            "others, or use LEFT JOIN if you want to display one without filtering on it."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_conformity_spelling",
        "type":        BLOCK,
        "code":        "AUDIT_CONFORMITY_SPELLING",
        "description": (
            "The compliance table stores 'Non Confirmity' (typo, missing 'o') "
            "— 'conformity' returns zero rows. Confirmed live — mirrors the "
            "same trap already guarded for inspection's impact table "
            "(block_rules.py's block_conformity_spelling)."
        ),
        "sql_pattern": r"\bconformity\b",
        "table_guard": "compliance",
        "message": (
            "DATA TYPO: the compliance table stores 'Non Confirmity' (NOT "
            "'Non Conformity') — confirmed live. ILIKE '%conformity%' always "
            "returns zero rows. Use: WHERE comp.name ILIKE '%Non Confirmity%'"
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_desc_limit_missing_nulls_last",
        "type":        BLOCK,
        "code":        "AUDIT_DESC_LIMIT_NULLS_NOT_HANDLED",
        "description": (
            "PostgreSQL sorts NULL as the LARGEST value by default, so "
            "'ORDER BY date_col DESC LIMIT 1' without NULLS LAST returns a "
            "NULL-dated row instead of the actual most recent one whenever "
            "ANY row has a NULL value in that column — this codebase's own "
            "seed examples already add NULLS LAST for exactly this reason "
            "(audit_seed_data.py has it on follow_up_date/submitted_on "
            "dozens of times) but nothing enforced it at validation time. "
            "Confirmed live: 'ORDER BY asch.follow_up_date DESC LIMIT 1' "
            "(no NULLS LAST) returned follow_up_date=NULL for the 'most "
            "recently conducted audit' — the user asked 'who conducted it "
            "and when was it conducted' in one question and only got a real "
            "answer for 'who', because the date came back NULL, and the "
            "reasoning model's own NEXT turn had to re-diagnose this from "
            "scratch ('Turn 8's SQL returned follow_up_date: N/A') instead "
            "of the query just being correct the first time."
        ),
        # Unconditional (no question_pattern) — there is no legitimate reading
        # where a NULL date should ever win a 'largest value' DESC sort; this
        # is a pure correctness footgun, not something intentional depending
        # on how the question was phrased.
        # LIMIT 100 excluded — confirmed live this rule blocked a correct,
        # textbook "list all audits with no CA raised" query (LIMIT 100 is
        # this codebase's generic full-listing safety cap, used 44x across
        # audit_seed_data.py) and kicked off an unnecessary 4-iteration retry
        # cascade. A NULL winning the DESC sort only actually drops a real
        # row from the answer when LIMIT is small enough to truncate it (top
        # N / "most recent" semantics) — with LIMIT 100 on a full listing,
        # every matching row is still present, just possibly reordered.
        "sql_pattern": (
            r"\bORDER\s+BY\s+\w+\.(?:follow_up_date|submitted_on|target_close_out_date|"
            r"close_on|completed_on|due_date)\s+DESC\b"
            r"(?!\s+NULLS\s+LAST)"
            r"(?:(?!;).)*?\bLIMIT\s+(?!100\b)\d+"
        ),
        "message": (
            "BLOCKED: ORDER BY <date_col> DESC LIMIT N without NULLS LAST can "
            "return a row with a NULL date as if it were the most recent — "
            "PostgreSQL sorts NULL as the largest value in a DESC sort by "
            "default. Add NULLS LAST, e.g.: "
            "ORDER BY asch.follow_up_date DESC NULLS LAST LIMIT 1."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_ca_for_observation_wrong_path",
        "type":        BLOCK,
        "code":        "AUDIT_CA_OBSERVATION_WRONG_PATH",
        "description": (
            "When fetching CAs for a specific observation text (ILIKE), must use "
            "observation_id -> submission_id path, not the whole-report join. "
            "Audit's mirror of inspection's block_ma_for_observation_wrong_path."
        ),
        # Fires when: ai_audit_answers ILIKE filter present + ACA joined via
        # audit_report (aca.audit_id = arep.id) but observation_id path is absent.
        "sql_pattern": (
            r"(?=.*\bai_audit_answers\b)"                    # ai_audit_answers present
            r"(?=.*\bilike\b)"                                # ILIKE filter present
            r"(?=.*\baudit_corrective_action\b)"              # ACA present
            r"(?=.*\baca\.audit_id\s*=\s*arep\.id\b)"         # wrong path: via whole report
            r"(?!.*\baca\.observation_id\b)"                  # correct path absent
        ),
        "message": (
            "WRONG PATH: When filtering CAs by observation text (ILIKE), join via "
            "aaa.submission_id = aca.observation_id::varchar — NOT via aca.audit_id = arep.id. "
            "The report join returns ALL CAs for the whole audit; the observation_id join "
            "returns only the CA directly linked to that specific observation."
        ),
        "hard_fail": True,
    },
    {
        "id":          "audit_block_degenerate_label_groupby",
        "type":        BLOCK,
        "code":        "AUDIT_DEGENERATE_LABEL_GROUPBY",
        "description": (
            "GROUP BY aq.label is meaningless when WHERE already pins aq.label to one "
            "literal value — collapses every row into a single bucket instead of grouping "
            "by the actual answer content (e.g. which specific observation recurred)."
        ),
        # Caught live 2026-07-29: "which observation triggered the most corrective
        # actions" generated WHERE aq.label = 'Findings / Remarks' ... GROUP BY aq.label
        # — every matching row already has that label, so GROUP BY aq.label produces
        # exactly one group; COUNT(*) just counts total matching rows ("Findings /
        # Remarks: 5"), not which specific observation recurred at all.
        "sql_pattern": (
            r"(?=.*\baq\.label\s*=\s*'[^']+')"
            r"(?=.*\bgroup\s+by\b[^;]*\baq\.label\b)"
        ),
        "message": (
            "WRONG GROUP BY: aq.label is already pinned to one value by the WHERE filter, "
            "so 'GROUP BY aq.label' collapses everything into a single meaningless group. "
            "Group by the actual answer instead: GROUP BY aaa.id, aaa.answer_text, arep.audit_id."
        ),
        "hard_fail": True,
    },
]


AUDIT_WARN_RULES: list[dict] = [
    {
        "id":          "audit_warn_ca_draft_literal",
        "type":        WARN,
        "code":        "AUDIT_CA_DRAFT_PLACEHOLDER",
        "description": "corrective_action_id = 'CA_DRAFT' is a placeholder shared by every CA pending adequacy approval (adequacy_status != 'APPROVE'), not a unique identifier — confirmed live this is independent of submission status.",
        "sql_pattern": r"corrective_action_id\s*=\s*'CA_DRAFT'",
        "message": (
            "WARNING: 'CA_DRAFT' is a literal placeholder string shared by every "
            "audit_corrective_action row pending adequacy approval (adequacy_status != "
            "'APPROVE') — it is NOT a unique ID, and NOT simply about submission status "
            "(a SUBMITTED-status CA can still be 'CA_DRAFT'). A lookup filtering on this "
            "literal will match many unrelated rows."
        ),
    },
    {
        "id":          "audit_warn_role_name_auditor_filter",
        "type":        WARN,
        "code":        "AUDIT_ROLE_NAME_UNRELIABLE",
        "description": "Filtering by role.name ILIKE '%auditor%' to enumerate who does audit work is unreliable — several assigned auditors hold unrelated roles.",
        "sql_pattern": r"role\.name\s+ILIKE\s+'%[Aa]udit",
        "message": (
            "WARNING: role.name ILIKE '%auditor%' misses people who are functionally assigned "
            "as auditor_id/auditee_id on real schedules but hold a different role label "
            "(HSE Officer, Internal Inspector, Super Admin, FM Manager all seen live). Query "
            "audit_schedule.auditor_id/.auditee_id directly for 'who does audit work' questions."
        ),
    },
]


AUDIT_INJECT_RULES: list[dict] = [
    {
        "id":             "audit_inject_auditor_assignment",
        "type":           INJECT,
        "priority":       80,
        "description":    "Who is the auditor/auditee — assignment column, not role table.",
        "question_pattern": r"(?i)\bwho\s+(is|are)\s+the\s+audit(?:or|ee)s?\b|\blist\s+(all\s+)?audit(?:or|ee)s?\b|\baudit(?:or|ee)s?\s+assign",
        "hint": (
            "\nAUDITOR/AUDITEE ASSIGNMENT — do NOT filter role.name ILIKE '%auditor%' to "
            "enumerate 'who is doing audit work': several people assigned as auditor_id hold "
            "unrelated roles (HSE Officer, Internal Inspector, Super Admin, FM Manager — "
            "confirmed live). Query the assignment column directly:\n"
            "  'who is the auditor' / 'who conducted this'  → audit_schedule.auditor_id → users.id\n"
            "  'who is the auditee' / 'who was audited'      → audit_schedule.auditee_id → users.id\n"
            "  Only join role/user_roles for an explicit role-management question, never to "
            "enumerate audit-work assignment.\n"
        ),
    },
    {
        "id":             "audit_inject_facility_from_report_or_ca",
        "type":           INJECT,
        "priority":       75,
        "description":    "Facility/project for a report or corrective action — multi-hop via audit_schedule, audit_report has no facility_id.",
        "question_pattern": r"(?i)(?:audit\s+)?(?:report|corrective\s+action)s?\b.{0,40}\b(facility|site|project)\b|\b(facility|site|project)\b.{0,40}(?:audit\s+)?(?:report|corrective\s+action)",
        "hint": (
            "\nFACILITY/PROJECT FROM REPORT OR CORRECTIVE ACTION — audit_report has NO "
            "facility_id/project_id column of its own (confirmed live, unlike inspection):\n"
            "  From audit_report:            .schedule_id → audit_schedule.id → .facility_id/.project_id\n"
            "  From audit_corrective_action: .audit_id → audit_report.id → .schedule_id → "
            "audit_schedule.id → .facility_id\n"
        ),
        "required_tables": ["audit_schedule"],
    },
    {
        "id":             "audit_inject_auditor_auditee_from_ca",
        "type":           INJECT,
        "priority":       75,
        "description":    "Auditor/auditee for a corrective action — same multi-hop trap as facility, caught live (model's retry silently returned 0 rows by joining aca.audit_id directly to audit_schedule.id).",
        "question_pattern": r"(?i)(?:audit(?:or|ee)s?)\b.{0,40}\bcorrective\s+actions?\b|\bcorrective\s+actions?\b.{0,40}\b(?:audit(?:or|ee)s?)\b",
        "hint": (
            "\nAUDITOR/AUDITEE FROM CORRECTIVE ACTION — audit_corrective_action.audit_id is a UUID "
            "FK to audit_report.id (NOT to audit_schedule, and NOT to audit_report.audit_id — that's "
            "a different column, the VARCHAR display code, despite the shared name). The auditor/"
            "auditee only exist on audit_schedule, reached via audit_report as a mandatory "
            "intermediate hop:\n"
            "  aca.audit_id → audit_report.id → audit_report.schedule_id → audit_schedule.id "
            "→ .auditor_id/.auditee_id → users.id\n"
            "NEVER join audit_corrective_action directly to audit_schedule — audit_report must be "
            "in the middle, even though skipping it doesn't error (both sides happen to be UUID-"
            "typed) — it just silently returns 0 rows because the join target is the wrong table.\n"
        ),
        "required_tables": ["audit_report", "audit_schedule"],
        "forbidden_patterns": [
            r"\baudit_id\s*=\s*\w+\.audit_id\b",
            r"\baca\.audit_id\s*=\s*asch\.id\b",
        ],
    },
    {
        "id":             "audit_inject_audit_type_resolution",
        "type":           INJECT,
        "priority":       75,
        "description":    "Audit type of a report/schedule — three/two-hop via audit_plan, audit_report.audit_type_id is dead.",
        "question_pattern": r"(?i)\baudit\s+type\b|\b(internal|external|statutory|third.?party|second\s+party|first\s+party)\s+audit",
        "hint": (
            "\nAUDIT TYPE RESOLUTION — audit_report.audit_type_id is DEAD (always NULL, "
            "confirmed live). The real path is via audit_plan:\n"
            "  From audit_report:  .schedule_id → audit_schedule.id → .audit_plan_id → "
            "audit_plan.id → .audit_type_id → audit_type.id\n"
            "  From audit_schedule: .audit_plan_id → audit_plan.id → .audit_type_id → audit_type.id\n"
        ),
        "required_tables": ["audit_plan"],
        "forbidden_patterns": [r"(?:audit_report|arep)\.audit_type_id"],
    },
    {
        "id":             "audit_inject_review_workflow",
        "type":           INJECT,
        "priority":       78,
        "description":    "Approval/pending-review questions across plan/portfolio/schedule/corrective-action route through the shared polymorphic review table.",
        "question_pattern": (
            r"(?i)\bpending\s+(approval|review)\b|\bawaiting\s+(approval|review)\b"
            r"|\bwho\s+approved\b|\bwho\s+reviewed\b|\blevel\s*[12]\s+review\b"
            r"|\badequacy\b|\bimplementation\s+(approval|status|review)\b"
            r"|\bapproved\s+by\b|\breturned\s+for\s+modification\b"
        ),
        "hint": (
            "\nAPPROVAL/REVIEW WORKFLOW — the 'review' table is a shared, POLYMORPHIC "
            "multi-level approval log (also used by inspection — audit_report itself is NOT "
            "reviewed this way). review.entity_id has NO real FK — ALWAYS filter "
            "WHERE entity_type = '<TYPE>' AND entity_id = <target>.id. Confirmed entity_type "
            "values for audit:\n"
            "  'AUDIT_PLAN'               → entity_id = audit_plan.id                (2 levels)\n"
            "  'PORTFOLIO'                → entity_id = auditor_portfolio.id         (1 level)\n"
            "  'SCHEDULE'                 → entity_id = audit_schedule.id            (auditee's "
            "accept/reschedule step on the scheduled visit — 1 level, rare 2nd)\n"
            "  'AUDIT_CORRECTIVE_ACTION'  → entity_id = audit_corrective_action.id, split by "
            "review_type: 'ADEQUACY' (auditor-side check) and 'IMPLEMENTATION' "
            "(two-level: auditee-side then auditor-side sign-off)\n"
            "review.status ∈ {'PENDING','COMPLETED'}, review.action ∈ {'APPROVE',"
            "'RETURN_FOR_MODIFICATION', NULL while pending}.\n"
            "NEVER hardcode a specific expected reviewer ROLE NAME as a filter or correctness "
            "check — role names are dynamic, permission-driven labels, not fixed identities. "
            "Only join review→users→user_roles→role when the question explicitly asks WHO "
            "reviewed, never to gate whether a row is 'correctly' approved.\n"
        ),
    },
    {
        "id":             "audit_inject_impact_compliance_risk",
        "type":           INJECT,
        "priority":       72,
        "description":    "impact/compliance/risk_level are three separate identically-shaped lookups on audit_corrective_action.",
        "question_pattern": r"(?i)\bimpact\b|\bcompliance\b|\brisk\b|\bdetails?\b|\bcorrection\b|\bcaus\w*\b",
        "hint": (
            "\nTHREE SEPARATE CLASSIFICATION AXES on audit_corrective_action — identically "
            "shaped (id, name, scale, short_code, is_positive_impact) but different FKs, "
            "answering different questions:\n"
            "  risk_level_id → risk_level   — 'how severe overall?' "
            "(No Active Risk < Low < Medium < High)\n"
            "  compliance_id → compliance   — 'was this compliant, how much of a gap?'\n"
            "  impact_id     → impact       — 'what business dimension does this affect?'\n"
            "  (confirmed live: risk_level, compliance, and impact currently have NO overlapping\n"
            "  label strings at all — risk_level: High/Low/Medium/No Active Risk; compliance:\n"
            "  Best Practice/Compliance/Exceeding Requirements/Non Confirmity - Major/Non\n"
            "  Confirmity - Minor/Not Applicable/Observation/Opportunity for Improvement;\n"
            "  impact: Client Relationship/Environmental/Financial/Health & Safety (HS)/Legal /\n"
            "  Regulatory/Operational - */Quality/Reputational.)\n"
            "SPELLING TRAP: compliance stores 'Non Confirmity' (typo, missing 'o'), NOT\n"
            "'Non Conformity' — confirmed live, ILIKE '%conformity%' returns zero rows.\n"
            "Pick the source table from what the question is actually asking about (severity vs.\n"
            "compliance gap vs. business dimension), not from label text alone.\n"
            "NEVER INNER JOIN more than one of these three axes in the same query unless the "
            "question explicitly asks about more than one — confirmed live: a 'show details' "
            "follow-up on 6 already-known high-risk corrective actions INNER-joined risk_level + "
            "impact + compliance together, and silently returned 0 rows (instead of the 6 rows "
            "with their correction/cause text) because impact_id/compliance_id were NULL on those "
            "rows. For a generic 'show details' / 'more info' / 'what's the correction or cause' "
            "follow-up, reuse ONLY the axis already established by the prior filter (e.g. "
            "risk_level, if that's what the conversation already filtered on) and leave the other "
            "two out entirely — aca.cause, aca.correction, and aca.corrective_action are plain "
            "text columns on audit_corrective_action itself and need NO extra axis join to "
            "select. If you do want to display an extra axis without filtering on it, use LEFT "
            "JOIN, never JOIN/INNER JOIN.\n"
        ),
    },
    {
        "id":             "audit_inject_correction_vs_corrective_action",
        "type":           INJECT,
        "priority":       73,
        "description":    "cause/correction/corrective_action are three separate TEXT columns on audit_corrective_action, easily conflated by name.",
        "question_pattern": r"(?i)\bcorrective\s+actions?\b|\bcorrection\b|\bcaus\w*\b|\bdetails?\b",
        "hint": (
            "\nCAUSE vs CORRECTION vs CORRECTIVE_ACTION — THREE SEPARATE TEXT columns on "
            "audit_corrective_action, confirmed live via DB introspection (do not assume only "
            "one exists just because the others sound similar, and never substitute one for "
            "another):\n"
            "  aca.cause              — why it happened\n"
            "  aca.correction         — the immediate fix already applied\n"
            "  aca.corrective_action  — the (root-cause/longer-term) action proposed to prevent "
            "recurrence — a DIFFERENT value from aca.correction, not a synonym for it\n"
            "  aca.corrective_action_id — NOT text at all; the VARCHAR display code (e.g. "
            "'2026/06/ST041/AUD003_CA004', or literal 'CA_DRAFT' pre-approval) — never confuse "
            "this with the aca.corrective_action text column despite the near-identical name.\n"
            "A question asking for 'the corrective action' (the proposed action) means "
            "aca.corrective_action, NOT aca.correction — confirmed live: a real follow-up asking "
            "for 'the corrective action text' got back aca.correction instead, because "
            "aca.corrective_action was missing from the schema knowledge entirely. When a "
            "question is ambiguous ('show details', 'what's the fix') and isn't clearly asking "
            "for only one of the three, select all three rather than guessing which one was "
            "meant.\n"
        ),
    },
    {
        "id":             "audit_inject_progress_stage",
        "type":           INJECT,
        "priority":       70,
        "description":    "'at risk'/behind schedule questions map to progress_stage, mirroring inspection's ica.progress_stage convention.",
        "question_pattern": r"(?i)\b(at\s+risk|behind\s+schedule|on\s+track)\b.{0,30}corrective\s+action|corrective\s+action.{0,30}\b(at\s+risk|behind\s+schedule|on\s+track)\b",
        "hint": (
            "\n'at risk' / 'behind schedule' / 'on track' on audit_corrective_action → "
            "progress_stage (schedule health), NOT risk_level or status. Values: NOT_STARTED, "
            "ON_TRACK, BEHIND, AT_RISK, COMPLETED, CLOSE_WITH_DEFERRED. NULL progress_stage = "
            "not started — do NOT OR it into an 'AT_RISK' filter, they are separate buckets.\n"
        ),
    },
]

# First audit AUTOCORRECT rules — implementation_status/adequacy_status are
# both 3-value enums (PENDING_FOR_REVIEW, APPROVE, RETURN_FOR_MODIFICATION —
# confirmed live via SELECT DISTINCT). "APPROVE", not "APPROVED" — a natural
# English-phrasing trap since users ask for "approved" rows. Confirmed live:
# a real query asking for "implementation status approved" generated
# aca.implementation_status = 'APPROVED', silently matching 0 of 5 real
# APPROVE rows (execution succeeded, no error — just a wrong, empty answer).
# Silent regex fix rather than a BLOCK+retry — this is an unambiguous,
# mechanical typo-class error with no legitimate reading, so autocorrecting
# is strictly better than spending a retry round-trip on it.
AUDIT_AUTOCORRECT_RULES: list[dict] = [
    {
        "id":          "audit_autocorrect_implementation_status_approved",
        "type":        AUTOCORRECT,
        "description": "implementation_status enum value is 'APPROVE', not 'APPROVED'.",
        "sql_pattern": r"(\bimplementation_status\s*(?:=|!=|<>|IN\s*\(|NOT\s+IN\s*\()\s*'?)APPROVED\b",
        "replacement": r"\1APPROVE",
        "message":     "[autocorrect] Fixed implementation_status enum value: 'APPROVED' -> 'APPROVE'",
    },
    {
        "id":          "audit_autocorrect_adequacy_status_approved",
        "type":        AUTOCORRECT,
        "description": "adequacy_status enum value is 'APPROVE', not 'APPROVED' — same trap, same enum shape.",
        "sql_pattern": r"(\badequacy_status\s*(?:=|!=|<>|IN\s*\(|NOT\s+IN\s*\()\s*'?)APPROVED\b",
        "replacement": r"\1APPROVE",
        "message":     "[autocorrect] Fixed adequacy_status enum value: 'APPROVED' -> 'APPROVE'",
    },
    {
        "id":          "audit_autocorrect_desc_limit_nulls_last",
        "type":        AUTOCORRECT,
        "description": (
            "Mechanically inserts NULLS LAST on ORDER BY <date_col> DESC for the "
            "same date columns audit_block_desc_limit_missing_nulls_last blocks on "
            "— added 2026-07-29 after DeepSeek ignored that BLOCK rule's explicit "
            "'add NULLS LAST' retry message 3 times in a row on 'show the most "
            "recently conducted audit' before the system force-executed. Runs via "
            "clean_sql() -> _registry_autocorrects() BEFORE validate(), so the BLOCK "
            "rule's negative lookahead (?!\\s+NULLS\\s+LAST) no longer matches by "
            "the time validation runs — it never even fires."
        ),
        "sql_pattern": (
            r"\bORDER\s+BY\s+(\w+\.(?:follow_up_date|submitted_on|target_close_out_date|"
            r"close_on|completed_on|due_date))\s+DESC\b(?!\s+NULLS\s+LAST)"
        ),
        "replacement": r"ORDER BY \1 DESC NULLS LAST",
        "message":     "[autocorrect] Added NULLS LAST to ORDER BY <date_col> DESC",
    },
]
