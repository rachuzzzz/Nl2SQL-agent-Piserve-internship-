"""
SQL Validator — production-grade refactor.

Architecture
============
  clean_sql()         — formatting normalization only; minimal semantic mutation
  validate()          — returns ValidationResult; HARD_FAIL vs SOFT_WARN separated
  validate_semantic() — SOFT_WARN by default (advisory, never blocks/retries); one
                        allowlisted HARD_FAIL exception exists, see semantic.py's
                        _ENFORCED_GLOSSARY_IDS (confirmed live 2026-08-07)
  ValidatorStats      — lightweight rule-hit counters for observability

Hard fails block execution and appear in retry prompts (concise codes only).
Soft warns are logged for observability but are NEVER sent to the LLM.

This separation eliminates:
  - validator-induced semantic drift on retries
  - verbose warning prose polluting retry prompts
  - false-positive blocks on valid complex SQL patterns

SQLValidator itself is composed from CleanerMixin/ShapeMixin/SemanticMixin/
ScopeMixin (see cleaner.py/shape.py/semantic.py/scope.py) -- this file has
__init__, the registry-consumption glue, and validate() (the main HARD_FAIL
entry point, which stays whole rather than being split further).
"""

import re
import logging

try:
    import sqlglot
    import sqlglot.expressions as exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False

logger = logging.getLogger("inspectai.sql")

from src.sql.validation.models import ValidationIssue, ValidationResult, ValidatorStats
from src.sql.validation.ast_checks import (
    _strip_nested_parens,
    _check_undefined_table_aliases,
    _check_ir_id_in_select_ast,
    _check_uuid_pk_in_select_ast,
    _check_display_code_uuid_compare_ast,
    _check_uuid_fk_compared_to_display_code_ast,
    _check_display_code_wrong_table_ast,
    _check_subquery_uuid_display_code_mismatch_ast,
    _check_uuid_leaks_ast,
    _check_uuid_leaks_regex,
    _check_ungrouped_select_columns_ast,
    _extract_top_level_limit,
)
from src.sql.validation.cleaner import CleanerMixin
from src.sql.validation.identity import IdentityMixin
from src.sql.validation.autocorrect_intent import IntentAutocorrectMixin
from src.sql.validation.shape import ShapeMixin
from src.sql.validation.semantic import SemanticMixin
from src.sql.validation.scope import ScopeMixin


class SQLValidator(CleanerMixin, IdentityMixin, IntentAutocorrectMixin, ShapeMixin, SemanticMixin, ScopeMixin):
    """
    Two-tier SQL validation.

    HARD_FAIL: blocks execution + concise code in retry prompt
    SOFT_WARN: logged only, invisible to LLM

    Clean separation means:
      - LLM retries get one focused error, not a wall of mixed warnings
      - Soft heuristics cannot cause retry cascades
      - Aggregation/window function SQL is never blocked by routing heuristics
    """


    # Columns that do not exist on ai_* tables — keyed by table → {col: hint}
    HALLUCINATED_AI_COLUMNS: dict[str, dict[str, str]] = {
        "ai_questions": {
            "question_label": "use aq.label instead",
            "form_name":      "use module_name (ai_questions has no form_name column)",
            "question_type":  "use entity_type instead",
            "page_name":      "column does not exist on ai_questions",
            "required":       "column does not exist on ai_questions",
            "active":         "ai_questions has no active column — no soft-delete on this table",
        },
        "ai_answers": {
            "answer_value":  "use answer_text (text) or answer_numeric (numeric)",
            "form_name":     "use module_name (ai_answers has no form_name column)",
            "label":         "ai_answers has no label column — JOIN ai_questions aq ON aa.element_id = aq.element_id, then filter aq.label = '...'",
            "active":        "ai_answers has no active column — filter via inspection_report: ir.active = true AND ir.status != 'DRAFT'",
        },
        # ai_audit_answers — audit's own flat answer table (confirmed live and populated
        # 2026-07-29, 445 rows — supersedes an earlier assumption it was empty). Columns
        # were renamed from a copy-pasted inspection-flavored schema: inspection_report_id
        # -> audit_report_id, inspection_id -> audit_id, inspector_user_id -> auditor_user_id.
        "ai_audit_answers": {
            "answer_value":       "use answer_text (text) or answer_numeric (numeric)",
            "form_name":          "use module_name (ai_audit_answers has no form_name column)",
            "label":              "ai_audit_answers has no label column — JOIN ai_questions aq ON aaa.element_id = aq.element_id, then filter aq.label = '...'",
            "active":             "ai_audit_answers has no active column — filter via audit_report: arep.active = true AND arep.status != 'DRAFT'",
            "inspection_report_id": "renamed to audit_report_id (2026-07-29) — use aaa.audit_report_id",
            "inspection_id":         "renamed to audit_id (2026-07-29) — use aaa.audit_id",
            "inspector_user_id":     "renamed to auditor_user_id (2026-07-29) — use aaa.auditor_user_id",
        },
        "inspection_schedule": {
            # deepseek training bias: uses scheduled_date (past participle) instead of schedule_date
            "scheduled_date": "column is 'schedule_date' not 'scheduled_date' on inspection_schedule",
            # inspection_id/inspection_report_id do not exist on inspection_schedule — the FK lives
            # on the report instead: inspection_report.schedule_id = isched.id
            "inspection_id":         "inspection_schedule has no inspection_id — reach the report via "
                                      "inspection_report.schedule_id = isched.id",
            "inspection_report_id":  "inspection_schedule has no inspection_report_id — reach the report via "
                                      "inspection_report.schedule_id = isched.id",
        },
    }


    # Columns that do not exist on inspection_report — must be resolved via JOIN
    # col → join hint (for logs only, not retry prompts)
    HALLUCINATED_IR_COLUMNS: dict[str, str] = {
        "facility_name":   "JOIN facility fac ON ir.facility_id = fac.id → fac.name",
        "inspector_name":  "JOIN users u ON ir.inspector_user_id = u.id → u.first_name || ' ' || u.last_name",
        "inspectee_name":  "JOIN users u ON ir.inspectee_user_id = u.id → u.first_name || ' ' || u.last_name",
        "client_name":     "JOIN client cl ON ir.client_id = cl.id → cl.name",
        "project_name":    "JOIN project proj ON ir.project_id = proj.id → proj.name",
        "type_name":       "JOIN inspection_type it ON ir.inspection_type_id = it.id → it.name",
        "inspection_type": "JOIN inspection_type it ON ir.inspection_type_id = it.id → it.name",
        # Confirmed live 2026-07-29 (qid=fe1426e8): "was there any inspection at
        # aldar hQ" hallucinated ir.inspection_type_name directly — a real Postgres
        # "column does not exist" error, self-corrected on retry, but the existing
        # "inspection_type" entry above doesn't catch this since the word-boundary
        # match requires an exact column name and "_name" continues the word.
        "inspection_type_name": "JOIN inspection_type it ON ir.inspection_type_id = it.id → it.name",
        # Quarter-comparison ghost columns — deepseek hallucinates these when
        # Qwen writes bad schema_hints like AVG(ir.current_qtr_score)
        "current_qtr_score": "does not exist — compute with CASE WHEN ir.submitted_on >= date_trunc('quarter', NOW()) THEN ir.inspection_score END",
        "last_qtr_score":    "does not exist — compute with CASE WHEN ir.submitted_on >= date_trunc('quarter', NOW()-INTERVAL '3 months') AND ir.submitted_on < date_trunc('quarter', NOW()) THEN ir.inspection_score END",
        "score_improvement": "does not exist as a column — compute as this_q - last_q in a CTE",
    }


    # Columns that do not exist on inspection_corrective_action.
    # deepseek hallucinates these when given vague schema_hints about risk/deferral.
    HALLUCINATED_ICA_COLUMNS: dict[str, str] = {
        # Deferral ghost columns — 'times_deferred' was seen in ORDER BY in trace
        "times_deferred":     "does not exist — use COUNT(*) GROUP BY cause or fac.name to count deferrals",
        # Risk-level ghost columns — risk is via risk_level_id FK, not a direct column.
        # NOTE: risk_level and impact are also TABLE names — only catch qualified refs (ica.risk_level)
        # capex_status DOES exist (unsuppressed) — do not flag it
        "adequacy_status":    "does not exist — adequacy workflow not active",
        "risk_level":         "does not exist as a column — risk_level_id is a UUID FK; JOIN risk_level rl ON ica.risk_level_id = rl.id → rl.name",
        # ICA columns that do not exist — common hallucinations
        "due_date":           "does not exist on inspection_corrective_action — use target_close_out_date",
        "actual_date":        "does not exist on inspection_corrective_action — use completed_on (for closed rows) or target_close_out_date (for deadline)",
        # mitigative_action_uuid is on po_details/pr_details — only block if qualified as ica.mitigative_action_uuid
        # "po.mitigative_action_uuid = ica.id" is CORRECT and must NOT be blocked
        "ica.mitigative_action_uuid": "does not exist on inspection_corrective_action — use po_details.mitigative_action_uuid = ica.id",
        "completion_date":    "does not exist on inspection_corrective_action — use completed_on (actual closure timestamp, populated for CLOSED rows)",
        "resolution_date":    "does not exist on inspection_corrective_action — use completed_on (actual closure timestamp for CLOSED rows)",
        "impact":             "does not exist as a column on ica — impact data is in ai_answers: JOIN ai_questions aq ON aa.element_id = aq.element_id WHERE aq.label = 'Impact'",
        # Other ghost columns seen in traces
        "corrective_action_description": "does not exist — use ica.corrective_action (the text field)",
    }


    # Subset of ICA ghost columns that are also table names — only check qualified refs.
    # If we check unqualified, we'd false-positive on "JOIN risk_level rl" clauses.
    _ICA_QUALIFIED_ONLY_COLS: frozenset = frozenset({"risk_level", "impact"})


    # Columns that do not exist on inspection_cycle. It has NO human-readable
    # name column — confirmed 2026-06-12 after a real DB error (see
    # core/business_rules.py block_inspection_cycle_name, which documents the
    # same fact but is never invoked at runtime — this is the live check).
    # inspection_cycle columns: id, status, start_date, end_date, due_date,
    # send_notification_to_inspectors, created_on, modified_on.
    HALLUCINATED_IC_COLUMNS: dict[str, str] = {
        "name":       "does not exist on inspection_cycle — cycles have no name, identify them by "
                       "ic.start_date / ic.end_date / ic.status instead (e.g. 'Jan 2026 cycle')",
        "cycle_name": "does not exist on inspection_cycle — cycles have no name, identify them by "
                       "ic.start_date / ic.end_date / ic.status instead (e.g. 'Jan 2026 cycle')",
        "title":      "does not exist on inspection_cycle — cycles have no name, identify them by "
                       "ic.start_date / ic.end_date / ic.status instead (e.g. 'Jan 2026 cycle')",
        "code":       "does not exist on inspection_cycle — cycles have no name, identify them by "
                       "ic.start_date / ic.end_date / ic.status instead (e.g. 'Jan 2026 cycle')",
    }


    # audit_cycle's audit-domain counterpart to HALLUCINATED_IC_COLUMNS above —
    # same underlying fact (no human-readable name column, confirmed live via
    # SQLAlchemy inspection: id, active, created_by/on, modified_by/on, due_date,
    # end_date, send_notification_to_auditors, start_date, status), caught live
    # in eval_audit.py's PLAN-08 run burning 7 LLM calls / 21s repeatedly
    # confabulating ac.name / acyc.name before finally dropping it.
    HALLUCINATED_AC_COLUMNS: dict[str, str] = {
        "name":       "does not exist on audit_cycle — cycles have no name, identify them by "
                       "acyc.start_date / acyc.end_date / acyc.status instead",
        "cycle_name": "does not exist on audit_cycle — cycles have no name, identify them by "
                       "acyc.start_date / acyc.end_date / acyc.status instead",
        "title":      "does not exist on audit_cycle — cycles have no name, identify them by "
                       "acyc.start_date / acyc.end_date / acyc.status instead",
        "code":       "does not exist on audit_cycle — cycles have no name, identify them by "
                       "acyc.start_date / acyc.end_date / acyc.status instead",
    }


    # audit_plan does NOT have audit_universe_id — that FK only lives on
    # audit_schedule. Caught live in eval_audit.py's PLAN-02 run: the model
    # joined audit_universe directly off audit_plan, got a real Postgres
    # UndefinedColumn error, and needed a get_schema round-trip to recover.
    HALLUCINATED_AP_COLUMNS: dict[str, str] = {
        "audit_universe_id": "does not exist on audit_plan — audit_universe_id only lives on "
                              "audit_schedule. Either join via audit_schedule.audit_plan_id, or "
                              "(for a plan's subject area) use audit_plan.category_id → "
                              "audit_category instead.",
    }


    # auditor_portfolio has NO display-name column at all (confirmed live:
    # id, active, created_by/on, modified_by/on, status, submitted_date,
    # audit_cycle_id, user_id) — identify a portfolio by its owning auditor
    # instead (JOIN users on po.user_id). Caught live in eval_audit.py's
    # PORT-03 run: DeepSeek hallucinated po.portfolio_name TWICE — once before
    # and once AFTER a get_schema round-trip confirmed the real column list —
    # despite a correct in-context seed example showing the right pattern.
    # This is a different trap from PORTFOLIO_AP_ALIAS_COLLISION (audit_rules.py),
    # which only fires when the alias itself is wrong ('ap' instead of 'po');
    # this one catches the hallucinated column regardless of alias.
    HALLUCINATED_PO_COLUMNS: dict[str, str] = {
        "name":           "does not exist on auditor_portfolio — it has no display-name column at "
                           "all, identify it by its owning auditor instead (JOIN users u ON "
                           "po.user_id = u.id)",
        "portfolio_name": "does not exist on auditor_portfolio — it has no display-name column at "
                           "all, identify it by its owning auditor instead (JOIN users u ON "
                           "po.user_id = u.id)",
        "title":          "does not exist on auditor_portfolio — that's audit_plan.title, a "
                           "different table. Identify it by its owning auditor instead (JOIN users "
                           "u ON po.user_id = u.id)",
        "code":           "does not exist on auditor_portfolio — identify it by its owning auditor "
                           "instead (JOIN users u ON po.user_id = u.id)",
        "submitted_on":   "does not exist on auditor_portfolio — the real column is submitted_date "
                           "(confirmed live: eval_audit.py's APPR-10 run hit a live UndefinedColumn "
                           "error on this after the ap->po alias was already fixed, burning a 3rd "
                           "LLM call). Use po.submitted_date instead.",
    }


    # Impossible JOIN conditions derived from join_graph — these columns
    # do not exist so the join can never be correct. Keyed by the bad pattern
    # (regex), value is the correct path.
    IMPOSSIBLE_JOIN_PATTERNS: list[tuple] = [
        # ipd.cycle_id — column does not exist on inspector_portfolio_details
        (r'\bipd\.cycle_id\b',
         'ipd.cycle_id does not exist. Correct path: '
         'ic→ip(ip.cycle_id=ic.id)→ipd(ipd.portfolio_id=ip.id)→fd'),
        # ic.inspection_cycle_id — column does not exist on inspection_cycle
        (r'\bic\.inspection_cycle_id\b',
         'ic.inspection_cycle_id does not exist on inspection_cycle'),
        # ir.cycle_id used as a join target on inspector_portfolio
        (r'\bipd\.id\s*=\s*ic\.id\b',
         'inspector_portfolio_details does not join directly to inspection_cycle. '
         'Use: ip.cycle_id = ic.id, then ipd.portfolio_id = ip.id'),
        # inspection_schedule.inspection_id / inspection_report_id — neither exists
        (r'\bisched\.inspection_id\b|\binspection_schedule\.inspection_id\b'
         r'|\bisched\.inspection_report_id\b|\binspection_schedule\.inspection_report_id\b',
         'inspection_schedule has no inspection_id or inspection_report_id column. '
         'The FK lives on the report instead: inspection_report.schedule_id = isched.id'),
        # ── observation → ai_answers cast direction ───────────────────────────
        # CORRECT cast: aa.submission_id = ica.observation_id::varchar
        #   (ica.observation_id is uuid, aa.submission_id is varchar — cast uuid to varchar)
        #   Verified in pgAdmin: aa.submission_id = ica.observation_id::varchar works correctly
        # WRONG cast caught below:
        #   ica.observation_id::text = aa.submission_id  (wrong side assignment)
        (r'\bica\.observation_id\s*::\s*text\s*=\s*aa\.submission_id\b',
         'Wrong cast — use: aa.submission_id = ica.observation_id::varchar'),
        # Audit's mirror of the observation cast trap above — same wrong-side-assignment shape.
        (r'\baca\.observation_id\s*::\s*text\s*=\s*aaa\.submission_id\b',
         'Wrong cast — use: aaa.submission_id = aca.observation_id::varchar'),
        (r'\baaa\.submission_id\s*::\s*uuid\s*=\s*aca\.observation_id\b',
         'Wrong cast direction — cast the UUID side, not submission_id: '
         'use aaa.submission_id = aca.observation_id::varchar'),
    ]


    # UUID FK columns — SOFT_WARN if selected without their lookup JOIN
    UUID_FK_COLUMNS: dict[str, tuple] = {
        "facility_id":            ("facility",           "fac",  "fac.name AS facility_name"),
        "inspector_user_id":      ("users",              "u",    "u.first_name || ' ' || u.last_name AS inspector_name"),
        "inspectee_user_id":      ("users",              "insp", "insp.first_name || ' ' || insp.last_name AS inspectee_name"),
        "client_id":              ("client",             "cl",   "cl.name AS client_name"),
        "project_id":             ("project",            "proj", "proj.name AS project_name"),
        "inspection_type_id":     ("inspection_type",    "it",   "it.name AS inspection_type_name"),
        "inspection_sub_type_id": ("inspection_sub_type","ist",  "ist.name AS subtype_name"),
        "module_id":              ("fb_modules",         "mod",  "mod.name AS module_name"),
        # inspection_cycle has no human-readable name — its "identity" for display
        # purposes is its date span + status, not a derived label from an unrelated
        # date column (e.g. TO_CHAR of an individual inspection's submitted_on).
        "cycle_id":               ("inspection_cycle",   "ic",   "ic.start_date, ic.end_date, ic.status AS cycle_status"),
    }


    # Audit-domain alias → display-column map for the generalized UUID-PK-in-
    # SELECT check (see _check_uuid_pk_in_select_ast) — confirmed live
    # 2026-07-28 that the reasoning model routinely adds bare `.id` to its
    # schema_hint across MULTIPLE audit tables (acyc.id, po.id, ap.id all
    # observed leaking into real answers in the same eval run), not just the
    # single inspection_report case this check originally covered. None
    # means "no display substitute exists at all" — confirmed live neither
    # audit_cycle nor auditor_portfolio has a name/code column.
    AUDIT_UUID_PK_DISPLAY_MAP: dict[str, str | None] = {
        "acyc": None,  # audit_cycle — identify by start_date/end_date/status, no id/name at all
        "po":   None,  # auditor_portfolio — identify by its owning auditor (JOIN users on user_id) or cycle+status
        "ap":   "ap.title",
        "asch": "asch.schedule_id (the human-readable varchar display code)",
        "arep": "arep.audit_id (the human-readable varchar display code)",
        "aca":  "aca.corrective_action_id (the varchar display code — 'CA_DRAFT' pre-approval)",
    }


    # Inverse of AUDIT_UUID_PK_DISPLAY_MAP, for the inverse failure shape —
    # see _check_display_code_uuid_compare_ast. Confirmed live: ordinal
    # resolution ("the 1st one in the list") correctly resolved a row's real
    # UUID from session context, but the reasoning model wrote
    # WHERE arep.audit_id = '<uuid>' instead of WHERE arep.id = '<uuid>' —
    # silently returned 0 rows instead of the real 4.
    #
    # Keyed by the EXACT (alias, column) pair, NOT the bare column name —
    # audit_id/schedule_id/inspection_id are each a genuine UUID foreign key
    # on a DIFFERENT table while being the display code on their OWN table
    # (audit_corrective_action.audit_id -> audit_report.id is a real UUID
    # FK; audit_report.schedule_id -> audit_schedule.id is a real UUID FK).
    # A bare column-name check would false-block those legitimate FK
    # comparisons, so this must stay alias-scoped.
    DISPLAY_CODE_UUID_COMPARE_MAP: dict[tuple[str, str], str] = {
        ("arep", "audit_id"): "arep.id",
        ("asch", "schedule_id"): "asch.id",
        ("aca", "corrective_action_id"): "aca.id",
        ("ica", "corrective_action_id"): "ica.id",
        ("ir", "inspection_id"): "ir.id",
    }


    # Inverse of the inverse: catches the OTHER failure shape on the same
    # (alias, column) family — a display-code-shaped string literal compared
    # against a column that is a genuine UUID FK on a DIFFERENT table,
    # merely sharing its bare name with the display-code column above.
    # See _check_uuid_fk_compared_to_display_code_ast. Confirmed live TWICE
    # (2026-07-29, 2026-08-04): the reasoning model wrote
    # WHERE aca.audit_id = '2026/06/ST0NN/AUDNNN' (a display code) instead of
    # joining audit_report and filtering arep.audit_id — aca.audit_id is a
    # UUID FK to arep.id, not a display code itself. Failed at DB execution
    # with psycopg2.errors.InvalidTextRepresentation, burning 2-4 retries.
    #
    # Keyed by the EXACT (alias, column) pair for the same false-positive
    # reason as DISPLAY_CODE_UUID_COMPARE_MAP above — a bare column-name
    # check would false-block legitimate uses of these columns as display
    # codes on their OWN table (arep.audit_id, asch.schedule_id,
    # ir.inspection_id are all legitimately compared to a display-code
    # string on their own alias).
    UUID_FK_DISPLAY_CODE_COMPARE_MAP: dict[tuple[str, str], tuple[str, str]] = {
        ("aca", "audit_id"): ("audit_report", "audit_id"),
        ("arep", "schedule_id"): ("audit_schedule", "schedule_id"),
        ("ica", "inspection_id"): ("inspection_report", "inspection_id"),
    }


    # A third failure shape on the same column family — see
    # _check_display_code_wrong_table_ast. Both prior maps only fire when a
    # UUID is involved on one side; this one catches two ordinary, well-
    # formed display codes from DIFFERENT tables being confused for each
    # other (e.g. a schedule's code fed into audit_report.audit_id).
    # Confirmed live 2026-08-04: ordinal resolution ("the second one")
    # correctly resolved a schedule's code, but the reasoning model then
    # filtered arep.audit_id with it instead of joining audit_schedule ->
    # audit_report — silently returned 0 rows (confirmed against the DB:
    # the schedule's real linked report has a genuinely different code).
    #
    # Keyed by each table's OWN display-code (alias, column) — the value is
    # the token that column's values always start their trailing segment
    # with (confirmed live via distinct real samples of each column:
    # '2026/08/ST040/AUD001', '2026/06/ST001/SCH002', '2026/07/ST006/INS001').
    OWN_DISPLAY_CODE_TOKEN_MAP: dict[tuple[str, str], str] = {
        ("arep", "audit_id"): "AUD",
        ("asch", "schedule_id"): "SCH",
        ("ir", "inspection_id"): "INS",
    }


    # Rule ids from core/business_rules.py's REGISTRY that validate()/clean_sql()
    # actually consume — an explicit allowlist so a new registry rule never
    # silently starts firing in production without deliberate opt-in here.
    #   Group A — pure consolidation of checks that used to be hand-written
    #             duplicates in this file (DML block, hallucinated-table names,
    #             dynamic fb_ tables, responsible/pending_with join, close_on
    #             arithmetic, missing-limit warn, hardcoded-year warn, plus the
    #             ICA/IR join and reserved-alias autocorrects).
    #   Group B — new enforcement: dormant registry-only rules turned on for
    #             the first time. block_po_pr_double_direct_join and
    #             block_capex_left_join_multiplication guard a confirmed
    #             production incident (2026-06-15, silent CAPEX/OPEX corruption
    #             via row fan-out) that had no guard anywhere until now.
    _CONSUMED_RULE_IDS: frozenset = frozenset({
        # Group A
        "block_dml", "block_table_ai_question_singular", "block_table_ai_answer_singular",
        "block_table_fb_users", "block_table_fb_submissions", "block_table_fb_answers",
        "block_table_fb_responses", "block_table_dynamic_fb", "block_responsible_fk_join",
        "block_close_on_arithmetic", "warn_no_limit", "block_hardcoded_year",
        "autocorrect_ica_ir_join_varchar_uuid", "autocorrect_ica_ir_join_reversed",
        "autocorrect_ica_ir_join_cast", "autocorrect_ica_ir_join_cast_typecast",
        "autocorrect_reserved_alias_is",
        # Group B
        "block_po_pr_double_direct_join", "block_capex_left_join_multiplication",
        "block_po_pr_join_to_ir", "block_mitigative_uuid_wrong_target",
        "block_expenditure_as_conversion", "block_inspector_not_in_pattern",
        # Group C — audit domain (get_audit_engine()'s registry, only ever
        # consulted when SQLValidator was constructed with domain="audit").
        # These existed in audit_rules.py but were never actually wired into
        # live validation until now — see AgentOrchestrator's SQLValidator
        # construction in orchestrator/loop.py.
        "audit_block_dead_schedule_status", "audit_block_dead_report_audit_type",
        "audit_block_report_missing_facility_project", "audit_block_responsible_join",
        "audit_block_ca_report_join_wrong_column", "audit_block_ca_schedule_direct_join",
        "audit_block_dead_portfolio_junctions", "audit_block_ca_draft_not_excluded",
        "audit_block_ca_existence_check_adequacy_filtered",
        "audit_block_conformity_spelling",
        "audit_warn_ca_draft_literal", "audit_warn_role_name_auditor_filter",
        "audit_block_review_completed_not_approved", "audit_block_portfolio_ap_alias",
        # Group Y — audit_schedule.schedule_status and audit_plan.status DRAFT
        # exclusion had zero enforcement (confirmed 2026-08-18) — same trap Group C's
        # audit_block_ca_draft_not_excluded guards for adequacy_status, now applied
        # to these two sibling columns.
        "audit_block_schedule_draft_not_excluded", "audit_block_plan_draft_not_excluded",
        # Group D — inspection domain's own review-table approval trap (shared
        # table, same status='COMPLETED' != approved issue as Group C's audit
        # rule above — see block_rules.py's inspection_block_review_completed_not_approved).
        "inspection_block_review_completed_not_approved",
        # Group E — audit domain's first AUTOCORRECT rules (implementation_status/
        # adequacy_status "APPROVED" -> "APPROVE" typo-class enum fix).
        "audit_autocorrect_implementation_status_approved",
        "audit_autocorrect_adequacy_status_approved",
        # Group F — review.entity_id must join to aca.id, never aca.corrective_action_id.
        "audit_block_review_entity_id_corrective_action_id",
        # Group G — a "show details"/follow-up query INNER-joining 2+ of the three
        # independent classification axes (risk_level/impact/compliance) that the
        # question didn't ask about silently zeroes the whole result (NULL FK on the
        # unasked axis) instead of showing the rows' correction/cause text.
        "audit_block_ca_multi_axis_inner_join",
        # Group H — ORDER BY <date_col> DESC LIMIT N without NULLS LAST can return a
        # NULL-dated row as "the most recent" (NULL sorts largest in DESC by default).
        "audit_block_desc_limit_missing_nulls_last",
        # Group I — CA-to-observation wrong-path guard, added 2026-07-29 alongside
        # the ai_audit_answers form-answer capability (see AUDIT_BUSINESS_RULES
        # "OBSERVATION -> CORRECTIVE ACTION").
        "audit_block_ca_for_observation_wrong_path",
        # Group J — GROUP BY aq.label is degenerate when WHERE already pins aq.label
        # to one literal value (caught live 2026-07-29: "which observation triggered
        # the most corrective actions" collapsed into one meaningless group instead
        # of grouping by the actual answer text). Same trap exists for inspection's
        # ai_answers/aq.label usage — see block_degenerate_label_groupby.
        "audit_block_degenerate_label_groupby", "block_degenerate_label_groupby",
        # Group L — mechanical NULLS LAST insertion (AUTOCORRECT), added 2026-07-29
        # so the fix lands before validate() ever runs, instead of relying on
        # DeepSeek to comply with the BLOCK rule's retry-prompt message.
        "audit_autocorrect_desc_limit_nulls_last",
        # Group K — inspection's own observation->MA wrong-path guard (block_rules.py's
        # block_ma_for_observation_wrong_path, the audit twin of Group A/audit_block_ca_
        # for_observation_wrong_path's Group I entry above) was defined but never added
        # here — silently dead until discovered 2026-07-29 while wiring up its audit twin.
        "block_ma_for_observation_wrong_path",
        # Group M — OPEN and OVERDUE are mutually exclusive status values; a
        # question asking only for "overdue" must not get status IN ('OPEN',
        # 'OVERDUE') copied in from the adjacent open+overdue seed example
        # (caught live 2026-07-29, qid=987552fd — see audit_seed_data.py's
        # now-separate 'list all corrective actions that are overdue' example
        # for the retrieval-side half of this fix).
        "audit_block_overdue_status_overbroad",
        # Group N — audit_schedule.audit_sub_category_id is NULL on ~28% of active
        # schedules (confirmed live 2026-07-29, qid=ff669ae7) — INNER JOIN silently
        # drops the whole row (auditor, category, everything) when only the
        # sub-category happens to be unset.
        "audit_block_sub_category_inner_join",
        # Group O — internal_stakeholder_id (per-CA) joined alongside auditor_id/
        # auditee_id (per-audit) without corrective_action_id in the SELECT list
        # fans out into rows that look like duplicate/inconsistent data (confirmed
        # live 2026-07-29, qid=4337464c).
        "audit_block_stakeholder_auditor_grain_mismatch",
        # Group Q — mechanical fix for review.entity_id joined to
        # aca.corrective_action_id (the VARCHAR display code) instead of aca.id
        # (the UUID PK) — same trap the Group F BLOCK rule above catches, but
        # this substitution has no legitimate alternate reading, so it's fixed
        # here before validate() ever runs instead of costing a retry.
        "audit_autocorrect_review_entity_id_corrective_action_id",
        "audit_autocorrect_corrective_action_id_entity_id_reversed",
        # Group P — audit domain had no DML/DDL guard of its own (confirmed live
        # 2026-07-29) — get_audit_engine()'s registry is entirely separate from
        # inspection's, so block_dml above never covered audit-domain queries.
        "audit_block_dml",
        # Group Q — 'closed_with_deferred' alias reused for a different concept
        # (closed-late timeliness) without renaming, confirmed live 2026-07-29
        # (qid=039d5f86) — see audit_seed_data.py's now-separate 'closed on time vs
        # closed late' example for the retrieval-side half of this fix.
        "audit_block_closed_with_deferred_mislabel",
        # Group R — audit_forms.status only ever holds DRAFT/SUBMITTED (confirmed
        # live 2026-07-29, qid=7f386051) — 'ACTIVE' is a hallucinated value that
        # silently matched zero rows instead of the correct count.
        "audit_block_forms_status_active_hallucination",
        # Group S — auditor_portfolio 'submitted' questions copying the full
        # 4-value CHECK constraint list (APPROVED, DRAFT, MODIFICATION, PENDING)
        # instead of narrowing to the submitted subset (confirmed live
        # 2026-08-03, qid=eee602ce/d73988fb) — same overbroad-status-list trap
        # shape as Group M's audit_block_overdue_status_overbroad.
        "audit_block_portfolio_submitted_status_overbroad",
        # Group T — sibling of Group I/K (ILIKE-text-search wrong-path guards), for the
        # ALREADY-RESOLVED-id direction instead: a uuid-guard retry re-inserting a dropped
        # aca.id/ica.id filter can silently swap the join to the whole-report/whole-
        # inspection path, returning unrelated form answers with no error (confirmed live
        # 2026-08-10, qid=5c9922a3 — see audit_block_ca_id_for_observation_wrong_path's
        # own docstring for the exact incident).
        "audit_block_ca_id_for_observation_wrong_path", "block_ma_id_for_observation_wrong_path",
        # Group U — self-join wrong-grain guard: combining two question labels (e.g. Risk
        # Level + Observation/Findings-Remarks) from the SAME repeatable entry requires
        # matching on submission_id, not report_id — confirmed live 2026-08-10 that two
        # existing seed examples got this wrong (see block_ai_answers_self_join_wrong_grain's
        # own docstring). No prior guard existed for this mistake shape in either domain.
        "block_ai_answers_self_join_wrong_grain", "audit_block_ai_audit_answers_self_join_wrong_grain",
        # Group V — a different failure mode of Group T's same uuid-guard-retry class: the
        # retry hardcoded a resolved uuid as a bare constant on the answers-table JOIN's ON
        # clause instead of using it as the *_corrective_action.observation_id value,
        # deleting the actual join predicate entirely (unintentional cross join, no error
        # raised — confirmed live 2026-08-10, qid=328ca155, see
        # audit_block_ai_audit_answers_constant_join_wrong_path's own docstring).
        "audit_block_ai_audit_answers_constant_join_wrong_path", "block_ai_answers_constant_join_wrong_path",
        # Group W — rev.entity_type on the audit domain must be exactly one of
        # AUDIT_PLAN/PORTFOLIO/SCHEDULE/AUDIT_CORRECTIVE_ACTION; any other literal
        # is UUID-typed like the real ones, so it silently matches zero rows
        # instead of erroring (confirmed live 2026-08-12, qid=4a85836b: "who
        # returned this for modification" generated entity_type = 'AUDIT_SCHEDULE',
        # a value that doesn't exist — the real one is bare 'SCHEDULE' — see
        # audit_rules.py's audit_block_review_entity_type_invalid_literal and
        # audit_seed_data.py's now-added 'who returned this audit schedule for
        # modification' example for the retrieval-side half of this fix).
        "audit_block_review_entity_type_invalid_literal",
        # Group X — sibling of Group W's trap, same silent-zero-rows shape,
        # different pair of confusable enum values: audit_plan.status's
        # RETURNED_FOR_MODIFICATION cross-contaminated onto
        # audit_schedule.schedule_status, whose real value is
        # RETURN_MODIFICATION (confirmed live 2026-08-12, qid=7a9a1309 — see
        # audit_rules.py's audit_block_schedule_status_invalid_literal and
        # audit_seed_data.py's now-added 'audit schedules returned for
        # modification' example for the retrieval-side half of this fix).
        "audit_block_schedule_status_invalid_literal",
    })


    # block_dml used to early-return non-retryable, before any other check ran.
    # Folding it into the generic registry-issue pass changes that control flow
    # (other checks can now also fire on the same SQL), but this keeps its
    # ValidatorStats/ValidationIssue observability semantics identical.
    _NON_RETRYABLE_CODES: frozenset = frozenset({"DML_BLOCKED"})


    def __init__(self, domain: str = "inspection"):
        from src.sql.rules import get_engine, get_audit_engine
        self.domain = domain
        self._rules = get_audit_engine() if domain == "audit" else get_engine()


    def _registry_block_warn_issues(self, sql: str, question: str = "", qid: str = "") -> list:
        """
        Consume business_rules.py's BLOCK/WARN registry rules, restricted to
        _CONSUMED_RULE_IDS. Applies the same table_guard / question_pattern
        gating RulesEngine precompiles onto each rule, but yields
        ValidationIssue objects so results slot into this class's existing
        pipeline (ValidatorStats, retry_message(), ...) unchanged.
        """
        issues: list[ValidationIssue] = []
        for rule in self._rules._block_rules:
            if rule["id"] not in self._CONSUMED_RULE_IDS:
                continue
            guard = rule.get("table_guard")
            if guard and not re.search(rf"\b{re.escape(guard)}\b", sql, re.IGNORECASE):
                continue
            q_re = rule.get("_q_re")
            if q_re and not (question and q_re.search(question)):
                continue
            if rule["_sql_re"].search(sql):
                code = rule["code"]
                retryable = code not in self._NON_RETRYABLE_CODES
                logger.debug("[VALID] qid=%s rule fired id=%s code=%s severity=HARD_FAIL",
                             qid, rule["id"], code)
                issues.append(ValidationIssue("HARD_FAIL", code, rule["message"], retryable=retryable))
                ValidatorStats.record(code, is_retry_trigger=retryable)
        for rule in self._rules._warn_rules:
            if rule["id"] not in self._CONSUMED_RULE_IDS:
                continue
            guard = rule.get("table_guard")
            if guard and not re.search(rf"\b{re.escape(guard)}\b", sql, re.IGNORECASE):
                continue
            if rule["_sql_re"].search(sql):
                code = rule["code"]
                logger.debug("[VALID] qid=%s rule fired id=%s code=%s severity=SOFT_WARN",
                             qid, rule["id"], code)
                issues.append(ValidationIssue("SOFT_WARN", code, rule["message"], retryable=False))
                ValidatorStats.record(code, is_retry_trigger=False)
        return issues


    def _registry_autocorrects(self, sql: str) -> str:
        """Consume business_rules.py's AUTOCORRECT registry rules, restricted to
        _CONSUMED_RULE_IDS. Called from cleaner.py's clean_sql(), which runs
        before ast_repair()/validate() in generation.py's pipeline — so a
        rule registered here (and added to _CONSUMED_RULE_IDS) lands before
        validation ever sees the SQL, instead of relying on the SQL-gen model
        to comply with a retry-prompt hint."""
        for rule in self._rules._autocorr_rules:
            if rule["id"] not in self._CONSUMED_RULE_IDS:
                continue
            sql = rule["_sql_re"].sub(rule["replacement"], sql)
        return sql


    # ------------------------------------------------------------------ #
    # Primary validation — HARD_FAIL vs SOFT_WARN separation
    # ------------------------------------------------------------------ #

    def validate(self, sql: str, question: str = "", qid: str = "") -> ValidationResult:
        """
        Validate SQL. Returns ValidationResult.

        Hard fails block execution and produce concise retry codes.
        Soft warns are advisory only and are NEVER forwarded to the LLM.

        Design rules:
          1. Early-return on non-retryable catastrophic failures (DML, empty)
          2. Accumulate hard fails — up to 3 before returning
          3. Soft warns accumulate separately, never mixed into retry message
          4. No extended hint prose in ValidationIssue.message — only in .detail
        """
        issues: list[ValidationIssue] = []

        # ── 1. Empty / error marker ───────────────────────────────────────
        if not sql or sql.startswith("-- ERROR:"):
            issue = ValidationIssue(
                "HARD_FAIL", "EMPTY_SQL",
                sql or "Empty SQL generated",
                retryable=False,
            )
            ValidatorStats.record("EMPTY_SQL", is_retry_trigger=True)
            return ValidationResult(passed=False, issues=[issue])

        sql_lower = sql.lower()

        # ── 2. Registry-driven BLOCK/WARN rules (core/business_rules.py) ──
        # DML block, hallucinated table names, dynamic fb_ tables, responsible/
        # pending_with join, close_on arithmetic, missing-limit, hardcoded-year,
        # plus the Group-B financial-integrity guards — see _CONSUMED_RULE_IDS.
        issues.extend(self._registry_block_warn_issues(sql, question, qid))

        # ── 4. Hallucinated columns on ai_* tables ────────────────────────
        for table, col_hints in self.HALLUCINATED_AI_COLUMNS.items():
            if not re.search(rf"\b{re.escape(table)}\b", sql_lower):
                continue
            # Collect the aliases used for this table in FROM/JOIN clauses
            # e.g. "FROM ai_answers aa" → alias "aa"
            # e.g. "JOIN ai_answers ans" → alias "ans"
            # We only flag qualified refs where the alias maps to THIS table.
            table_aliases: set[str] = set()
            for m in re.finditer(
                rf'\b{re.escape(table)}\s+(\w+)\b', sql_lower
            ):
                table_aliases.add(m.group(1))
            # Also include the bare table name itself as a possible qualifier
            table_aliases.add(table)

            for col, hint in col_hints.items():
                # Only match qualified references (alias.col) where alias ∈ table_aliases
                found = False
                for alias in table_aliases:
                    if re.search(rf'\b{re.escape(alias)}\.{re.escape(col)}\b', sql_lower):
                        found = True
                        break
                if found:
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "WRONG_COLUMN",
                        f"Column '{col}' does not exist on {table} — {hint}",
                        detail=f"Hallucinated column: {table}.{col}",
                    ))
                    ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)

        # ── 5. Hallucinated columns on inspection_report ──────────────────
        ir_used = bool(re.search(r'\b(inspection_report|ir)\b', sql_lower))
        if ir_used:
            for col, join_hint in self.HALLUCINATED_IR_COLUMNS.items():
                for alias in ("ir", "inspection_report"):
                    if re.search(rf"\b{re.escape(alias)}\.{re.escape(col)}\b", sql_lower):
                        issues.append(ValidationIssue(
                            "HARD_FAIL", "WRONG_COLUMN",
                            f"Column '{alias}.{col}' does not exist on inspection_report",
                            detail=f"Requires JOIN: {join_hint}",
                        ))
                        ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)
                        break

        # ── 5b. Hallucinated columns on inspection_corrective_action ──────
        # deepseek confabulates risk/deferral columns that don't exist on ICA.
        # Proven in traces: capex_status, adequacy_status, times_deferred.
        # Also guards ir.current_qtr_score / ir.last_qtr_score ghost columns.
        #
        # ALIAS AMBIGUITY (fixed 2026-06-11): "ic" is ALSO the conventional
        # alias for inspection_cycle (e.g. "JOIN inspection_cycle ic ON ...").
        # A bare \bic\b match previously activated this whole section for
        # queries that have NO inspection_corrective_action table at all —
        # then a query selecting isched.due_date (valid — inspection_schedule
        # HAS a due_date column) was wrongly flagged as a hallucinated ICA
        # column. Fix: only treat "ica"/"ic" as ICA aliases if
        # inspection_corrective_action actually appears in FROM/JOIN; extract
        # its real alias(es) from there instead of guessing.
        _ica_table_used = bool(re.search(r'\binspection_corrective_action\b', sql_lower))
        _ica_aliases: set[str] = {"ica", "inspection_corrective_action"}
        for _m in re.finditer(r'\binspection_corrective_action\s+(?:as\s+)?(\w+)', sql_lower):
            _ica_aliases.add(_m.group(1))
        ica_used = _ica_table_used or bool(re.search(r'\bica\b', sql_lower))
        if ica_used:
            for col, hint in self.HALLUCINATED_ICA_COLUMNS.items():
                qualified_only = col in self._ICA_QUALIFIED_ONLY_COLS
                hit = False
                for alias in _ica_aliases:
                    if re.search(rf"\b{re.escape(alias)}\.{re.escape(col)}\b", sql_lower):
                        hit = True
                        break
                if not hit and not qualified_only:
                    # Unqualified bare reference — only flag in clause positions
                    # (ORDER BY, WHERE, SELECT column list) to avoid matching aliases/comments.
                    # (?<![\w.]) excludes BOTH "foo_due_date" (different identifier) AND
                    # "isched.due_date" (validly qualified to a DIFFERENT table that
                    # legitimately has this column, e.g. inspection_schedule.due_date).
                    if re.search(
                        rf"(?:order\s+by|where\s|select\s)[^;]*(?<![\w.]){re.escape(col)}\b",
                        sql_lower, re.DOTALL
                    ) and not re.search(
                        rf"\bAS\s+{re.escape(col)}\b",
                        sql, re.IGNORECASE
                    ):
                        hit = True
                if hit:
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "WRONG_COLUMN",
                        f"Column '{col}' does not exist on inspection_corrective_action — {hint}",
                        detail=f"Hallucinated ICA column: {col}",
                    ))
                    ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)

        # ── 5b2. Hallucinated columns on inspection_cycle ─────────────────
        # deepseek/llama repeatedly confabulate a display-name column on
        # inspection_cycle (ic.name, ic.cycle_name, ...) even after get_schema
        # returns the real column list — seen looping for 5 straight retries
        # in production traces (2026-07-08). Only flag when inspection_cycle
        # is actually joined, using its real alias(es) from FROM/JOIN, so we
        # don't false-positive on unrelated tables that legitimately have a
        # "name" column (facility.name, project.name, etc.).
        _ic_table_used = bool(re.search(r'\binspection_cycle\b', sql_lower))
        if _ic_table_used:
            _ic_aliases: set[str] = {"inspection_cycle"}
            for _m in re.finditer(r'\binspection_cycle\s+(?:as\s+)?(\w+)', sql_lower):
                _ic_aliases.add(_m.group(1))
            for col, hint in self.HALLUCINATED_IC_COLUMNS.items():
                for alias in _ic_aliases:
                    if re.search(rf"\b{re.escape(alias)}\.{re.escape(col)}\b", sql_lower):
                        issues.append(ValidationIssue(
                            "HARD_FAIL", "WRONG_COLUMN",
                            f"Column '{alias}.{col}' does not exist on inspection_cycle — {hint}",
                            detail=f"Hallucinated IC column: {col}",
                        ))
                        ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)
                        break

        # ── 5b3. Hallucinated columns on audit_cycle ──────────────────────
        # Audit-domain counterpart to 5b2 — see HALLUCINATED_AC_COLUMNS.
        _ac_table_used = bool(re.search(r'\baudit_cycle\b', sql_lower))
        if _ac_table_used:
            _ac_aliases: set[str] = {"audit_cycle"}
            for _m in re.finditer(r'\baudit_cycle\s+(?:as\s+)?(\w+)', sql_lower):
                _ac_aliases.add(_m.group(1))
            for col, hint in self.HALLUCINATED_AC_COLUMNS.items():
                for alias in _ac_aliases:
                    if re.search(rf"\b{re.escape(alias)}\.{re.escape(col)}\b", sql_lower):
                        issues.append(ValidationIssue(
                            "HARD_FAIL", "WRONG_COLUMN",
                            f"Column '{alias}.{col}' does not exist on audit_cycle — {hint}",
                            detail=f"Hallucinated AC column: {col}",
                        ))
                        ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)
                        break

        # ── 5b4. Hallucinated columns on audit_plan ───────────────────────
        # See HALLUCINATED_AP_COLUMNS — currently just audit_universe_id,
        # which only exists on audit_schedule.
        _ap_table_used = bool(re.search(r'\baudit_plan\b', sql_lower))
        if _ap_table_used:
            _ap_aliases: set[str] = {"audit_plan"}
            for _m in re.finditer(r'\baudit_plan\s+(?:as\s+)?(\w+)', sql_lower):
                _ap_aliases.add(_m.group(1))
            for col, hint in self.HALLUCINATED_AP_COLUMNS.items():
                for alias in _ap_aliases:
                    if re.search(rf"\b{re.escape(alias)}\.{re.escape(col)}\b", sql_lower):
                        issues.append(ValidationIssue(
                            "HARD_FAIL", "WRONG_COLUMN",
                            f"Column '{alias}.{col}' does not exist on audit_plan — {hint}",
                            detail=f"Hallucinated AP column: {col}",
                        ))
                        ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)
                        break

        # ── 5b5. Hallucinated columns on auditor_portfolio ────────────────
        # See HALLUCINATED_PO_COLUMNS — checks every alias regardless of
        # whether it's the correct 'po' or the wrong 'ap', since the column
        # doesn't exist under either name.
        _po_table_used = bool(re.search(r'\bauditor_portfolio\b', sql_lower))
        if _po_table_used:
            _po_aliases: set[str] = {"auditor_portfolio"}
            for _m in re.finditer(r'\bauditor_portfolio\s+(?:as\s+)?(\w+)', sql_lower):
                _po_aliases.add(_m.group(1))
            for col, hint in self.HALLUCINATED_PO_COLUMNS.items():
                for alias in _po_aliases:
                    if re.search(rf"\b{re.escape(alias)}\.{re.escape(col)}\b", sql_lower):
                        issues.append(ValidationIssue(
                            "HARD_FAIL", "WRONG_COLUMN",
                            f"Column '{alias}.{col}' does not exist on auditor_portfolio — {hint}",
                            detail=f"Hallucinated PO column: {col}",
                        ))
                        ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)
                        break

        # ── 5c. Impossible JOIN conditions (from join graph) ─────────────
        for _pattern, _correct in self.IMPOSSIBLE_JOIN_PATTERNS:
            if re.search(_pattern, sql_lower):
                # Distinguish cast-direction errors with a dedicated code
                # so the retry message is specific, not generic IMPOSSIBLE_JOIN.
                _code = "WRONG_CAST_DIR" if "cast" in _correct.lower() or "submission_id" in _pattern else "IMPOSSIBLE_JOIN"
                issues.append(ValidationIssue(
                    "HARD_FAIL", _code,
                    f"{'Wrong cast direction' if _code == 'WRONG_CAST_DIR' else 'Impossible join'} — {_correct}",
                    detail=f"Pattern matched: {_pattern}",
                ))
                ValidatorStats.record(_code, is_retry_trigger=True)

        # ── 6. ir.id in SELECT clause (UUID PK exposed for display) ───────
        # IMPORTANT: ir.id is CORRECT and required as a JOIN key, e.g.:
        #   WHERE aa.inspection_report_id = (SELECT ir.id FROM ...)  ← valid
        #   JOIN ica ON ica.inspection_id = ir.id                    ← valid
        # Only flag ir.id when it appears in the OUTERMOST SELECT column list
        # (where it would be displayed as UUID garbage instead of inspection_id).
        if ir_used:
            if _HAS_SQLGLOT:
                _check_ir_id_in_select_ast(sql, issues)
            else:
                # Regex fallback: strip subquery content first to avoid matching
                # ir.id that lives inside a WHERE (SELECT ir.id ...) subquery.
                outermost = _strip_nested_parens(sql)
                m = re.search(r'\bSELECT\b(.*?)\bFROM\b',
                              outermost, re.IGNORECASE | re.DOTALL)
                if m and re.search(r'\bir\.id\b', m.group(1), re.IGNORECASE):
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "UUID_IN_SELECT",
                        "ir.id is the UUID primary key — use ir.inspection_id for display",
                        detail="ir.inspection_id is the human-readable varchar '2026/04/ST001/INS001'",
                    ))
                    ValidatorStats.record("UUID_IN_SELECT", is_retry_trigger=True)

        # ── 6a. audit-domain UUID PK exposed for display (same trap as #6,
        # generalized) — confirmed live 2026-07-28 across multiple audit
        # tables in the same eval run, not just one. AST-only (no regex
        # fallback needed — sqlglot is a hard dependency of this module).
        if self.domain == "audit" and _HAS_SQLGLOT:
            _check_uuid_pk_in_select_ast(sql, issues, self.AUDIT_UUID_PK_DISPLAY_MAP)

        # ── 6a-inverse. A UUID-shaped literal compared against a VARCHAR
        # display-code column instead of the row's real id — confirmed live:
        # ordinal resolution ("the 1st one") handed the model a real UUID,
        # but it filtered arep.audit_id = '<uuid>' instead of arep.id =
        # '<uuid>', silently returning 0 rows. Not gated to domain=="audit"
        # (the map covers both audit and inspection aliases) — cheap AST
        # check, same pattern as 6a above.
        if _HAS_SQLGLOT:
            _check_display_code_uuid_compare_ast(sql, issues, self.DISPLAY_CODE_UUID_COMPARE_MAP)

        # ── 6a-inverse-2. The mirror failure shape: a display-code-shaped
        # literal compared against a column that is a genuine UUID FK on a
        # different table (e.g. aca.audit_id = '2026/06/ST074/AUD001') —
        # confirmed live twice, see UUID_FK_DISPLAY_CODE_COMPARE_MAP above.
        # Same domain-agnostic, cheap-AST-check pattern as 6a-inverse.
        if _HAS_SQLGLOT:
            _check_uuid_fk_compared_to_display_code_ast(sql, issues, self.UUID_FK_DISPLAY_CODE_COMPARE_MAP)

        # ── 6a-inverse-3. Two ordinary, well-formed display codes from
        # DIFFERENT tables confused for each other (e.g. a schedule's code
        # fed into audit_report.audit_id) — see OWN_DISPLAY_CODE_TOKEN_MAP
        # above. Same domain-agnostic, cheap-AST-check pattern.
        if _HAS_SQLGLOT:
            _check_display_code_wrong_table_ast(sql, issues, self.OWN_DISPLAY_CODE_TOKEN_MAP)

        # ── 6a-inverse-4. Same column family, through a scalar subquery
        # instead of a literal — confirmed live twice (2026-07-30,
        # 2026-08-04), see _check_subquery_uuid_display_code_mismatch_ast's
        # docstring. Self-contained (no domain-specific map — spans both
        # domains), a guaranteed Postgres type error either way, so zero
        # false-positive risk.
        if _HAS_SQLGLOT:
            _check_subquery_uuid_display_code_mismatch_ast(sql, issues)

        # ── 6b. Undefined table aliases — column refs to aliases not in FROM/JOIN ──
        # Catches: SELECT ... WHERE ir.active = true  with NO "FROM/JOIN ... ir"
        # anywhere in the query. Postgres rejects this at execution time with
        # 'missing FROM-clause entry for table "ir"', but plain-regex validation
        # passes it since 'ir.active' LOOKS like a valid qualified column.
        # Observed 2026-06-11: this exact pattern passed validation twice in a
        # row (Deepseek kept dropping the JOIN while keeping ir.active in WHERE),
        # burning 3 retries before Llama finally added the missing JOIN.
        if _HAS_SQLGLOT:
            _check_undefined_table_aliases(sql, issues)

        # ── 6c. Ungrouped SELECT column alongside an aggregate — HARD_FAIL ──
        # Catches: SELECT a, COUNT(b) ... GROUP BY a, c  where a stray column
        # (not in GROUP BY, not aggregated) sits next to a real aggregate.
        # Domain-agnostic — this is plain Postgres GROUP BY semantics, not an
        # audit/inspection-specific trap. See _check_ungrouped_select_columns_ast.
        if _HAS_SQLGLOT:
            _check_ungrouped_select_columns_ast(sql, issues)

        # ── 9b. Stale hardcoded date literal — HARD_FAIL ─────────────────
        # deepseek hallucinates DATE '2023-01-01' or '2022-...' from training data
        # when generating date range filters. These are always wrong — the data is
        # from 2025/2026. Only fire for years clearly in the past (≤ current year - 2)
        # to avoid blocking intentional historical queries like "inspections in 2024".
        _stale_date_re = re.compile(
            r"""(?:date\s+['"]|['"])(20(?:2[0-4]|1\d))-\d{2}-\d{2}['"]""",
            re.IGNORECASE,
        )
        for m in _stale_date_re.finditer(sql_lower):
            stale_year = m.group(0)
            issues.append(ValidationIssue(
                "HARD_FAIL", "STALE_DATE_LITERAL",
                f"Hardcoded stale date literal detected — use EXTRACT(MONTH/YEAR FROM CURRENT_DATE) "
                f"or pass the year from context instead of a fixed past date",
                detail=f"Matched: {stale_year}",
            ))
            ValidatorStats.record("STALE_DATE_LITERAL", is_retry_trigger=True)
            break  # one report is enough

        # ── 10b. TO_CHAR used as a WHERE filter — HARD_FAIL ───────────────
        # TO_CHAR(col, 'Month YYYY') = 'January 2026' silently returns 0 rows
        # because PostgreSQL pads month names: 'January   2026' ≠ 'January 2026'.
        # This pattern is ONLY valid for GROUP BY labels / SELECT display.
        if re.search(r'\bto_char\b', sql_lower):
            # Detect: to_char(..., 'Month YYYY') = '...' — the = must immediately follow
            # the closing ) of the to_char() call to avoid false positives where a distant
            # = ' later in the query (e.g. ir.status != 'DRAFT') is incorrectly matched.
            tochar_month_eq = re.search(
                r"\bto_char\b[^;]*'month\s+yyyy'\s*\)\s*=\s*'",
                sql_lower, re.IGNORECASE | re.DOTALL,
            )
            if tochar_month_eq:
                # Confirm a WHERE keyword precedes this pattern (not just SELECT/GROUP BY).
                # Only look for WHERE after the last SELECT to avoid matching CTE WHERE clauses.
                before = sql_lower[:tochar_month_eq.start()]
                last_select = max((m.end() for m in re.finditer(r'\bselect\b', before)), default=0)
                in_where = bool(re.search(r'\bwhere\b', before[last_select:]))
                if in_where:
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "TOCHAR_WHERE_FILTER",
                        "TO_CHAR(..., 'Month YYYY') = '...' in WHERE always returns 0 rows — "
                        "use submitted_on >= 'YYYY-MM-01' AND < next month instead",
                        detail="PostgreSQL pads month names with spaces: 'January   2026' != 'January 2026'",
                    ))
                    ValidatorStats.record("TOCHAR_WHERE_FILTER", is_retry_trigger=True)

        # ── 10c. Truncated inspection_id — HARD_FAIL ──────────────────────
        # inspection_id is always 'YYYY/MM/STXXX/INSXXX' (full VARCHAR).
        # deepseek sometimes drops the 'YYYY/MM/' prefix, producing 'STXXX/INSXXX'.
        # Detect: inspection_id = 'STXXX/INSXXX' (missing year/month prefix).
        _trunc_id = re.search(
            r"""inspection_id\s*=\s*['"](?!\d{4}/)(st\d+/ins\d+)['"]""",
            sql_lower, re.IGNORECASE,
        )
        if _trunc_id:
            issues.append(ValidationIssue(
                "HARD_FAIL", "TRUNCATED_INSPECTION_ID",
                f"Truncated inspection_id '{_trunc_id.group(1)}' — full format is 'YYYY/MM/STXXX/INSXXX'",
                detail="Use the full inspection_id string as provided by the user",
            ))
            ValidatorStats.record("TRUNCATED_INSPECTION_ID", is_retry_trigger=True)
        # ── 10d. Mismatched first/last name OR-split — HARD_FAIL ──────────
        # WHERE (x.first_name ILIKE '%George%' OR x.last_name ILIKE '%Auditor%')
        # matches anyone named George OR anyone surnamed Auditor — not people
        # named "George Auditor". A same-literal OR split (both '%george%') is
        # the correct single-word check and is NOT flagged. Two-word full names
        # must use the concatenated form instead:
        #   (x.first_name || ' ' || x.last_name) ILIKE '%George Auditor%'
        _name_or_pat = re.compile(
            r"(\w+)\.first_name\s+ILIKE\s+'%([^%']+)%'\s+OR\s+\1\.last_name\s+ILIKE\s+'%([^%']+)%'"
            r"|(\w+)\.last_name\s+ILIKE\s+'%([^%']+)%'\s+OR\s+\4\.first_name\s+ILIKE\s+'%([^%']+)%'",
            re.IGNORECASE,
        )
        _name_or_match = _name_or_pat.search(sql)
        if _name_or_match:
            g = _name_or_match.groups()
            first_val, last_val = (g[1], g[2]) if g[0] else (g[5], g[4])
            if first_val.strip().lower() != last_val.strip().lower():
                issues.append(ValidationIssue(
                    "HARD_FAIL", "SPLIT_NAME_MISMATCH",
                    f"first_name/last_name OR-split uses different literals "
                    f"('{first_val}' vs '{last_val}') — matches wrong people",
                    detail=(
                        "Use the concatenated form for a full name: "
                        f"(alias.first_name || ' ' || alias.last_name) ILIKE "
                        f"'%{first_val} {last_val}%'"
                    ),
                ))
                ValidatorStats.record("SPLIT_NAME_MISMATCH", is_retry_trigger=True)

        # ── 10e. "overdue" mitigative actions via date arithmetic — HARD_FAIL ──
        # status='OVERDUE' is the authoritative, maintained field. Reconstructing
        # it via `NOT status IN (...) AND target_close_out_date < CURRENT_DATE`
        # silently diverges from that field's business rule (grace periods,
        # deferrals, etc. are baked into the maintained status, not the raw date
        # comparison) — both 3B and 7B were observed doing this identically in
        # eval, always without error, so it was invisible until the SQL was
        # read by hand. Scoped tightly to "overdue" + mitigative/action to avoid
        # over-firing on unrelated "active"/"status" questions.
        _overdue_ca_q = re.compile(
            r"\boverdue\b.{0,25}\b(?:mitigative|actions?)\b"
            r"|\b(?:mitigative|actions?)\b.{0,25}\boverdue\b",
            re.IGNORECASE,
        )
        if question and _overdue_ca_q.search(question):
            uses_overdue_status = "'OVERDUE'" in sql.upper()
            uses_date_compare = bool(re.search(
                r"target_close_out_date\s*<\s*(?:now|current_date)", sql_lower, re.IGNORECASE))
            if not uses_overdue_status and uses_date_compare:
                issues.append(ValidationIssue(
                    "HARD_FAIL", "OVERDUE_DATE_ARITHMETIC",
                    "Overdue mitigative actions must use status='OVERDUE', not date arithmetic",
                    detail="status='OVERDUE' is the authoritative field — do not recompute "
                           "overdue from target_close_out_date/CLOSED/CLOSE_WITH_DEFERRED",
                ))
                ValidatorStats.record("OVERDUE_DATE_ARITHMETIC", is_retry_trigger=True)

        # ── 10f. "which cycle" returning a bare cycle_id — HARD_FAIL ────────────
        # inspection_cycle has no human-readable name — its only identity is
        # start_date/end_date/status. A query that answers "which cycle ..." by
        # selecting/grouping on the raw cycle_id UUID without joining
        # inspection_cycle is unusable in the UI and easily read as duplicate
        # rows when unrelated derived labels (e.g. month of an individual
        # inspection's submitted_on) coincide across several distinct cycles.
        # Reuses the precise AST-based SELECT/GROUP BY leak detector (not a
        # blind substring match) so a legitimate `WHERE cycle_id = '<uuid>'`
        # scoping filter is never flagged.
        if question and re.search(r"\bcycle\b", question, re.IGNORECASE):
            _cycle_leak: list[ValidationIssue] = []
            # Domain-scoped — audit_plan/audit_schedule's cycle FK is audit_cycle,
            # not inspection_cycle. Using the inspection tuple unconditionally here
            # meant `lookup_table.lower() in sql_lower` ("inspection_cycle" in sql)
            # was ALWAYS False for audit SQL, so this HARD_FAILed every audit-domain
            # query that selected/grouped cycle_id — even ones that correctly
            # joined audit_cycle. Caught live via eval_audit.py's PLAN-08 run.
            if self.domain == "audit":
                _cycle_table, _cycle_alias = "audit_cycle", "acyc"
                _cycle_fk = {"cycle_id": (
                    _cycle_table, _cycle_alias,
                    "acyc.start_date, acyc.end_date, acyc.status AS cycle_status",
                )}
            else:
                _cycle_table, _cycle_alias = "inspection_cycle", "ic"
                _cycle_fk = {"cycle_id": self.UUID_FK_COLUMNS["cycle_id"]}
            if _HAS_SQLGLOT:
                _check_uuid_leaks_ast(sql, _cycle_fk, _cycle_leak)
            else:
                _check_uuid_leaks_regex(sql_lower, _cycle_fk, _cycle_leak)
            if _cycle_leak:
                issues.append(ValidationIssue(
                    "HARD_FAIL", "CYCLE_UUID_NO_JOIN",
                    f"Question asks about a cycle but SQL exposes cycle_id without "
                    f"joining {_cycle_table} for its date span/status",
                    detail=_cycle_leak[0].detail,
                ))
                ValidatorStats.record("CYCLE_UUID_NO_JOIN", is_retry_trigger=True)

        # ── 10g. "opex incurred/converted" answered via po_details — HARD_FAIL ──
        # po_details (Purchase Orders) is the CAPEX conversion table; pr_details
        # (Purchase Requisitions) is the OPEX one. Both share the same
        # mitigative_action_uuid join shape, so joining the wrong one produces
        # a plausible, non-erroring result that is actually measuring CAPEX
        # data under an "opex" label — observed live 2026-07-07: a query asking
        # for "opex generated and converted" joined po_details for the
        # "converted" figure instead of pr_details.
        _opex_convert_q = re.compile(
            r"\bopex\b.{0,25}\b(?:converted|incurred|spend|spent)\b"
            r"|\b(?:converted|incurred|spend|spent)\b.{0,25}\bopex\b",
            re.IGNORECASE,
        )
        if question and _opex_convert_q.search(question):
            uses_po = bool(re.search(r"\bpo_details\b|\bpo\.po_amount\b", sql_lower))
            uses_pr = bool(re.search(r"\bpr_details\b|\bpr\.pr_amount\b", sql_lower))
            if uses_po and not uses_pr:
                issues.append(ValidationIssue(
                    "HARD_FAIL", "OPEX_WRONG_TABLE",
                    "OPEX incurred/converted must come from pr_details (Purchase "
                    "Requisitions), not po_details (that's the CAPEX conversion table)",
                    detail="JOIN pr_details pr ON pr.mitigative_action_uuid = ica.id, "
                           "SELECT SUM(pr.pr_amount)",
                ))
                ValidatorStats.record("OPEX_WRONG_TABLE", is_retry_trigger=True)

        # ── 10h. Requested row count vs. actual LIMIT — SOFT_WARN ────────────
        # loop.py's LIMIT-preservation hint ("user asked for exactly N
        # results — use LIMIT N", appended to schema_hint before generate_sql)
        # is unenforced prose the SQL-gen model can ignore. SOFT_WARN, not
        # HARD_FAIL: a top-level LIMIT mismatch isn't always wrong — a
        # legitimate query can need a LIMIT inside a CTE/subquery to control
        # join fan-out, with no matching top-level LIMIT at all (e.g. "the
        # last 5 inspections and their corrective actions" may need `LIMIT 5`
        # scoped to the inspections CTE, not the final joined row count).
        # HARD_FAILing on that shape would block correct SQL — exactly what
        # this whole validator redesign exists to avoid. Reuses the same
        # (already date-window-excluding) regex loop.py's hint itself uses,
        # so this check can never fire on a case the hint wouldn't also have
        # fired on.
        _limit_hint_q = re.search(
            r'\b(?:last|top|first|recent|latest)\s+(\d+)'
            r'(?!\s*(?:days?|weeks?|months?|hours?|minutes?|years?)\b)\b',
            question, re.IGNORECASE,
        ) or re.search(
            r'\b(\d+)\s+(?:most\s+recent|latest|last|inspections?|actions?|reports?)\b',
            question, re.IGNORECASE,
        ) if question else None
        if _limit_hint_q:
            _requested_n = int(_limit_hint_q.group(1))
            if 1 <= _requested_n <= 500:
                _actual_limit = _extract_top_level_limit(sql)
                if _actual_limit is not None and _actual_limit != _requested_n:
                    issues.append(ValidationIssue(
                        "SOFT_WARN", "LIMIT_MISMATCH",
                        f"User asked for exactly {_requested_n} results but SQL's "
                        f"top-level LIMIT is {_actual_limit}",
                        detail=f"schema_hint requested LIMIT {_requested_n}",
                        retryable=False,
                    ))
                    ValidatorStats.record("LIMIT_MISMATCH", is_retry_trigger=False)

        hard_fails = [i for i in issues if i.severity == "HARD_FAIL"]
        passed = len(hard_fails) == 0

        if hard_fails:
            logger.info("[VALID] qid=%s HARD_FAIL (%d): %s",
                        qid, len(hard_fails), "; ".join(str(f) for f in hard_fails))
        soft = [i for i in issues if i.severity == "SOFT_WARN"]
        if soft:
            logger.debug("[VALID] qid=%s SOFT_WARN (%d, not sent to LLM): %s",
                         qid, len(soft), "; ".join(str(w) for w in soft))

        return ValidationResult(passed=passed, issues=issues)

