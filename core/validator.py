"""
SQL Validator — production-grade refactor.

Architecture
============
  clean_sql()         — formatting normalization only; minimal semantic mutation
  validate()          — returns ValidationResult; HARD_FAIL vs SOFT_WARN separated
  validate_semantic() — SOFT_WARN only; NEVER blocks execution; NEVER in retry prompts
  ValidatorStats      — lightweight rule-hit counters for observability

Hard fails block execution and appear in retry prompts (concise codes only).
Soft warns are logged for observability but are NEVER sent to the LLM.

This separation eliminates:
  - validator-induced semantic drift on retries
  - verbose warning prose polluting retry prompts
  - false-positive blocks on valid complex SQL patterns
"""

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

try:
    import sqlglot
    import sqlglot.expressions as exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False

try:
    from core.business_glossary import check_glossary as _check_glossary
    _HAS_GLOSSARY = True
except ImportError:
    _HAS_GLOSSARY = False

logger = logging.getLogger("inspectai.sql")
_log_security = logging.getLogger("inspectai.security")


# ---------------------------------------------------------------------------
# Structured validation output
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """
    A single validation finding.

    severity  : "HARD_FAIL" — blocks execution, appears in retry prompt (as code+message only)
                "SOFT_WARN" — logged only; NEVER sent to LLM; NEVER forces retry
    code      : machine-readable identifier, e.g. "WRONG_COLUMN"
    message   : short, actionable description (≤ 80 chars)
    detail    : extended context for logs/debug only — never injected into prompts
    retryable : False for structural issues the LLM can't fix (e.g. DML blocked)
    """
    severity: str
    code: str
    message: str
    detail: str = ""
    retryable: bool = True

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


@dataclass
class ValidationResult:
    """
    Replaces tuple[bool, list[str]].

    passed      : True if no HARD_FAILs present
    issues      : all issues (hard + soft)

    Callers:
      result.passed              → gate execution
      result.retry_message()     → single concise string for LLM retry prompt
      result.hard_fails          → list of blocking issues (for logging)
      result.soft_warns          → list of advisory issues (for logging only)
    """
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def hard_fails(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "HARD_FAIL"]

    @property
    def soft_warns(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "SOFT_WARN"]

    def retry_message(self) -> str:
        """
        Concise, machine-readable string for the retry prompt.

        Rules:
          - Only HARD_FAILs, maximum 3 (prevents prompt bloat)
          - Format: "[CODE] short message" — no extended hints, no prose
          - Soft warnings are intentionally excluded
        """
        fails = self.hard_fails[:3]
        if not fails:
            return "SQL validation failed."
        return "; ".join(str(f) for f in fails)


# ---------------------------------------------------------------------------
# Observability — lightweight rule hit counter
# Zero external dependencies. Inspect at runtime via ValidatorStats.snapshot().
# ---------------------------------------------------------------------------

class ValidatorStats:
    """
    Tracks validator rule hit frequency.
    Use to identify which rules fire most, which are noisy, and which drive retries.

    Usage:
        ValidatorStats.record("WRONG_COLUMN")
        ValidatorStats.snapshot()   # → dict of code → count
        ValidatorStats.reset()
    """
    _counts: dict[str, int] = defaultdict(int)
    _retry_triggers: dict[str, int] = defaultdict(int)

    @classmethod
    def record(cls, code: str, is_retry_trigger: bool = False) -> None:
        cls._counts[code] += 1
        if is_retry_trigger:
            cls._retry_triggers[code] += 1

    @classmethod
    def snapshot(cls) -> dict:
        return {
            "rule_hits": dict(cls._counts),
            "retry_triggers": dict(cls._retry_triggers),
            "total_issues": sum(cls._counts.values()),
            "total_retry_triggers": sum(cls._retry_triggers.values()),
        }

    @classmethod
    def reset(cls) -> None:
        cls._counts.clear()
        cls._retry_triggers.clear()


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

class SQLValidator:
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
    })

    # block_dml used to early-return non-retryable, before any other check ran.
    # Folding it into the generic registry-issue pass changes that control flow
    # (other checks can now also fire on the same SQL), but this keeps its
    # ValidatorStats/ValidationIssue observability semantics identical.
    _NON_RETRYABLE_CODES: frozenset = frozenset({"DML_BLOCKED"})

    def __init__(self):
        from core.business_rules import get_engine
        self._rules = get_engine()

    def _registry_block_warn_issues(self, sql: str, question: str = "") -> list:
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
                issues.append(ValidationIssue("SOFT_WARN", code, rule["message"], retryable=False))
                ValidatorStats.record(code, is_retry_trigger=False)
        return issues

    def _registry_autocorrects(self, sql: str) -> str:
        """Consume business_rules.py's AUTOCORRECT registry rules, restricted to _CONSUMED_RULE_IDS."""
        for rule in self._rules._autocorr_rules:
            if rule["id"] not in self._CONSUMED_RULE_IDS:
                continue
            sql = rule["_sql_re"].sub(rule["replacement"], sql)
        return sql

    # ------------------------------------------------------------------ #
    # Primary validation — HARD_FAIL vs SOFT_WARN separation
    # ------------------------------------------------------------------ #

    def validate(self, sql: str, question: str = "") -> ValidationResult:
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
        issues.extend(self._registry_block_warn_issues(sql, question))

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
            _cycle_fk = {"cycle_id": self.UUID_FK_COLUMNS["cycle_id"]}
            if _HAS_SQLGLOT:
                _check_uuid_leaks_ast(sql, _cycle_fk, _cycle_leak)
            else:
                _check_uuid_leaks_regex(sql_lower, _cycle_fk, _cycle_leak)
            if _cycle_leak:
                issues.append(ValidationIssue(
                    "HARD_FAIL", "CYCLE_UUID_NO_JOIN",
                    "Question asks about a cycle but SQL exposes cycle_id without "
                    "joining inspection_cycle for its date span/status",
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

        hard_fails = [i for i in issues if i.severity == "HARD_FAIL"]
        passed = len(hard_fails) == 0

        if hard_fails:
            logger.info("[VALID] HARD_FAIL (%d): %s",
                        len(hard_fails), "; ".join(str(f) for f in hard_fails))
        soft = [i for i in issues if i.severity == "SOFT_WARN"]
        if soft:
            logger.debug("[VALID] SOFT_WARN (%d, not sent to LLM): %s",
                         len(soft), "; ".join(str(w) for w in soft))

        return ValidationResult(passed=passed, issues=issues)

    # ------------------------------------------------------------------ #
    # Shape validation — HARD_FAIL for structural mismatches
    # Call after generate_sql when the expected shape is known.
    # ------------------------------------------------------------------ #

    # Map of question patterns → expected shape
    _SINGLE_ROW_PATTERNS = re.compile(
        r"\b(this\s+(month|quarter|year|week)\s+vs\.?\s+last|"
        r"last\s+(month|quarter|year|week)\s+vs\.?|"
        r"how\s+many.{0,30}(vs\.?|versus|compared?\s+to)|"
        r"raised\s+vs\.?\s+closed|"
        r"comparison|"
        r"faster\s+or\s+slower)\b",
        re.IGNORECASE,
    )

    def validate_shape(self, sql: str, question: str) -> ValidationResult:
        """
        AST-level shape validation. Called when query intent implies a
        specific result structure that the generated SQL must match.

        Currently enforces:
          - SINGLE_ROW intent: GROUP BY is forbidden (use FILTER aggregates)

        Returns ValidationResult with HARD_FAIL if shape is violated.
        This triggers a retry with an explicit structural correction message.
        """
        if not _HAS_SQLGLOT:
            return ValidationResult(passed=True, issues=[])

        issues: list[ValidationIssue] = []
        q_lower = question.lower()

        # ── Completed status filter ───────────────────────────────────────
        # "completed inspections" must filter status IN ('CLOSED','SUBMITTED').
        # status != 'DRAFT' alone is insufficient — it includes IN_PROGRESS etc.
        # NOTE: SOFT_WARN only — deepseek cannot fix this via retry because it
        # copies the seed example. Fix is in the seed SQL and CORE_CONTEXT.
        _COMPLETED_Q = re.compile(
            r"\b(completed|finished|done|finalized)\s+(inspection|inspections?)\b"
            r"|\binspections?\s+(were\s+)?(completed|finished|done|finalized)\b"
            r"|\bhow\s+many\s+inspections?.{0,20}(completed|finished|done)\b",
            re.IGNORECASE,
        )
        if _COMPLETED_Q.search(question):
            has_submitted = "'SUBMITTED'" in sql.upper() or "\"SUBMITTED\"" in sql
            has_closed = "'CLOSED'" in sql.upper() or "\"CLOSED\"" in sql
            if not (has_submitted or has_closed):
                issues.append(ValidationIssue(
                    "SOFT_WARN", "MISSING_COMPLETED_STATUS",
                    "Query asks about 'completed' inspections but SQL has neither 'CLOSED' nor 'SUBMITTED'",
                    detail="completed_inspection = status IN ('CLOSED','SUBMITTED')",
                    retryable=False,
                ))
                ValidatorStats.record("MISSING_COMPLETED_STATUS", is_retry_trigger=False)
                return ValidationResult(passed=True, issues=issues)  # warn but don't block

        # Detect single-row comparison intent
        is_single_row_intent = bool(self._SINGLE_ROW_PATTERNS.search(q_lower))

        if not is_single_row_intent:
            return ValidationResult(passed=True, issues=[])

        # Parse SQL AST and check for GROUP BY on the outermost SELECT
        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            return ValidationResult(passed=True, issues=[])

        # Find outermost SELECT
        outermost = None
        for node in tree.walk():
            if isinstance(node, sqlglot.exp.Select):
                outermost = node
                break

        if outermost is None:
            return ValidationResult(passed=True, issues=[])

        has_group_by = outermost.args.get("group") is not None

        # Allow GROUP BY for legitimate trend queries (quarter/month breakdown)
        _is_trend_groupby = bool(re.search(
            r'GROUP\s+BY\s+(date_trunc|EXTRACT|TO_CHAR|\d+)',
            sql, re.IGNORECASE
        ))
        # Allow GROUP BY when FILTER aggregates are already present — the query
        # is using the correct FILTER pattern and GROUP BY just adds a breakdown
        # dimension (e.g. CASE WHEN status='CLOSED' GROUP BY that CASE alias).
        _has_filter_aggregates = bool(re.search(r'\bFILTER\s*\(', sql, re.IGNORECASE))

        if has_group_by and not _is_trend_groupby and not _has_filter_aggregates:
            # Build domain-specific retry message
            q_lower_shape = question.lower()
            _is_ca_comparison = bool(re.search(
                r'\braised.{0,20}closed\b|\bclosed.{0,20}raised\b'
                r'|\b(faster|slower|closure.{0,20}rate)\b', q_lower_shape))
            _is_time_comparison = bool(re.search(
                r'\b(vs\.?|versus|this.{0,10}(month|quarter).{0,10}last'
                r'|last.{0,10}(month|quarter).{0,10}this)\b', q_lower_shape))

            if _is_ca_comparison:
                _fix_example = (
                    "  SELECT 'Q2 2026' AS quarter,\n"
                    "    COUNT(*) AS total_raised,\n"
                    "    COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS closed_count\n"
                    "  FROM inspection_corrective_action\n"
                    "  WHERE target_close_out_date >= date_trunc('quarter', CURRENT_DATE);"
                )
                _msg = "Use FILTER aggregates for raised/closed comparison (single row, no GROUP BY):\n"
            elif _is_time_comparison:
                _fix_example = (
                    "  SELECT\n"
                    "    COUNT(*) FILTER (WHERE date_trunc('month',col)=date_trunc('month',NOW())) AS this_month,\n"
                    "    COUNT(*) FILTER (WHERE date_trunc('month',col)=date_trunc('month',NOW()-INTERVAL '1 month')) AS last_month\n"
                    "  FROM table WHERE ...;"
                )
                _msg = "Use FILTER aggregates for time comparison (single row, no GROUP BY):\n"
            else:
                # Non-comparison GROUP BY: likely simple aggregate with unnecessary grouping
                _fix_example = (
                    "  -- For simple counts/aggregates, do NOT add GROUP BY:\n"
                    "  SELECT COUNT(DISTINCT col) AS count FROM table WHERE ...;\n"
                    "  -- Only use GROUP BY when the question asks for breakdown BY something."
                )
                _msg = "WRONG_SHAPE: remove unnecessary GROUP BY for a simple aggregate query:\n"

            issue = ValidationIssue(
                "HARD_FAIL",
                "WRONG_SHAPE",
                _msg + _fix_example,
                retryable=True,
            )
            ValidatorStats.record("WRONG_SHAPE", is_retry_trigger=True)
            return ValidationResult(passed=False, issues=[issue])

        return ValidationResult(passed=True, issues=[])

    def validate_semantic(self, sql: str, question: str) -> list[ValidationIssue]:
        """
        Advisory heuristics.

        Returns SOFT_WARN issues only.
        These are for observability (logs, stats) only.
        They must NEVER be forwarded to the LLM retry prompt.
        They must NEVER cause retries.

        Callers: log these, count them via ValidatorStats, then discard.
        """
        warns: list[ValidationIssue] = []
        sql_lower = sql.lower()
        q_lower = question.lower()

        # Routing mismatch: asks about form answers but SQL avoids ai_answers
        asks_about_answers = bool(re.search(
            r'\b(answer|submission|response|submitted|filled|responded)\b', q_lower))
        uses_ai_answers = "ai_answers" in sql_lower
        uses_inspection = bool(re.search(
            r'\binspection_(report|corrective_action|schedule|cycle)\b', sql_lower))

        if asks_about_answers and not uses_inspection and not uses_ai_answers:
            warns.append(ValidationIssue(
                "SOFT_WARN", "ROUTING_MISMATCH",
                "Question mentions answers but SQL avoids ai_answers",
                detail="Consider get_answers tool or join via ai_answers.element_id",
                retryable=False,
            ))
            ValidatorStats.record("ROUTING_MISMATCH", is_retry_trigger=False)

        # UUID FK columns selected/grouped without their lookup JOIN.
        # Uses AST when available for precise SELECT clause detection.
        if _HAS_SQLGLOT:
            _check_uuid_leaks_ast(sql, self.UUID_FK_COLUMNS, warns)
        else:
            _check_uuid_leaks_regex(sql_lower, self.UUID_FK_COLUMNS, warns)

        # Required predicate checks — verify semantic constraints are present.
        # These are SOFT_WARN because the query may still be correct (e.g. user
        # is explicitly asking about DRAFT inspections).

        # "completed" inspections must include SUBMITTED or CLOSED
        _COMPLETED_Q = re.compile(
            r"\b(completed|finished|done|finalized)\s+(inspection|inspections?)\b"
            r"|\binspections?\s+(were\s+)?(completed|finished|done|finalized)\b"
            r"|\bhow\s+many\s+inspections?.{0,20}(completed|finished|done)\b",
            re.IGNORECASE,
        )
        if _COMPLETED_Q.search(question):
            has_submitted = "'SUBMITTED'" in sql.upper() or "\"SUBMITTED\"" in sql
            has_closed = "'CLOSED'" in sql.upper() or "\"CLOSED\"" in sql
            if not (has_submitted or has_closed):
                warns.append(ValidationIssue(
                    "SOFT_WARN", "MISSING_COMPLETED_STATUS",
                    "Query asks about 'completed' inspections but SQL has neither 'CLOSED' nor 'SUBMITTED'",
                    detail="completed_inspection = status IN ('CLOSED','SUBMITTED')",
                    retryable=False,
                ))
                ValidatorStats.record("MISSING_COMPLETED_STATUS", is_retry_trigger=False)

        # NOTE: "overdue mitigative actions" via date arithmetic is now a HARD_FAIL,
        # enforced in validate() (see OVERDUE_DATE_ARITHMETIC) — promoted out of this
        # advisory-only method because it was firing consistently (both 3B and 7B,
        # in eval) with no retry and no user-visible signal.

        # ── Business glossary term checks ─────────────────────────────────────
        if _HAS_GLOSSARY:
            for severity, msg in _check_glossary(question, sql):
                warns.append(ValidationIssue(
                    "SOFT_WARN", "GLOSSARY_VIOLATION", msg,
                    detail="business_glossary check",
                    retryable=False,
                ))
                ValidatorStats.record("GLOSSARY_VIOLATION", is_retry_trigger=False)

        return warns

    # ------------------------------------------------------------------ #
    # clean_sql — formatting only + structural ICA join fix
    # ------------------------------------------------------------------ #

    def clean_sql(self, sql: str) -> str:
        """
        Normalize raw LLM SQL output for execution.

        ALLOWED mutations (formatting / structural type-fixes only):
          ✓ Strip markdown fences and LLM formatting artifacts
          ✓ Normalize whitespace
          ✓ Extract SELECT/WITH from preamble prose
          ✓ Normalize trailing semicolon
          ✓ Fix reserved word aliases (is → isch)
          ✓ Fix ICA→IR join varchar/UUID type mismatch (structural, not semantic)

        NOT ALLOWED (these would change query semantics):
          ✗ Adding/removing WHERE conditions
          ✗ Modifying GROUP BY / HAVING / aggregates
          ✗ Adding/removing JOINs
          ✗ Injecting filters
          ✗ Rewriting ORDER BY

        The ICA→IR join fix is retained because:
          - It is a type-level structural error (VARCHAR vs UUID)
          - The wrong form ALWAYS causes a DB type mismatch error
          - The fix NEVER changes what data is returned — only fixes the join type
          - It has 4 variants covering deepseek's CAST/:: fallback attempts
        """
        # Strip markdown fences and LLM formatting artifacts
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```", "", sql)
        sql = re.sub(r"\[/?[A-Z_]+\]", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^###.*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"-\s*\[.*?\]\(.*?\)", "", sql)
        sql = re.sub(r"^[^\n;]*:\s*$", "", sql, flags=re.MULTILINE)
        sql = sql.strip()

        # Extract SELECT/WITH from any LLM preamble prose
        if "SELECT" in sql.upper():
            select_pos = sql.upper().find("SELECT")
            cte_match = re.search(
                r'\bWITH\s+(?:RECURSIVE\s+)?\w+\s+AS\s*\(',
                sql, re.IGNORECASE,
            )
            if cte_match and cte_match.start() < select_pos:
                sql = sql[cte_match.start():]
            else:
                sql = sql[select_pos:]

        # Truncate at first semicolon (LLM sometimes appends explanation after ;)
        semi = re.search(r";", sql)
        if semi:
            sql = sql[:semi.end()]

        # ICA→IR join structural fix (4 variants: direct, reversed, CAST, ::type)
        # and reserved-alias fix ('is' → 'isch') — both consumed from
        # core/business_rules.py's REGISTRY via _registry_autocorrects().
        # Rationale for the ICA/IR fix: ica.inspection_id is UUID FK;
        # ir.inspection_id is VARCHAR — comparing them is always a type error.
        _sql_before_autocorrect = sql
        sql = self._registry_autocorrects(sql)
        # fix_reserved_aliases (removed) additionally rewrote bare "is."
        # qualifiers anywhere in the query, once "is" was confirmed used as a
        # table alias — the registry's autocorrect_reserved_alias_is rule only
        # fixes the alias *declaration* site itself. No registry equivalent for
        # this broader cleanup; preserved here directly, same trigger condition.
        _is_alias_rule = next(
            (r for r in self._rules._autocorr_rules if r["id"] == "autocorrect_reserved_alias_is"),
            None,
        )
        if _is_alias_rule and _is_alias_rule["_sql_re"].search(_sql_before_autocorrect):
            sql = re.sub(r'\bis\.', 'isch.', sql)

        sql = re.sub(r"\s*\[/?[A-Z_]+\]\s*$", "", sql, flags=re.IGNORECASE).rstrip()

        if sql and not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"

        return sql

    # ------------------------------------------------------------------ #
    # AST repair — deterministic post-generation fixes via sqlglot
    # ------------------------------------------------------------------ #

    # Tables that have an `active` boolean soft-delete column.
    _SOFT_DELETE_TABLES: frozenset = frozenset({
        "inspection_report", "inspection_corrective_action", "facility",
        "users", "inspection_type", "inspection_sub_type",
        "inspector_portfolio", "inspector_portfolio_details",
        "frequency_definition", "po_details", "pr_details", "risk_level",
        "client", "project", "country", "region", "entity", "organisation",
        "inspection_schedule", "impact",
    })

    # Tables that NEVER have an active column — never inject for these.
    _NO_ACTIVE_TABLES: frozenset = frozenset({
        "ai_answers", "ai_questions", "role", "user_roles",
    })

    def ast_repair(self, sql: str) -> tuple:
        """
        Deterministic AST-level repairs using sqlglot. Every mutation is
        logged in the returned mutations list. Falls back to original SQL
        on parse failure — never corrupts.

        Repairs:
          1. Remove aa.active / aq.active  (columns don't exist)
          2. Inject {alias}.active = true  for soft-delete tables missing it
        """
        if not _HAS_SQLGLOT:
            return sql, []

        mutations: list = []

        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            return sql, []

        # ── 1. Drop hallucinated active columns on aliases that can't have one ──
        # Three cases, unioned into one alias->reason map. deepseek reflexively
        # appends `<alias>.active = true` to every table/CTE it references,
        # regardless of whether that alias actually has (or even IS) a
        # soft-delete-tracked table:
        #  (a) alias's table is in _NO_ACTIVE_TABLES — known real tables with
        #      no soft-delete column (ai_answers, ai_questions, role, etc.)
        #  (b) alias is a CTE whose own SELECT never outputs an "active"
        #      column — pre-aggregated derived tables were never soft-delete
        #      tracked in the first place, so `<cte_alias>.active` can never
        #      resolve to a real column no matter what the CTE's base table is
        #  (c) alias isn't introduced anywhere in the statement at all — same
        #      conservative whole-statement alias union as
        #      _check_undefined_table_aliases() below (see its docstring for
        #      why: false negatives are OK, false positives would block valid
        #      SQL). This is typically leftover from copying a WHERE fragment
        #      out of an unrelated few-shot example without its JOIN.
        _bogus_active_aliases: dict = {}

        for node in tree.find_all(exp.Table):
            if node.name and node.name.lower() in self._NO_ACTIVE_TABLES:
                _bogus_active_aliases[(node.alias or node.name).lower()] = (
                    "table has no active column"
                )

        _introduced: set = set()
        for t in tree.find_all(exp.Table):
            if t.alias:
                _introduced.add(t.alias.lower())
            if t.name:
                _introduced.add(t.name.lower())
        for cte in tree.find_all(exp.CTE):
            if cte.alias:
                _introduced.add(cte.alias.lower())
        for sq in tree.find_all(exp.Subquery):
            if sq.alias:
                _introduced.add(sq.alias.lower())

        # cte.alias is the CTE's OWN declared name (e.g. "po_by_ica" in
        # `WITH po_by_ica AS (...)`) — NOT the alias used to reference it in a
        # JOIN (e.g. the "p" in `LEFT JOIN po_by_ica p ON ...`, which is a
        # separate exp.Table node). Column qualifiers use the JOIN alias, so
        # this must resolve CTE-name -> JOIN-alias(es) via exp.Table, exactly
        # like the _NO_ACTIVE_TABLES loop above does for real tables.
        _cte_names_without_active: set = set()
        for cte in tree.find_all(exp.CTE):
            if cte.alias and isinstance(cte.this, exp.Select):
                has_active = any(
                    e.output_name and e.output_name.lower() == "active"
                    for e in cte.this.expressions
                )
                if not has_active:
                    _cte_names_without_active.add(cte.alias.lower())
        for node in tree.find_all(exp.Table):
            if node.name and node.name.lower() in _cte_names_without_active:
                _bogus_active_aliases.setdefault(
                    (node.alias or node.name).lower(), "CTE has no active column"
                )

        for eq in tree.find_all(exp.EQ):
            if (isinstance(eq.left, exp.Column)
                    and eq.left.name.lower() == "active"
                    and eq.left.table
                    and eq.left.table.lower() not in _introduced):
                _bogus_active_aliases.setdefault(
                    eq.left.table.lower(),
                    "alias not joined/defined anywhere in this query",
                )

        _no_active_aliases: set = set(_bogus_active_aliases)

        if _no_active_aliases:
            def _drop_ai_active(node):
                if (isinstance(node, exp.EQ)
                        and isinstance(node.left, exp.Column)
                        and node.left.name.lower() == "active"
                        and node.left.table
                        and node.left.table.lower() in _no_active_aliases):
                    reason = _bogus_active_aliases.get(
                        node.left.table.lower(), "table has no active column"
                    )
                    mutations.append(
                        f"Removed {node.left.table}.active ({reason})"
                    )
                    return exp.true()
                return node

            tree = tree.transform(_drop_ai_active)
            # Clean up AND(TRUE, x) → x and AND(x, TRUE) → x
            def _simplify_and(node):
                if isinstance(node, exp.And):
                    if isinstance(node.left, exp.Boolean) and node.left.this:
                        return node.right
                    if isinstance(node.right, exp.Boolean) and node.right.this:
                        return node.left
                return node
            tree = tree.transform(_simplify_and)

        # ── 2. Inject active = true for soft-delete tables ────────────────
        def _direct_tables(sel: exp.Select) -> dict:
            """
            alias -> table_name for tables that are NEVER NULL for a matched
            row in this SELECT (FROM + INNER-joined tables) — safe to filter
            via WHERE without changing which rows the query returns.

            A LEFT-joined table's own alias is always excluded (it can be
            NULL). A RIGHT/FULL join makes every table accumulated BEFORE it
            nullable too (that's the side RIGHT/FULL JOIN pads with NULLs),
            so those are dropped from the safe set at that point; a RIGHT
            join's own right-hand table is NOT nullable (all its rows are
            kept) and stays in the safe set, while a FULL join's own table
            is nullable like the LEFT-joined case.
            """
            def _alias(t):
                return (t.alias or t.name).lower() if isinstance(t, exp.Table) and t.name else None
            def _tname(t):
                return t.name.lower() if isinstance(t, exp.Table) and t.name else None

            result = {}
            # sqlglot v20+ uses 'from_' key; older versions use 'from'
            from_clause = sel.args.get("from_") or sel.args.get("from")
            if from_clause and _alias(from_clause.this):
                result[_alias(from_clause.this)] = _tname(from_clause.this)

            for j in (sel.args.get("joins") or []):
                _side = (j.args.get("side") or "").upper()
                if _side == "LEFT":
                    continue  # this join's table is nullable — never mark safe
                if _side in ("RIGHT", "FULL"):
                    result.clear()  # everything accumulated so far is now nullable
                    if _side == "FULL":
                        continue  # this join's own table is nullable too
                _alias_ = _alias(j.this)
                if _alias_:
                    result[_alias_] = _tname(j.this)
            return result

        def _active_aliases(where) -> set:
            """Aliases already filtered by active = true/TRUE/1 (qualified or bare)."""
            found = set()
            if where is None:
                return found
            for eq in where.find_all(exp.EQ):
                if (isinstance(eq.left, exp.Column)
                        and eq.left.name.lower() == "active"):
                    right = eq.right
                    is_true = (
                        (isinstance(right, exp.Boolean) and right.this is True)
                        or (isinstance(right, exp.Literal)
                            and str(right.this).lower() in ("1", "true"))
                    )
                    if is_true:
                        if eq.left.table:
                            found.add(eq.left.table.lower())
                        else:
                            # Bare `active = true` without alias covers all tables
                            # in this scope — mark with sentinel so we skip injection
                            found.add("__bare__")
            return found

        def _inject_active(sel: exp.Select) -> exp.Select:
            direct = _direct_tables(sel)
            existing = _active_aliases(sel.args.get("where"))
            # If bare `active = true` present, skip injection — it already filters
            if "__bare__" in existing:
                return sel
            for alias, tname in sorted(direct.items()):
                if (tname in self._SOFT_DELETE_TABLES
                        and alias not in existing):
                    cond = exp.EQ(
                        this=exp.Column(
                            this=exp.Identifier(this="active"),
                            table=exp.Identifier(this=alias),
                        ),
                        expression=exp.true(),
                    )
                    sel = sel.where(cond)
                    mutations.append(f"Injected {alias}.active = true ({tname})")

            # For LEFT/FULL-joined soft-delete tables (excluded from `direct`
            # above because they're nullable), the active filter belongs in
            # the JOIN's own ON clause, not WHERE. This still hides
            # soft-deleted matches but — unlike a WHERE-clause filter — doesn't
            # turn the outer join into an inner join, so rows with no match at
            # all (genuinely NULL on that side) are still kept.
            #
            # The LLM sometimes writes `alias.active = true` in WHERE itself
            # for one of these tables (it dutifully adds the filter it was
            # told to, without realizing that placement breaks outer-join
            # semantics — e.g. `LEFT JOIN organisation org ON ... WHERE
            # org.active = true` silently drops every row whose org is
            # genuinely NULL). `_relocate_aliases` collects those so they can
            # be stripped from WHERE and re-added to the correct ON clause
            # below, instead of just being left in place as "already handled".
            _relocate_aliases: set = set()
            for j in (sel.args.get("joins") or []):
                _join_side = (j.args.get("side") or "").upper()
                # LEFT: the joined table is nullable. FULL: the joined table is
                # also nullable, and ON-clause injection is equally safe there —
                # a false match (soft-deleted row) just falls back to NULL on
                # that side, same as a genuine non-match. (The FROM-side of a
                # FULL JOIN is nullable too, but that side isn't reachable from
                # this loop — FULL JOIN doesn't appear anywhere in this
                # codebase's generated SQL patterns, so it's not worth the
                # extra complexity to handle here.)
                if _join_side not in ("LEFT", "FULL"):
                    continue
                t = j.this
                if not (isinstance(t, exp.Table) and t.name):
                    continue
                alias = (t.alias or t.name).lower()
                tname = t.name.lower()
                if tname not in self._SOFT_DELETE_TABLES:
                    continue
                on_cond = j.args.get("on")
                already_filtered = on_cond is not None and any(
                    isinstance(eq.left, exp.Column)
                    and eq.left.name.lower() == "active"
                    and eq.left.table and eq.left.table.lower() == alias
                    for eq in on_cond.find_all(exp.EQ)
                )
                if already_filtered:
                    continue
                cond = exp.EQ(
                    this=exp.Column(
                        this=exp.Identifier(this="active"),
                        table=exp.Identifier(this=alias),
                    ),
                    expression=exp.true(),
                )
                j.set("on", exp.And(this=on_cond, expression=cond) if on_cond else cond)
                if alias in existing:
                    _relocate_aliases.add(alias)
                    mutations.append(
                        f"Relocated {alias}.active = true from WHERE into {_join_side} "
                        f"JOIN ON clause ({tname}) — WHERE placement silently drops "
                        f"legitimate no-match rows on an outer join"
                    )
                else:
                    mutations.append(
                        f"Injected {alias}.active = true into {_join_side} JOIN ON clause ({tname})"
                    )

            if _relocate_aliases:
                def _strip_relocated(node):
                    if (isinstance(node, exp.EQ)
                            and isinstance(node.left, exp.Column)
                            and node.left.name.lower() == "active"
                            and node.left.table
                            and node.left.table.lower() in _relocate_aliases):
                        return exp.true()
                    return node
                where = sel.args.get("where")
                if where is not None:
                    new_where = where.this.transform(_strip_relocated)
                    def _simplify(n):
                        if isinstance(n, exp.And):
                            if isinstance(n.left, exp.Boolean) and n.left.this:
                                return n.right
                            if isinstance(n.right, exp.Boolean) and n.right.this:
                                return n.left
                        return n
                    new_where = new_where.transform(_simplify)
                    if isinstance(new_where, exp.Boolean) and new_where.this:
                        sel.set("where", None)
                    else:
                        where.set("this", new_where)
            return sel

        # ── 4. Strip date_trunc wrapping bare string literals in comparisons ─
        # deepseek generates: ir.submitted_on >= date_trunc('month', '2026-03-01')
        # PostgreSQL can't infer the type of the bare string inside date_trunc
        # → raises "function date_trunc(unknown, unknown) is ambiguous".
        # Fix: replace with CAST('YYYY-MM-DD' AS DATE) which is unambiguous.
        # sqlglot parses date_trunc('month', 'literal') as TimestampTrunc.
        def _fix_date_trunc_literal(node):
            if not isinstance(node, exp.TimestampTrunc):
                return node
            date_arg = node.args.get("this")
            if isinstance(date_arg, exp.Literal) and date_arg.is_string:
                mutations.append(
                    f"Replaced date_trunc('month', '{date_arg.this}') "
                    f"→ CAST('{date_arg.this}' AS DATE)"
                )
                return exp.Cast(
                    this=exp.Literal.string(date_arg.this),
                    to=exp.DataType(this=exp.DataType.Type.DATE),
                )
            return node

        tree = tree.transform(_fix_date_trunc_literal)

        tree = tree.transform(
            lambda n: _inject_active(n) if isinstance(n, exp.Select) else n
        )

        # ── 5. Fix sched.scheduled_date → sched.schedule_date ────────────
        # deepseek training bias: uses 'scheduled_date' (past participle) but
        # the actual column on inspection_schedule is 'schedule_date'.
        _sched_aliases: set = set()
        for node in tree.find_all(exp.Table):
            if node.name and node.name.lower() == "inspection_schedule":
                _sched_aliases.add((node.alias or node.name).lower())

        if _sched_aliases:
            def _fix_scheduled_date(node):
                if (isinstance(node, exp.Column)
                        and node.name.lower() == "scheduled_date"
                        and node.table
                        and node.table.lower() in _sched_aliases):
                    mutations.append(
                        f"Renamed {node.table}.scheduled_date → {node.table}.schedule_date"
                    )
                    return exp.Column(
                        this=exp.Identifier(this="schedule_date"),
                        table=exp.Identifier(this=node.table),
                    )
                return node
            tree = tree.transform(_fix_scheduled_date)
        # ── 6. Drop ORDER BY / GROUP BY terms referencing an undefined alias ──
        # Weak SQL-gen models (e.g. small local models) frequently copy a seed
        # example's ORDER BY/GROUP BY clause referencing a SELECT alias they
        # themselves dropped from the SELECT list one line earlier — e.g.
        # `SELECT ... FROM ... ORDER BY last_inspection_date` where
        # `last_inspection_date` was never actually selected. Postgres would
        # reject this with "column ... does not exist" / the validator's
        # UNDEFINED_ALIAS check. Removing the dangling sort/group term is
        # always safe: it loses only ordering/grouping, never row content.
        # Deliberately conservative — UNSAFE to touch SELECT columns, WHERE,
        # HAVING, or JOIN ON referencing an undefined alias (those change
        # which rows/values come back), so those are left for the validator
        # to hard-fail and trigger a real LLM retry instead.
        _introduced_all: set = set()
        for t in tree.find_all(exp.Table):
            if t.alias:
                _introduced_all.add(t.alias.lower())
            if t.name:
                _introduced_all.add(t.name.lower())
        for cte in tree.find_all(exp.CTE):
            if cte.alias:
                _introduced_all.add(cte.alias.lower())
        for sq in tree.find_all(exp.Subquery):
            if sq.alias:
                _introduced_all.add(sq.alias.lower())

        if _introduced_all:
            _undefined_all: set = set()
            for col in tree.find_all(exp.Column):
                if col.table and col.table.lower() not in _introduced_all:
                    _undefined_all.add(col.table.lower())

            if _undefined_all:
                def _refs_undefined(node) -> bool:
                    return any(
                        isinstance(c, exp.Column)
                        and c.table and c.table.lower() in _undefined_all
                        for c in node.find_all(exp.Column)
                    )

                for sel in tree.find_all(exp.Select):
                    order = sel.args.get("order")
                    if order is not None:
                        kept = [o for o in order.expressions if not _refs_undefined(o)]
                        if len(kept) < len(order.expressions):
                            for o in order.expressions:
                                if _refs_undefined(o):
                                    mutations.append(
                                        f"Removed ORDER BY term referencing "
                                        f"undefined alias: {o.sql(dialect='postgres')}"
                                    )
                            sel.set("order", exp.Order(expressions=kept) if kept else None)

                    group = sel.args.get("group")
                    if group is not None:
                        kept_g = [g for g in group.expressions if not _refs_undefined(g)]
                        if len(kept_g) < len(group.expressions):
                            for g in group.expressions:
                                if _refs_undefined(g):
                                    mutations.append(
                                        f"Removed GROUP BY term referencing "
                                        f"undefined alias: {g.sql(dialect='postgres')}"
                                    )
                            sel.set("group", exp.Group(expressions=kept_g) if kept_g else None)

        if not mutations:
            return sql, []

        try:
            repaired = tree.sql(dialect="postgres", pretty=False)
            if repaired and not repaired.rstrip().endswith(";"):
                repaired = repaired.rstrip() + ";"
            return repaired, mutations
        except Exception:
            return sql, []

    # facility is intentionally excluded — public reference data.
    _SECONDARY_SCOPE_TABLES: dict = {
        'inspection_cycle': {
            'mode': 'forward', 'fk_col': 'cycle_id',
        },
        'inspection_corrective_action': {
            'mode': 'reverse', 'own_col': 'inspection_id',
        },
        'po_details': {
            'mode': 'reverse2', 'own_col': 'mitigative_action_uuid',
            'mid_table': 'inspection_corrective_action', 'mid_fk': 'inspection_id',
        },
        'pr_details': {
            'mode': 'reverse2', 'own_col': 'mitigative_action_uuid',
            'mid_table': 'inspection_corrective_action', 'mid_fk': 'inspection_id',
        },
        'inspection_schedule': {
            'mode': 'direct',
            'column_map': {'inspector_user_id': 'inspector_id', 'inspectee_user_id': 'inspectee_id'},
        },
        'inspector_portfolio': {
            'mode': 'inspector_only', 'own_col': 'user_id',
        },
        'inspector_portfolio_details': {
            'mode': 'inspector_only_reverse', 'own_col': 'portfolio_id',
            'mid_table': 'inspector_portfolio', 'mid_col': 'user_id',
        },
        'users': {
            'mode': 'connected_users', 'own_col': 'id',
        },
    }

    _USER_SUBQ_RE = re.compile(
        r"\(SELECT id FROM users WHERE id_fk = \d+ AND active = true AND is_synced = true\)"
    )

    @classmethod
    def _build_secondary_predicate(cls, tbl_name: str, spec: dict, alias: str,
                                    scope_filter: str, ir_predicate: str) -> str:
        mode = spec['mode']

        if mode == 'forward':
            return (
                f"{alias}.id IN (SELECT DISTINCT {spec['fk_col']} "
                f"FROM inspection_report WHERE {ir_predicate} AND active = true)"
            )

        if mode == 'reverse':
            return (
                f"{alias}.{spec['own_col']} IN (SELECT id FROM inspection_report "
                f"WHERE {ir_predicate} AND active = true)"
            )

        if mode == 'reverse2':
            return (
                f"{alias}.{spec['own_col']} IN ("
                f"SELECT _sc_mid.id FROM {spec['mid_table']} _sc_mid "
                f"JOIN inspection_report _sc_ir ON _sc_mid.{spec['mid_fk']} = _sc_ir.id "
                f"WHERE {ir_predicate} AND _sc_mid.active = true AND _sc_ir.active = true)"
            )

        if mode == 'direct':
            pred = scope_filter
            for old_col, new_col in spec['column_map'].items():
                pred = pred.replace(f"ir.{old_col}", f"{alias}.{new_col}")
            return pred

        if mode == 'inspector_only':
            if 'inspector_user_id' not in scope_filter:
                return "1 = 0"
            m = cls._USER_SUBQ_RE.search(scope_filter)
            subq = m.group(0) if m else "(SELECT NULL)"
            return f"{alias}.{spec['own_col']} = {subq}"

        if mode == 'inspector_only_reverse':
            if 'inspector_user_id' not in scope_filter:
                return "1 = 0"
            m = cls._USER_SUBQ_RE.search(scope_filter)
            subq = m.group(0) if m else "(SELECT NULL)"
            return (
                f"{alias}.{spec['own_col']} IN (SELECT id FROM {spec['mid_table']} "
                f"WHERE {spec['mid_col']} = {subq} AND active = true)"
            )

        if mode == 'connected_users':
            return (
                f"{alias}.{spec['own_col']} IN ("
                f"SELECT inspector_user_id FROM inspection_report WHERE {ir_predicate} AND active = true "
                f"UNION "
                f"SELECT inspectee_user_id FROM inspection_report WHERE {ir_predicate} AND active = true)"
            )

        raise ValueError(f"unknown secondary-scope mode {mode!r} for {tbl_name}")

    def enforce_scope(self, sql: str, scope_filter: str) -> str:
        """Hard data-visibility guard — AST-level injection, always applied.

        scope_filter == ''       → unrestricted, return SQL unchanged
        scope_filter == 'BLOCK'  → raise ValueError('SCOPE_BLOCKED')
        anything else            → inject into every Select that directly
                                   FROM/JOINs inspection_report, or a
                                   secondary-scoped table (see
                                   _SECONDARY_SCOPE_TABLES / _build_secondary_predicate)

        Raises ValueError('SCOPE_BLOCKED') — caller converts to HTTP 403.
        """
        if not scope_filter:
            return sql
        if scope_filter == 'BLOCK':
            _log_security.warning("[SCOPE] enforce_scope called with BLOCK — raising SCOPE_BLOCKED")
            raise ValueError('SCOPE_BLOCKED')

        sql_lower = sql.lower()
        has_ir    = 'inspection_report' in sql_lower

        # Strip "ir." alias so the predicate works inside a raw subquery
        # (e.g., dual-role "(ir.inspector_user_id=X OR ir.inspectee_user_id=X)"
        #  → "(inspector_user_id=X OR inspectee_user_id=X)")
        ir_predicate = scope_filter.replace('ir.', '')

        secondary_tables = [
            (tbl, spec) for tbl, spec in self._SECONDARY_SCOPE_TABLES.items()
            if tbl in sql_lower
        ]

        if not has_ir and not secondary_tables:
            # Pure reference tables (frequency_definition, inspection_type, etc.)
            # carry no row-level owner — no scope injection required.
            return sql

        _log_security.info(
            "[SCOPE] injecting scope filter: via=%s filter=%s",
            (["inspection_report"] if has_ir else []) + [t for t, _ in secondary_tables],
            scope_filter[:120],
        )

        # ── Primary path: AST injection via sqlglot ──────────────────────────
        if _HAS_SQLGLOT:
            try:
                tree       = sqlglot.parse_one(sql, dialect="postgres")
                scope_expr = sqlglot.parse_one(scope_filter, dialect="postgres")

                def _alias_for_table(sel: exp.Select, tbl_name: str):
                    """Return alias if sel directly FROM/JOINs tbl_name, else None."""
                    # sqlglot v20+ uses 'from_' key; older versions use 'from'.
                    # Matches the fallback already used by _direct_tables() above —
                    # without it this always returns None on the installed
                    # sqlglot version, silently disabling scope injection.
                    from_clause = sel.args.get("from_") or sel.args.get("from")
                    if from_clause and isinstance(from_clause.this, exp.Table):
                        if from_clause.this.name.lower() == tbl_name:
                            return from_clause.this.alias or tbl_name
                    for join in (sel.args.get("joins") or []):
                        if isinstance(join.this, exp.Table):
                            if join.this.name.lower() == tbl_name:
                                return join.this.alias or tbl_name
                    return None

                # Each SELECT node (including every branch of a UNION, and every
                # subquery) is scoped independently — a query can mix a branch
                # that joins inspection_report with a branch that reaches a
                # secondary-scoped table bare; both must be caught on their
                # own terms, not decided once for the whole statement.
                def _inject(node):
                    if not isinstance(node, exp.Select):
                        return node
                    if _alias_for_table(node, 'inspection_report'):
                        return node.where(scope_expr.copy())
                    for tbl_name, spec in secondary_tables:
                        alias = _alias_for_table(node, tbl_name)
                        if alias:
                            subq_where = self._build_secondary_predicate(
                                tbl_name, spec, alias, scope_filter, ir_predicate
                            )
                            node = node.where(
                                sqlglot.parse_one(subq_where, dialect="postgres")
                            )
                    return node

                rewritten = tree.transform(_inject).sql(dialect="postgres", pretty=False)

                if rewritten and not rewritten.rstrip().endswith(";"):
                    rewritten = rewritten.rstrip() + ";"
                return rewritten

            except Exception:
                logging.warning("enforce_scope: AST injection failed, using string fallback",
                                exc_info=True)

        # ── Fallback: string injection (no sqlglot) ───────────────────────────
        sql_work  = sql.rstrip().rstrip(";").rstrip()
        has_where = bool(re.search(r"\bWHERE\b", sql_work, re.IGNORECASE))
        connector = "AND" if has_where else "WHERE"
        term      = re.search(
            r"\b(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT)\b", sql_work, re.IGNORECASE
        )

        parts = ([scope_filter] if has_ir else []) + [
            self._build_secondary_predicate(tbl_name, spec, tbl_name, scope_filter, ir_predicate)
            for tbl_name, spec in secondary_tables
        ]
        fragment = " AND ".join(f"({p})" for p in parts)

        if term:
            pos = term.start()
            return f"{sql_work[:pos].rstrip()}\n  {connector} {fragment}\n{sql_work[pos:]};"
        return f"{sql_work}\n  {connector} {fragment};"


# ---------------------------------------------------------------------------
# AST helpers (sqlglot) — used when available, regex fallback otherwise
# ---------------------------------------------------------------------------

def _strip_nested_parens(sql: str) -> str:
    """
    Replace content inside balanced parentheses with spaces, preserving string length.
    Used to isolate the outermost SELECT clause before regex matching,
    preventing false-positive matches on subquery content.

    Example:
      "SELECT ir.id FROM t WHERE x = (SELECT ir.id FROM t2)"
      → "SELECT ir.id FROM t WHERE x =                      "
    The outer ir.id is preserved; the inner one (in the subquery) is blanked.
    """
    result = list(sql)
    depth = 0
    for i, ch in enumerate(sql):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        elif depth > 0:
            result[i] = ' '
    return ''.join(result)


def _check_undefined_table_aliases(sql: str, issues: list[ValidationIssue]) -> None:
    """
    Use sqlglot AST to verify every table-qualified column reference (alias.col)
    corresponds to a table/alias actually introduced by FROM/JOIN in some scope
    that contains it (including CTEs and subqueries).

    This catches the class of error where Deepseek drops a JOIN but leaves a
    WHERE/SELECT reference to that table's alias — SQL that LOOKS syntactically
    plausible to regex checks but Postgres rejects with:
        missing FROM-clause entry for table "ir"

    Conservative by design — false negatives are OK (we'd rather miss a case
    than block valid SQL): we collect ALL aliases/table-names introduced
    ANYWHERE in the statement (any scope) into one set, plus a few universal
    aliases that legitimately appear without local definition in some dialects.
    A reference is only flagged if its qualifier appears NOWHERE as an
    introduced table/alias/CTE-name in the entire statement.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return

    introduced: set[str] = set()

    # Every table reference contributes its alias (if any) and its bare name.
    for t in tree.find_all(exp.Table):
        if t.alias:
            introduced.add(t.alias.lower())
        # bare table name is also a valid qualifier (e.g. "inspection_report.facility_id")
        if t.name:
            introduced.add(t.name.lower())

    # CTE names (WITH foo AS (...) — "foo" can be referenced like a table)
    for cte in tree.find_all(exp.CTE):
        if cte.alias:
            introduced.add(cte.alias.lower())

    # Subquery aliases: (SELECT ...) AS sub
    for sq in tree.find_all(exp.Subquery):
        if sq.alias:
            introduced.add(sq.alias.lower())

    if not introduced:
        return  # couldn't determine scope — fail safe, don't block

    # Collect all qualified column references: <alias>.<col>
    undefined: set[str] = set()
    for col in tree.find_all(exp.Column):
        if col.table and col.table.lower() not in introduced:
            undefined.add(col.table.lower())

    if undefined:
        bad = ", ".join(sorted(undefined))
        issues.append(ValidationIssue(
            "HARD_FAIL", "UNDEFINED_ALIAS",
            f"Table alias '{bad}' is used in WHERE/SELECT but not introduced "
            f"by any FROM/JOIN — add the missing JOIN or remove the reference",
            detail=(
                f"Aliases referenced but never joined: {bad}. "
                f"Tables/aliases actually in scope: {', '.join(sorted(introduced))}. "
                f"Postgres would reject this with 'missing FROM-clause entry for table \"{sorted(undefined)[0]}\"'."
            ),
        ))
        ValidatorStats.record("UNDEFINED_ALIAS", is_retry_trigger=True)


def _check_ir_id_in_select_ast(sql: str, issues: list[ValidationIssue]) -> None:
    """
    Use sqlglot AST to detect ir.id in the OUTERMOST SELECT column list only.

    CRITICAL: only check the top-level Select node's direct expressions.
    find_all(exp.Select) walks ALL Select nodes including subqueries — if we
    used that, we would false-positive on valid patterns like:
        WHERE aa.inspection_report_id = (SELECT ir.id FROM inspection_report ir ...)
    where ir.id is correctly used as a UUID FK key, never displayed.

    CTE structure: With → this (the final SELECT after WITH definitions).
    Simple query: Select directly.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")

        # Navigate to the outermost SELECT only — do NOT use find_all()
        top_select = None
        if isinstance(tree, exp.Select):
            top_select = tree
        elif isinstance(tree, exp.With):
            # CTE: tree.this is the final SELECT after the CTE definitions
            final = tree.this
            if isinstance(final, exp.Select):
                top_select = final
            elif isinstance(final, exp.Subquery):
                inner = final.this
                if isinstance(inner, exp.Select):
                    top_select = inner

        if top_select is None:
            return

        # Inspect only the direct column expressions of the outermost SELECT.
        # Do NOT recurse into nested Subquery or Anonymous nodes.
        for col_expr in top_select.expressions:
            # Unwrap alias: SELECT ir.id AS something → col_expr is Alias, col_expr.this is Column
            col = col_expr.this if isinstance(col_expr, exp.Alias) else col_expr
            if isinstance(col, exp.Column):
                if (col.table and col.table.lower() == "ir"
                        and col.name.lower() == "id"):
                    issues.append(ValidationIssue(
                        "HARD_FAIL", "UUID_IN_SELECT",
                        "ir.id is the UUID primary key — use ir.inspection_id for display",
                        detail="ir.inspection_id is the human-readable varchar '2026/04/ST001/INS001'",
                    ))
                    ValidatorStats.record("UUID_IN_SELECT", is_retry_trigger=True)
                    return  # one report is enough

    except Exception:
        # AST parse failed — fail safe: do not block on uncertainty
        pass


def _check_uuid_leaks_ast(
    sql: str,
    uuid_fk_cols: dict,
    warns: list[ValidationIssue],
) -> None:
    """
    Use sqlglot AST to detect UUID FK columns selected/grouped without their lookup JOIN.
    AST approach avoids the DOTALL regex false positive where a subquery's GROUP BY
    triggers a warning on a parent query that correctly uses the lookup JOIN.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        # Fallback to regex on parse failure
        _check_uuid_leaks_regex(sql.lower(), uuid_fk_cols, warns)
        return

    # Collect all table names/aliases present in any FROM/JOIN
    joined_tables: set[str] = set()
    for node in tree.find_all(exp.Table):
        if node.name:
            joined_tables.add(node.name.lower())

    # Check outermost SELECT columns and GROUP BY for raw FK column references
    for node in tree.find_all(exp.Select):
        select_cols = {
            c.name.lower()
            for c in node.expressions
            if isinstance(c, exp.Column)
        }
        group_cols: set[str] = set()
        group = node.find(exp.Group)
        if group:
            group_cols = {
                c.name.lower()
                for c in group.find_all(exp.Column)
            }

        exposed = select_cols | group_cols

        for fk_col, (lookup_table, alias, select_expr) in uuid_fk_cols.items():
            if fk_col in exposed and lookup_table.lower() not in joined_tables:
                warns.append(ValidationIssue(
                    "SOFT_WARN", "UUID_LEAK",
                    f"'{fk_col}' selected/grouped without joining '{lookup_table}' — returns raw UUIDs",
                    detail=f"Fix: JOIN {lookup_table} {alias} ON ...{fk_col} = {alias}.id  "
                           f"SELECT {select_expr}",
                    retryable=False,
                ))
                ValidatorStats.record("UUID_LEAK", is_retry_trigger=False)


def _check_uuid_leaks_regex(
    sql_lower: str,
    uuid_fk_cols: dict,
    warns: list[ValidationIssue],
) -> None:
    """
    Regex fallback for UUID leak detection when sqlglot is unavailable.
    Less precise than AST — restricts DOTALL scope to reduce false positives.
    """
    for fk_col, (lookup_table, alias, select_expr) in uuid_fk_cols.items():
        # Check SELECT list only — use non-greedy, stop at first newline cluster
        col_in_select = bool(re.search(
            rf'\bSELECT\b[^\n]{{0,200}}\b{re.escape(fk_col)}\b',
            sql_lower, re.IGNORECASE,
        ))
        col_in_groupby = bool(re.search(
            rf'\bGROUP\s+BY\b[^\n]{{0,200}}\b{re.escape(fk_col)}\b',
            sql_lower, re.IGNORECASE,
        ))
        lookup_joined = lookup_table.lower() in sql_lower

        if (col_in_select or col_in_groupby) and not lookup_joined:
            warns.append(ValidationIssue(
                "SOFT_WARN", "UUID_LEAK",
                f"'{fk_col}' selected/grouped without joining '{lookup_table}' — returns raw UUIDs",
                detail=f"Fix: JOIN {lookup_table} {alias} ON ...{fk_col} = {alias}.id  "
                       f"SELECT {select_expr}",
                retryable=False,
            ))
            ValidatorStats.record("UUID_LEAK", is_retry_trigger=False)