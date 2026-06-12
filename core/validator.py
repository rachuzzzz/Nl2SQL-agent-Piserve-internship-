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

logger = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------
    # Backward compatibility shim
    # Remove once all callers migrate to ValidationResult.
    # ------------------------------------------------------------------
    def as_legacy_tuple(self) -> tuple[bool, list[str]]:
        """
        Returns (passed, errors_list) matching the old validate() return type.
        Hard fails → error strings.
        Soft warns → prefixed with 'WARNING:' to preserve old filtering logic.
        """
        hard = [str(i) for i in self.hard_fails]
        soft = [f"WARNING: {i.message}" for i in self.soft_warns]
        return self.passed, hard + soft


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

    # Tables the LLM may hallucinate that do not exist.
    HALLUCINATED_TABLES: set[str] = {
        # Singular forms — most common mistake
        "ai_question", "ai_answer",
        # fb_* hallucinations
        "fb_users", "fb_user",
        "fb_answers", "fb_answer", "fb_form_answer", "fb_form_answers",
        "fb_submissions", "fb_submission",
        "fb_responses", "fb_response",
    }

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
            # inspection_schedule has no active column
            "active":         "inspection_schedule has no active column — no soft-delete on this table",
            # inspection_id does not exist — use inspection_report_id
            "inspection_id":  "inspection_schedule has no inspection_id — use inspection_report_id FK",
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
        "actual_date":        "does not exist on inspection_corrective_action — use close_on",
        # mitigative_action_uuid is on po_details/pr_details — only block if qualified as ica.mitigative_action_uuid
        # "po.mitigative_action_uuid = ica.id" is CORRECT and must NOT be blocked
        "ica.mitigative_action_uuid": "does not exist on inspection_corrective_action — use po_details.mitigative_action_uuid = ica.id",
        "completion_date":    "does not exist on inspection_corrective_action — use close_on or completed_on",
        "resolution_date":    "does not exist on inspection_corrective_action — use close_on",
        "impact":             "does not exist as a column on ica — impact data is in ai_answers: JOIN ai_questions aq ON aa.element_id = aq.element_id WHERE aq.label = 'Impact'",
        # Other ghost columns seen in traces
        "corrective_action_description": "does not exist — use ica.corrective_action (the text field)",
    }

    # Subset of ICA ghost columns that are also table names — only check qualified refs.
    # If we check unqualified, we'd false-positive on "JOIN risk_level rl" clauses.
    _ICA_QUALIFIED_ONLY_COLS: frozenset = frozenset({"risk_level", "impact"})

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
        # inspection_schedule.inspection_id — does not exist
        (r'\bisched\.inspection_id\b|\binspection_schedule\.inspection_id\b',
         'inspection_schedule has no inspection_id — use inspection_report_id FK'),
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
    }

    # Precompiled patterns — class-level, built once
    _DYNAMIC_TABLE_RE = re.compile(
        r'\bfb_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}\b'
    )
    _HARDCODED_YEAR_RE = re.compile(
        r"\bextract\s*\(\s*year\s+from\s+\w[\w.]*\s*\)\s*=\s*(20\d\d)\b",
        re.IGNORECASE,
    )
    _DML_RE = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE)\b",
        re.IGNORECASE,
    )
    _AGG_RE = re.compile(
        r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(",
        re.IGNORECASE,
    )
    _LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)
    _CLOSE_ON_ARITH_RE = re.compile(
        r'\bclose_on\s*[-+]\s*\w|\bextract\s*\(.*?\bclose_on\b',
        re.IGNORECASE | re.DOTALL,
    )

    # ------------------------------------------------------------------ #
    # Primary validation — HARD_FAIL vs SOFT_WARN separation
    # ------------------------------------------------------------------ #

    def validate(self, sql: str) -> ValidationResult:
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

        # ── 2. DML/DDL — non-retryable, return immediately ───────────────
        if self._DML_RE.search(sql):
            issue = ValidationIssue(
                "HARD_FAIL", "DML_BLOCKED",
                "Only SELECT queries are allowed — DML/DDL detected",
                retryable=False,
            )
            ValidatorStats.record("DML_BLOCKED", is_retry_trigger=False)
            return ValidationResult(passed=False, issues=[issue])

        sql_lower = sql.lower()

        # ── 3. Hallucinated tables ────────────────────────────────────────
        for table in self.HALLUCINATED_TABLES:
            if re.search(rf"\b{re.escape(table)}\b", sql_lower):
                issues.append(ValidationIssue(
                    "HARD_FAIL", "UNKNOWN_TABLE",
                    f"Table '{table}' does not exist — use ai_answers / ai_questions (plural)",
                    detail=f"Hallucinated table name: {table}",
                ))
                ValidatorStats.record("UNKNOWN_TABLE", is_retry_trigger=True)

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

        # ── 7. Dynamic fb_{uuid} table references ─────────────────────────
        if self._DYNAMIC_TABLE_RE.search(sql):
            issues.append(ValidationIssue(
                "HARD_FAIL", "DYNAMIC_TABLE",
                "Dynamic fb_<uuid> table detected — use ai_answers instead",
            ))
            ValidatorStats.record("DYNAMIC_TABLE", is_retry_trigger=True)

        # ── 8. responsible / pending_with wrongly joined to users ─────────
        if re.search(r'\bjoin\s+users\b', sql_lower):
            if (re.search(r'\b(?:responsible|pending_with)\s*=\s*\w+\.id\b', sql_lower)
                    or re.search(r'\bON\s+\w+\.(?:responsible|pending_with)\s*=',
                                 sql, re.IGNORECASE)):
                issues.append(ValidationIssue(
                    "HARD_FAIL", "WRONG_JOIN",
                    "'responsible' / 'pending_with' are string ENUMs, not UUID FKs to users",
                    detail="Use: WHERE ica.responsible = 'CLIENT'  "
                           "Values: CLIENT, INTERNAL_OPERATIONS, SUB_CONTRACTOR",
                ))
                ValidatorStats.record("WRONG_JOIN", is_retry_trigger=True)

        # ── 9. Hardcoded year — SOFT_WARN only ───────────────────────────
        # Moved from HARD_FAIL: a user may intentionally query a specific
        # historical year (e.g. "inspector with most inspections in 2024").
        # Blocking that is validator overreach — it caused Qwen to corrupt
        # the query intent on retry (seen in traces: "projects without
        # inspections in 2024" → retried → produced WHERE CURRENT_YEAR != 2024,
        # which always evaluates TRUE, returning all projects).
        # Log for observability; do NOT block execution.
        for m in self._HARDCODED_YEAR_RE.finditer(sql_lower):
            year = m.group(1)
            issues.append(ValidationIssue(
                "SOFT_WARN", "HARDCODED_YEAR",
                f"Hardcoded year '= {year}' — consider EXTRACT(YEAR FROM CURRENT_DATE) for current-year queries",
                detail="Only an issue for 'this year' intent; intentional historical year queries are fine",
                retryable=False,
            ))
            ValidatorStats.record("HARDCODED_YEAR", is_retry_trigger=False)

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

        # ── 10. close_on arithmetic — SOFT_WARN (not a hard fail) ─────────
        # close_on is NULL for most records but arithmetic IS valid for the
        # subset where it is populated. Block only with a soft advisory.
        if self._CLOSE_ON_ARITH_RE.search(sql):
            issues.append(ValidationIssue(
                "SOFT_WARN", "UNRELIABLE_COLUMN",
                "Arithmetic on 'close_on' may be unreliable — column is NULL for most records",
                detail="Alternative: COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED'))",
                retryable=False,
            ))
            ValidatorStats.record("UNRELIABLE_COLUMN", is_retry_trigger=False)

        # ── 10b. TO_CHAR used as a WHERE filter — HARD_FAIL ───────────────
        # TO_CHAR(col, 'Month YYYY') = 'January 2026' silently returns 0 rows
        # because PostgreSQL pads month names: 'January   2026' ≠ 'January 2026'.
        # This pattern is ONLY valid for GROUP BY labels / SELECT display.
        if re.search(r'\bto_char\b', sql_lower):
            # Detect: to_char(...'Month YYYY'...) = '...' anywhere in the SQL
            # Use a flat scan (not [^)]+) to handle nested calls like date_trunc('month',...)
            tochar_month_eq = re.search(
                r"\bto_char\b[^;]*'month\s+yyyy'[^;]*=\s*'",
                sql_lower, re.IGNORECASE | re.DOTALL,
            )
            if tochar_month_eq:
                # Confirm a WHERE keyword precedes this pattern (not just SELECT/GROUP BY)
                before = sql_lower[:tochar_month_eq.start()]
                in_where = bool(re.search(r'\bwhere\b', before))
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
        has_agg = bool(self._AGG_RE.search(sql))
        if not has_agg and not self._LIMIT_RE.search(sql):
            issues.append(ValidationIssue(
                "SOFT_WARN", "MISSING_LIMIT",
                "Non-aggregate query has no LIMIT — may return large result sets",
                retryable=False,
            ))
            ValidatorStats.record("MISSING_LIMIT", is_retry_trigger=False)

        hard_fails = [i for i in issues if i.severity == "HARD_FAIL"]
        passed = len(hard_fails) == 0

        if hard_fails:
            logger.debug("Validation HARD_FAIL: %s",
                         "; ".join(str(f) for f in hard_fails))
        soft = [i for i in issues if i.severity == "SOFT_WARN"]
        if soft:
            logger.debug("Validation SOFT_WARN (not forwarded to LLM): %s",
                         "; ".join(str(w) for w in soft))

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
        if has_group_by and not _is_trend_groupby:
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

        # "overdue" corrective actions should use status='OVERDUE' not date arithmetic
        _OVERDUE_CA_Q = re.compile(
            r"\boverdue\s+(corrective|action)\b|\b(corrective|action).{0,15}overdue\b",
            re.IGNORECASE,
        )
        if _OVERDUE_CA_Q.search(question):
            uses_overdue_status = "'OVERDUE'" in sql.upper()
            uses_date_compare = bool(re.search(
                r"target_close_out_date\s*<\s*(?:NOW|CURRENT_DATE)", sql_lower))
            if not uses_overdue_status and uses_date_compare:
                warns.append(ValidationIssue(
                    "SOFT_WARN", "SEMANTIC_DATE_VS_STATUS",
                    "Overdue corrective actions: prefer status='OVERDUE' over date arithmetic",
                    detail="status='OVERDUE' is the authoritative field — do not recompute from target_close_out_date",
                    retryable=False,
                ))
                ValidatorStats.record("SEMANTIC_DATE_VS_STATUS", is_retry_trigger=False)

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

        # ICA→IR join structural fix.
        # deepseek generates: ON ir.inspection_id = ica.inspection_id
        # Correct form:       ON ica.inspection_id = ir.id
        # Rationale: ica.inspection_id is UUID FK; ir.inspection_id is VARCHAR.
        # Comparing them causes a PostgreSQL type mismatch error every time.
        sql = re.sub(
            r'\bir\.inspection_id\s*=\s*ica\.inspection_id\b',
            'ica.inspection_id = ir.id',
            sql, flags=re.IGNORECASE,
        )
        sql = re.sub(
            r'\bica\.inspection_id\s*=\s*ir\.inspection_id\b',
            'ica.inspection_id = ir.id',
            sql, flags=re.IGNORECASE,
        )
        # CAST variant: CAST(ir.inspection_id AS TEXT) = ica.inspection_id
        sql = re.sub(
            r'CAST\s*\(\s*ir\.inspection_id\s+AS\s+\w+\s*\)\s*=\s*ica\.inspection_id\b',
            'ica.inspection_id = ir.id',
            sql, flags=re.IGNORECASE,
        )
        # Cast operator variant: ir.inspection_id::varchar(255) = ica.inspection_id
        sql = re.sub(
            r'\bir\.inspection_id\s*::\s*\w+(?:\(\d+\))?\s*=\s*ica\.inspection_id\b',
            'ica.inspection_id = ir.id',
            sql, flags=re.IGNORECASE,
        )

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

        sql = self.fix_reserved_aliases(sql)
        sql = re.sub(r"\s*\[/?[A-Z_]+\]\s*$", "", sql, flags=re.IGNORECASE).rstrip()

        if sql and not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"

        return sql

    def fix_reserved_aliases(self, sql: str) -> str:
        """Fix SQL reserved words used as table aliases (e.g. 'is' for inspection_schedule)."""
        replacements = {"is": "isch"}
        for reserved, safe in replacements.items():
            alias_pattern = re.compile(
                r'(\b\w+\s+)' + r'\b' + re.escape(reserved) + r'\b'
                r'(?=\s*\.|\s+ON\b|\s+WHERE\b|\s+GROUP\b|\s+ORDER\b'
                r'|\s+LEFT\b|\s+JOIN\b|\s+INNER\b|\s*\n|\s*,)',
                re.IGNORECASE,
            )
            if alias_pattern.search(sql):
                sql = alias_pattern.sub(lambda m, s=safe: m.group(1) + s, sql)
                sql = re.sub(r'\b' + re.escape(reserved) + r'\.', safe + '.', sql)
        return sql

    def is_dynamic_answer_table(self, table_name: str) -> bool:
        return bool(re.match(
            r'^fb_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}$',
            table_name,
        ))

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
    })

    # Tables that NEVER have an active column — never inject for these.
    _NO_ACTIVE_TABLES: frozenset = frozenset({
        "ai_answers", "ai_questions", "role", "user_roles",
        "inspection_schedule",
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

        # ── 1. Drop hallucinated active columns on no-active tables ─────────
        # ALL tables in _NO_ACTIVE_TABLES have no soft-delete column.
        # Covers ai_answers, ai_questions, role, user_roles, inspection_schedule.
        # deepseek writes `aa.active`, `sched.active` etc from training bias.
        _no_active_aliases: set = set()
        for node in tree.find_all(exp.Table):
            if node.name and node.name.lower() in self._NO_ACTIVE_TABLES:
                _no_active_aliases.add((node.alias or node.name).lower())

        if _no_active_aliases:
            def _drop_ai_active(node):
                if (isinstance(node, exp.EQ)
                        and isinstance(node.left, exp.Column)
                        and node.left.name.lower() == "active"
                        and node.left.table
                        and node.left.table.lower() in _no_active_aliases):
                    mutations.append(
                        f"Removed {node.left.table}.active "
                        f"(table has no active column)"
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
            """alias → table_name for tables directly in this SELECT's FROM/JOIN."""
            result = {}
            def _add(t):
                if isinstance(t, exp.Table) and t.name:
                    result[(t.alias or t.name).lower()] = t.name.lower()
            # sqlglot v20+ uses 'from_' key; older versions use 'from'
            from_clause = sel.args.get("from_") or sel.args.get("from")
            if from_clause:
                _add(from_clause.this)
            for j in (sel.args.get("joins") or []):
                _add(j.this)
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
        # deepseek hallucinates a standalone 'impact' table every time it generates
        # CA detail queries. ast_repair removes both the JOIN and any SELECT columns
        # referencing the impact alias, so the BLOCK rule never fires.
        _impact_aliases: set = set()
        for node in tree.find_all(exp.Table):
            if node.name and node.name.lower() == "impact":
                _impact_aliases.add((node.alias or node.name).lower())

        if _impact_aliases:
            # Remove impact columns from SELECT expressions
            def _drop_impact_cols(node):
                if isinstance(node, exp.Select):
                    new_exprs = []
                    for expr in node.expressions:
                        col = expr.this if isinstance(expr, exp.Alias) else expr
                        if (isinstance(col, exp.Column)
                                and col.table
                                and col.table.lower() in _impact_aliases):
                            mutations.append(
                                f"Removed {col.table}.{col.name} "
                                f"(table 'impact' does not exist — use ai_answers WHERE aq.label='Impact')"
                            )
                            continue
                        new_exprs.append(expr)
                    if len(new_exprs) < len(node.expressions):
                        node = node.copy()
                        node.set("expressions", new_exprs)
                return node

            tree = tree.transform(_drop_impact_cols)

            # Remove the JOIN to impact from every Select node's joins list
            # (transform can't delete JOIN nodes — must mutate the joins list directly)
            for sel in tree.find_all(exp.Select):
                joins = sel.args.get("joins")
                if not joins:
                    continue
                new_joins = []
                for j in joins:
                    tbl = j.this
                    if (isinstance(tbl, exp.Table)
                            and tbl.name
                            and tbl.name.lower() == "impact"):
                        mutations.append("Removed JOIN impact (table does not exist)")
                    else:
                        new_joins.append(j)
                if len(new_joins) < len(joins):
                    sel.set("joins", new_joins)

        if not mutations:
            return sql, []

        try:
            repaired = tree.sql(dialect="postgres", pretty=False)
            if repaired and not repaired.rstrip().endswith(";"):
                repaired = repaired.rstrip() + ";"
            return repaired, mutations
        except Exception:
            return sql, []


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