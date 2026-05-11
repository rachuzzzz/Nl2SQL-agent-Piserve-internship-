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
        },
        "ai_answers": {
            "answer_value": "use answer_text (text) or answer_numeric (numeric)",
            "form_name":    "use module_name (ai_answers has no form_name column)",
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
    }

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
            for col, hint in col_hints.items():
                # Only match qualified references (alias.col or table.col)
                # to avoid flagging valid AS aliases like "SELECT ... AS form_name"
                if re.search(rf"\b\w+\.{re.escape(col)}\b", sql_lower):
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

        # ── 9. ICA bare column + UUID-vs-string BLOCK rules ──────────────────
        # inspection_corrective_action.risk_level_id and impact_id are UUID FKs.
        # Deepseek generates two wrong patterns:
        #   (a) ica.risk_level / ica.impact  — columns don't exist (only *_id variants)
        #   (b) WHERE ica.risk_level_id = 'High'  — UUID FK compared to a string
        # Both fail silently or crash at DB time and require a JOIN through the lookup table.
        ica_used = bool(re.search(
            r'\b(inspection_corrective_action|ica)\b', sql_lower))
        if ica_used:
            # (a) bare column without _id suffix
            if re.search(
                r'\b(ica|inspection_corrective_action)\.risk_level\b(?!_id)',
                sql_lower,
            ):
                issues.append(ValidationIssue(
                    "HARD_FAIL", "WRONG_COLUMN",
                    "ica.risk_level does not exist — JOIN risk_level rl ON ica.risk_level_id = rl.id",
                    detail="Filter: WHERE rl.name ILIKE '%High%'  "
                           "Values: High, Medium, Low, No Active Risk",
                ))
                ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)

            if re.search(
                r'\b(ica|inspection_corrective_action)\.impact\b(?!_id)',
                sql_lower,
            ):
                issues.append(ValidationIssue(
                    "HARD_FAIL", "WRONG_COLUMN",
                    "ica.impact does not exist — JOIN impact im ON ica.impact_id = im.id",
                    detail="Typo in live data: 'Non Confirmity' (not 'Conformity')",
                ))
                ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)

            # (b) UUID FK column compared to a string literal (not a UUID)
            if re.search(
                r"\brisk_level_id\s*(?:=|ilike|like|!=|<>)\s*"
                r"'(?![0-9a-f]{8}-)[^']*'",
                sql_lower,
            ):
                issues.append(ValidationIssue(
                    "HARD_FAIL", "WRONG_COLUMN",
                    "risk_level_id is a UUID FK — cannot compare to 'High'; join risk_level table instead",
                    detail="JOIN risk_level rl ON ica.risk_level_id = rl.id WHERE rl.name ILIKE '%High%'",
                ))
                ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)

            if re.search(
                r"\bimpact_id\s*(?:=|ilike|like|!=|<>)\s*"
                r"'(?![0-9a-f]{8}-)[^']*'",
                sql_lower,
            ):
                issues.append(ValidationIssue(
                    "HARD_FAIL", "WRONG_COLUMN",
                    "impact_id is a UUID FK — cannot compare to string; note: live data spells 'Non Confirmity'",
                    detail="JOIN impact im ON ica.impact_id = im.id WHERE im.name ILIKE '%Non Confirmity%'",
                ))
                ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)

        # (c) conformity spelling error when querying the impact table
        if "impact" in sql_lower and re.search(r'\bconformity\b', sql_lower):
            issues.append(ValidationIssue(
                "HARD_FAIL", "WRONG_COLUMN",
                "impact table spells 'Non Confirmity' (with 'i') — 'conformity' always returns zero rows",
                detail="Use: WHERE im.name ILIKE '%Non Confirmity%'",
            ))
            ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)

        # ── 10. Hardcoded year — SOFT_WARN only ───────────────────────────
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

        # ── 11. close_on arithmetic — SOFT_WARN (not a hard fail) ─────────
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

        # ── 12. Missing LIMIT — SOFT_WARN ─────────────────────────────────
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
    # Semantic heuristics — SOFT_WARN only, never blocks, never retries
    # ------------------------------------------------------------------ #

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