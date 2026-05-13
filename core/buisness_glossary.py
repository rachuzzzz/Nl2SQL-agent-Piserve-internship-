"""
business_glossary.py — Domain term → SQL constraint definitions.

Purpose
-------
Fixes the silent metric corruption problem: when a user says "completed
inspections", the system must use the correct status set, not the nearest
pattern match from training data.

Each entry defines:
  - terms: user-facing words that trigger this definition
  - required_pattern: regex that MUST appear in SQL when term is used
  - forbidden_patterns: regexes that MUST NOT appear in SQL when term is used
  - correct_sql: the canonical constraint (for error messages)
  - severity: HARD_FAIL (blocks) or SOFT_WARN (logs only)

The GlossaryValidator is called from validate_semantic() so violations
appear as SOFT_WARN by default — observable but non-blocking.
Flip to HARD_FAIL for production enforcement once eval confirms stability.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Glossary entry
# ---------------------------------------------------------------------------

@dataclass
class GlossaryTerm:
    id: str
    terms: list[str]                    # question keywords triggering this entry
    required_pattern: Optional[str]     # SQL must match this regex
    forbidden_patterns: list[str]       # SQL must NOT match any of these
    correct_sql: str                    # shown in violation message
    message: str                        # violation message
    severity: str = "SOFT_WARN"        # SOFT_WARN or HARD_FAIL
    # Pre-compiled — populated at load time
    _term_re: Optional[re.Pattern] = field(default=None, repr=False, compare=False)
    _required_re: Optional[re.Pattern] = field(default=None, repr=False, compare=False)
    _forbidden_res: list[re.Pattern] = field(default_factory=list, repr=False, compare=False)

    def compile(self) -> "GlossaryTerm":
        # Terms may be plain strings OR regex patterns (contain .* etc).
        # Plain strings get word-boundary anchors + re.escape.
        # Terms containing regex metacharacters (. * + ? [ ] ( ) |) are used as-is.
        _META = re.compile(r'[.*+?()\[\]|{}\\]')
        parts = []
        for t in self.terms:
            if _META.search(t):
                parts.append(t)          # already a regex pattern
            else:
                parts.append(r"\b" + re.escape(t) + r"\b")
        self._term_re = re.compile("(?:" + "|".join(parts) + ")", re.IGNORECASE)
        if self.required_pattern:
            self._required_re = re.compile(self.required_pattern, re.IGNORECASE)
        self._forbidden_res = [
            re.compile(p, re.IGNORECASE) for p in self.forbidden_patterns
        ]
        return self

    def matches_question(self, question: str) -> bool:
        return bool(self._term_re and self._term_re.search(question))

    def violation(self, sql: str) -> Optional[str]:
        """Return violation message if SQL violates this term's constraints, else None."""
        if self._required_re and not self._required_re.search(sql):
            return (
                f"[TERM_VIOLATION] '{self.terms[0]}' requires: {self.message}. "
                f"Use: {self.correct_sql}"
            )
        for pat in self._forbidden_res:
            if pat.search(sql):
                return (
                    f"[TERM_VIOLATION] '{self.terms[0]}' forbids this pattern. "
                    f"Use: {self.correct_sql}"
                )
        return None


# ---------------------------------------------------------------------------
# Glossary registry
# ---------------------------------------------------------------------------

_RAW: list[dict] = [

    # ── Inspection status terms ──────────────────────────────────────────────

    {
        "id": "completed_inspections",
        "terms": ["completed inspection", "completed inspections", "completion"],
        "required_pattern": r"status\s*(?:IN|!=|=|NOT\s+IN)",
        "forbidden_patterns": [
            # The single biggest silent corruption: "completed" → status='CLOSED' only
            r"status\s*=\s*'CLOSED'\s*(?:AND|ORDER|GROUP|LIMIT|;|$)",
        ],
        "correct_sql": "status IN ('CLOSED', 'SUBMITTED', 'UNDER_REVIEW', 'RETURN_FOR_MODIFICATION')",
        "message": "completed inspections include SUBMITTED, CLOSED, UNDER_REVIEW, RETURN_FOR_MODIFICATION — not only CLOSED",
        "severity": "SOFT_WARN",
    },
    {
        "id": "active_inspections",
        "terms": ["active inspection", "active inspections", "non-draft", "non draft"],
        "required_pattern": r"status\s*!=\s*'DRAFT'",
        "forbidden_patterns": [],
        "correct_sql": "status != 'DRAFT'",
        "message": "active inspections filter: status != 'DRAFT'",
        "severity": "SOFT_WARN",
    },
    {
        "id": "open_actions",
        "terms": ["open action", "open actions", "open corrective", "open ca"],
        "required_pattern": r"status\s*=\s*'OPEN'",
        "forbidden_patterns": [],
        "correct_sql": "WHERE status = 'OPEN'",
        "message": "open corrective actions: status = 'OPEN'",
        "severity": "SOFT_WARN",
    },
    {
        "id": "overdue_actions",
        "terms": ["overdue action", "overdue actions", "overdue corrective"],
        "required_pattern": r"status\s*=\s*'OVERDUE'",
        "forbidden_patterns": [],
        "correct_sql": "WHERE status = 'OVERDUE'",
        "message": "overdue corrective actions: status = 'OVERDUE'",
        "severity": "SOFT_WARN",
    },
    {
        "id": "closed_actions",
        "terms": ["closed action", "closed actions", "closed corrective"],
        "required_pattern": (
            r"status\s*(=\s*'CLOSED'|IN\s*\([^)]*'CLOSED'[^)]*\))"
        ),
        "forbidden_patterns": [],
        "correct_sql": "status IN ('CLOSED', 'CLOSE_WITH_DEFERRED')",
        "message": "closed corrective actions include CLOSE_WITH_DEFERRED",
        "severity": "SOFT_WARN",
    },
    {
        "id": "deferred_actions",
        "terms": ["deferred action", "deferred actions", "deferred corrective", "carried forward"],
        "required_pattern": r"CLOSE_WITH_DEFERRED",
        "forbidden_patterns": [],
        "correct_sql": "WHERE status = 'CLOSE_WITH_DEFERRED'",
        "message": "deferred corrective actions: status = 'CLOSE_WITH_DEFERRED'",
        "severity": "SOFT_WARN",
    },

    # ── Time terms ───────────────────────────────────────────────────────────

    {
        "id": "this_year",
        "terms": ["this year", "current year", "year to date", "ytd"],
        # Required: if submitted_on is filtered, EXTRACT must be used
        # Only fires if the SQL has a date filter that looks year-based
        "required_pattern": None,
        "forbidden_patterns": [
            # Hardcoded year in EXTRACT comparison — wrong for 'this year' queries
            r"EXTRACT\s*\(\s*YEAR\s+FROM\s+\w[\w.]*\s*\)\s*=\s*20\d\d\b",
        ],
        "correct_sql": "EXTRACT(YEAR FROM submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)",
        "message": "current year queries must use EXTRACT(YEAR FROM CURRENT_DATE), not a hardcoded year",
        "severity": "SOFT_WARN",
    },
    {
        "id": "this_quarter",
        "terms": ["this quarter", "current quarter"],
        "required_pattern": r"date_trunc\s*\(\s*'quarter'",
        "forbidden_patterns": [],
        "correct_sql": "created_on >= date_trunc('quarter', CURRENT_DATE)",
        "message": "current quarter: use date_trunc('quarter', CURRENT_DATE)",
        "severity": "SOFT_WARN",
    },
    {
        "id": "this_month",
        "terms": ["this month", "current month"],
        "required_pattern": r"date_trunc\s*\(\s*'month'",
        "forbidden_patterns": [
            r"TO_CHAR\s*\(.*?\)\s*=\s*'[A-Za-z]+\s+\d{4}'",
        ],
        "correct_sql": "date_trunc('month', submitted_on) = date_trunc('month', NOW())",
        "message": "current month: use date_trunc('month', ...) — never TO_CHAR equality",
        "severity": "SOFT_WARN",
    },

    # ── Responsible party terms ──────────────────────────────────────────────

    {
        "id": "client_responsible",
        "terms": ["client responsible", "responsible.*client", "client.*responsibility"],
        "required_pattern": r"responsible\s*=\s*'CLIENT'",
        "forbidden_patterns": [],
        "correct_sql": "WHERE ica.responsible = 'CLIENT'",
        "message": "client-responsible actions: responsible = 'CLIENT' (uppercase ENUM string)",
        "severity": "SOFT_WARN",
    },
    {
        "id": "internal_responsible",
        "terms": ["internal operations", "internal responsible"],
        "required_pattern": r"responsible\s*=\s*'INTERNAL_OPERATIONS'",
        "forbidden_patterns": [],
        "correct_sql": "WHERE ica.responsible = 'INTERNAL_OPERATIONS'",
        "message": "internal operations: responsible = 'INTERNAL_OPERATIONS'",
        "severity": "SOFT_WARN",
    },

    # ── Score terms ──────────────────────────────────────────────────────────

    {
        "id": "inspection_score",
        "terms": ["inspection score", "average score", "avg score", "score distribution"],
        "required_pattern": r"inspection_score",
        "forbidden_patterns": [],
        "correct_sql": "AVG(ir.inspection_score) FROM inspection_report WHERE status != 'DRAFT'",
        "message": "inspection scores are in inspection_report.inspection_score (not ai_answers)",
        "severity": "SOFT_WARN",
    },

    # ── Aggregation operator terms ───────────────────────────────────────────

    {
        "id": "average_per_inspection",
        "terms": [
            r"average.*per inspection",
            r"avg.*per inspection",
            r"average number.*per",
            r"mean.*per inspection",
            r"average.*each inspection",
        ],
        "required_pattern": r"AVG\s*\(",
        "forbidden_patterns": [],
        "correct_sql": "SELECT AVG(obs_count) FROM (SELECT ir.inspection_id, COUNT(*) AS obs_count FROM ... GROUP BY ir.inspection_id) sub",
        "message": "average per inspection requires nested aggregation: AVG(COUNT per group) — not just GROUP BY inspection_id",
        "severity": "SOFT_WARN",
    },
    {
        "id": "percentage_ratio",
        "terms": ["percentage", "percent", "ratio", "proportion", "rate"],
        "required_pattern": r"100\s*[\*\.]\s*|/\s*NULLIF\s*\(|/\s*COUNT",
        "forbidden_patterns": [],
        "correct_sql": "ROUND(100.0 * COUNT(matching) / NULLIF(COUNT(*), 0), 1) AS percentage",
        "message": "percentage queries require division — use 100.0 * count / NULLIF(total, 0)",
        "severity": "SOFT_WARN",
    },
]


def _build() -> list[GlossaryTerm]:
    entries = []
    for raw in _RAW:
        entry = GlossaryTerm(
            id=raw["id"],
            terms=raw["terms"],
            required_pattern=raw.get("required_pattern"),
            forbidden_patterns=raw.get("forbidden_patterns", []),
            correct_sql=raw["correct_sql"],
            message=raw["message"],
            severity=raw.get("severity", "SOFT_WARN"),
        ).compile()
        entries.append(entry)
    return entries


GLOSSARY: list[GlossaryTerm] = _build()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def check_glossary(question: str, sql: str) -> list[tuple[str, str]]:
    """
    Check SQL against all glossary terms that match the question.

    Returns list of (severity, message) tuples for any violations.
    Empty list = no violations.
    """
    violations = []
    for entry in GLOSSARY:
        if not entry.matches_question(question):
            continue
        msg = entry.violation(sql)
        if msg:
            violations.append((entry.severity, msg))
    return violations


def get_term(term_id: str) -> Optional[GlossaryTerm]:
    for entry in GLOSSARY:
        if entry.id == term_id:
            return entry
    return None


def summary() -> str:
    return f"BusinessGlossary: {len(GLOSSARY)} terms"