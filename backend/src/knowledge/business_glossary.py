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
appear as SOFT_WARN by default — observable but non-blocking, REGARDLESS
of an entry's own `severity` field.

Confirmed live 2026-08-07 (debug_llm_20260807_153128): "list all conducted
audits" generated status IN ('SUBMITTED','CLOSED','CLOSED_WITH_EXCEPTION')
— wrong per AUDIT_REPORT_SEMANTICS's own conducted=SUBMITTED-only
definition. The glossary caught it exactly right (conducted_audits fired),
but the SOFT_WARN-always behavior meant nothing blocked or corrected the
query — 18 rows shipped instead of the correct 13, silently.

Enforcement is now opt-in per entry id, via sql/validation/semantic.py's
_ENFORCED_GLOSSARY_IDS allowlist — the direct analogue of engine.py's
_CONSUMED_RULE_IDS gotcha for BLOCK/WARN rules, so a `"severity":
"HARD_FAIL"` tag here never silently starts blocking production on its
own. Only conducted_audits/completed_audits are currently allowlisted
(exact-literal checks against a 4-value enum from a single source of
truth). The 5 CAPEX/OPEX entries also tagged HARD_FAIL below (search
capex_conversion_rate, capex_converted, opex_incurred, fm_manager_capex,
capex_row_multiplication) are DELIBERATELY left off that allowlist —
they're heuristic phrase-matchers (e.g. bare "rate"/"percent"), not yet
proven stable enough to block on. Their severity tag is real, just dormant
until they're added to the allowlist after their own eval.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

# ── Semantics registry (single source of truth for status definitions) ───────
try:
    from src.knowledge.semantics_registry import (
        completed_required_regex as _completed_required_regex,
        conducted_required_regex as _conducted_required_regex,
        old_completed_forbidden_regex as _old_completed_forbidden_regex,
        completed_filter as _completed_filter,
        conducted_in_list as _conducted_in_list,
        AUDIT_SEMANTICS as _AUDIT_SEMANTICS,
        AUDIT_REPORT_SEMANTICS as _AUDIT_REPORT_SEMANTICS,
    )
    _COMPLETED_REQUIRED  = _completed_required_regex()
    _CONDUCTED_REQUIRED  = _conducted_required_regex()
    _OLD_COMPLETED_BAD   = _old_completed_forbidden_regex()
    _COMPLETED_SQL       = f"status {_completed_filter()}"
    _CONDUCTED_SQL       = f"status IN {_conducted_in_list()}"
    # Audit domain — TWO separate concepts, deliberately not conflated:
    #   "completed/conducted audit PLANS" (word "plan" said explicitly) ->
    #     audit_plan.status (conducted = SUBMITTED/RETURNED_FOR_MODIFICATION,
    #     completed = APPROVED) — was this planning document approved.
    #   bare "completed/conducted audits" (no qualifier) -> audit_report.status
    #     (conducted = SUBMITTED, completed = CLOSED/CLOSED_WITH_EXCEPTION) —
    #     was the audit itself actually carried out and closed out. Confirmed
    #     live: this bare phrasing is what people actually mean almost every
    #     time, not the plan-approval question — audit_plan is "just the
    #     plan," not the audit.
    # No "old wrong definition" forbidden pattern exists yet for either audit
    # case (nothing confirmed live to need one), unlike inspection's
    # _OLD_COMPLETED_BAD.
    _AUDIT_PLAN_COMPLETED_REQUIRED = _completed_required_regex(_AUDIT_SEMANTICS)
    _AUDIT_PLAN_CONDUCTED_REQUIRED = _conducted_required_regex(_AUDIT_SEMANTICS)
    _AUDIT_PLAN_COMPLETED_SQL      = f"status {_completed_filter(_AUDIT_SEMANTICS)}"
    _AUDIT_PLAN_CONDUCTED_SQL      = f"status IN {_conducted_in_list(_AUDIT_SEMANTICS)}"
    _AUDIT_REPORT_COMPLETED_REQUIRED = _completed_required_regex(_AUDIT_REPORT_SEMANTICS)
    _AUDIT_REPORT_CONDUCTED_REQUIRED = _conducted_required_regex(_AUDIT_REPORT_SEMANTICS)
    _AUDIT_REPORT_COMPLETED_SQL      = f"status {_completed_filter(_AUDIT_REPORT_SEMANTICS)}"
    _AUDIT_REPORT_CONDUCTED_SQL      = f"status IN {_conducted_in_list(_AUDIT_REPORT_SEMANTICS)}"
except ImportError:
    # Fallback — keep in sync with knowledge/semantics_registry.py
    _COMPLETED_REQUIRED = r"status\s*=\s*'CLOSED'"
    _CONDUCTED_REQUIRED = (
        r"status\s*IN\s*\([^)]*(?:'SUBMITTED'|'UNDER_REVIEW'|'RETURN_FOR_MODIFICATION')[^)]*\)"
    )
    _OLD_COMPLETED_BAD  = r"status\s*IN\s*\(\s*'CLOSED'\s*,\s*'SUBMITTED'\s*\)"
    _COMPLETED_SQL      = "status = 'CLOSED'"
    _CONDUCTED_SQL      = "status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')"
    _AUDIT_PLAN_COMPLETED_REQUIRED = r"status\s*=\s*'APPROVED'"
    _AUDIT_PLAN_CONDUCTED_REQUIRED = (
        r"status\s*IN\s*\([^)]*(?:'SUBMITTED'|'RETURNED_FOR_MODIFICATION')[^)]*\)"
    )
    _AUDIT_PLAN_COMPLETED_SQL = "status = 'APPROVED'"
    _AUDIT_PLAN_CONDUCTED_SQL = "status IN ('SUBMITTED','RETURNED_FOR_MODIFICATION')"
    _AUDIT_REPORT_COMPLETED_REQUIRED = (
        r"status\s*IN\s*\([^)]*(?:'CLOSED'|'CLOSED_WITH_EXCEPTION')[^)]*\)"
    )
    _AUDIT_REPORT_CONDUCTED_REQUIRED = r"status\s*=\s*'SUBMITTED'"
    _AUDIT_REPORT_COMPLETED_SQL = "status IN ('CLOSED','CLOSED_WITH_EXCEPTION')"
    _AUDIT_REPORT_CONDUCTED_SQL = "status = 'SUBMITTED'"


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
    # Human-readable name for LLM-facing messages when this entry is
    # actually enforced (HARD_FAIL) — defaults to terms[0] in _build() below
    # if not set explicitly, but terms[0] is often a raw regex (e.g.
    # r"conducted\s+audits?\b(?!\s+plans?\b)"), not fit to show the model.
    # Set explicitly on any entry that gets allowlisted for enforcement in
    # sql/validation/semantic.py's _ENFORCED_GLOSSARY_IDS.
    label: str = ""
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
        "required_pattern": _COMPLETED_REQUIRED,
        "forbidden_patterns": [
            # OLD WRONG definition — completed != CLOSED+SUBMITTED.
            # SUBMITTED is 'conducted' (in-flight), not 'completed'.
            _OLD_COMPLETED_BAD,
        ],
        "correct_sql": _COMPLETED_SQL,
        "message": (
            f"completed inspections = {_COMPLETED_SQL} only "
            f"(fully reviewed and approved). "
            f"SUBMITTED/UNDER_REVIEW/RETURN_FOR_MODIFICATION are 'conducted' (in-flight), not completed."
        ),
        "severity": "SOFT_WARN",
    },
    {
        "id": "conducted_inspections",
        "terms": ["conducted inspection", "conducted inspections", "in-flight inspection",
                   "in progress inspection", "inspections in progress"],
        "required_pattern": _CONDUCTED_REQUIRED,
        "forbidden_patterns": [
            # status != 'DRAFT' alone is too broad — it includes CLOSED (completed)
            r"status\s*!=\s*'DRAFT'\s*(?:AND|ORDER|GROUP|LIMIT|;|$)",
        ],
        "correct_sql": _CONDUCTED_SQL,
        "message": (
            f"conducted inspections = {_CONDUCTED_SQL} — in-flight, "
            f"excludes both DRAFT (not started) and CLOSED (completed)."
        ),
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
        "terms": ["open action", "open actions", "open mitigative", "open ma"],
        "required_pattern": r"status\s*=\s*'OPEN'",
        "forbidden_patterns": [],
        "correct_sql": "WHERE status = 'OPEN'",
        "message": "open mitigative actions: status = 'OPEN'",
        "severity": "SOFT_WARN",
    },
    {
        "id": "overdue_actions",
        "terms": ["overdue action", "overdue actions", "overdue mitigative"],
        "required_pattern": r"status\s*=\s*'OVERDUE'",
        "forbidden_patterns": [],
        "correct_sql": "WHERE status = 'OVERDUE'",
        "message": "overdue mitigative actions: status = 'OVERDUE'",
        "severity": "SOFT_WARN",
    },
    {
        "id": "closed_actions",
        "terms": ["closed action", "closed actions", "closed mitigative"],
        "required_pattern": (
            r"status\s*(=\s*'CLOSED'|IN\s*\([^)]*'CLOSED'[^)]*\))"
        ),
        "forbidden_patterns": [],
        "correct_sql": "status IN ('CLOSED', 'CLOSE_WITH_DEFERRED')",
        "message": "closed mitigative actions include CLOSE_WITH_DEFERRED",
        "severity": "SOFT_WARN",
    },
    {
        "id": "deferred_actions",
        "terms": ["deferred action", "deferred actions", "deferred mitigative", "carried forward"],
        "required_pattern": r"CLOSE_WITH_DEFERRED",
        "forbidden_patterns": [],
        "correct_sql": "WHERE status = 'CLOSE_WITH_DEFERRED'",
        "message": "deferred mitigative actions: status = 'CLOSE_WITH_DEFERRED'",
        "severity": "SOFT_WARN",
    },

    # ── Audit status terms — TWO SEPARATE tables/concepts, deliberately not
    # conflated. Confirmed live: bare "completed/conducted audits" (no
    # qualifier) is what people actually mean almost every time — whether the
    # audit itself was carried out and closed out (audit_report.status) — NOT
    # whether its planning document was approved (audit_plan.status, "just
    # the plan"). The plan-specific terms below only match when the question
    # explicitly says "audit plan(s)". ─────────────────────────────────────

    {
        "id": "completed_audits",
        # Regex terms (not plain phrases) with a negative lookahead excluding
        # "plan(s)" — "completed audit" as a plain word-boundary phrase would
        # also match inside "completed audit PLANS" (word boundaries don't
        # anchor on what follows), which would wrongly fire this
        # report-scoped entry on a plan-scoped question too. Confirmed live.
        "terms": [r"completed\s+audits?\b(?!\s+plans?\b)"],
        "label": "completed audits",
        "required_pattern": _AUDIT_REPORT_COMPLETED_REQUIRED,
        "forbidden_patterns": [],
        "correct_sql": _AUDIT_REPORT_COMPLETED_SQL,
        "message": (
            f"completed audits = {_AUDIT_REPORT_COMPLETED_SQL} on audit_report "
            f"(fully closed out, with or without exceptions). SUBMITTED is "
            f"'conducted' (audit done, report submitted, not yet closed) — not "
            f"completed. This is about audit_report, NOT audit_plan.status."
        ),
        # Enforced (HARD_FAIL) — see sql/validation/semantic.py's
        # _ENFORCED_GLOSSARY_IDS. Confirmed live 2026-08-07
        # (debug_llm_20260807_153128): the mirror-image conducted_audits
        # entry below was caught firing on real wrong SQL, silently, while
        # SOFT_WARN-only. Escalated alongside it — same table, same
        # registry source, same question class.
        "severity": "HARD_FAIL",
    },
    {
        "id": "conducted_audits",
        # Same negative-lookahead reasoning as completed_audits above for the
        # "conducted audit(s)" phrase; the other three terms don't collide
        # with any "audit plan(s)" phrasing so stay plain.
        "terms": [r"conducted\s+audits?\b(?!\s+plans?\b)", "in-flight audit",
                   "in-flight audits", "audits in progress"],
        "label": "conducted audits",
        "required_pattern": _AUDIT_REPORT_CONDUCTED_REQUIRED,
        "forbidden_patterns": [],
        "correct_sql": _AUDIT_REPORT_CONDUCTED_SQL,
        "message": (
            f"conducted audits = {_AUDIT_REPORT_CONDUCTED_SQL} on audit_report "
            f"— the audit was carried out and the report submitted, but not yet "
            f"closed out. Excludes DRAFT (not yet submitted) and CLOSED/"
            f"CLOSED_WITH_EXCEPTION (completed). This is about audit_report, "
            f"NOT audit_plan.status."
        ),
        # Enforced (HARD_FAIL) — see sql/validation/semantic.py's
        # _ENFORCED_GLOSSARY_IDS. Confirmed live 2026-08-07
        # (debug_llm_20260807_153128, qid=88b9c71e): "list all conducted
        # audits" generated status IN ('SUBMITTED','CLOSED',
        # 'CLOSED_WITH_EXCEPTION') — 18 rows instead of the correct 13 — and
        # this entry fired correctly but the SOFT_WARN-always behavior meant
        # nothing blocked it. Requires semantics_registry.py's
        # conducted_required_regex() len==1 fix (accepts both '=' and
        # 'IN (single)') to avoid an unfixable retry loop — see that
        # function's docstring.
        "severity": "HARD_FAIL",
    },
    {
        "id": "completed_audit_plans",
        "terms": ["completed audit plan", "completed audit plans",
                   "approved audit plan", "approved audit plans"],
        "required_pattern": _AUDIT_PLAN_COMPLETED_REQUIRED,
        "forbidden_patterns": [],
        "correct_sql": _AUDIT_PLAN_COMPLETED_SQL,
        "message": (
            f"completed audit plans = {_AUDIT_PLAN_COMPLETED_SQL} on audit_plan "
            f"(fully reviewed and approved). SUBMITTED/RETURNED_FOR_MODIFICATION "
            f"are 'conducted' (in-flight), not completed."
        ),
        "severity": "SOFT_WARN",
    },
    {
        "id": "conducted_audit_plans",
        "terms": ["conducted audit plan", "conducted audit plans",
                   "audit plans in progress", "in-flight audit plan",
                   "in-flight audit plans"],
        "required_pattern": _AUDIT_PLAN_CONDUCTED_REQUIRED,
        "forbidden_patterns": [],
        "correct_sql": _AUDIT_PLAN_CONDUCTED_SQL,
        "message": (
            f"conducted audit plans = {_AUDIT_PLAN_CONDUCTED_SQL} on audit_plan "
            f"— in-flight, excludes both DRAFT (not started) and APPROVED "
            f"(completed)."
        ),
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
        "correct_sql": (
            "SELECT AVG(obs_count) FROM (SELECT ir.inspection_id, COUNT(*) AS obs_count "
            "FROM ... GROUP BY ir.inspection_id) sub  -- generic shape; "
            "if obs_count = OBSERVATIONS specifically, use "
            "COUNT(DISTINCT aa.submission_id) not COUNT(*) (see registry counting.observations_per_inspection — "
            "ai_answers has multiple rows per observation)"
        ),
        "message": (
            "average per inspection requires nested aggregation: AVG(COUNT per group) — not just GROUP BY inspection_id. "
            "If counting OBSERVATIONS per inspection, the inner COUNT must be COUNT(DISTINCT aa.submission_id), not COUNT(*)."
        ),
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

    # ── CAPEX / Financial terms ──────────────────────────────────────────────

    {
        "id": "capex_conversion_rate",
        "terms": ["conversion rate", "capex conversion", "how much was converted"],
        "required_pattern": r"po_details|po_by_ica|po\.po_amount",
        "forbidden_patterns": [
            # Wrong: using inspection count / MA count as conversion rate
            r"inspections_with_ca|inspections_with_ma|inspection.*ca.*rate|ca.*inspection.*rate",
        ],
        "correct_sql": "ROUND(SUM(po.po_amount) / NULLIF(SUM(ica.capex), 0) * 100, 2) AS conversion_rate_pct",
        "message": "CAPEX conversion rate = SUM(po_amount) / SUM(ica.capex) * 100 — NOT inspection-to-MA percentage",
        "severity": "HARD_FAIL",
    },

    {
        "id": "capex_converted",
        "terms": ["capex converted", "converted capex", "capex conversion"],
        "required_pattern": r"po_details|po\.po_amount|po_by_ica",
        "forbidden_patterns": [
            r"capex_status\s*=\s*'ACCEPTED'",
            r"SUM\s*\(\s*ica\.capex\s*\).*LEFT\s+JOIN\s+po_details",
        ],
        "correct_sql": "(SELECT SUM(po_amount) FROM po_details) AS capex_converted",
        "message": "CAPEX Converted = SUM(po_details.po_amount) — NOT ica.capex filtered by capex_status='ACCEPTED'",
        "severity": "HARD_FAIL",
    },
    {
        "id": "capex_lost",
        "terms": ["capex lost", "lost capex", "capex loss"],
        # Two valid shapes: (1) the arithmetic AGGREGATE (SUM(ica.capex) - SUM(po.po_amount))
        # via po_details, OR (2) a LISTING query filtered by capex_status='REJECTED'
        # (e.g. "capex lost from rejected mitigative actions" — see business_rules.py
        # block_capex_status_rejected_as_loss, narrowed 2026-06-12 to allow exactly this).
        # Previously required_pattern only accepted (1), contradicting this entry's
        # own message which already said (2) is valid — aligned now.
        "required_pattern": r"po_details|po\.po_amount|po_by_ica|capex_status\s*=\s*'REJECTED'",
        "forbidden_patterns": [],
        "correct_sql": "SUM(ica.capex) WHERE ica.active=true AND ica.capex>0 MINUS SUM(po_details.po_amount) WHERE po_details.active=true — arithmetic gap",
        "message": "CAPEX Lost = SUM(ica.capex) - SUM(po.po_amount) — arithmetic gap. capex_status=REJECTED is a valid filter for rejected MA queries.",
        "severity": "SOFT_WARN",
    },
    {
        "id": "capex_generated",
        "terms": ["capex generated", "generated capex", "total capex generated"],
        "required_pattern": r"SUM\s*\(\s*(?:ica\.)?capex\s*\)|inspection_corrective_action",
        "forbidden_patterns": [],
        "correct_sql": "SUM(ica.capex) AS capex_generated FROM inspection_corrective_action",
        "message": "CAPEX Generated = SUM(ica.capex) from inspection_corrective_action",
        "severity": "SOFT_WARN",
    },
    {
        "id": "opex_generated",
        "terms": ["opex generated", "generated opex", "total opex generated"],
        "required_pattern": r"SUM\s*\(\s*(?:ica\.)?opex\s*\)|inspection_corrective_action",
        "forbidden_patterns": [],
        "correct_sql": "SUM(ica.opex) AS opex_generated FROM inspection_corrective_action",
        "message": "OPEX Generated = SUM(ica.opex) from inspection_corrective_action — "
                   "the inspectee's estimate, parallel to CAPEX Generated. Distinct from "
                   "OPEX Incurred/Converted (SUM(pr_details.pr_amount), the actual realized spend).",
        "severity": "SOFT_WARN",
    },
    {
        "id": "opex_incurred",
        # Plain phrases PLUS proximity patterns — bare "opex converted" as a
        # contiguous literal misses realistic phrasing like "opex generated
        # AND converted" (the word "converted" isn't adjacent to "opex").
        # Same class of miss that let 'overdue' slip through in seed_examples.py.
        "terms": [
            "opex incurred", "opex spend", "purchase requisition amount", "pr amount",
            r"opex\b.{0,25}\b(?:converted|incurred|spend|spent)",
            r"(?:converted|incurred|spend|spent)\b.{0,25}\bopex",
        ],
        "required_pattern": r"pr_details|pr\.pr_amount|pr_by_ica",
        "forbidden_patterns": [
            r"SUM\s*\(\s*ica\.opex\s*\)",
        ],
        "correct_sql": "(SELECT SUM(pr_amount) FROM pr_details) AS opex_incurred",
        "message": "OPEX Incurred = SUM(pr_details.pr_amount) — NOT SUM(ica.opex) which is just an estimate",
        "severity": "HARD_FAIL",
    },
    {
        "id": "fm_manager_capex",
        "terms": [
            r"capex generator",
            r"capex converter",
            r"fm manager.*capex",
            r"capex.*fm manager",
            r"top.*capex.*generator",
            r"top.*capex.*converter",
        ],
        "required_pattern": r"inspectee_user_id",
        "forbidden_patterns": [
            r"GROUP\s+BY.*corrective_action_id",
        ],
        "correct_sql": "JOIN users u ON ir.inspectee_user_id = u.id GROUP BY u.id, u.first_name, u.last_name",
        "message": "CAPEX Generator/Converter = FM Manager (inspectee_user_id) — GROUP BY person not corrective_action_id",
        "severity": "HARD_FAIL",
    },
    {
        "id": "capex_row_multiplication",
        "terms": ["capex generated", "total capex", "capex by facility", "capex by inspector"],
        "required_pattern": None,
        "forbidden_patterns": [
            r"SUM\s*\(\s*ica\.capex\s*\).*LEFT\s+JOIN\s+po_details",
        ],
        "correct_sql": "WITH po_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) FROM po_details GROUP BY 1) LEFT JOIN po_by_ica p ON p.ica_id = ica.id",
        "message": "SUM(ica.capex) + LEFT JOIN po_details causes row multiplication — use CTE pre-aggregation",
        "severity": "HARD_FAIL",
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
            label=raw.get("label") or raw["terms"][0],
        ).compile()
        entries.append(entry)
    return entries


GLOSSARY: list[GlossaryTerm] = _build()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def check_glossary(question: str, sql: str) -> list[tuple[GlossaryTerm, str]]:
    """
    Check SQL against all glossary terms that match the question.

    Returns list of (entry, message) tuples for any violations — the full
    entry (not just its severity) so the one caller (sql/validation/
    semantic.py) can check entry.id against its enforcement allowlist and
    use entry.label/correct_sql for a clean retry message. Empty list = no
    violations.
    """
    violations = []
    for entry in GLOSSARY:
        if not entry.matches_question(question):
            continue
        msg = entry.violation(sql)
        if msg:
            violations.append((entry, msg))
    return violations