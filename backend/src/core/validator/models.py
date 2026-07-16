"""
Structured validation output: ValidationIssue, ValidationResult, and the
process-lifetime ValidatorStats counters. Extracted verbatim from the
original core/validator.py.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional




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
