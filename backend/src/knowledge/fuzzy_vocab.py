"""
fuzzy_vocab.py — typo/abbreviation tolerance for the regex-gated
CONDITIONAL_BLOCKS/INJECT-rule matching (agent/prompts/builder.py,
agent/prompts/builder_audit.py, sql/rules/engine.py). Those layers do
literal (case-insensitive only) regex matching against the user's raw
question — a misspelling or truncation ("mit actions", "crrctive action",
"coorecctive", "mittigattive") silently loses the domain-vocabulary
guidance those regexes exist to inject (MA-workflow routing, "at risk" vs
risk_level, dead-column warnings, etc.), with no fallback. Confirmed live
2026-08-11: no spell-correction/fuzzy-matching mechanism existed anywhere
in this codebase before this module — api/intent_recap.py's LLM
paraphrase looks related but is a parallel, display-only call, never
routed back into this matching pipeline (see that module's docstring).

fuzzy_augment() APPENDS matched canonical phrases to the question text
rather than replacing anything, so every existing exact-match regex now
finds a literal hit without changing what the LLMs actually read as the
user's original wording. Called once per turn from
agent/orchestrator/loop.py's query() — see that call site for which two
downstream matchers this actually feeds.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

# Starting set — the highest-value domain phrases already implicit in
# question_patterns.py's MA_WORKFLOW_TOPIC/MA_APPROVAL_WORKFLOW_TOPIC,
# builder.py/builder_audit.py's CONDITIONAL_BLOCKS topic patterns, and the
# terms called out in this codebase's own "confirmed live" trap comments
# (semantics_registry.py, inject_rules.py, audit_rules.py). Not
# exhaustive by design — extend as new gaps are found rather than trying
# to enumerate every possible term up front.
CANONICAL_TERMS: list[str] = [
    "mitigative action",
    "corrective action",
    "at risk",
    "overdue",
    "risk level",
    "capex",
    "opex",
    "progress stage",
    "implementation status",
    "schedule",
    "portfolio",
    "cycle",
    "audit plan",
    "audit schedule",
    "audit report",
    "observation",
    "adequacy",
    "compliance",
    "impact",
    "responsible party",
    "deferred",
]

# Connector/function words inside a canonical phrase carry no
# discriminative signal and are too short to fuzzy/prefix-match safely
# (e.g. query word "attend" would falsely prefix-match canonical word
# "at") — dropped from the per-word match requirement rather than
# tightened further, since a phrase like "at risk" should still be
# satisfiable by the query containing a match for "risk" alone.
_STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "for", "to", "and", "or"}

_MIN_PREFIX = 3          # shortest abbreviation worth trusting, e.g. "mit"
# 82, not a rounder 75/80 — confirmed live: short unrelated word pairs
# ("open" vs "opex", "apex" vs "opex", "cap" vs "capex") score exactly
# 75.0 on rapidfuzz.fuzz.ratio, a real false-positive collision between a
# common status word and a financial term. Genuine misspellings in this
# codebase's own vocabulary score comfortably higher (85.7-96.3 measured
# across corrective/mitigative/implementation/progress/stage/status) —
# 82 clears the collision with margin on both sides.
_RATIO_THRESHOLD = 82


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _word_matches(query_word: str, canonical_word: str) -> bool:
    """True if query_word is a plausible typo or truncation of canonical_word."""
    if len(query_word) < _MIN_PREFIX or len(canonical_word) < _MIN_PREFIX:
        return query_word == canonical_word
    if canonical_word.startswith(query_word) or query_word.startswith(canonical_word):
        return True
    return fuzz.ratio(query_word, canonical_word) >= _RATIO_THRESHOLD


def fuzzy_augment(question: str, terms: list[str] = CANONICAL_TERMS) -> str:
    """
    Return `question` with any fuzzily-matched canonical phrase from
    `terms` appended verbatim (never replacing or reordering the original
    text) — e.g. "show mit actions overdue" ->
    "show mit actions overdue [matched: mitigative action] [matched: overdue]".

    A multi-word canonical phrase only counts as matched when every one
    of its non-stopword words gets a hit somewhere in the question
    (order-independent — the phrase's words don't need to be adjacent or
    in the same order the user typed them). Already-present exact
    phrases are skipped — nothing to add when the literal regex will
    already catch it directly.
    """
    if not question:
        return question
    q_words = _tokenize(question)
    if not q_words:
        return question
    q_lower = question.lower()

    matched: list[str] = []
    for phrase in terms:
        if phrase in q_lower:
            continue
        phrase_words = [w for w in phrase.split() if w not in _STOPWORDS]
        if phrase_words and all(
            any(_word_matches(qw, pw) for qw in q_words) for pw in phrase_words
        ):
            matched.append(phrase)

    if not matched:
        return question
    return question + "".join(f" [matched: {m}]" for m in matched)
