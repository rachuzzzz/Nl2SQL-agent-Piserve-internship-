"""
question_patterns.py — SINGLE SOURCE OF TRUTH for shared "is this question
about topic X" regexes.

Several independent prompt-assembly layers need to agree on the same
topic-detection regex — e.g. whether a question is "about mitigative
actions" — so that they fire together on the same questions:
  1. agent/prompts/builder.py            → CONDITIONAL_BLOCKS (Llama-side facts)
  2. agent/prompts/examples_tier2.py     → EXAMPLES_TIER2 (Llama-side few-shots)
  3. sql/prompts.py                      → CORE_CONTEXT_CONDITIONAL (Deepseek-side)
  4. sql/rules/inject_rules.py           → INJECT hints (SQL-gen routing)
  5. knowledge/schema_introspector.py    → DatabaseSchema.for_schema_hint's
                                            table_relationships topic-gating
                                            (Deepseek-side, Phase 3.16)

Before this module existed these regexes were hand-copied between files
with a "KEPT IN SYNC — if you edit one, edit the other" comment and no
enforcement. Define each shared pattern ONCE here and import it everywhere
it's needed — editing a pattern now updates every consumer at once.

Not every question_pattern in the codebase belongs here — only ones actually
shared by 2+ files. Patterns used in exactly one place should stay defined
inline where they're used.
"""

# Shared by CONDITIONAL_BLOCKS["MA workflow columns"] (prompts/builder.py),
# CORE_CONTEXT_CONDITIONAL["MA workflow columns (Deepseek)"] (examples/builder.py),
# and EXAMPLES_TIER2["MA-related worked examples..."] (examples_tier2.py) —
# all three gate on "is this question about the MA workflow at all".
MA_WORKFLOW_TOPIC = (
    r"(?i)\bmitigative\s+actions?\b|\bMAs?\b"
    r"|\bcapex\b|\bopex\b|\bdeferred\b"
    r"|\bprogress[_\s]stage\b|\bimplementation[_\s]status\b"
    r"|\bneeds?\s+attention\b|\bcritical\s+issues?\b"
)

# Shared by inject_rules.py's "inject_ma_approval_workflow" (canonical — the
# SQL-routing hint this was written for) and examples_tier2.py's "MAs
# awaiting approval" worked example — both should fire on the same
# approval/progress-status questions.
MA_APPROVAL_WORKFLOW_TOPIC = (
    r"(?i)\b(awaiting|pending)\s*(for\s+)?(approval|review)\b.{0,40}\b(action|MA)s?\b"
    r"|\b(action|MA)s?\b.{0,40}\b(awaiting|pending)\s*(for\s+)?(approval|review)\b"
    r"|\bapproval\b.{0,30}\b(action|MA)s?\b"
    r"|\b(approved|sent\s+back|returned)\b.{0,30}\b(action|MA)s?\b"
    r"|\b(at\s+risk|behind\s+schedule|on\s+track|not\s+started)\b.{0,30}\b(action|MA)s?\b"
    r"|\b(action|MA)s?\b.{0,40}\b(at\s+risk|behind|on\s+track|approved?|pending|sent\s+back|returned)\b"
)

# Shared by inject_rules.py's "inject_financial_routing" (canonical) and
# examples_tier2.py's "Total capex generated converted and lost" worked
# example — both should fire on the same CAPEX/OPEX financial questions.
# Used by sql/prompts.py's CORE_CONTEXT_CONDITIONAL to gate the ai_answers/
# ai_questions form-answer schema block (aa.score semantics, answer_text
# encoding, module_name filtering, submission_id grouping — ~1250 tokens) out
# of the SQL-gen prompt on questions that never touch form-answer data at
# all. Was previously ALWAYS-ON (baked into sql_generation_prompt.txt's
# static skeleton), unlike the CAPEX/P&L block, which was already gated —
# Phase 3.7 latency profiling (2026-08-19) found this the largest remaining
# unconditional, topic-specific chunk in the inspection skeleton.
#
# Validated against all 99 single-turn questions in eval.py's real eval
# corpus (ground truth = does generate_sql()'s ACTUAL output reference
# ai_answers/ai_questions/submission_id/aa.score/answer_text/module_name):
# TP=20 TN=76 FP=1 FN=2, fires on 21/99 (21%). The 2 false negatives are
# accepted, not chased further — EDG-04 is literal nonsense input with no
# lexical signal to key off; CROSS-03 only touches ai_answers via a
# pre-existing, unrelated model bug (misapplies the skeleton's own
# FACILITY-LEVEL NEGATION worked example), not a real need for this
# guidance.
AI_ANSWERS_TOPIC = (
    r"(?i)\bobservations?\b"
    r"|\brisk\s+levels?\b"
    r"|\bform\s+(answers?|data)\b|\binspection\s+form\b"
    r"|\bquestions?\b.{0,30}\banswers?\b|\banswers?\b.{0,30}\bquestions?\b"
    r"|\bscores?\b.{0,30}\bquestions?\b|\bquestions?\b.{0,30}\bscores?\b"
    r"|\bindexed\b"
    r"|\bperform(ing|ance)?\b"
)

# Shared by CONDITIONAL_BLOCKS["User roles"/"Approval workflow"/"Communication
# groups"] (agent/prompts/builder.py, reasoning-LLM side — moved here 2026-08-24
# rather than duplicated) and DatabaseSchema.for_schema_hint's table_relationships
# topic-gating (schema_introspector.py, SQL-gen side, Phase 3.16) — both prompt
# layers should agree on which niche topics a question touches, same rationale
# as MA_WORKFLOW_TOPIC above. Patterns are deliberately broad — a false positive
# only costs tokens, a false negative silently drops a "confirmed live" rule.
USER_ROLES_TOPIC = (
    r"(?i)\brole\b|\binspector\b|\bfm\s+(manager|engineer|admin|supervisor)\b"
    r"|\basst\s+fm\b|\bhse\s+officer\b"
)
APPROVAL_WORKFLOW_TOPIC = r"(?i)\bapprov|\breview\b|\breject|\bcompleted\b|\bapproval\s+levels?\b"
COMMUNICATION_GROUPS_TOPIC = r"(?i)\bcommunication\s+group|\bnotif|\brecipients?\b"

FINANCIAL_ROUTING_TOPIC = (
    r"\b(po[_\s]detail|pr[_\s]detail|purchase\s+order|purchase\s+requisition|"
    r"purchase\s+order\s+amount|po\s+amount|total\s+po|"
    r"capex.{0,30}(generat|convert|lost|loss|incurr)|"
    r"opex.{0,30}(incurr|facilit|spend|higher|exceed)|"
    r"(generat|convert|lost|incurr).{0,30}(capex|opex)|"
    r"how\s+much.{0,40}(convert|capex|opex|po|purchase)|"
    r"opex.{0,30}(higher|more|greater|exceed).{0,30}capex|"
    r"conversion\s+rate|capex\s+conversion|"
    r"(convert|capex).{0,20}(inspectee|fm|manager|facility)|"
    # plural 'converters' / 'generators' explicitly
    r"(capex|opex)\s+(converter|generator)s?|"
    r"(converter|generator)s?\s+(capex|opex)|"
    r"top.{0,15}(capex|opex).{0,15}(converter|generator)s?|"
    # facility cost / expenditure queries
    r"(facility|site).{0,30}(costs?|spend|expenditure|expensive)|"
    r"(cost|spend|expenditure).{0,30}(facility|site)|"
    r"most\s+expensive\s+(facility|site)|"
    r"rival.{0,20}(quote|bid|amount)|financial\s+summary)\b"
)
