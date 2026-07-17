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
