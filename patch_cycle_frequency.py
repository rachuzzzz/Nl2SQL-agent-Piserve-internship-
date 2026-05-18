#!/usr/bin/env python3
"""
patch_cycle_frequency.py
Run from project root (where orchestrator.py, seed_examples.py, validator.py live):
    python3 patch_cycle_frequency.py

Fixes 9 faults in the cycle-frequency multi-turn failure trace.
"""
import os, sys, re, ast

ROOT = os.path.dirname(os.path.abspath(__file__))

def load(name):
    # Files are flat in project root — strip agent/ and core/ prefixes
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        print(f"  ERROR: {basename} not found at {p}"); sys.exit(1)
    with open(p) as f: return f.read()

def save(name, src):
    with open(os.path.join(ROOT, name), 'w') as f: f.write(src)

def check(name, src):
    try: ast.parse(src)
    except SyntaxError as e:
        print(f"  SYNTAX ERROR in {name}: {e}"); sys.exit(1)

changes = []

# ── seed_examples.py ──────────────────────────────────────────────────────────
src = load("agent/seed_examples.py")

MARKER2 = "# FIX2_CORE_CONTEXT"
if MARKER2 not in src:
    OLD = ('-- NEVER: ir.id in SELECT | NEVER: ica.risk_level_id vs string | NEVER: GROUP BY ica.risk_level_id\n"""')
    NEW = ('-- NEVER: ir.id in SELECT | NEVER: ica.risk_level_id vs string | NEVER: GROUP BY ica.risk_level_id\n'
           "-- completed_cycle = status='COMPLETED' (NOT 'CLOSED', NOT status!='DRAFT')\n"
           "-- NEVER: ipd.cycle_id (no such column on inspector_portfolio_details)\n"
           "-- Cycle->Freq: inspection_cycle.id -> inspector_portfolio.cycle_id\n"
           "--   -> inspector_portfolio_details.portfolio_id -> frequency_definition.id\n"
           '"""')
    src = src.replace(OLD, NEW, 1)
    src += "\n" + MARKER2 + "\n"
    changes.append("seed_examples.py: CORE_CONTEXT — completed_cycle + ipd.cycle_id rules")

MARKER1 = "# FIX1_CYCLE_SEEDS"
if MARKER1 not in src:
    seed1_sql = ("-- CYCLE FREQUENCY PATH (ipd has NO cycle_id):\n"
        "--   ic.id -> ip.cycle_id -> ipd.portfolio_id -> fd.id\n"
        "-- NEVER: ipd.cycle_id  |  completed = status='COMPLETED'\n"
        "SELECT ic.id AS cycle_id,\n"
        "       TO_CHAR(ic.start_date, 'Month YYYY') AS cycle_month,\n"
        "       ic.status, fd.name AS frequency_name,\n"
        "       fd.repeat_count, fd.repeat_interval, fd.repeat_unit,\n"
        "       COUNT(DISTINCT ipd.id) AS portfolio_detail_count\n"
        "FROM inspection_cycle ic\n"
        "JOIN inspector_portfolio ip          ON ip.cycle_id = ic.id\n"
        "JOIN inspector_portfolio_details ipd ON ipd.portfolio_id = ip.id\n"
        "JOIN frequency_definition fd         ON fd.id = ipd.frequency_definition_id\n"
        "WHERE ic.id = '<CYCLE_UUID>'\n"
        "GROUP BY ic.id, ic.start_date, ic.status,\n"
        "         fd.name, fd.repeat_count, fd.repeat_interval, fd.repeat_unit\n"
        "ORDER BY portfolio_detail_count DESC;")
    seed2_sql = ("-- completed_cycle = status = 'COMPLETED'\n"
        "SELECT ic.id AS cycle_id,\n"
        "       TO_CHAR(ic.start_date, 'Month YYYY') AS cycle_month,\n"
        "       ic.status, fd.name AS frequency_name,\n"
        "       fd.repeat_count, fd.repeat_interval, fd.repeat_unit,\n"
        "       COUNT(DISTINCT ipd.id) AS portfolio_detail_count\n"
        "FROM inspection_cycle ic\n"
        "JOIN inspector_portfolio ip          ON ip.cycle_id = ic.id\n"
        "JOIN inspector_portfolio_details ipd ON ipd.portfolio_id = ip.id\n"
        "JOIN frequency_definition fd         ON fd.id = ipd.frequency_definition_id\n"
        "WHERE ic.id = (\n"
        "    SELECT id FROM inspection_cycle WHERE status = 'COMPLETED'\n"
        "    ORDER BY end_date DESC LIMIT 1\n"
        ")\n"
        "GROUP BY ic.id, ic.start_date, ic.status,\n"
        "         fd.name, fd.repeat_count, fd.repeat_interval, fd.repeat_unit\n"
        "ORDER BY portfolio_detail_count DESC;")
    seed3_sql = ("-- completed_cycle: status = 'COMPLETED' only\n"
        "SELECT ic.id AS inspection_cycle_id,\n"
        "       TO_CHAR(ic.start_date, 'Month YYYY') AS month,\n"
        "       ic.status, ic.start_date, ic.end_date, ic.modified_on\n"
        "FROM inspection_cycle ic\n"
        "WHERE ic.status = 'COMPLETED'\n"
        "ORDER BY ic.end_date DESC LIMIT 1;")

    new_block = (
        '\n\n    # -- Cycle / Frequency multi-turn --\n'
        '    (\n        "what is the frequency of a specific inspection cycle",\n'
        '        """\\\n' + seed1_sql + '\n""",\n    ),\n'
        '    (\n        "frequency of the most recently completed inspection cycle",\n'
        '        """\\\n' + seed2_sql + '\n""",\n    ),\n'
        '    (\n        "most recently completed inspection cycle",\n'
        '        """\\\n' + seed3_sql + '\n""",\n    ),\n'
    )
    idx = src.rfind("\n]")
    src = src[:idx] + new_block + "\n]\n" + MARKER1 + "\n"
    changes.append("seed_examples.py: 3 new cycle/frequency seed examples")

check("agent/seed_examples.py", src)
save("agent/seed_examples.py", src)
print("  ✓ seed_examples.py")

# ── orchestrator.py ───────────────────────────────────────────────────────────
src = load("agent/orchestrator.py")

MARKER3 = "# FIX3_CYCLE_IDS"
if MARKER3 not in src:
    old3 = "    single_row_values: dict = field(default_factory=dict) # col\u2192val when result is 1 row"
    if old3 in src:
        src = src.replace(old3, old3 + "\n    cycle_ids: list = field(default_factory=list)  " + MARKER3, 1)
        changes.append("orchestrator.py: cycle_ids on ConversationTurn")

MARKER4 = "# FIX4_CYCLE_CTX"
if MARKER4 not in src:
    old4 = ('            elif last.single_row_values:\n'
            '                lines.append(f"CONTEXT \u2014 Last query returned a single result: {last.single_row_values}")')
    if old4 in src:
        src = src.replace(old4, old4 + "\n"
            "            _cyc = getattr(last, 'cycle_ids', [])\n"
            "            if _cyc and len(_cyc) == 1:\n"
            "                lines.append(\n"
            "                    f\"CONTEXT \u2014 conversation about inspection_cycle id='{_cyc[0]}'.\"\n"
            "                    f\" Filter: WHERE ic.id = '{_cyc[0]}' AND ic.status = 'COMPLETED'.\"\n"
            "                    f\" NEVER change status to != 'DRAFT'.\"\n"
            "                )\n"
            "            " + MARKER4, 1)
        changes.append("orchestrator.py: build_context_block — cycle_id CONTEXT line")

MARKER5 = "# FIX5_FREQ_HINT"
if MARKER5 not in src:
    old5 = "                # Preserve explicit LIMIT from user question"
    if old5 in src:
        src = src.replace(old5,
            "                _CYC_FREQ_RE = re.compile(\n"
            "                    r'\\b(frequency|how\\s+often|repeat).{0,60}(cycle|portfolio)\\b'\n"
            "                    r'|\\b(cycle|portfolio).{0,60}(frequency|how\\s+often)\\b',\n"
            "                    re.IGNORECASE)\n"
            "                if _CYC_FREQ_RE.search(question):\n"
            "                    args['schema_hint'] = args.get('schema_hint','') + (\n"
            "                        '\\nCRITICAL Cycle->Freq (ipd NO cycle_id):'\n"
            "                        '\\n  ic.id->ip.cycle_id->ipd.portfolio_id->fd.id'\n"
            "                        '\\nNEVER ipd.cycle_id | completed=\\'COMPLETED\\''\n"
            "                        '\\nGROUP BY all non-agg SELECT cols.\\n')\n"
            "                " + MARKER5 + "\n\n"
            "                # Preserve explicit LIMIT from user question", 1)
        changes.append("orchestrator.py: auto-inject cycle->frequency join hint")

MARKER6 = "# FIX6_IPD_MSG"
if MARKER6 not in src:
    old6 = ("        if cm and not tm:\n"
        "            parts.append(f\"Column '{cm.group(1)}' doesn't exist. Use get_schema to check columns.\")")
    if old6 in src:
        src = src.replace(old6,
            "        if cm and not tm:\n"
            "            _bc = cm.group(1)\n"
            "            if _bc == 'cycle_id':\n"
            "                parts.append('inspector_portfolio_details has NO cycle_id.'\n"
            "                    ' Path: ic.id->ip.cycle_id->ipd.portfolio_id->fd.id')\n"
            "            else:\n"
            "                parts.append(f\"Column '{_bc}' doesn't exist. Use get_schema.\")\n"
            "            " + MARKER6, 1)
        changes.append("orchestrator.py: _schema_hint_for_error — ipd.cycle_id message")

MARKER78 = "# FIX78"
if MARKER78 not in src:
    old78 = "        gen_sql_failures = 0\n        _last_sql = \"\""
    if old78 in src:
        src = src.replace(old78,
            "        gen_sql_failures = 0\n"
            "        _failed_patterns: set = set()  " + MARKER78 + "\n"
            "        _last_sql = \"\"", 1)
        changes.append("orchestrator.py: _failed_patterns retry tracker")

MARKER3B = "# FIX3B"
if MARKER3B not in src:
    old3b = "        # If only 1 row, store all values for maximum context"
    if old3b in src:
        src = src.replace(old3b,
            "        cycle_ids = []\n"
            "        for _cr in last_rows[:5]:\n"
            "            if not isinstance(_cr, dict): continue\n"
            "            for _cc in ('inspection_cycle_id', 'cycle_id'):\n"
            "                _cv = _cr.get(_cc)\n"
            "                if _cv and re.match(r'^[0-9a-f-]{36}$', str(_cv), re.I):\n"
            "                    if str(_cv) not in cycle_ids: cycle_ids.append(str(_cv))\n"
            "        " + MARKER3B + "\n\n"
            "        # If only 1 row, store all values for maximum context", 1)
        changes.append("orchestrator.py: extract cycle_ids in _finish")

    old_turn = ("                single_row_values=single_row_vals,\n            ))")
    if old_turn in src:
        src = src.replace(old_turn,
            "                single_row_values=single_row_vals,\n"
            "                cycle_ids=cycle_ids[:3],\n"
            "            ))", 1)
        changes.append("orchestrator.py: cycle_ids passed to session.add_turn")

check("agent/orchestrator.py", src)
save("agent/orchestrator.py", src)
print("  ✓ orchestrator.py")

# ── validator.py ──────────────────────────────────────────────────────────────
src = load("core/validator.py")

MARKER9 = "# FIX9_IPD_CYCLE"
if MARKER9 not in src:
    m9 = re.search(r'        # [\u2500-]{2,} 7\. Dynamic fb_\{uuid\} table', src)
    if m9:
        old9 = m9.group(0)
        src = src.replace(old9,
            "        # 6b. ipd.cycle_id HARD_FAIL  " + MARKER9 + "\n"
            "        if re.search(r'\\bipd\\.cycle_id\\b', sql_lower):\n"
            "            issues.append(ValidationIssue(\n"
            '                "HARD_FAIL", "WRONG_COLUMN",\n'
            '                "ipd.cycle_id does not exist; join via inspector_portfolio.cycle_id",\n'
            '                detail="ic.id -> ip.cycle_id -> ipd.portfolio_id -> fd.id",\n'
            "            ))\n"
            '            ValidatorStats.record("WRONG_COLUMN", is_retry_trigger=True)\n\n'
            + old9, 1)
        changes.append("validator.py: HARD_FAIL for ipd.cycle_id")
    else:
        print("  WARN: validator section 7 anchor not found — skipping")

check("core/validator.py", src)
save("core/validator.py", src)
print("  ✓ validator.py")

print(f"\n{'='*52}")
print(f"  {len(changes)} fix(es) applied:\n")
for i, c in enumerate(changes, 1): print(f"  {i}. {c}")
if not changes: print("  Nothing to do — all patches already applied.")
print(f"\n  Restart uvicorn to apply.\n{'='*52}\n")