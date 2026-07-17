"""
Pure-Python formatting/detection helpers for the /api/chat and /api/page
routes — no FastAPI, DB, or agent-layer dependencies. Extracted out of
routes/chat.py, which had grown to mix HTTP routing with this business
logic (chart follow-up detection, UUID scrubbing, security redaction,
answer cleaning, and the zero-LLM insight summary). Keeping these here
lets the route handlers stay thin request/response glue and lets this
logic be tested without spinning up the FastAPI app.
"""

import re


# ── Chart follow-up fast-path ──────────────────────────────────────────────────
# Detects pure chart requests that need no new SQL — handled before the LLM.

_VISUAL_KW = re.compile(
    r'\b(chart|graph|plot|visual(i[sz]e|ization)?|bar|pie|line|doughnut|donut|scatter|draw)\b'
    r'|\bshow\s+as\s+chart\b|\bas\s+(a\s+)?chart\b',
    re.IGNORECASE,
)
_DOMAIN_KW = re.compile(
    r'\b(inspection|facility|capex|opex|score|mitigative|action|overdue|report|month|'
    r'year|week|average|total|count|inspector|manager|status|pending|approved|rejected|'
    r'deferred|finding|observation|po|purchase|order|amount|budget|responsible|closed|open)\b',
    re.IGNORECASE,
)
_TYPE_RE = re.compile(
    r'\b(horizontal\s+bar|bar|line|trend|pie|doughnut|donut|scatter)\b',
    re.IGNORECASE,
)
_TYPE_MAP = {
    'bar': 'bar', 'horizontal bar': 'bar', 'line': 'line', 'trend': 'line',
    'pie': 'pie', 'doughnut': 'doughnut', 'donut': 'doughnut', 'scatter': 'scatter',
}

def is_chart_followup(q: str) -> bool:
    """True when message is purely a chart request with no new data intent."""
    return (bool(_VISUAL_KW.search(q))
            and not bool(_DOMAIN_KW.search(q))
            and len(q.split()) <= 20)

def extract_requested_type(q: str) -> str | None:
    m = _TYPE_RE.search(q)
    return _TYPE_MAP.get(m.group(1).lower().strip()) if m else None


# ── UUID column scrubber ──────────────────────────────────────────────────────

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE
)


def strip_uuid_columns(columns: list, rows: list) -> tuple[list, list]:
    """
    Drop any column whose values are raw UUIDs before the result table reaches
    the frontend. UUIDs (user_id, facility_id, etc.) are needed server-side to
    resolve/join entities but are meaningless — and look like a bug — to a
    person reading the chat table, so they're scrubbed here regardless of
    which tool/path produced the rows.
    """
    if not columns or not rows:
        return columns, rows
    uuid_cols = {
        col for col in columns
        if any(_UUID_RE.match(str(row.get(col, "")).strip()) for row in rows if row.get(col))
    }
    if not uuid_cols:
        return columns, rows
    kept_cols = [c for c in columns if c not in uuid_cols]
    kept_rows = [{k: v for k, v in row.items() if k not in uuid_cols} for row in rows]
    return kept_cols, kept_rows


_SCOPE_PREDICATE_RE = re.compile(
    r"ir\.(?:inspector|inspectee)_user_id\s*=\s*\(\s*SELECT\s+id\s+FROM\s+users\s+WHERE\s+id_fk\s*=\s*\d+[^)]*\)",
    re.IGNORECASE,
)
_ID_FK_RE = re.compile(r"id_fk\s*=\s*\d+", re.IGNORECASE)
# No trailing \)? — consuming an optional closing paren here ate it on redact,
# leaving unbalanced markdown like "(Drupal UID [redacted]" in the response.
_DRUPAL_UID_RE = re.compile(r"Drupal\s+UID[\s:(]*\d+", re.IGNORECASE)
# Catches the model explaining the scope mechanism in prose instead of the
# literal SQL shape _SCOPE_PREDICATE_RE expects — e.g. "Inspectors see
# inspections they conduct (via inspector_user_id)" or "WHERE ur.user_id =
# 11223" — observed live in production logs from an ordinary, non-adversarial
# question ("how do you determine what I'm allowed to see"), which the SQL-
# shaped regex above does not match at all.
_FK_COLUMN_MENTION_RE = re.compile(
    r"`?\b(?:inspector_user_id|inspectee_user_id)\b`?", re.IGNORECASE
)
_USER_ID_VALUE_RE = re.compile(r"\buser_id\s*=\s*\d+", re.IGNORECASE)


def redact_security_internals(text: str | None) -> str | None:
    """Deterministic backstop for the system prompt's "never expose this
    constraint" instruction, which the model doesn't reliably honor when
    directly asked about its filters/mechanism. Applied to LLM-authored
    text right before it reaches the client — enforce_scope() itself
    doesn't care what the LLM says, so this is purely about not handing an
    attacker the enforcement mechanism to study.
    """
    if not text:
        return text
    text = _SCOPE_PREDICATE_RE.sub("[access-control filter]", text)
    text = _ID_FK_RE.sub("id_fk = [redacted]", text)
    text = _DRUPAL_UID_RE.sub("[redacted]", text)
    text = _FK_COLUMN_MENTION_RE.sub("[access-control column]", text)
    text = _USER_ID_VALUE_RE.sub("user_id = [redacted]", text)
    return text


# ── Answer cleaner ────────────────────────────────────────────────────────────

def clean_answer(answer: str, result_rows: list) -> str:
    """
    Keep the LLM's natural-language INSIGHT but strip raw data lists/tables
    since the UI renders the data table separately.

    Strips:
    - Markdown tables (| col | col |)
    - Numbered data rows: "1. facility_name: X | total_capex: Y"
    - Bullet data rows: "- X: 123.00" or "• X: 123"
    - "Found N result(s):" / "Here are the results:" preamble lines
    - "... and N more shown" trailing lines
    - Excessive blank lines

    Keeps: natural language sentences, insights, summaries.
    """
    if not answer:
        return answer

    lines = answer.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines at this stage (re-add single blank later)
        if not stripped:
            cleaned.append("")
            continue

        # Skip markdown table rows and separators
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        if re.match(r"^[\|\-\s:]+$", stripped) and "|" in stripped:
            continue

        # Skip numbered data rows: "1. col: val | col: val" or "  1. col: val"
        if re.match(r"^\d+\.\s+\w.*\|", stripped):
            continue
        # Also catch "1. facility_name: X" without pipes
        if re.match(r"^\d+\.\s+\w+[\w_]+:\s+", stripped):
            continue

        # Skip bullet data rows: "- X: 123" or "• X: 123" or "* X: 123"
        if re.match(r"^[-•*]\s+\S.*:\s+[\d,\.]+\s*$", stripped):
            continue
        # Skip bullet rows with pipes: "• facility: X | capex: Y"
        if re.match(r"^[-•*]\s+\S.*\|", stripped):
            continue

        # Skip "Found N result(s)..." lines — count is shown in table meta
        if re.match(r"^found\s+\d+\s+result", stripped, re.IGNORECASE):
            continue
        if re.match(r"^results?\s*\(\d+", stripped, re.IGNORECASE):
            continue
        if re.match(r"^showing\s+(first\s+)?\d+", stripped, re.IGNORECASE):
            continue

        # Skip "... and N more shown" trailing lines
        if re.match(r"^\.\.\.\s+and\s+\d+\s+more", stripped, re.IGNORECASE):
            continue

        # Skip "Here are the results:" / "Here is the data:" preambles
        if re.match(r"^here (are|is) the (results?|data|following|capex|list)", stripped, re.IGNORECASE):
            continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # Collapse 3+ consecutive blank lines to 1
    result = re.sub(r"\n{3,}", "\n\n", result)

    # If after cleaning we have nothing meaningful, fall back gracefully
    if not result or len(result.strip()) < 15:
        n = len(result_rows)
        return f"Found {n} result{'s' if n != 1 else ''}."

    return result.strip()


# ── Summary generator ────────────────────────────────────────────────────────

def generate_summary(question: str, columns: list, rows: list) -> str:
    """Pure Python insight summary — zero LLM calls."""
    if not rows or not columns or len(rows) < 2:
        return ""
    try:
        def _coerce(v):
            try:
                return float(str(v).replace(",", "").strip())
            except Exception:
                return None

        def fmt(v):
            if v >= 1_000_000: return f"{v/1_000_000:.2f}M"
            if v >= 1_000: return f"{v:,.0f}"
            return f"{v:.2f}"

        def _fmt_label(v) -> str:
            """Return a human-readable label, formatting datetimes nicely."""
            from datetime import datetime, date

            def _fmt_dt(dt: datetime) -> str:
                # day==1 at midnight → DATE_TRUNC('month', ...) pattern → show month only
                if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.day == 1:
                    return dt.strftime("%b %Y")
                if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                    return dt.strftime("%b %d, %Y").replace(" 0", " ")
                return dt.strftime("%b %d, %Y %H:%M").replace(" 0", " ")

            s = str(v).strip()
            # Already a date/datetime object
            if isinstance(v, datetime):
                return _fmt_dt(v)
            if isinstance(v, date):
                if v.day == 1:
                    return v.strftime("%b %Y")
                return v.strftime("%b %d, %Y").replace(" 0", " ")
            # ISO timestamp strings: 2026-01-01 00:00:00+00:00 or 2026-01-01T00:00:00Z
            if re.match(r'^\d{4}-\d{2}-\d{2}[ T]', s):
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                    return _fmt_dt(dt)
                except Exception:
                    pass
            # Plain date string: 2026-01-01
            if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
                try:
                    dt = datetime.strptime(s, "%Y-%m-%d")
                    return _fmt_dt(dt)
                except Exception:
                    pass
            return s

        num_cols, cat_cols = [], []
        for col in columns:
            sample = [rows[i].get(col) for i in range(min(5, len(rows))) if rows[i].get(col) is not None]
            if sample and all(_coerce(v) is not None for v in sample):
                num_cols.append(col)
            else:
                cat_cols.append(col)

        n = len(rows)
        parts = []
        for nc in num_cols[:2]:
            vals = [(row, _coerce(row.get(nc))) for row in rows if _coerce(row.get(nc)) is not None]
            if not vals:
                continue
            vals.sort(key=lambda x: x[1], reverse=True)
            top_row, top_val = vals[0]
            total = sum(v for _, v in vals)
            label = None
            for cc in cat_cols:
                lv = top_row.get(cc)
                if lv and str(lv).strip() not in ("None", ""):
                    label = _fmt_label(lv)
                    break
            col_label = nc.replace("_", " ").title()
            if label:
                parts.append(f"{label} leads with {fmt(top_val)} in {col_label}.")
                if n > 2 and abs(total - top_val) > 0.01:
                    entity = "facilities" if any("facilit" in c for c in cat_cols) else "records"
                    parts.append(f"Total across {n} {entity} is {fmt(total)}.")
            else:
                parts.append(f"Highest {col_label} is {fmt(top_val)} across {n} records.")

        if not parts and cat_cols:
            parts.append(f"{n} records returned.")

        return " ".join(parts)
    except Exception:
        return ""


# ── Chart suggestion sentence ─────────────────────────────────────────────────

TYPE_LABELS = {
    'bar': 'Bar Chart', 'line': 'Line Chart', 'pie': 'Pie Chart',
    'doughnut': 'Doughnut Chart', 'scatter': 'Scatter Plot',
}
ALT_MAP = {
    'bar':      ['line', 'pie'],
    'line':     ['bar', 'scatter'],
    'pie':      ['doughnut', 'bar'],
    'doughnut': ['pie', 'bar'],
    'scatter':  ['line', 'bar'],
}


# ── SQL pagination ────────────────────────────────────────────────────────────

def paginate_sql(sql: str, page: int, page_size: int) -> str:
    """
    Replace (or append) LIMIT/OFFSET in an SQL query for the requested page.
    page is 0-based.
    """
    offset = page * page_size
    # Remove existing LIMIT / OFFSET
    stripped = re.sub(
        r'\bLIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*;?\s*$', '',
        sql.strip(), flags=re.IGNORECASE
    ).strip().rstrip(';')
    return f"{stripped}\nLIMIT {page_size} OFFSET {offset};"
