"""
Regression test: inject_row_identity()'s __id-suffixed columns (see
src/sql/validation/identity.py) must never leak into the deterministic
answer text _synthesize() builds for execute_sql results.

Confirmed live 2026-07-30 via an eval run: 18 of 74 questions had a raw
UUID appear directly in the user-facing answer, e.g.
"1. title: del 01 | ... | title__id: 54327f53-8fb2-454b-bcde-d9d2a8f25ae7"
— a real production leak, not eval-specific (eval_audit.py's
answer_preview is just result.answer[:250], the SAME text a real user
would see).
"""

import re

from src.agent.orchestrator.synthesis import SynthesisMixin


class _FakeOrchestrator(SynthesisMixin):
    domain = "audit"
    db_engine = None
    _COUNT_RE = re.compile(r"\bhow many\b|\bcount\b", re.IGNORECASE)


def test_id_column_never_appears_in_multi_row_answer_text():
    orch = _FakeOrchestrator()
    rows = [
        {"title": "del 01", "status": "APPROVED",
         "title__id": "54327f53-8fb2-454b-bcde-d9d2a8f25ae7"},
        {"title": "del 02", "status": "SUBMITTED",
         "title__id": "aaaa1111-0000-0000-0000-000000000002"},
    ]
    tool_result = {
        "success": True,
        "result": {"rows": rows, "columns": list(rows[0].keys()),
                   "total_count": 2, "row_count": 2, "truncated": False},
    }
    answer = orch._synthesize("list all audit plans", "execute_sql", tool_result)
    assert "__id" not in answer
    assert "54327f53" not in answer
    assert "del 01" in answer and "del 02" in answer  # real data still present


def test_id_column_never_appears_when_promoted_as_constant():
    """A single-valued __id column across all rows would otherwise get
    promoted into the header's constant-column summary — must still be
    excluded there too."""
    orch = _FakeOrchestrator()
    rows = [
        {"title": "level 1", "audit_plan_title__id": "c58b5540-6cac-41ea-a308-2020ee9497b2"},
        {"title": "level 2", "audit_plan_title__id": "c58b5540-6cac-41ea-a308-2020ee9497b2"},
    ]
    tool_result = {
        "success": True,
        "result": {"rows": rows, "columns": list(rows[0].keys()),
                   "total_count": 2, "row_count": 2, "truncated": False},
    }
    answer = orch._synthesize("review status", "execute_sql", tool_result)
    assert "__id" not in answer
    assert "c58b5540" not in answer
