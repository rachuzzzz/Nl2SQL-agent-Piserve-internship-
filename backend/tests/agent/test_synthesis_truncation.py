"""
Regression test for synthesis.py's execute_sql multi-row formatter — the
100-char per-value truncation exists to stop a wall of text across MANY rows,
not to shorten the one thing a small result is actually about.

Confirmed live 2026-07-29 (qid=25cbdee7): "show the observation raised for
which the above action was proposed" returned exactly 1 row whose whole point
was aca.observation_text (a long free-text finding) — the chat summary cut it
to 97 chars + "..." even though the separate rows/columns payload the
frontend also renders had the full text, so the chat bubble and the table
visibly disagreed.
"""

from src.agent.orchestrator.synthesis import SynthesisMixin

LONG_TEXT = (
    "The organization has identified risks and opportunities related to its QMS; "
    "however, there was insufficient evidence to demonstrate that actions taken to "
    "address these risks and opportunities are consistently planned, implemented, and "
    "monitored for effectiveness. As a result, the organization cannot fully demonstrate "
    "that the QMS is achieving intended results, preventing undesired effects, and "
    "driving continual improvement."
)


def _tool_result(rows, columns):
    return {
        "success": True,
        "result": {
            "rows": rows,
            "columns": columns,
            "truncated": False,
            "total_count": len(rows),
        },
    }


def test_small_result_does_not_truncate_long_text_field():
    synth = SynthesisMixin()
    tool_result = _tool_result(
        [{
            "corrective_action_id": "2026/06/ST041/AUD003_CA004",
            "observation_text": LONG_TEXT,
            "facility_name": "Aldar HQ",
            "report_id": "2026/06/ST041/AUD003",
        }],
        ["corrective_action_id", "observation_text", "facility_name", "report_id"],
    )
    answer = synth._synthesize("show the observation for this CA", "execute_sql", tool_result)
    assert LONG_TEXT in answer
    assert "..." not in answer


def test_large_result_still_truncates_long_text_field():
    synth = SynthesisMixin()
    # Vary the text per row — an identical value on every row gets promoted to
    # the (untruncated) header by the constant-column optimization, which
    # would defeat this test rather than exercise the per-value truncation.
    rows = [
        {
            "corrective_action_id": f"CA{i:03d}",
            "observation_text": f"{LONG_TEXT} (row {i})",
            "facility_name": "Aldar HQ",
        }
        for i in range(10)
    ]
    tool_result = _tool_result(rows, ["corrective_action_id", "observation_text", "facility_name"])
    answer = synth._synthesize("list overdue corrective actions", "execute_sql", tool_result)
    assert LONG_TEXT not in answer
    assert "..." in answer
