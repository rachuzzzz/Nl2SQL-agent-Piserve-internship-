"""
data_health.py — pre-flight snapshot health check for the NL2SQL eval.

PROBLEM THIS SOLVES
===================
The normalized answer layer (`ai_answers` / `ai_questions` for inspection,
`ai_audit_answers` for audit) was populated by a one-time bulk insert. In a
stripped DB snapshot the `ai_answers.inspection_report_id` FK can dangle —
values that don't resolve to any `inspection_report.id` — so every observation
/ form-reconstruction query comes back empty. That is a *data* problem, not an
*agent* problem, but the eval scores it as a `real_pass` failure and a
`truth_match` miss, which drags the headline number down and hides real
regressions.

This module runs a handful of cheap COUNT queries BEFORE the suite, decides
whether each answer-layer dependency is `ok` / `degraded` / `broken`, and hands
the runner a set of "blocked" dependencies. Questions tagged with a
`data_dependency` that is blocked still run (so a repaired snapshot
auto-recovers them) but their failures are bucketed as `snapshot_blocked` and
excluded from the headline `real_pass` / `truth_pass` rates.

Never raises — every check is wrapped; a query error records
`status="broken", reason="preflight query error: ..."` (mirrors
eval_truth.run_truth_check's contract).

USAGE
=====
    from data_health import run_preflight, blocked_dependencies, classify_row
    report      = run_preflight(orch.db_engine, "inspection")
    blocked     = blocked_dependencies(report)          # set[str]
    snap_blocked = classify_row(test.get("data_dependency"),
                                blocked, real_pass, truth_match)
"""

from __future__ import annotations

from typing import Any

# ── dependency tags (must match the `data_dependency` values in the TEST_SETs) ─
DEP_AI_ANSWERS = "ai_answers"
DEP_AI_AUDIT_ANSWERS = "ai_audit_answers"

# ── thresholds ───────────────────────────────────────────────────────────────
# form_rows_resolved == 0 is the unambiguous "the load is broken" signal: a
# healthy snapshot resolves essentially every conduct-form answer row to its
# parent report. The ratio band catches a partial load. MIN_FORM_ROWS is kept
# low on purpose — the eval snapshots are small (100-600 form answer rows even
# when healthy) — so it only distinguishes "some real data" from "a handful of
# stragglers", never gates on absolute volume.
MIN_FORM_ROWS = {"inspection": 100, "audit": 50}
FK_OK_RATIO = 0.80
FK_BROKEN_RATIO = 0.20


def _scalar(conn, sql: str) -> Any:
    from sqlalchemy import text
    return conn.execute(text(sql)).scalar()


def _check_ai_answers(conn) -> dict:
    """Inspection answer layer: ai_answers -> inspection_report."""
    total_fk = _scalar(conn, """
        SELECT COUNT(*) FROM ai_answers WHERE inspection_report_id IS NOT NULL
    """) or 0
    fk_dangling = _scalar(conn, """
        SELECT COUNT(*) FROM ai_answers aa
        WHERE aa.inspection_report_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM inspection_report ir
                          WHERE ir.id = aa.inspection_report_id)
    """) or 0
    fk_resolve_ratio = 1.0 - (fk_dangling / total_fk) if total_fk else 0.0

    form_rows_resolved = _scalar(conn, """
        SELECT COUNT(*)
        FROM ai_answers aa
        JOIN inspection_report ir ON ir.id = aa.inspection_report_id
        WHERE aa.module_name = 'Inspection Form'
    """) or 0

    obs_total = _scalar(conn, """
        SELECT COUNT(*) FROM inspection_corrective_action
        WHERE observation_id IS NOT NULL
    """) or 0
    obs_resolved = _scalar(conn, """
        SELECT COUNT(*) FROM inspection_corrective_action ica
        WHERE ica.observation_id IS NOT NULL
          AND EXISTS (SELECT 1 FROM ai_answers aa
                      WHERE aa.submission_id = ica.observation_id::varchar)
    """) or 0
    obs_join_resolve_ratio = (obs_resolved / obs_total) if obs_total else 0.0

    observation_answer_rows = _scalar(conn, """
        SELECT COUNT(*)
        FROM ai_answers aa
        JOIN ai_questions aq ON aa.element_id = aq.element_id
        WHERE aq.label = 'Observation' AND aa.module_name = 'Inspection Form'
    """) or 0

    return _verdict("inspection", dict(
        total_fk=total_fk, fk_dangling=fk_dangling,
        fk_resolve_ratio=round(fk_resolve_ratio, 4),
        form_rows_resolved=form_rows_resolved,
        obs_total=obs_total, obs_resolved=obs_resolved,
        obs_join_resolve_ratio=round(obs_join_resolve_ratio, 4),
        observation_answer_rows=observation_answer_rows,
    ))


def _check_ai_audit_answers(conn) -> dict:
    """Audit answer layer: ai_audit_answers -> audit_report."""
    total_rows = _scalar(conn, "SELECT COUNT(*) FROM ai_audit_answers") or 0
    fk_dangling = _scalar(conn, """
        SELECT COUNT(*) FROM ai_audit_answers aaa
        WHERE aaa.audit_report_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM audit_report ar
                          WHERE ar.id = aaa.audit_report_id)
    """) or 0
    total_fk = _scalar(conn, """
        SELECT COUNT(*) FROM ai_audit_answers WHERE audit_report_id IS NOT NULL
    """) or 0
    fk_resolve_ratio = 1.0 - (fk_dangling / total_fk) if total_fk else 0.0

    form_rows_resolved = _scalar(conn, """
        SELECT COUNT(*)
        FROM ai_audit_answers aaa
        JOIN audit_report ar ON ar.id = aaa.audit_report_id
    """) or 0

    obs_total = _scalar(conn, """
        SELECT COUNT(*) FROM audit_corrective_action
        WHERE observation_id IS NOT NULL
    """) or 0
    obs_resolved = _scalar(conn, """
        SELECT COUNT(*) FROM audit_corrective_action aca
        WHERE aca.observation_id IS NOT NULL
          AND EXISTS (SELECT 1 FROM ai_audit_answers aaa
                      WHERE aaa.submission_id = aca.observation_id::varchar)
    """) or 0
    obs_join_resolve_ratio = (obs_resolved / obs_total) if obs_total else 0.0

    return _verdict("audit", dict(
        total_rows=total_rows, total_fk=total_fk, fk_dangling=fk_dangling,
        fk_resolve_ratio=round(fk_resolve_ratio, 4),
        form_rows_resolved=form_rows_resolved,
        obs_total=obs_total, obs_resolved=obs_resolved,
        obs_join_resolve_ratio=round(obs_join_resolve_ratio, 4),
    ))


def _verdict(domain: str, metrics: dict) -> dict:
    """Turn raw metrics into ok / degraded / broken."""
    frr = metrics["fk_resolve_ratio"]
    resolved = metrics["form_rows_resolved"]
    floor = MIN_FORM_ROWS[domain]

    if resolved == 0 or frr < FK_BROKEN_RATIO:
        status = "broken"
        reason = (f"form_rows_resolved={resolved}, fk_resolve_ratio={frr:.2f} — "
                  f"answer-layer FK does not resolve to parent reports "
                  f"(one-time load is stale/incomplete in this snapshot)")
    elif resolved >= floor and frr >= FK_OK_RATIO:
        status = "ok"
        reason = f"form_rows_resolved={resolved} (>= {floor}), fk_resolve_ratio={frr:.2f}"
    else:
        status = "degraded"
        reason = (f"form_rows_resolved={resolved} (floor {floor}), "
                  f"fk_resolve_ratio={frr:.2f} — partial load, not blocking")
    return {"status": status, "metrics": metrics, "reason": reason}


# per-domain check registry
_CHECKS = {
    "inspection": [(DEP_AI_ANSWERS, _check_ai_answers)],
    "audit": [(DEP_AI_AUDIT_ANSWERS, _check_ai_audit_answers)],
}


def run_preflight(db_engine, domain: str) -> dict:
    """
    Run the answer-layer health checks for `domain`.
    Returns {"domain", "checks": {dep: {"status","metrics","reason"}}, "blocked": [dep,...]}.
    Never raises.
    """
    checks: dict[str, dict] = {}
    try:
        with db_engine.connect() as conn:
            for dep, fn in _CHECKS.get(domain, []):
                try:
                    checks[dep] = fn(conn)
                except Exception as exc:  # noqa: BLE001
                    checks[dep] = {"status": "broken", "metrics": {},
                                   "reason": f"preflight query error: {exc}"}
    except Exception as exc:  # noqa: BLE001 — connection failure, etc.
        for dep, _ in _CHECKS.get(domain, []):
            checks.setdefault(dep, {"status": "broken", "metrics": {},
                                    "reason": f"preflight connection error: {exc}"})

    blocked = [dep for dep, c in checks.items() if c["status"] == "broken"]
    return {"domain": domain, "checks": checks, "blocked": blocked}


def blocked_dependencies(report: dict) -> set[str]:
    return set(report.get("blocked", []))


def classify_row(dep: str | None, blocked: set[str],
                 real_pass: bool, truth_match) -> bool:
    """
    A row is `snapshot_blocked` when it depends on a broken answer layer AND it
    actually went wrong (empty / not real_pass, or a truth mismatch). A blocked
    question that still passes is left alone — it is not "blocked", it worked.
    """
    if not dep or dep not in blocked:
        return False
    return (not real_pass) or (truth_match is False)


def format_report(report: dict) -> str:
    """One block of text for the runner to print."""
    lines = [f"  DATA HEALTH PREFLIGHT — {report['domain']} domain"]
    for dep, c in report["checks"].items():
        mark = {"ok": "🟢", "degraded": "🟡", "broken": "🔴"}.get(c["status"], "??")
        lines.append(f"    {mark} {dep:<18} {c['status']:<9} {c['reason']}")
    if report["blocked"]:
        lines.append(f"    ⚠  BLOCKED: {', '.join(report['blocked'])} — "
                     f"dependent questions run but are excluded from headline rates")
    return "\n".join(lines)
