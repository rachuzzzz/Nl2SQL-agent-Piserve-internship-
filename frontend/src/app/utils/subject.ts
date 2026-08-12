/**
 * Infers the subject noun ("corrective actions", "audit schedules", ...) a
 * question was actually about, from its own text — used to build a sensible
 * follow-up when the user drills into a table row, a dashboard KPI card, or
 * a chart segment, rather than defaulting every drilldown to "inspections".
 * Shared across result-table, dashboard-display, and chart-display, which
 * all need the exact same inference for their own click-to-drill features.
 */
const SUBJECT_PATTERNS: Array<[RegExp, string]> = [
  [/corrective\s*actions?/i,          'corrective actions'],
  [/mitigative\s*actions?/i,          'mitigative actions'],
  [/inspection\s*cycles?|\bcycles?\b/i, 'inspection cycles'],
  [/audit\s*schedules?/i,             'audit schedules'],
  [/audit\s*plans?/i,                 'audit plans'],
  [/inspection\s*schedules?|\bschedules?\b/i, 'inspection schedules'],
  [/form\s*answers?|question\s*answers?/i, 'form answers'],
  [/findings?/i,                      'findings'],
  [/observations?/i,                  'observations'],
];

/** Falls back to `fallback` (default 'inspections') when nothing matches. */
export function resolveSubject(question: string | undefined | null, fallback = 'inspections'): string {
  const q = question ?? '';
  for (const [re, label] of SUBJECT_PATTERNS) {
    if (re.test(q)) return label;
  }
  return fallback;
}
