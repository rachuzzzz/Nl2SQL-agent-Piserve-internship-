/* ============================================================
   OptimaX mock data — EHS / inspections / audits / permits
   Exposes window.OPTIMAX_DATA
   ============================================================ */
(function () {
  // ---------------- Schema ----------------
  const SCHEMA = [
    { name: 'inspection_report', rows: '48,210', desc: 'Site inspections & scores', cols: [
      ['id','uuid'],['site_id','int → site'],['inspector_id','int → inspector'],['inspection_score','numeric'],
      ['status','enum'],['submitted_on','timestamptz'],['active','bool'],['template','text'],['findings_count','int'] ] },
    { name: 'audit', rows: '9,884', desc: 'Compliance & internal audits', cols: [
      ['id','uuid'],['site_id','int → site'],['type','enum'],['result','enum'],['auditor_id','int'],
      ['opened_on','date'],['closed_on','date'],['nonconformities','int'] ] },
    { name: 'permit', rows: '12,640', desc: 'Work & environmental permits', cols: [
      ['id','uuid'],['site_id','int → site'],['permit_type','enum'],['status','enum'],
      ['issued_on','date'],['expires_on','date'],['holder','text'] ] },
    { name: 'finding', rows: '131,902', desc: 'Findings raised on inspections', cols: [
      ['id','uuid'],['report_id','uuid → inspection_report'],['severity','enum'],['category','text'],
      ['is_open','bool'],['raised_on','date'] ] },
    { name: 'corrective_action', rows: '54,771', desc: 'CAPA tracking', cols: [
      ['id','uuid'],['finding_id','uuid → finding'],['site_id','int'],['status','enum'],
      ['due_on','date'],['closed_on','date'],['owner','text'] ] },
    { name: 'incident', rows: '7,355', desc: 'Safety incidents & near-misses', cols: [
      ['id','uuid'],['site_id','int'],['severity','enum'],['type','text'],
      ['reported_on','timestamptz'],['lost_time','bool'] ] },
    { name: 'site', rows: '212', desc: 'Facilities / locations', cols: [
      ['id','int'],['name','text'],['region','text'],['risk_tier','enum'] ] },
    { name: 'inspector', rows: '486', desc: 'Field inspectors', cols: [
      ['id','int'],['name','text'],['certified','bool'],['region','text'] ] },
  ];

  // ---------------- Canned query examples ----------------
  // Each: id, q, sql, reasoning, summary, kpis[], table{cols, rows}, chart{type,series,x},
  //       debug{calls, tokens, prompt, completion, responseTime, llmCalls, sqlErrors, status, banner, trace[]}
  const teal = 'var(--viz-teal)', coral = 'var(--viz-coral)', blue = 'var(--viz-blue)',
        amber = 'var(--viz-amber)', violet = 'var(--viz-violet)';

  const EX = {
    avg_score: {
      id: 'avg_score',
      q: 'What is the average inspection score across months?',
      summary: 'Average inspection score peaked in January 2026 at 91.64 and dipped to 81.35 in April. Across the 5 months, 195 inspections were completed at a blended average of 85.52.',
      sql: `SELECT
  DATE_TRUNC('month', ir.submitted_on)            AS month_idx,
  TO_CHAR(DATE_TRUNC('month', ir.submitted_on),
          'FMMonth YYYY')                         AS month_label,
  ROUND(AVG(ir.inspection_score)::numeric, 2)     AS avg_score,
  COUNT(*)                                        AS inspection_count
FROM inspection_report ir
WHERE ir.active = true
  AND ir.status <> 'DRAFT'
  AND ir.inspection_score IS NOT NULL
GROUP BY month_idx, month_label
ORDER BY month_idx;`,
      reasoning: 'User wants the average inspection score grouped by month. This requires querying inspection_report, truncating submitted_on to month, averaging inspection_score, and excluding DRAFT rows. First attempt aliased a window function inside GROUP BY — validator flagged it and triggered a retry with DATE_TRUNC.',
      kpis: [
        { label: 'AVG SCORE', value: '85.52', tag: 'AVERAGE', icon: 'star', tone: 'blue' },
        { label: 'SCORE RANGE', value: '81.35–91.64', tag: 'MIN – MAX', icon: 'range', tone: 'amber' },
        { label: 'INSPECTIONS', value: '195', tag: 'TOTAL', icon: 'rows', tone: 'teal' },
      ],
      chart: { type: 'line', best: 'line', x: ['Jan','Feb','Mar','Apr','May'],
        series: [
          { name: 'Avg Score', color: teal, data: [91.64, 81.75, 87.31, 81.35, 85.55] },
          { name: 'Inspection Count', color: coral, data: [59, 42, 31, 28, 35] },
        ] },
      table: { cols: ['MONTH IDX','MONTH LABEL','AVG SCORE','COUNT'], rows: [
        ['2026-01-01 00:00:00+00','January 2026','91.64','59'],
        ['2026-02-01 00:00:00+00','February 2026','81.75','42'],
        ['2026-03-01 00:00:00+00','March 2026','87.31','31'],
        ['2026-04-01 00:00:00+00','April 2026','81.35','28'],
        ['2026-05-01 00:00:00+00','May 2026','85.55','35'],
      ], total: 5 },
      debug: { calls: 2, tokens: 30519, prompt: 29932, completion: 587, responseTime: '10.17s',
        llmCalls: 2, sqlErrors: 1, status: 'warn',
        banner: { tone: 'warn', text: '1 SQL error — validator triggered retry' },
        trace: [
          { step: 'Parse intent', model: 'qwen3.5:9b', ms: 410, status: 'ok' },
          { step: 'Resolve schema', model: 'retriever', ms: 230, status: 'ok', note: 'matched inspection_report (0.94)' },
          { step: 'Generate SQL', model: 'qwen2.5-coder', ms: 3120, status: 'ok' },
          { step: 'Validate SQL', model: 'validator', ms: 90, status: 'err', note: 'aggregate alias in GROUP BY' },
          { step: 'Repair + retry', model: 'qwen2.5-coder', ms: 4880, status: 'ok' },
          { step: 'Execute', model: 'postgres', ms: 740, status: 'ok', note: '5 rows · 28ms' },
          { step: 'Summarize', model: 'qwen3.5:9b', ms: 700, status: 'ok' },
        ] },
    },

    open_capa: {
      id: 'open_capa',
      q: 'Show open corrective actions by site',
      summary: 'Houston Terminal carries the heaviest backlog with 38 open corrective actions, followed by Rotterdam (29). 7 sites account for 142 of 168 open CAPAs.',
      sql: `SELECT
  s.name                       AS site,
  s.region                     AS region,
  COUNT(*)                     AS open_actions,
  COUNT(*) FILTER (
    WHERE ca.due_on < CURRENT_DATE
  )                            AS overdue
FROM corrective_action ca
JOIN site s ON s.id = ca.site_id
WHERE ca.status = 'OPEN'
GROUP BY s.name, s.region
ORDER BY open_actions DESC
LIMIT 7;`,
      reasoning: 'User wants a count of open corrective actions grouped by site, with overdue surfaced. Joined corrective_action to site, filtered status = OPEN, and added a FILTER clause to compute overdue in the same pass. Clean on first attempt.',
      kpis: [
        { label: 'OPEN CAPAS', value: '168', tag: 'TOTAL', icon: 'rows', tone: 'blue' },
        { label: 'OVERDUE', value: '41', tag: '24.4%', icon: 'warn', tone: 'coral' },
        { label: 'SITES', value: '7', tag: 'TOP', icon: 'site', tone: 'teal' },
      ],
      chart: { type: 'bar', best: 'bar', x: ['Houston','Rotterdam','Singapore','Antwerp','Jubail','Texas City','Pernis'],
        series: [
          { name: 'Open', color: blue, data: [38, 29, 22, 18, 14, 12, 9] },
          { name: 'Overdue', color: coral, data: [11, 8, 5, 6, 3, 4, 4] },
        ] },
      table: { cols: ['SITE','REGION','OPEN','OVERDUE'], rows: [
        ['Houston Terminal','US Gulf','38','11'],
        ['Rotterdam Port','NW Europe','29','8'],
        ['Singapore Jurong','APAC','22','5'],
        ['Antwerp Dock 4','NW Europe','18','6'],
        ['Jubail Plant','MEA','14','3'],
        ['Texas City','US Gulf','12','4'],
        ['Pernis Refinery','NW Europe','9','4'],
      ], total: 7 },
      debug: { calls: 1, tokens: 18204, prompt: 17790, completion: 414, responseTime: '4.62s',
        llmCalls: 1, sqlErrors: 0, status: 'ok',
        banner: { tone: 'ok', text: 'Executed cleanly on first attempt' },
        trace: [
          { step: 'Parse intent', model: 'qwen3.5:9b', ms: 380, status: 'ok' },
          { step: 'Resolve schema', model: 'retriever', ms: 210, status: 'ok', note: 'corrective_action ⨝ site' },
          { step: 'Generate SQL', model: 'qwen2.5-coder', ms: 2980, status: 'ok' },
          { step: 'Validate SQL', model: 'validator', ms: 80, status: 'ok' },
          { step: 'Execute', model: 'postgres', ms: 410, status: 'ok', note: '7 rows · 19ms' },
          { step: 'Summarize', model: 'qwen3.5:9b', ms: 560, status: 'ok' },
        ] },
    },

    permits: {
      id: 'permits',
      q: 'How many permits expire in the next 30 days?',
      summary: '23 permits expire within 30 days — 6 of them are high-risk environmental permits. Jubail Plant has the most exposure with 5 expiring permits.',
      sql: `SELECT
  p.permit_type,
  s.name                       AS site,
  p.expires_on,
  (p.expires_on - CURRENT_DATE) AS days_left
FROM permit p
JOIN site s ON s.id = p.site_id
WHERE p.status = 'ACTIVE'
  AND p.expires_on BETWEEN CURRENT_DATE
                       AND CURRENT_DATE + INTERVAL '30 days'
ORDER BY p.expires_on ASC;`,
      reasoning: 'User asks for permits expiring within a 30-day window. Filtered permit.status = ACTIVE and a BETWEEN range on expires_on, joined site for context, and computed days_left as a date difference. Single pass, no repair needed.',
      kpis: [
        { label: 'EXPIRING ≤30d', value: '23', tag: 'PERMITS', icon: 'rows', tone: 'blue' },
        { label: 'HIGH RISK', value: '6', tag: 'ENV', icon: 'warn', tone: 'coral' },
        { label: 'NEXT EXPIRY', value: '4 days', tag: 'SOONEST', icon: 'range', tone: 'amber' },
      ],
      chart: { type: 'bar', best: 'bar', x: ['Jubail','Houston','Antwerp','Singapore','Rotterdam','Pernis'],
        series: [ { name: 'Expiring', color: amber, data: [5, 4, 4, 4, 3, 3] } ] },
      table: { cols: ['PERMIT TYPE','SITE','EXPIRES ON','DAYS LEFT'], rows: [
        ['Air Emissions','Jubail Plant','2026-07-01','8'],
        ['Hot Work','Houston Terminal','2026-06-27','4'],
        ['Wastewater Disch.','Antwerp Dock 4','2026-07-03','10'],
        ['Confined Space','Singapore Jurong','2026-07-09','16'],
        ['Air Emissions','Rotterdam Port','2026-07-14','21'],
        ['Excavation','Pernis Refinery','2026-07-19','26'],
      ], total: 23 },
      debug: { calls: 1, tokens: 15910, prompt: 15602, completion: 308, responseTime: '3.91s',
        llmCalls: 1, sqlErrors: 0, status: 'ok',
        banner: { tone: 'ok', text: 'Executed cleanly on first attempt' },
        trace: [
          { step: 'Parse intent', model: 'qwen3.5:9b', ms: 350, status: 'ok' },
          { step: 'Resolve schema', model: 'retriever', ms: 190, status: 'ok', note: 'permit ⨝ site' },
          { step: 'Generate SQL', model: 'qwen2.5-coder', ms: 2410, status: 'ok' },
          { step: 'Validate SQL', model: 'validator', ms: 70, status: 'ok' },
          { step: 'Execute', model: 'postgres', ms: 320, status: 'ok', note: '23 rows · 14ms' },
          { step: 'Summarize', model: 'qwen3.5:9b', ms: 480, status: 'ok' },
        ] },
    },

    incidents: {
      id: 'incidents',
      q: 'Trend of safety incidents by severity this year',
      summary: 'Incidents are trending down — from 41 in January to 24 in May. Minor incidents drive the volume while serious incidents stay in the low single digits, with zero lost-time incidents since March.',
      sql: `SELECT
  TO_CHAR(DATE_TRUNC('month', i.reported_on), 'Mon') AS month,
  i.severity,
  COUNT(*) AS incidents
FROM incident i
WHERE i.reported_on >= DATE_TRUNC('year', CURRENT_DATE)
GROUP BY DATE_TRUNC('month', i.reported_on), i.severity
ORDER BY DATE_TRUNC('month', i.reported_on);`,
      reasoning: 'User wants a time trend of incidents split by severity for the current year. Truncated reported_on to month, grouped by month and severity, and filtered to year-to-date. Pivoted client-side into stacked series for charting.',
      kpis: [
        { label: 'INCIDENTS YTD', value: '163', tag: 'TOTAL', icon: 'rows', tone: 'blue' },
        { label: 'LOST-TIME', value: '2', tag: 'YTD', icon: 'warn', tone: 'coral' },
        { label: 'TREND', value: '−41%', tag: 'JAN→MAY', icon: 'range', tone: 'teal' },
      ],
      chart: { type: 'area', best: 'area', x: ['Jan','Feb','Mar','Apr','May'],
        series: [
          { name: 'Minor', color: blue, data: [31, 28, 25, 22, 18] },
          { name: 'Serious', color: amber, data: [8, 6, 5, 5, 4] },
          { name: 'Critical', color: coral, data: [2, 1, 1, 2, 2] },
        ] },
      table: { cols: ['MONTH','MINOR','SERIOUS','CRITICAL'], rows: [
        ['January','31','8','2'],
        ['February','28','6','1'],
        ['March','25','5','1'],
        ['April','22','5','2'],
        ['May','18','4','2'],
      ], total: 5 },
      debug: { calls: 1, tokens: 16740, prompt: 16395, completion: 345, responseTime: '4.18s',
        llmCalls: 1, sqlErrors: 0, status: 'ok',
        banner: { tone: 'ok', text: 'Executed cleanly on first attempt' },
        trace: [
          { step: 'Parse intent', model: 'qwen3.5:9b', ms: 360, status: 'ok' },
          { step: 'Resolve schema', model: 'retriever', ms: 200, status: 'ok', note: 'matched incident (0.91)' },
          { step: 'Generate SQL', model: 'qwen2.5-coder', ms: 2620, status: 'ok' },
          { step: 'Validate SQL', model: 'validator', ms: 80, status: 'ok' },
          { step: 'Execute', model: 'postgres', ms: 360, status: 'ok', note: '15 rows · 17ms' },
          { step: 'Summarize', model: 'qwen3.5:9b', ms: 560, status: 'ok' },
        ] },
    },

    top_inspectors: {
      id: 'top_inspectors',
      q: 'Top 5 inspectors by audits completed',
      summary: 'M. Okafor leads with 218 completed audits and the highest pass rate (96%). The top 5 inspectors closed 894 audits between them this period.',
      sql: `SELECT
  ins.name                                     AS inspector,
  ins.region,
  COUNT(a.id)                                  AS audits_done,
  ROUND(100.0 * COUNT(*) FILTER (
        WHERE a.result = 'PASS') / COUNT(*), 0) AS pass_rate
FROM audit a
JOIN inspector ins ON ins.id = a.auditor_id
WHERE a.closed_on IS NOT NULL
GROUP BY ins.name, ins.region
ORDER BY audits_done DESC
LIMIT 5;`,
      reasoning: 'User wants the top 5 inspectors ranked by completed audits. Joined audit to inspector, filtered to closed audits, counted per inspector, and added a pass-rate using a FILTER aggregate. Ordered descending with LIMIT 5.',
      kpis: [
        { label: 'AUDITS (TOP 5)', value: '894', tag: 'TOTAL', icon: 'rows', tone: 'blue' },
        { label: 'BEST PASS RATE', value: '96%', tag: 'M. OKAFOR', icon: 'star', tone: 'teal' },
        { label: 'AVG / INSPECTOR', value: '179', tag: 'AUDITS', icon: 'range', tone: 'amber' },
      ],
      chart: { type: 'bar', best: 'bar', x: ['Okafor','Tan','Khan','Silva','Novak'],
        series: [ { name: 'Audits done', color: violet, data: [218, 196, 174, 161, 145] } ] },
      table: { cols: ['INSPECTOR','REGION','AUDITS','PASS %'], rows: [
        ['M. Okafor','MEA','218','96'],
        ['L. Tan','APAC','196','94'],
        ['A. Khan','MEA','174','91'],
        ['R. Silva','LATAM','161','93'],
        ['P. Novak','NW Europe','145','89'],
      ], total: 5 },
      debug: { calls: 1, tokens: 17120, prompt: 16760, completion: 360, responseTime: '4.05s',
        llmCalls: 1, sqlErrors: 0, status: 'ok',
        banner: { tone: 'ok', text: 'Executed cleanly on first attempt' },
        trace: [
          { step: 'Parse intent', model: 'qwen3.5:9b', ms: 370, status: 'ok' },
          { step: 'Resolve schema', model: 'retriever', ms: 200, status: 'ok', note: 'audit ⨝ inspector' },
          { step: 'Generate SQL', model: 'qwen2.5-coder', ms: 2550, status: 'ok' },
          { step: 'Validate SQL', model: 'validator', ms: 80, status: 'ok' },
          { step: 'Execute', model: 'postgres', ms: 330, status: 'ok', note: '5 rows · 12ms' },
          { step: 'Summarize', model: 'qwen3.5:9b', ms: 540, status: 'ok' },
        ] },
    },
  };

  const SUGGESTIONS = [
    { q: EX.avg_score.q, ex: 'avg_score', icon: 'chart' },
    { q: EX.open_capa.q, ex: 'open_capa', icon: 'warn' },
    { q: EX.permits.q, ex: 'permits', icon: 'permit' },
    { q: EX.incidents.q, ex: 'incidents', icon: 'incident' },
    { q: EX.top_inspectors.q, ex: 'top_inspectors', icon: 'star' },
  ];

  // ---------------- Conversations ----------------
  const CONVERSATIONS = {
    today: [
      { id: 'c1', title: 'Average inspection score across months', ex: 'avg_score' },
      { id: 'c2', title: 'Open corrective actions by site', ex: 'open_capa' },
    ],
    earlier: [
      { id: 'c3', title: 'Permits expiring in 30 days', ex: 'permits' },
      { id: 'c4', title: 'Safety incident trend by severity', ex: 'incidents' },
      { id: 'c5', title: 'Top inspectors by audits', ex: 'top_inspectors' },
    ],
  };

  // ---------------- Saved queries ----------------
  const SAVED = [
    { name: 'Monthly inspection score', ex: 'avg_score', tag: 'Inspections', runs: 142, chart: 'line', color: teal },
    { name: 'Open CAPAs by site', ex: 'open_capa', tag: 'CAPA', runs: 98, chart: 'bar', color: blue },
    { name: 'Permits expiring ≤30d', ex: 'permits', tag: 'Permits', runs: 211, chart: 'bar', color: amber },
    { name: 'Incident severity trend', ex: 'incidents', tag: 'Safety', runs: 76, chart: 'area', color: coral },
    { name: 'Top inspectors', ex: 'top_inspectors', tag: 'Audits', runs: 54, chart: 'bar', color: violet },
    { name: 'Audit pass rate by region', ex: 'top_inspectors', tag: 'Audits', runs: 33, chart: 'bar', color: blue },
  ];

  // ---------------- Dashboard tiles ----------------
  const DASH_KPIS = [
    { label: 'INSPECTIONS THIS MONTH', value: '1,284', delta: '+8.2%', up: true, icon: 'rows' },
    { label: 'OPEN CORRECTIVE ACTIONS', value: '168', delta: '−12', up: true, icon: 'warn' },
    { label: 'PERMITS EXPIRING ≤30d', value: '23', delta: '+5', up: false, icon: 'permit' },
    { label: 'LOST-TIME INCIDENTS YTD', value: '2', delta: '0', up: true, icon: 'incident' },
  ];

  window.OPTIMAX_DATA = { SCHEMA, EX, SUGGESTIONS, CONVERSATIONS, SAVED, DASH_KPIS };
})();
