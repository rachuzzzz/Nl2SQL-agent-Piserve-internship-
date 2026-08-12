// ── Step info (from debug panel) ─────────────────────────────────────────────

export interface StepInfo {
  tool:            string;
  thought:         string;
  sql?:            string | null;
  result_preview?: string | null;
  success?:        boolean;
}

// ── Result data (table pagination) ───────────────────────────────────────────

export interface DisambiguationMeta {
  kind:        string;  // "person" | "facility" | "project" | "client" | "organisation"
  search_term: string;  // literal text from the STEP 1 ILIKE clause, e.g. "adnoc"
}

export interface ResultData {
  rows:            Record<string, any>[];
  columns:         string[];
  row_count:       number;
  total_count?:    number | null;
  page?:           number;
  page_size?:      number;
  has_more?:       boolean;
  sql:             string | null;
  disambiguation?: DisambiguationMeta | null;
  // True only when this turn's rows are resolvable server-side against a
  // checkbox selection (see resolve_selected_rows) — false for stats-style
  // tables and always false on anything fetched via /api/page.
  row_selectable?: boolean;
}

// ── Row selection (checkbox pick -> scoped follow-up) ──────────────────────

export interface SelectedRow {
  position: number;  // 1-based, matches backend ChatRequest.selected_positions
  label:    string;  // first-column display value, for the chip UI
}

// ── Chart types ───────────────────────────────────────────────────────────────

export interface ChartAlt {
  chart_type: string;
  label:      string;
  config:     any;
}

export interface ChartConfig {
  chart_type:       string | null;
  reason:           string;
  confidence:       string;
  config:           any | null;
  conflict:         boolean;
  requested_type:   string | null;
  requested_config: any | null;
  alternatives:     ChartAlt[];
  x_label?:         string | null;
  y_labels?:        string[];
  // Top-N truncation metadata — see ChartConfig in backend/src/api/schemas.py.
  // For aggregated chart types these count CATEGORIES (post-aggregation), not
  // raw rows. rows_shown == rows_total when nothing was truncated.
  rows_shown?:      number | null;
  rows_total?:      number | null;
  is_aggregated?:   boolean;
  can_view_all?:    boolean;
}

export interface ColumnMeta {
  columns:    string[];
  categories: string[];
  numerics:   string[];
  dates:      string[];
  session_id: string;
}

export interface ChartRequest {
  session_id:  string;
  chart_type?: string | null;
  x_col?:      string | null;
  y_cols?:     string[] | null;
  view_all?:   boolean;
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export interface DashboardCard {
  label:   string;
  value:   number;
  value2?: number;          // max value for 'range' cards
  kind:    'avg' | 'sum' | 'range' | 'distinct' | 'status';
  icon:    'star' | 'list' | 'range' | 'hash' | 'status';
  // 'status'-kind cards only (see backend/src/charts/advisor/kpi.py's
  // _status_breakdown_cards) — one card per status VALUE (e.g. "Overdue: 5")
  // instead of a single generic sum, for a status-breakdown-shaped result.
  severity?:     'critical' | 'warning' | 'ok' | null;  // drives card coloring
  filter_value?: string;    // raw status value, for click-to-drill filtering
}

export interface DashboardData {
  show:  boolean;
  cards: DashboardCard[];
}

// ── Chart signal (parsed from __CHART_AVAILABLE__ marker) ────────────────────

export interface ChartSignal {
  recommendedType: string;
}

// ── Chat response (from /api/chat) ────────────────────────────────────────────

export interface ChatResponse {
  answer:            string;
  sql:               string | null;
  reasoning:         string | null;
  // Which domain answered this turn — "inspection" | "audit" | null
  // (ambiguous / no orchestrator ran for this reply).
  domain?:           string | null;
  // Short "here's what I understood" recap shown above the answer — see
  // backend/src/api/intent_recap.py. Null when a recap doesn't make sense
  // for this turn (a /chart-followup reply, a domain-ambiguity reply).
  intent_recap?:     string | null;
  prompt_tokens:     number;
  completion_tokens: number;
  total_tokens:      number;
  time_ms:           number;
  llm_calls:         number;
  sql_errors:        number;
  steps:             StepInfo[];
  session_id:        string;
  result_data?:      ResultData | null;
  chart_config?:     ChartConfig | null;
  dashboard?:        DashboardData | null;
}

// ── Message ───────────────────────────────────────────────────────────────────

export interface Message {
  id:              string;
  role:            'user' | 'ai';
  text:            string;
  timestamp:       Date;
  debug?:          ChatResponse | null;
  domain?:         string | null;   // "inspection" | "audit" | null — which domain answered this turn
  intentRecap?:    string | null;   // "here's what I understood" recap shown above the bubble
  resultData?:     ResultData | null;
  chartConfig?:    ChartConfig | null;
  chartSignal?:    ChartSignal | null;   // parsed from __CHART_AVAILABLE__ signal
  chartMode?:      'quick' | 'control';  // 'quick' = plain render, 'control' = switcher+axis
  dashboard?:      DashboardData | null;
  sourceQuestion?: string;  // the user question that produced this AI message's resultData —
                             // lets a drilldown click on the resulting table build a follow-up
                             // that preserves what the user actually asked about (e.g. "mitigative
                             // actions") instead of always assuming "inspections"
}

// ── Conversation ──────────────────────────────────────────────────────────────

export interface Conversation {
  id:        string;
  title:     string;
  createdAt: Date;
  messages:  Message[];
}

// ── Helper: parse __CHART_AVAILABLE__ signal out of answer text ───────────────

const CHART_SIGNAL_RE = /\n*📊__CHART_AVAILABLE__:(\w+):[^\n]*/;

export function parseChartSignal(text: string): {
  cleanText:   string;
  chartSignal: ChartSignal | null;
} {
  const m = text.match(CHART_SIGNAL_RE);
  if (!m) return { cleanText: text, chartSignal: null };
  return {
    cleanText:   text.replace(CHART_SIGNAL_RE, '').trimEnd(),
    chartSignal: { recommendedType: m[1] },
  };
}