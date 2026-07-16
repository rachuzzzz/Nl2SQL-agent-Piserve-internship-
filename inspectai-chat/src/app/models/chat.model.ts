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
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export interface DashboardCard {
  label:   string;
  value:   number;
  value2?: number;          // max value for 'range' cards
  kind:    'avg' | 'sum' | 'range' | 'distinct';
  icon:    'star' | 'list' | 'range' | 'hash';
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