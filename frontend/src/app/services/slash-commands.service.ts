import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of, throwError } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { ChartConfig } from '../models/chat.model';

export interface SlashCommand {
  command:     string;
  args?:       string[];
  description: string;
  example:     string;
}

export const SLASH_COMMANDS: SlashCommand[] = [
  {
    command: '/chart',
    args: ['bar','horizontal','stacked','line','area','pie','doughnut','nightingale','scatter','radar','heatmap','calendar','waterfall','funnel','gauge'],
    description: 'Visualize the last result as a chart',
    example: '/chart line',
  },
  {
    command: '/sql',
    description: 'Show the SQL that generated the last result',
    example: '/sql',
  },
  {
    command: '/export',
    description: 'Download last result as CSV',
    example: '/export',
  },
  {
    command: '/clear',
    description: 'Clear this conversation',
    example: '/clear',
  },
  {
    command: '/help',
    description: 'Show available commands',
    example: '/help',
  },
];

const CHART_TYPE_MAP: Record<string, string> = {
  bar:         'bar',
  horizontal:  'horizontal_bar',
  hbar:        'horizontal_bar',
  stacked:     'stacked_bar',
  line:        'line',
  area:        'area',
  pie:         'pie',
  doughnut:    'doughnut',
  donut:       'doughnut',
  nightingale:  'nightingale',
  rose:         'nightingale',
  calendar:     'calendar_heatmap',
  scatter:     'scatter',
  radar:       'radar',
  heatmap:     'heatmap',
  waterfall:   'waterfall',
  funnel:      'funnel',
  gauge:       'gauge',
};

const CHART_DESCRIPTIONS: Record<string, string> = {
  bar:         'Bar — compare values across categories',
  horizontal:  'Horizontal bar — long category labels, ranked lists',
  stacked:     'Stacked bar — part-to-whole across groups',
  line:        'Line — trends over time or sequence',
  area:        'Area — trends with volume emphasis',
  pie:         'Pie — proportions of a whole (≤6 slices)',
  doughnut:    'Doughnut — proportions with centre callout',
  nightingale: 'Nightingale — circular bars, dramatic category comparison',
  scatter:     'Scatter — correlation between two numeric values',
  radar:       'Radar — multi-metric profile (spider chart)',
  heatmap:     'Heatmap — intensity across two dimensions',
  calendar:    'Calendar — date-wise activity grid (GitHub-style)',
  waterfall:   'Waterfall — cumulative running totals',
  funnel:      'Funnel — stages in a pipeline or process',
  gauge:       'Gauge — single KPI vs a target',
};

export type SlashCommandResult =
  | { type: 'chart';  chartType: string | null }
  | { type: 'sql' }
  | { type: 'export' }
  | { type: 'clear' }
  | { type: 'help' }
  | { type: 'unknown'; raw: string };

export interface ColumnSelection {
  xCol:  string;
  yCols: string[];
}

@Injectable({ providedIn: 'root' })
export class SlashCommandService {

  constructor(private http: HttpClient) {}

  isSlashCommand(input: string): boolean {
    return input.trim().startsWith('/');
  }

  parse(input: string): SlashCommandResult | null {
    const trimmed = input.trim().toLowerCase();
    if (!trimmed.startsWith('/')) return null;
    const [cmd, ...args] = trimmed.split(/\s+/);
    switch (cmd) {
      case '/chart': {
        const arg = args[0];
        const chartType = arg ? (CHART_TYPE_MAP[arg] ?? null) : null;
        return { type: 'chart', chartType };
      }
      case '/sql':    return { type: 'sql' };
      case '/export': return { type: 'export' };
      case '/clear':  return { type: 'clear' };
      case '/help':   return { type: 'help' };
      default:        return { type: 'unknown', raw: input };
    }
  }

  getSuggestions(input: string): Array<{ display: string; completion: string; description: string }> {
    const trimmed = input.trim().toLowerCase();
    if (!trimmed.startsWith('/')) return [];
    const parts = trimmed.split(/\s+/);
    const cmd   = parts[0];
    const arg   = parts[1] ?? '';
    if (parts.length === 1) {
      return SLASH_COMMANDS
        .filter(c => c.command.startsWith(cmd))
        .map(c => ({
          display:     c.command,
          completion:  c.command + ' ',
          description: c.description,
        }));
    }
    const found = SLASH_COMMANDS.find(c => c.command === cmd);
    if (!found?.args) return [];
    return found.args
      .filter(a => a.startsWith(arg))
      .map(a => ({
        display:     cmd + ' ' + a,
        completion:  cmd + ' ' + a,
        description: CHART_DESCRIPTIONS[a] ?? ('Chart type: ' + a),
      }));
  }

  fetchChart(sessionId: string, chartType: string | null = null, xCol?: string, yCols?: string[]): Observable<ChartConfig> {
    return this.http.post<ChartConfig>('/api/chart', {
      session_id: sessionId,
      chart_type: chartType ?? null,
      x_col:      xCol ?? null,
      y_cols:     yCols ?? null,
    });
  }

  fetchSql(sessionId: string): Observable<string> {
    return this.http.get<{sql: string}>('/api/sql/' + sessionId).pipe(
      map(r => r.sql)
    );
  }

  fetchColumns(sessionId: string): Observable<{columns: string[]; categories: string[]; numerics: string[]; dates: string[]}> {
    return this.http.get<any>('/api/columns/' + sessionId);
  }

  exportCsv(columns: string[], rows: Record<string, any>[], filename = 'octaai-export.csv'): void {
    const escape = (v: any): string => {
      const s = String(v ?? '');
      return s.includes(',') || s.includes('"') || s.includes('\n') ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    const header = columns.map(escape).join(',');
    const body   = rows.map(r => columns.map(c => escape(r[c])).join(',')).join('\n');
    const csv    = header + '\n' + body;
    const blob   = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url    = URL.createObjectURL(blob);
    const a      = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  }

  buildHelpText(): string {
    const lines = ['**Available commands:**', ''];
    for (const cmd of SLASH_COMMANDS) {
      lines.push('**' + cmd.command + '** — ' + cmd.description);
      if (cmd.args?.length) {
        lines.push('  Chart types:');
        cmd.args.forEach(a => {
          const desc = CHART_DESCRIPTIONS[a];
          lines.push('  • `' + cmd.command + ' ' + a + '`' + (desc ? ' — ' + desc.split(' — ')[1] : ''));
        });
      }
      lines.push('  Example: `' + cmd.example + '`');
      lines.push('');
    }
    return lines.join('\n');
  }
}
