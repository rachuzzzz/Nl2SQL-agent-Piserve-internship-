import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { ChatService } from '../../services/chat.service';
import { ChatResponse } from '../../models/chat.model';

const SQL_KW = /\b(SELECT|FROM|WHERE|AND|OR|NOT|NULL|IS|IN|AS|JOIN|LEFT|RIGHT|INNER|FULL|OUTER|ON|GROUP|ORDER|BY|LIMIT|HAVING|DISTINCT|CASE|WHEN|THEN|ELSE|END|WITH|UNION|ALL|TRUE|FALSE|ASC|DESC|BETWEEN|FILTER|INTERVAL|CURRENT_DATE|DATE_TRUNC|TO_CHAR|ROUND|COUNT|AVG|SUM|MIN|MAX|COALESCE|NULLIF|EXTRACT)\b/g;

function highlightSql(raw: string): string {
  const esc = raw
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  return esc
    .split('\n')
    .map(line => {
      let out = line
        .replace(/'([^']*)'/g, `<span class="sql-s">'$1'</span>`)
        .replace(/--.*/g, m => `<span class="sql-c">${m}</span>`)
        .replace(/\b(\d+(?:\.\d+)?)\b/g, `<span class="sql-n">$1</span>`);
      SQL_KW.lastIndex = 0;
      out = out.replace(SQL_KW, m => `<span class="sql-k">${m}</span>`);
      return `<div class="sql-line">${out || '&nbsp;'}</div>`;
    })
    .join('');
}

function fmtTok(n: number): string {
  return n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k' : String(n);
}

@Component({
  selector: 'app-debug-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    <aside class="dp" [class.dp-collapsed]="collapsed">

      <!-- ── Header ── -->
      <div class="dh">
        <button class="dh-toggle" (click)="collapsed = !collapsed"
                [title]="collapsed ? 'Expand debug panel' : 'Collapse debug panel'">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
               width="14" height="14" stroke-linecap="round" stroke-linejoin="round"
               [style.transform]="collapsed ? 'rotate(180deg)' : 'none'"
               style="transition:transform .22s">
            <polyline points="9 18 15 12 9 6"/>
          </svg>
        </button>

        <ng-container *ngIf="!collapsed">
          <span class="status-ring">
            <span class="status-dot" [class.dot-warn]="hasErrors"></span>
            <span class="status-pulse" [class.dot-warn]="hasErrors"></span>
          </span>
          <span class="dh-label">DEBUG</span>
          <span class="dh-stats" *ngIf="d">
            {{ d!.llm_calls }} call{{ d!.llm_calls !== 1 ? 's' : '' }}
            &middot; {{ fmtTok(d!.total_tokens) }} tok
          </span>
        </ng-container>
      </div>

      <!-- ── Body ── -->
      <div class="db" *ngIf="!collapsed">

        <!-- Empty -->
        <div class="empty" *ngIf="!d">
          <div class="empty-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"
                 width="26" height="26" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="16 18 22 12 16 6"/>
              <polyline points="8 6 2 12 8 18"/>
            </svg>
          </div>
          <p class="empty-text">Send a query to see pipeline trace, SQL, and metrics.</p>
        </div>

        <ng-container *ngIf="d">

          <!-- Banner -->
          <div class="banner" [class.banner-warn]="hasErrors" [class.banner-ok]="!hasErrors">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                 width="14" height="14" stroke-linecap="round" stroke-linejoin="round">
              <polyline *ngIf="!hasErrors" points="20 6 9 17 4 12"/>
              <path *ngIf="hasErrors" d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01"/>
            </svg>
            <span>{{ hasErrors
              ? (d!.sql_errors + ' SQL error' + (d!.sql_errors !== 1 ? 's' : '') + ' — validator triggered retry')
              : 'Executed cleanly on first attempt' }}</span>
          </div>

          <!-- ── Pipeline Trace ── -->
          <div class="section" *ngIf="d!.steps.length">
            <button class="sec-hdr" (click)="open.steps = !open.steps">
              <span class="sdot" style="background:var(--accent)"></span>
              <span class="sec-title">Pipeline Trace</span>
              <span class="sec-right">
                <span class="sec-meta">{{ d!.steps.length }} step{{ d!.steps.length !== 1 ? 's' : '' }}</span>
                <svg class="chev" [class.chev-open]="open.steps" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" stroke-width="2.5" width="13" height="13"
                     stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="6 9 12 15 18 9"/>
                </svg>
              </span>
            </button>
            <div class="sec-body" *ngIf="open.steps">
              <div class="trace">
                <div *ngFor="let step of d!.steps; let i = index; let last = last" class="trace-row">
                  <div class="trace-spine">
                    <span class="trace-node" [class.node-ok]="step.success !== false"
                          [class.node-err]="step.success === false">
                      <svg *ngIf="step.success !== false" viewBox="0 0 24 24" fill="none"
                           stroke="#fff" stroke-width="3" width="8" height="8"
                           stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"/>
                      </svg>
                      <span *ngIf="step.success === false" class="node-x"></span>
                    </span>
                    <span class="trace-line" *ngIf="!last"></span>
                  </div>
                  <div class="trace-info">
                    <div class="trace-tool">{{ step.tool }}()</div>
                    <div class="trace-thought" *ngIf="step.thought">
                      {{ step.thought.length > 90 ? (step.thought | slice:0:90) + '…' : step.thought }}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- ── Generated SQL ── -->
          <div class="section" *ngIf="d!.sql">
            <button class="sec-hdr" (click)="open.sql = !open.sql">
              <span class="sdot" style="background:var(--viz-teal)"></span>
              <span class="sec-title">Generated SQL</span>
              <svg class="chev" [class.chev-open]="open.sql" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2.5" width="13" height="13"
                   stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"/>
              </svg>
            </button>
            <div class="sec-body" *ngIf="open.sql">
              <div class="sql-block">
                <button class="copy-btn" (click)="copySql()" [class.copy-done]="copied">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                       width="11" height="11" stroke-linecap="round" stroke-linejoin="round">
                    <polyline *ngIf="copied" points="20 6 9 17 4 12"/>
                    <ng-container *ngIf="!copied">
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                      <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                    </ng-container>
                  </svg>
                  {{ copied ? 'Copied' : 'Copy' }}
                </button>
                <pre class="sql-pre mono" [innerHTML]="sqlHtml"></pre>
              </div>
            </div>
          </div>

          <!-- ── Model Reasoning ── -->
          <div class="section" *ngIf="d!.reasoning">
            <button class="sec-hdr" (click)="open.reasoning = !open.reasoning">
              <span class="sdot" style="background:var(--viz-violet)"></span>
              <span class="sec-title">Model Reasoning</span>
              <svg class="chev" [class.chev-open]="open.reasoning" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2.5" width="13" height="13"
                   stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"/>
              </svg>
            </button>
            <div class="sec-body" *ngIf="open.reasoning">
              <div class="reasoning">"{{ d!.reasoning }}"</div>
            </div>
          </div>

          <!-- ── Performance ── -->
          <div class="section">
            <button class="sec-hdr" (click)="open.metrics = !open.metrics">
              <span class="sdot" style="background:var(--ok)"></span>
              <span class="sec-title">Performance</span>
              <svg class="chev" [class.chev-open]="open.metrics" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2.5" width="13" height="13"
                   stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"/>
              </svg>
            </button>
            <div class="sec-body" *ngIf="open.metrics">
              <div class="metrics-grid">
                <div class="metric">
                  <div class="metric-lbl">RESPONSE TIME</div>
                  <div class="metric-val accent">{{ ((d!.time_ms || 0) / 1000).toFixed(2) }}s</div>
                </div>
                <div class="metric">
                  <div class="metric-lbl">LLM CALLS</div>
                  <div class="metric-val">{{ d!.llm_calls }}</div>
                </div>
                <div class="metric">
                  <div class="metric-lbl">PROMPT TOKENS</div>
                  <div class="metric-val">{{ d!.prompt_tokens.toLocaleString() }}</div>
                </div>
                <div class="metric">
                  <div class="metric-lbl">COMPLETION</div>
                  <div class="metric-val">{{ d!.completion_tokens.toLocaleString() }}</div>
                </div>
                <div class="metric">
                  <div class="metric-lbl">SQL ERRORS</div>
                  <div class="metric-val" [class.err]="(d!.sql_errors || 0) > 0">
                    {{ d!.sql_errors || 0 }}
                  </div>
                </div>
                <div class="metric">
                  <div class="metric-lbl">TOTAL TOKENS</div>
                  <div class="metric-val">{{ d!.total_tokens.toLocaleString() }}</div>
                </div>
              </div>
            </div>
          </div>

          <!-- ── Session Details ── -->
          <div class="section">
            <button class="sec-hdr" (click)="open.session = !open.session">
              <span class="sdot" style="background:var(--text-3)"></span>
              <span class="sec-title">Session Details</span>
              <svg class="chev" [class.chev-open]="open.session" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2.5" width="13" height="13"
                   stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"/>
              </svg>
            </button>
            <div class="sec-body" *ngIf="open.session">
              <div class="kv-table">
                <div class="kv-row">
                  <span class="kv-k">Session ID</span>
                  <span class="kv-v mono">{{ d!.session_id | slice:0:16 }}…</span>
                </div>
                <div class="kv-row alt">
                  <span class="kv-k">LLM Calls</span>
                  <span class="kv-v mono">{{ d!.llm_calls }}</span>
                </div>
                <div class="kv-row">
                  <span class="kv-k">SQL Errors</span>
                  <span class="kv-v mono" [class.kv-err]="(d!.sql_errors || 0) > 0">{{ d!.sql_errors || 0 }}</span>
                </div>
                <div class="kv-row alt">
                  <span class="kv-k">Total Tokens</span>
                  <span class="kv-v mono">{{ d!.total_tokens.toLocaleString() }}</span>
                </div>
                <div class="kv-row">
                  <span class="kv-k">Steps</span>
                  <span class="kv-v mono">{{ d!.steps.length || 0 }}</span>
                </div>
              </div>
              <div class="live-ind">
                <span class="live-ring">
                  <span class="live-dot-c"></span>
                  <span class="live-pulse-c"></span>
                </span>
                Live · pipeline healthy
              </div>
            </div>
          </div>

        </ng-container>
      </div>
    </aside>
  `,
  styles: [`
    :host { display: contents; }

    .dp {
      width: 362px; min-width: 362px;
      display: flex; flex-direction: column; overflow: hidden;
      background: var(--surface); border-left: 1px solid var(--border);
      transition: width 0.25s cubic-bezier(0.4,0,0.2,1),
                  min-width 0.25s cubic-bezier(0.4,0,0.2,1);
    }
    .dp.dp-collapsed { width: 48px; min-width: 48px; }

    /* Header */
    .dh {
      display: flex; align-items: center; gap: 9px;
      padding: 0 14px; height: 48px; flex-shrink: 0;
      border-bottom: 1px solid var(--border);
    }
    .dh-toggle {
      width: 28px; height: 28px; display: grid; place-items: center;
      background: var(--surface-2); border: 1px solid var(--border);
      border-radius: 10px; cursor: pointer; color: var(--text-2);
      flex-shrink: 0; transition: background 0.14s, color 0.14s, box-shadow 0.14s;
    }
    .dh-toggle:hover { background: var(--surface-3); color: var(--text); box-shadow: var(--shadow-sm); }
    .status-ring { position: relative; display: inline-flex; flex-shrink: 0; }
    .status-dot {
      width: 9px; height: 9px; border-radius: 50%; background: var(--ok);
    }
    .status-dot.dot-warn { background: var(--warn); }
    .status-pulse {
      position: absolute; inset: 0; border-radius: 50%; background: var(--ok);
      animation: dp-pulse 1.6s ease-in-out infinite;
    }
    .status-pulse.dot-warn { background: var(--warn); }
    @keyframes dp-pulse {
      0%,100% { opacity:0.5; transform:scale(1); }
      50%      { opacity:0;   transform:scale(2.2); }
    }

    .dh-label {
      font-size: 11.5px; font-weight: 700; letter-spacing: 0.14em;
      color: var(--text); font-family: var(--font-mono);
    }
    .dh-stats {
      margin-left: auto; font-size: 11px; color: var(--text-3);
      font-family: var(--font-mono); white-space: nowrap;
    }

    /* Body */
    .db { flex: 1; overflow-y: auto; }

    /* Empty */
    .empty {
      display: flex; flex-direction: column; align-items: center;
      padding: 40px 20px; gap: 12px; text-align: center;
    }
    .empty-icon {
      width: 52px; height: 52px; border-radius: 50%;
      background: linear-gradient(145deg, var(--blue-50), var(--blue-100));
      border: 1px solid var(--blue-200);
      box-shadow: 0 0 0 6px rgba(27,67,255,0.06);
      display: grid; place-items: center; color: var(--accent);
    }
    .empty-text {
      font-size: 12.5px; color: var(--text-3); line-height: 1.55; margin: 0;
      max-width: 200px;
    }

    /* Banner */
    .banner {
      display: flex; align-items: center; gap: 8px;
      margin: 12px 14px 4px; padding: 9px 14px; border-radius: 14px;
      font-size: 11.5px; font-weight: 600; font-family: var(--font-mono);
    }
    .banner-ok {
      background: color-mix(in srgb, var(--ok) 11%, transparent);
      border: 1px solid color-mix(in srgb, var(--ok) 28%, transparent);
      color: var(--ok);
    }
    .banner-warn {
      background: color-mix(in srgb, var(--warn) 12%, transparent);
      border: 1px solid color-mix(in srgb, var(--warn) 32%, transparent);
      color: color-mix(in srgb, var(--warn) 80%, var(--text));
    }

    /* Section */
    .section {
      margin: 6px 10px 0;
      background: var(--surface-2); border: 1px solid var(--border);
      border-radius: 14px; overflow: hidden;
    }
    .section:last-child { margin-bottom: 10px; }
    .sec-hdr {
      width: 100%; display: flex; align-items: center; gap: 8px;
      background: none; border: none; padding: 11px 14px;
      cursor: pointer; transition: background 0.13s; border-radius: 14px;
    }
    .sec-hdr:hover { background: var(--surface-3); }
    .sdot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .sec-title {
      font-size: 10.5px; font-weight: 700; letter-spacing: 0.11em;
      text-transform: uppercase; color: var(--text-2);
      font-family: var(--font-mono);
    }
    .sec-right {
      margin-left: auto; display: flex; align-items: center; gap: 8px;
    }
    .sec-meta {
      font-size: 10.5px; color: var(--text-3); font-family: var(--font-mono);
    }
    .chev {
      color: var(--text-3); transition: transform 0.2s;
    }
    .chev.chev-open { transform: rotate(180deg); }
    .sec-body { padding: 0 14px 14px; }

    /* Pipeline Trace */
    .trace { padding-left: 4px; }
    .trace-row {
      display: flex; gap: 10px;
      padding-bottom: 11px; position: relative;
    }
    .trace-row:last-child { padding-bottom: 0; }
    .trace-spine {
      display: flex; flex-direction: column; align-items: center;
      flex-shrink: 0; width: 14px;
    }
    .trace-node {
      width: 14px; height: 14px; border-radius: 50%;
      display: grid; place-items: center; flex-shrink: 0; z-index: 1;
      box-shadow: 0 0 0 3px var(--surface);
    }
    .trace-node.node-ok  { background: var(--ok); }
    .trace-node.node-err { background: var(--err); }
    .node-x {
      width: 5px; height: 5px; background: #fff; border-radius: 1px;
    }
    .trace-line {
      flex: 1; width: 2px; background: var(--border); margin-top: 2px;
    }
    .trace-info { flex: 1; min-width: 0; padding-top: 1px; }
    .trace-tool {
      font-size: 12.5px; font-weight: 600; color: var(--text);
      font-family: var(--font-mono);
    }
    .trace-thought {
      font-size: 11px; color: var(--text-3); margin-top: 3px; line-height: 1.4;
    }

    /* SQL block */
    .sql-block {
      position: relative; border-radius: 16px; overflow: hidden;
      background: var(--code-bg); border: 1px solid var(--rail-border, #232a4d);
      box-shadow: var(--shadow-sm);
    }
    .copy-btn {
      position: absolute; top: 8px; right: 8px; z-index: 2;
      display: inline-flex; align-items: center; gap: 4px;
      background: rgba(255,255,255,0.08); color: #fff;
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 10px; padding: 4px 9px;
      font-size: 10.5px; font-weight: 600; cursor: pointer;
      transition: background 0.15s;
    }
    .copy-btn:hover  { background: rgba(255,255,255,0.14); }
    .copy-btn.copy-done { background: var(--ok); border-color: var(--ok); }
    .sql-pre {
      margin: 0; padding: 13px 14px; font-size: 11.5px; line-height: 1.65;
      color: var(--code-text, #D9E0FF); overflow-x: auto;
      white-space: pre; font-family: var(--font-mono);
    }

    /* Reasoning */
    .reasoning {
      font-size: 12.5px; line-height: 1.62; font-style: italic;
      color: var(--text-2);
      border-left: 2.5px solid var(--accent-soft-border, rgba(27,67,255,0.25));
      padding-left: 11px; border-radius: 0 8px 8px 0;
    }

    /* Metrics */
    .metrics-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
    }
    .metric {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 14px; padding: 10px 12px;
      transition: box-shadow 0.18s, transform 0.18s;
    }
    .metric:hover { box-shadow: var(--shadow-sm); transform: translateY(-1px); }
    .metric-lbl {
      font-size: 9.5px; font-weight: 700; letter-spacing: 0.09em;
      color: var(--text-3); text-transform: uppercase;
      font-family: var(--font-mono); margin-bottom: 5px;
    }
    .metric-val {
      font-size: 18px; font-weight: 700; color: var(--text);
      font-family: var(--font-mono); line-height: 1;
    }
    .metric-val.accent { color: var(--accent); }
    .metric-val.err    { color: var(--err); }

    /* Session KV table */
    .kv-table {
      border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
      margin-bottom: 10px;
    }
    .kv-row {
      display: flex; align-items: center; gap: 8px; padding: 8px 11px;
    }
    .kv-row + .kv-row { border-top: 1px solid var(--border); }
    .kv-row.alt { background: var(--surface-2); }
    .kv-k { font-size: 11.5px; color: var(--text-3); font-weight: 500; }
    .kv-v {
      margin-left: auto; font-size: 11.5px; font-weight: 600; color: var(--text);
      text-align: right;
    }
    .kv-err { color: var(--err); }

    /* Live indicator */
    .live-ind {
      display: flex; align-items: center; gap: 7px;
      font-size: 11.5px; color: var(--text-2); font-weight: 600;
    }
    .live-ring { position: relative; display: inline-flex; }
    .live-dot-c {
      width: 7px; height: 7px; border-radius: 50%; background: var(--ok);
    }
    .live-pulse-c {
      position: absolute; inset: 0; border-radius: 50%; background: var(--ok);
      animation: dp-pulse 1.8s ease-in-out infinite;
    }
  `],
})
export class DebugPanelComponent {
  collapsed = false;
  d: ChatResponse | null = null;
  sqlHtml: SafeHtml = '';
  copied = false;

  open = { steps: true, sql: true, reasoning: true, metrics: true, session: false };

  readonly fmtTok = fmtTok;

  get hasErrors(): boolean { return (this.d?.sql_errors ?? 0) > 0; }

  constructor(private svc: ChatService, private san: DomSanitizer) {
    svc.lastDebug$.subscribe(d => {
      this.d = d;
      this.open = { steps: true, sql: true, reasoning: true, metrics: true, session: false };
      if (d?.sql) {
        this.sqlHtml = this.san.bypassSecurityTrustHtml(highlightSql(d.sql));
      }
    });
  }

  copySql() {
    if (!this.d?.sql || this.copied) return;
    navigator.clipboard?.writeText(this.d.sql).catch(() => {});
    this.copied = true;
    setTimeout(() => { this.copied = false; }, 1500);
  }
}
