import { Component } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { ChatService } from '../../services/chat.service';
import { ChatResponse } from '../../models/chat.model';

@Component({
  selector: 'app-debug-panel',
  standalone: true,
  imports: [CommonModule, DecimalPipe],
  template: `
    <aside class="sidebar-right" [class.collapsed]="collapsed">
      <div class="debug-header">
        <div class="debug-header-left">
          <button class="btn-col-dark" (click)="collapsed = !collapsed" title="Toggle debug panel">
            <svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"/></svg>
          </button>
          <div class="status-dot" [style.background]="statusColor"></div>
          <span class="debug-title">Debug</span>
        </div>
        <span class="call-tag sr-hide" *ngIf="d">
          {{ d!.llm_calls }} call{{ d!.llm_calls !== 1 ? 's' : '' }}
          · {{ d!.total_tokens | number }} tok
        </span>
      </div>

      <div class="debug-body sr-hide">
        <div class="debug-empty" *ngIf="!d">
          <div class="debug-empty-icon">⚡</div>
          <div class="debug-empty-text">
            Send a query to see SQL, reasoning, tokens, and execution steps.
          </div>
        </div>

        <ng-container *ngIf="d">
          <div class="error-bar" *ngIf="(d!.sql_errors || 0) > 0">
            ⚠ {{ (d!.sql_errors || 0) }} SQL error{{ (d!.sql_errors || 0) !== 1 ? 's' : '' }} — validator triggered retry
          </div>

          <div class="dcard" [class.open]="open.sql" *ngIf="d!.sql">
            <div class="dcard-hdr" (click)="open.sql = !open.sql">
              <div class="dcard-title"><div class="dtdot dot-sql"></div>Generated SQL</div>
              <span class="dcard-chev">▼</span>
            </div>
            <div class="dcard-body" *ngIf="open.sql">
              <pre class="sql-pre" [innerHTML]="sqlHtml"></pre>
            </div>
          </div>

          <div class="dcard" [class.open]="open.reasoning" *ngIf="d!.reasoning">
            <div class="dcard-hdr" (click)="open.reasoning = !open.reasoning">
              <div class="dcard-title"><div class="dtdot dot-think"></div>Model Reasoning</div>
              <span class="dcard-chev">▼</span>
            </div>
            <div class="dcard-body" *ngIf="open.reasoning">
              <div class="think-text">"{{ d!.reasoning }}"</div>
            </div>
          </div>

          <div class="dcard" [class.open]="open.metrics">
            <div class="dcard-hdr" (click)="open.metrics = !open.metrics">
              <div class="dcard-title"><div class="dtdot dot-perf"></div>Performance</div>
              <span class="dcard-chev">▼</span>
            </div>
            <div class="dcard-body" *ngIf="open.metrics">
              <div class="metrics-grid">
                <div class="metric hi">
                  <div class="metric-lbl">Response Time</div>
                  <div class="metric-val">{{ ((d!.time_ms || 0)/1000).toFixed(2) }}<span class="metric-unit">s</span></div>
                </div>
                <div class="metric">
                  <div class="metric-lbl">LLM Calls</div>
                  <div class="metric-val">{{ d!.llm_calls }}</div>
                </div>
                <div class="metric">
                  <div class="metric-lbl">Prompt Tokens</div>
                  <div class="metric-val">{{ d!.prompt_tokens | number }}</div>
                </div>
                <div class="metric">
                  <div class="metric-lbl">Completion</div>
                  <div class="metric-val">{{ d!.completion_tokens | number }}</div>
                </div>
                <div class="metric" [class.ok]="(d!.sql_errors || 0) === 0" [class.err]="(d!.sql_errors || 0) > 0">
                  <div class="metric-lbl">SQL Errors</div>
                  <div class="metric-val">{{ (d!.sql_errors || 0) }}</div>
                </div>
                <div class="metric">
                  <div class="metric-lbl">Total Tokens</div>
                  <div class="metric-val">{{ d!.total_tokens | number }}</div>
                </div>
              </div>
            </div>
          </div>

          <div class="dcard" [class.open]="open.steps" *ngIf="d!.steps?.length">
            <div class="dcard-hdr" (click)="open.steps = !open.steps">
              <div class="dcard-title"><div class="dtdot dot-steps"></div>Execution Steps</div>
              <span class="dcard-chev">▼</span>
            </div>
            <div class="dcard-body" *ngIf="open.steps">
              <div class="steps-list">
                <div class="step" *ngFor="let step of d!.steps; let i = index">
                  <div class="step-num"
                    [class.done]="step.success !== false"
                    [class.err]="step.success === false">{{ i + 1 }}</div>
                  <div>
                    <div class="step-tool">{{ step.tool }}()</div>
                    <div class="step-thought">{{ step.thought }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </ng-container>
      </div>
    </aside>
  `,
})
export class DebugPanelComponent {
  collapsed = false;
  d: ChatResponse | null = null;
  sqlHtml: SafeHtml = '';
  open = { sql: true, reasoning: true, metrics: true, steps: true };

  get statusColor(): string {
    if (!this.d) return '#10b981';
    return (this.d.sql_errors ?? 0) > 0 ? '#ef4444' : '#10b981';
  }

  constructor(private svc: ChatService, private san: DomSanitizer) {
    svc.lastDebug$.subscribe(d => {
      this.d = d;
      if (d?.sql) this.sqlHtml = this.san.bypassSecurityTrustHtml(this.highlight(d.sql));
    });
  }

  highlight(sql: string): string {
    const KW = ['SELECT','FROM','WHERE','JOIN','LEFT','INNER','ON','GROUP BY','ORDER BY',
      'HAVING','LIMIT','AND','OR','NOT','IN','AS','DISTINCT','CASE','WHEN','THEN',
      'ELSE','END','WITH','UNION','ALL','NULL','IS','FILTER'];
    const FN = ['AVG','COUNT','SUM','MAX','MIN','ROUND','EXTRACT','TO_CHAR',
      'DATE_TRUNC','NULLIF','COALESCE'];
    let h = sql.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    h = h.replace(/'([^']*)'/g, "<span class=\"ss\">'$1'</span>");
    h = h.replace(/--.*/g, (m: string) => `<span class="sc">${m}</span>`);
    FN.forEach(f => { h = h.replace(new RegExp(`\\b${f}\\b`,'g'), `<span class="sf">${f}</span>`); });
    KW.forEach(k => { h = h.replace(new RegExp(`\\b${k}\\b`,'g'), `<span class="sk">${k}</span>`); });
    h = h.replace(/\b(ir|fac|ica|u|it|rl|aa|aq|ipd|fd|isched|per_inspection)\./g,
      '<span class="sa">$1</span>.');
    return h;
  }
}
