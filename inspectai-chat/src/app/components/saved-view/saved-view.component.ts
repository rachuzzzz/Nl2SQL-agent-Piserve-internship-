import { Component, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

interface SavedCard {
  name:   string;
  tag:    string;
  runs:   number;
  color:  string;
  sparkType: 'line' | 'bar' | 'area';
  sparkPoly: string;
  sparkArea: string;
  sparkBars: { x: number; y: number; bw: number; bh: number }[];
}

const SW = 280, SH = 90;

function spark(data: number[], w: number, h: number) {
  const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
  const step = w / (data.length - 1);
  const pad = h * 0.12;
  const pts = data.map((v, i) => ({
    x: +(i * step).toFixed(1),
    y: +(pad + (1 - (v - min) / range) * (h - 2 * pad)).toFixed(1),
  }));
  const poly = pts.map(p => `${p.x},${p.y}`).join(' ');
  const last = pts[pts.length - 1];
  const area = `M${poly.replace(/,/g, ',')} L${last.x},${h} L0,${h} Z`;
  return { poly, area };
}

function bars(data: number[], w: number, h: number) {
  const max = Math.max(...data) || 1;
  const gap = w / data.length;
  const bw = gap * 0.58;
  const pad = h * 0.06;
  return data.map((v, i) => {
    const bh = (v / max) * (h - pad);
    return { x: +(i * gap + (gap - bw) / 2).toFixed(1), y: +(h - bh).toFixed(1), bw: +bw.toFixed(1), bh: +bh.toFixed(1) };
  });
}

const empty: any[] = [];

const AVG_SCORE_SPARK   = spark([91.64, 81.75, 87.31, 81.35, 85.55], SW, SH);
const OPEN_CAPA_BARS    = bars([38, 29, 22, 18, 14, 12, 9], SW, SH);
const PERMITS_BARS      = bars([5, 4, 4, 4, 3, 3], SW, SH);
const INCIDENTS_SPARK   = spark([41, 35, 31, 29, 24], SW, SH);
const INSPECTOR_BARS    = bars([218, 196, 174, 161, 145], SW, SH);
const AUDIT_BARS        = bars([218, 196, 174, 161, 145], SW, SH);

const SAVED_CARDS: SavedCard[] = [
  { name: 'Monthly inspection score', tag: 'Inspections', runs: 142, color: 'var(--accent)',
    sparkType: 'line', sparkPoly: AVG_SCORE_SPARK.poly, sparkArea: AVG_SCORE_SPARK.area, sparkBars: empty },
  { name: 'Open CAPAs by site',       tag: 'CAPA',        runs: 98,  color: 'var(--viz-blue)',
    sparkType: 'bar',  sparkPoly: '', sparkArea: '', sparkBars: OPEN_CAPA_BARS },
  { name: 'Permits expiring ≤30d',    tag: 'Permits',     runs: 211, color: 'var(--viz-amber)',
    sparkType: 'bar',  sparkPoly: '', sparkArea: '', sparkBars: PERMITS_BARS },
  { name: 'Incident severity trend',  tag: 'Safety',      runs: 76,  color: 'var(--viz-coral)',
    sparkType: 'area', sparkPoly: INCIDENTS_SPARK.poly, sparkArea: INCIDENTS_SPARK.area, sparkBars: empty },
  { name: 'Top inspectors',           tag: 'Audits',      runs: 54,  color: 'var(--viz-violet)',
    sparkType: 'bar',  sparkPoly: '', sparkArea: '', sparkBars: INSPECTOR_BARS },
  { name: 'Audit pass rate by region',tag: 'Audits',      runs: 33,  color: 'var(--viz-blue)',
    sparkType: 'bar',  sparkPoly: '', sparkArea: '', sparkBars: AUDIT_BARS },
];

@Component({
  selector: 'app-saved-view',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="sv-scroll">
      <div class="sv-inner">

        <div class="page-head">
          <div>
            <h1 class="page-title">Saved queries</h1>
            <p class="page-sub">{{ cards.length }} pinned questions · reusable across your team</p>
          </div>
        </div>

        <div class="card-grid">
          <div *ngFor="let c of cards" class="s-card">
            <div class="s-card-top">
              <div class="s-tag-row">
                <svg viewBox="0 0 24 24" fill="none" width="13" height="13" stroke-linecap="round"
                     stroke-linejoin="round" stroke-width="2" [style.stroke]="c.color">
                  <path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>
                </svg>
                <span class="s-tag">{{ c.tag }}</span>
                <span class="s-runs">{{ c.runs }} runs</span>
              </div>
              <h3 class="s-name">{{ c.name }}</h3>
            </div>

            <!-- Mini sparkline -->
            <div class="spark-wrap">
              <svg [attr.viewBox]="'0 0 ' + SW + ' ' + SH" class="spark-svg" style="overflow:visible">
                <ng-container *ngIf="c.sparkType === 'line'">
                  <path [attr.d]="c.sparkArea" [attr.fill]="c.color" [attr.fill-opacity]="0.10" stroke="none"/>
                  <polyline [attr.points]="c.sparkPoly" fill="none" [attr.stroke]="c.color"
                            stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </ng-container>
                <ng-container *ngIf="c.sparkType === 'area'">
                  <path [attr.d]="c.sparkArea" [attr.fill]="c.color" [attr.fill-opacity]="0.15" stroke="none"/>
                  <polyline [attr.points]="c.sparkPoly" fill="none" [attr.stroke]="c.color"
                            stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </ng-container>
                <ng-container *ngIf="c.sparkType === 'bar'">
                  <rect *ngFor="let b of c.sparkBars"
                        [attr.x]="b.x" [attr.y]="b.y"
                        [attr.width]="b.bw" [attr.height]="b.bh"
                        rx="2.5" [attr.fill]="c.color" opacity="0.78"/>
                </ng-container>
              </svg>
            </div>

            <div class="s-footer">
              <button class="run-btn" (click)="switchToChat.emit()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"
                     stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                Run query
              </button>
              <button class="copy-btn" title="Copy question">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"
                     stroke-linecap="round" stroke-linejoin="round">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                  <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                </svg>
              </button>
            </div>
          </div>
        </div>

        <!-- Empty state if no cards -->
        <div class="empty" *ngIf="cards.length === 0">
          <div class="empty-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="28" height="28"
                 stroke-linecap="round" stroke-linejoin="round">
              <path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>
            </svg>
          </div>
          <p class="empty-text">No saved queries yet</p>
          <p class="empty-sub">Run a query in AI Chat and pin it here to save it for your team.</p>
        </div>

      </div>
    </div>
  `,
  styles: [`
    :host {
      flex: 1; display: flex; flex-direction: column; overflow: hidden;
      background: var(--bg); min-width: 0;
    }
    .sv-scroll { flex: 1; overflow-y: auto; }
    .sv-inner  { max-width: 1100px; margin: 0 auto; padding: 30px 32px 60px; }

    .page-head { margin-bottom: 24px; }
    .page-title {
      margin: 0 0 5px; font-size: 28px; font-weight: 800; letter-spacing: -0.02em;
      color: var(--text);
    }
    .page-sub { margin: 0; font-size: 14px; color: var(--text-2); }

    .card-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 14px;
    }
    .s-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 22px; overflow: hidden; display: flex; flex-direction: column;
      box-shadow: var(--shadow-sm); transition: box-shadow 0.22s, transform 0.22s, border-color 0.22s;
    }
    .s-card:hover { box-shadow: var(--shadow-md), 0 0 0 1px rgba(27,67,255,0.08); transform: translateY(-3px); border-color: var(--blue-100); }

    .s-card-top { padding: 15px 16px 6px; }
    .s-tag-row {
      display: flex; align-items: center; gap: 7px; margin-bottom: 8px;
    }
    .s-tag {
      font-size: 11px; font-weight: 700; color: var(--text-2);
      background: var(--surface-3); padding: 2px 9px; border-radius: 999px;
    }
    .s-runs {
      font-size: 11px; font-weight: 600; color: var(--text-3);
      font-family: var(--font-mono); margin-left: auto;
    }
    .s-name {
      margin: 0; font-size: 15px; font-weight: 700; color: var(--text);
      line-height: 1.35;
    }

    .spark-wrap {
      padding: 6px 10px 4px; overflow: hidden;
    }
    .spark-svg { width: 100%; display: block; }

    .s-footer {
      margin-top: auto; padding: 11px 14px;
      border-top: 1px solid var(--border);
      display: flex; gap: 8px; align-items: center;
    }
    .run-btn {
      flex: 1; display: flex; align-items: center; justify-content: center; gap: 6px;
      padding: 8px 14px; border-radius: 999px;
      background: var(--accent); border: none; color: #fff;
      font-size: 13px; font-weight: 600; cursor: pointer; font-family: var(--font);
      transition: opacity 0.15s;
    }
    .run-btn:hover { opacity: 0.86; }
    .copy-btn {
      display: flex; align-items: center; justify-content: center;
      width: 34px; height: 34px; padding: 0; border-radius: 999px;
      background: var(--surface-2); border: 1px solid var(--border);
      color: var(--text-2); cursor: pointer; transition: background 0.14s;
    }
    .copy-btn:hover { background: var(--surface-3); }

    /* Empty state */
    .empty {
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      padding: 64px 24px; gap: 12px; color: var(--text-3);
    }
    .empty-icon {
      width: 56px; height: 56px; border-radius: 16px;
      background: var(--blue-50); border: 1px solid var(--blue-100);
      display: flex; align-items: center; justify-content: center; color: var(--accent);
    }
    .empty-text { font-size: 16px; font-weight: 600; color: var(--text); margin: 0; }
    .empty-sub  { font-size: 13px; color: var(--text-3); margin: 0; text-align: center; max-width: 320px; line-height: 1.6; }
  `],
})
export class SavedViewComponent {
  @Output() switchToChat = new EventEmitter<void>();

  readonly cards = SAVED_CARDS;
  readonly SW = SW;
  readonly SH = SH;
}
