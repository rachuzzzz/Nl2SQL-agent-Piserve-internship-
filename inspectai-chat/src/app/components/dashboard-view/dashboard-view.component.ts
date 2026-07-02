import { Component, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

interface Kpi { label: string; value: string; delta: string; up: boolean; iconPath: string; }
interface RecentItem { title: string; tag: string; }
interface BarRect { x: number; y: number; bw: number; bh: number; }
interface ScorePt { cx: number; cy: number; label: string; }

const W = 340, H = 140;

function computeLinePts(data: number[]): ScorePt[] {
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const step = W / (data.length - 1);
  const pad = H * 0.12;
  return data.map((v, i) => ({
    cx: +(i * step).toFixed(1),
    cy: +(pad + (1 - (v - min) / range) * (H - 2 * pad)).toFixed(1),
    label: v.toFixed(1),
  }));
}

function ptsToPolyline(pts: ScorePt[]): string {
  return pts.map(p => `${p.cx},${p.cy}`).join(' ');
}

function ptsToArea(pts: ScorePt[]): string {
  const line = pts.map(p => `${p.cx},${p.cy}`).join(' L');
  const last = pts[pts.length - 1];
  return `M${line} L${last.cx},${H} L0,${H} Z`;
}

function computeBars(data: number[]): BarRect[] {
  const max = Math.max(...data) || 1;
  const count = data.length;
  const gap = W / count;
  const bw = gap * 0.56;
  const pad = H * 0.06;
  return data.map((v, i) => {
    const bh = (v / max) * (H - pad);
    return { x: +(i * gap + (gap - bw) / 2).toFixed(1), y: +(H - bh).toFixed(1), bw: +bw.toFixed(1), bh: +bh.toFixed(1) };
  });
}

const SCORE_DATA   = [91.64, 81.75, 87.31, 81.35, 85.55];
const SCORE_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May'];
const CAPA_DATA    = [38, 29, 22, 18, 14, 12, 9];
const CAPA_LABELS  = ['Houston', 'Rotterdam', 'Singapore', 'Antwerp', 'Jubail', 'Tex.City', 'Pernis'];

const ICON_ROWS     = 'M3 12h18M3 6h18M3 18h18';
const ICON_WARN     = 'M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01';
const ICON_PERMIT   = 'M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6M16 13H8M16 17H8M10 9H8';
const ICON_INCIDENT = 'M13 2L3 14h9l-1 8 10-12h-9l1-8z';

const KPIS: Kpi[] = [
  { label: 'INSPECTIONS THIS MONTH', value: '1,284', delta: '+8.2%', up: true,  iconPath: ICON_ROWS },
  { label: 'OPEN CORRECTIVE ACTIONS', value: '168',   delta: '−12',   up: true,  iconPath: ICON_WARN },
  { label: 'PERMITS EXPIRING ≤30d',   value: '23',    delta: '+5',    up: false, iconPath: ICON_PERMIT },
  { label: 'LOST-TIME INCIDENTS YTD', value: '2',     delta: '0',     up: true,  iconPath: ICON_INCIDENT },
];

const RECENT: RecentItem[] = [
  { title: 'What is the average inspection score across months?', tag: 'Inspections' },
  { title: 'Show open corrective actions by site',                tag: 'CAPA' },
  { title: 'How many permits expire in the next 30 days?',        tag: 'Permits' },
  { title: 'Trend of safety incidents by severity this year',     tag: 'Safety' },
  { title: 'Top 5 inspectors by audits completed',               tag: 'Audits' },
];

@Component({
  selector: 'app-dashboard-view',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="d-scroll">
      <div class="d-inner">

        <!-- Page head -->
        <div class="page-head">
          <div>
            <h1 class="page-title">Operations overview</h1>
            <p class="page-sub">Live EHS posture across all sites · updated 2 min ago</p>
          </div>
          <button class="pill-btn" (click)="switchToChat.emit()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"
                 stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"/>
              <path d="M19 3v4M21 5h-4"/>
            </svg>
            Ask OctaAI
          </button>
        </div>

        <!-- KPI grid -->
        <div class="kpi-grid">
          <div *ngFor="let k of kpis" class="kpi-card">
            <div class="kpi-top">
              <span class="kpi-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                     width="17" height="17" stroke-linecap="round" stroke-linejoin="round">
                  <path [attr.d]="k.iconPath"/>
                </svg>
              </span>
              <span class="kpi-label">{{ k.label }}</span>
            </div>
            <div class="kpi-value">{{ k.value }}</div>
            <div class="kpi-delta">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
                   width="12" height="12" stroke-linecap="round" stroke-linejoin="round"
                   [style.color]="k.up ? 'var(--ok)' : 'var(--err)'">
                <polyline *ngIf="k.up"  points="18 15 12 9 6 15"/>
                <polyline *ngIf="!k.up" points="6 9 12 15 18 9"/>
              </svg>
              <span [style.color]="k.up ? 'var(--ok)' : 'var(--err)'" style="font-weight:700">{{ k.delta }}</span>
              <span class="kpi-cmp">vs last month</span>
            </div>
          </div>
        </div>

        <!-- Chart cards -->
        <div class="chart-grid">
          <!-- Line: inspection score trend -->
          <div class="chart-card">
            <div class="chart-card-head">
              <h3 class="chart-title">Inspection score trend</h3>
              <button class="open-btn" (click)="switchToChat.emit()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="11" height="11"
                     stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                Open
              </button>
            </div>
            <svg [attr.viewBox]="'0 0 ' + W + ' ' + (H + 28)" class="chart-svg">
              <!-- Area fill -->
              <path [attr.d]="scoreAreaPath" fill="rgba(27,67,255,0.08)" stroke="none"/>
              <!-- Line -->
              <polyline [attr.points]="scorePolyline"
                        fill="none" stroke="var(--accent)" stroke-width="2.2"
                        stroke-linecap="round" stroke-linejoin="round"/>
              <!-- Dots -->
              <circle *ngFor="let p of scorePts"
                      [attr.cx]="p.cx" [attr.cy]="p.cy" r="3.5"
                      fill="var(--accent)" stroke="var(--surface)" stroke-width="2"/>
              <!-- X labels -->
              <text *ngFor="let p of scorePts; let i = index"
                    [attr.x]="p.cx" [attr.y]="H + 22" text-anchor="middle"
                    fill="var(--text-3)" font-size="10" font-family="var(--font-mono)">
                {{ scoreLabels[i] }}
              </text>
            </svg>
          </div>

          <!-- Bar: open CAPAs by site -->
          <div class="chart-card">
            <div class="chart-card-head">
              <h3 class="chart-title">Open corrective actions by site</h3>
              <button class="open-btn" (click)="switchToChat.emit()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="11" height="11"
                     stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                Open
              </button>
            </div>
            <svg [attr.viewBox]="'0 0 ' + W + ' ' + (H + 28)" class="chart-svg">
              <rect *ngFor="let b of capaBars; let i = index"
                    [attr.x]="b.x" [attr.y]="b.y"
                    [attr.width]="b.bw" [attr.height]="b.bh"
                    rx="3" fill="var(--accent)" opacity="0.82"/>
              <!-- X labels -->
              <text *ngFor="let b of capaBars; let i = index"
                    [attr.x]="b.x + b.bw / 2" [attr.y]="H + 22" text-anchor="middle"
                    fill="var(--text-3)" font-size="9" font-family="var(--font-mono)">
                {{ capaLabels[i] }}
              </text>
            </svg>
          </div>
        </div>

        <!-- Recent queries -->
        <div class="recent-card">
          <div class="recent-head">
            <h3 class="chart-title">Recent queries</h3>
          </div>
          <button *ngFor="let r of recent; let i = index"
                  class="recent-row"
                  [class.not-first]="i > 0"
                  (click)="switchToChat.emit()">
            <span class="recent-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"
                   stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
              </svg>
            </span>
            <span class="recent-title">{{ r.title }}</span>
            <span class="recent-tag">{{ r.tag }}</span>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="13" height="13"
                 stroke-linecap="round" stroke-linejoin="round" style="color:var(--text-3);flex-shrink:0">
              <polyline points="9 18 15 12 9 6"/>
            </svg>
          </button>
        </div>

      </div>
    </div>
  `,
  styles: [`
    :host {
      flex: 1; display: flex; flex-direction: column; overflow: hidden;
      background: var(--bg); min-width: 0;
    }
    .d-scroll { flex: 1; overflow-y: auto; }
    .d-inner  { max-width: 1100px; margin: 0 auto; padding: 30px 32px 60px; }

    .page-head {
      display: flex; align-items: flex-end; justify-content: space-between;
      gap: 12px; margin-bottom: 24px; flex-wrap: wrap;
    }
    .page-title {
      margin: 0 0 5px; font-size: 28px; font-weight: 800; letter-spacing: -0.02em;
      color: var(--text);
    }
    .page-sub { margin: 0; font-size: 14px; color: var(--text-2); }

    .pill-btn {
      display: flex; align-items: center; gap: 6px;
      padding: 9px 16px; border-radius: 999px;
      background: var(--accent); border: none; color: #fff;
      font-size: 13px; font-weight: 600; cursor: pointer;
      font-family: var(--font); white-space: nowrap;
      transition: opacity 0.15s, transform 0.15s;
    }
    .pill-btn:hover { opacity: 0.88; transform: translateY(-1px); }

    /* KPI */
    .kpi-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px; margin-bottom: 20px;
    }
    .kpi-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 22px; padding: 18px 20px;
      box-shadow: var(--shadow-sm); transition: box-shadow 0.22s, transform 0.22s, border-color 0.22s;
    }
    .kpi-card:hover { box-shadow: var(--shadow-md), 0 0 0 1px rgba(27,67,255,0.08); transform: translateY(-3px); border-color: var(--blue-100); }
    .kpi-top {
      display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
    }
    .kpi-icon {
      width: 34px; height: 34px; border-radius: 12px;
      background: var(--accent-soft); display: grid; place-items: center;
      flex-shrink: 0; color: var(--accent);
    }
    .kpi-label {
      font-size: 9.5px; font-weight: 700; letter-spacing: 0.07em;
      color: var(--text-3); text-transform: uppercase; line-height: 1.3;
      font-family: var(--font-mono);
    }
    .kpi-value {
      font-size: 30px; font-weight: 700; letter-spacing: -0.02em;
      color: var(--text); font-family: var(--font-mono);
    }
    .kpi-delta {
      display: flex; align-items: center; gap: 4px; margin-top: 8px;
      font-size: 12px;
    }
    .kpi-cmp { color: var(--text-3); }

    /* Charts */
    .chart-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px; margin-bottom: 20px;
    }
    .chart-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 22px; padding: 18px 20px; box-shadow: var(--shadow-sm);
      transition: box-shadow 0.22s, transform 0.22s;
    }
    .chart-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
    .chart-card-head {
      display: flex; align-items: center; margin-bottom: 14px;
    }
    .chart-title {
      margin: 0; font-size: 15px; font-weight: 700; color: var(--text);
    }
    .open-btn {
      margin-left: auto; display: flex; align-items: center; gap: 5px;
      padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600;
      background: var(--blue-50); border: 1px solid var(--blue-100);
      color: var(--accent); cursor: pointer; font-family: var(--font);
      transition: background 0.14s;
    }
    .open-btn:hover { background: var(--blue-100); }
    .chart-svg { width: 100%; overflow: visible; display: block; }

    /* Recent queries */
    .recent-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 22px; overflow: hidden; box-shadow: var(--shadow-sm);
    }
    .recent-head {
      padding: 15px 20px; border-bottom: 1px solid var(--border);
    }
    .recent-row {
      width: 100%; display: flex; align-items: center; gap: 12px;
      padding: 13px 20px; background: transparent; border: none;
      text-align: left; cursor: pointer; transition: background 0.12s;
    }
    .recent-row.not-first { border-top: 1px solid var(--border); }
    .recent-row:hover { background: var(--surface-2); }
    .recent-icon {
      width: 32px; height: 32px; border-radius: 50%;
      background: var(--accent-soft); display: grid; place-items: center;
      flex-shrink: 0; color: var(--accent);
    }
    .recent-title {
      flex: 1; min-width: 0; font-size: 13.5px; font-weight: 600;
      color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .recent-tag {
      font-size: 11px; font-weight: 600; color: var(--text-2);
      background: var(--surface-3); padding: 3px 9px; border-radius: 999px;
      white-space: nowrap;
    }
  `],
})
export class DashboardViewComponent {
  @Output() switchToChat = new EventEmitter<void>();

  readonly kpis        = KPIS;
  readonly recent      = RECENT;
  readonly scoreLabels = SCORE_LABELS;
  readonly capaLabels  = CAPA_LABELS;
  readonly W = W;
  readonly H = H;

  readonly scorePts     = computeLinePts(SCORE_DATA);
  readonly scorePolyline = ptsToPolyline(this.scorePts);
  readonly scoreAreaPath = ptsToArea(this.scorePts);
  readonly capaBars      = computeBars(CAPA_DATA);
}
