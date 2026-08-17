import {
  Component, Input, Output, EventEmitter, OnChanges, AfterViewInit, OnDestroy,
  SimpleChanges, ChangeDetectionStrategy, ChangeDetectorRef
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChartDisplayComponent } from '../chart-display/chart-display.component';
import { ResultTableComponent } from '../result-table/result-table.component';
import { DashboardCard, ChartConfig, ResultData, SelectedRow } from '../../models/chat.model';
import { resolveSubject } from '../../utils/subject';

@Component({
  selector: 'app-dashboard-display',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, ChartDisplayComponent, ResultTableComponent],
  template: `
    <div class="dashboard">

      <!-- KPI stat cards -->
      <div class="kpi-row">
        <div class="kpi-card"
             *ngFor="let card of cards; let i = index"
             [class.kpi-in]="entered[i]"
             [class.kpi-card--clickable]="!!card.filter_value"
             [class.kpi-card--critical]="card.severity === 'critical'"
             [class.kpi-card--warning]="card.severity === 'warning'"
             [class.kpi-card--ok]="card.severity === 'ok'"
             [attr.title]="card.filter_value ? 'Click to see these rows' : null"
             (click)="onCardClick(card)">
          <div class="kpi-icon" [ngClass]="'kpi-icon--' + card.icon">
            <svg *ngIf="card.icon === 'star'" viewBox="0 0 20 20" fill="currentColor">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
            </svg>
            <svg *ngIf="card.icon === 'list'" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h4a1 1 0 110 2H4a1 1 0 01-1-1z" clip-rule="evenodd"/>
            </svg>
            <!-- range: up-down arrows -->
            <svg *ngIf="card.icon === 'range'" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L11 6.414V13.586l1.293-1.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 011.414-1.414L9 13.586V6.414L7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3z" clip-rule="evenodd"/>
            </svg>
            <!-- hash: count distinct -->
            <svg *ngIf="card.icon === 'hash'" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M9.243 3.03a1 1 0 01.727 1.213L9.53 6h2.94l.56-2.243a1 1 0 111.94.486L14.53 6H17a1 1 0 110 2h-2.97l-.6 2.4H16a1 1 0 110 2h-3.07l-.56 2.243a1 1 0 11-1.94-.486L10.97 12H8.03l-.56 2.243a1 1 0 11-1.94-.486L6.47 12H4a1 1 0 110-2h2.97l.6-2.4H5a1 1 0 010-2h3.07l.56-2.243a1 1 0 011.213-.727zM9.03 8l-.6 2.4h2.94l.6-2.4H9.03z" clip-rule="evenodd"/>
            </svg>
            <!-- status: dot-in-ring, colored by severity -->
            <svg *ngIf="card.icon === 'status'" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.5 2.5a1 1 0 001.414-1.414L11 9.586V6z" clip-rule="evenodd"/>
            </svg>
          </div>
          <div class="kpi-body">
            <div class="kpi-label">{{ card.label }}</div>

            <!-- Range card: show min ↔ max inline -->
            <ng-container *ngIf="card.kind === 'range'; else normalValue">
              <div class="kpi-range">
                <span class="kpi-range-val">{{ fmtVal(displayValues[i]) }}</span>
                <span class="kpi-range-sep">–</span>
                <span class="kpi-range-val kpi-range-max">{{ fmtVal(card.value2 ?? 0) }}</span>
              </div>
            </ng-container>
            <ng-template #normalValue>
              <div class="kpi-value">{{ fmtVal(displayValues[i]) }}</div>
            </ng-template>

            <span class="kpi-badge"
              [class.kpi-badge--avg]="card.kind === 'avg'"
              [class.kpi-badge--sum]="card.kind === 'sum'"
              [class.kpi-badge--range]="card.kind === 'range'"
              [class.kpi-badge--distinct]="card.kind === 'distinct'"
              [class.kpi-badge--critical]="card.severity === 'critical'"
              [class.kpi-badge--warning]="card.severity === 'warning'"
              [class.kpi-badge--ok]="card.severity === 'ok'">
              {{ card.kind === 'avg' ? 'Average' : card.kind === 'sum' ? 'Total' : card.kind === 'range' ? 'Min – Max' : card.kind === 'status' ? (card.severity ?? 'Status') : 'Distinct' }}
            </span>
          </div>
        </div>
      </div>

      <!-- Chart + Table: side-by-side or stacked -->
      <div class="data-grid"
           [class.stacked]="stackedMode !== 'none'"
           [class.table-first]="stackedMode === 'table-first'">

        <!-- Chart panel -->
        <div class="panel panel-chart">
          <div class="panel-toolbar">
            <button class="btn-panel-expand"
                    (click)="toggleChartStack()"
                    [title]="stackedMode === 'chart-first' ? 'Restore side-by-side' : 'Expand chart'">
              <svg *ngIf="stackedMode !== 'chart-first'" viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
                <path d="M1.5 1.5v4h1.5v-2.5h2.5v-1.5h-4zm4 13h-2.5v-2.5h-1.5v4h4v-1.5zm9-9h-1.5v2.5h-2.5v1.5h4v-4zm-4 7h2.5v-2.5h1.5v4h-4v-1.5z"/>
              </svg>
              <svg *ngIf="stackedMode === 'chart-first'" viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
                <path d="M5.5 5.5v-4h-1.5v2.5h-2.5v1.5h4zm-4 5h2.5v2.5h1.5v-4h-4v1.5zm9-5h1.5v-2.5h2.5v-1.5h-4v4zm4 5h-2.5v2.5h-1.5v-4h4v1.5z"/>
              </svg>
            </button>
          </div>
          <app-chart-display
            [cfg]="chartConfig"
            [sessionId]="sessionId"
            [mode]="'control'"
            [noOverlay]="true"
            [resizeTrigger]="chartResizeTrigger"
            [sourceQuestion]="sourceQuestion"
            (fullscreenChange)="onChartFullscreen($event)"
            (drilldown)="onChartDrilldown($event)">
          </app-chart-display>
        </div>

        <!-- Table panel -->
        <div class="panel panel-table">
          <div class="panel-toolbar">
            <button class="btn-panel-expand"
                    (click)="toggleTableStack()"
                    [title]="stackedMode === 'table-first' ? 'Restore side-by-side' : 'Expand table & chart'">
              <svg *ngIf="stackedMode !== 'table-first'" viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
                <path d="M1.5 1.5v4h1.5v-2.5h2.5v-1.5h-4zm4 13h-2.5v-2.5h-1.5v4h4v-1.5zm9-9h-1.5v2.5h-2.5v1.5h4v-4zm-4 7h2.5v-2.5h1.5v4h-4v-1.5z"/>
              </svg>
              <svg *ngIf="stackedMode === 'table-first'" viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
                <path d="M5.5 5.5v-4h-1.5v2.5h-2.5v1.5h4zm-4 5h2.5v2.5h1.5v-4h-4v1.5zm9-5h1.5v-2.5h2.5v-1.5h-4v4zm4 5h-2.5v2.5h-1.5v-4h4v1.5z"/>
              </svg>
            </button>
          </div>
          <app-result-table
            [data]="resultData"
            [sessionId]="sessionId"
            [messageId]="messageId"
            [selectable]="selectable"
            [resetSelection]="resetSelection"
            (selectionChange)="selectionChange.emit($event)">
          </app-result-table>
        </div>

      </div>
    </div>
  `,
  styles: [`
    :host { display: block; margin-top: 12px; }

    /* ── KPI cards ── */
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }

    .kpi-card {
      position: relative;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 16px;
      display: flex;
      align-items: flex-start;
      gap: 12px;
      box-shadow: var(--shadow-sm);
      overflow: hidden;
      opacity: 0;
      transform: translateY(14px) scale(0.97);
      transition: opacity 0.45s cubic-bezier(0.16, 1, 0.3, 1),
                  transform 0.45s cubic-bezier(0.16, 1, 0.3, 1),
                  box-shadow 0.22s ease;
    }
    /* Gradient top-border accent on hover */
    .kpi-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0; height: 2.5px;
      background: linear-gradient(90deg, #1B43FF, #5C73FF);
      opacity: 0;
      transition: opacity 0.25s ease;
      border-radius: 14px 14px 0 0;
    }
    .kpi-card.kpi-in { opacity: 1; transform: translateY(0) scale(1); }
    .kpi-card:hover  { box-shadow: var(--shadow); }
    .kpi-card:hover::before { opacity: 1; }

    /* Clickable status cards — drill into the underlying rows for that status */
    .kpi-card--clickable { cursor: pointer; }
    .kpi-card--clickable:hover { transform: translateY(-1px); }
    .kpi-card--clickable:active { transform: translateY(0) scale(0.98); }

    /* Severity accents — critical/warning/ok replace the default blue
       gradient top-border with a semantic color, so a status breakdown
       ("Overdue: 5", "Open: 3", "Closed: 12") reads at a glance instead of
       needing the label text to carry all the meaning. */
    .kpi-card--critical::before { background: linear-gradient(90deg, #dc2626, #ef4444); opacity: 1; }
    .kpi-card--warning::before  { background: linear-gradient(90deg, #d97706, #f59e0b); opacity: 1; }
    .kpi-card--ok::before       { background: linear-gradient(90deg, #059669, #10b981); opacity: 1; }

    .kpi-icon--status.kpi-icon--status { background: rgba(107,114,128,0.12); color: var(--gray-500); }
    .kpi-card--critical .kpi-icon--status {
      background: linear-gradient(135deg, rgba(220,38,38,0.16), rgba(239,68,68,0.10));
      color: #dc2626;
    }
    .kpi-card--warning .kpi-icon--status {
      background: linear-gradient(135deg, rgba(217,119,6,0.16), rgba(245,158,11,0.10));
      color: #d97706;
    }
    .kpi-card--ok .kpi-icon--status {
      background: linear-gradient(135deg, rgba(5,150,105,0.16), rgba(16,185,129,0.10));
      color: #059669;
    }

    .kpi-badge--critical { background: rgba(220,38,38,0.1);  color: #dc2626; border: 1px solid rgba(220,38,38,0.2); }
    .kpi-badge--warning  { background: rgba(217,119,6,0.1);  color: #d97706; border: 1px solid rgba(217,119,6,0.2); }
    .kpi-badge--ok       { background: rgba(5,150,105,0.1);  color: #059669; border: 1px solid rgba(5,150,105,0.2); }

    /* A critical card gets a slow, subtle pulse on its icon once settled —
       purposeful liveliness (draws the eye to what actually needs
       attention) rather than decorative motion everywhere. */
    .kpi-card--critical.kpi-in .kpi-icon--status {
      animation: kpi-pulse 2.2s ease-in-out infinite;
    }
    @keyframes kpi-pulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.35); }
      50%      { box-shadow: 0 0 0 6px rgba(220,38,38,0); }
    }

    .kpi-icon {
      width: 38px; height: 38px;
      border-radius: 11px;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
      transition: transform 0.22s ease;
    }
    .kpi-card:hover .kpi-icon { transform: scale(1.1) rotate(-4deg); }
    .kpi-icon svg { width: 17px; height: 17px; }
    .kpi-icon--star {
      background: linear-gradient(135deg, rgba(27,67,255,0.14), rgba(92,115,255,0.10));
      color: var(--accent);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.5);
    }
    .kpi-icon--list {
      background: linear-gradient(135deg, rgba(224,122,110,0.13), rgba(214,104,89,0.1));
      color: #e07a6e;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.5);
    }
    .kpi-icon--range {
      background: linear-gradient(135deg, rgba(217,119,6,0.13), rgba(234,88,12,0.1));
      color: #d97706;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.5);
    }
    .kpi-icon--hash {
      background: linear-gradient(135deg, rgba(124,58,237,0.13), rgba(27,67,255,0.10));
      color: #7c3aed;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.5);
    }

    .kpi-body { min-width: 0; flex: 1; }

    .kpi-label {
      font-size: 10.5px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.6px; color: var(--gray-500); margin-bottom: 4px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .kpi-value {
      font-size: 23px; font-weight: 800; color: var(--gray-900);
      font-family: ui-monospace, SFMono-Regular, monospace;
      line-height: 1.15; margin-bottom: 5px;
      letter-spacing: -0.5px;
    }
    .kpi-badge {
      display: inline-block; padding: 2px 8px; border-radius: 20px;
      font-size: 9.5px; font-weight: 600; letter-spacing: 0.3px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      text-transform: uppercase;
    }
    .kpi-badge--avg      { background: rgba(99,102,241,0.1);  color: #6366f1; border: 1px solid rgba(99,102,241,0.2); }
    .kpi-badge--sum      { background: rgba(5,150,105,0.1);   color: #059669; border: 1px solid rgba(5,150,105,0.2); }
    .kpi-badge--range    { background: rgba(217,119,6,0.1);   color: #d97706; border: 1px solid rgba(217,119,6,0.2); }
    .kpi-badge--distinct { background: rgba(8,145,178,0.1);   color: #0891b2; border: 1px solid rgba(8,145,178,0.2); }

    /* ── Range card value layout ── */
    .kpi-range {
      display: flex; align-items: baseline; gap: 4px;
      margin-bottom: 5px;
    }
    .kpi-range-val {
      font-size: 20px; font-weight: 800; color: var(--gray-900);
      font-family: ui-monospace, SFMono-Regular, monospace;
      line-height: 1.15; letter-spacing: -0.5px;
    }
    .kpi-range-sep {
      font-size: 14px; font-weight: 500; color: var(--gray-400);
    }
    .kpi-range-max { color: #d97706; }

    /* ── Side-by-side grid (default) ── */
    .data-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      align-items: start;
    }

    /* Stacked mode: collapse to single column — each panel fills 100% width */
    .data-grid.stacked {
      grid-template-columns: 1fr;
    }

    .panel { min-width: 0; overflow: hidden; width: 100%; }

    /* Default visual order */
    .panel-chart { order: 1; }
    .panel-table  { order: 2; }

    /* Table-first: swap so table appears above chart */
    .data-grid.table-first .panel-chart { order: 2; }
    .data-grid.table-first .panel-table  { order: 1; }

    /* ── Panel toolbars (expand buttons) ── */
    .panel-toolbar {
      display: flex; align-items: center; justify-content: flex-end;
      padding: 4px 2px 4px;
    }
    .btn-panel-expand {
      display: flex; align-items: center; justify-content: center;
      width: 26px; height: 26px; border-radius: 7px;
      background: var(--surface); border: 1px solid var(--border);
      color: var(--gray-400); cursor: pointer;
      transition: all 0.15s ease; padding: 0;
      box-shadow: var(--shadow-sm);
    }
    .btn-panel-expand:hover {
      background: var(--blue-50); border-color: var(--blue-500);
      color: var(--blue-500); transform: scale(1.05);
      box-shadow: 0 0 0 2px rgba(37,99,235,0.1);
    }

    /* Always stack on very narrow containers */
    @media (max-width: 680px) {
      .data-grid { grid-template-columns: 1fr; }
    }

    /* Respect reduced motion */
    @media (prefers-reduced-motion: reduce) {
      .kpi-card { transition: none; opacity: 1; transform: none !important; }
      .kpi-card--critical.kpi-in .kpi-icon--status { animation: none; }
    }
  `],
})
export class DashboardDisplayComponent implements OnChanges, AfterViewInit, OnDestroy {
  @Input() cards:        DashboardCard[] = [];
  @Input() resultData!:  ResultData;
  @Input() chartConfig!: ChartConfig;
  @Input() sessionId:    string = '';
  @Input() messageId:    string = '';
  // Row-selection passthrough — this component wraps its own <app-result-table>
  // (the "panel-table" below), which was missed when checkbox selection was
  // first wired up: any query that triggers the KPI-dashboard view instead of
  // the plain table never got these bindings, so its table had no checkboxes
  // at all regardless of ResultData.row_selectable.
  @Input() selectable = false;
  @Input() resetSelection = 0;
  @Output() selectionChange = new EventEmitter<SelectedRow[]>();
  // Card click-to-drill — the question that produced this dashboard, used
  // to infer the subject noun ("corrective actions", "audit schedules", ...)
  // the same way result-table.component.ts's onRowClick/resolveSubject does.
  @Input() sourceQuestion?: string;
  @Output() drilldown = new EventEmitter<string>();

  displayValues: number[] = [];
  entered:       boolean[] = [];
  stackedMode: 'none' | 'chart-first' | 'table-first' = 'none';
  chartResizeTrigger = 0;

  private timers: any[] = [];
  private destroyed = false;

  /** Click a status KPI card ("Overdue: 5") to drill into just those rows. */
  onCardClick(card: DashboardCard): void {
    if (!card.filter_value) return;
    const subject = resolveSubject(this.sourceQuestion, 'items');
    this.drilldown.emit(`show me the ${subject} that are ${card.label.toLowerCase()}`);
  }

  /** Bubble a chart-segment click (see chart-display.component.ts) up as
   * the same kind of drilldown request a card click produces. */
  onChartDrilldown(text: string): void {
    this.drilldown.emit(text);
  }

  constructor(private cdr: ChangeDetectorRef) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['cards']) {
      this.displayValues = this.cards.map(() => 0);
      this.entered       = this.cards.map(() => false);
    }
  }

  ngAfterViewInit(): void {
    this.cards.forEach((card, i) => {
      const t = setTimeout(() => {
        if (this.destroyed) return;
        this.entered[i] = true;
        this.cdr.markForCheck();
        this._countUp(i, card.value);
      }, 100 + i * 120);
      this.timers.push(t);
    });
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    this.timers.forEach(t => clearTimeout(t));
    this.timers = [];
  }

  toggleChartStack(): void {
    this.stackedMode = this.stackedMode === 'chart-first' ? 'none' : 'chart-first';
    this.chartResizeTrigger++;
    this.cdr.markForCheck();
  }

  toggleTableStack(): void {
    this.stackedMode = this.stackedMode === 'table-first' ? 'none' : 'table-first';
    this.chartResizeTrigger++;
    this.cdr.markForCheck();
  }

  onChartFullscreen(isFullscreen: boolean): void {
    this.stackedMode = isFullscreen ? 'chart-first' : 'none';
    this.chartResizeTrigger++;
    this.cdr.markForCheck();
  }

  fmtVal(v: number): string {
    if (!isFinite(v)) return '0';
    const a = Math.abs(v);
    if (a >= 1_000_000) return (v / 1_000_000).toFixed(2) + 'M';
    if (a >= 10_000)    return Math.round(v).toLocaleString('en-US');
    if (a >= 100)       return (Math.round(v * 10) / 10).toString();
    return (Math.round(v * 100) / 100).toString();
  }

  private _countUp(index: number, target: number): void {
    const steps    = 30;
    const interval = 20; // 30 × 20ms = 600ms total
    let   step     = 0;
    const timer = setInterval(() => {
      if (this.destroyed) { clearInterval(timer); return; }
      step++;
      const progress = step / steps;
      const ease     = 1 - Math.pow(1 - progress, 3); // cubic ease-out
      this.displayValues[index] = target * ease;
      this.cdr.markForCheck();
      if (step >= steps) {
        clearInterval(timer);
        this.displayValues[index] = target;
        this.cdr.markForCheck();
      }
    }, interval);
    this.timers.push(timer);
  }
}
