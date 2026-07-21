import {
  Component, Input, Output, EventEmitter, OnChanges,
  SimpleChanges, inject, signal, computed,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { ChartConfig, ColumnMeta } from '../../models/chat.model';

@Component({
  selector: 'app-chart-suggestion',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
<div class="suggestion-wrap" *ngIf="!dismissed">

  <!-- ── Top strip: icon + label + type pills + dismiss ── -->
  <div class="suggestion-strip">
    <div class="strip-left">
      <div class="strip-icon">
        <svg viewBox="0 0 20 20" fill="currentColor">
          <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM6 5a1 1 0 011-1h2a1 1 0 011 1v11a1 1 0 01-1 1H7a1 1 0 01-1-1V5zm6-3a1 1 0 00-1 1v13a1 1 0 102 0V3a1 1 0 00-1-1z"/>
        </svg>
      </div>
      <div class="strip-text">
        <span class="strip-label">Visualize as chart</span>
        <span class="strip-hint" *ngIf="!loading && !activeConfig">pick a type to render</span>
        <span class="strip-hint" *ngIf="loading">loading…</span>
        <span class="strip-hint strip-reason" *ngIf="activeConfig && !loading">{{ activeReason }}</span>
      </div>
    </div>

    <div class="strip-right">
      <!-- Type pills — recommended first, then alternatives from backend -->
      <div class="type-pills">
        <button
          *ngFor="let t of allTypes"
          class="type-pill"
          [class.type-pill-rec]="t.type === recommendedType && t.type !== activeType"
          [class.type-pill-active]="t.type === activeType"
          [disabled]="loading"
          (click)="pickType(t.type)">
          <span class="pill-icon">{{ typeIcon(t.type) }}</span>
          {{ t.label }}
          <span class="rec-badge" *ngIf="t.type === recommendedType && t.type !== activeType">Best</span>
        </button>
      </div>

      <button class="dismiss-btn" (click)="dismiss()" title="Dismiss">
        <svg viewBox="0 0 20 20" fill="currentColor"><path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z"/></svg>
      </button>
    </div>
  </div>

  <!-- ── Axis picker — shown once a chart type is active ── -->
  <div class="axis-row" *ngIf="activeConfig && cols">
    <div class="axis-group">
      <label class="axis-label">X axis</label>
      <select class="axis-select" [(ngModel)]="selectedX" (ngModelChange)="onAxisChange()">
        <option *ngFor="let c of axisXOptions" [value]="c">{{ colLabel(c) }}</option>
      </select>
    </div>

    <div class="axis-group axis-y">
      <label class="axis-label">Y axis</label>
      <div class="y-options">
        <label *ngFor="let c of cols.numerics" class="y-checkbox">
          <input type="checkbox"
            [checked]="selectedY.includes(c)"
            (change)="toggleY(c)"
            [disabled]="loading" />
          <span>{{ colLabel(c) }}</span>
        </label>
      </div>
    </div>

    <button class="btn-apply" [disabled]="loading" (click)="applyAxes()">
      <svg viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
        <path d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"/>
      </svg>
      Apply
    </button>
  </div>

  <!-- ── Error ── -->
  <div class="strip-error" *ngIf="error">{{ error }}</div>

</div>
  `,
  styles: [`
    :host { display: block; margin-top: 10px; }

    .suggestion-wrap {
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid rgba(27,67,255,0.25);
      background: linear-gradient(135deg, rgba(12,16,36,0.92) 0%, rgba(20,26,54,0.96) 100%);
    }

    /* ── strip ── */
    .suggestion-strip {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 9px 12px;
      flex-wrap: wrap;
    }
    .strip-left { display: flex; align-items: center; gap: 9px; }
    .strip-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

    .strip-icon {
      width: 28px; height: 28px;
      border-radius: 7px;
      background: rgba(27,67,255,0.18);
      border: 1px solid rgba(27,67,255,0.32);
      display: flex; align-items: center; justify-content: center;
      color: var(--blue-300); flex-shrink: 0;
    }
    .strip-icon svg { width: 14px; height: 14px; }

    .strip-text { display: flex; flex-direction: column; gap: 1px; }
    .strip-label { font-size: 12px; font-weight: 700; color: rgba(255,255,255,0.82); letter-spacing: 0.15px; }
    .strip-hint { font-size: 11px; color: rgba(255,255,255,0.35); }
    .strip-reason { color: rgba(255,255,255,0.45); font-style: italic; }

    /* ── type pills ── */
    .type-pills { display: flex; gap: 5px; flex-wrap: wrap; }

    .type-pill {
      display: flex; align-items: center; gap: 5px;
      padding: 5px 11px;
      border-radius: 7px;
      font-size: 11.5px; font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      color: rgba(255,255,255,0.5);
      white-space: nowrap;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .type-pill:hover:not(:disabled) {
      background: rgba(255,255,255,0.1);
      border-color: rgba(255,255,255,0.2);
      color: rgba(255,255,255,0.85);
    }
    .type-pill:disabled { opacity: 0.5; cursor: not-allowed; }

    .type-pill-rec {
      background: rgba(27,67,255,0.14);
      border-color: rgba(27,67,255,0.35);
      color: var(--blue-300);
    }
    .type-pill-rec:hover:not(:disabled) {
      background: rgba(27,67,255,0.24);
      border-color: rgba(27,67,255,0.55);
      color: var(--blue-200);
    }
    .type-pill-active {
      background: rgba(27,67,255,0.28) !important;
      border-color: rgba(27,67,255,0.6) !important;
      color: var(--blue-200) !important;
      font-weight: 600 !important;
    }

    .pill-icon { font-size: 12px; line-height: 1; }

    .rec-badge {
      padding: 1px 5px; border-radius: 20px;
      font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px;
      background: rgba(27,67,255,0.20); color: var(--blue-300);
      border: 1px solid rgba(27,67,255,0.35);
    }

    /* ── dismiss ── */
    .dismiss-btn {
      width: 24px; height: 24px; border-radius: 5px;
      background: transparent; border: none;
      color: rgba(255,255,255,0.22); cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      padding: 0; transition: all 0.13s; flex-shrink: 0;
    }
    .dismiss-btn svg { width: 13px; height: 13px; }
    .dismiss-btn:hover { color: rgba(255,255,255,0.55); background: rgba(255,255,255,0.06); }

    /* ── axis row ── */
    .axis-row {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 12px 10px;
      border-top: 1px solid rgba(255,255,255,0.06);
      background: rgba(0,0,0,0.12);
      flex-wrap: wrap;
    }

    .axis-group { display: flex; align-items: center; gap: 7px; }
    .axis-y { flex-wrap: wrap; }

    .axis-label {
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.6px; color: rgba(255,255,255,0.3);
      white-space: nowrap;
    }

    .axis-select {
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11.5px;
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.12);
      color: rgba(255,255,255,0.72);
      outline: none; cursor: pointer;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .axis-select:focus { border-color: rgba(27,67,255,0.50); }
    .axis-select option { background: #1c2135; color: #e2e8f0; }

    .y-options { display: flex; gap: 8px; flex-wrap: wrap; }
    .y-checkbox {
      display: flex; align-items: center; gap: 4px;
      font-size: 11.5px; color: rgba(255,255,255,0.6);
      cursor: pointer;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .y-checkbox input { accent-color: var(--accent); cursor: pointer; }
    .y-checkbox span { user-select: none; }

    .btn-apply {
      display: flex; align-items: center; gap: 5px;
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 11.5px; font-weight: 600;
      background: rgba(27,67,255,0.20);
      border: 1px solid rgba(27,67,255,0.40);
      color: var(--blue-300); cursor: pointer;
      transition: all 0.14s;
      font-family: var(--font);
      margin-left: auto;
    }
    .btn-apply:hover:not(:disabled) { background: rgba(27,67,255,0.32); color: var(--blue-200); }
    .btn-apply:disabled { opacity: 0.4; cursor: not-allowed; }

    /* ── error ── */
    .strip-error {
      padding: 6px 12px 8px;
      font-size: 11px; color: #fca5a5;
      border-top: 1px solid rgba(239,68,68,0.2);
    }
  `],
})
export class ChartSuggestionComponent implements OnChanges {
  /** session_id from the parent chat message */
  @Input() sessionId!: string;

  /**
   * Seed data from the chat response — the `__CHART_AVAILABLE__` signal
   * is parsed by the parent into { recommended_type, reason }.
   * If the parent already has a full ChartConfig (from chart fast-path),
   * pass it as initialConfig and the component skips the first API call.
   */
  @Input() recommendedType: string = 'bar';
  @Input() recommendedReason: string = '';
  @Input() initialConfig: ChartConfig | null = null;

  /** Emits the full ChartConfig whenever the user selects/changes a chart */
  @Output() chartChosen = new EventEmitter<ChartConfig>();
  @Output() chartDismissed = new EventEmitter<void>();

  private http = inject(HttpClient);
  private base = '/api';

  dismissed    = false;
  loading      = false;
  error        = '';
  activeConfig: ChartConfig | null = null;
  activeReason = '';
  activeType: string | null = null;
  cols: ColumnMeta | null = null;

  // axis state
  selectedX  = '';
  selectedY: string[] = [];

  // all type pills = recommended + alternatives
  allTypes: Array<{ type: string; label: string }> = [];

  private readonly LABELS: Record<string, string> = {
    bar: 'Bar', horizontal_bar: 'H-Bar', stacked_bar: 'Stacked',
    line: 'Line', area: 'Area', pie: 'Pie', doughnut: 'Doughnut',
    scatter: 'Scatter', radar: 'Radar', heatmap: 'Heatmap',
    waterfall: 'Waterfall', gauge: 'Gauge',
  };
  private readonly ICONS: Record<string, string> = {
    bar: '▊', horizontal_bar: '▬', stacked_bar: '▨',
    line: '↗', area: '◭', pie: '◕', doughnut: '◎',
    scatter: '⠿', radar: '⬡', heatmap: '⊞', waterfall: '▄▆█', gauge: '◉',
  };

  typeLabel(t: string): string { return this.LABELS[t] ?? t.replace(/_/g, ' '); }
  typeIcon(t: string): string  { return this.ICONS[t] ?? '◉'; }
  colLabel(c: string): string  { return c.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()); }

  get axisXOptions(): string[] {
    if (!this.cols) return [];
    return [...this.cols.categories, ...this.cols.dates];
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['initialConfig'] && this.initialConfig) {
      this._applyConfig(this.initialConfig);
      this._loadColumns();
      return;
    }
    if ((changes['sessionId'] || changes['recommendedType']) && this.sessionId) {
      this._buildPills([]);   // start with just recommended, expand after first call
      this._loadColumns();
    }
  }

  // ── User picks a chart type pill ──────────────────────────────────────────
  pickType(type: string): void {
    if (this.loading) return;

    // If we already have this alt rendered by a previous /api/chart call, reuse it
    if (this.activeConfig && type !== this.activeConfig.chart_type) {
      const existingAlt = this.activeConfig.alternatives?.find(a => a.chart_type === type);
      if (existingAlt) {
        this.activeType = type;
        this.activeReason = existingAlt.label;
        // Build a synthetic ChartConfig so chart-display can render it
        const synth: ChartConfig = {
          ...this.activeConfig,
          chart_type: type,
          config: existingAlt.config,
          reason: existingAlt.label,
          conflict: false,
          requested_type: null,
          requested_config: null,
        };
        this.chartChosen.emit(synth);
        return;
      }
    }

    // Otherwise call /api/chart
    this._fetchChart({ session_id: this.sessionId, chart_type: type });
  }

  // ── Axis change handlers ──────────────────────────────────────────────────
  toggleY(col: string): void {
    const idx = this.selectedY.indexOf(col);
    if (idx === -1) this.selectedY = [...this.selectedY, col];
    else            this.selectedY = this.selectedY.filter(c => c !== col);
  }

  onAxisChange(): void {
    // debounce: wait for blur/apply button, not immediate
  }

  applyAxes(): void {
    if (!this.activeType) return;
    this._fetchChart({
      session_id: this.sessionId,
      chart_type: this.activeType,
      x_col:  this.selectedX || null,
      y_cols: this.selectedY.length ? this.selectedY : null,
    });
  }

  dismiss(): void {
    this.dismissed = true;
    this.chartDismissed.emit();
  }

  // ── Private ───────────────────────────────────────────────────────────────
  private _fetchChart(req: {
    session_id: string; chart_type?: string | null;
    x_col?: string | null; y_cols?: string[] | null;
  }): void {
    this.loading = true;
    this.error   = '';
    this.http.post<ChartConfig>(`${this.base}/chart`, req).subscribe({
      next: cfg => {
        this.loading = false;
        this._applyConfig(cfg);
        this.chartChosen.emit(cfg);
      },
      error: err => {
        this.loading = false;
        this.error = err?.error?.detail ?? 'Could not build chart.';
      },
    });
  }

  private _loadColumns(): void {
    if (!this.sessionId) return;
    this.http.get<ColumnMeta>(`${this.base}/columns/${this.sessionId}`).subscribe({
      next: cols => {
        this.cols = cols;
        // Set sensible defaults for axis pickers
        if (cols.categories.length)  this.selectedX = cols.categories[0];
        else if (cols.dates.length)  this.selectedX = cols.dates[0];
        if (cols.numerics.length)    this.selectedY = [cols.numerics[0]];
      },
      error: () => { /* columns endpoint failure is non-fatal */ },
    });
  }

  private _applyConfig(cfg: ChartConfig): void {
    this.activeConfig = cfg;
    this.activeType   = cfg.chart_type;
    this.activeReason = cfg.reason;
    this._buildPills(cfg.alternatives ?? []);

    // Sync axis pickers to what the backend actually chose
    if (cfg.config) {
      const xData = cfg.config.xAxis?.data;
      // don't override user selection; only set if pickers are empty
      if (!this.selectedX && this.cols) {
        if (this.cols.categories.length) this.selectedX = this.cols.categories[0];
      }
    }
  }

  private _buildPills(alts: Array<{ chart_type: string; label: string; config: any }>): void {
    const seen = new Set<string>();
    const pills: Array<{ type: string; label: string }> = [];

    // Recommended first
    seen.add(this.recommendedType);
    pills.push({ type: this.recommendedType, label: this.typeLabel(this.recommendedType) });

    // Then backend alternatives
    for (const a of alts) {
      if (!seen.has(a.chart_type)) {
        seen.add(a.chart_type);
        pills.push({ type: a.chart_type, label: a.label });
      }
    }

    this.allTypes = pills;
  }
}