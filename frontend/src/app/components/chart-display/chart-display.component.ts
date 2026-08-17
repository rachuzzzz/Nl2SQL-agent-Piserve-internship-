import {
  Component, Input, Output, EventEmitter, OnInit, OnDestroy, OnChanges,
  AfterViewChecked, ViewChild, ElementRef, SimpleChanges, NgZone, inject,
  ChangeDetectorRef, ChangeDetectionStrategy, HostBinding
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { ChartConfig, ChartAlt, ColumnMeta } from '../../models/chat.model';
import { resolveSubject } from '../../utils/subject';
import * as echarts from 'echarts';

@Component({
  selector: 'app-chart-display',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="chart-backdrop" *ngIf="isFullscreen && !noOverlay"
         [style.top.px]="fsBounds?.top"
         [style.left.px]="fsBounds?.left"
         [style.right.px]="fsBounds?.right"
         [style.bottom.px]="fsBounds?.bottom"
         (click)="toggleFullscreen()"></div>

    <div class="chart-placeholder" *ngIf="isFullscreen && !noOverlay">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
      <span>Chart is expanded in a separate view</span>
    </div>

    <div class="chart-card" [class.fullscreen]="isFullscreen && !noOverlay"
         [style.top.px]="isFullscreen && !noOverlay && fsBounds ? fsBounds.top    : null"
         [style.left.px]="isFullscreen && !noOverlay && fsBounds ? fsBounds.left   : null"
         [style.right.px]="isFullscreen && !noOverlay && fsBounds ? fsBounds.right  : null"
         [style.bottom.px]="isFullscreen && !noOverlay && fsBounds ? fsBounds.bottom : null">
      <!-- ── Conflict banner ── -->
      <div class="suggestion-bar" *ngIf="showConflict">
        <span class="suggestion-text">
          I'd suggest a <strong>{{ typeLabel(cfg.chart_type) }}</strong> here —
          <span class="suggestion-reason">{{ cfg.reason }}</span>
        </span>
        <div class="suggestion-actions">
          <button class="btn-accept" (click)="useRecommended()">Use {{ typeLabel(cfg.chart_type) }}</button>
          <button class="btn-keep"   (click)="useRequested()">Keep {{ typeLabel(cfg.requested_type) }}</button>
        </div>
      </div>

      <!-- ── Top-N truncation banner — drill down to the full dataset ── -->
      <div class="suggestion-bar view-all-bar" *ngIf="!showConflict && cfg.can_view_all && !resolvedViewAll">
        <span class="suggestion-text">
          Showing top <strong>{{ cfg.rows_shown }}</strong> of <strong>{{ cfg.rows_total }}</strong> categories
          <span class="suggestion-reason" *ngIf="viewAllError">{{ viewAllError }}</span>
        </span>
        <div class="suggestion-actions">
          <button class="btn-keep" [disabled]="viewAllLoading" (click)="viewAll()">
            {{ viewAllLoading ? 'Loading…' : 'View all' }}
          </button>
        </div>
      </div>

      <!-- ── Header ── -->
      <div class="chart-header" *ngIf="activeConfig">
        <!-- Left: icon + label + confidence badge -->
        <div class="chart-header-left">
          <div class="chart-icon">
            <ng-container [ngSwitch]="iconGroup(activeType)">
              <svg *ngSwitchCase="'bar'"       viewBox="0 0 20 20" fill="currentColor"><path d="M12 3a1 1 0 00-1 1v12a1 1 0 102 0V4a1 1 0 00-1-1zM8 7a1 1 0 00-1 1v8a1 1 0 102 0V8a1 1 0 00-1-1zM4 11a1 1 0 00-1 1v4a1 1 0 102 0v-4a1 1 0 00-1-1zM16 9a1 1 0 00-1 1v6a1 1 0 102 0v-6a1 1 0 00-1-1z"/></svg>
              <svg *ngSwitchCase="'hbar'"      viewBox="0 0 20 20" fill="currentColor"><path d="M3 5a1 1 0 011-1h8a1 1 0 010 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 010 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h5a1 1 0 010 2H4a1 1 0 01-1-1z"/></svg>
              <svg *ngSwitchCase="'line'"      viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 0l-2 2a1 1 0 101.414 1.414L8 10.414l1.293 1.293a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>
              <svg *ngSwitchCase="'pie'"       viewBox="0 0 20 20" fill="currentColor"><path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z"/><path d="M12 2.252A8.014 8.014 0 0117.748 8H12V2.252z"/></svg>
              <svg *ngSwitchCase="'scatter'"   viewBox="0 0 20 20" fill="currentColor"><circle cx="4" cy="15" r="2"/><circle cx="9" cy="9" r="2"/><circle cx="14" cy="5" r="2"/></svg>
              <svg *ngSwitchCase="'radar'"     viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 2a8 8 0 100 16A8 8 0 0010 2zm0 2a6 6 0 110 12A6 6 0 0110 4z" clip-rule="evenodd"/></svg>
              <svg *ngSwitchCase="'heatmap'"   viewBox="0 0 20 20" fill="currentColor"><path d="M5 3h2v2H5V3zm4 0h2v2H9V3zm4 0h2v2h-2V3zM5 7h2v2H5V7zm4 0h2v2H9V7zm4 0h2v2h-2V7zM5 11h2v2H5v-2zm4 0h2v2H9v-2zm4 0h2v2h-2v-2z"/></svg>
              <svg *ngSwitchCase="'calendar'"  viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clip-rule="evenodd"/></svg>
              <svg *ngSwitchCase="'waterfall'" viewBox="0 0 20 20" fill="currentColor"><path d="M4 14h2v2H4v-2zM7 10h2v6H7v-6zM10 8h2v8h-2V8zM13 5h2v11h-2V5z"/></svg>
              <svg *ngSwitchCase="'funnel'"    viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3 3a1 1 0 011-1h12a1 1 0 011 1v3a1 1 0 01-.293.707L12 11.414V15a1 1 0 01-.293.707l-2 2A1 1 0 018 17v-5.586L3.293 6.707A1 1 0 013 6V3z" clip-rule="evenodd"/></svg>
              <svg *ngSwitchDefault            viewBox="0 0 20 20" fill="currentColor"><path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5z"/></svg>
            </ng-container>
          </div>
          <span class="chart-type-label">{{ typeLabel(activeType) }}</span>
          <span class="chart-confidence" [class]="'conf-' + cfg.confidence"
            *ngIf="cfg.confidence && cfg.confidence !== 'low'">
            {{ cfg.confidence === 'high' ? 'Best fit' : 'Good fit' }}
          </span>
        </div>

        <!-- Right: type switcher + axis toggle + zoom -->
        <div class="chart-header-right">
          <!-- Control mode header bar: axis button only (switcher moved to footer) -->
          <ng-container *ngIf="mode === 'control'">
            <button class="btn-ctrl btn-axis"
              [class.active]="showAxisPicker"
              [disabled]="!cols"
              (click)="showAxisPicker = !showAxisPicker"
              title="Configure axes">
              <svg viewBox="0 0 16 16" fill="currentColor"><path d="M1 1v14h14v-2H3V1H1zm3 10l3-4 2 2 3-5 2 3v4H4z"/></svg>
              <span>Axes</span>
            </button>
          </ng-container>

          <!-- Zoom controls -->
          <div class="zoom-controls" *ngIf="isZoomable">
            <button class="btn-ctrl" title="Zoom in"    (click)="zoomIn()"><svg viewBox="0 0 16 16" fill="currentColor"><path d="M6.5 1a5.5 5.5 0 104.396 8.803l3.85 3.85a.75.75 0 001.061-1.06l-3.85-3.85A5.5 5.5 0 006.5 1zm0 1.5a4 4 0 110 8 4 4 0 010-8zm0 1.75a.75.75 0 01.75.75v1h1a.75.75 0 010 1.5H7.25v1a.75.75 0 01-1.5 0V7.5h-1a.75.75 0 010-1.5h1V5a.75.75 0 01.75-.75z"/></svg></button>
            <button class="btn-ctrl" title="Zoom out"   (click)="zoomOut()"><svg viewBox="0 0 16 16" fill="currentColor"><path d="M6.5 1a5.5 5.5 0 104.396 8.803l3.85 3.85a.75.75 0 001.061-1.06l-3.85-3.85A5.5 5.5 0 006.5 1zm0 1.5a4 4 0 110 8 4 4 0 010-8zm-2 3.75a.75.75 0 01.75-.75h2.5a.75.75 0 010 1.5h-2.5a.75.75 0 01-.75-.75z"/></svg></button>
            <button class="btn-ctrl" title="Reset zoom" (click)="resetZoom()"><svg viewBox="0 0 16 16" fill="currentColor"><path fill-rule="evenodd" d="M2.5 7.5a5 5 0 119.316 2.537.75.75 0 001.376.588A6.5 6.5 0 102.25 8.37V7a.75.75 0 00-1.5 0v2.5a.75.75 0 00.75.75H4a.75.75 0 000-1.5H2.5V7.5z" clip-rule="evenodd"/></svg></button>
          </div>
          <!-- Fullscreen control — hidden when parent handles expand (noOverlay mode) -->
          <button *ngIf="!noOverlay" class="btn-ctrl" title="Toggle Fullscreen" (click)="toggleFullscreen()">
            <svg *ngIf="!isFullscreen" viewBox="0 0 16 16" fill="currentColor"><path d="M1.5 1.5v4h1.5v-2.5h2.5v-1.5h-4zm4 13h-2.5v-2.5h-1.5v4h4v-1.5zm9-9h-1.5v2.5h-2.5v1.5h4v-4zm-4 7h2.5v-2.5h1.5v4h-4v-1.5z"/></svg>
            <svg *ngIf="isFullscreen" viewBox="0 0 16 16" fill="currentColor"><path d="M5.5 5.5v-4h-1.5v2.5h-2.5v1.5h4zm-4 5h2.5v2.5h1.5v-4h-4v1.5zm9-5h1.5v-2.5h2.5v-1.5h-4v4zm4 5h-2.5v2.5h-1.5v-4h4v1.5z"/></svg>
          </button>
        </div>
      </div>

      <!-- ── Axis picker panel (collapsed by default, open in control mode) ── -->
      <div class="axis-panel" *ngIf="showAxisPicker && activeConfig">
        <div class="axis-row" *ngIf="cols; else colsLoading">
          <div class="axis-group">
            <label class="axis-label">X axis</label>
            <select class="axis-select" [(ngModel)]="selectedX">
              <option *ngFor="let c of axisXOptions" [value]="c">{{ colLabel(c) }}</option>
            </select>
          </div>
          <div class="axis-group">
            <label class="axis-label">Y axis</label>
            <div class="y-options">
              <label *ngFor="let c of cols.numerics" class="y-checkbox">
                <input type="checkbox"
                  [checked]="selectedY.includes(c)"
                  (change)="toggleY(c)"
                  [disabled]="axisLoading" />
                <span>{{ colLabel(c) }}</span>
              </label>
            </div>
          </div>
          <button class="btn-apply" [disabled]="axisLoading" (click)="applyAxes()">
            <svg viewBox="0 0 16 16" fill="currentColor" width="11" height="11">
              <path d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"/>
            </svg>
            {{ axisLoading ? 'Updating…' : 'Apply' }}
          </button>
        </div>
        <div class="axis-error" *ngIf="axisError">{{ axisError }}</div>
        <ng-template #colsLoading>
          <span class="axis-hint">{{ colsLoadError ? 'Column data unavailable — re-run your query.' : 'Loading columns…' }}</span>
        </ng-template>
      </div>

      <!-- ── Chart canvas ── -->
      <div class="chart-body" *ngIf="activeConfig && !renderError">
        <div #chartDiv [style.height]="chartHeight" class="echarts-host"></div>
      </div>

      <!-- ── Render-error fallback — never leave the user staring at a blank div ── -->
      <div class="chart-render-error" *ngIf="activeConfig && renderError">
        <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.72-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>
        <span>Couldn't render this {{ typeLabel(activeType) }} — the underlying data table still has the full result.</span>
      </div>

      <!-- ── Chart-type switcher footer (control mode only) ── -->
      <div class="alt-footer" *ngIf="mode === 'control' && availableAlts.length > 0 && activeConfig">
        <span class="alt-footer-label">View as</span>
        <div class="alt-switcher">
          <button *ngFor="let alt of availableAlts" class="btn-alt"
            [class.active]="activeType === alt.chart_type"
            (click)="switchToAlt(alt)" [title]="alt.label">
            <ng-container [ngSwitch]="iconGroup(alt.chart_type)">
              <svg *ngSwitchCase="'bar'"      viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path d="M9.5 2a.5.5 0 00-.5.5v11a.5.5 0 001 0v-11a.5.5 0 00-.5-.5zM6.5 5a.5.5 0 00-.5.5v8a.5.5 0 001 0v-8a.5.5 0 00-.5-.5zM3.5 8a.5.5 0 00-.5.5v5a.5.5 0 001 0v-5a.5.5 0 00-.5-.5z"/></svg>
              <svg *ngSwitchCase="'hbar'"     viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path d="M2 4h6v2H2V4zm0 3h10v2H2V7zm0 3h4v2H2v-2z"/></svg>
              <svg *ngSwitchCase="'line'"     viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path fill-rule="evenodd" d="M1 13l4-4 3 2 4-5 3 2-1 1-2-1-4 5-3-2-3 3-1-1z" clip-rule="evenodd"/></svg>
              <svg *ngSwitchCase="'pie'"      viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zM0 8a8 8 0 1116 0A8 8 0 010 8z"/><path d="M8 1v7h7A7 7 0 008 1z"/></svg>
              <svg *ngSwitchCase="'scatter'"  viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><circle cx="3" cy="12" r="1.5"/><circle cx="7" cy="7" r="1.5"/><circle cx="12" cy="4" r="1.5"/></svg>
              <svg *ngSwitchCase="'radar'"    viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path d="M8 2l1.5 4.5H14l-3.7 2.7 1.4 4.3L8 11 4.3 13.5l1.4-4.3L2 6.5h4.5L8 2z"/></svg>
              <svg *ngSwitchCase="'heatmap'"  viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path d="M1 1h4v4H1V1zm5 0h4v4H6V1zm5 0h4v4h-4V1zM1 6h4v4H1V6zm5 0h4v4H6V6zm5 0h4v4h-4V6z"/></svg>
              <svg *ngSwitchCase="'calendar'" viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path fill-rule="evenodd" d="M4 1a.5.5 0 01.5.5V2h7v-.5a.5.5 0 011 0V2h.5A1.5 1.5 0 0114.5 3.5v10A1.5 1.5 0 0113 15H3a1.5 1.5 0 01-1.5-1.5v-10A1.5 1.5 0 013 2h.5v-.5A.5.5 0 014 1zM2 5.5v8A.5.5 0 003 14h10a.5.5 0 00.5-.5V5.5H2zm2 1h2v2H4v-2zm3 0h2v2H7v-2zm3 0h2v2h-2v-2zM4 9h2v2H4V9zm3 0h2v2H7V9z" clip-rule="evenodd"/></svg>
              <svg *ngSwitchCase="'waterfall'" viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path d="M1 12h2v2H1v-2zM4 8h2v6H4V8zM7 6h2v8H7V6zM10 3h2v11h-2V3z"/></svg>
              <svg *ngSwitchCase="'funnel'"    viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path fill-rule="evenodd" d="M2.5 3A1.5 1.5 0 014 1.5h8A1.5 1.5 0 0113.5 3v1.5a1.5 1.5 0 01-.44 1.06L10 8.62v4.88a1.5 1.5 0 01-2.22 1.31l-1.5-.75A1.5 1.5 0 015.5 12.7V8.62L2.44 5.56A1.5 1.5 0 012.5 4.5V3z" clip-rule="evenodd"/></svg>
              <svg *ngSwitchDefault            viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path d="M8 1a7 7 0 100 14A7 7 0 008 1z"/></svg>
            </ng-container>
            <span>{{ alt.label }}</span>
          </button>
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host { display: block; margin-top: 12px; margin-bottom: 12px; }
    
    .chart-backdrop {
      position: fixed; z-index: 9998;
      /* top/left/right/bottom are set via [style] bindings to match .chat-main */
      background: rgba(15, 23, 42, 0.55); backdrop-filter: blur(3px);
      animation: fadeIn 0.2s ease-out forwards;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

    .chart-placeholder {
      display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px;
      height: 320px; border: 2px dashed var(--chart-border); border-radius: 16px;
      background: var(--surface-2); color: var(--chart-text-muted);
      font-size: 13px; font-style: italic;
    }
    .chart-placeholder svg { width: 28px; height: 28px; opacity: 0.4; }

    .chart-card { border-radius: 16px; overflow: hidden; background: var(--chart-card-bg); border: 1px solid var(--chart-border); box-shadow: 0 10px 30px -10px rgba(0,0,0,0.08); backdrop-filter: blur(8px); transition: box-shadow 0.3s ease; display: flex; flex-direction: column; animation: cardReveal 0.45s cubic-bezier(0.16, 1, 0.3, 1) both; }
    @keyframes cardReveal {
      from { opacity: 0; transform: translateY(14px) scale(0.97); }
      to   { opacity: 1; transform: none; }
    }
    @media (prefers-reduced-motion: reduce) { .chart-card { animation: none; } }
    .chart-card:hover { box-shadow: 0 12px 40px -12px rgba(0,0,0,0.12); }

    .chart-card.fullscreen {
      animation: none;
      position: fixed !important;
      /* top/left/right/bottom are set via [style] bindings — see toggleFullscreen() */
      width: auto !important; height: auto !important;
      z-index: 9999 !important; margin: 0 !important;
      border-radius: 12px; box-shadow: 0 20px 48px -12px rgba(0,0,0,0.45);
      background: var(--surface);
    }
    .chart-card.fullscreen .chart-body { flex: 1; min-height: 0; padding: 24px; }
    .chart-card.fullscreen .echarts-host { height: 100% !important; }

    .suggestion-bar { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; padding: 12px 18px; background: var(--chart-suggestion-bg); border-bottom: 1px solid var(--chart-border); backdrop-filter: blur(4px); }
    .suggestion-text { font-size: 13px; color: var(--chart-text); font-family: ui-sans-serif,system-ui,sans-serif; }
    .suggestion-text strong { color: var(--accent); font-weight: 600; }
    .suggestion-reason { color: var(--chart-text-muted); font-size: 12px; font-style: italic; }
    .suggestion-actions { display: flex; gap: 8px; }
    .btn-accept { padding: 6px 16px; border-radius: 8px; font-size: 12px; font-weight: 600; background: var(--accent); border: none; color: #fff; cursor: pointer; transition: background 0.2s, transform 0.1s; }
    .btn-accept:hover { background: var(--blue-700); transform: translateY(-1px); }
    .btn-accept:active { transform: translateY(0); }
    .btn-keep { padding: 6px 16px; border-radius: 8px; font-size: 12px; background: var(--chart-card-bg); border: 1px solid #cbd0da; color: #4b5563; cursor: pointer; transition: all 0.2s; backdrop-filter: blur(2px); }
    .btn-keep:hover { border-color: #9aa0ad; color: #111827; background: var(--surface); }

    .chart-header { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; gap: 10px; background: var(--chart-header-bg); border-bottom: 1px solid var(--chart-border); backdrop-filter: blur(4px); }
    .chart-header-left { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
    .chart-header-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .chart-icon { width: 30px; height: 30px; border-radius: 8px; background: var(--blue-50); border: 1px solid var(--blue-100); display: flex; align-items: center; justify-content: center; color: var(--accent); box-shadow: 0 2px 5px rgba(27,67,255,0.10); }
    .chart-icon svg { width: 15px; height: 15px; }
    .chart-type-label { font-size: 13px; font-weight: 600; color: var(--chart-text); font-family: ui-sans-serif,system-ui,sans-serif; letter-spacing: 0.2px; }
    .chart-confidence { font-size: 10.5px; font-weight: 600; padding: 2px 8px; border-radius: 20px; font-family: ui-sans-serif,system-ui,sans-serif; text-transform: uppercase; letter-spacing: 0.5px; }
    .conf-high   { background: rgba(220, 252, 231, 0.8); color: #166534; border: 1px solid rgba(134, 239, 172, 0.5); }
    .conf-medium { background: rgba(254, 249, 195, 0.8); color: #854d0e; border: 1px solid rgba(253, 224, 71, 0.5); }

    .alt-switcher { display: flex; gap: 4px; background: var(--chart-axis-bg); border: 1px solid rgba(228, 231, 237, 0.8); border-radius: 10px; padding: 4px; backdrop-filter: blur(2px); }
    .btn-alt { display: flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 6px; border: none; background: transparent; color: var(--chart-text-muted); font-size: 11.5px; cursor: pointer; font-family: ui-sans-serif,system-ui,sans-serif; transition: all 0.15s ease; white-space: nowrap; }
    .btn-alt:hover { background: var(--chart-card-bg); color: var(--chart-text); box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
    .btn-alt.active { background: var(--surface); color: var(--accent); font-weight: 600; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }

    .zoom-controls { display: flex; gap: 4px; }
    .btn-ctrl { width: 30px; height: 30px; border-radius: 8px; background: var(--chart-card-bg); border: 1px solid rgba(228, 231, 237, 0.8); color: var(--chart-text-muted); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.15s ease; padding: 0; backdrop-filter: blur(2px); }
    .btn-ctrl svg { width: 14px; height: 14px; }
    .btn-ctrl:hover { background: var(--surface); border-color: var(--blue-100); color: var(--accent); box-shadow: 0 2px 6px rgba(27,67,255,0.08); transform: translateY(-1px); }

    /* Axes button — wider to fit label */
    .btn-axis { width: auto; padding: 0 10px; gap: 5px; font-size: 11.5px; font-family: var(--font); font-weight: 500; }
    .btn-axis span { line-height: 1; }
    .btn-axis.active { background: var(--surface); border-color: var(--blue-100); color: var(--accent); box-shadow: 0 2px 8px rgba(27,67,255,0.10); }

    /* ── Axis panel ── */
    .axis-panel { padding: 12px 16px; background: var(--chart-axis-bg); border-bottom: 1px solid var(--chart-border); backdrop-filter: blur(8px); }
    .axis-row { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
    .axis-group { display: flex; align-items: center; gap: 8px; background: var(--surface); padding: 6px 12px; border-radius: 10px; border: 1px solid #e4e7ed; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
    .axis-label { font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--chart-text-muted); white-space: nowrap; }
    .axis-select { padding: 4px 8px; border-radius: 6px; font-size: 12px; background: transparent; border: none; color: #111827; outline: none; cursor: pointer; font-family: ui-sans-serif,system-ui,sans-serif; font-weight: 500; }
    .axis-select:focus { color: var(--accent); }
    .y-options { display: flex; gap: 12px; flex-wrap: wrap; }
    .y-checkbox { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--chart-text); cursor: pointer; font-family: var(--font); font-weight: 500; }
    .y-checkbox input { accent-color: var(--accent); cursor: pointer; width: 14px; height: 14px; }
    .y-checkbox span { user-select: none; }
    .btn-apply { display: flex; align-items: center; gap: 6px; padding: 8px 16px; border-radius: 8px; font-size: 12px; font-weight: 600; background: var(--accent); border: none; color: #fff; cursor: pointer; margin-left: auto; transition: all 0.2s ease; font-family: var(--font); box-shadow: 0 2px 6px rgba(27,67,255,0.22); }
    .btn-apply:hover:not(:disabled) { background: var(--blue-700); box-shadow: 0 4px 12px rgba(27,67,255,0.32); transform: translateY(-1px); }
    .btn-apply:disabled { opacity: 0.5; cursor: not-allowed; box-shadow: none; transform: none; }
    .axis-error { margin-top: 6px; font-size: 11.5px; color: #dc2626; font-weight: 500; }
    .axis-hint  { font-size: 11.5px; color: var(--chart-text-muted); font-style: italic; }

    .chart-body { padding: 16px 16px 20px; background: transparent; min-width: 0; overflow: hidden; }
    .echarts-host { width: 100%; display: block; overflow: hidden; }

    .chart-render-error { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; padding: 40px 24px; min-height: 200px; color: var(--chart-text-muted); font-size: 13px; text-align: center; font-family: ui-sans-serif,system-ui,sans-serif; }
    .chart-render-error svg { width: 26px; height: 26px; color: #dc2626; opacity: 0.75; }

    /* Alt-switcher footer — full-width row below the chart */
    .alt-footer { display: flex; align-items: center; gap: 10px; padding: 10px 16px; border-top: 1px solid var(--chart-border); background: var(--chart-axis-bg); flex-wrap: wrap; }
    .alt-footer-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; color: var(--chart-text-muted); white-space: nowrap; font-family: ui-sans-serif,system-ui,sans-serif; flex-shrink: 0; }
  `],
})
export class ChartDisplayComponent implements OnInit, OnChanges, OnDestroy, AfterViewChecked {
  @HostBinding('class.is-fullscreen') get isFullscreenClass() { return this.isFullscreen && !this.noOverlay; }
  @Input()  cfg!: ChartConfig;
  @Input()  sessionId: string = '';
  @Input()  mode: 'quick' | 'control' = 'quick';  // 'control' shows switcher + axis panel
  @Input()  noOverlay: boolean = false;  // when true, fullscreen emits event only; parent handles layout
  @Input()  resizeTrigger: number = 0;  // increment from parent to force chart.resize()
  @Input()  sourceQuestion?: string;    // the question that produced `cfg` — used to infer
                                         // drilldown intent (see resolveSubject() in utils/subject.ts)
  @Output() fullscreenChange = new EventEmitter<boolean>();
  @Output() drilldown = new EventEmitter<string>();  // emits message text to send, on segment click
  @ViewChild('chartDiv') chartDivRef!: ElementRef<HTMLDivElement>;

  private http   = inject(HttpClient);
  private zone   = inject(NgZone);
  private cdr    = inject(ChangeDetectorRef);
  private elRef  = inject(ElementRef);
  private base   = '/api';

  showConflict     = false;
  resolvedConflict = false;
  resolvedViewAll  = false;
  activeConfig: any = null;
  activeType: string | null = null;
  hasZoomed        = false;
  isFullscreen     = false;
  fsBounds: { top: number; left: number; right: number; bottom: number } | null = null;
  availableAlts: Array<{chart_type: string; label: string; config: any}> = [];

  // axis picker state
  showAxisPicker  = false;
  cols: ColumnMeta | null = null;
  colsLoadError   = false;
  selectedX       = '';
  selectedY: string[] = [];
  axisLoading     = false;
  axisError       = '';

  // Set when buildOption()/setOption() throws — e.g. a malformed value in the
  // ECharts config (the historical 'transparentaa' invalid-color case was one
  // instance of this whole failure class: one bad value can throw a canvas
  // DOMException that silently blanks the ENTIRE chart with no indication).
  // Shows a visible fallback instead of an empty div.
  renderError     = false;

  // Top-N truncation drill-down — see ChartConfig.rows_shown/rows_total/
  // can_view_all (schemas.py) and viewAll() below.
  viewAllLoading  = false;
  viewAllError    = '';

  private chart: echarts.ECharts | null = null;
  private needsRender = false;
  private resizeObs: ResizeObserver | null = null;
  private resizeTimer: any = null;
  private transitionEndCb: ((e: Event) => void) | null = null;
  private initTimers: any[] = [];
  private calZoom = 1;

  private readonly TYPE_LABELS: Record<string, string> = {
    bar:'Bar Chart', horizontal_bar:'Horizontal Bar', stacked_bar:'Stacked Bar',
    line:'Line Chart', area:'Area Chart', pie:'Pie Chart', doughnut:'Doughnut Chart',
    nightingale:'Nightingale Chart', scatter:'Scatter Plot', radar:'Radar Chart',
    heatmap:'Heatmap', calendar_heatmap:'Calendar Heatmap',
    waterfall:'Waterfall', funnel:'Funnel Chart', gauge:'Gauge',
  };

  typeLabel(t: string | null | undefined): string {
    return t ? (this.TYPE_LABELS[t] ?? t.replace(/_/g,' ').replace(/\b\w/g, l => l.toUpperCase())) : 'Chart';
  }
  iconGroup(t: string | null | undefined): string {
    if (!t) return 'default';
    if (t === 'horizontal_bar' || t === 'stacked_bar') return 'hbar';
    if (t === 'bar' || t === 'waterfall')              return 'bar';
    if (t === 'line' || t === 'area')                  return 'line';
    if (t === 'pie' || t === 'doughnut' || t === 'nightingale') return 'pie';
    if (t === 'scatter')          return 'scatter';
    if (t === 'radar')            return 'radar';
    if (t === 'heatmap')          return 'heatmap';
    if (t === 'calendar_heatmap') return 'calendar';
    if (t === 'waterfall')        return 'waterfall';
    if (t === 'funnel')           return 'funnel';
    return 'default';
  }
  colLabel(c: string): string { return c.replace(/_/g,' ').replace(/\b\w/g, l => l.toUpperCase()); }

  get isZoomable(): boolean {
    return ['bar','horizontal_bar','stacked_bar','line','area','scatter','calendar_heatmap'].includes(this.activeType ?? '');
  }
  get altType(): string | null {
    if (!this.cfg) return null;
    return this.activeType === this.cfg.chart_type ? this.cfg.requested_type : this.cfg.chart_type;
  }
  get axisXOptions(): string[] {
    if (!this.cols) return [];
    return [...this.cols.categories, ...this.cols.dates];
  }
  get chartHeight(): string {
    if (this.isFullscreen && !this.noOverlay) return '100%';
    if (!this.activeConfig) return '400px';
    if (this.activeType === 'horizontal_bar') {
      const yData = this.activeConfig?.yAxis?.data;
      const rows = Array.isArray(yData) ? yData.length : 8;
      return Math.min(720, Math.max(240, 60 + rows * 32)) + 'px';
    }
    if (this.activeType === 'heatmap') {
      const yData = this.activeConfig?.yAxis?.data;
      const rows = Array.isArray(yData) ? yData.length : 5;
      return Math.min(600, Math.max(280, 80 + rows * 42)) + 'px';
    }
    if (this.activeType === 'calendar_heatmap') {
      const range = this.activeConfig?.calendar?.range;
      const baseYearH = Math.round(190 * this.calZoom);
      if (Array.isArray(range) && range.length === 2) {
        const y0 = parseInt(String(range[0]).substring(0, 4), 10);
        const y1 = parseInt(String(range[1]).substring(0, 4), 10);
        const years = (isNaN(y0) || isNaN(y1)) ? 1 : Math.max(1, y1 - y0 + 1);
        return Math.max(240, baseYearH * years + 80) + 'px';
      }
      return Math.max(240, baseYearH + 80) + 'px';
    }
    if (this.activeType === 'radar')   return '420px';
    if (this.activeType === 'gauge')   return '320px';
    if (this.activeType === 'pie' || this.activeType === 'doughnut' || this.activeType === 'nightingale') return '360px';
    return '400px';
  }

  private themeObserver: MutationObserver | null = null;

  ngOnInit()  {
    this.init();
    this.themeObserver = new MutationObserver((mutations) => {
      mutations.forEach((m) => {
        if (m.attributeName === 'data-theme' && this.chartDivRef?.nativeElement && this.activeConfig) {
          this.renderChart();
        }
      });
    });
    this.themeObserver.observe(document.body, { attributes: true, attributeFilter: ['data-theme'] });
  }
  ngOnChanges(c: SimpleChanges) {
    if ((c['cfg'] || c['mode']) && !(c['cfg']?.firstChange && c['mode']?.firstChange)) {
      this.destroyChart();
      this.init();
    }
    if (c['resizeTrigger'] && !c['resizeTrigger'].firstChange) {
      this.zone.runOutsideAngular(() => {
        requestAnimationFrame(() => {
          this.chart?.resize();
          setTimeout(() => this.chart?.resize(), 50);
          setTimeout(() => this.chart?.resize(), 200);
          setTimeout(() => this.chart?.resize(), 500);
        });
      });
    }
  }

  private init() {
    if (!this.cfg) return;
    this.hasZoomed = false; this.resolvedConflict = false; this.resolvedViewAll = false;
    this.renderError = false; this.viewAllError = '';
    this.showAxisPicker = this.mode === 'control'; this.axisError = '';
    this.cols = null; this.colsLoadError = false;
    const altList = (this.cfg.alternatives ?? []).filter(a => a.chart_type !== this.cfg.chart_type).slice(0, 3);
    // Prepend the primary chart so the user can always switch back to it.
    // The switcher is only shown when there is at least one alternative.
    const primaryAlt: ChartAlt = {
      chart_type: this.cfg.chart_type ?? '',
      label: this.typeLabel(this.cfg.chart_type),
      config: this.cfg.config,
    };
    this.availableAlts = altList.length > 0 ? [primaryAlt, ...altList] : [];
    if (this.cfg.conflict) {
      this.showConflict = true; this.activeConfig = null; this.activeType = null;
    } else {
      this.showConflict  = false;
      this.activeConfig  = this.cfg.config;
      this.activeType    = this.cfg.chart_type;
      this.needsRender   = true;
    }
    // Load column metadata for the axis picker (non-blocking)
    if (this.sessionId) this._loadColumns();
  }

  useRecommended() { this.activeConfig = this.cfg.config; this.activeType = this.cfg.chart_type; this.showConflict = false; this.resolvedConflict = true; this.needsRender = true; }
  useRequested()   { this.activeConfig = this.cfg.requested_config ?? this.cfg.config; this.activeType = this.cfg.requested_type ?? this.cfg.chart_type; this.showConflict = false; this.resolvedConflict = true; this.needsRender = true; }

  switchToAlt(alt: ChartAlt) {
    this.destroyChart();
    this.activeConfig = alt.config; this.activeType = alt.chart_type; this.needsRender = true;
    this.cdr.markForCheck();
  }

  // ── Axis picker ───────────────────────────────────────────────────────────
  toggleY(col: string): void {
    const idx = this.selectedY.indexOf(col);
    if (idx === -1) this.selectedY = [...this.selectedY, col];
    else            this.selectedY = this.selectedY.filter(c => c !== col);
  }

  applyAxes(): void {
    if (!this.sessionId || this.axisLoading) return;
    this.axisLoading = true; this.axisError = '';
    this.http.post<ChartConfig>(`${this.base}/chart`, {
      session_id: this.sessionId,
      chart_type: this.activeType,
      x_col:  this.selectedX || null,
      y_cols: this.selectedY.length ? this.selectedY : null,
    }).subscribe({
      next: cfg => {
        this.axisLoading = false;
        this.destroyChart();
        this.activeConfig = cfg.config;
        this.activeType   = cfg.chart_type;
        const axisAltList = (cfg.alternatives ?? []).filter(a => a.chart_type !== cfg.chart_type).slice(0, 3);
        const axisPrimary: ChartAlt = { chart_type: cfg.chart_type ?? '', label: this.typeLabel(cfg.chart_type), config: cfg.config };
        this.availableAlts = axisAltList.length > 0 ? [axisPrimary, ...axisAltList] : [];
        this.needsRender  = true;
        this.cdr.markForCheck();
      },
      error: err => {
        this.axisLoading = false;
        this.axisError = err?.error?.detail ?? 'Could not update chart.';
        this.cdr.markForCheck();
      },
    });
  }

  // ── Top-N drill-down ─────────────────────────────────────────────────────
  viewAll(): void {
    if (!this.sessionId || this.viewAllLoading) return;
    this.viewAllLoading = true; this.viewAllError = '';
    this.http.post<ChartConfig>(`${this.base}/chart`, {
      session_id: this.sessionId,
      chart_type: this.activeType,
      view_all: true,
    }).subscribe({
      next: cfg => {
        this.viewAllLoading = false;
        this.resolvedViewAll = true;
        this.destroyChart();
        this.cfg           = cfg;
        this.activeConfig  = cfg.config;
        this.activeType    = cfg.chart_type;
        this.needsRender   = true;
        this.cdr.markForCheck();
      },
      error: err => {
        this.viewAllLoading = false;
        this.viewAllError = err?.error?.detail ?? 'Could not load the full dataset.';
        this.cdr.markForCheck();
      },
    });
  }

  private _loadColumns(): void {
    this.colsLoadError = false;
    this.http.get<ColumnMeta>(`${this.base}/columns/${this.sessionId}`).subscribe({
      next: cols => {
        this.cols = cols;
        if (cols.categories.length) this.selectedX = cols.categories[0];
        else if (cols.dates.length) this.selectedX = cols.dates[0];
        if (cols.numerics.length)   this.selectedY = [cols.numerics[0]];
        this.cdr.markForCheck();
      },
      error: () => {
        this.colsLoadError = true;
        this.cdr.markForCheck();
      },
    });
  }

  // ── Zoom ─────────────────────────────────────────────────────────────────
  private _zoom(factor: number) {
    if (!this.chart) return; this.hasZoomed = true;
    if (this.activeType === 'calendar_heatmap') {
      this.calZoom = Math.max(0.4, Math.min(3, this.calZoom * (factor < 1 ? 1.35 : 0.72)));
      this._applyCalendarCellSize();
      return;
    }
    const opt = this.chart.getOption() as any;
    const dz = (opt.dataZoom as any[])?.[0] ?? { start: 0, end: 100 };
    const mid = (dz.start + dz.end) / 2; const span = (dz.end - dz.start) * factor; const half = Math.min(span/2,50);
    const s = Math.max(0,mid-half), e = Math.min(100,mid+half);
    [0,1].forEach(i => this.chart!.dispatchAction({ type:'dataZoom', dataZoomIndex:i, start:s, end:e }));
  }
  zoomIn()    { this._zoom(0.55); }
  zoomOut()   { this._zoom(1.6); }
  resetZoom() {
    if (!this.chart) return; this.hasZoomed = false;
    if (this.activeType === 'calendar_heatmap') {
      this.calZoom = 1;
      this._applyCalendarCellSize();
      return;
    }
    [0,1].forEach(i => this.chart!.dispatchAction({ type:'dataZoom', dataZoomIndex:i, start:0, end:100 }));
  }
  /** Clicked a bar/pie/etc segment labeled `category` — drill into just
   * that category's rows, mirroring result-table.component.ts's onRowClick. */
  private onSegmentClick(category: string): void {
    const subject = resolveSubject(this.sourceQuestion);
    this.drilldown.emit(`show me the list of ${subject} where ${this.cfg?.x_label || 'category'} is ${category}`);
  }

  private _applyCalendarCellSize() {
    if (!this.chart) return;
    const size = Math.max(10, Math.round(20 * this.calZoom));
    const opt = this.chart.getOption() as any;
    const calCount = Array.isArray(opt.calendar) ? opt.calendar.length : 1;
    const cals = Array.from({ length: calCount }, () => ({ cellSize: [size, size] }));
    this.chart.setOption({ calendar: cals }, { notMerge: false });
    this.cdr.markForCheck();
    setTimeout(() => this.chart?.resize(), 50);
  }

  toggleFullscreen(): void {
    this.isFullscreen = !this.isFullscreen;

    if (!this.noOverlay && this.isFullscreen) {
      // Measure the nearest .chat-main ancestor so the expanded chart fills
      // exactly that area (bounded by header + sidebars), not the full viewport.
      const host: HTMLElement = this.elRef.nativeElement;
      const container: HTMLElement | null = host.closest('.chat-main');
      if (container) {
        const r = container.getBoundingClientRect();
        this.fsBounds = {
          top:    r.top,
          left:   r.left,
          right:  window.innerWidth  - r.right,
          bottom: window.innerHeight - r.bottom,
        };
      } else {
        // Fallback: small inset from viewport edges
        this.fsBounds = { top: 56, left: 8, right: 8, bottom: 0 };
      }
    } else {
      this.fsBounds = null;
    }

    this.fullscreenChange.emit(this.isFullscreen);
    this.cdr.markForCheck();
    setTimeout(() => { this.chart?.resize(); }, 120);
  }

  ngAfterViewChecked() {
    if (this.needsRender && this.chartDivRef?.nativeElement && this.activeConfig) {
      this.needsRender = false;
      // Defer into the browser's next animation frame so the layout pass has
      // already run. Without this, echarts.init() reads clientWidth = 0 because
      // the DOM was just mutated (by *ngIf) but the browser hasn't measured yet.
      this.zone.runOutsideAngular(() => {
        requestAnimationFrame(() => {
          if (this.chartDivRef?.nativeElement && this.activeConfig) {
            this.renderChart();
          }
        });
      });
    }
  }

  private renderChart() {
    this.destroyChart();
    this.renderError = false;
    const el = this.chartDivRef.nativeElement;
    const isDark = document.body.getAttribute('data-theme') === 'dark';

    // A single malformed value anywhere in the ECharts option (e.g. the
    // historical 'transparentaa' invalid-color case — see the waterfall
    // itemStyle guard in buildOption()) can throw during option-build or
    // setOption() and silently leave a blank canvas with no user-visible
    // indication. Catch here and show a visible fallback instead — this is
    // the general fix; per-case guards like the waterfall one stay in place
    // as defense-in-depth, not a replacement for this.
    let option: any;
    try {
      option = this.buildOption(this.activeConfig, this.activeType, isDark);
    } catch (e) {
      console.error('Chart option build failed', e);
      this.renderError = true;
      this.cdr.markForCheck();
      return;
    }

    this.zone.runOutsideAngular(() => {
      try {
        this.chart = echarts.init(el, isDark ? 'dark' : null, { renderer: 'canvas' });
        this.chart.setOption(option, { notMerge: true });
      } catch (e) {
        console.error('Chart render failed', e);
        this.chart?.dispose();
        this.chart = null;
        this.zone.run(() => { this.renderError = true; this.cdr.markForCheck(); });
        return;
      }
      this.chart.on('dataZoom', () => this.zone.run(() => this.hasZoomed = true));
      // Click a bar/pie/etc segment to drill into just that category —
      // params.name is the clicked category's axis/legend label for every
      // chart type this advisor produces (bar, pie, doughnut, line, ...).
      // Aggregate-only shapes (waterfall's synthetic Income/Cost/Profit
      // bars, gauge, heatmap cells with composite keys) still emit
      // something clickable here; harmless since it just becomes a
      // slightly odd follow-up question rather than a broken interaction.
      this.chart.on('click', (params: any) => {
        if (!params?.name) return;
        this.zone.run(() => this.onSegmentClick(String(params.name)));
      });
      if (this.activeType === 'calendar_heatmap') {
        el.addEventListener('wheel', (e: WheelEvent) => {
          e.preventDefault();
          this.zone.run(() => e.deltaY < 0 ? this.zoomIn() : this.zoomOut());
        }, { passive: false });
      }
      el.addEventListener('dblclick', () => this.zone.run(() => this.resetZoom()));
      this.resizeObs = new ResizeObserver(() => {
        this.chart?.resize();
        this._syncHbarXRotation();
        this._syncPieLegendSize();
        clearTimeout(this.resizeTimer);
        this.resizeTimer = setTimeout(() => {
          this.chart?.resize();
          this._syncHbarXRotation();
          this._syncPieLegendSize();
        }, 320);
      });
      this.resizeObs.observe(el);
      this.resizeObs.observe(this.elRef.nativeElement);

      // transitionend fires on document whenever a CSS transition completes.
      // The sidebars use `transition: width`, so this fires reliably when a
      // panel is opened or closed — even if the ResizeObserver misses it.
      this.transitionEndCb = (e: Event) => {
        if ((e as TransitionEvent).propertyName === 'width') {
          this.chart?.resize();
          this._syncHbarXRotation();
          this._syncPieLegendSize();
        }
      };
      document.addEventListener('transitionend', this.transitionEndCb);

      // Timed retries: guarantee the chart reaches its correct size on first
      // render and after any layout change, regardless of browser quirks.
      this.initTimers = [0, 150, 350, 700].map(ms =>
        setTimeout(() => this.chart?.resize(), ms)
      );
    });
  }

  private buildOption(raw: any, chartType: string | null, isDark: boolean): any {
    const opt: any = JSON.parse(JSON.stringify(raw));
    delete opt['_lib'];
    const isCartesian = !!(opt.xAxis || opt.yAxis);
    const isRadar     = !isCartesian && opt.series?.some((s: any) => s.type === 'radar');
    const isCalendar  = opt.calendar != null || opt.series?.some((s: any) => s.coordinateSystem === 'calendar');
    const isHeatmap   = !isCalendar && opt.series?.some((s: any) => s.type === 'heatmap');
    const isWaterfall = chartType === 'waterfall';
    const seriesCount = opt.series?.length ?? 0;
    // Vivid, high-contrast professional palette
    const PALETTE = ['#1B43FF','#e07a6e','#d97706','#7c3aed','#059669','#0891b2','#5C73FF','#e11d48'];
    opt.color = PALETTE; opt.backgroundColor = 'transparent';

    const tTextColor  = isDark ? '#f1f5f9' : '#111827';
    const tTextMuted  = isDark ? '#94a3b8' : '#4b5563';
    const tBorder     = isDark ? '#334155' : '#e5e7eb';
    const tBg         = isDark ? 'rgba(22, 32, 50, 0.97)' : 'rgba(255, 255, 255, 0.97)';

    // Number formatter for tooltips
    const fmtNum = (v: any): string => {
      if (v === null || v === undefined) return '—';
      const n = Number(v);
      if (!isFinite(n)) return String(v);
      if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
      if (Math.abs(n) >= 10_000)    return Math.round(n).toLocaleString('en-US');
      if (Number.isInteger(n))      return n.toLocaleString('en-US');
      return (Math.round(n * 100) / 100).toLocaleString('en-US');
    };

    // Rich HTML Tooltips
    const htmlFormatter = (params: any) => {
      let p = Array.isArray(params) ? params : [params];
      if (!p.length) return '';
      let title = p[0].axisValue || p[0].name || '';
      let html = '<div style="margin-bottom:6px;font-weight:700;font-size:12.5px;color:' + tTextColor + ';border-bottom:1px solid ' + tBorder + ';padding-bottom:5px;">' + title + '</div>';
      p.forEach((item: any) => {
        let val = item.value;
        if (Array.isArray(val)) val = val[item.seriesType === 'scatter' ? 1 : 2] ?? val[1] ?? val[0];
        const displayVal = (item.seriesType === 'pie' || item.seriesType === 'funnel')
          ? fmtNum(item.value) + ' <span style="color:' + tTextMuted + ';font-size:10.5px;font-weight:500">(' + item.percent + '%)</span>'
          : fmtNum(val);
        let marker = item.marker || '<span style="display:inline-block;margin-right:5px;border-radius:50%;width:9px;height:9px;background:' + item.color + '"></span>';
        html += '<div style="display:flex;justify-content:space-between;align-items:center;gap:18px;margin-top:5px;font-size:12px;color:' + tTextMuted + ';">' +
                  '<span style="display:flex;align-items:center;gap:2px">' + marker + (item.seriesName || item.name) + '</span>' +
                  '<span style="font-weight:700;color:' + tTextColor + ';font-family:ui-monospace,SFMono-Regular,monospace;font-size:12.5px">' + displayVal + '</span>' +
                '</div>';
      });
      return html;
    };

    // Scatter uses trigger:'item' (hover a point) and its string formatter from Python
    // (shows both X and Y values). Overriding to 'axis' or htmlFormatter breaks it.
    const isScatter = chartType === 'scatter';
    opt.tooltip = { ...(opt.tooltip ?? {}), trigger: (isCartesian && !isScatter) ? 'axis' : 'item', formatter: isScatter ? (opt.tooltip?.formatter ?? htmlFormatter) : htmlFormatter, backgroundColor: tBg, borderColor: tBorder, borderWidth: 1, padding: [12,16], confine: true, textStyle: { color: tTextColor, fontSize: 12, fontFamily: 'ui-sans-serif,system-ui,sans-serif' }, extraCssText: 'border-radius:12px;box-shadow:0 10px 25px -5px rgba(0,0,0,0.25); backdrop-filter: blur(8px);', axisPointer: (isCartesian && !isScatter) ? { type:'cross', crossStyle:{color:tBorder,width:1}, lineStyle:{color:isDark ? '#475569' : '#dce9ff',width:1,type:'dashed'}, label:{backgroundColor:'#2563eb',color:'#fff',fontSize:11,borderRadius:4,padding:[3,6]} } : undefined };
    if (opt.legend) {
      const isPieFamily = ['pie','doughnut','nightingale'].includes(chartType ?? '');
      const containerW  = this.elRef.nativeElement.clientWidth || 9999;
      const { fontSize: legendFontSize, iconSize: legendIconSize } =
        this._legendSizeForWidth(containerW, isPieFamily);
      opt.legend = {
        ...opt.legend,
        type: 'scroll',
        inactiveColor: isDark ? '#475569' : '#d1d5db',
        // Merge textStyle — preserve width/overflow/truncate set by chart_advisor
        textStyle: {
          ...(opt.legend.textStyle ?? {}),
          color: tTextMuted,
          fontSize: legendFontSize,
          fontFamily: 'ui-sans-serif,system-ui,sans-serif',
        },
        itemWidth:  legendIconSize,
        itemHeight: legendIconSize,
        pageTextStyle: { color: tTextMuted, fontSize: 10 },
        pageIconColor: '#2563eb',
        pageIconInactiveColor: isDark ? '#475569' : '#d1d5db',
      };
    }
    if (isCartesian) {
      const isHbar = chartType === 'horizontal_bar';
      opt.grid = { ...(opt.grid ?? {}), left:'1%', right:'2%', top: seriesCount > 1 ? '52px' : '18px', bottom:'68px', containLabel:true };
      const styleAxis = (ax: any) => { ax.axisLabel = { ...(ax.axisLabel ?? {}), color:tTextMuted, fontSize:11, fontFamily:'ui-sans-serif,system-ui,sans-serif', hideOverlap:false }; ax.axisLine = {lineStyle:{color:tBorder}}; ax.axisTick = {lineStyle:{color:tBorder},show:false}; ax.splitLine = {lineStyle:{color:isDark ? '#1e293b' : '#f1f3f7',type:'solid'}}; return ax; };
      if (opt.xAxis) { const a = Array.isArray(opt.xAxis)?opt.xAxis:[opt.xAxis]; opt.xAxis = Array.isArray(opt.xAxis)?a.map(styleAxis):styleAxis(a[0]); }
      if (opt.yAxis) { const a = Array.isArray(opt.yAxis)?opt.yAxis:[opt.yAxis]; opt.yAxis = Array.isArray(opt.yAxis)?a.map(styleAxis):styleAxis(a[0]); }
      // Rotate category x-axis labels when many values would overlap
      if (!isHbar && opt.xAxis) {
        const xArr = Array.isArray(opt.xAxis) ? opt.xAxis : [opt.xAxis];
        const xLen = xArr[0]?.data?.length ?? 0;
        if (xLen > 6) {
          const rotate = xLen > 12 ? 90 : 45;
          xArr.forEach((ax: any) => {
            ax.axisLabel = { ...(ax.axisLabel ?? {}), rotate, interval: 0 };
          });
        }
      }
      // Rotate horizontal-bar value x-axis labels vertically when container is narrow
      if (isHbar && opt.xAxis) {
        const containerW = this.elRef.nativeElement.clientWidth || 9999;
        const rotate = containerW < 480 ? 90 : 0;
        const xArr = Array.isArray(opt.xAxis) ? opt.xAxis : [opt.xAxis];
        xArr.forEach((ax: any) => {
          ax.axisLabel = { ...(ax.axisLabel ?? {}), rotate, hideOverlap: true };
        });
      }
      if (this.isZoomable) {
        const xData = Array.isArray(opt.xAxis)?opt.xAxis[0]?.data:opt.xAxis?.data;
        const yData = Array.isArray(opt.yAxis)?opt.yAxis[0]?.data:opt.yAxis?.data;
        const refData = isHbar ? yData : xData; const dataLen = refData?.length ?? 0;
        const winPct = dataLen > 30 ? Math.round(3000/dataLen) : 100;
        opt.dataZoom = [
          { type:'inside', xAxisIndex:isHbar?undefined:0, yAxisIndex:isHbar?0:undefined, filterMode:'none', zoomOnMouseWheel:true, moveOnMouseMove:true, moveOnMouseWheel:true, start:100-winPct, end:100 },
          { type:'slider', xAxisIndex:isHbar?undefined:0, yAxisIndex:isHbar?0:undefined, filterMode:'none', start:100-winPct, end:100, height:isHbar?undefined:18, width:isHbar?18:undefined, right:isHbar?8:undefined, bottom:isHbar?undefined:28, orient:isHbar?'vertical':'horizontal', borderColor:tBorder, backgroundColor:isDark ? '#1e293b' : '#f8f9fb', fillerColor:'rgba(37,99,235,0.15)', handleStyle:{color:'#2563eb',borderColor:'#93c5fd',borderWidth:1}, textStyle:{color:tTextMuted,fontSize:10}, showDetail:true, showDataShadow:'auto', dataBackground:{lineStyle:{color:isDark ? '#475569' : '#dce9ff',width:1},areaStyle:{color:'rgba(37,99,235,0.04)'}}, selectedDataBackground:{lineStyle:{color:'#93c5fd',width:1},areaStyle:{color:'rgba(37,99,235,0.1)'}} },
        ];
      }
    }
    if ((isHeatmap || isCalendar) && opt.visualMap) {
      opt.visualMap = { ...opt.visualMap, textStyle:{color:tTextMuted,fontSize:11}, inRange:{color: isDark ? ['#1e2a42','#6366f1','#10b981'] : ['#eff6ff','#2563eb','#10b981']} };
    }
    if (isCalendar && opt.calendar) {
      const cals = Array.isArray(opt.calendar) ? opt.calendar : [opt.calendar];
      cals.forEach((cal: any) => {
        cal.dayLabel   = { ...(cal.dayLabel   ?? {}), color: tTextMuted, fontSize: 10 };
        cal.monthLabel = { ...(cal.monthLabel ?? {}), color: tTextMuted, fontSize: 10 };
        cal.yearLabel  = { ...(cal.yearLabel  ?? {}), color: tTextColor, fontSize: 12 };
        cal.itemStyle  = {
          color: isDark ? 'rgba(255,255,255,0.03)' : '#f8fafc',
          borderColor: isDark ? 'rgba(255,255,255,0.06)' : '#e2e8f0',
          borderWidth: 1,
        };
        cal.splitLine  = { lineStyle: { color: isDark ? 'rgba(255,255,255,0.06)' : '#e2e8f0', width: 1 } };
      });
    }
    if (isRadar && opt.radar) opt.radar = { ...opt.radar, axisName:{color:'#6b7280',fontSize:11}, splitLine:{lineStyle:{color:'#f1f3f7'}}, splitArea:{show:true,areaStyle:{color:['#fafafa','#f3f4f6']}}, axisLine:{lineStyle:{color:'#e4e7ed'}} };
    // Gauge: ticks and labels use "inherit" (arc color) which is readable in dark mode
    // but may clash in light mode. Override explicitly for reliable cross-theme rendering.
    if (chartType === 'gauge' && opt.series) {
      opt.series.forEach((s: any) => {
        if (s.type !== 'gauge') return;
        if (s.axisTick?.lineStyle)  s.axisTick.lineStyle  = { ...s.axisTick.lineStyle,  color: isDark ? '#ffffff' : tBorder };
        if (s.splitLine?.lineStyle) s.splitLine.lineStyle = { ...s.splitLine.lineStyle, color: isDark ? '#ffffff' : tBorder };
        if (s.axisLabel) s.axisLabel = { ...s.axisLabel,   color: tTextMuted, fontSize: 11 };
        if (s.detail)   s.detail    = { ...s.detail,       color: tTextColor };
      });
    }
    if (opt.series) {
      opt.series.forEach((s: any, i: number) => {
        const c = PALETTE[i % PALETTE.length];
        if (s.type === 'bar') {
          if (!isWaterfall) {
            s.itemStyle = { ...(s.itemStyle??{}), borderRadius:[5,5,0,0], color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:c},{offset:1,color:c+'99'}]) };
          } else {
            // Waterfall: only gradient-ify series that have an explicit hex color.
            // The transparent base series has color='transparent'; appending 'aa' yields
            // 'transparentaa' — an invalid CSS color that throws a canvas DOMException
            // and silently aborts rendering of ALL series, leaving an empty chart.
            const bc = s.itemStyle?.color;
            if (bc && typeof bc === 'string' && bc.startsWith('#')) {
              s.itemStyle = {...s.itemStyle, color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:bc},{offset:1,color:bc+'aa'}])};
            }
            // transparent base and per-item-colored bars: leave itemStyle untouched
          }
          s.barMaxWidth = s.barMaxWidth??48;
          s.emphasis = {focus:'series',itemStyle:{shadowBlur:12,shadowColor:c+'55',shadowOffsetY:-3,borderRadius:[5,5,0,0]}};
        }
        if (s.type === 'line') {
          s.smooth=0.4; s.symbolSize=5; s.symbol='circle';
          s.lineStyle={...(s.lineStyle??{}),width:2.5,shadowColor:c+'33',shadowBlur:4};
          s.itemStyle={...(s.itemStyle??{}),borderWidth:2.5,borderColor:'#fff',color:c};
          if (!s.areaStyle) s.areaStyle={color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:c+'2a'},{offset:0.65,color:c+'08'},{offset:1,color:c+'00'}])};
          s.emphasis={focus:'series',itemStyle:{shadowBlur:12,shadowColor:c+'66',scale:true,borderWidth:3}};
        }
        if (s.type === 'scatter') { s.symbolSize=9; s.itemStyle={...(s.itemStyle??{}),borderWidth:2,borderColor:'#fff',color:c}; s.emphasis={focus:'self',itemStyle:{shadowBlur:10,shadowColor:c+'66'}}; }
        if (s.type === 'pie') { s.radius=s.radius??['0%','68%']; s.itemStyle={...(s.itemStyle??{}),borderColor:'#fff',borderWidth:3,borderRadius:5}; s.label={show:true,formatter:'{b|{b}}\n{per|{d}%}',rich:{b:{fontSize:11,color:'#6b7280',lineHeight:16},per:{fontSize:13,fontWeight:700,color:'#111827',lineHeight:18}}}; s.labelLine={length:12,length2:8,lineStyle:{color:'#d1d5db'}}; s.emphasis={scale:true,scaleSize:5,itemStyle:{shadowBlur:14,shadowColor:'rgba(0,0,0,0.15)'}}; }
        if (s.type === 'radar') { s.areaStyle={...(s.areaStyle??{}),opacity:0.08}; s.lineStyle={...(s.lineStyle??{}),width:2}; s.symbolSize=s.symbolSize??4; s.symbol='circle'; }
      });
    }
    // Per-chart-type animation: tuned easing + duration for each feel
    const isBarType  = ['bar','horizontal_bar','stacked_bar','waterfall'].includes(chartType ?? '');
    const isLineType = ['line','area'].includes(chartType ?? '');
    const isPieType  = ['pie','doughnut','nightingale'].includes(chartType ?? '');

    opt.animation               = true;
    opt.animationDurationUpdate = 500;
    opt.animationEasingUpdate   = 'cubicInOut';
    opt.animationDelayUpdate    = (idx: number) => idx * 15;

    if (isBarType) {
      // backOut: bars grow past target and spring back — satisfying "pop" effect
      opt.animationDuration = 900;
      opt.animationEasing   = 'backOut';
      opt.animationDelay    = (idx: number) => 50 + idx * 40;
    } else if (isLineType) {
      // sinusoidalOut: smooth wave-like draw from left-to-right
      opt.animationDuration = 1200;
      opt.animationEasing   = 'cubicInOut';
      opt.animationDelay    = (idx: number) => idx * 80;
    } else if (isPieType) {
      // backOut: slices pop in with a slight overshoot
      opt.animationDuration = 750;
      opt.animationEasing   = 'backOut';
      opt.animationDelay    = (idx: number) => idx * 70;
    } else {
      opt.animationDuration = 900;
      opt.animationEasing   = 'cubicOut';
      opt.animationDelay    = (idx: number) => 30 + idx * 55;
    }

    return opt;
  }

  // Single source of truth for legend text/icon sizing — was previously
  // computed independently (and identically, only by accident of never
  // having drifted) in buildOption(), here, and via the ResizeObserver
  // callback that just calls this method. Pie-family charts (pie/doughnut/
  // nightingale) shrink in narrow containers; other chart types are fixed.
  private _legendSizeForWidth(w: number, isPieFamily: boolean): { fontSize: number; iconSize: number } {
    if (!isPieFamily) return { fontSize: 12, iconSize: 10 };
    return { fontSize: w < 420 ? 9 : w < 560 ? 10 : 11, iconSize: w < 560 ? 8 : 10 };
  }

  private _syncPieLegendSize() {
    const isPieFamily = ['pie','doughnut','nightingale'].includes(this.activeType ?? '');
    if (!isPieFamily || !this.chart) return;
    const w = this.elRef.nativeElement.clientWidth;
    if (!w) return;
    const { fontSize, iconSize } = this._legendSizeForWidth(w, true);
    this.chart.setOption({ legend: { textStyle: { fontSize }, itemWidth: iconSize, itemHeight: iconSize } }, false);
  }

  private _syncHbarXRotation() {
    if (this.activeType !== 'horizontal_bar' || !this.chart) return;
    const w = this.elRef.nativeElement.clientWidth;
    if (!w) return;
    const rotate = w < 480 ? 90 : 0;
    this.chart.setOption({ xAxis: { axisLabel: { rotate } } }, false);
  }

  private destroyChart() {
    this.calZoom = 1;
    clearTimeout(this.resizeTimer); this.resizeTimer = null;
    this.resizeObs?.disconnect(); this.resizeObs = null;
    if (this.transitionEndCb) {
      document.removeEventListener('transitionend', this.transitionEndCb);
      this.transitionEndCb = null;
    }
    this.initTimers.forEach(t => clearTimeout(t)); this.initTimers = [];
    if (this.chart) { this.chart.dispose(); this.chart = null; }
  }
  ngOnDestroy() { 
    this.destroyChart(); 
    if (this.themeObserver) this.themeObserver.disconnect();
  }
}