import {
  Component, Input, Output, EventEmitter, OnChanges, SimpleChanges,
  ChangeDetectionStrategy, ChangeDetectorRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';
import { ResultData } from '../../models/chat.model';

@Component({
  selector: 'app-result-table',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="table-card" *ngIf="display && display.rows.length > 0">

      <!-- Drilldown hint bar -->
      <div class="drilldown-hint" *ngIf="isDrilldown">
        <svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm.75 4.75a.75.75 0 00-1.5 0v3.5H5.5a.75.75 0 000 1.5h3.25V5.75z"/></svg>
        Click a row to drill down
      </div>

      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th *ngFor="let col of display.columns">{{ formatHeader(col) }}</th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let row of display.rows; let even = even"
                [class.row-even]="even"
                [class.row-drilldown]="isDrilldown"
                (click)="isDrilldown ? onRowClick(row) : null">
              <td *ngFor="let col of display.columns">{{ row[col] ?? '—' }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="table-footer">
        <span class="row-label">
          <ng-container *ngIf="display.total_count && display.total_count > 0; else simpleCount">
            {{ pageStart }}–{{ pageEnd }}&nbsp;of&nbsp;<strong>{{ display.total_count }}</strong>&nbsp;rows
          </ng-container>
          <ng-template #simpleCount>
            <strong>{{ display.row_count }}</strong>&nbsp;row{{ display.row_count !== 1 ? 's' : '' }}
          </ng-template>
        </span>

        <div class="pager" *ngIf="totalPages > 1">
          <button class="pg-arrow" [disabled]="currentPage === 0 || loading"
                  (click)="goToPage(currentPage - 1)" title="Previous">
            <svg viewBox="0 0 16 16" fill="currentColor">
              <path d="M10.06 3.47a.75.75 0 010 1.06L6.59 8l3.47 3.47a.75.75 0 11-1.06 1.06l-4-4a.75.75 0 010-1.06l4-4a.75.75 0 011.06 0z"/>
            </svg>
          </button>
          <span class="pg-label">
            <span class="pg-current">{{ currentPage + 1 }}</span>
            <span class="pg-sep">/</span>
            <span class="pg-total">{{ totalPages }}</span>
          </span>
          <button class="pg-arrow" [disabled]="currentPage >= totalPages - 1 || loading"
                  (click)="goToPage(currentPage + 1)" title="Next">
            <svg viewBox="0 0 16 16" fill="currentColor">
              <path d="M5.94 3.47a.75.75 0 000 1.06L9.41 8 5.94 11.47a.75.75 0 001.06 1.06l4-4a.75.75 0 000-1.06l-4-4a.75.75 0 00-1.06 0z"/>
            </svg>
          </button>
          <svg *ngIf="loading" class="pg-spin" viewBox="0 0 24 24"
               fill="none" stroke="currentColor" stroke-width="2.5">
            <path stroke-linecap="round" d="M12 3a9 9 0 019 9"/>
          </svg>
        </div>

        <div class="page-size-wrap" *ngIf="display.total_count && display.total_count > 20">
          <span class="ps-label">Per page</span>
          <select class="ps-select" [(ngModel)]="pageSize" (change)="changePageSize()">
            <option [value]="20">20</option>
            <option [value]="50">50</option>
            <option [value]="100">100</option>
          </select>
        </div>
      </div>

      <div class="pg-error" *ngIf="error">
        <svg viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 3.5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 018 4.5zm0 7a1 1 0 110-2 1 1 0 010 2z"/>
        </svg>
        {{ error }}
      </div>
    </div>
  `,
  styles: [`
    :host { display: block; margin-top: 10px; }

    .table-card {
      border-radius: 10px; overflow: hidden;
      background: var(--surface);
      border: 1.5px solid #c6eeeb;
      box-shadow: 0 2px 12px rgba(63,189,173,0.08);
    }

    /* Drilldown hint */
    .drilldown-hint {
      display: flex; align-items: center; gap: 6px;
      padding: 6px 14px;
      background: #edf9f8; border-bottom: 1px solid #c6eeeb;
      font-size: 11px; color: #32a89a; font-weight: 500;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .drilldown-hint svg { width: 12px; height: 12px; flex-shrink: 0; }

    .table-scroll { overflow-x: auto; max-height: 420px; overflow-y: auto; }
    table {
      width: 100%; border-collapse: collapse;
      font-size: 13px; font-family: ui-sans-serif, system-ui, sans-serif;
    }
    thead { position: sticky; top: 0; z-index: 2; }
    thead tr { background: var(--surface-2); border-bottom: 2px solid #3fbdad; }
    th {
      text-align: left; padding: 9px 14px; color: #32a89a;
      font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
      text-transform: uppercase; white-space: nowrap;
    }
    td {
      padding: 8px 14px; color: var(--gray-800);
      border-bottom: 1px solid var(--border);
      max-width: 280px; overflow: hidden;
      text-overflow: ellipsis; white-space: nowrap;
    }
    tr:last-child td { border-bottom: none; }
    .row-even { background: var(--surface-2); }
    tbody tr:hover { background: var(--surface-2); }

    /* Drilldown rows */
    .row-drilldown { cursor: pointer; transition: all 0.12s; }
    .row-drilldown:hover {
      background: #edf9f8 !important;
      box-shadow: inset 3px 0 0 #3fbdad;
    }
    .row-drilldown td:first-child { color: #32a89a; font-weight: 600; }

    .table-footer {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 14px; gap: 12px; flex-wrap: wrap;
      border-top: 1.5px solid #c6eeeb;
      background: var(--surface);
    }
    .row-label { font-size: 12px; color: var(--gray-500); }
    .row-label strong { color: #32a89a; font-weight: 700; }

    .pager { display: flex; align-items: center; gap: 6px; }
    .pg-arrow {
      display: flex; align-items: center; justify-content: center;
      width: 28px; height: 28px; border-radius: 6px;
      background: var(--surface); border: 1.5px solid #c6eeeb;
      color: #3fbdad; cursor: pointer; transition: all 0.15s;
    }
    .pg-arrow svg { width: 14px; height: 14px; }
    .pg-arrow:hover:not([disabled]) { background: #3fbdad; border-color: #3fbdad; color: #fff; }
    .pg-arrow[disabled] { opacity: 0.3; cursor: not-allowed; }

    .pg-label { display: flex; align-items: center; gap: 4px; font-size: 13px; font-weight: 600; }
    .pg-current {
      color: #32a89a; background: #edf9f8;
      border: 1.5px solid #3fbdad; border-radius: 6px;
      padding: 2px 10px; min-width: 28px; text-align: center; font-weight: 700;
    }
    .pg-sep   { color: #94a3b8; font-size: 12px; }
    .pg-total { color: var(--gray-500); font-size: 13px; }

    .pg-spin { width: 14px; height: 14px; color: #3fbdad; animation: spin 0.75s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    .page-size-wrap { display: flex; align-items: center; gap: 6px; }
    .ps-label { font-size: 11px; color: #94a3b8; }
    .ps-select {
      background: var(--surface); border: 1.5px solid #c6eeeb; border-radius: 6px;
      color: #32a89a; font-size: 12px; padding: 3px 8px; cursor: pointer; outline: none;
    }
    .ps-select:hover { border-color: #3fbdad; }

    .pg-error {
      display: flex; align-items: center; gap: 7px;
      padding: 7px 14px; font-size: 12px; color: #dc2626;
      background: #fef2f2; border-top: 1px solid #fecaca;
    }
    .pg-error svg { width: 13px; height: 13px; flex-shrink: 0; }
  `],
})
export class ResultTableComponent implements OnChanges {

  @Input()  data!:      ResultData;
  @Input()  sessionId!: string;
  @Input()  messageId!: string;
  @Output() drilldown = new EventEmitter<string>();  // emits message text to send

  display!:    ResultData;
  currentPage = 0;
  pageSize    = 50;
  loading     = false;
  error: string | null = null;

  constructor(private svc: ChatService, private cdr: ChangeDetectorRef) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data'] && this.data) {
      this.display     = { ...this.data };
      this.currentPage = this.data.page      ?? 0;
      this.pageSize    = this.data.page_size ?? 50;
      this.error       = null;
    }
  }

  /**
   * Detect if this result looks like a drillable entity list.
   *
   * Requirements (ALL must be true):
   *  1. Exactly 2 columns, 2–20 rows
   *  2. Second column name looks like a count/frequency metric
   *  3. First column is a short entity string — NOT a UUID, date, inspection ID,
   *     long text, OR a known summary/category label (conducted, completed, etc.)
   *  4. Second column value is numeric
   *
   * The "conducted | inspection_count" two-row summary must NOT be drillable —
   * 'conducted' and 'completed' are summary labels, not entity names.
   */
  get isDrilldown(): boolean {
    if (!this.display?.columns || this.display.columns.length !== 2) return false;
    if (this.display.rows.length > 20 || this.display.rows.length < 2) return false;

    const cols   = this.display.columns;
    const col1   = cols[1].toLowerCase();
    const sample = this.display.rows[0] ?? {};
    const first  = sample[cols[0]];
    const second = sample[cols[1]];

    // Second column must look like a count/frequency metric
    const isCountCol = /count|total|num_|frequency|inspection_count|ca_count|occurrences|assignment/.test(col1);
    if (!isCountCol) return false;

    // First column must be a short text label
    if (typeof first !== 'string') return false;

    const isUUID         = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(first);
    const isDate         = /^\d{4}[-\/]\d{2}/.test(first);
    const isLong         = first.length > 80;
    const isInspId       = /^\d{4}\/\w+\/INS\d+/.test(first);
    // Exclude known summary/category label values — these are grouped summaries not entities
    const isSummaryLabel = /^(conducted|completed|category|status|total|all|draft|submitted|closed|under_review|return_for_modification)$/i.test(first.trim());

    if (isUUID || isDate || isLong || isInspId || isSummaryLabel) return false;

    // Second column must be numeric
    return typeof second === 'number' || !isNaN(Number(second));
  }

  /**
   * When a drilldown row is clicked, emit a message requesting the DETAIL LIST
   * for that entity — not a recount. Phrasing "list of X inspections" signals
   * Llama to return inspection rows (inspection_id, facility, status, submitted_on)
   * rather than a COUNT(*).
   */
  onRowClick(row: Record<string, any>): void {
    const labelCol = this.display.columns[0];
    const countCol = this.display.columns[1];
    const value    = row[labelCol];
    const count    = row[countCol];
    if (!value) return;

    // Build a natural-language detail request based on the column name
    // so Llama understands to return rows, not an aggregate
    const colName = labelCol.toLowerCase();
    if (colName.includes('inspector') || colName.includes('name')) {
      this.drilldown.emit(`show me the list of inspections by ${value}`);
    } else if (colName.includes('facility')) {
      this.drilldown.emit(`show me the list of inspections at ${value}`);
    } else if (colName.includes('region') || colName.includes('country') || colName.includes('project')) {
      this.drilldown.emit(`show me the list of inspections for ${value}`);
    } else {
      // Generic fallback — "show inspections where <col> is <value>"
      this.drilldown.emit(`show me inspections where ${labelCol.replace(/_/g, ' ')} is ${value}`);
    }
  }

  get totalPages(): number {
    if (this.display?.total_count && this.display.total_count > 0) {
      return Math.ceil(this.display.total_count / this.pageSize);
    }
    return this.display?.has_more ? this.currentPage + 2 : this.currentPage + 1;
  }
  get pageStart(): number { return this.currentPage * this.pageSize + 1; }
  get pageEnd():   number {
    const total = (this.display?.total_count && this.display.total_count > 0) 
      ? this.display.total_count 
      : null;
    return Math.min(
      (this.currentPage + 1) * this.pageSize,
      total ?? this.pageStart + this.display.row_count - 1,
    );
  }

  formatHeader(col: string): string {
    return col.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  goToPage(page: number): void {
    if (page < 0 || page >= this.totalPages || this.loading) return;
    this.loading = true;
    this.error   = null;
    this.svc.fetchPage(this.sessionId, page, this.pageSize, this.display.sql).subscribe({
      next: res => {
        this.display     = { ...res };
        this.currentPage = page;
        this.loading     = false;
        this.svc.updateMessageResultData(this.messageId, { ...res });
        this.cdr.markForCheck();
      },
      error: err => {
        this.error   = err?.error?.detail ?? 'Failed to load page.';
        this.loading = false;
        this.cdr.markForCheck();
      },
    });
  }

  changePageSize(): void {
    this.loading = true;
    this.error   = null;
    this.svc.fetchPage(this.sessionId, 0, this.pageSize, this.display.sql).subscribe({
      next: res => {
        this.display     = { ...res };
        this.currentPage = 0;
        this.loading     = false;
        this.svc.updateMessageResultData(this.messageId, { ...res });
        this.cdr.markForCheck();
      },
      error: err => {
        this.error   = err?.error?.detail ?? 'Failed to load page.';
        this.loading = false;
        this.cdr.markForCheck();
      },
    });
  }
}