import {
  Component, Input, Output, EventEmitter, OnChanges, SimpleChanges,
  ChangeDetectionStrategy, ChangeDetectorRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';
import { ResultData, SelectedRow } from '../../models/chat.model';
import { resolveSubject } from '../../utils/subject';

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
        {{ display.disambiguation ? 'Click a row to select' : 'Click a row to drill down' }}
      </div>
      <div class="drilldown-hint" *ngIf="showCheckboxes">
        <svg viewBox="0 0 16 16" fill="currentColor"><path d="M13.485 1.929a1 1 0 00-1.414 0L6.15 7.85 3.929 5.629a1 1 0 10-1.414 1.414l3 3a1 1 0 001.414 0l6.556-6.556a1 1 0 000-1.414z"/></svg>
        Check rows to ask a follow-up about just those
      </div>

      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th *ngIf="showCheckboxes" class="col-check"></th>
              <th *ngFor="let col of display.columns">{{ formatHeader(col) }}</th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let row of display.rows; let i = index; let even = even"
                [class.row-even]="even"
                [class.row-drilldown]="isDrilldown"
                [class.row-selected]="showCheckboxes && selectedPositions.has(i + 1)"
                (click)="isDrilldown ? onRowClick(row) : null">
              <td *ngIf="showCheckboxes" class="col-check" (click)="$event.stopPropagation()">
                <input type="checkbox"
                  [checked]="selectedPositions.has(i + 1)"
                  (change)="onRowCheck(row, i, $event)">
              </td>
              <td *ngFor="let col of display.columns" [title]="formatCell(row[col])">{{ formatCell(row[col]) }}</td>
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
      border-radius: 16px; overflow: hidden;
      background: var(--surface);
      border: 1px solid var(--border);
      box-shadow: var(--shadow-sm);
    }

    /* Drilldown hint */
    .drilldown-hint {
      display: flex; align-items: center; gap: 6px;
      padding: 6px 14px;
      background: var(--blue-50); border-bottom: 1px solid var(--blue-100);
      font-size: 11px; color: var(--accent); font-weight: 500;
    }
    .drilldown-hint svg { width: 12px; height: 12px; flex-shrink: 0; }

    .table-scroll { overflow-x: auto; max-height: 420px; overflow-y: auto; }
    table {
      width: 100%; border-collapse: collapse;
      font-size: 13px; font-family: var(--font);
    }
    thead { position: sticky; top: 0; z-index: 2; }
    thead tr { background: var(--surface-2); border-bottom: 1.5px solid var(--border-2); }
    th {
      text-align: left; padding: 9px 14px; color: var(--accent);
      font-size: 10.5px; font-weight: 700; letter-spacing: 0.6px;
      text-transform: uppercase; white-space: nowrap;
    }
    td {
      padding: 8px 14px; color: var(--text-2);
      border-bottom: 1px solid var(--border);
      max-width: 280px; overflow: hidden;
      text-overflow: ellipsis; white-space: nowrap;
    }
    tr:last-child td { border-bottom: none; }
    .row-even { background: var(--surface-2); }
    tbody tr:hover { background: var(--surface-2); }

    /* Row selection */
    .col-check { width: 34px; padding: 8px 10px 8px 14px; }
    .col-check input[type="checkbox"] { cursor: pointer; width: 14px; height: 14px; }
    .row-selected { background: var(--blue-50) !important; }

    /* Drilldown rows */
    .row-drilldown { cursor: pointer; transition: all 0.12s; }
    .row-drilldown:hover {
      background: var(--blue-50) !important;
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .row-drilldown td:first-child { color: var(--accent); font-weight: 600; }

    .table-footer {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 14px; gap: 12px; flex-wrap: wrap;
      border-top: 1px solid var(--border);
      background: var(--surface-2);
    }
    .row-label { font-size: 12px; color: var(--text-3); }
    .row-label strong { color: var(--accent); font-weight: 700; }

    .pager { display: flex; align-items: center; gap: 6px; }
    .pg-arrow {
      display: flex; align-items: center; justify-content: center;
      width: 28px; height: 28px; border-radius: 6px;
      background: var(--surface); border: 1.5px solid var(--blue-100);
      color: var(--accent); cursor: pointer; transition: all 0.15s;
    }
    .pg-arrow svg { width: 14px; height: 14px; }
    .pg-arrow:hover:not([disabled]) { background: var(--accent); border-color: var(--accent); color: #fff; }
    .pg-arrow[disabled] { opacity: 0.3; cursor: not-allowed; }

    .pg-label { display: flex; align-items: center; gap: 4px; font-size: 13px; font-weight: 600; }
    .pg-current {
      color: var(--accent); background: var(--blue-50);
      border: 1.5px solid var(--accent); border-radius: 6px;
      padding: 2px 10px; min-width: 28px; text-align: center; font-weight: 700;
    }
    .pg-sep   { color: var(--text-3); font-size: 12px; }
    .pg-total { color: var(--text-2); font-size: 13px; }

    .pg-spin { width: 14px; height: 14px; color: var(--accent); animation: spin 0.75s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    .page-size-wrap { display: flex; align-items: center; gap: 6px; }
    .ps-label { font-size: 11px; color: var(--text-3); }
    .ps-select {
      background: var(--surface); border: 1.5px solid var(--blue-100); border-radius: 6px;
      color: var(--accent); font-size: 12px; padding: 3px 8px; cursor: pointer; outline: none;
    }
    .ps-select:hover { border-color: var(--accent); }

    .pg-error {
      display: flex; align-items: center; gap: 7px;
      padding: 7px 14px; font-size: 12px; color: var(--err);
      background: var(--red-bg); border-top: 1px solid color-mix(in srgb, var(--err) 25%, transparent);
    }
    .pg-error svg { width: 13px; height: 13px; flex-shrink: 0; }
  `],
})
export class ResultTableComponent implements OnChanges {

  @Input()  data!:           ResultData;
  @Input()  sessionId!:      string;
  @Input()  messageId!:      string;
  @Input()  sourceQuestion?: string;  // the question that produced `data` — used to infer
                                       // drilldown intent (see resolveSubject())
  @Input()  selectable = false;       // server said this turn's rows are resolvable
                                       // (ResultData.row_selectable) AND the caller has
                                       // restricted this to the latest AI message.
  @Input()  resetSelection = 0;       // bump to clear checkboxes from outside (e.g. the
                                       // chip bar's "clear" action) — watched via ngOnChanges.
  @Output() drilldown = new EventEmitter<string>();  // emits message text to send
  @Output() selectionChange = new EventEmitter<SelectedRow[]>();

  display!:    ResultData;
  currentPage = 0;
  pageSize    = 50;
  loading     = false;
  error: string | null = null;
  selectedPositions = new Set<number>();

  constructor(private svc: ChatService, private cdr: ChangeDetectorRef) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data'] && this.data) {
      this.display     = { ...this.data };
      this.currentPage = this.data.page      ?? 0;
      this.pageSize    = this.data.page_size ?? 50;
      this.error       = null;
    }
    if (changes['resetSelection'] && !changes['resetSelection'].firstChange) {
      this.selectedPositions.clear();
    }
  }

  // Checkboxes only make sense on an ungrouped, unpaginated, non-drilldown
  // table — selection resolves server-side against ConversationTurn.row_titles
  // (page 1 only, see ResultData.row_selectable's docstring), so hide them
  // once the user has paged or changed page size within THIS table.
  get showCheckboxes(): boolean {
    return this.selectable && !this.isDrilldown && this.currentPage === 0;
  }

  onRowCheck(row: Record<string, any>, rowIndex: number, event: Event): void {
    const position = this.currentPage * this.pageSize + rowIndex + 1;
    const checked = (event.target as HTMLInputElement).checked;
    if (checked) {
      this.selectedPositions.add(position);
    } else {
      this.selectedPositions.delete(position);
    }
    this.emitSelection();
  }

  private emitSelection(): void {
    const labelCol = this.display?.columns?.[0];
    const selected: SelectedRow[] = Array.from(this.selectedPositions)
      .sort((a, b) => a - b)
      .map(position => {
        const row = this.display.rows[position - 1 - this.currentPage * this.pageSize];
        const label = labelCol ? this.formatCell(row?.[labelCol]) : `row ${position}`;
        return { position, label };
      });
    this.selectionChange.emit(selected);
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
    if (!this.display?.columns) return false;

    // Backend-confirmed disambiguation pick-list (core.disambiguation's
    // is_disambiguation_result / is_entity_disambiguation_result) — always
    // drillable regardless of row count. These can legitimately run into the
    // hundreds (a broad name search), paginated at 50/page, so the generic
    // heuristic's 20-row cap below must not apply here.
    if (this.display.disambiguation && this.display.rows.length >= 1) return true;

    if (this.display.columns.length !== 2) return false;
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
   * for that entity — not a recount. Preferred path: the backend already told
   * us (via `display.disambiguation`, computed from the same SQL that
   * produced this table — see core.disambiguation.extract_search_term)
   * exactly what search term was ambiguous. Substituting the resolved entity
   * name for just that term inside `sourceQuestion` preserves everything else
   * the user actually asked for — subject, verb, even typos — e.g.
   * "list corrective actions at adnoc" → "list corrective actions at Adnoc
   * Schools- Ruwais". Works uniformly for any kind (person, facility,
   * project, client, organisation) with no per-kind branching needed here.
   * Falls back to a generic "list of <subject> by/at/for <value>" phrasing
   * (subject guessed from keywords) when there's no disambiguation metadata —
   * e.g. clicking a plain "inspections per inspector" summary row.
   */
  onRowClick(row: Record<string, any>): void {
    const labelCol = this.display.columns[0];
    const value    = row[labelCol];
    if (!value) return;

    const origin = this.sourceQuestion?.trim();
    const term   = this.display.disambiguation?.search_term;
    if (origin && term) {
      const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const re = new RegExp(escaped, 'i');
      if (re.test(origin)) {
        this.drilldown.emit(origin.replace(re, value));
        return;
      }
    }

    // Fallback — build a generic detail request based on the column name
    // so Llama understands to return rows, not an aggregate
    const subject = resolveSubject(this.sourceQuestion);
    const colName  = labelCol.toLowerCase();
    if (colName.includes('inspector') || colName.includes('name')) {
      this.drilldown.emit(`show me the list of ${subject} by ${value}`);
    } else if (colName.includes('facility')) {
      this.drilldown.emit(`show me the list of ${subject} at ${value}`);
    } else if (colName.includes('region') || colName.includes('country') || colName.includes('project')) {
      this.drilldown.emit(`show me the list of ${subject} for ${value}`);
    } else {
      // Generic fallback — "show <subject> where <col> is <value>"
      this.drilldown.emit(`show me ${subject} where ${labelCol.replace(/_/g, ' ')} is ${value}`);
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

  formatCell(value: any): string {
    if (value === null || value === undefined) return '—';
    const s = String(value).trim();
    if (!s || s === 'null' || s === 'None') return '—';

    // ISO timestamp: 2026-01-01 00:00:00+00:00 or 2026-01-01T00:00:00Z
    if (/^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(s)) {
      try {
        const d = new Date(s.replace(' ', 'T'));
        if (!isNaN(d.getTime())) {
          const isMonthStart = d.getUTCDate() === 1 && d.getUTCHours() === 0 && d.getUTCMinutes() === 0 && d.getUTCSeconds() === 0;
          if (isMonthStart) {
            return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });
          }
          const hasTime = d.getUTCHours() !== 0 || d.getUTCMinutes() !== 0;
          const opts: Intl.DateTimeFormatOptions = hasTime
            ? { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC' }
            : { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' };
          return d.toLocaleDateString('en-US', opts);
        }
      } catch { /* fall through */ }
    }

    return s;
  }

  goToPage(page: number): void {
    if (page < 0 || page >= this.totalPages || this.loading) return;
    this.loading = true;
    this.error   = null;
    // /api/page bypasses the session entirely (see ResultData.row_selectable's
    // docstring) — no row_titles backing exists to resolve a selection made
    // on a different page against, so clear it rather than carry it forward.
    this.selectedPositions.clear();
    this.emitSelection();
    this.svc.fetchPage(this.sessionId, page, this.pageSize).subscribe({
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
    this.selectedPositions.clear();
    this.emitSelection();
    this.svc.fetchPage(this.sessionId, 0, this.pageSize).subscribe({
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