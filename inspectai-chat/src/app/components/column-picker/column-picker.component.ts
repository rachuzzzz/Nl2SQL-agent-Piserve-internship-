import {
  Component, Input, Output, EventEmitter, OnInit,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

export interface ColumnSelection {
  xCol:  string;
  yCols: string[];
}

@Component({
  selector: 'app-column-picker',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="picker-card">
      <div class="picker-header">
        <span class="picker-title">Customise chart columns</span>
        <button class="btn-close" (click)="cancel.emit()">✕</button>
      </div>

      <div class="picker-section">
        <label class="section-label">X axis <span class="section-hint">(label / category)</span></label>
        <div class="col-options">
          <button
            *ngFor="let col of allCols"
            class="col-pill"
            [class.selected]="selectedX === col"
            (click)="selectedX = col">
            <span class="col-type-dot" [class]="typeClass(col)"></span>
            {{ colLabel(col) }}
          </button>
        </div>
      </div>

      <div class="picker-section">
        <label class="section-label">Y axis <span class="section-hint">(values — select one or more)</span></label>
        <div class="col-options">
          <button
            *ngFor="let col of numericCols"
            class="col-pill"
            [class.selected]="isSelectedY(col)"
            [disabled]="col === selectedX"
            (click)="toggleY(col)">
            <span class="col-type-dot numeric"></span>
            {{ colLabel(col) }}
            <span class="checkmark" *ngIf="isSelectedY(col)">✓</span>
          </button>
          <span class="empty-note" *ngIf="numericCols.length === 0">No numeric columns in result</span>
        </div>
      </div>

      <div class="picker-footer">
        <button class="btn-cancel" (click)="cancel.emit()">Cancel</button>
        <button class="btn-draw" (click)="submit()" [disabled]="!canSubmit">Draw Chart</button>
      </div>
    </div>
  `,
  styles: [`
    .picker-card {
      border-radius: 12px; overflow: hidden;
      background: #ffffff; border: 1px solid #e4e7ed;
      box-shadow: 0 4px 16px rgba(0,0,0,0.08);
      margin: 8px 16px;
    }
    .picker-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 10px 14px; background: #f8f9fb; border-bottom: 1px solid #e4e7ed;
    }
    .picker-title { font-size: 13px; font-weight: 600; color: #374151; font-family: ui-sans-serif,system-ui,sans-serif; }
    .btn-close { background: none; border: none; color: #9aa0ad; font-size: 14px; cursor: pointer; padding: 2px 6px; border-radius: 4px; }
    .btn-close:hover { background: #f3f4f6; color: #374151; }
    .picker-section { padding: 12px 14px; border-bottom: 1px solid #f3f4f6; }
    .section-label { display: block; font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; font-family: ui-sans-serif,system-ui,sans-serif; }
    .section-hint { font-weight: 400; text-transform: none; letter-spacing: 0; color: #9aa0ad; }
    .col-options { display: flex; flex-wrap: wrap; gap: 6px; }
    .col-pill {
      display: flex; align-items: center; gap: 5px;
      padding: 4px 11px; border-radius: 20px; border: 1px solid #e4e7ed;
      background: #f9fafb; color: #4b5563; font-size: 12px; cursor: pointer;
      font-family: ui-sans-serif,system-ui,sans-serif; transition: all 0.12s;
    }
    .col-pill:hover:not([disabled]) { border-color: var(--accent); background: var(--blue-50); color: var(--accent); }
    .col-pill.selected { border-color: var(--accent); background: var(--blue-50); color: var(--accent); font-weight: 600; }
    .col-pill[disabled] { opacity: 0.4; cursor: not-allowed; }
    .checkmark { font-size: 11px; color: var(--accent); margin-left: 2px; }
    .col-type-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; background: var(--border-2); }
    .col-type-dot.numeric  { background: var(--accent); }
    .col-type-dot.date     { background: var(--ok); }
    .col-type-dot.category { background: var(--viz-violet); }
    .empty-note { font-size: 12px; color: #9aa0ad; font-style: italic; }
    .picker-footer { display: flex; align-items: center; justify-content: flex-end; gap: 8px; padding: 10px 14px; background: #f8f9fb; }
    .btn-cancel { padding: 6px 14px; border-radius: 7px; font-size: 12px; background: #ffffff; border: 1px solid #e4e7ed; color: #6b7280; cursor: pointer; font-family: ui-sans-serif,system-ui,sans-serif; }
    .btn-cancel:hover { border-color: #9aa0ad; color: #374151; }
    .btn-draw { padding: 6px 18px; border-radius: 7px; font-size: 12px; font-weight: 600; background: var(--accent); border: none; color: #fff; cursor: pointer; font-family: var(--font); }
    .btn-draw:hover:not([disabled]) { background: var(--blue-700); }
    .btn-draw[disabled] { opacity: 0.5; cursor: not-allowed; }
  `],
})
export class ColumnPickerComponent implements OnInit {
  @Input()  allCols:     string[] = [];
  @Input()  numericCols: string[] = [];
  @Input()  dateCols:    string[] = [];
  @Input()  catCols:     string[] = [];
  @Output() confirmed = new EventEmitter<ColumnSelection>();
  @Output() cancel    = new EventEmitter<void>();

  selectedX  = '';
  selectedYs = new Set<string>();

  ngOnInit() {
    this.selectedX = this.catCols[0] ?? this.dateCols[0] ?? this.allCols[0] ?? '';
    const firstNum = this.numericCols.find(c => c !== this.selectedX);
    if (firstNum) this.selectedYs.add(firstNum);
  }

  colLabel(col: string): string {
    return col.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  typeClass(col: string): string {
    if (this.numericCols.includes(col)) return 'numeric';
    if (this.dateCols.includes(col))    return 'date';
    return 'category';
  }

  isSelectedY(col: string): boolean { return this.selectedYs.has(col); }

  toggleY(col: string) {
    if (this.selectedYs.has(col)) this.selectedYs.delete(col);
    else                          this.selectedYs.add(col);
  }

  get canSubmit(): boolean { return !!this.selectedX && this.selectedYs.size > 0; }

  submit() {
    if (!this.canSubmit) return;
    this.confirmed.emit({ xCol: this.selectedX, yCols: Array.from(this.selectedYs) });
  }
}
