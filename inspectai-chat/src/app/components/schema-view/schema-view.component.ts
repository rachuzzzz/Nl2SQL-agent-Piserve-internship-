import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

const EHS_SCHEMA = [
  { name: 'inspection_report', rows: '48,210', desc: 'Site inspections & scores', cols: [
      ['id','uuid'],['site_id','int → site'],['inspector_id','int → inspector'],
      ['inspection_score','numeric'],['status','enum'],['submitted_on','timestamptz'],
      ['active','bool'],['template','text'],['findings_count','int']] },
  { name: 'audit', rows: '9,884', desc: 'Compliance & internal audits', cols: [
      ['id','uuid'],['site_id','int → site'],['type','enum'],['result','enum'],
      ['auditor_id','int'],['opened_on','date'],['closed_on','date'],['nonconformities','int']] },
  { name: 'permit', rows: '12,640', desc: 'Work & environmental permits', cols: [
      ['id','uuid'],['site_id','int → site'],['permit_type','enum'],['status','enum'],
      ['issued_on','date'],['expires_on','date'],['holder','text']] },
  { name: 'finding', rows: '131,902', desc: 'Findings raised on inspections', cols: [
      ['id','uuid'],['report_id','uuid → inspection_report'],['severity','enum'],
      ['category','text'],['is_open','bool'],['raised_on','date']] },
  { name: 'corrective_action', rows: '54,771', desc: 'CAPA tracking', cols: [
      ['id','uuid'],['finding_id','uuid → finding'],['site_id','int'],['status','enum'],
      ['due_on','date'],['closed_on','date'],['owner','text']] },
  { name: 'incident', rows: '7,355', desc: 'Safety incidents & near-misses', cols: [
      ['id','uuid'],['site_id','int'],['severity','enum'],['type','text'],
      ['reported_on','timestamptz'],['lost_time','bool']] },
  { name: 'site', rows: '212', desc: 'Facilities / locations', cols: [
      ['id','int'],['name','text'],['region','text'],['risk_tier','enum']] },
  { name: 'inspector', rows: '486', desc: 'Field inspectors', cols: [
      ['id','int'],['name','text'],['certified','bool'],['region','text']] },
];

@Component({
  selector: 'app-schema-view',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="s-scroll">
      <div class="s-inner">

        <div class="page-head">
          <div>
            <h1 class="page-title">Schema</h1>
            <p class="page-sub">{{ schema.length }} tables · the grounding OctaAI uses to generate SQL</p>
          </div>
          <label class="search-pill">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                 width="15" height="15" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            </svg>
            <input class="search-in" [(ngModel)]="searchQuery" placeholder="Search tables…"/>
          </label>
        </div>

        <div class="acc-list">
          <ng-container *ngFor="let t of filteredSchema">
            <div class="acc-item" [class.acc-open]="openTable === t.name">
              <button class="acc-hdr" (click)="toggle(t.name)">
                <span class="acc-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                       width="17" height="17" stroke-linecap="round" stroke-linejoin="round">
                    <ellipse cx="12" cy="5" rx="9" ry="3"/>
                    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
                    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
                  </svg>
                </span>
                <div class="acc-info">
                  <div class="acc-name">{{ t.name }}</div>
                  <div class="acc-desc">{{ t.desc }}</div>
                </div>
                <span class="acc-rows">{{ t.rows }} rows</span>
                <svg class="acc-chev" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2.5" width="15" height="15" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="6 9 12 15 18 9"/>
                </svg>
              </button>

              <div class="acc-body" *ngIf="openTable === t.name">
                <div class="col-grid">
                  <div *ngFor="let col of t.cols" class="col-chip">
                    <span class="col-n">{{ col[0] }}</span>
                    <span class="col-t">{{ col[1] }}</span>
                  </div>
                </div>
              </div>
            </div>
          </ng-container>

          <div class="empty-state" *ngIf="filteredSchema.length === 0">
            No tables match "{{ searchQuery }}"
          </div>
        </div>

      </div>
    </div>
  `,
  styles: [`
    :host {
      flex: 1; display: flex; flex-direction: column; overflow: hidden;
      background: var(--bg); min-width: 0;
    }
    .s-scroll { flex: 1; overflow-y: auto; }
    .s-inner  { max-width: 1100px; margin: 0 auto; padding: 30px 32px 60px; }

    .page-head {
      display: flex; align-items: flex-end; justify-content: space-between;
      gap: 16px; margin-bottom: 26px; flex-wrap: wrap;
    }
    .page-title {
      margin: 0 0 5px; font-size: 28px; font-weight: 800; letter-spacing: -0.02em;
      color: var(--text);
    }
    .page-sub { margin: 0; font-size: 14px; color: var(--text-2); }
    .search-pill {
      display: flex; align-items: center; gap: 8px;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 999px; padding: 8px 14px;
      box-shadow: var(--shadow-sm); color: var(--text-3); cursor: text;
    }
    .search-in {
      border: none; outline: none; background: transparent;
      color: var(--text); font-size: 13.5px; width: 150px; font-family: var(--font);
    }
    .search-in::placeholder { color: var(--text-3); }

    .acc-list { display: flex; flex-direction: column; gap: 10px; }
    .acc-item {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 18px; overflow: hidden; box-shadow: var(--shadow-sm);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .acc-item:hover { box-shadow: var(--shadow); }
    .acc-item.acc-open { border-color: var(--accent-soft-border); }
    .acc-hdr {
      width: 100%; display: flex; align-items: center; gap: 12px;
      padding: 14px 18px; background: transparent; border: none;
      text-align: left; cursor: pointer; transition: background 0.14s;
    }
    .acc-hdr:hover { background: var(--surface-2); }
    .acc-icon {
      width: 38px; height: 38px; border-radius: 12px;
      background: var(--accent-soft); display: grid; place-items: center;
      flex-shrink: 0; color: var(--accent);
    }
    .acc-info  { flex: 1; min-width: 0; }
    .acc-name  {
      font-size: 13.5px; font-weight: 700; color: var(--text);
      font-family: var(--font-mono); line-height: 1.3;
    }
    .acc-desc  { font-size: 12px; color: var(--text-2); margin-top: 2px; }
    .acc-rows  {
      font-size: 11.5px; font-weight: 600; color: var(--text-3);
      font-family: var(--font-mono); white-space: nowrap; margin-right: 4px;
    }
    .acc-chev {
      color: var(--text-3); flex-shrink: 0;
      transition: transform 0.22s cubic-bezier(0.4,0,0.2,1);
    }
    .acc-item.acc-open .acc-chev { transform: rotate(180deg); }

    .acc-body {
      border-top: 1px solid var(--border);
      padding: 4px 18px 16px;
    }
    .col-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
      gap: 7px; padding-top: 12px;
    }
    .col-chip {
      display: flex; align-items: center; justify-content: space-between;
      gap: 6px; padding: 7px 12px; background: var(--surface-2);
      border-radius: 12px; border: 1px solid var(--border);
      transition: background 0.14s, box-shadow 0.14s;
    }
    .col-chip:hover { background: var(--surface); box-shadow: var(--shadow-sm); }
    .col-n {
      font-size: 12.5px; font-weight: 600; color: var(--text);
      font-family: var(--font-mono);
    }
    .col-t {
      font-size: 11px; font-family: var(--font-mono);
      color: var(--accent); background: var(--accent-soft);
      padding: 2px 8px; border-radius: 999px; white-space: nowrap;
    }
    .empty-state {
      text-align: center; padding: 48px 24px;
      color: var(--text-3); font-size: 14px; font-style: italic;
    }
  `],
})
export class SchemaViewComponent {
  searchQuery = '';
  openTable   = 'inspection_report';
  readonly schema = EHS_SCHEMA;

  get filteredSchema() {
    const q = this.searchQuery.toLowerCase().trim();
    if (!q) return this.schema;
    return this.schema.filter(t =>
      t.name.includes(q) || t.desc.toLowerCase().includes(q),
    );
  }

  toggle(name: string) {
    this.openTable = this.openTable === name ? '' : name;
  }
}
