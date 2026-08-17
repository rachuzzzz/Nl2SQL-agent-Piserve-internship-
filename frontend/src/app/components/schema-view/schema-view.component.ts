import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-schema-view',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="s-scroll">
      <div class="s-inner">

        <div class="page-head">
          <div>
            <h1 class="page-title">Schema</h1>
            <p class="page-sub">The grounding OctaAI uses to generate SQL</p>
          </div>
        </div>

        <div class="empty">
          <div class="empty-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="28" height="28"
                 stroke-linecap="round" stroke-linejoin="round">
              <ellipse cx="12" cy="5" rx="9" ry="3"/>
              <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
              <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
            </svg>
          </div>
          <p class="empty-text">Schema browsing isn't available yet</p>
          <p class="empty-sub">Ask OctaAI a question in AI Chat and it will use the real database schema to answer — a browsable view of it is coming soon.</p>
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

    .page-head { margin-bottom: 24px; }
    .page-title {
      margin: 0 0 5px; font-size: 28px; font-weight: 800; letter-spacing: -0.02em;
      color: var(--text);
    }
    .page-sub { margin: 0; font-size: 14px; color: var(--text-2); }

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
export class SchemaViewComponent {}
