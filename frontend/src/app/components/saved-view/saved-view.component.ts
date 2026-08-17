import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

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
            <p class="page-sub">Pinned questions, reusable across your team</p>
          </div>
        </div>

        <div class="empty">
          <div class="empty-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="28" height="28"
                 stroke-linecap="round" stroke-linejoin="round">
              <path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>
            </svg>
          </div>
          <p class="empty-text">Saved queries aren't available yet</p>
          <p class="empty-sub">Pinning and reusing questions from AI Chat is coming soon.</p>
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
export class SavedViewComponent {}
