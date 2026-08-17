import { Component, EventEmitter, OnDestroy, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { ChatService } from '../../services/chat.service';
import { Conversation, Message } from '../../models/chat.model';

interface DashboardEntry {
  conversation: Conversation;
  message:      Message;
}

@Component({
  selector: 'app-dashboard-view',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="d-scroll">
      <div class="d-inner">

        <div class="page-head">
          <div>
            <h1 class="page-title">Dashboards</h1>
            <p class="page-sub">Rich answers from this session &middot; click one to jump back to it in chat</p>
          </div>
        </div>

        <div class="gallery-grid" *ngIf="entries.length">
          <button class="g-card" *ngFor="let e of entries" (click)="open(e)">
            <div class="g-card-top">
              <span class="msg-pill msg-pill-domain" *ngIf="e.message.domain">
                {{ e.message.domain === 'audit' ? '📋 Audit' : '🏢 Inspection' }}
              </span>
              <span class="g-time">{{ e.message.timestamp | date:'MMM d, h:mm a' }}</span>
            </div>
            <h3 class="g-question">{{ e.message.sourceQuestion || e.conversation.title }}</h3>
            <div class="g-cards" *ngIf="e.message.dashboard?.cards?.length">
              <div class="g-mini" *ngFor="let c of e.message.dashboard!.cards.slice(0, 3)"
                   [class.g-mini--critical]="c.severity === 'critical'"
                   [class.g-mini--warning]="c.severity === 'warning'"
                   [class.g-mini--ok]="c.severity === 'ok'">
                <span class="g-mini-label">{{ c.label }}</span>
                <span class="g-mini-value">
                  {{ c.value | number:'1.0-2' }}{{ c.value2 != null ? ' – ' + (c.value2 | number:'1.0-2') : '' }}
                </span>
              </div>
            </div>
          </button>
        </div>

        <div class="empty" *ngIf="!entries.length">
          <div class="empty-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="28" height="28"
                 stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="3" width="7" height="9" rx="1.5"/>
              <rect x="14" y="3" width="7" height="5" rx="1.5"/>
              <rect x="14" y="12" width="7" height="9" rx="1.5"/>
              <rect x="3" y="16" width="7" height="5" rx="1.5"/>
            </svg>
          </div>
          <p class="empty-text">No dashboard-shaped answers yet this session</p>
          <p class="empty-sub">Ask a question with a rich result &mdash; a breakdown, trend, or status summary &mdash; and it'll show up here. Scoped to this session only, same as your chat history: it clears on refresh.</p>
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

    .page-head { margin-bottom: 24px; }
    .page-title {
      margin: 0 0 5px; font-size: 28px; font-weight: 800; letter-spacing: -0.02em;
      color: var(--text);
    }
    .page-sub { margin: 0; font-size: 14px; color: var(--text-2); }

    /* Gallery */
    .gallery-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 14px;
    }
    .g-card {
      display: flex; flex-direction: column; gap: 10px; text-align: left;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 22px; padding: 16px 18px; cursor: pointer;
      font-family: var(--font); box-shadow: var(--shadow-sm);
      transition: box-shadow 0.22s, transform 0.22s, border-color 0.22s;
    }
    .g-card:hover {
      box-shadow: var(--shadow-md), 0 0 0 1px rgba(27,67,255,0.08);
      transform: translateY(-3px); border-color: var(--blue-100);
    }
    .g-card-top {
      display: flex; align-items: center; justify-content: space-between; gap: 8px;
    }
    .g-time { font-size: 10.5px; color: var(--text-3); white-space: nowrap; }
    .g-question {
      margin: 0; font-size: 14px; font-weight: 700; color: var(--text);
      line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2;
      -webkit-box-orient: vertical; overflow: hidden;
    }

    .g-cards { display: flex; flex-direction: column; gap: 6px; margin-top: 2px; }
    .g-mini {
      display: flex; align-items: baseline; justify-content: space-between;
      gap: 10px; padding: 7px 10px; background: var(--surface-2);
      border-radius: 10px; border: 1px solid var(--border);
    }
    .g-mini-label {
      font-size: 10px; font-weight: 700; letter-spacing: 0.04em;
      color: var(--text-3); text-transform: uppercase;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0;
    }
    .g-mini-value {
      font-size: 13px; font-weight: 700; color: var(--text);
      font-family: var(--font-mono); white-space: nowrap; flex-shrink: 0;
    }
    .g-mini--critical .g-mini-value { color: #dc2626; }
    .g-mini--warning  .g-mini-value { color: #d97706; }
    .g-mini--ok       .g-mini-value { color: #059669; }

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
    .empty-sub  { font-size: 13px; color: var(--text-3); margin: 0; text-align: center; max-width: 380px; line-height: 1.6; }
  `],
})
export class DashboardViewComponent implements OnDestroy {
  @Output() switchToChat = new EventEmitter<void>();

  entries: DashboardEntry[] = [];
  private sub: Subscription;

  constructor(private chatSvc: ChatService) {
    this.sub = chatSvc.conversations$.subscribe(convs => {
      const flat: DashboardEntry[] = [];
      for (const conversation of convs) {
        for (const message of conversation.messages) {
          if (message.role === 'ai' && message.dashboard?.show) {
            flat.push({ conversation, message });
          }
        }
      }
      flat.sort((a, b) => b.message.timestamp.getTime() - a.message.timestamp.getTime());
      this.entries = flat;
    });
  }

  ngOnDestroy(): void {
    this.sub.unsubscribe();
  }

  open(e: DashboardEntry): void {
    this.chatSvc.jumpToMessage(e.conversation.id, e.message.id);
    this.switchToChat.emit();
  }
}
