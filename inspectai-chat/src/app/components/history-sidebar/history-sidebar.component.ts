import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatService } from '../../services/chat.service';
import { Conversation } from '../../models/chat.model';

@Component({
  selector: 'app-history-sidebar',
  standalone: true,
  imports: [CommonModule],
  template: `
    <aside class="sidebar-left" [class.collapsed]="collapsed">
      <div class="sl-header">
        <div class="sl-header-left">
          <button class="btn-col" (click)="collapsed = !collapsed" title="Toggle sidebar">
            <svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"/></svg>
          </button>
          <span class="sl-label sl-hide">Conversations</span>
        </div>
        <button class="btn-new sl-hide" (click)="svc.newConversation()">+ New</button>
      </div>
      <div class="history-list sl-hide">
        <ng-container *ngFor="let group of groups">
          <div class="h-group">
            <div class="h-group-label">{{ group.label }}</div>
            <div *ngFor="let conv of group.items"
              class="h-item" [class.active]="(svc.activeConversation$ | async)?.id === conv.id"
              (click)="svc.selectConversation(conv.id)">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
              </svg>
              <span>{{ conv.title }}</span>
            </div>
          </div>
        </ng-container>
      </div>
    </aside>
  `,
})
export class HistorySidebarComponent {
  collapsed = false;
  groups: { label: string; items: Conversation[] }[] = [];

  constructor(public svc: ChatService) {
    svc.conversations$.subscribe(convs => this.buildGroups(convs));
  }

  buildGroups(convs: Conversation[]) {
    const now = Date.now();
    const today: Conversation[] = [];
    const yesterday: Conversation[] = [];
    const older: Conversation[] = [];
    convs.forEach(c => {
      const diff = (now - c.createdAt.getTime()) / 86400000;
      if (diff < 1) today.push(c);
      else if (diff < 2) yesterday.push(c);
      else older.push(c);
    });
    this.groups = [
      ...(today.length    ? [{ label: 'Today',     items: today }]     : []),
      ...(yesterday.length ? [{ label: 'Yesterday', items: yesterday }] : []),
      ...(older.length    ? [{ label: 'Earlier',   items: older }]     : []),
    ];
  }
}
