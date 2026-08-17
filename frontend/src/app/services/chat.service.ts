import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';
import { ChatResponse, Conversation, Message, ResultData, parseChartSignal } from '../models/chat.model';

function uuid(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly API = '/api';
  conversations$      = new BehaviorSubject<Conversation[]>([]);
  activeConversation$ = new BehaviorSubject<Conversation | null>(null);
  lastDebug$          = new BehaviorSubject<ChatResponse | null>(null);
  loading$            = new BehaviorSubject<boolean>(false);
  // Message id the chat window should scroll to and highlight next — a
  // BehaviorSubject (not a plain Subject) because *ngSwitchCase destroys and
  // recreates ChatWindowComponent on every tab switch, so a freshly
  // constructed instance must still see a jump that was requested just
  // before it was created (see jumpToMessage() below and its consumer in
  // ChatWindowComponent, which resets this back to null once handled).
  scrollToMessage$    = new BehaviorSubject<string | null>(null);

  constructor(private http: HttpClient) {
    this.newConversation();
  }

  newConversation(): void {
    const conv: Conversation = {
      id: uuid(), title: 'New conversation', createdAt: new Date(), messages: [],
    };
    this.conversations$.next([conv, ...this.conversations$.value]);
    this.activeConversation$.next(conv);
    this.lastDebug$.next(null);
  }

  selectConversation(id: string): void {
    const conv = this.conversations$.value.find(c => c.id === id);
    if (!conv) return;
    this.activeConversation$.next(conv);
    const last = [...conv.messages].reverse().find(m => m.role === 'ai');
    this.lastDebug$.next(last?.debug ?? null);
  }

  sendMessage(text: string, selectedPositions?: number[]): Observable<ChatResponse> {
    const conv = this.activeConversation$.value!;
    conv.messages.push({ id: uuid(), role: 'user', text, timestamp: new Date() });
    if (conv.messages.length === 1) {
      conv.title = text.length > 42 ? text.slice(0, 42) + '\u2026' : text;
    }
    this.activeConversation$.next({ ...conv });
    this.loading$.next(true);

    return this.http.post<ChatResponse>(this.API + '/chat', {
      message:            text,
      session_id:         conv.id,
      selected_positions: selectedPositions?.length ? selectedPositions : undefined,
    }).pipe(
      tap({
        next: res => {
          // Parse __CHART_AVAILABLE__ signal out of the answer text
          const { cleanText, chartSignal } = parseChartSignal(res.answer);

          const aiMsg: Message = {
            id:             uuid(),
            role:           'ai',
            text:           cleanText,
            timestamp:      new Date(),
            debug:          res,
            domain:         res.domain ?? null,
            intentRecap:    res.intent_recap ?? null,
            resultData:     res.result_data ?? null,
            chartConfig:    res.chart_config ?? null,
            chartSignal:    chartSignal,
            dashboard:      res.dashboard ?? null,
            sourceQuestion: text,
          };
          conv.messages.push(aiMsg);
          this.activeConversation$.next({ ...conv });
          this.lastDebug$.next(res);
          this.loading$.next(false);
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'An error occurred. Please try again.';
          const errMsg: Message = {
            id: uuid(), role: 'ai', text: `⚠ ${detail}`, timestamp: new Date(),
          };
          conv.messages.push(errMsg);
          this.activeConversation$.next({ ...conv });
          this.loading$.next(false);
        },
      })
    );
  }

  addMessage(msg: Message): void {
    const conv = this.activeConversation$.value;
    if (!conv) return;
    conv.messages.push(msg);
    this.activeConversation$.next({ ...conv });
  }

  clearConversation(): void {
    const conv = this.activeConversation$.value;
    if (!conv) return;
    // Also wipe the backend ConversationSession (scope/entity memory, raw
    // turn history) — without this, /clear only wiped what's shown on
    // screen while the server-side session (including cross-domain scope
    // tracking) silently survived and could leak into what looks like a
    // brand-new chat. Best-effort: a failed delete shouldn't block the
    // visible clear the user is expecting.
    this.http.delete(this.API + '/sessions/' + conv.id).subscribe({ error: () => {} });
    conv.messages = [];
    conv.title    = 'New conversation';
    this.activeConversation$.next({ ...conv });
    this.lastDebug$.next(null);
  }

  reset(): void {
    this.conversations$.next([]);
    this.activeConversation$.next(null);
    this.lastDebug$.next(null);
    this.loading$.next(false);
    this.newConversation();
  }

  fetchPage(sessionId: string, page: number, pageSize: number): Observable<any> {
    return this.http.post<any>(this.API + '/page', {
      session_id: sessionId,
      page,
      page_size:  pageSize,
    });
  }

  // Switches to the conversation containing messageId, then signals
  // ChatWindowComponent to scroll to and highlight it. Used by the
  // Dashboard gallery to jump back to the originating chat message.
  jumpToMessage(conversationId: string, messageId: string): void {
    this.selectConversation(conversationId);
    this.scrollToMessage$.next(messageId);
  }

  updateMessageResultData(messageId: string, data: ResultData): void {
    const conv = this.activeConversation$.value;
    if (!conv) return;
    const msg = conv.messages.find(m => m.id === messageId);
    if (msg) {
      msg.resultData = data;
      this.activeConversation$.next({ ...conv });
    }
  }
}