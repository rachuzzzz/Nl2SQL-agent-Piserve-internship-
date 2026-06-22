import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';
import { ChatResponse, Conversation, Message, ResultData, DashboardData, parseChartSignal } from '../models/chat.model';

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

  sendMessage(text: string): Observable<ChatResponse> {
    const conv = this.activeConversation$.value!;
    conv.messages.push({ id: uuid(), role: 'user', text, timestamp: new Date() });
    if (conv.messages.length === 1) {
      conv.title = text.length > 42 ? text.slice(0, 42) + '\u2026' : text;
    }
    this.activeConversation$.next({ ...conv });
    this.loading$.next(true);

    return this.http.post<ChatResponse>(this.API + '/chat', {
      message:    text,
      session_id: conv.id,
    }).pipe(
      tap({
        next: res => {
          // Parse __CHART_AVAILABLE__ signal out of the answer text
          const { cleanText, chartSignal } = parseChartSignal(res.answer);

          const aiMsg: Message = {
            id:          uuid(),
            role:        'ai',
            text:        cleanText,
            timestamp:   new Date(),
            debug:       res,
            resultData:  res.result_data ?? null,
            chartConfig: res.chart_config ?? null,
            chartSignal: chartSignal,
            dashboard:   res.dashboard ?? null,
          };
          conv.messages.push(aiMsg);
          this.activeConversation$.next({ ...conv });
          this.lastDebug$.next(res);
          this.loading$.next(false);
        },
        error: () => { this.loading$.next(false); },
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
    conv.messages = [];
    conv.title    = 'New conversation';
    this.activeConversation$.next({ ...conv });
    this.lastDebug$.next(null);
  }

  fetchPage(sessionId: string, page: number, pageSize: number, sql: string | null): Observable<any> {
    return this.http.post<any>(this.API + '/page', {
      session_id: sessionId,
      page,
      page_size:  pageSize,
      sql:        sql ?? null,
    });
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