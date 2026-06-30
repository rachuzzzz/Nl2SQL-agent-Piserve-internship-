import {
  Component, ElementRef, ViewChild, AfterViewChecked,
  ChangeDetectionStrategy, ChangeDetectorRef, NgZone, OnDestroy, OnInit
} from '@angular/core';
import { CommonModule, DatePipe, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { ChatService } from '../../services/chat.service';
import { Message, ResultData, ChartConfig } from '../../models/chat.model';
import { ChartDisplayComponent } from '../chart-display/chart-display.component';
import { ResultTableComponent } from '../result-table/result-table.component';
import { DashboardDisplayComponent } from '../dashboard-display/dashboard-display.component';
import { FormatMsgPipe } from '../../pipes/format-msg.pipe';
import { SlashAutocompleteComponent } from '../slash-autocomplete/slash-autocomplete.component';
import { ColumnPickerComponent } from '../column-picker/column-picker.component';
import { SlashCommandService, ColumnSelection } from '../../services/slash-commands.service';

@Component({
  selector: 'app-chat-window',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule, FormsModule, DatePipe, DecimalPipe,
    ChartDisplayComponent, ResultTableComponent, DashboardDisplayComponent, FormatMsgPipe,
    SlashAutocompleteComponent, ColumnPickerComponent,
  ],
  template: `
    <main class="chat-main">

      <!-- Ambient background — same language as the splash, but whisper-quiet -->
      <div class="chat-bg" aria-hidden="true">
        <div class="cb-orb cb-orb-1"></div>
        <div class="cb-orb cb-orb-2"></div>
        <div class="cb-orb cb-orb-3"></div>
        <div class="cb-grid"></div>
        <div class="cb-particles">
          <span class="cbp cbp-1"></span><span class="cbp cbp-2"></span>
          <span class="cbp cbp-3"></span><span class="cbp cbp-4"></span>
          <span class="cbp cbp-5"></span>
        </div>
      </div>

      <div class="chat-messages" #msgs>

        <div class="welcome" *ngIf="messages.length === 0 && !loading && !inputText.trim()">
          <div class="welcome-icon">
            <!-- Piserve "p" mark -->
            <svg viewBox="0 0 100 100" fill="none" style="overflow:visible">
              <defs>
                <linearGradient id="lg-welcome" x1="20" y1="90" x2="86" y2="10" gradientUnits="userSpaceOnUse">
                  <stop offset="0" stop-color="var(--blue-700)"/>
                  <stop offset="0.55" stop-color="var(--blue-600)"/>
                  <stop offset="1" stop-color="var(--blue-400)"/>
                </linearGradient>
              </defs>
              <path d="M34 36 V87" stroke="url(#lg-welcome)" stroke-width="13" stroke-linecap="round"/>
              <circle cx="52" cy="42" r="18" stroke="url(#lg-welcome)" stroke-width="13" fill="none"/>
              <path d="M67 31 L85 12" stroke="url(#lg-welcome)" stroke-width="13" stroke-linecap="round"/>
            </svg>
          </div>
          <h2>Ask your data anything</h2>
          <p>Type a question in plain English — OctaAI generates SQL,<br>
             executes it, and explains the results automatically.</p>
          <div class="slash-hint">
            Type <kbd>/</kbd> for commands &nbsp;·&nbsp;
            <kbd>/chart</kbd> to visualize &nbsp;·&nbsp;
            <kbd>/export</kbd> to download &nbsp;·&nbsp;
            <kbd>/help</kbd> for all commands
          </div>
          <div class="quick-queries">
            <button class="qq" *ngFor="let q of quickQueries" (click)="send(q)">{{ q }}</button>
          </div>
        </div>

        <div *ngFor="let msg of messages; trackBy: trackMsg"
             class="msg"
             [class.msg-user]="msg.role === 'user'"
             [class.msg-wide]="(msg.role === 'ai' && !!msg.dashboard?.show) || expandedChartMsgs.has(msg.id)">
          <div class="avatar"
            [class.avatar-ai]="msg.role === 'ai'"
            [class.avatar-user]="msg.role === 'user'">
            {{ msg.role === 'user' ? 'U' : 'AI' }}
          </div>
          <div class="msg-content">
            <div class="bubble"
              [class.bubble-ai]="msg.role === 'ai'"
              [class.bubble-user]="msg.role === 'user'"
              [class.bubble-command]="isCommand(msg.text)"
              [innerHTML]="msg.text | formatMsg">
            </div>

            <!-- Dashboard path: cards + chart + table in one animated layout -->
            <app-dashboard-display
              *ngIf="msg.role === 'ai' && msg.dashboard?.show && msg.chartConfig && msg.resultData"
              [cards]="msg.dashboard!.cards"
              [resultData]="msg.resultData!"
              [chartConfig]="msg.chartConfig!"
              [sessionId]="sessionId"
              [messageId]="msg.id">
            </app-dashboard-display>

            <!-- Non-dashboard path: existing "Show as chart" + table + chart -->
            <ng-container *ngIf="!(msg.role === 'ai' && msg.dashboard?.show)">

              <!-- "Show as chart" button — appears when backend signals a chart is available -->
              <div class="chart-cta" *ngIf="msg.role === 'ai' && msg.chartSignal && !msg.chartConfig">
                <button class="btn-show-chart" (click)="showChart(msg)">
                  <svg viewBox="0 0 16 16" fill="currentColor" width="13" height="13">
                    <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v3a1 1 0 01-1 1H3a1 1 0 01-1-1v-3zM6 6a1 1 0 011-1h2a1 1 0 011 1v8a1 1 0 01-1 1H7a1 1 0 01-1-1V6zm6-4a1 1 0 00-1 1v11a1 1 0 102 0V3a1 1 0 00-1-1z"/>
                  </svg>
                  Show as chart
                </button>
              </div>

              <app-result-table
                *ngIf="msg.role === 'ai' && msg.resultData && msg.resultData.rows.length > 0"
                [data]="msg.resultData"
                [sessionId]="sessionId"
                [messageId]="msg.id"
                (drilldown)="send($event)">
              </app-result-table>

              <app-chart-display
                *ngIf="msg.role === 'ai' && msg.chartConfig"
                [cfg]="msg.chartConfig!"
                [sessionId]="sessionId"
                [mode]="msg.chartMode ?? 'quick'"
                [noOverlay]="true"
                [resizeTrigger]="chartResizeTriggers.get(msg.id) ?? 0"
                (fullscreenChange)="onMsgChartFullscreen(msg.id, $event)">
              </app-chart-display>

            </ng-container>

            <div class="msg-meta">
              <button *ngIf="msg.role === 'ai'"
                class="btn-speak"
                [class.btn-speak--loading]="ttsLoadingId === msg.id"
                [class.btn-speak--playing]="ttsPlayingId === msg.id"
                [title]="ttsPlayingId === msg.id ? 'Stop narration' : 'Read aloud'"
                (click)="toggleTTS(msg)">
                {{ ttsLoadingId === msg.id ? '⏳' : ttsPlayingId === msg.id ? '⏹' : '🔊' }}
              </button>
              <span class="msg-time">{{ msg.timestamp | date:'shortTime' }}</span>
              <span *ngIf="msg.debug as d" class="msg-pill"
                [class.msg-pill-warn]="(d.sql_errors || 0) > 0">
                ✓ {{ d.llm_calls || 1 }} call{{ (d.llm_calls || 1) !== 1 ? 's' : '' }}
                · {{ ((d.time_ms || 0) / 1000).toFixed(1) }}s
                · {{ d.total_tokens || 0 | number }} tok
              </span>
            </div>
          </div>
        </div>
        <div class="msg inline-thinking" *ngIf="loading">
          <svg viewBox="0 0 100 100" fill="none" width="20" height="20" style="overflow:visible;flex-shrink:0">
            <defs>
              <linearGradient id="lg-think" x1="20" y1="90" x2="86" y2="10" gradientUnits="userSpaceOnUse">
                <stop offset="0" stop-color="#1129A8"/>
                <stop offset="0.55" stop-color="#1B43FF"/>
                <stop offset="1" stop-color="#5C73FF"/>
              </linearGradient>
            </defs>
            <g style="opacity:.2">
              <path d="M34 36 V87" stroke="url(#lg-think)" stroke-width="13" stroke-linecap="round"/>
              <circle cx="52" cy="42" r="18" stroke="url(#lg-think)" stroke-width="13" fill="none"/>
              <path d="M67 31 L85 12" stroke="url(#lg-think)" stroke-width="13" stroke-linecap="round"/>
            </g>
            <path d="M34 87 L34 42 A18 18 0 1 1 70 42 A18 18 0 1 1 34 42 A18 18 0 0 1 67 31 L85 12"
              fill="none" stroke="url(#lg-think)" stroke-width="13" stroke-linecap="round"
              stroke-dasharray="26 74"
              style="animation: om-worm-run 1.3s linear infinite"/>
          </svg>
          <span class="inline-think-text">{{ loadingMessage }}</span>
        </div>

      </div>

      <app-column-picker
        *ngIf="showColumnPicker"
        [allCols]="columnPickerInfo?.columns ?? []"
        [numericCols]="columnPickerInfo?.numerics ?? []"
        [dateCols]="columnPickerInfo?.dates ?? []"
        [catCols]="columnPickerInfo?.categories ?? []"
        (confirmed)="onColumnPickerConfirmed($event)"
        (cancel)="showColumnPicker = false">
      </app-column-picker>

      <div class="chat-input-area">
        <div class="mic-error" *ngIf="micError" (click)="micError=''">
          ⚠ {{ micError }}
        </div>
        <div class="input-wrap" [class.focused]="focused" style="position: relative">
          <app-slash-autocomplete
            #autocomplete
            [input]="inputText"
            (chosen)="onAutoCompleteChosen($event)">
          </app-slash-autocomplete>
          <textarea #inputEl class="chat-input" [(ngModel)]="inputText"
            [placeholder]="inputPlaceholder"
            rows="1"
            (keydown)="onKeydown($event)"
            (input)="autoResize($event)"
            (focus)="focused=true"
            (blur)="focused=false">
          </textarea>
          <!-- Mic button -->
          <button class="btn-mic"
            [class.btn-mic--recording]="isRecording"
            [class.btn-mic--busy]="isTranscribing"
            [disabled]="loading || isTranscribing"
            (click)="toggleRecording()"
            [title]="isRecording ? 'Stop recording' : 'Speak your query'">
            <!-- idle: mic icon -->
            <svg *ngIf="!isRecording && !isTranscribing" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="9" y="2" width="6" height="11" rx="3"/>
              <path d="M19 10a7 7 0 01-14 0"/>
              <line x1="12" y1="19" x2="12" y2="23"/>
              <line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
            <!-- recording: stop square -->
            <svg *ngIf="isRecording" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" rx="2"/>
            </svg>
            <!-- transcribing: spinner -->
            <svg *ngIf="isTranscribing && !isRecording" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="2.5" class="spin">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83
                       M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
            </svg>
          </button>

          <button class="btn-send" [disabled]="loading || !inputText.trim()" (click)="sendInput()">
            <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5"
              stroke-linecap="round" stroke-linejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"/>
              <polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        </div>
      </div>
    </main>
  `,
  styles: [`

    /* ── Ambient background layer ── */
    .chat-bg {
      position: absolute; inset: 0; pointer-events: none; z-index: 0; overflow: hidden;
    }

    /* Orbs — light mode: barely-there cobalt blue halos */
    .cb-orb { position: absolute; border-radius: 50%; filter: blur(90px); }
    .cb-orb-1 {
      width: 560px; height: 560px; background: rgba(27,67,255,0.055);
      top: -200px; right: -140px;
      animation: cb-drift 12s ease-in-out infinite alternate;
    }
    .cb-orb-2 {
      width: 420px; height: 420px; background: rgba(92,115,255,0.038);
      bottom: -140px; left: -100px;
      animation: cb-drift 15s ease-in-out 2.5s infinite alternate-reverse;
    }
    .cb-orb-3 {
      width: 280px; height: 280px; background: rgba(27,67,255,0.028);
      top: 45%; left: 35%;
      animation: cb-pulse 9s ease-in-out infinite;
    }
    @keyframes cb-drift {
      from { transform: translate(0,0) scale(1); }
      to   { transform: translate(28px,18px) scale(1.08); }
    }
    @keyframes cb-pulse {
      0%,100% { opacity: 0.5; transform: translate(-50%,-50%) scale(1); }
      50%      { opacity: 1;   transform: translate(-50%,-50%) scale(1.18); }
    }

    /* Dot grid */
    .cb-grid {
      position: absolute; inset: 0;
      background-image: radial-gradient(circle, rgba(27,67,255,0.07) 1px, transparent 1px);
      background-size: 38px 38px;
      mask-image: radial-gradient(ellipse 85% 75% at 65% 25%, rgba(0,0,0,0.6) 0%, transparent 100%);
    }

    /* Floating particles — light mode */
    .cb-particles { position: absolute; inset: 0; }
    .cbp {
      position: absolute; border-radius: 50%;
      background: rgba(27,67,255,0.22);
      box-shadow: 0 0 5px 1px rgba(27,67,255,0.12);
      animation: cbp-float linear infinite;
    }
    .cbp-1 { width:3px; height:3px; left:82%; top:88%; animation-duration:13s; animation-delay:0s; }
    .cbp-2 { width:2px; height:2px; left:18%; top:65%; animation-duration:17s; animation-delay:4s; }
    .cbp-3 { width:3px; height:3px; left:60%; top:78%; animation-duration:11s; animation-delay:2s;
              background:rgba(92,115,255,0.25); box-shadow:0 0 5px 1px rgba(92,115,255,0.12); }
    .cbp-4 { width:2px; height:2px; left:40%; top:90%; animation-duration:20s; animation-delay:7s; }
    .cbp-5 { width:3px; height:3px; left:92%; top:55%; animation-duration:14s; animation-delay:1s; }
    @keyframes cbp-float {
      0%   { transform: translateY(0)    scale(1);   opacity: 0; }
      8%   { opacity: 0.9; }
      92%  { opacity: 0.4; }
      100% { transform: translateY(-280px) scale(0.4); opacity: 0; }
    }

    /* Dark mode — same elements, more visible */
    :host-context([data-theme="dark"]) .cb-orb-1 { background: rgba(27,67,255,0.11); }
    :host-context([data-theme="dark"]) .cb-orb-2 { background: rgba(92,115,255,0.08); }
    :host-context([data-theme="dark"]) .cb-orb-3 { background: rgba(27,67,255,0.06); }
    :host-context([data-theme="dark"]) .cb-grid {
      background-image: radial-gradient(circle, rgba(27,67,255,0.12) 1px, transparent 1px);
    }
    :host-context([data-theme="dark"]) .cbp {
      background: rgba(27,67,255,0.5);
      box-shadow: 0 0 7px 2px rgba(27,67,255,0.28);
    }
    :host-context([data-theme="dark"]) .cbp-3 {
      background: rgba(92,115,255,0.5);
      box-shadow: 0 0 7px 2px rgba(92,115,255,0.28);
    }

    /* Push actual content above the bg layer */
    .chat-messages {
      position: relative; z-index: 1;
      padding-bottom: 16px;
      /* soft bottom fade into the floating pill */
      mask-image: linear-gradient(to bottom, black 85%, transparent 100%);
      -webkit-mask-image: linear-gradient(to bottom, black 85%, transparent 100%);
    }
    .chat-input-area { position: relative; z-index: 2; }

    .slash-hint {
      font-size: 12px; color: var(--text-3);
      margin: 8px 0 16px;
    }
    kbd {
      display: inline-block; padding: 1px 7px; border-radius: 5px;
      background: var(--surface-2); border: 1px solid var(--border);
      font-family: var(--font-mono); font-size: 11px; color: var(--text-2);
    }
    .bubble-command {
      font-family: var(--font-mono) !important;
      font-size: 13px !important; background: var(--surface-2) !important;
      color: var(--text-2) !important; border: 1px solid var(--border) !important;
      border-radius: 8px !important; padding: 6px 12px !important;
    }

    /* ── Inline thinking ── */
    .inline-thinking {
      display: flex; align-items: center; gap: 10px;
      padding: 4px 0; animation: om-rise .3s var(--ease);
    }
    .inline-think-text {
      font-size: 13.5px; font-weight: 500; color: var(--text-2);
      animation: om-text-in .35s var(--ease);
    }

    /* ── Mic error banner ── */
    .mic-error {
      font-size: 12px; color: var(--warn); background: color-mix(in srgb, var(--warn) 8%, transparent);
      border: 1px solid color-mix(in srgb, var(--warn) 28%, transparent); border-radius: 8px;
      padding: 8px 12px; margin-bottom: 8px; cursor: pointer;
      line-height: 1.5;
    }

    /* ── Mic / ASR button ── */
    .btn-mic {
      width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
      background: transparent; border: 1.5px solid rgba(154,163,200,0.35);
      color: var(--text-3); cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      transition: all 0.18s ease;
      svg { width: 14px; height: 14px; }
    }
    .btn-mic:hover:not(:disabled) {
      background: rgba(27,67,255,0.07); border-color: rgba(27,67,255,0.40); color: var(--accent);
    }
    .btn-mic--recording {
      background: var(--red-bg) !important;
      border-color: color-mix(in srgb, var(--err) 35%, transparent) !important;
      color: var(--err) !important;
      animation: mic-pulse 1.2s ease-in-out infinite;
    }
    .btn-mic--busy { opacity: 0.6; cursor: wait !important; }
    @keyframes mic-pulse {
      0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--err) 0%, transparent); }
      50%       { box-shadow: 0 0 0 5px color-mix(in srgb, var(--err) 18%, transparent); }
    }
    .spin { animation: spin 0.9s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Speaker / TTS button ── */
    .btn-speak {
      background: none; border: none; cursor: pointer;
      padding: 2px 5px; border-radius: 5px; font-size: 12px;
      opacity: 0.35; transition: opacity 0.15s, background 0.15s; line-height: 1;
    }
    .btn-speak:hover { opacity: 1; background: rgba(0,0,0,0.05); }
    .btn-speak--playing { opacity: 1; color: var(--accent); }
    .btn-speak--loading { opacity: 0.7; cursor: wait; }

    /* ── Show as chart button ── */
    .chart-cta { margin-top: 8px; }
    .btn-show-chart {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 600;
      background: var(--blue-50); border: 1px solid var(--blue-100);
      color: var(--blue-600); cursor: pointer; transition: all 0.15s;
      font-family: var(--font);
    }
    .btn-show-chart:hover {
      background: var(--blue-100); border-color: var(--blue-200);
    }
  `],
})
export class ChatWindowComponent implements AfterViewChecked {
  @ViewChild('msgs')         msgsEl!:         ElementRef;
  @ViewChild('inputEl')      inputEl!:         ElementRef;
  @ViewChild('autocomplete') autocompleteRef!: SlashAutocompleteComponent;

  messages:  Message[] = [];
  sessionId            = '';
  inputText            = '';
  loading              = false;
  loadingMessage       = 'Parsing your question…';
  private loadingInterval: any;
  private readonly loadingPhrases = [
    'Parsing your question…',
    'Resolving schema…',
    'Generating SQL…',
    'Validating & executing…',
    'Summarizing results…',
  ];
  focused              = false;

  showColumnPicker   = false;
  pendingChartType: string | null = null;
  columnPickerInfo: { columns: string[]; categories: string[]; numerics: string[]; dates: string[] } | null = null;

  expandedChartMsgs   = new Set<string>();
  chartResizeTriggers = new Map<string, number>();

  isRecording    = false;
  isTranscribing = false;
  micError       = '';
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];

  ttsPlayingId:  string | null = null;
  ttsLoadingId:  string | null = null;
  private _ttsAudio: HTMLAudioElement | null = null;
  private readonly API = '/api';

  private lastResultData: ResultData | null = null;
  private lastMsgCount     = 0;
  private lastLoadingState = false;

  quickQueries = [
    'What is the average inspection score?',
    'Show overdue corrective actions',
    'Which facility has the most high-risk CAs?',
    'How are we doing overall?',
    'Rank inspection types by avg CAs per inspection',
  ];

  get inputPlaceholder(): string {
    return this.lastResultData
      ? 'Ask a question or type / for commands (/chart, /export, /sql...)'
      : 'Ask about inspections, scores, corrective actions, facilities...';
  }

  constructor(
    public svc: ChatService,
    private cdr: ChangeDetectorRef,
    private slashSvc: SlashCommandService,
    private http: HttpClient,
    private zone: NgZone
  ) {
    svc.activeConversation$.subscribe(c => {
      this.messages  = c?.messages ?? [];
      this.sessionId = c?.id ?? '';
      const lastAi = [...this.messages].reverse().find(m => m.role === 'ai' && m.resultData);
      this.lastResultData = lastAi?.resultData ?? null;
      this.cdr.markForCheck();
    });
    svc.loading$.subscribe(l => {
      if (l === this.loading) return; // Prevent duplicate triggers
      this.loading = l;
      if (l) {
        this.loadingMessage = this.loadingPhrases[0];
        let idx = 0;
        if (this.loadingInterval) clearInterval(this.loadingInterval);
        this.loadingInterval = setInterval(() => {
          idx = Math.min(idx + 1, this.loadingPhrases.length - 1);
          this.loadingMessage = this.loadingPhrases[idx];
          this.cdr.detectChanges();
        }, 1500);
      } else {
        if (this.loadingInterval) {
           clearInterval(this.loadingInterval);
           this.loadingInterval = null;
        }
      }
      this.cdr.markForCheck();
    });
  }
  
  ngOnDestroy() {
    if (this.loadingInterval) clearInterval(this.loadingInterval);
    if (this._ttsAudio) { this._ttsAudio.pause(); this._ttsAudio = null; }
  }

  ngAfterViewChecked(): void {
    const el = this.msgsEl?.nativeElement;
    if (!el) return;
    const changed = this.messages.length !== this.lastMsgCount
                 || this.loading          !== this.lastLoadingState;
    if (changed) {
      this.lastMsgCount     = this.messages.length;
      this.lastLoadingState = this.loading;
      requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
    }
  }

  trackMsg(_: number, msg: Message): string { return msg.id; }
  isCommand(text: string): boolean { return text.trim().startsWith('/'); }

  onKeydown(e: KeyboardEvent): void {
    if (this.autocompleteRef?.handleKey(e)) return;
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      this.sendInput();
    }
  }

  onAutoCompleteChosen(completion: string): void {
    this.inputText = completion;
    this.cdr.markForCheck();
    setTimeout(() => this.inputEl?.nativeElement?.focus(), 0);
  }

  /** Called when user clicks "Show as chart" button on a message */
  showChart(msg: Message): void {
    this.http.post<ChartConfig>(`${this.API}/chart`, {
      session_id: this.sessionId,
      chart_type: msg.chartSignal?.recommendedType ?? null,
    }).subscribe({
      next: cfg => {
        msg.chartConfig = cfg;
        msg.chartSignal = null;
        msg.chartMode   = 'quick';
        this.svc.activeConversation$.next({ ...this.svc.activeConversation$.value! });
        this.cdr.markForCheck();
      },
      error: () => {
        // silently ignore — button stays visible for retry
      },
    });
  }

  // ── TTS / narration ──────────────────────────────────────────────────────
  toggleTTS(msg: Message): void {
    if (this._ttsAudio) {
      this._ttsAudio.pause();
      this._ttsAudio = null;
      const wasId = this.ttsPlayingId;
      this.ttsPlayingId = null;
      this.ttsLoadingId = null;
      this.cdr.markForCheck();
      if (wasId === msg.id) return;  // same button — toggle off
    }

    this.ttsLoadingId = msg.id;
    this.cdr.markForCheck();

    // strip HTML tags from rendered message text
    const tmp = document.createElement('div');
    tmp.innerHTML = msg.text;
    let text = (tmp.textContent || tmp.innerText || msg.text).trim();

    // Append chart description if this message has an associated chart
    if (msg.chartConfig?.chart_type) {
      const ct  = msg.chartConfig.chart_type.replace(/_/g, ' ');
      const xl  = msg.chartConfig.x_label?.replace(/_/g, ' ') ?? null;
      const yls = (msg.chartConfig.y_labels ?? []).map(l => l.replace(/_/g, ' '));
      let chartDesc = `You can see the ${ct} to the left`;
      if (xl)          chartDesc += `, with ${xl} on the horizontal axis`;
      if (yls.length)  chartDesc += ` and ${yls.join(', ')} on the vertical axis`;
      chartDesc += '.';
      text = text + ' ' + chartDesc;
    }

    this.http.post(`${this.API}/tts`, { text }, { responseType: 'blob' }).subscribe({
      next: (blob: Blob) => {
        const url = URL.createObjectURL(blob);
        this._ttsAudio   = new Audio(url);
        this.ttsLoadingId = null;
        this.ttsPlayingId = msg.id;
        this.cdr.markForCheck();
        this._ttsAudio.onended = () => {
          this.ttsPlayingId = null;
          this._ttsAudio    = null;
          URL.revokeObjectURL(url);
          this.cdr.markForCheck();
        };
        this._ttsAudio.play();
      },
      error: () => {
        this.ttsLoadingId = null;
        this.ttsPlayingId = null;
        this.cdr.markForCheck();
      },
    });
  }

  // ── ASR / voice input ────────────────────────────────────────────────────
  toggleRecording(): void {
    this.micError = '';
    if (this.isRecording) {
      this.mediaRecorder?.stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      this.micError = 'Microphone requires HTTPS. Open Chrome, go to chrome://flags/#unsafely-treat-insecure-origin-as-secure, add http://10.77.20.13 and relaunch.';
      this.cdr.markForCheck();
      return;
    }
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      this.audioChunks = [];
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus' : 'audio/webm';
      this.mediaRecorder = new MediaRecorder(stream, { mimeType });
      this.mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) this.audioChunks.push(e.data);
      };
      this.mediaRecorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(this.audioChunks, { type: mimeType });
        this.isRecording = false;
        this._transcribe(blob);
      };
      this.mediaRecorder.start();
      this.isRecording = true;
      this.cdr.markForCheck();
    }).catch((err: Error) => {
      this.isRecording = false;
      this.micError = err.name === 'NotAllowedError'
        ? 'Microphone permission denied. Allow it in your browser settings.'
        : `Mic error: ${err.message}`;
      this.cdr.markForCheck();
    });
  }

  private _transcribe(blob: Blob): void {
    this.isTranscribing = true;
    this.cdr.markForCheck();
    const form = new FormData();
    form.append('audio', blob, 'recording.webm');
    this.http.post<{ text: string }>(`${this.API}/transcribe`, form).subscribe({
      next: res => {
        const t = res.text?.trim();
        if (t) this.inputText = (this.inputText + ' ' + t).trim();
        this.isTranscribing = false;
        this.cdr.markForCheck();
        setTimeout(() => this.inputEl?.nativeElement?.focus(), 0);
      },
      error: () => {
        this.isTranscribing = false;
        this.cdr.markForCheck();
      },
    });
  }

  onMsgChartFullscreen(msgId: string, isFullscreen: boolean): void {
    if (isFullscreen) {
      this.expandedChartMsgs.add(msgId);
    } else {
      this.expandedChartMsgs.delete(msgId);
    }
    this.chartResizeTriggers.set(msgId, (this.chartResizeTriggers.get(msgId) ?? 0) + 1);
    this.cdr.markForCheck();
  }

  send(text: string): void {
    if (!text.trim() || this.loading) return;
    if (this.slashSvc.isSlashCommand(text)) {
      this.inputText = '';
      if (this.inputEl) this.inputEl.nativeElement.style.height = 'auto';
      this.handleSlashCommand(text.trim());
      return;
    }
    this.svc.sendMessage(text).subscribe();
    this.inputText = '';
    if (this.inputEl) this.inputEl.nativeElement.style.height = 'auto';
  }

  sendInput(): void { this.send(this.inputText); }
  onEnter(e: any): void { if (!e.shiftKey) { e.preventDefault(); this.sendInput(); } }
  autoResize(e: Event): void {
    const el = e.target as HTMLTextAreaElement;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  private handleSlashCommand(raw: string): void {
    const parsed = this.slashSvc.parse(raw);
    if (!parsed) return;

    switch (parsed.type) {
      case 'chart': {
        this.addUserMessage(raw);
        const cols = this.lastResultData?.columns ?? [];
        if (cols.length > 4 && !parsed.chartType) {
          this.slashSvc.fetchColumns(this.sessionId).subscribe({
            next: (info) => {
              this.columnPickerInfo = info;
              this.pendingChartType = null;
              this.showColumnPicker = true;
              this.cdr.markForCheck();
            },
            error: () => this.addAiMessage('No result in session. Ask a data question first.'),
          });
          return;
        }
        this.slashSvc.fetchChart(this.sessionId, parsed.chartType ?? null).subscribe({
          next: (cfg) => {
            const label = parsed.chartType
              ? parsed.chartType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
              : 'best fit';
            let msg = `Here's the ${label} chart.`;
            if (cfg.confidence === 'low' && parsed.chartType) {
              msg += ' ⚠ This chart type may not suit your data — the advisor recommends a different shape. The result might look unusual.';
            }
            this.addAiMessage(msg, cfg, 'control');
          },
          error: (err) => this.addAiMessage(err.error?.detail ?? 'Could not generate chart. Run a query first.'),
        });
        break;
      }
      case 'sql': {
        this.addUserMessage(raw);
        this.slashSvc.fetchSql(this.sessionId).subscribe({
          next: (sql) => this.addAiMessage('```sql\n' + sql + '\n```'),
          error: ()    => this.addAiMessage('No SQL available. Ask a data question first.'),
        });
        break;
      }
      case 'export': {
        this.addUserMessage(raw);
        if (!this.lastResultData?.rows?.length) {
          this.addAiMessage('No data to export. Ask a data question first.');
          return;
        }
        this.slashSvc.exportCsv(this.lastResultData.columns, this.lastResultData.rows);
        this.addAiMessage('Exported ' + this.lastResultData.row_count + ' rows as CSV.');
        break;
      }
      case 'clear': {
        this.svc.clearConversation();
        this.lastResultData   = null;
        this.showColumnPicker = false;
        this.cdr.markForCheck();
        break;
      }
      case 'help': {
        this.addUserMessage(raw);
        this.addAiMessage(this.slashSvc.buildHelpText());
        break;
      }
      case 'unknown': {
        this.addAiMessage('Unknown command `' + raw + '`. Type `/help` to see available commands.');
        break;
      }
    }
  }

  onColumnPickerConfirmed(selection: ColumnSelection): void {
    this.showColumnPicker = false;
    this.slashSvc.fetchChart(this.sessionId, this.pendingChartType, selection.xCol, selection.yCols).subscribe({
      next: (cfg) => this.addAiMessage('Here\'s your customised chart.', cfg, 'control'),
      error: (err) => this.addAiMessage(err.error?.detail ?? 'Could not generate chart.'),
    });
  }

  private addUserMessage(text: string): void {
    const msg: Message = {
      id:        Math.random().toString(36).slice(2) + Date.now().toString(36),
      role:      'user',
      text,
      timestamp: new Date(),
    };
    this.svc.addMessage(msg);
    this.cdr.markForCheck();
  }

  private addAiMessage(text: string, chartConfig?: ChartConfig, chartMode?: 'quick' | 'control'): void {
    const msg: Message = {
      id:          Math.random().toString(36).slice(2) + Date.now().toString(36),
      role:        'ai',
      text,
      timestamp:   new Date(),
      chartConfig: chartConfig ?? null,
      chartMode:   chartMode ?? (chartConfig ? 'quick' : undefined),
    };
    this.svc.addMessage(msg);
    this.cdr.markForCheck();
  }
}