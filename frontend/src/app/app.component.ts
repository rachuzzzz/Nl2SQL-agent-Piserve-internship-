import { Component, Inject, PLATFORM_ID, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { ChatWindowComponent } from './components/chat-window/chat-window.component';
import { DebugPanelComponent } from './components/debug-panel/debug-panel.component';
import { HistorySidebarComponent } from './components/history-sidebar/history-sidebar.component';
import { DashboardViewComponent } from './components/dashboard-view/dashboard-view.component';
import { SavedViewComponent } from './components/saved-view/saved-view.component';
import { SchemaViewComponent } from './components/schema-view/schema-view.component';
import { AuthService } from './services/auth.service';
import { ChatService } from './services/chat.service';

const TOKEN_KEY    = 'octaai_access_token';
const USERNAME_KEY = 'octaai_username';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, ChatWindowComponent, DebugPanelComponent, HistorySidebarComponent,
            DashboardViewComponent, SavedViewComponent, SchemaViewComponent],
  template: `
    <!-- ── Login page ── -->
    <div class="login-screen" *ngIf="showLogin">
      <div class="oa-orb oa-orb-1"></div>
      <div class="oa-orb oa-orb-2"></div>
      <div class="oa-grid"></div>
      <div class="login-card">
        <!-- Piserve "p" mark -->
        <div class="login-logo-ring">
          <svg viewBox="0 0 100 100" fill="none" style="overflow:visible">
            <defs>
              <linearGradient id="lg-login" x1="20" y1="90" x2="86" y2="10" gradientUnits="userSpaceOnUse">
                <stop offset="0" stop-color="#1129A8"/>
                <stop offset="0.55" stop-color="#1B43FF"/>
                <stop offset="1" stop-color="#5C73FF"/>
              </linearGradient>
            </defs>
            <path d="M34 36 V87" stroke="url(#lg-login)" stroke-width="13" stroke-linecap="round"/>
            <circle cx="52" cy="42" r="18" stroke="url(#lg-login)" stroke-width="13" fill="none"/>
            <path d="M67 31 L85 12" stroke="url(#lg-login)" stroke-width="13" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="login-brand">OctaAI</div>
        <div class="login-sub">Natural Language · SQL Intelligence</div>

        <form class="login-form" (ngSubmit)="doLogin()" #loginForm="ngForm">
          <div class="field-wrap">
            <label class="field-label">Username</label>
            <div class="field-inner" [class.field-focus]="userFocus">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                <path stroke-linecap="round" stroke-linejoin="round" d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 000 8z"/>
              </svg>
              <input
                type="text" name="username" [(ngModel)]="loginUsername"
                placeholder="Enter your username" autocomplete="username" required
                (focus)="userFocus=true" (blur)="userFocus=false" />
            </div>
          </div>

          <div class="field-wrap">
            <label class="field-label">Password</label>
            <div class="field-inner" [class.field-focus]="passFocus">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                <path stroke-linecap="round" d="M7 11V7a5 5 0 0110 0v4"/>
              </svg>
              <input
                [type]="showPass ? 'text' : 'password'" name="password" [(ngModel)]="loginPassword"
                placeholder="Enter your password" autocomplete="current-password" required
                (focus)="passFocus=true" (blur)="passFocus=false" />
              <button type="button" class="pass-toggle" (click)="showPass=!showPass" tabindex="-1">
                <svg *ngIf="!showPass" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
                </svg>
                <svg *ngIf="showPass" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                  <path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24M1 1l22 22"/>
                </svg>
              </button>
            </div>
          </div>

          <div class="login-error" *ngIf="loginError">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            {{ loginError }}
          </div>

          <button type="submit" class="login-btn" [disabled]="loginLoading || !loginUsername || !loginPassword">
            <span *ngIf="!loginLoading">Sign In</span>
            <span *ngIf="loginLoading" class="login-spinner"></span>
          </button>
        </form>
      </div>
    </div>

    <!-- ── OctaAI splash ── -->
    <div class="oa-splash" *ngIf="showSplash" [class.pan-away]="splashOut">
      <div class="oa-orb oa-orb-1"></div>
      <div class="oa-orb oa-orb-2"></div>
      <div class="oa-orb oa-orb-3"></div>
      <div class="oa-grid"></div>

      <div class="oa-splash-center">
        <!-- Spinning Piserve "p" mark -->
        <div class="oa-splash-mark">
          <svg viewBox="0 0 100 100" fill="none" style="overflow:visible">
            <defs>
              <linearGradient id="lg-splash" x1="20" y1="90" x2="86" y2="10" gradientUnits="userSpaceOnUse">
                <stop offset="0" stop-color="#1129A8"/>
                <stop offset="0.55" stop-color="#1B43FF"/>
                <stop offset="1" stop-color="#94A5FF"/>
              </linearGradient>
              <filter id="glow-splash" x="-60%" y="-60%" width="220%" height="220%">
                <feGaussianBlur stdDeviation="3" result="b"/>
                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>
            <!-- static ghost -->
            <g style="opacity:.18" filter="url(#glow-splash)">
              <path d="M34 36 V87" stroke="url(#lg-splash)" stroke-width="13" stroke-linecap="round"/>
              <circle cx="52" cy="42" r="18" stroke="url(#lg-splash)" stroke-width="13" fill="none"/>
              <path d="M67 31 L85 12" stroke="url(#lg-splash)" stroke-width="13" stroke-linecap="round"/>
            </g>
            <!-- animated worm -->
            <path d="M34 87 L34 42 A18 18 0 1 1 70 42 A18 18 0 1 1 34 42 A18 18 0 0 1 67 31 L85 12"
              fill="none" stroke="url(#lg-splash)" stroke-width="13" stroke-linecap="round"
              stroke-dasharray="26 74"
              style="animation: om-worm-run 1.3s linear infinite"/>
          </svg>
        </div>

        <div class="oa-splash-wordmark">OctaAI</div>
        <div class="oa-splash-tagline">{{ splashStageLabel }}</div>

        <!-- Progress bar -->
        <div class="oa-progress-track">
          <div class="oa-progress-bar" [style.width.%]="splashProgress"></div>
        </div>
      </div>

      <!-- Faint giant watermark -->
      <div class="oa-bg-mark" aria-hidden="true">
        <svg viewBox="0 0 100 100" fill="none" style="width:100%;height:100%;overflow:visible">
          <path d="M34 36 V87" stroke="#1B43FF" stroke-width="6" stroke-linecap="round"/>
          <circle cx="52" cy="42" r="18" stroke="#1B43FF" stroke-width="6" fill="none"/>
          <path d="M67 31 L85 12" stroke="#1B43FF" stroke-width="6" stroke-linecap="round"/>
        </svg>
      </div>
    </div>

    <!-- ── App shell ── -->
    <div class="app-shell">
      <header class="app-header">
        <!-- Brand -->
        <div class="brand">
          <div class="brand-logo">
            <svg viewBox="0 0 100 100" fill="none" style="overflow:visible">
              <defs>
                <linearGradient id="lg-brand" x1="20" y1="90" x2="86" y2="10" gradientUnits="userSpaceOnUse">
                  <stop offset="0" stop-color="rgba(255,255,255,.55)"/>
                  <stop offset="1" stop-color="#ffffff"/>
                </linearGradient>
              </defs>
              <path d="M34 36 V87" stroke="url(#lg-brand)" stroke-width="13" stroke-linecap="round"/>
              <circle cx="52" cy="42" r="18" stroke="url(#lg-brand)" stroke-width="13" fill="none"/>
              <path d="M67 31 L85 12" stroke="url(#lg-brand)" stroke-width="13" stroke-linecap="round"/>
            </svg>
          </div>
          <div>
            <span class="brand-name">OctaAI</span>
            <span class="brand-sub">NL · SQL Intelligence</span>
          </div>
        </div>

        <!-- Tab navigation -->
        <nav class="header-nav">
          <button class="nav-tab" [class.active]="activeTab==='chat'" (click)="activeTab='chat'">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
            </svg>
            AI Chat
          </button>
          <button class="nav-tab" [class.active]="activeTab==='dashboard'" (click)="activeTab='dashboard'">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="4"/>
              <rect x="14" y="10" width="7" height="9"/><rect x="3" y="13" width="7" height="7"/>
            </svg>
            Dashboard
          </button>
          <button class="nav-tab" [class.active]="activeTab==='saved'" (click)="activeTab='saved'">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>
            </svg>
            Saved Queries
          </button>
          <button class="nav-tab" [class.active]="activeTab==='schema'" (click)="activeTab='schema'">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
              <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
            </svg>
            Schema
          </button>
        </nav>

        <!-- Right controls -->
        <div class="header-badges">
          <span class="badge badge-blue">
            <span class="bdot" style="background:var(--blue-300)"></span>
            {{ modelLabel }}
          </span>
          <span class="badge badge-green">
            <span class="live-dot"></span>
            Live
          </span>
          <div class="user-pill" *ngIf="loggedInUser">
            <div class="user-avatar">{{ userInitials }}</div>
            <span class="user-name">{{ loggedInUser }}</span>
            <button class="btn-logout" (click)="logout()" title="Sign out">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/>
                <polyline points="16 17 21 12 16 7"/>
                <line x1="21" y1="12" x2="9" y2="12"/>
              </svg>
            </button>
          </div>
          <button class="btn-theme" (click)="toggleTheme()" [title]="isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'">
            <svg *ngIf="!isDark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>
            <svg *ngIf="isDark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
          </button>
        </div>
      </header>

      <div class="app-body">
        <ng-container [ngSwitch]="activeTab">
          <ng-container *ngSwitchCase="'chat'">
            <app-history-sidebar></app-history-sidebar>
            <app-chat-window></app-chat-window>
            <app-debug-panel></app-debug-panel>
          </ng-container>
          <app-dashboard-view *ngSwitchCase="'dashboard'" (switchToChat)="activeTab='chat'"></app-dashboard-view>
          <app-saved-view     *ngSwitchCase="'saved'"></app-saved-view>
          <app-schema-view    *ngSwitchCase="'schema'"></app-schema-view>
        </ng-container>
      </div>
    </div>
  `,
  styles: [`
    /* ── Shared: orbs + grid used by both splash + login ── */
    .oa-orb {
      position: absolute; border-radius: 50%; filter: blur(90px); pointer-events: none;
    }
    .oa-orb-1 {
      width: 560px; height: 560px; background: rgba(27,67,255,0.18);
      top: -180px; left: -160px;
      animation: oa-orb-drift 7s ease-in-out infinite alternate;
    }
    .oa-orb-2 {
      width: 440px; height: 440px; background: rgba(94,115,255,0.12);
      bottom: -140px; right: -120px;
      animation: oa-orb-drift 9s ease-in-out 1s infinite alternate-reverse;
    }
    .oa-orb-3 {
      width: 320px; height: 320px; background: rgba(27,67,255,0.09);
      top: 50%; left: 50%; transform: translate(-50%,-50%);
      animation: oa-orb-pulse 5s ease-in-out infinite;
    }
    @keyframes oa-orb-drift {
      from { transform: translate(0,0) scale(1); }
      to   { transform: translate(45px,28px) scale(1.1); }
    }
    @keyframes oa-orb-pulse {
      0%,100% { opacity: .5; transform: translate(-50%,-50%) scale(1); }
      50%     { opacity: 1;  transform: translate(-50%,-50%) scale(1.18); }
    }
    .oa-grid {
      position: absolute; inset: 0; pointer-events: none;
      background-image: radial-gradient(circle, rgba(27,67,255,0.14) 1px, transparent 1px);
      background-size: 40px 40px;
      mask-image: radial-gradient(ellipse 75% 75% at 50% 50%, black, transparent);
      animation: oa-grid-in 1s ease .1s both;
    }
    @keyframes oa-grid-in { from { opacity:0; } to { opacity:1; } }

    /* ── Login screen ── */
    .login-screen {
      position: fixed; inset: 0; z-index: 10000;
      background: radial-gradient(ellipse at 50% 35%, #0e153a 0%, #080c22 55%, #04060f 100%);
      display: flex; align-items: center; justify-content: center; overflow: hidden;
    }
    .login-card {
      position: relative; z-index: 2;
      background: rgba(255,255,255,0.035);
      border: 1px solid rgba(27,67,255,0.22);
      border-radius: 28px; padding: 40px 36px 36px;
      width: 100%; max-width: 380px;
      box-shadow: 0 24px 64px rgba(0,0,0,0.55), 0 0 0 1px rgba(27,67,255,0.10), inset 0 1px 0 rgba(255,255,255,0.05);
      backdrop-filter: blur(20px);
      animation: card-in 0.55s cubic-bezier(0.16,1,0.3,1) 0.1s both;
      display: flex; flex-direction: column; align-items: center;
    }
    @keyframes card-in {
      from { opacity:0; transform: translateY(24px) scale(0.97); }
      to   { opacity:1; transform: translateY(0) scale(1); }
    }
    .login-logo-ring {
      width: 64px; height: 64px; border-radius: 50%;
      background: rgba(27,67,255,0.10);
      border: 1px solid rgba(27,67,255,0.30);
      box-shadow: 0 0 36px rgba(27,67,255,0.20), inset 0 1px 0 rgba(255,255,255,0.06);
      display: flex; align-items: center; justify-content: center; margin-bottom: 18px;
      animation: mark-in 0.65s cubic-bezier(0.16,1,0.3,1) 0.2s both;
    }
    .login-logo-ring svg { width: 38px; height: 38px; }
    @keyframes mark-in {
      from { opacity:0; transform: scale(0.7) translateY(8px); }
      to   { opacity:1; transform: scale(1) translateY(0); }
    }
    .login-brand {
      font-size: 22px; font-weight: 800; color: #fff; letter-spacing: -0.5px; margin-bottom: 6px;
      background: linear-gradient(160deg, #ffffff 30%, #94A5FF 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .login-sub {
      font-size: 11.5px; color: rgba(154,163,200,0.7); letter-spacing: 0.4px;
      margin-bottom: 28px; text-align: center;
    }
    .login-form { width: 100%; display: flex; flex-direction: column; gap: 14px; }
    .field-wrap { display: flex; flex-direction: column; gap: 6px; }
    .field-label {
      font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;
      color: rgba(154,163,200,0.65);
    }
    .field-inner {
      display: flex; align-items: center; gap: 10px;
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
      border-radius: 14px; padding: 0 14px;
      transition: border-color 0.18s, box-shadow 0.18s;
    }
    .field-inner.field-focus {
      border-color: rgba(27,67,255,0.55); box-shadow: 0 0 0 3px rgba(27,67,255,0.12);
    }
    .field-inner svg { width: 15px; height: 15px; stroke: rgba(154,163,200,0.45); flex-shrink: 0; }
    .field-inner input {
      flex: 1; background: transparent; border: none; outline: none;
      font-size: 13.5px; color: #fff; padding: 12px 0;
    }
    .field-inner input::placeholder { color: rgba(154,163,200,0.30); }
    .pass-toggle {
      background: transparent; border: none; cursor: pointer; padding: 4px;
      color: rgba(154,163,200,0.40); display: flex; align-items: center; transition: color 0.15s;
    }
    .pass-toggle:hover { color: rgba(92,115,255,0.85); }
    .pass-toggle svg { width: 15px; height: 15px; }
    .login-error {
      display: flex; align-items: center; gap: 8px;
      background: rgba(220,38,38,0.09); border: 1px solid rgba(220,38,38,0.22);
      border-radius: 12px; padding: 9px 12px;
      font-size: 12px; color: #f87171; line-height: 1.4;
    }
    .login-error svg { width: 15px; height: 15px; flex-shrink: 0; }
    .login-btn {
      margin-top: 4px; width: 100%; padding: 13px;
      background: linear-gradient(145deg, #1B43FF, #1535D6);
      border: none; border-radius: 999px;
      font-size: 14px; font-weight: 700; color: #fff; cursor: pointer;
      box-shadow: 0 4px 18px rgba(27,67,255,0.38);
      transition: all 0.18s;
      display: flex; align-items: center; justify-content: center; min-height: 46px;
      letter-spacing: 0.1px;
    }
    .login-btn:hover:not(:disabled) {
      background: linear-gradient(145deg, #2F4CFF, #1B43FF);
      box-shadow: 0 6px 24px rgba(27,67,255,0.48); transform: translateY(-1px);
    }
    .login-btn:active:not(:disabled) { transform: translateY(0); }
    .login-btn:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; }
    .login-spinner {
      width: 18px; height: 18px; border-radius: 50%;
      border: 2px solid rgba(255,255,255,0.25); border-top-color: #fff;
      animation: oa-spin 0.7s linear infinite; display: inline-block;
    }
    @keyframes oa-spin { to { transform: rotate(360deg); } }

    /* ── OctaAI Splash ── */
    .oa-splash {
      position: fixed; inset: 0; z-index: 9999;
      background: radial-gradient(ellipse at 45% 40%, #0e153a 0%, #070d18 60%, #04060f 100%);
      display: flex; align-items: center; justify-content: center; overflow: hidden;
      transition: transform 0.75s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .oa-splash.pan-away { transform: translateY(-100%); }

    .oa-splash-center {
      position: relative; z-index: 2;
      display: flex; flex-direction: column; align-items: center; gap: 0;
    }
    .oa-splash-mark {
      width: 72px; height: 72px; margin-bottom: 28px;
      animation: mark-in 0.65s cubic-bezier(0.16,1,0.3,1) 0.15s both;
      filter: drop-shadow(0 0 24px rgba(27,67,255,0.55));
    }
    .oa-splash-mark svg { width: 72px; height: 72px; }

    .oa-splash-wordmark {
      font-size: 72px; font-weight: 800; letter-spacing: -2px; line-height: 1;
      background: linear-gradient(160deg, #ffffff 25%, #C2CCFF 65%, #5C73FF 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
      filter: drop-shadow(0 0 40px rgba(27,67,255,0.4));
      animation: wordmark-in 0.9s cubic-bezier(0.16,1,0.3,1) 0.35s both;
      margin-bottom: 14px;
    }
    @keyframes wordmark-in {
      from { opacity:0; transform: translateY(22px) scale(0.9); }
      to   { opacity:1; transform: none; }
    }

    .oa-splash-tagline {
      font-size: 12px; font-weight: 500; letter-spacing: 4px; text-transform: uppercase;
      color: rgba(154,163,200,0.65); margin-bottom: 36px;
      animation: text-up 0.6s ease 1.0s both;
      min-height: 18px;
    }
    @keyframes text-up {
      from { opacity:0; transform: translateY(8px); }
      to   { opacity:1; transform: none; }
    }

    .oa-progress-track {
      width: 200px; height: 3px; background: rgba(27,67,255,0.18);
      border-radius: 999px; overflow: hidden;
      animation: text-up 0.5s ease 1.2s both;
    }
    .oa-progress-bar {
      height: 100%; background: linear-gradient(90deg, #1B43FF, #94A5FF);
      border-radius: 999px; transition: width 0.4s ease;
      box-shadow: 0 0 8px rgba(27,67,255,0.6);
    }

    /* Faint giant watermark */
    .oa-bg-mark {
      position: absolute; right: -8%; top: 5%; width: 520px; height: 520px;
      opacity: 0.04; pointer-events: none;
      animation: om-float 9s ease-in-out infinite;
    }

    /* ── User pill ── */
    .user-pill {
      display: flex; align-items: center; gap: 8px;
      padding: 4px 10px 4px 4px;
      background: rgba(27,67,255,0.10); border: 1px solid rgba(27,67,255,0.22);
      border-radius: 20px; cursor: default;
    }
    .user-avatar {
      width: 24px; height: 24px; border-radius: 50%; flex-shrink: 0;
      background: linear-gradient(145deg, #1535D6, #1B43FF);
      display: flex; align-items: center; justify-content: center;
      font-size: 9px; font-weight: 700; color: #fff;
    }
    .user-name {
      font-size: 11.5px; font-weight: 500; color: rgba(154,163,200,0.9);
      max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .btn-logout {
      display: flex; align-items: center; justify-content: center;
      width: 20px; height: 20px; padding: 0; margin-left: 2px;
      background: none; border: none; cursor: pointer;
      color: rgba(154,163,200,0.45); border-radius: 8px;
      transition: color 0.15s, background 0.15s;
    }
    .btn-logout:hover { color: #f87171; background: rgba(248,113,113,0.10); }
    .btn-logout svg { width: 13px; height: 13px; }

    /* ── Theme toggle ── */
    .btn-theme {
      background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px; padding: 5px 9px; cursor: pointer;
      color: rgba(154,163,200,0.75);
      display: flex; align-items: center; justify-content: center; transition: all 0.18s;
    }
    .btn-theme:hover { background: rgba(255,255,255,0.09); border-color: rgba(255,255,255,0.13); color: #fff; }
    .btn-theme svg { width: 15px; height: 15px; }

    /* ── Tab placeholder ── */
    .tab-placeholder {
      flex: 1; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 12px;
      background: var(--bg); color: var(--text-3);
      animation: fadeUp 0.4s var(--ease) both;
    }
    .tab-placeholder-icon {
      width: 56px; height: 56px; border-radius: 16px;
      background: var(--blue-50); border: 1px solid var(--blue-100);
      display: flex; align-items: center; justify-content: center;
      color: var(--accent);
    }
    .tab-placeholder-text { font-size: 16px; font-weight: 600; color: var(--text); }
    .tab-placeholder-sub  { font-size: 13px; color: var(--text-3); max-width: 340px; text-align: center; line-height: 1.6; }
    @keyframes fadeUp {
      from { opacity:0; transform: translateY(12px); }
      to   { opacity:1; transform: none; }
    }
  `]
})
export class AppComponent implements OnInit {
  isDark         = false;
  modelLabel     = 'Loading…';
  showSplash     = false;
  splashOut      = false;
  showLogin      = true;
  loggedInUser   = '';
  activeTab      = 'chat';

  splashStageLabel = 'Initializing…';
  splashProgress   = 0;

  loginUsername = '';
  loginPassword = '';
  loginError    = '';
  loginLoading  = false;
  showPass      = false;
  userFocus     = false;
  passFocus     = false;

  get userInitials(): string {
    return this.loggedInUser
      .split(/[\s._-]+/).filter(Boolean).slice(0, 2)
      .map(w => w[0].toUpperCase()).join('') || '?';
  }

  get tabPlaceholderLabel(): string {
    const labels: Record<string, string> = {
      dashboard: 'Dashboard',
      saved: 'Saved Queries',
      schema: 'Schema Explorer',
    };
    return labels[this.activeTab] ?? this.activeTab;
  }

  private readonly API = '/api';

  constructor(
    @Inject(PLATFORM_ID) private platformId: Object,
    private http: HttpClient,
    private cdr: ChangeDetectorRef,
    private authService: AuthService,
    private chatService: ChatService,
  ) {}

  ngOnInit() {
    if (isPlatformBrowser(this.platformId)) {
      const stored = localStorage.getItem('theme');
      if (stored === 'dark' || (!stored && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        this.isDark = true;
        document.body.setAttribute('data-theme', 'dark');
      }
      this.loggedInUser = sessionStorage.getItem(USERNAME_KEY) || '';
      const existingToken = sessionStorage.getItem(TOKEN_KEY);
      if (existingToken) {
        this.showLogin = false;
        this.authService.fetchPermissions().subscribe({
          next: res => {
            this.authService.setAuth(existingToken, res.sub, res.permissions, res.first_name || '');
            this._startSplash();
            this.cdr.detectChanges();
          },
          error: err => {
            if (err?.status === 401 || err?.status === 403) {
              this.authService.clearAuth();
              sessionStorage.removeItem(TOKEN_KEY);
              sessionStorage.removeItem(USERNAME_KEY);
              this.loggedInUser = '';
              this.showLogin    = true;
              if (err?.status === 403) {
                this.loginError = err?.error?.detail || 'You have no permission to access this feature.';
              }
            } else {
              this._startSplash();
            }
            this.cdr.detectChanges();
          },
        });

        document.addEventListener('visibilitychange', () => {
          if (document.visibilityState === 'visible' && this.authService.token) {
            this.authService.fetchPermissions().subscribe({
              next: res => this.authService.setAuth(this.authService.token!, res.sub, res.permissions, res.first_name || ''),
              error: () => {},
            });
          }
        });
      }
    }
  }

  doLogin() {
    if (!this.loginUsername || !this.loginPassword || this.loginLoading) return;
    this.loginLoading = true;
    this.loginError   = '';

    this.http.post<{ access_token: string; sub: string }>('/api/login', {
      username: this.loginUsername,
      password: this.loginPassword,
    }).subscribe({
      next: res => {
        if (isPlatformBrowser(this.platformId)) {
          sessionStorage.setItem(TOKEN_KEY, res.access_token);
          sessionStorage.setItem(USERNAME_KEY, this.loginUsername);
        }
        this.loggedInUser = this.loginUsername;
        // Reset chat state on every successful login — logout already does this
        // (see _clearAndRedirectToLogin below). Without it, a stale conversation/
        // session_id from a previous login in this tab would carry over and get
        // sent under the new user's token.
        this.chatService.reset();

        this.authService.fetchPermissions().subscribe({
          next: permsRes => {
            this.authService.setAuth(res.access_token, res.sub, permsRes.permissions, permsRes.first_name || '');
            this.loginLoading = false;
            this.showLogin    = false;
            this._startSplash();
            this.cdr.detectChanges();
          },
          error: (err) => {
            this.loginLoading = false;
            if (err?.status === 403) {
              sessionStorage.removeItem(TOKEN_KEY);
              sessionStorage.removeItem(USERNAME_KEY);
              this.loggedInUser = '';
              this.loginError   = err?.error?.detail || 'You have no permission to access this feature.';
            } else {
              this.authService.setAuth(res.access_token, res.sub, []);
              this.showLogin = false;
              this._startSplash();
            }
            this.cdr.detectChanges();
          },
        });
      },
      error: err => {
        this.loginLoading = false;
        this.loginError   = err?.error?.detail || err?.error?.message || err?.error?.error_description || 'Invalid credentials. Please try again.';
        this.cdr.detectChanges();
      },
    });
  }

  private readonly SPLASH_STAGES = [
    'Connecting to backend…',
    'Loading schema context…',
    'Warming up models…',
    'Almost ready…',
  ];

  private _startSplash() {
    this.showSplash      = true;
    this.splashProgress  = 0;
    this.splashStageLabel = this.SPLASH_STAGES[0];

    this.http.get<{ status: string; model: string }>(this.API + '/health').subscribe({
      next: res  => { this.modelLabel = res.model ?? 'unknown'; this.cdr.detectChanges(); },
      error: ()  => { this.modelLabel = 'offline'; this.cdr.detectChanges(); },
    });

    // Animate progress bar + stage labels over 2.6s
    const steps = [
      { at: 200,  pct: 22,  stage: 1 },
      { at: 800,  pct: 55,  stage: 2 },
      { at: 1500, pct: 82,  stage: 3 },
      { at: 2200, pct: 100, stage: 3 },
    ];
    steps.forEach(({ at, pct, stage }) => {
      setTimeout(() => {
        this.splashProgress   = pct;
        this.splashStageLabel = this.SPLASH_STAGES[stage];
        this.cdr.detectChanges();
      }, at);
    });

    setTimeout(() => { this.splashOut  = true;  this.cdr.detectChanges(); }, 2600);
    setTimeout(() => { this.showSplash = false; this.cdr.detectChanges(); }, 3400);
  }

  logout() {
    this.authService.logout().subscribe({
      complete: () => this._clearAndRedirectToLogin(),
      error:    () => this._clearAndRedirectToLogin(),
    });
  }

  private _clearAndRedirectToLogin() {
    this.authService.clearAuth();
    this.chatService.reset();
    if (isPlatformBrowser(this.platformId)) {
      sessionStorage.removeItem(TOKEN_KEY);
      sessionStorage.removeItem(USERNAME_KEY);
    }
    this.loggedInUser = '';
    this.showSplash   = false;
    this.splashOut    = false;
    this.showLogin    = true;
    this.activeTab    = 'chat';
    this.cdr.detectChanges();
  }

  toggleTheme() {
    this.isDark = !this.isDark;
    if (isPlatformBrowser(this.platformId)) {
      if (this.isDark) {
        document.body.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
      } else {
        document.body.removeAttribute('data-theme');
        localStorage.setItem('theme', 'light');
      }
    }
  }
}
