import { Component, Inject, PLATFORM_ID, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { ChatWindowComponent } from './components/chat-window/chat-window.component';
import { DebugPanelComponent } from './components/debug-panel/debug-panel.component';
import { HistorySidebarComponent } from './components/history-sidebar/history-sidebar.component';

const OAUTH_URL       = 'https://10.77.20.51/oauth/token?_format=json';
const OAUTH_CLIENT_ID = '78XrNvZXXG4sZEggH1G3RKzcCtKxXWyr1VGxf_gfWws';
const OAUTH_SECRET    = '78XrNvZXXG4sZEggH1G3RKzcCtKxXWyr1VGxf_gfWws';
const TOKEN_KEY       = 'inspectai_access_token';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, ChatWindowComponent, DebugPanelComponent, HistorySidebarComponent],
  template: `
    <!-- ── Login page ── -->
    <div class="login-screen" *ngIf="showLogin">
      <div class="splash-orb splash-orb-1"></div>
      <div class="splash-orb splash-orb-2"></div>
      <div class="splash-orb splash-orb-3"></div>
      <div class="splash-grid"></div>
      <div class="login-card">
        <div class="login-logo-ring">
          <svg viewBox="0 0 24 24" fill="none" stroke="#3fbdad" stroke-width="1.5">
            <path stroke-linecap="round" stroke-linejoin="round"
              d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18"/>
          </svg>
        </div>
        <div class="login-brand">InspectAI</div>
        <div class="login-sub">نظام التفتيش الذكي &nbsp;·&nbsp; Sign in to continue</div>

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

    <!-- Arabic welcome splash -->
    <div class="splash" *ngIf="showSplash" [class.pan-away]="splashOut">
      <div class="splash-orb splash-orb-1"></div>
      <div class="splash-orb splash-orb-2"></div>
      <div class="splash-orb splash-orb-3"></div>
      <div class="splash-grid"></div>
      <div class="splash-particles">
        <span class="sp sp-1"></span><span class="sp sp-2"></span>
        <span class="sp sp-3"></span><span class="sp sp-4"></span>
        <span class="sp sp-5"></span><span class="sp sp-6"></span>
        <span class="sp sp-7"></span><span class="sp sp-8"></span>
      </div>
      <div class="splash-content">
        <div class="splash-logo-ring">
          <svg viewBox="0 0 24 24" fill="none" stroke="#3fbdad" stroke-width="1.5">
            <path stroke-linecap="round" stroke-linejoin="round"
              d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18"/>
          </svg>
        </div>
        <div class="splash-arabic">مرحباً</div>
        <div class="splash-divider"><span></span></div>
        <div class="splash-welcome">Welcome to InspectAI</div>
        <div class="splash-sub">نظام التفتيش الذكي &nbsp;·&nbsp; Intelligent Inspection Analytics</div>
      </div>
      <div class="splash-fade-bottom"></div>
    </div>

    <div class="app-shell">
      <header class="app-header">
        <!-- Brand block (same width as sidebar) -->
        <div class="brand">
          <div class="brand-logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round"
                d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18"/>
            </svg>
          </div>
          <div>
            <span class="brand-name">InspectAI</span>
            <span class="brand-sub">NL · SQL Intelligence</span>
          </div>
        </div>

        <!-- Khidmah-style tab navigation -->
        <nav class="header-nav">
          <button class="nav-tab active">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
            </svg>
            AI Chat
          </button>
          <button class="nav-tab">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
              <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
            </svg>
            Dashboard
          </button>
          <button class="nav-tab">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            Inspection List
          </button>
          <button class="nav-tab">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>
            Reports
          </button>
          <button class="nav-tab">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.07 4.93l-1.41 1.41M4.93 19.07l1.41-1.41M12 2v2M12 20v2M4.93 4.93l1.41 1.41M19.07 19.07l-1.41-1.41M2 12h2M20 12h2"/>
            </svg>
            Configurations
          </button>
        </nav>

        <!-- Right controls -->
        <div class="header-badges">
          <span class="badge badge-blue">
            <span class="bdot" style="background:#3fbdad"></span>
            {{ modelLabel }}
          </span>
          <span class="badge badge-green">
            <span class="bdot" style="background:#34d399"></span>
            Live
          </span>
          <button class="btn-theme" (click)="toggleTheme()" [title]="isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'">
            <svg *ngIf="!isDark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"></path></svg>
            <svg *ngIf="isDark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
          </button>
        </div>
      </header>
      <div class="app-body">
        <app-history-sidebar></app-history-sidebar>
        <app-chat-window></app-chat-window>
        <app-debug-panel></app-debug-panel>
      </div>
    </div>
  `,
  styles: [`
    /* ── Splash screen ── */
    .splash {
      position: fixed; inset: 0; z-index: 9999;
      background: radial-gradient(ellipse at 50% 38%, #162034 0%, #0e1826 55%, #070d18 100%);
      display: flex; align-items: center; justify-content: center; overflow: hidden;
      transition: transform 0.75s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .splash.pan-away { transform: translateY(-100%); }

    /* Ambient orbs */
    .splash-orb {
      position: absolute; border-radius: 50%; filter: blur(90px); pointer-events: none;
    }
    .splash-orb-1 {
      width: 520px; height: 520px; background: rgba(63,189,173,0.13);
      top: -160px; left: -140px;
      animation: orb-drift 6s ease-in-out infinite alternate;
    }
    .splash-orb-2 {
      width: 420px; height: 420px; background: rgba(224,122,110,0.09);
      bottom: -120px; right: -100px;
      animation: orb-drift 7s ease-in-out 1s infinite alternate-reverse;
    }
    .splash-orb-3 {
      width: 300px; height: 300px; background: rgba(63,189,173,0.07);
      top: 55%; left: 48%; transform: translate(-50%,-50%);
      animation: orb-pulse 4s ease-in-out infinite;
    }
    @keyframes orb-drift {
      from { transform: translate(0,0) scale(1); }
      to   { transform: translate(40px,25px) scale(1.12); }
    }
    @keyframes orb-pulse {
      0%,100% { opacity: 0.5; transform: translate(-50%,-50%) scale(1); }
      50%      { opacity: 1;   transform: translate(-50%,-50%) scale(1.2); }
    }

    /* Subtle dot grid */
    .splash-grid {
      position: absolute; inset: 0; pointer-events: none;
      background-image: radial-gradient(circle, rgba(63,189,173,0.12) 1px, transparent 1px);
      background-size: 38px 38px;
      mask-image: radial-gradient(ellipse 70% 70% at 50% 50%, black 0%, transparent 100%);
      animation: grid-fade 1.2s ease 0.1s both;
    }
    @keyframes grid-fade { from { opacity:0; } to { opacity:1; } }

    /* Floating light particles */
    .splash-particles { position: absolute; inset: 0; pointer-events: none; }
    .sp {
      position: absolute; border-radius: 50%;
      background: rgba(63,189,173,0.55);
      box-shadow: 0 0 8px 2px rgba(63,189,173,0.4);
      animation: sp-float linear infinite;
    }
    .sp-1 { width:4px;height:4px; left:12%; top:20%; animation-duration:8s;  animation-delay:0s; }
    .sp-2 { width:3px;height:3px; left:80%; top:15%; animation-duration:11s; animation-delay:1.5s; }
    .sp-3 { width:5px;height:5px; left:65%; top:70%; animation-duration:9s;  animation-delay:0.8s; background:rgba(224,122,110,0.6); box-shadow:0 0 8px 2px rgba(224,122,110,0.4); }
    .sp-4 { width:3px;height:3px; left:30%; top:75%; animation-duration:12s; animation-delay:2s; }
    .sp-5 { width:4px;height:4px; left:88%; top:48%; animation-duration:10s; animation-delay:0.3s; }
    .sp-6 { width:2px;height:2px; left:20%; top:55%; animation-duration:14s; animation-delay:3s; }
    .sp-7 { width:3px;height:3px; left:50%; top:10%; animation-duration:9s;  animation-delay:1s; background:rgba(224,122,110,0.5); box-shadow:0 0 6px 2px rgba(224,122,110,0.3); }
    .sp-8 { width:4px;height:4px; left:73%; top:85%; animation-duration:13s; animation-delay:2.5s; }
    @keyframes sp-float {
      0%   { transform: translateY(0) scale(1);   opacity: 0; }
      10%  { opacity: 1; }
      90%  { opacity: 0.7; }
      100% { transform: translateY(-120px) scale(0.6); opacity: 0; }
    }

    /* Central content */
    .splash-content {
      position: relative; z-index: 2;
      display: flex; flex-direction: column; align-items: center; gap: 0;
    }

    .splash-logo-ring {
      width: 72px; height: 72px; border-radius: 20px;
      background: rgba(63,189,173,0.08);
      border: 1px solid rgba(63,189,173,0.25);
      box-shadow: 0 0 40px rgba(63,189,173,0.2), inset 0 1px 0 rgba(255,255,255,0.06);
      display: flex; align-items: center; justify-content: center; margin-bottom: 28px;
      animation: logo-in 0.6s cubic-bezier(0.16,1,0.3,1) 0.15s both;
    }
    .splash-logo-ring svg { width: 34px; height: 34px; }
    @keyframes logo-in {
      from { opacity:0; transform: scale(0.7) translateY(12px); }
      to   { opacity:1; transform: scale(1) translateY(0); }
    }

    .splash-arabic {
      font-family: 'Cairo', 'Segoe UI', 'Arial', sans-serif;
      font-size: 96px; font-weight: 900; line-height: 1;
      direction: rtl;
      background: linear-gradient(160deg, #ffffff 30%, #a8e8e2 70%, #3fbdad 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
      text-shadow: none;
      filter: drop-shadow(0 0 50px rgba(63,189,173,0.45)) drop-shadow(0 0 100px rgba(63,189,173,0.2));
      animation: arabic-in 0.9s cubic-bezier(0.16,1,0.3,1) 0.35s both;
      margin-bottom: 20px;
    }
    @keyframes arabic-in {
      from { opacity:0; transform: translateY(28px) scale(0.88); filter: drop-shadow(0 0 0 rgba(63,189,173,0)); }
      to   { opacity:1; transform: translateY(0) scale(1); }
    }

    .splash-divider {
      display: flex; align-items: center; gap: 10px; margin-bottom: 18px;
      animation: divider-in 0.6s ease 0.9s both;
    }
    .splash-divider span {
      display: block; height: 1px; width: 120px;
      background: linear-gradient(90deg, transparent, rgba(63,189,173,0.8), rgba(63,189,173,0.3), transparent);
      box-shadow: 0 0 10px rgba(63,189,173,0.5);
    }
    @keyframes divider-in {
      from { opacity:0; transform: scaleX(0); }
      to   { opacity:1; transform: scaleX(1); }
    }

    .splash-welcome {
      font-family: 'Inter', sans-serif;
      font-size: 15px; font-weight: 400; letter-spacing: 5px; text-transform: uppercase;
      color: rgba(255,255,255,0.7); margin-bottom: 10px;
      animation: text-up 0.6s ease 1.1s both;
    }
    .splash-sub {
      font-family: 'Cairo', 'Inter', sans-serif;
      font-size: 12.5px; color: rgba(63,189,173,0.65); letter-spacing: 1px;
      animation: text-up 0.6s ease 1.3s both;
    }
    @keyframes text-up {
      from { opacity:0; transform: translateY(10px); }
      to   { opacity:1; transform: translateY(0); }
    }

    /* Bottom vignette so chat isn't visible underneath */
    .splash-fade-bottom {
      position: absolute; bottom: 0; left: 0; right: 0; height: 120px; pointer-events: none;
      background: linear-gradient(to bottom, transparent, #070d18);
    }

    /* ── Login screen ── */
    .login-screen {
      position: fixed; inset: 0; z-index: 10000;
      background: radial-gradient(ellipse at 50% 38%, #162034 0%, #0e1826 55%, #070d18 100%);
      display: flex; align-items: center; justify-content: center; overflow: hidden;
    }

    .login-card {
      position: relative; z-index: 2;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(63,189,173,0.18);
      border-radius: 20px;
      padding: 40px 36px 36px;
      width: 100%; max-width: 380px;
      box-shadow: 0 24px 64px rgba(0,0,0,0.5), 0 0 0 1px rgba(63,189,173,0.08), inset 0 1px 0 rgba(255,255,255,0.06);
      backdrop-filter: blur(20px);
      animation: card-in 0.55s cubic-bezier(0.16,1,0.3,1) 0.1s both;
      display: flex; flex-direction: column; align-items: center;
    }
    @keyframes card-in {
      from { opacity:0; transform: translateY(24px) scale(0.97); }
      to   { opacity:1; transform: translateY(0) scale(1); }
    }

    .login-logo-ring {
      width: 60px; height: 60px; border-radius: 16px;
      background: rgba(63,189,173,0.08);
      border: 1px solid rgba(63,189,173,0.25);
      box-shadow: 0 0 32px rgba(63,189,173,0.18), inset 0 1px 0 rgba(255,255,255,0.06);
      display: flex; align-items: center; justify-content: center; margin-bottom: 18px;
    }
    .login-logo-ring svg { width: 28px; height: 28px; }

    .login-brand {
      font-family: 'Inter', sans-serif;
      font-size: 20px; font-weight: 700; color: #fff; letter-spacing: -0.4px; margin-bottom: 6px;
    }
    .login-sub {
      font-family: 'Cairo', 'Inter', sans-serif;
      font-size: 11.5px; color: rgba(63,189,173,0.6); letter-spacing: 0.5px;
      margin-bottom: 28px; text-align: center;
    }

    .login-form { width: 100%; display: flex; flex-direction: column; gap: 14px; }

    .field-wrap { display: flex; flex-direction: column; gap: 6px; }
    .field-label {
      font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;
      color: rgba(160,174,192,0.7);
    }
    .field-inner {
      display: flex; align-items: center; gap: 10px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 10px; padding: 0 14px;
      transition: border-color 0.18s, box-shadow 0.18s;
    }
    .field-inner.field-focus {
      border-color: rgba(63,189,173,0.5);
      box-shadow: 0 0 0 3px rgba(63,189,173,0.1);
    }
    .field-inner svg { width: 15px; height: 15px; stroke: rgba(160,174,192,0.5); flex-shrink: 0; }
    .field-inner input {
      flex: 1; background: transparent; border: none; outline: none;
      font-family: 'Inter', sans-serif; font-size: 13.5px; color: #fff;
      padding: 11px 0;
    }
    .field-inner input::placeholder { color: rgba(160,174,192,0.35); }

    .pass-toggle {
      background: transparent; border: none; cursor: pointer; padding: 4px;
      color: rgba(160,174,192,0.45); display: flex; align-items: center;
      transition: color 0.15s;
    }
    .pass-toggle:hover { color: rgba(63,189,173,0.8); }
    .pass-toggle svg { width: 15px; height: 15px; }

    .login-error {
      display: flex; align-items: center; gap: 8px;
      background: rgba(220,38,38,0.1); border: 1px solid rgba(220,38,38,0.25);
      border-radius: 8px; padding: 9px 12px;
      font-size: 12px; color: #f87171; line-height: 1.4;
    }
    .login-error svg { width: 15px; height: 15px; flex-shrink: 0; }

    .login-btn {
      margin-top: 4px;
      width: 100%; padding: 12px;
      background: linear-gradient(145deg, #3fbdad, #32a89a);
      border: none; border-radius: 10px;
      font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 600;
      color: #0f1923; cursor: pointer;
      box-shadow: 0 4px 16px rgba(63,189,173,0.3);
      transition: all 0.18s;
      display: flex; align-items: center; justify-content: center; min-height: 44px;
    }
    .login-btn:hover:not(:disabled) {
      background: linear-gradient(145deg, #4dcebf, #3fbdad);
      box-shadow: 0 6px 20px rgba(63,189,173,0.4); transform: translateY(-1px);
    }
    .login-btn:active:not(:disabled) { transform: translateY(0); }
    .login-btn:disabled { opacity: 0.5; cursor: not-allowed; box-shadow: none; }

    .login-spinner {
      width: 18px; height: 18px; border-radius: 50%;
      border: 2px solid rgba(15,25,35,0.3);
      border-top-color: #0f1923;
      animation: spin 0.7s linear infinite; display: inline-block;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Theme button ── */
    .btn-theme {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 7px; padding: 5px 9px; cursor: pointer;
      color: rgba(160,174,192,0.8);
      display: flex; align-items: center; justify-content: center;
      transition: all 0.18s ease;
    }
    .btn-theme:hover {
      background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.14);
      color: #fff; transform: scale(1.04);
    }
    .btn-theme svg { width: 15px; height: 15px; }
  `]
})
export class AppComponent implements OnInit {
  isDark      = false;
  modelLabel  = 'Loading…';
  showSplash  = false;
  splashOut   = false;
  showLogin   = true;

  loginUsername = '';
  loginPassword = '';
  loginError    = '';
  loginLoading  = false;
  showPass      = false;
  userFocus     = false;
  passFocus     = false;

  private readonly API = '/api';

  constructor(
    @Inject(PLATFORM_ID) private platformId: Object,
    private http: HttpClient,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit() {
    if (isPlatformBrowser(this.platformId)) {
      const stored = localStorage.getItem('theme');
      if (stored === 'dark' || (!stored && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        this.isDark = true;
        document.body.setAttribute('data-theme', 'dark');
      }
      if (localStorage.getItem(TOKEN_KEY)) {
        this.showLogin = false;
        this._startSplash();
      }
    }
  }

  doLogin() {
    if (!this.loginUsername || !this.loginPassword || this.loginLoading) return;
    this.loginLoading = true;
    this.loginError   = '';

    const body = new URLSearchParams({
      grant_type:    'password',
      client_id:     OAUTH_CLIENT_ID,
      client_secret: OAUTH_SECRET,
      username:      this.loginUsername,
      password:      this.loginPassword,
    });
    const headers = new HttpHeaders({ 'Content-Type': 'application/x-www-form-urlencoded' });

    this.http.post<{ access_token: string }>(OAUTH_URL, body.toString(), { headers }).subscribe({
      next: res => {
        if (isPlatformBrowser(this.platformId)) {
          localStorage.setItem(TOKEN_KEY, res.access_token);
        }
        this.loginLoading = false;
        this.showLogin    = false;
        this._startSplash();
        this.cdr.detectChanges();
      },
      error: err => {
        this.loginLoading = false;
        this.loginError   = err?.error?.message || err?.error?.error_description || 'Invalid credentials. Please try again.';
        this.cdr.detectChanges();
      },
    });
  }

  private _startSplash() {
    this.showSplash = true;
    this.http.get<{ status: string; model: string }>(this.API + '/health').subscribe({
      next: res  => { this.modelLabel = res.model ?? 'unknown'; this.cdr.detectChanges(); },
      error: ()  => { this.modelLabel = 'offline'; this.cdr.detectChanges(); },
    });
    setTimeout(() => { this.splashOut  = true;  this.cdr.detectChanges(); }, 2600);
    setTimeout(() => { this.showSplash = false; this.cdr.detectChanges(); }, 3400);
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
