import { Injectable, PLATFORM_ID, Inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';
import { isPlatformBrowser } from '@angular/common';

const TOKEN_KEY     = 'octaai_access_token';
const SUB_KEY       = 'octaai_sub';
const FIRSTNAME_KEY = 'octaai_firstname';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly _permissions$ = new BehaviorSubject<string[]>([]);
  private readonly _sub$         = new BehaviorSubject<string | null>(null);

  readonly permissions$ = this._permissions$.asObservable();
  readonly sub$         = this._sub$.asObservable();

  constructor(
    private readonly http: HttpClient,
    @Inject(PLATFORM_ID) private readonly platformId: object,
  ) {}

  get token(): string | null {
    if (!isPlatformBrowser(this.platformId)) return null;
    return localStorage.getItem(TOKEN_KEY);
  }

  get sub(): string | null { return this._sub$.value; }

  get firstName(): string {
    if (!isPlatformBrowser(this.platformId)) return '';
    return localStorage.getItem(FIRSTNAME_KEY) || '';
  }

  setAuth(token: string, sub: string, permissions: string[], firstName = ''): void {
    if (isPlatformBrowser(this.platformId)) {
      localStorage.setItem(TOKEN_KEY, token);
      localStorage.setItem(SUB_KEY, sub);
      if (firstName) localStorage.setItem(FIRSTNAME_KEY, firstName);
    }
    this._sub$.next(sub);
    // Deduplicate on receipt — the Java API sometimes repeats entries
    this._permissions$.next([...new Set(permissions)]);
  }

  clearAuth(): void {
    if (isPlatformBrowser(this.platformId)) {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(SUB_KEY);
      localStorage.removeItem(FIRSTNAME_KEY);
      localStorage.removeItem('octaai_username');
    }
    this._sub$.next(null);
    this._permissions$.next([]);
  }

  hasPermission(permission: string): boolean {
    return this._permissions$.value.includes(permission);
  }

  hasAnyPermission(permissions: string[]): boolean {
    const current = this._permissions$.value;
    return permissions.some(p => current.includes(p));
  }

  fetchPermissions(): Observable<{ sub: string; permissions: string[]; first_name?: string }> {
    return this.http.get<{ sub: string; permissions: string[]; first_name?: string }>('/api/me/permissions');
  }

  logout(): Observable<{ status: string }> {
    return this.http.post<{ status: string }>('/api/logout', {});
  }
}
