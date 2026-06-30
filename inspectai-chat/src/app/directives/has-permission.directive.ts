import {
  Directive, Input, OnInit, OnDestroy,
  TemplateRef, ViewContainerRef,
} from '@angular/core';
import { Subscription } from 'rxjs';
import { AuthService } from '../services/auth.service';

@Directive({ selector: '[appHasPermission]', standalone: true })
export class HasPermissionDirective implements OnInit, OnDestroy {
  @Input('appHasPermission') permission = '';

  private _sub?: Subscription;
  private _hasView = false;

  constructor(
    private readonly tpl: TemplateRef<unknown>,
    private readonly vcr: ViewContainerRef,
    private readonly auth: AuthService,
  ) {}

  ngOnInit(): void {
    this._sub = this.auth.permissions$.subscribe(() => this._update());
  }

  ngOnDestroy(): void {
    this._sub?.unsubscribe();
  }

  private _update(): void {
    const allowed = !this.permission || this.auth.hasPermission(this.permission);
    if (allowed && !this._hasView) {
      this.vcr.createEmbeddedView(this.tpl);
      this._hasView = true;
    } else if (!allowed && this._hasView) {
      this.vcr.clear();
      this._hasView = false;
    }
  }
}
