import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from '../services/auth.service';

export const permissionGuard: CanActivateFn = (route) => {
  const auth   = inject(AuthService);
  const router = inject(Router);
  const required = route.data?.['permission'] as string | undefined;
  if (!required || auth.hasPermission(required)) return true;
  router.navigate(['/']);
  return false;
};
