import { HttpInterceptorFn, HttpRequest, HttpHandlerFn, HttpErrorResponse } from '@angular/common/http';
import { catchError, throwError } from 'rxjs';

const TOKEN_KEY = 'octaai_access_token';

export const authInterceptor: HttpInterceptorFn = (
  req: HttpRequest<unknown>,
  next: HttpHandlerFn
) => {
  if (req.url.includes('/api/login')) {
    return next(req);
  }

  const token = sessionStorage.getItem(TOKEN_KEY);
  const authReq = token
    ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } })
    : req;

  return next(authReq).pipe(
    catchError((err: HttpErrorResponse) => {
      if (err.status === 401) {
        sessionStorage.removeItem(TOKEN_KEY);
        sessionStorage.removeItem('octaai_username');
        window.location.reload();
      }
      // 403 — propagate; route guards / components handle the UI response
      return throwError(() => err);
    })
  );
};
