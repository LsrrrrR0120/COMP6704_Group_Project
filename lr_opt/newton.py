"""Newton-Raphson.

Owner: ZHANG Qigang

b_{k+1} = b_k - [Hess J(b_k)]^{-1} grad J(b_k) with unit steps by default. With
damped=True the Newton direction is combined with Armijo backtracking, which makes the
method globally convergent on any strictly convex J (Boyd & Vandenberghe, Sec. 9.5).

Without regularization (lam = 0) and with (quasi-)separable classes, J has no finite
minimizer: the iterates grow without bound and the Hessian becomes numerically singular.
The solve then falls back to least squares, and the run stops if the iterate stops being finite.
"""
import numpy as np

from .linesearch import LineSearchError, backtracking
from .result import Trace


def newton_raphson(obj, x0, tol=1e-6, max_iter=300, damped=False):
    tr = Trace(obj)
    x = np.array(x0, dtype=float)
    f, g = obj.fun_grad(x)
    tr.log(f, g)
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            return tr.result(x, f, g, k, tol, "converged")
        H = obj.hess(x)
        try:
            d = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            d = -np.linalg.lstsq(H, g, rcond=None)[0]
        a = 1.0
        if damped:
            try:
                a, _ = backtracking(obj.fun, x, f, g, d)
            except LineSearchError as e:
                return tr.result(x, f, g, k, tol, str(e), success=False)
        x_new = x + a * d
        if not np.all(np.isfinite(x_new)):
            return tr.result(x, f, g, k, tol, "non-finite iterate", success=False)
        x = x_new
        f, g = obj.fun_grad(x)
        tr.log(f, g)
    return tr.result(x, f, g, max_iter, tol, "maximum number of iterations reached")
