"""L-BFGS (limited-memory BFGS).

Owner: PAN Pan

The BFGS update of an inverse-Hessian approximation,
    H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T,   rho = 1 / (y^T s),
is never formed as a d x d matrix. L-BFGS stores the last m pairs (s, y) and computes H_k g
with the two-loop recursion, starting from H_k^0 = (s^T y / y^T y) I. A strong Wolfe line
search (c1 = 1e-4, c2 = 0.9) guarantees s^T y > 0, so the implied H_k stays positive definite.
"""
from collections import deque

import numpy as np

from .linesearch import LineSearchError, strong_wolfe
from .result import Trace


def lbfgs(obj, x0, m=10, tol=1e-6, max_iter=8000):
    tr = Trace(obj)
    x = np.array(x0, dtype=float)
    f, g = obj.fun_grad(x)
    tr.log(f, g)
    pairs = deque(maxlen=m)          # (s, y, rho), oldest first
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            return tr.result(x, f, g, k, tol, "converged")
        d = -_two_loop(g, pairs)
        alpha0 = 1.0 if pairs else 1.0 / max(1.0, np.linalg.norm(g))
        try:
            a, f_new, g_new = strong_wolfe(obj.fun_grad, x, f, g, d, alpha0=alpha0)
        except LineSearchError as e:
            return tr.result(x, f, g, k, tol, str(e), success=False)
        s, y = a * d, g_new - g
        sy = s @ y
        if sy > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            pairs.append((s, y, 1.0 / sy))
        x, f, g = x + s, f_new, g_new
        tr.log(f, g)
    return tr.result(x, f, g, max_iter, tol, "maximum number of iterations reached")


def _two_loop(g, pairs):
    """H_k g for the L-BFGS inverse-Hessian approximation (Nocedal & Wright Algorithm 7.4)."""
    q = g.copy()
    alphas = []
    for s, y, rho in reversed(pairs):
        a = rho * (s @ q)
        q -= a * y
        alphas.append(a)
    if pairs:
        s, y, _ = pairs[-1]
        q *= (s @ y) / (y @ y)
    for (s, y, rho), a in zip(pairs, reversed(alphas)):
        q += (a - rho * (y @ q)) * s
    return q
