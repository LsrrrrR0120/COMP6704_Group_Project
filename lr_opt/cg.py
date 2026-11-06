"""Nonlinear conjugate gradient.

Owner: LIN Shenrui

d_0 = -g_0,  b_{k+1} = b_k + a_k d_k,  d_{k+1} = -g_{k+1} + beta_k d_k.

beta="FR" is the Fletcher-Reeves coefficient (the default), beta="PR+" the non-negative
Polak-Ribiere coefficient (scipy's CG is a Polak-Ribiere method). The step comes from a strong
Wolfe search with c2 = 0.1 (c2 < 1/2 guarantees that every FR direction is a descent
direction, Nocedal & Wright Lemma 5.6). The direction is reset to -g whenever it is not a
descent direction, and every `restart` iterations if `restart` is set.
"""
import numpy as np

from .linesearch import LineSearchError, strong_wolfe
from .result import Trace


def conjugate_gradient(obj, x0, beta="FR", tol=1e-6, max_iter=8000, c2=0.1, restart=None):
    if beta not in ("FR", "PR+"):
        raise ValueError(f"unknown beta {beta!r}")
    tr = Trace(obj)
    x = np.array(x0, dtype=float)
    f, g = obj.fun_grad(x)
    tr.log(f, g)
    d = -g
    alpha0 = 1.0 / max(1.0, np.linalg.norm(g))
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            return tr.result(x, f, g, k, tol, "converged")
        if g @ d >= 0:
            d = -g
        try:
            a, f_new, g_new = strong_wolfe(obj.fun_grad, x, f, g, d, alpha0=alpha0, c2=c2)
        except LineSearchError as e:
            return tr.result(x, f, g, k, tol, str(e), success=False)
        x = x + a * d
        gg = g @ g
        b = g_new @ g_new / gg if beta == "FR" else max(0.0, g_new @ (g_new - g) / gg)
        if restart and (k + 1) % restart == 0:
            b = 0.0
        # next initial trial step: assume the same first-order change as this step (N&W Eq. 3.60)
        d_new = -g_new + b * d
        alpha0 = a * (g @ d) / (g_new @ d_new) if g_new @ d_new < 0 else 1.0
        f, g, d = f_new, g_new, d_new
        tr.log(f, g)
    return tr.result(x, f, g, max_iter, tol, "maximum number of iterations reached")
