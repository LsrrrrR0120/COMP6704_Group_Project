"""Gradient descent.

Owner: LIN Shenrui

b_{k+1} = b_k - a * grad J(b_k), stopped when ||grad J||_2 < tol. The default is the fixed
step a = 0.1 on the 1/n-scaled objective (below 2 / L_bar on both datasets). With
step="armijo" the step is chosen by backtracking instead.
"""
import numpy as np

from .linesearch import LineSearchError, backtracking
from .result import Trace


def gradient_descent(obj, x0, step=0.1, tol=1e-6, max_iter=8000):
    tr = Trace(obj)
    x = np.array(x0, dtype=float)
    f, g = obj.fun_grad(x)
    tr.log(f, g)
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            return tr.result(x, f, g, k, tol, "converged")
        if step == "armijo":
            try:
                a, _ = backtracking(obj.fun, x, f, g, -g)
            except LineSearchError as e:
                return tr.result(x, f, g, k, tol, str(e), success=False)
        else:
            a = step
        x = x - a * g
        f, g = obj.fun_grad(x)
        tr.log(f, g)
    return tr.result(x, f, g, max_iter, tol, "maximum number of iterations reached")
