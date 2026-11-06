"""Line searches shared by CG, BFGS and L-BFGS (Nocedal & Wright, Numerical Optimization, Ch. 3).

strong_wolfe finds a step a > 0 along a descent direction d with
    f(x + a d) <= f(x) + c1 a g^T d                 (sufficient decrease)
    |grad f(x + a d)^T d| <= c2 |g^T d|              (strong curvature condition)
using the bracketing phase of Algorithm 3.5 and the zoom phase of Algorithm 3.6 with
cubic interpolation.
"""
import numpy as np


class LineSearchError(RuntimeError):
    pass


def strong_wolfe(fun_grad, x, f0, g0, d, alpha0=1.0, c1=1e-4, c2=0.9, alpha_max=1e10, max_iter=30):
    """Returns (alpha, f(x + alpha d), grad f(x + alpha d))."""
    dphi0 = float(g0 @ d)
    if dphi0 >= 0:
        raise LineSearchError("d is not a descent direction")

    a_prev, f_prev, dphi_prev = 0.0, f0, dphi0
    a = alpha0
    for i in range(max_iter):
        f_a, g_a = fun_grad(x + a * d)
        dphi_a = float(g_a @ d)
        if f_a > f0 + c1 * a * dphi0 or (i > 0 and f_a >= f_prev):
            return _zoom(fun_grad, x, d, f0, dphi0, (a_prev, f_prev, dphi_prev), (a, f_a, dphi_a), c1, c2)
        if abs(dphi_a) <= -c2 * dphi0:
            return a, f_a, g_a
        if dphi_a >= 0:
            return _zoom(fun_grad, x, d, f0, dphi0, (a, f_a, dphi_a), (a_prev, f_prev, dphi_prev), c1, c2)
        a_prev, f_prev, dphi_prev = a, f_a, dphi_a
        a = min(2.0 * a, alpha_max)
    raise LineSearchError("bracketing phase did not terminate")


def _zoom(fun_grad, x, d, f0, dphi0, lo, hi, c1, c2, max_iter=40):
    (a_lo, f_lo, dphi_lo), (a_hi, f_hi, dphi_hi) = lo, hi
    for _ in range(max_iter):
        a = _cubic_min(a_lo, f_lo, dphi_lo, a_hi, f_hi, dphi_hi)
        left, right = min(a_lo, a_hi), max(a_lo, a_hi)
        margin = 0.1 * (right - left)
        if a is None or not (left + margin <= a <= right - margin):
            a = 0.5 * (a_lo + a_hi)          # safeguard: bisection
        f_a, g_a = fun_grad(x + a * d)
        dphi_a = float(g_a @ d)
        if f_a > f0 + c1 * a * dphi0 or f_a >= f_lo:
            a_hi, f_hi, dphi_hi = a, f_a, dphi_a
        else:
            if abs(dphi_a) <= -c2 * dphi0:
                return a, f_a, g_a
            if dphi_a * (a_hi - a_lo) >= 0:
                a_hi, f_hi, dphi_hi = a_lo, f_lo, dphi_lo
            a_lo, f_lo, dphi_lo = a, f_a, dphi_a
        if abs(a_hi - a_lo) < 1e-16 * max(1.0, abs(a_lo)):
            break
    raise LineSearchError("zoom phase did not find a strong Wolfe point")


def _cubic_min(a, fa, da, b, fb, db):
    """Minimizer of the cubic interpolating f and f' at a and b (N&W Eq. 3.59), or None."""
    if a == b:
        return None
    d1 = da + db - 3.0 * (fa - fb) / (a - b)
    disc = d1 * d1 - da * db
    if disc < 0:
        return None
    d2 = np.sign(b - a) * np.sqrt(disc)
    denom = db - da + 2.0 * d2
    if denom == 0:
        return None
    return b - (b - a) * (db + d2 - d1) / denom


def backtracking(fun, x, f0, g0, d, alpha0=1.0, rho=0.5, c1=1e-4, max_iter=60):
    """Armijo backtracking. Returns (alpha, f(x + alpha d))."""
    dphi0 = float(g0 @ d)
    a = alpha0
    for _ in range(max_iter):
        f_a = fun(x + a * d)
        if f_a <= f0 + c1 * a * dphi0:
            return a, f_a
        a *= rho
    raise LineSearchError("backtracking did not find a sufficient decrease")
