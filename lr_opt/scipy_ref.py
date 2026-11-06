"""Reference solvers: scipy.optimize.minimize with method CG, BFGS or L-BFGS-B.

Run with options = {"gtol": 1e-6, "maxiter": 8000}, from the same starting point as lr_opt.
Note that scipy's gtol is a test on the max-norm of the gradient (and L-BFGS-B also stops on a
relative decrease of f), while lr_opt stops on the 2-norm ||grad J||_2 < 1e-6. scipy's BFGS and
L-BFGS-B therefore report success at points where ||grad J||_2 is still above 1e-6.
"""
import numpy as np
from scipy.optimize import minimize

from .result import Trace

SCIPY_METHODS = {"CG": "CG", "BFGS": "BFGS", "L-BFGS": "L-BFGS-B"}


def scipy_minimize(obj, x0, method, tol=1e-6, max_iter=8000, record=True):
    """Runs scipy and returns an OptResult; `record` adds a history (one extra f/grad per iterate)."""
    tr = Trace(obj)
    f0, g0 = obj.fun_grad(np.asarray(x0, dtype=float))
    obj.reset_counts()
    if record:
        tr.log(f0, g0)

    def callback(xk, *args):
        counts = obj.nfev, obj.ngev
        tr.log(*obj.fun_grad(xk))
        obj.nfev, obj.ngev = counts

    r = minimize(obj.fun, x0, jac=obj.grad, method=SCIPY_METHODS[method],
                 callback=callback if record else None, options=dict(gtol=tol, maxiter=max_iter))
    counts = obj.nfev, obj.ngev
    f, g = obj.fun_grad(r.x)
    obj.nfev, obj.ngev = counts
    return tr.result(r.x, f, g, int(r.nit), tol, str(r.message), success=bool(r.success))
