"""Five unconstrained minimization methods for ridge-regularized logistic regression."""
from .cg import conjugate_gradient
from .gd import gradient_descent
from .newton import newton_raphson
from .objective import RidgeLogistic
from .quasi_newton import bfgs, lbfgs
from .scipy_ref import SCIPY_METHODS, scipy_minimize

# The five methods with their default settings: tolerance 1e-6 on ||grad J||_2,
# at most 8,000 iterations (300 for Newton-Raphson).
OPTIMIZERS = {
    "GD": lambda obj, x0: gradient_descent(obj, x0, step=0.1, max_iter=8000),
    "NR": lambda obj, x0: newton_raphson(obj, x0, max_iter=300),
    "CG": lambda obj, x0: conjugate_gradient(obj, x0, beta="FR", max_iter=8000),
    "BFGS": lambda obj, x0: bfgs(obj, x0, max_iter=8000),
    "L-BFGS": lambda obj, x0: lbfgs(obj, x0, m=10, max_iter=8000),
}


def scipy_protocol(name):
    """Reference runs: CG/BFGS/L-BFGS through scipy; GD and NR (no scipy counterpart) from lr_opt."""
    if name in SCIPY_METHODS:
        return lambda obj, x0, record=True: scipy_minimize(obj, x0, name, record=record)
    return lambda obj, x0, record=True: OPTIMIZERS[name](obj, x0)
