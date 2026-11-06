"""Ridge-regularized logistic regression objective, scaled by 1/n.

    J(b) = (1/n) * [ sum_i ( log(1 + exp(x_i^T b)) - y_i x_i^T b ) + (lam / 2) * ||b_{1:}||^2 ]

The intercept b_0 is not penalized, as in scikit-learn's LogisticRegression with C = 1 / lam.
The 1/n scaling does not change the minimizer; it divides J, its gradient and its Hessian by n,
so that one fixed step size and one gradient-norm tolerance mean the same on datasets of
different size.
"""
import numpy as np
from scipy.special import expit


class RidgeLogistic:
    """J, grad J and Hessian of J for a design matrix X whose first column is the intercept."""

    def __init__(self, X, y, lam):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.lam = float(lam)
        self.n, self.d = self.X.shape
        self.mask = np.ones(self.d)
        self.mask[0] = 0.0          # the intercept is not penalized
        self.reset_counts()

    def reset_counts(self):
        self.nfev = self.ngev = self.nhev = 0

    def fun(self, b):
        self.nfev += 1
        z = self.X @ b
        nll = np.logaddexp(0.0, z) - self.y * z
        return (nll.sum() + 0.5 * self.lam * np.sum((self.mask * b) ** 2)) / self.n

    def grad(self, b):
        self.ngev += 1
        r = expit(self.X @ b) - self.y
        return (self.X.T @ r + self.lam * self.mask * b) / self.n

    def fun_grad(self, b):
        """Value and gradient from one product X b (used by the line searches)."""
        self.nfev += 1
        self.ngev += 1
        z = self.X @ b
        nll = np.logaddexp(0.0, z) - self.y * z
        f = (nll.sum() + 0.5 * self.lam * np.sum((self.mask * b) ** 2)) / self.n
        g = (self.X.T @ (expit(z) - self.y) + self.lam * self.mask * b) / self.n
        return f, g

    def hess(self, b):
        self.nhev += 1
        p = expit(self.X @ b)
        w = p * (1.0 - p)
        return (self.X.T * w) @ self.X / self.n + np.diag(self.lam * self.mask) / self.n

    def predict_proba(self, b, X=None):
        return expit((self.X if X is None else X) @ b)
