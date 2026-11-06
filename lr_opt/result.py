"""Common result type and per-iteration trace shared by all optimizers."""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


@dataclass
class OptResult:
    x: np.ndarray
    fun: float
    grad_norm: float
    n_iter: int                 # number of parameter updates
    converged: bool             # ||grad J||_2 < tol (the stopping test of lr_opt)
    success: bool               # solver terminated normally (converged, or no further progress possible)
    message: str
    time: float                 # wall-clock seconds, including the per-iteration logging
    nfev: int
    ngev: int
    nhev: int
    history: dict = field(default_factory=dict)


class Trace:
    """Records f, ||grad f|| and elapsed time at every iterate (iterate 0 = starting point)."""

    def __init__(self, obj):
        self.obj = obj
        obj.reset_counts()
        self.t0 = perf_counter()
        self.f, self.gnorm, self.time = [], [], []

    def log(self, f, g):
        self.f.append(float(f))
        self.gnorm.append(float(np.linalg.norm(g)))
        self.time.append(perf_counter() - self.t0)

    def result(self, x, f, g, n_iter, tol, message, success=None):
        elapsed = perf_counter() - self.t0
        gnorm = float(np.linalg.norm(g))
        converged = gnorm < tol
        return OptResult(
            x=x, fun=float(f), grad_norm=gnorm, n_iter=n_iter, converged=converged,
            success=converged if success is None else success, message=message, time=elapsed,
            nfev=self.obj.nfev, ngev=self.obj.ngev, nhev=self.obj.nhev,
            history=dict(f=self.f, grad_norm=self.gnorm, time=self.time))
