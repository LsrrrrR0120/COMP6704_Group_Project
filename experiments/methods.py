"""Per-method experiments -> results/methods.json.

    python -m experiments.methods

1. newton_starts  pure vs damped Newton from starting points at growing distance t from the
                  origin (10 random directions per t), at the selected lambda.
2. newton_cost    time to form the Hessian, to solve the Newton system and to evaluate the gradient,
                  on the synthetic data of the scaling study (n = 4000, d = 10 ... 1000).
3. cg_c2          FR and PR+ with the strong Wolfe parameter c2 in {0.01, 0.1, 0.3, 0.5, 0.9}.
4. lbfgs_memory   L-BFGS with m = 1 ... 50 on the synthetic problem with d = 1000.
"""
import json
import time

import numpy as np

from experiments.run import ROOT, SEED, select_lambda
from lr_opt import RidgeLogistic, conjugate_gradient, lbfgs, newton_raphson
from lr_opt.data import DATASETS, train_test

STARTS = (0, 1, 2, 5, 10, 20, 50)
N_DIRECTIONS = 10
C2 = (0.01, 0.1, 0.3, 0.5, 0.9)
MEMORY = (1, 3, 5, 10, 20, 50)
DIMS = (10, 30, 100, 300, 1000)


def problem(name):
    D = train_test(name)
    lam, _ = select_lambda(D["X_raw"], D["y"])
    return RidgeLogistic(D["A"], D["y"], lam)


def synthetic(n=4000, dims=DIMS, lam=1.0):
    """The same data as the scaling study in experiments/extra.py (one generator, dims in order)."""
    rng = np.random.default_rng(SEED)
    out = {}
    for d in dims:
        X = rng.standard_normal((n, d))
        w = rng.standard_normal(d) * 2 / np.sqrt(d)
        y = (rng.random(n) < 1 / (1 + np.exp(-X @ w))).astype(int)
        out[d] = RidgeLogistic(np.hstack([np.ones((n, 1)), X]), y, lam)
    return out


def newton_starts(obj):
    rng = np.random.default_rng(SEED)
    U = rng.standard_normal((N_DIRECTIONS, obj.d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    out = {}
    for t in STARTS:
        row = {}
        for damped in (False, True):
            runs = [newton_raphson(obj, t * u, damped=damped) for u in (U[:1] if t == 0 else U)]
            row["damped" if damped else "pure"] = dict(
                converged=sum(r.converged for r in runs), runs=len(runs),
                iterations=[r.n_iter for r in runs], messages=sorted({r.message for r in runs}))
        out[t] = row
    return out


def median_time(fn, repeats=7):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def newton_cost(problems):
    out = {}
    for d, obj in problems.items():
        b = np.zeros(obj.d)
        H, g = obj.hess(b), obj.grad(b)
        out[d] = dict(hessian=median_time(lambda: obj.hess(b)), solve=median_time(lambda: np.linalg.solve(H, g)),
                      gradient=median_time(lambda: obj.grad(b)))
        print(f"  newton cost d = {d:4d}: " + ", ".join(f"{k} {1000 * v:.3f} ms" for k, v in out[d].items()))
    return out


def cg_c2(obj):
    out = {}
    for beta in ("FR", "PR+"):
        for c2 in C2:
            r = conjugate_gradient(obj, np.zeros(obj.d), beta=beta, c2=c2)
            out[f"{beta}, c2 = {c2:g}"] = dict(beta=beta, c2=c2, n_iter=r.n_iter, nfev=r.nfev, converged=r.converged,
                                              message=r.message, grad_norm=r.grad_norm)
    return out


def lbfgs_memory(obj):
    out = {}
    for m in MEMORY:
        runs = [lbfgs(obj, np.zeros(obj.d), m=m, max_iter=20000) for _ in range(3)]
        r = runs[0]
        out[m] = dict(n_iter=r.n_iter, converged=r.converged, nfev=r.nfev,
                      time=float(np.median([x.time for x in runs])))
        print(f"  L-BFGS d = 1000, m = {m:2d}: {r.n_iter} iterations, {out[m]['time']:.3f} s")
    return out


def main():
    out = dict(newton_starts={}, cg_c2={})
    for name in DATASETS:
        obj = problem(name)
        out["newton_starts"][name] = newton_starts(obj)
        out["cg_c2"][name] = cg_c2(obj)
        print(f"{name}: Newton starting points and CG c2 done")
    syn = synthetic()
    out["newton_cost"] = newton_cost(syn)
    out["lbfgs_memory"] = lbfgs_memory(syn[1000])
    path = ROOT / "results" / "methods.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
