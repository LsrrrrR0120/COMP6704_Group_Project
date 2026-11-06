"""Ablations, curvature, random splits and scaling -> results/extra.json.

    python -m experiments.extra

1. ablation   design choices inside each method, at the selected lambda: GD step size (fixed
              0.01 / 0.1 / 1/L, Armijo), CG coefficient (FR / PR+) and restarts, BFGS initial
              matrix, L-BFGS memory m. No iteration cap (10^6), so every variant converges.
2. rates      Hessian spectrum at the minimizer, the predicted linear rate of fixed-step GD
              and the measured one, and the per-iteration gradient-norm ratios of all methods.
3. splits     the whole pipeline (split, CV choice of lambda, five optimizers) on 20 random
              70/30 splits instead of the single fixed split.
4. scaling    per-iteration cost and time to tolerance on synthetic data with growing dimension d,
              to check the complexity analysis (n = 4000, d = 10 ... 1000, lambda = 1).
"""
import json

import numpy as np

from experiments.run import ROOT, SEED, is_separable, select_lambda
from lr_opt import OPTIMIZERS, RidgeLogistic, bfgs, conjugate_gradient, gradient_descent, lbfgs, newton_raphson
from lr_opt.data import DATASETS, train_test
from lr_opt.metrics import classification_metrics

MAX_ITER = 10**6
N_SPLITS = 20
TIMING_REPEATS = 3


def timed(fn, repeats=TIMING_REPEATS):
    runs = [fn() for _ in range(repeats)]
    r = runs[0]
    return dict(n_iter=r.n_iter, converged=r.converged, fun=r.fun, grad_norm=r.grad_norm,
                nfev=r.nfev, ngev=r.ngev, nhev=r.nhev, time=float(np.median([x.time for x in runs])),
                history=dict(f=r.history["f"], grad_norm=r.history["grad_norm"]))


def global_L(obj):
    """Upper bound on the largest Hessian eigenvalue: lambda_max(A^T A) / (4n) + lam / n."""
    return np.linalg.eigvalsh(obj.X.T @ obj.X)[-1] / (4 * obj.n) + obj.lam / obj.n


def ablation(obj):
    x0 = np.zeros(obj.d)
    L = global_L(obj)
    variants = {
        "GD, a = 0.01": lambda: gradient_descent(obj, x0, step=0.01, max_iter=MAX_ITER),
        "GD, a = 0.1 (default)": lambda: gradient_descent(obj, x0, step=0.1, max_iter=MAX_ITER),
        "GD, a = 1/L": lambda: gradient_descent(obj, x0, step=1 / L, max_iter=MAX_ITER),
        "GD, Armijo": lambda: gradient_descent(obj, x0, step="armijo", max_iter=MAX_ITER),
        "CG, FR (default)": lambda: conjugate_gradient(obj, x0, beta="FR", max_iter=MAX_ITER),
        "CG, FR, restart every d": lambda: conjugate_gradient(obj, x0, beta="FR", restart=obj.d, max_iter=MAX_ITER),
        "CG, PR+": lambda: conjugate_gradient(obj, x0, beta="PR+", max_iter=MAX_ITER),
        "CG, PR+, restart every d": lambda: conjugate_gradient(obj, x0, beta="PR+", restart=obj.d, max_iter=MAX_ITER),
        "BFGS, H0 = I (default)": lambda: bfgs(obj, x0, init="identity", max_iter=MAX_ITER),
        "BFGS, H0 scaled": lambda: bfgs(obj, x0, init="scaled", max_iter=MAX_ITER),
        "BFGS, H0 = inverse Hessian": lambda: bfgs(obj, x0, init="hessian", max_iter=MAX_ITER),
        **{f"L-BFGS, m = {m}" + (" (default)" if m == 10 else ""):
           (lambda m=m: lbfgs(obj, x0, m=m, max_iter=MAX_ITER)) for m in (1, 3, 5, 10, 20, 50)},
        "NR, pure (default)": lambda: newton_raphson(obj, x0),
        "NR, damped": lambda: newton_raphson(obj, x0, damped=True),
    }
    return {name: timed(fn) for name, fn in variants.items()}, L


def rates(obj, L_glob):
    b_star = newton_raphson(obj, np.zeros(obj.d), tol=1e-13).x
    ev = np.linalg.eigvalsh(obj.hess(b_star))
    mu, L = float(ev[0]), float(ev[-1])
    # fixed-step GD near b*: ||b_{k+1} - b*|| ~ rho ||b_k - b*||, so J - J* contracts by rho^2
    rho = max(abs(1 - 0.1 * mu), abs(1 - 0.1 * L))
    gd = gradient_descent(obj, np.zeros(obj.d), step=0.1, max_iter=MAX_ITER)
    gap = np.array(gd.history["f"]) - obj.fun(b_star)
    idx = np.where((gap > 1e-13) & (gap < 1e-8))[0]
    measured = float(np.exp(np.polyfit(idx, np.log(gap[idx]), 1)[0]))
    return dict(mu=mu, L=L, kappa=L / mu, L_global=float(L_glob), gd_step=0.1,
                gd_predicted_rate=rho**2, gd_measured_rate=measured, gd_iterations_to_tol=gd.n_iter)


def splits(name):
    out = []
    for seed in range(N_SPLITS):
        D = train_test(name, seed=seed)
        lam, _ = select_lambda(D["X_raw"], D["y"])
        obj = RidgeLogistic(D["A"], D["y"], lam)
        row = dict(seed=seed, lam=lam, separable=is_separable(D["A"], D["y"]), runs={})
        for k, opt in OPTIMIZERS.items():
            r = opt(obj, np.zeros(obj.d))
            m = classification_metrics(D["y_test"], obj.predict_proba(r.x, D["A_test"]))
            row["runs"][k] = dict(n_iter=r.n_iter, converged=r.converged, time=r.time, fun=r.fun,
                                  accuracy=m["accuracy"], auc=m["auc"], log_loss=m["log_loss"])
        out.append(row)
        print(f"  {name} split {seed}: lambda = {lam:g}, separable = {row['separable']}")
    return out


def scaling(n=4000, dims=(10, 30, 100, 300, 1000), lam=1.0):
    rng = np.random.default_rng(SEED)
    out = {}
    for d in dims:
        X = rng.standard_normal((n, d))
        w = rng.standard_normal(d) * 2 / np.sqrt(d)
        y = (rng.random(n) < 1 / (1 + np.exp(-X @ w))).astype(int)
        obj = RidgeLogistic(np.hstack([np.ones((n, 1)), X]), y, lam)
        x0 = np.zeros(obj.d)
        variants = {
            "GD": lambda: gradient_descent(obj, x0, step="armijo", max_iter=20000),
            "NR": lambda: newton_raphson(obj, x0),
            "CG": lambda: conjugate_gradient(obj, x0, beta="FR", max_iter=20000),
            "BFGS": lambda: bfgs(obj, x0, max_iter=20000),
            "L-BFGS": lambda: lbfgs(obj, x0, max_iter=20000),
        }
        out[d] = {}
        for k, fn in variants.items():
            r = timed(fn, repeats=1 if d >= 300 else TIMING_REPEATS)
            r.pop("history")
            r["time_per_iter"] = r["time"] / max(r["n_iter"], 1)
            out[d][k] = r
            print(f"  scaling d = {d:4d} {k:7s} iter = {r['n_iter']:5d} time = {r['time']:.3f}s conv = {r['converged']}")
    return out


def main():
    out = dict(ablation={}, rates={}, splits={})
    for name in DATASETS:
        D = train_test(name)
        lam, _ = select_lambda(D["X_raw"], D["y"])
        obj = RidgeLogistic(D["A"], D["y"], lam)
        abl, L_glob = ablation(obj)
        out["ablation"][name] = abl
        out["rates"][name] = rates(obj, L_glob)
        print(f"{name}: ablation and rates done")
        out["splits"][name] = splits(name)
    out["scaling"] = scaling()
    path = ROOT / "results" / "extra.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
