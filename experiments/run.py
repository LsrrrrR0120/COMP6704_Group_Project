"""Main experiments on the fixed split -> results/results.json.

    python -m experiments.run

For each dataset:
  1. lambda selection by stratified 5-fold cross-validation on the training set (mean validation
     log-loss, L-BFGS), grid {1e-4, ..., 100}
  2. the five optimizers at the selected lambda and at lambda = 0
  3. test log-loss over a lambda grid including 0
  4. bootstrap 95% intervals of the Newton-Raphson model, 2,000 resamples
  5. a linear-programming check of whether the training classes are linearly separable,
     i.e. whether the unregularized maximum-likelihood estimate exists (Albert & Anderson, 1984)

Steps 2-3 run twice:
  "own"   all five methods implemented in lr_opt, and
  "scipy" scipy.optimize.minimize for CG, BFGS and L-BFGS-B as a reference (GD and NR from lr_opt).
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from sklearn.model_selection import StratifiedKFold

from lr_opt import OPTIMIZERS, RidgeLogistic, lbfgs, scipy_protocol
from lr_opt.data import DATASETS, Preprocessor, train_test
from lr_opt.metrics import bootstrap_ci, classification_metrics

ROOT = Path(__file__).resolve().parent.parent
CV_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SWEEP_GRID = [0.0] + CV_GRID
TIMING_REPEATS = 5
SEED = 42


def select_lambda(X_raw, y):
    """Mean validation log-loss per lambda; preprocessing is refitted inside every fold."""
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_raw, y))
    scores = []
    for lam in CV_GRID:
        losses = []
        for tr, va in folds:
            pre = Preprocessor().fit(X_raw[tr])
            obj = RidgeLogistic(pre.transform(X_raw[tr]), y[tr], lam)
            b = lbfgs(obj, np.zeros(obj.d)).x
            losses.append(classification_metrics(y[va], obj.predict_proba(b, pre.transform(X_raw[va])))["log_loss"])
        scores.append(float(np.mean(losses)))
    return CV_GRID[int(np.argmin(scores))], scores


def is_separable(A, y):
    """True iff some b satisfies (2 y_i - 1) a_i^T b >= 1 for all i (an LP feasibility problem)."""
    s = 2.0 * y - 1.0
    lp = linprog(np.zeros(A.shape[1]), A_ub=-(s[:, None] * A), b_ub=-np.ones(len(y)),
                 bounds=[(None, None)] * A.shape[1], method="highs")
    return bool(lp.status == 0)


def run(opt, obj, x0, A_test, y_test, protocol):
    """One run with history, plus the median wall-clock time of TIMING_REPEATS runs."""
    r = opt(obj, x0)
    times = [r.time]
    for _ in range(TIMING_REPEATS - 1):
        times.append((opt(obj, x0, record=False) if protocol == "scipy" else opt(obj, x0)).time)
    p = obj.predict_proba(r.x, A_test)
    train_acc = float(np.mean((obj.predict_proba(r.x) >= 0.5) == obj.y))
    return dict(fun=r.fun, grad_norm=r.grad_norm, n_iter=r.n_iter, converged=r.converged,
                success=r.success, message=r.message, time=float(np.median(times)),
                nfev=r.nfev, ngev=r.ngev, nhev=r.nhev, coef_norm=float(np.linalg.norm(r.x)),
                coef=r.x.tolist(), train_accuracy=100 * train_acc,
                metrics=classification_metrics(y_test, p), proba_test=p.tolist(), history=r.history)


def protocols():
    return {"own": {k: (lambda o: lambda obj, x0, record=True: o(obj, x0))(o) for k, o in OPTIMIZERS.items()},
            "scipy": {k: scipy_protocol(k) for k in OPTIMIZERS}}


def main():
    out = {}
    for name in DATASETS:
        D = train_test(name)
        lam, cv_scores = select_lambda(D["X_raw"], D["y"])
        x0 = np.zeros(D["A"].shape[1])
        res = dict(n_train=len(D["y"]), n_test=len(D["y_test"]), features=D["features"],
                   test_positives=int(D["y_test"].sum()), cv=dict(grid=CV_GRID, mean_log_loss=cv_scores, selected=lam),
                   train_separable=is_separable(D["A"], D["y"]), runs={}, sweep=dict(grid=SWEEP_GRID))
        print(f"{name}: selected lambda = {lam}, training set linearly separable: {res['train_separable']}")
        for proto, opts in protocols().items():
            res["runs"][proto] = {}
            for key, lam_k in (("selected", lam), ("0", 0.0)):
                obj = RidgeLogistic(D["A"], D["y"], lam_k)
                res["runs"][proto][key] = {k: run(o, obj, x0, D["A_test"], D["y_test"], proto) for k, o in opts.items()}
                for k, v in res["runs"][proto][key].items():
                    print(f"  {proto:5s} lam={lam_k:<5g} {k:7s} iter={v['n_iter']:5d} f={v['fun']:.6f} "
                          f"|g|={v['grad_norm']:.1e} t={v['time']:.4f}s acc={v['metrics']['accuracy']:.2f}")
            res["sweep"][proto] = {}
            for k, o in opts.items():
                losses = []
                for lam_k in SWEEP_GRID:
                    obj = RidgeLogistic(D["A"], D["y"], lam_k)
                    b = o(obj, x0, record=False).x if proto == "scipy" else o(obj, x0).x
                    losses.append(classification_metrics(D["y_test"], obj.predict_proba(b, D["A_test"]))["log_loss"])
                res["sweep"][proto][k] = losses
        nr = res["runs"]["own"]["selected"]["NR"]
        ci = bootstrap_ci(D["y_test"], np.array(nr["proba_test"]), n_boot=2000, seed=SEED)
        ci["accuracy"] = tuple(v / 100 for v in ci["accuracy"])      # accuracy as a fraction, like the other metrics
        est = dict(nr["metrics"], accuracy=nr["metrics"]["accuracy"] / 100)
        res["bootstrap"] = {m: [est[m], *ci[m]] for m in ci}
        out[name] = res
    path = ROOT / "results" / "results.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
