# Unconstrained optimization for regularized logistic regression in tumour classification

COMP6704 (PolyU, Fall 2026) group project, Group **22**.

We fit ridge-regularized logistic regression on two tumour datasets (breast cancer, 569 samples;
prostate cancer, 100 samples) with four unconstrained minimization methods: gradient descent,
Newton–Raphson, nonlinear conjugate gradient and L-BFGS. All four are implemented from scratch in
`lr_opt/`, with one shared strong Wolfe line search, and are checked against SciPy's CG and
L-BFGS-B. The repository contains the code, the data and every experiment behind our report.

## Main results

1. **Same model, different cost.** At the cross-validated λ every method that reaches
   ‖∇J‖₂ < 10⁻⁶ ends at the same objective value and makes the same test predictions
   (breast: accuracy 97.08 %, AUC 0.9977; prostate: 83.33 %, 0.8804). Newton–Raphson needs 4–8
   iterations, CG and L-BFGS 11–35, and gradient descent with a fixed step 728 on
   prostate and 26,420 on breast cancer.
2. **Curvature explains the iteration counts.** At the breast-cancer minimizer the smallest Hessian
   eigenvalue is μ* = 0.0025 ≈ λ/n and κ* = 61 (prostate: 3.9). The predicted per-iteration
   contraction of fixed-step GD, (1 − 0.1 μ*)² = 0.99950, matches the measured 0.99943. The global
   bound L̄ = 3.28 is 22 times the local curvature L* = 0.15, so a step of 1/L̄ is too short.
3. **SciPy reports success above the tolerance.** SciPy's `gtol` bounds ‖∇J‖∞ (and L-BFGS-B also
   stops on a small relative decrease of J), so L-BFGS-B stops with ‖∇J‖₂ between
   2.8 × 10⁻⁶ and 9.4 × 10⁻⁶. Our 2-norm test is reached in a similar number of iterations.
4. **Without regularization the breast-cancer training set is linearly separable** (LP check in
   `experiments/run.py`), so the unregularized problem has no minimizer: J → 0 and ‖b‖ → ∞
   (`figures/lambda0.png`). NR, CG and L-BFGS still pass the gradient test, at ‖b‖ between 265
   and 979 and with 100 % training accuracy. Their test log-loss (1.6–2.3) depends on where each
   solver stops. Gradient descent stops after 8,000 iterations at ‖b‖ = 6.2, and this early stop acts
   as a regularizer (log-loss 0.10). Prostate is not separable, and all methods reach the same minimum.
5. **Ablations** (no iteration cap): Armijo backtracking cuts GD on breast from 26,420 to 2,633
   iterations. FR vs. PR+ and restarts hardly matter (35–40 iterations). L-BFGS improves from 48 (m = 1) to 30 (m = 50) iterations.
6. **20 random splits**: breast is separable in 20/20 splits, and GD never converges within 8,000
   iterations. On prostate, CV picks λ = 1 or 10, and the test accuracy is 87.8 ± 5.9 % (fixed split: 83.33 %).
7. **Scaling with d** (synthetic, n = 4000): per-iteration time grows like d^1.1 for GD, CG and L-BFGS
   and d^1.6 for NR. At d = 1000, L-BFGS reaches the tolerance in 0.08 s, CG in 0.17 s and NR in 0.36 s.

## Setup

| Item | Setting |
|---|---|
| Objective | J(b) = (1/n)[Σᵢ log(1+e^{aᵢᵀb}) − yᵢaᵢᵀb + (λ/2)‖b₁:ₚ‖²], intercept not penalized. Dividing by n lets one step size and one tolerance serve both datasets. |
| Data | malignant = 1; stratified 70/30 split, `random_state=42`; median imputation and z-scoring fitted on the training part (see `data/README.md`). |
| λ | stratified 5-fold CV (`shuffle=True, random_state=42`), mean validation log-loss, grid 10⁻⁴…10² → λ = 1 (breast), 10 (prostate); also λ = 0. |
| Stopping | ‖∇J‖₂ < 10⁻⁶ from b₀ = 0; at most 8,000 iterations (NR: 300). |
| Defaults | GD fixed step 0.1; NR unit step; CG Fletcher–Reeves with strong Wolfe c₂ = 0.1; L-BFGS m = 10. |
| SciPy reference | `minimize(method="CG" / "L-BFGS-B", options={"gtol": 1e-6, "maxiter": 8000})`. |
| Bootstrap | 2,000 resamples of the test set, `numpy.random.default_rng(42)`. |

## Layout

```
lr_opt/                   the optimization library
  objective.py            RidgeLogistic: J, grad J, Hessian (1/n-scaled, intercept unpenalized)
  linesearch.py           strong Wolfe (bracketing + zoom, cubic interpolation) and Armijo backtracking
  gd.py                   gradient descent (fixed step or Armijo)                  -- YU Kaijing
  newton.py               Newton-Raphson (pure or damped)                          -- ZHANG Qigang
  cg.py                   nonlinear CG (Fletcher-Reeves / Polak-Ribiere+)          -- LIN Shenrui
  quasi_newton.py         L-BFGS (two-loop recursion)                              -- PAN Pan
  scipy_ref.py            SciPy's CG / L-BFGS-B as a reference
  data.py                 datasets, split and preprocessing
  metrics.py              accuracy, AUC, precision, sensitivity, specificity, F1, log-loss, bootstrap CI
  result.py               OptResult and the per-iteration trace
experiments/
  run.py                  main experiments on the fixed split -> results/results.json
  make_tables.py          -> results/tables.md
  make_figures.py         -> figures/*.png, *.pdf
  extra.py                ablations, curvature/rates, 20 random splits, scaling -> results/extra.json
  make_extra.py           -> results/extra.md, figures/{ablation,order,splits,scaling}.*
  methods.py              per-method studies: Newton starting points and cost, CG line search c2,
                          L-BFGS memory at d = 1000 -> results/methods.json
data/                     prostate_cancer.csv and its provenance (data/README.md, data/fetch_prostate.py)
tests/                    derivative checks, convergence of every method to the minimizer, Wolfe conditions
results/  figures/        generated outputs (committed)
```

In `results/results.json`, `runs.own` holds our implementations and `runs.scipy` the SciPy
reference runs (GD and NR have no SciPy counterpart, so those entries repeat our runs).

| Output | File |
|---|---|
| optimization at the selected λ and at λ = 0, test metrics, bootstrap CI | `results/tables.md` |
| test log-loss against λ | `figures/lambda_breast.png`, `figures/lambda_prostate.png` |
| convergence, time and iterations | `figures/convergence_*.png` |
| order of convergence | `figures/order.png` |
| ROC, confusion matrices, coefficients | `figures/roc_*.png`, `figures/confusion_*.png`, `figures/coef_*.png` |
| λ = 0 and separability | `figures/lambda0.png` |
| ablations, curvature, splits, scaling | `results/extra.md`, `figures/{ablation,splits,scaling}.png` |

## Running

Tested with Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, pandas 2.2.3,
Matplotlib 3.10.0 on Windows 11 (CPU only; the main experiments run in under a minute, the
additional ones in a few minutes).

```bash
pip install -r requirements.txt
python -m pytest tests                  # 11 tests
python -m experiments.run               # main runs -> results/results.json
python -m experiments.make_tables       # -> results/tables.md
python -m experiments.make_figures      # -> figures/
python -m experiments.extra             # additional experiments -> results/extra.json
python -m experiments.make_extra        # -> results/extra.md, figures/
python -m experiments.methods           # per-method studies -> results/methods.json
```

Wall-clock times depend on the machine; iteration counts, costs and metrics are deterministic.

## Team

| Member | Student ID | Implemented |
|---|---|---|
| LIN Shenrui | 25028243r | nonlinear conjugate gradient (`lr_opt/cg.py`) |
| PAN Pan | 25021421r | L-BFGS (`lr_opt/quasi_newton.py`) |
| YU Kaijing | 25145003r | gradient descent (`lr_opt/gd.py`) |
| ZHANG Qigang | 25016402r | Newton–Raphson (`lr_opt/newton.py`) |
