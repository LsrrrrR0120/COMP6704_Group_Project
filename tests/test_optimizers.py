"""python -m pytest tests"""
import numpy as np
import pytest
from scipy.optimize import minimize

from lr_opt import OPTIMIZERS, RidgeLogistic, bfgs, conjugate_gradient, gradient_descent
from lr_opt.data import train_test
from lr_opt.linesearch import strong_wolfe

CASES = [("breast", 1.0), ("prostate", 10.0)]


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def problem(request):
    name, lam = request.param
    D = train_test(name)
    obj = RidgeLogistic(D["A"], D["y"], lam)
    ref = minimize(obj.fun, np.zeros(obj.d), jac=obj.grad, method="L-BFGS-B",
                   options=dict(gtol=1e-12, ftol=1e-15, maxiter=10000))
    return obj, ref.x


def near_minimizer(obj, b_star, r):
    """Strong convexity with mu = smallest Hessian eigenvalue gives ||b - b*|| <= ||grad J(b)|| / mu
    and J(b) - J* <= ||grad J(b)||^2 / (2 mu). mu is taken at b* rather than as a global lower
    bound, so both checks allow a factor 2 of slack."""
    mu = np.linalg.eigvalsh(obj.hess(b_star))[0]
    return (np.linalg.norm(r.x - b_star) <= 2 * r.grad_norm / mu + 1e-8
            and r.fun - obj.fun(b_star) <= r.grad_norm ** 2 / mu + 1e-14)


def test_derivatives(problem):
    obj, _ = problem
    rng = np.random.default_rng(0)
    b, v = rng.normal(size=obj.d), rng.normal(size=obj.d)
    eps = 1e-6
    fd_grad = (obj.fun(b + eps * v) - obj.fun(b - eps * v)) / (2 * eps)
    fd_hess = (obj.grad(b + eps * v) - obj.grad(b - eps * v)) / (2 * eps)
    assert np.isclose(obj.grad(b) @ v, fd_grad, rtol=1e-6)
    assert np.allclose(obj.hess(b) @ v, fd_hess, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("name", [k for k in OPTIMIZERS if k != "GD"])
def test_reaches_minimizer(problem, name):
    obj, b_star = problem
    r = OPTIMIZERS[name](obj, np.zeros(obj.d))
    assert r.converged and near_minimizer(obj, b_star, r)


def test_variants(problem):
    obj, b_star = problem
    x0 = np.zeros(obj.d)
    for r in (gradient_descent(obj, x0, step="armijo", max_iter=100000),
              conjugate_gradient(obj, x0, beta="PR+"), conjugate_gradient(obj, x0, restart=obj.d),
              bfgs(obj, x0, init="scaled"), bfgs(obj, x0, init="hessian")):
        assert r.converged and near_minimizer(obj, b_star, r)


def test_strong_wolfe_conditions():
    A = np.diag([1.0, 10.0, 100.0])
    fun_grad = lambda x: (0.5 * x @ A @ x, A @ x)
    x = np.ones(3)
    f0, g0 = fun_grad(x)
    d = -g0
    for c2 in (0.1, 0.9):
        a, f, g = strong_wolfe(fun_grad, x, f0, g0, d, c2=c2)
        assert f <= f0 + 1e-4 * a * (g0 @ d)
        assert abs(g @ d) <= c2 * abs(g0 @ d)
