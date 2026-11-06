"""Figures from results/results.json -> figures/*.png (and .pdf).

    python -m experiments.make_figures

Per dataset: lambda_<ds> (test log-loss against lambda), convergence_<ds> (objective gap,
gradient norm, time and iterations), roc_<ds>, confusion_<ds> and coef_<ds>. lambda0 shows the
objective at lambda = 0 for both datasets.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from experiments.make_tables import NAMES, OPTS
from experiments.run import ROOT
from lr_opt.data import train_test
from lr_opt.metrics import confusion

COLORS = {"GD": "#E69F00", "NR": "#0072B2", "CG": "#009E73", "L-BFGS": "#D55E00"}   # Okabe-Ito
FIG = ROOT / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_lambda(ds, res):
    grid = np.array(res["sweep"]["grid"])
    x = np.arange(len(grid))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for k, marker in zip(OPTS, "osD^v"):
        ax.plot(x, res["sweep"]["own"][k], marker=marker, ms=4, lw=1.4, color=COLORS[k], label=k, mfc="none")
    sel = list(grid).index(res["cv"]["selected"])
    ax.axvline(sel, color="0.4", ls=":", lw=1)
    ax.annotate(f"CV choice λ = {grid[sel]:g}", (sel, ax.get_ylim()[1]), xytext=(-4, -2),
                textcoords="offset points", fontsize=8, color="0.3", va="top", ha="right")
    ax.set_xticks(x, ["0"] + [f"$10^{{{int(np.log10(g))}}}$" for g in grid[1:]])
    ax.set_xlabel("ridge parameter λ")
    ax.set_ylabel("test log-loss")
    ax.set_title(f"{NAMES[ds]}: effect of λ")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, f"lambda_{ds}")


def fig_convergence(ds, res):
    runs = {p: res["runs"][p]["selected"] for p in ("own", "scipy")}
    f_star = min(min(r["history"]["f"]) for p in runs.values() for r in p.values())
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), layout="constrained")
    for k in OPTS:
        for p, ls in (("own", "-"), ("scipy", "--")):
            if p == "scipy" and k in ("GD", "NR"):
                continue          # identical to "own"
            h = runs[p][k]["history"]
            it = np.arange(1, len(h["f"]) + 1)
            label = k if p == "own" else f"{k} (SciPy)"
            axes[0, 0].loglog(it, np.maximum(np.array(h["f"]) - f_star, 1e-16), ls, color=COLORS[k], lw=1.3, label=label)
            axes[0, 1].loglog(it, h["grad_norm"], ls, color=COLORS[k], lw=1.3)
    axes[0, 1].axhline(1e-6, color="0.4", ls=":", lw=1)
    axes[0, 1].annotate("tol = 1e-6", (1.2, 1e-6), xytext=(0, 3), textcoords="offset points", fontsize=8, color="0.3")
    axes[0, 0].set(xlabel="iteration + 1", ylabel="J(b) − J*", title="objective gap")
    axes[0, 1].set(xlabel="iteration + 1", ylabel="‖∇J(b)‖₂", title="gradient norm")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8)

    x = np.arange(len(OPTS))
    w = 0.38
    for j, (p, hatch) in enumerate((("own", None), ("scipy", "///"))):
        times = [1000 * runs[p][k]["time"] for k in OPTS]
        iters = [runs[p][k]["n_iter"] for k in OPTS]
        kw = dict(width=w, color=[COLORS[k] for k in OPTS], edgecolor="white", hatch=hatch,
                  label="our implementations" if p == "own" else "SciPy for CG/L-BFGS")
        axes[1, 0].bar(x + (j - 0.5) * w, times, **kw)
        axes[1, 1].bar(x + (j - 0.5) * w, iters, **kw)
    for ax, title, ylabel, values in (
            (axes[1, 0], "wall-clock time (median of 5)", "time (ms)",
             [1000 * runs[p][k]["time"] for p in runs for k in OPTS]),
            (axes[1, 1], "iterations", "iterations", [runs[p][k]["n_iter"] for p in runs for k in OPTS])):
        ax.set_yscale("log")
        ax.set_ylim(min(values) / 3, max(values) * 40)        # headroom for the legend
        ax.set_xticks(x, OPTS)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="x", visible=False)
    axes[1, 1].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"{NAMES[ds]}, λ = {res['cv']['selected']:g}")
    save(fig, f"convergence_{ds}")


def fig_roc(ds, res, y_test):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for k, ls in zip(OPTS, ["-", "--", "-.", ":", (0, (5, 1))]):
        r = res["runs"]["own"]["selected"][k]
        fpr, tpr, _ = roc_curve(y_test, r["proba_test"])
        ax.plot(fpr, tpr, ls=ls, color=COLORS[k], lw=1.4, label=f"{k} (AUC {r['metrics']['auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
    ax.set(xlabel="false positive rate", ylabel="sensitivity", title=f"{NAMES[ds]}: ROC", aspect="equal")
    ax.legend(loc="lower right", fontsize=7)
    save(fig, f"roc_{ds}")


def fig_confusion(ds, res, y_test):
    fig, axes = plt.subplots(1, len(OPTS), figsize=(9, 2.2))
    for ax, k in zip(axes, OPTS):
        tp, fp, tn, fn = confusion(y_test, np.array(res["runs"]["own"]["selected"][k]["proba_test"]))
        M = np.array([[tn, fp], [fn, tp]])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > M.max() / 2 else "black")
        ax.set_xticks([0, 1], ["benign", "malig."])
        ax.set_yticks([0, 1], ["benign", "malig."] if k == OPTS[0] else ["", ""])
        ax.set_title(k)
        ax.grid(False)
    axes[0].set_ylabel("true")
    axes[len(OPTS) // 2].set_xlabel("predicted")
    fig.suptitle(f"{NAMES[ds]}: confusion matrices, λ = {res['cv']['selected']:g}", y=1.05)
    save(fig, f"confusion_{ds}")


def fig_coef(ds, res):
    C = np.array([res["runs"]["own"]["selected"][k]["coef"] for k in OPTS]).T
    labels = ["intercept"] + res["features"]
    lim = np.abs(C).max()
    fig, ax = plt.subplots(figsize=(3.6, 0.2 * len(labels) + 1.0))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(OPTS)), OPTS)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="coefficient (standardized features)")
    ax.set_title(f"{NAMES[ds]}: coefficients, λ = {res['cv']['selected']:g}")
    save(fig, f"coef_{ds}")


def fig_lambda0(results):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, (ds, res) in zip(axes, results.items()):
        for k in OPTS:
            h = res["runs"]["own"]["0"][k]["history"]
            ax.loglog(np.arange(1, len(h["f"]) + 1), h["f"], color=COLORS[k], lw=1.3, label=k)
        sep = "linearly separable: no minimizer" if res["train_separable"] else "not separable: finite minimizer"
        ax.set(xlabel="iteration + 1", ylabel="J(b), λ = 0", title=f"{NAMES[ds]}\n({sep})")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, "lambda0")


def main():
    FIG.mkdir(exist_ok=True)
    results = json.loads((ROOT / "results" / "results.json").read_text())
    for ds, res in results.items():
        y_test = train_test(ds)["y_test"]
        fig_lambda(ds, res)
        fig_convergence(ds, res)
        fig_roc(ds, res, y_test)
        fig_confusion(ds, res, y_test)
        fig_coef(ds, res)
    fig_lambda0(results)
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    main()
