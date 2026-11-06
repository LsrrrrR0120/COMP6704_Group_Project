"""Classification metrics (accuracy, AUC, precision, sensitivity, specificity, F1, log-loss) and the percentile bootstrap."""
import numpy as np
from sklearn.metrics import roc_auc_score

METRICS = ("accuracy", "auc", "precision", "sensitivity", "specificity", "f1", "log_loss")


def confusion(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def classification_metrics(y, p, threshold=0.5):
    """Accuracy in %, the other metrics as fractions; log-loss with probabilities clipped to [1e-15, 1 - 1e-15]."""
    tp, fp, tn, fn = confusion(y, p, threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return dict(
        accuracy=100.0 * (tp + tn) / len(y),
        auc=float(roc_auc_score(y, p)),
        precision=precision,
        sensitivity=sensitivity,
        specificity=tn / (tn + fp) if tn + fp else 0.0,
        f1=2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0,
        log_loss=float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
    )


def bootstrap_ci(y, p, n_boot=2000, level=0.95, seed=42):
    """Percentile intervals from resampling the test predictions with replacement.
    Resamples that contain a single class (AUC undefined) are drawn again."""
    rng = np.random.default_rng(seed)
    samples = {m: [] for m in METRICS}
    while len(samples["auc"]) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        for m, v in classification_metrics(y[idx], p[idx]).items():
            samples[m].append(v)
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {m: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for m, v in samples.items()}
