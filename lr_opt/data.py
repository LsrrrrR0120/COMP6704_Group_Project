"""Datasets and preprocessing.

- Breast Cancer Wisconsin (Diagnostic): 569 x 30, from scikit-learn (a copy of the UCI file).
  scikit-learn encodes malignant = 0; we use malignant = 1, so the labels are flipped.
- Prostate Cancer: 100 x 8, data/prostate_cancer.csv (see data/README.md for its provenance),
  malignant ("M") = 1.

Stratified 70/30 train/test split with random_state = 42, median imputation and z-score
standardization fitted on the training part only, then an intercept column of ones.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASETS = ("breast", "prostate")


def load_dataset(name):
    """Returns (X, y, feature_names) with y = 1 for malignant."""
    if name == "breast":
        d = load_breast_cancer()
        return d.data.astype(float), (1 - d.target).astype(int), list(d.feature_names)
    if name == "prostate":
        df = pd.read_csv(DATA_DIR / "prostate_cancer.csv")
        y = (df["diagnosis_result"] == "M").astype(int).to_numpy()
        X = df.drop(columns=["id", "diagnosis_result"])
        return X.to_numpy(dtype=float), y, list(X.columns)
    raise ValueError(f"unknown dataset {name!r}")


class Preprocessor:
    """Median imputation + standardization fitted on training rows, then an intercept column."""

    def fit(self, X):
        self.imputer = SimpleImputer(strategy="median").fit(X)
        self.scaler = StandardScaler().fit(self.imputer.transform(X))
        return self

    def transform(self, X):
        Z = self.scaler.transform(self.imputer.transform(X))
        return np.hstack([np.ones((len(Z), 1)), Z])


def train_test(name, test_size=0.3, seed=42):
    """Raw and preprocessed train/test parts of a dataset."""
    X, y, features = load_dataset(name)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    pre = Preprocessor().fit(X_tr)
    return dict(features=features, X_raw=X_tr, y=y_tr, A=pre.transform(X_tr),
                A_test=pre.transform(X_te), y_test=y_te)
