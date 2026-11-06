# Data

## Breast Cancer Wisconsin (Diagnostic)

Loaded with `sklearn.datasets.load_breast_cancer()`, a copy of the UCI file
(https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)).
569 samples, 30 features, 212 malignant / 357 benign. scikit-learn encodes malignant as 0, so
`lr_opt/data.py` flips the labels to malignant = 1. Nothing is stored here.

## Prostate Cancer (`prostate_cancer.csv`)

100 samples, 8 features (radius, texture, perimeter, area, smoothness, compactness, symmetry,
fractal_dimension), 62 malignant (`M`) / 38 benign (`B`), no missing values.

Source: https://github.com/SAZZZO99/PROSTATE-CANCER-DETECTION. That repository contains notebooks
but no CSV file. The notebook `PROSTATE_CANCER_KAGGLE.ipynb` uploads the file `PTRAIN.csv`
(4,397 bytes), and its saved cell output contains the full file contents. `prostate_cancer.csv` is
that file, byte for byte:

    sha256 1db89cad00041ab8840d6913f0075c1e2f6cf2f4ae4060c14f59d689686e4476

`python data/fetch_prostate.py` downloads the notebook again, extracts the file and checks the hash.
With the stratified 70/30 split (`random_state=42`) the test set has 30 samples, 19 malignant and
11 benign.
