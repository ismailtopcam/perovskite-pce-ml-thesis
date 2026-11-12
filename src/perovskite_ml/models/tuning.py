"""Grup-guvenli hiperparametre aramasi. Ayni DOI-grup CV protokolu altinda
default vs tuned karsilastirmasi yapar — 'tune ettin mi?' sorusunun cevabi."""
import numpy as np
from itertools import product
from sklearn.base import clone
from sklearn.metrics import r2_score
from perovskite_ml.validation.splits import group_kfold

def grouped_cv_r2_katlar(estimator, X, y, groups, n_splits=5, atama="kayit"):
    """Kat basina R2 listesi. `atama`: bkz. perovskite_ml.validation.splits."""
    scores = []
    for tr, te in group_kfold(X, y, groups, n_splits, atama):
        m = clone(estimator); m.fit(X.iloc[tr], y[tr])
        scores.append(float(r2_score(y[te], m.predict(X.iloc[te]))))
    return scores

def grouped_cv_r2(estimator, X, y, groups, n_splits=5, atama="kayit"):
    scores = grouped_cv_r2_katlar(estimator, X, y, groups, n_splits, atama)
    return float(np.mean(scores)), float(np.std(scores))

def grid_search_grouped(factory, param_grid, X, y, groups, n_splits=3, atama="kayit"):
    keys = list(param_grid)
    rows, best = [], None
    for combo in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        mean, std = grouped_cv_r2(factory(**params), X, y, groups, n_splits, atama)
        row = {**params, "cv_r2_mean": round(mean, 4), "cv_r2_std": round(std, 4)}
        rows.append(row)
        if best is None or mean > best["cv_r2_mean"]:
            best = row
    return best, rows
