import numpy as np, pandas as pd
from perovskite_ml.validation.splits import holdout_split, group_kfold

def _data(n=200, n_groups=20):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f": rng.randn(n)})
    y = rng.randn(n)
    groups = np.array([f"doi{i%n_groups}" for i in range(n)])
    return X, y, groups

def test_holdout_groups_disjoint():
    X, y, g = _data()
    tr, te = holdout_split(X, y, g, 0.2, 42)
    assert set(g[tr]).isdisjoint(set(g[te]))   # ayni DOI iki tarafta olamaz

def test_kfold_groups_disjoint():
    X, y, g = _data()
    for tr, te in group_kfold(X, y, g, 5):
        assert set(g[tr]).isdisjoint(set(g[te]))
