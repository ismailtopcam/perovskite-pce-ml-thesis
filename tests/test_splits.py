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


def test_nan_doi_sentinel_single_group():
    """Betiklerdeki sozlesme: DOI'siz (NaN) kayitlar fillna('unknown_doi') ile TEK grup olur
    ve bu grup hicbir bolmede egitim/test arasinda bolunmez (bkz. tez, Bolum 3.8 / P313)."""
    X, y, g = _data()
    raw = pd.Series(g).astype(object)
    raw.iloc[::10] = np.nan                      # her 10. kayit DOI'siz
    groups = raw.fillna("unknown_doi").astype(str).values
    assert (groups == "unknown_doi").sum() == 20  # tek sentinel grup, tum NaN'lar toplandi

    tr, te = holdout_split(X, y, groups, 0.2, 42)
    sides = {"tr" if i in set(tr) else "te" for i in np.where(groups == "unknown_doi")[0]}
    assert len(sides) == 1                        # sentinel grup tek tarafta

    for tr_i, te_i in group_kfold(X, y, groups, 5):
        assert set(groups[tr_i]).isdisjoint(set(groups[te_i]))
