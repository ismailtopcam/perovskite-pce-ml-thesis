"""DOI-grup-guvenli veri ayrimlari. Ayni yayina ait kayitlar egitim ve teste
birlikte dusmez -> ezberden kaynakli yapay yuksek skor onlenir."""
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

def holdout_split(X, y, groups, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(X, y, groups))

def group_kfold(X, y, groups, n_splits=5):
    return list(GroupKFold(n_splits=n_splits).split(X, y, groups))
