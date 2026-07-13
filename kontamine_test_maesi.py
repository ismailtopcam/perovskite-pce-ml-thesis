"""Kontamine test kayitlari analizi — holdout'ta sinir asan yayinlarin
test tarafindaki kayit sayisi ve bu kayitlarda model hatasi.
Amaç: sizintinin skoru sisirip sisirmedigini kayit duzeyinde olcmek.
Repo kokunden: python kontamine_test_maesi.py  (~15 sn, CatBoost gerekli)
"""
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from catboost import CatBoostRegressor
from collections import defaultdict

M = pd.read_csv("data/processed/model_ready_dataset.csv", low_memory=False)
y = M["JV_default_PCE"].values
X = M.drop(columns=["JV_default_PCE", "Ref_DOI_number"])
groups = M["Ref_DOI_number"].fillna("unknown_doi").astype(str).values

def normalize_doi(s: str) -> str:
    s = s.strip().lstrip("\ufeff").strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.rstrip("/.")

# Pipeline ile birebir ayni holdout
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(X, y, groups))
side = np.empty(len(M), dtype=object)
side[tr] = "train"; side[te] = "test"

# Carpisan varyantlar ve holdout sinirini asanlar
nm = defaultdict(set)
for g in pd.unique(groups):
    if g != "unknown_doi":
        nm[normalize_doi(g)].add(g)

contam_test_idx, train_side_n = [], 0
for k, variants in nm.items():
    if len(variants) < 2:
        continue
    sides = {s for v in variants for s in set(side[groups == v])}
    if len(sides) > 1:
        for v in variants:
            n = int((groups == v).sum())
            if set(side[groups == v]) == {"test"}:
                contam_test_idx.extend(np.where(groups == v)[0].tolist())
            else:
                train_side_n += n

contam_test_idx = np.array(sorted(contam_test_idx))
print(f"Holdout sinirini asan yayinlarin TEST tarafindaki kayitlari : {len(contam_test_idx)} "
      f"(test setinin %{100*len(contam_test_idx)/len(te):.2f}'si)")
print(f"Ayni yayinlarin TRAIN tarafindaki kayitlari                : {train_side_n}")

# CatBoost — registry ile birebir ayni
m = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                      random_seed=42, verbose=0, allow_writing_files=False)
m.fit(X.iloc[tr], y[tr])
p = m.predict(X.iloc[te])
ae = np.abs(y[te] - p)

contam_set = set(contam_test_idx.tolist())
mask_contam = np.array([j for j, idx in enumerate(te) if idx in contam_set])
mask_clean  = np.array([j for j, idx in enumerate(te) if idx not in contam_set])

print(f"\nKontamine test kayitlari (n={len(mask_contam)}): MAE = {ae[mask_contam].mean():.3f}")
print(f"Temiz test kayitlari     (n={len(mask_clean)}): MAE = {ae[mask_clean].mean():.3f}")
print("\nYorum: kontamine MAE, temiz MAE'den YUKSEK ise sizinti bu kayitlarda")
print("olculebilir bir skor sisirmesi yaratmamis demektir.")

# Ozet JSON (tez 5.7'deki 35 kayit / MAE 3,36-3,15 degerleri commit'li dosyadan izlenebilsin)
import json
from pathlib import Path

summary = {
    "n_contaminated_test": int(len(mask_contam)),
    "n_clean_test": int(len(mask_clean)),
    "mae_contaminated": round(float(ae[mask_contam].mean()), 3),
    "mae_clean": round(float(ae[mask_clean].mean()), 3),
    "n_train_side_records": int(train_side_n),
}
outp = Path("outputs/robustness/kontamine_test_maesi.json")
outp.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\nOzet JSON: {outp}")
