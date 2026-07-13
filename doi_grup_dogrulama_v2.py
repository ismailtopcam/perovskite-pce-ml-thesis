"""DOI grup dogrulama — commit'li repo verisi uzerinde.
Kontroller:
  A) model_ready_dataset.csv icinde DOI'siz kayit var mi? (tezin 'temizlikte cikarildi' iddiasi)
  B) Ham DOI stringleri normalize edilmemis mi? (varyant carpismalari)
  C) KESIN TEST: Carpisan varyantlar holdout'ta veya GroupKFold fold'larinda AYRILIYOR mu?
     (Pipeline'in birebir ayni split mantigi: fillna('unknown_doi').astype(str),
      GroupShuffleSplit(test_size=0.2, seed=42), GroupKFold(5))
"""
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from collections import defaultdict

M = pd.read_csv("data/processed/model_ready_dataset.csv", low_memory=False)
GCOL = "Ref_DOI_number"
print(f"model_ready satir sayisi: {len(M)}")

# ---------- A) DOI'siz kayitlar ----------
raw_series = M[GCOL]
n_missing = raw_series.isna().sum()
# CSV'de bos olmayan ama bosluk/placeholder olanlar da olabilir
as_str = raw_series.astype(str)
n_ws_only = ((~raw_series.isna()) & (as_str.str.strip() == "")).sum()
print(f"\n[A] DOI'siz (NaN) kayit: {n_missing}")
print(f"[A] Bosluk-string DOI  : {n_ws_only}")
print(f"[A] Benzersiz DOI (NaN haric, ham): {raw_series.nunique()}")

# Pipeline'in yaptigi gibi grup vektoru:
groups = raw_series.fillna("unknown_doi").astype(str).values
uniq_raw = pd.unique(groups)
print(f"[A] Pipeline grup sayisi (unknown_doi dahil): {len(uniq_raw)}")
n_unknown = (groups == "unknown_doi").sum()
print(f"[A] 'unknown_doi' grubunun boyutu: {n_unknown} kayit")

# ---------- B) Normalizasyon ----------
def normalize_doi(s: str) -> str:
    s = s.strip().lstrip("\ufeff").strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    s = s.rstrip("/.")
    return s

norm_map = defaultdict(set)
for g in uniq_raw:
    if g == "unknown_doi":
        continue
    norm_map[normalize_doi(g)].add(g)

collisions = {k: v for k, v in norm_map.items() if len(v) > 1}
print(f"\n[B] Benzersiz ham DOI (unknown haric)      : {len(uniq_raw) - (1 if 'unknown_doi' in uniq_raw else 0)}")
print(f"[B] Normalize sonrasi benzersiz DOI        : {len(norm_map)}")
print(f"[B] Carpisan (>=2 varyantli) normalize DOI : {len(collisions)}")

n_affected_records = 0
if collisions:
    print("\n[B] Carpisma ornekleri (ilk 15):")
    for i, (k, variants) in enumerate(sorted(collisions.items())):
        counts = {v: int((groups == v).sum()) for v in variants}
        n_affected_records += sum(counts.values())
        if i < 15:
            print(f"    norm='{k}'")
            for v, c in counts.items():
                print(f"        ham={v!r}  ({c} kayit)")
print(f"[B] Carpismalardan etkilenen toplam kayit  : {n_affected_records}")

# Hangi tur farklar? (case / whitespace / prefix)
kinds = defaultdict(int)
for k, variants in collisions.items():
    vs = list(variants)
    stripped = {v.strip() for v in vs}
    lowered = {v.lower() for v in vs}
    if len(stripped) < len(vs) or any(v != v.strip() for v in vs):
        kinds["bosluk (leading/trailing whitespace)"] += 1
    if len(lowered) < len(vs):
        kinds["buyuk/kucuk harf"] += 1
    if any(re.match(r"^(https?://)?(dx\.)?doi\.org/", v.lower()) for v in vs):
        kinds["url oneki (doi.org/...)"] += 1
    if any(re.match(r"^doi:", v.lower().strip()) for v in vs):
        kinds["'doi:' oneki"] += 1
if collisions:
    print(f"[B] Fark turleri: {dict(kinds)}")

# ---------- C) KESIN TEST: fold ayrismasi ----------
y = M["JV_default_PCE"].values
X = M.drop(columns=["JV_default_PCE", GCOL])

# Holdout — pipeline ile birebir ayni
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(X, y, groups))
side = np.empty(len(M), dtype=object)
side[tr] = "train"; side[te] = "test"

print(f"\n[C] Holdout: train={len(tr)}  test={len(te)}  (test orani={len(te)/len(M):.3f})")
print(f"[C] 'unknown_doi' grubu holdout'ta: {set(side[groups=='unknown_doi']) if n_unknown else 'yok'}")

split_holdout = []
for k, variants in collisions.items():
    sides = set()
    for v in variants:
        sides |= set(side[groups == v])
    if len(sides) > 1:
        split_holdout.append((k, sorted(variants), {v: sorted(set(side[groups == v])) for v in variants}))

print(f"[C] HOLDOUT'ta train/test'e AYRILAN carpisma sayisi: {len(split_holdout)}")
for k, variants, detail in split_holdout[:20]:
    print(f"    norm='{k}' -> {detail}")

# GroupKFold(5) — pipeline ile birebir ayni
gkf = GroupKFold(n_splits=5)
fold_of = np.full(len(M), -1)
for f, (tri, vai) in enumerate(gkf.split(X, y, groups)):
    fold_of[vai] = f

split_cv = []
for k, variants in collisions.items():
    folds = set()
    for v in variants:
        folds |= set(fold_of[groups == v].tolist())
    if len(folds) > 1:
        split_cv.append((k, sorted(variants), {v: sorted(set(fold_of[groups == v].tolist())) for v in variants}))

print(f"\n[C] GroupKFold(5) icinde FARKLI fold'lara dusen carpisma sayisi: {len(split_cv)}")
for k, variants, detail in split_cv[:20]:
    print(f"    norm='{k}' -> {detail}")

# Etkilenen kayit sayilari
n_rec_split_holdout = sum(int((np.isin(groups, v)).sum()) for _, v, _ in split_holdout)
n_rec_split_cv = sum(int((np.isin(groups, v)).sum()) for _, v, _ in split_cv)
print(f"\n[C] Holdout sinirini asan yayinlarin toplam kayit sayisi : {n_rec_split_holdout} / {len(M)} ({100*n_rec_split_holdout/len(M):.3f}%)")
print(f"[C] CV fold sinirini asan yayinlarin toplam kayit sayisi : {n_rec_split_cv} / {len(M)} ({100*n_rec_split_cv/len(M):.3f}%)")

# ---------- Ham veri capraz kontrolu ----------
print("\n[D] Ham veri kontrolu (Perovskite_database_content_all_data.csv):")
raw = pd.read_csv("data/raw/Perovskite_database_content_all_data.csv",
                  encoding="utf-8-sig", usecols=["Ref_DOI_number"], low_memory=False)
print(f"    Ham satir: {len(raw)}, DOI'siz (NaN): {raw['Ref_DOI_number'].isna().sum()}")

# ---------- Ozet JSON (tez 5.7 / README'deki degerler commit'li dosyadan izlenebilsin) ----------
import json
from pathlib import Path

summary = {
    "n_doi_missing": int(n_missing),
    "n_unknown_doi_group": int(n_unknown),
    "n_groups_pipeline": int(len(uniq_raw)),
    "n_collisions_normalized": int(len(collisions)),
    "n_records_affected": int(n_affected_records),
    "n_collisions_split_holdout": int(len(split_holdout)),
    "n_collisions_split_cv": int(len(split_cv)),
    "n_records_split_holdout": int(n_rec_split_holdout),
    "n_records_split_cv": int(n_rec_split_cv),
    "pct_records_split_cv": round(100 * n_rec_split_cv / len(M), 3),
}
outp = Path("outputs/robustness/doi_grup_dogrulama.json")
outp.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\nOzet JSON: {outp}")
