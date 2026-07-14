# -*- coding: utf-8 -*-
"""Aday uyumluluk filtresi — (mimari, ETL, HTL) uclulerinin YALNIZ EGITIM
bolmesindeki birlikte-gorulme sikligina dayali veri-gudumlu filtre.

Onemli: birlikte-gorulme sayilari, pipeline'in DOI-grup holdout ayrimindaki
(GroupShuffleSplit, seed 42) EGITIM kaydi uzerinden hesaplanir; holdout
kayitlari sayima dahil edilmez (tez Bolum 4.5). Onceki surum tum temiz veri
kumesini kullaniyordu (546 aday); egitim-bolmesi duzeltmesiyle destekli aday
sayisi 504'tur.

Girdi : data/processed/model_ready_dataset.csv, outputs/candidates_full/candidate_predictions.csv
Cikti : outputs/candidates_full/triple_cooccurrence.csv
        outputs/candidates_full/candidates_compat_N5.csv
        outputs/candidates_full/top30_compat_N5.csv
Calistirma (repo kokunden): python aday_uyumluluk_filtresi.py
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ARCH = ["arch_nip", "arch_pin"]
ETL  = ["ETL_TiO2_c___TiO2_mp", "ETL_SnO2_np", "ETL_PCBM_60"]
HTL  = ["HTL_Spiro_MeOTAD", "HTL_PTAA", "HTL_PEDOT_PSS"]
SEED, TEST_SIZE = 42, 0.2  # config.yaml ile ayni (pipeline holdout ayrimi)

M = pd.read_csv("data/processed/model_ready_dataset.csv", low_memory=False)
groups = M["Ref_DOI_number"].fillna("unknown_doi").astype(str).values
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
tr_idx, te_idx = next(gss.split(M, M["JV_default_PCE"], groups))
Mt = M.iloc[tr_idx]
C = pd.read_csv("outputs/candidates_full/candidate_predictions.csv")
print(f"Egitim: {len(Mt)} kayit (holdout {len(te_idx)} haric) | Aday: {len(C)}")

# --- [A] Egitim bolmesinde uclu birlikte-gorulme sayimlari ---
rows = []
for a in ARCH:
    for e in ETL:
        for h in HTL:
            n = int(((Mt[a] == 1) & (Mt[e] == 1) & (Mt[h] == 1)).sum())
            rows.append({"arch": a.replace("arch_", ""),
                         "ETL": e.replace("ETL_", ""),
                         "HTL": h.replace("HTL_", ""),
                         "egitim_kayit_sayisi": n})
co = pd.DataFrame(rows).sort_values("egitim_kayit_sayisi", ascending=False).reset_index(drop=True)
co.to_csv("outputs/candidates_full/triple_cooccurrence.csv", index=False)
print("\n[A] 18 aday uclusunun egitim bolmesindeki birlikte-gorulme sayilari:")
print(co.to_string(index=False))

# --- [B] Filtre: N esikleri ---
key = ["arch", "ETL", "HTL"]
C2 = C.merge(co, on=key, how="left")
for N in (1, 5, 10):
    n_c = int((C2["egitim_kayit_sayisi"] >= N).sum())
    n_t = int((co["egitim_kayit_sayisi"] >= N).sum())
    print(f"[B] N>={N:2}: gecerli uclu {n_t}/18, gecerli aday {n_c}/{len(C)}")

Np = 5  # birincil esik
F = C2[C2["egitim_kayit_sayisi"] >= Np].copy()
F = F.sort_values("Predicted_PCE", ascending=False).reset_index(drop=True)
F.to_csv("outputs/candidates_full/candidates_compat_N5.csv", index=False)
top30_f = F.drop_duplicates(subset=["A", "B", "X", "HTL"]).head(30)
top30_f.to_csv("outputs/candidates_full/top30_compat_N5.csv", index=False)

# --- [C] Karsilastirma ozeti ---
div = pd.read_csv("outputs/candidates_full/top30_diverse.csv")
kk = ["A", "B", "X", "arch", "ETL", "HTL"]
ortak = pd.merge(top30_f[kk], div[kk], on=kk).shape[0]
print(f"\n[C] Filtreli ilk-30 mimari dagilimi: {top30_f['arch'].value_counts().to_dict()}")
print(f"[C] Filtreli tavan: {F['Predicted_PCE'].max():.2f} | cesitlendirilmis ilk-30 ile ortak: {ortak}")
