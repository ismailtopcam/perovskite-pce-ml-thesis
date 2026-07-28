# -*- coding: utf-8 -*-
"""Uyumluluk esigi duyarliligi — aday_uyumluluk_filtresi.py'deki 'egitimde en
az N kez birlikte-gorulme' esigi N=5 olarak secilmisti; bu sezgisel bir
muhendislik tercihidir. Bu betik N ∈ {1, 3, 5, 10} icin destekli aday sayisini,
tahmin tavanini ve filtreli ilk-30 listesinin N=5 tabaniyla ortusmesini
raporlar (tez Bolum 4.5).

Girdi : data/processed/model_ready_dataset.csv, outputs/candidates_full/candidate_predictions.csv
Cikti : outputs/robustness/aday_esik_duyarliligi.json
Calistirma (repo kokunden): python aday_esik_duyarliligi.py
"""
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ARCH = ["arch_nip", "arch_pin"]
ETL  = ["ETL_TiO2_c___TiO2_mp", "ETL_SnO2_np", "ETL_PCBM_60"]
HTL  = ["HTL_Spiro_MeOTAD", "HTL_PTAA", "HTL_PEDOT_PSS"]
SEED, TEST_SIZE = 42, 0.2  # config.yaml ile ayni (pipeline holdout ayrimi)
ESIKLER = [1, 3, 5, 10]

M = pd.read_csv("data/processed/model_ready_dataset.csv", low_memory=False)
groups = M["Ref_DOI_number"].fillna("unknown_doi").astype(str).values
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
tr_idx, _ = next(gss.split(M, M["JV_default_PCE"], groups))
Mt = M.iloc[tr_idx]

# egitim bolmesinde uclu birlikte-gorulme sayimlari (aday_uyumluluk_filtresi ile ayni)
cooc = {}
for a in ARCH:
    for e in ETL:
        for h in HTL:
            cooc[(a.replace("arch_", ""), e.replace("ETL_", ""), h.replace("HTL_", ""))] = \
                int(((Mt[a] == 1) & (Mt[e] == 1) & (Mt[h] == 1)).sum())

C = pd.read_csv("outputs/candidates_full/candidate_predictions.csv")
C["destek"] = [cooc.get((r.arch, r.ETL, r.HTL), 0) for r in C.itertuples()]

def ilk30(df):
    return set((df["A"] + "|" + df["B"] + "|" + df["X"] + "|" + df["arch"] + "|" +
                df["ETL"] + "|" + df["HTL"]).head(30))

sonuc = {"egitim_kayit": int(len(Mt)), "esikler": {}}
taban30 = None
for n in ESIKLER:
    S = C[C["destek"] >= n].sort_values("Predicted_PCE", ascending=False)
    t30 = ilk30(S)
    if n == 5: taban30 = t30
    sonuc["esikler"][f"N>={n}"] = {
        "destekli_aday": int(len(S)),
        "tahmin_tavani": round(float(S["Predicted_PCE"].max()), 2) if len(S) else None,
        "ilk30_medyan": round(float(S["Predicted_PCE"].head(30).median()), 2) if len(S) else None,
    }
for n in ESIKLER:
    S = C[C["destek"] >= n].sort_values("Predicted_PCE", ascending=False)
    sonuc["esikler"][f"N>={n}"]["ilk30_N5_ortusme"] = len(ilk30(S) & taban30)

json.dump(sonuc, open("outputs/robustness/aday_esik_duyarliligi.json", "w"), indent=2)
print(json.dumps(sonuc, indent=2))
