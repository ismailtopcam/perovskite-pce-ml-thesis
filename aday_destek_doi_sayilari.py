# -*- coding: utf-8 -*-
"""Aday destegi DOI cesitliligi denetimi — (mimari, ETL, HTL) uclulerinin
egitim bolmesindeki desteginin yalniz KAYIT sayisiyla degil, BENZERSIZ DOI
sayisiyla da raporlanmasi (tez Bolum 4.5). Bir yayin icindeki cok sayida
kardes cihaz bagimsiz kanit sayilamayacagi icin, her uclu icin ayrica en
buyuk tek yayinin destek payi da verilir.

Bolme, pipeline holdout ayrimiyla birebir aynidir (GroupShuffleSplit, seed 42;
bkz. aday_uyumluluk_filtresi.py).

Girdi : data/processed/model_ready_dataset.csv
Cikti : outputs/robustness/aday_destek_doi_sayilari.json
        outputs/candidates_full/triple_doi_support.csv
Calistirma (repo kokunden): python aday_destek_doi_sayilari.py
"""
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ARCH = ["arch_nip", "arch_pin"]
ETL = ["ETL_TiO2_c___TiO2_mp", "ETL_SnO2_np", "ETL_PCBM_60"]
HTL = ["HTL_Spiro_MeOTAD", "HTL_PTAA", "HTL_PEDOT_PSS"]
SEED, TEST_SIZE = 42, 0.2  # config.yaml ile ayni (pipeline holdout ayrimi)

M = pd.read_csv("data/processed/model_ready_dataset.csv", low_memory=False)
groups = M["Ref_DOI_number"].fillna("unknown_doi").astype(str)
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
tr_idx, _ = next(gss.split(M, M["JV_default_PCE"], groups.values))
Mt = M.iloc[tr_idx].copy()
Mt["_doi"] = groups.iloc[tr_idx].values
print(f"Egitim bolmesi: {len(Mt)} kayit, {Mt['_doi'].nunique()} benzersiz DOI")

rows = []
for a in ARCH:
    for e in ETL:
        for h in HTL:
            sub = Mt[(Mt[a] == 1) & (Mt[e] == 1) & (Mt[h] == 1)]
            n_kayit = int(len(sub))
            n_doi = int(sub["_doi"].nunique())
            if n_kayit:
                top = sub["_doi"].value_counts()
                max_pay = round(float(top.iloc[0]) / n_kayit, 3)
            else:
                max_pay = None
            rows.append({"arch": a.replace("arch_", ""),
                         "ETL": e.replace("ETL_", ""),
                         "HTL": h.replace("HTL_", ""),
                         "egitim_kayit_sayisi": n_kayit,
                         "benzersiz_doi_sayisi": n_doi,
                         "en_buyuk_tek_yayin_payi": max_pay})
df = pd.DataFrame(rows).sort_values("egitim_kayit_sayisi", ascending=False).reset_index(drop=True)
df.to_csv("outputs/candidates_full/triple_doi_support.csv", index=False)
print(df.to_string(index=False))

json.dump({"aciklama": "Uclu (mimari, ETL, HTL) basina egitim bolmesi destegi: kayit sayisi, "
                        "benzersiz DOI sayisi ve en buyuk tek yayinin payi",
           "egitim_kayit": int(len(Mt)), "egitim_doi": int(Mt["_doi"].nunique()),
           "ucluler": rows},
          open("outputs/robustness/aday_destek_doi_sayilari.json", "w"), indent=2)
print("\nCikti: outputs/robustness/aday_destek_doi_sayilari.json + outputs/candidates_full/triple_doi_support.csv")
