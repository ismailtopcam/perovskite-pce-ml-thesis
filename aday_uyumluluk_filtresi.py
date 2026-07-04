# -*- coding: utf-8 -*-
"""Aday uyumluluk filtresi — (mimari, ETL, HTL) uclulerinin egitim verisindeki
birlikte-gorulme sikligina dayali veri-gudumlu filtre.
Girdi : data/processed/model_ready_dataset.csv, outputs/candidates_full/candidate_predictions.csv
Cikti : outputs/candidates_full/triple_cooccurrence.csv
        outputs/candidates_full/candidates_compat_N5.csv
        outputs/candidates_full/top30_compat_N5.csv
Calistirma (repo kokunden): python aday_uyumluluk_filtresi.py
"""
import pandas as pd

ARCH = ["arch_nip", "arch_pin"]
ETL  = ["ETL_TiO2_c___TiO2_mp", "ETL_SnO2_np", "ETL_PCBM_60"]
HTL  = ["HTL_Spiro_MeOTAD", "HTL_PTAA", "HTL_PEDOT_PSS"]

M = pd.read_csv("data/processed/model_ready_dataset.csv", low_memory=False)
C = pd.read_csv("outputs/candidates_full/candidate_predictions.csv")
print(f"Egitim: {len(M)} kayit | Aday: {len(C)}")

# --- [A] Egitimde uclu birlikte-gorulme sayimlari (aday soz dagarciginda) ---
rows = []
for a in ARCH:
    for e in ETL:
        for h in HTL:
            n = int(((M[a] == 1) & (M[e] == 1) & (M[h] == 1)).sum())
            rows.append({"arch": a.replace("arch_", ""),
                         "ETL": e.replace("ETL_", ""),
                         "HTL": h.replace("HTL_", ""),
                         "egitim_kayit_sayisi": n})
co = pd.DataFrame(rows).sort_values("egitim_kayit_sayisi", ascending=False).reset_index(drop=True)
co.to_csv("outputs/candidates_full/triple_cooccurrence.csv", index=False)
print("\n[A] 18 aday uclusunun egitimdeki birlikte-gorulme sayilari:")
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
orig30 = pd.read_csv("outputs/candidates_full/top30_diverse.csv")
def sig(df): return set(map(tuple, df[["A", "B", "X", "arch", "ETL", "HTL"]].values))
ortak = len(sig(orig30) & sig(top30_f))
print(f"\n[C] Birincil esik N>={Np}:")
print(f"    Aday: {len(C)} -> {len(F)}  (elenen {len(C)-len(F)})")
print(f"    Ham top-30 mimari dagilimi     : {orig30['arch'].value_counts().to_dict()}")
print(f"    Filtreli top-30 mimari dagilimi: {top30_f['arch'].value_counts().to_dict()}")
print(f"    Filtreli top-30 ETL/HTL        : {top30_f['ETL'].value_counts().to_dict()} / {top30_f['HTL'].value_counts().to_dict()}")
print(f"    Iki top-30 arasindaki ortak aday: {ortak}/30")
print(f"    Tahmin tavani: ham {C['Predicted_PCE'].max():.2f} -> filtreli {F['Predicted_PCE'].max():.2f}")
print(f"    pin+SnO2_np+Spiro_MeOTAD egitim sayisi: "
      f"{int(co.set_index(key).loc[('pin','SnO2_np','Spiro_MeOTAD'),'egitim_kayit_sayisi'])}")
