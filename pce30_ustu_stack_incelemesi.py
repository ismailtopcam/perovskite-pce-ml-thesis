# -*- coding: utf-8 -*-
"""PCE>30 kayitlarinin stack incelemesi — cift-emici (tandem) izi var mi?
Girdi : data/processed/cleaned_dataset.csv
Cikti : outputs/robustness/pce30_ustu_stack.csv + konsol ozeti
Calistirma (repo kokunden): python pce30_ustu_stack_incelemesi.py
"""
import pandas as pd

D = pd.read_csv("data/processed/cleaned_dataset.csv", low_memory=False,
                usecols=lambda c: c in ["Ref_DOI_number", "JV_default_PCE", "Cell_architecture"])
pce = pd.to_numeric(D["JV_default_PCE"], errors="coerce")
S = D[pce > 30].copy()
S["JV_default_PCE"] = pce[pce > 30]

# Stack bilgisi temiz sette tasinmadigindan ham veriden DOI+PCE ile eslenir
R = pd.read_csv("data/raw/Perovskite_database_content_all_data.csv", encoding="utf-8-sig",
                low_memory=False,
                usecols=["Ref_DOI_number", "JV_default_PCE", "Cell_stack_sequence",
                         "Perovskite_composition_long_form"])
R["JV_default_PCE"] = pd.to_numeric(R["JV_default_PCE"], errors="coerce")
S = S.merge(R, on=["Ref_DOI_number", "JV_default_PCE"], how="left") \
     .drop_duplicates(subset=["Ref_DOI_number", "JV_default_PCE"]) \
     .sort_values("JV_default_PCE", ascending=False)
S.to_csv("outputs/robustness/pce30_ustu_stack.csv", index=False)

print(f"PCE>30 kayit sayisi: {len(S)}  |  yayin: {S['Ref_DOI_number'].nunique()}")
print(f"PCE degerleri: {sorted(S['JV_default_PCE'].round(1).tolist(), reverse=True)}")
print(f"Mimari dagilimi: {S['Cell_architecture'].value_counts(dropna=False).to_dict()}")
# cift-emici / tandem sozcuk izleri
kw = ["tandem", "perovskite | perovskite", "si |", "| si", "cigs", "silicon"]
for _, r in S.iterrows():
    st = str(r["Cell_stack_sequence"]).lower()
    hits = [k for k in kw if k in st]
    n_pvk = st.count("perovskite")
    print(f"\nDOI={r['Ref_DOI_number']}  PCE={r['JV_default_PCE']:.1f}  arch={r['Cell_architecture']}")
    print(f"  stack: {str(r['Cell_stack_sequence'])[:150]}")
    print(f"  'perovskite' katman sayisi: {n_pvk}  | tandem izi: {hits if hits else 'yok'}")
