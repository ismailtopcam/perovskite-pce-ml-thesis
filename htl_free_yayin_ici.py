# -*- coding: utf-8 -*-
"""HTL-free yayin-ici (paired) karsilastirma — tez Bolum 5.3.

Calismalar-arasi HTL-free medyan farkinin (8,0 vs 12,77) bir veri-kaynagi
artefakti olup olmadigini sinar: ayni yayin (DOI) icinde hem HTL-free hem
HTL'li hucre bulunan "karma" yayinlarda, yayin-ici medyan PCE farkini olcer.
Fark ayni yonde ve buyukse, oruntu yayin-duzeyi karisiklikla aciklanamaz.

Repo kokunden: python htl_free_yayin_ici.py   (~5 sn, ham veri gerektirmez;
model_ready_dataset.csv yeterlidir)
Cikti: outputs/robustness/htl_free_yayin_ici.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

M = pd.read_csv("data/processed/model_ready_dataset.csv", low_memory=False)

TARGET, GROUP, HTL_FREE = "JV_default_PCE", "Ref_DOI_number", "HTL_none"
for col in (TARGET, GROUP, HTL_FREE):
    if col not in M.columns:
        raise SystemExit(f"Beklenen kolon yok: {col}")

pce = M[TARGET].astype(float)
doi = M[GROUP].fillna("unknown_doi").astype(str)
free = M[HTL_FREE].astype(int) == 1

# Tutarlilik kontrolleri (tez Bolum 4.1/5.3'teki degerlerle)
n_free, n_free_doi = int(free.sum()), int(doi[free].nunique())
print(f"HTL-free kayit: {n_free} (beklenen 2494) | HTL-free iceren yayin: {n_free_doi} (beklenen 647)")
print(f"HTL-free medyan PCE: {pce[free].median():.2f} (beklenen 8,0) | genel medyan: {pce.median():.2f} (beklenen 12,77)")

# Karma yayinlar: ayni DOI icinde hem HTL-free hem HTL'li kayit
df = pd.DataFrame({"doi": doi, "pce": pce, "free": free})
per_doi = df.groupby("doi")["free"].agg(["any", "all"])
mixed = per_doi[per_doi["any"] & ~per_doi["all"]].index
sub = df[df["doi"].isin(mixed)]

# Yayin-ici fark: medyan(HTL-free) - medyan(HTL'li)
diffs = sub.groupby("doi").apply(
    lambda g: g.loc[g["free"], "pce"].median() - g.loc[~g["free"], "pce"].median(),
    include_groups=False,
)
w = stats.wilcoxon(diffs)

res = {
    "n_htl_free_kayit": n_free,
    "n_htl_free_yayin": n_free_doi,
    "htl_free_medyan_pce": round(float(pce[free].median()), 2),
    "genel_medyan_pce": round(float(pce.median()), 2),
    "n_karma_yayin": int(len(mixed)),
    "yayin_ici_medyan_fark": round(float(diffs.median()), 2),
    "ceyrekler": [round(float(diffs.quantile(0.25)), 2), round(float(diffs.quantile(0.75)), 2)],
    "negatif_fark_yayin_orani": round(float((diffs < 0).mean()), 3),
    "wilcoxon_p": float(w.pvalue),
}
print(f"\nKarma yayin: {res['n_karma_yayin']} (beklenen 222)")
print(f"Yayin-ici medyan fark (HTL-free − HTL'li): {res['yayin_ici_medyan_fark']} PCE puani (beklenen −3,74)")
print(f"Ceyrekler: {res['ceyrekler']} (beklenen [−6,90, −1,50])")
print(f"Negatif fark orani: %{100*res['negatif_fark_yayin_orani']:.1f} (beklenen %89,2) | Wilcoxon p = {w.pvalue:.2e}")

out = Path("outputs/robustness"); out.mkdir(parents=True, exist_ok=True)
with open(out / "htl_free_yayin_ici.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(f"\nYazildi: {out/'htl_free_yayin_ici.json'}")
