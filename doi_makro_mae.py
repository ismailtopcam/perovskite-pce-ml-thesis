# -*- coding: utf-8 -*-
"""DOI-makro hata denetimi — kayit-agirlikli holdout MAE'sinin yani sira
yayin (DOI) duzeyinde makro MAE raporlar (tez Bolum 4.7). Kayit-agirlikli
ortalama buyuk yayinlara daha fazla agirlik verir; DOI-makro ortalama her
yayina esit agirlik vererek bu duyarliligi sinar.

Girdi : outputs/v4/best_model_holdout_predictions.csv  (DOI, y_true, y_pred)
Cikti : outputs/robustness/doi_makro_mae.json
Calistirma (repo kokunden): python doi_makro_mae.py
"""
import json
import numpy as np
import pandas as pd

df = pd.read_csv("outputs/v4/best_model_holdout_predictions.csv")
df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
kayit_mae = float(df["abs_err"].mean())
per_doi = df.groupby("DOI")["abs_err"].mean()
makro_mae = float(per_doi.mean())
medyan_doi_mae = float(per_doi.median())
out = {"n_kayit": int(len(df)), "n_doi": int(per_doi.size),
       "kayit_agirlikli_MAE": round(kayit_mae, 3),
       "doi_makro_MAE": round(makro_mae, 3),
       "doi_MAE_medyani": round(medyan_doi_mae, 3)}
json.dump(out, open("outputs/robustness/doi_makro_mae.json", "w"), indent=2)
print(json.dumps(out, indent=2))
