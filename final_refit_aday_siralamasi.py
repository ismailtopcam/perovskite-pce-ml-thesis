# -*- coding: utf-8 -*-
"""Final-refit denetimi — arsivlenmis aday siralamasi, %80 egitim bolmesinde
egitilmis degerlendirme modeliyle (best_model.joblib) uretilmistir. Bu betik,
secilen model ailesini ayni hiperparametrelerle TUM temiz veri (41.485 kayit)
uzerinde yeniden egitir (final refit), 756 adayi yeniden puanlar ve iki
siralamayi karsilastirir (tez Bolum 5.4). Ana sonuclar ve arsiv degismez;
bu bir sonradan-denetim katmanidir.

Girdi : data/processed/model_ready_dataset.csv, outputs/v4/best_model.joblib,
        outputs/candidates_full/candidate_predictions.csv
Cikti : outputs/robustness/final_refit_aday_siralamasi.json
        outputs/robustness/final_refit_model.joblib (yerel; commit'lenmez)
Calistirma (repo kokunden): python final_refit_aday_siralamasi.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.candidates.candidate_space import enumerate_candidates
from perovskite_ml.models.registry import build_models

cfg = load_config(str(ROOT / "config.yaml"))
M = pd.read_csv(ROOT / "data/processed/model_ready_dataset.csv", low_memory=False)
y = M[cfg["target"]].values
X = M.drop(columns=[cfg["target"], cfg["group_col"]])

bundle = joblib.load(ROOT / "outputs/v4/best_model.joblib")
eval_model, features, best = bundle["model"], bundle["features"], bundle["best_name"]
assert list(X.columns) == features, "Ozellik sirasi bundle ile uyusmuyor"

# --- final refit: ayni aile + ayni hiperparametreler, TUM veri ---
refit = clone(build_models(cfg["seed"])[best])
refit.fit(X, y)

# --- adaylar: 06 ile ayni sabitler (veri kumesi medyanlari) ---
bg = float(M["band_gap"].median()) if "band_gap" in M else 1.6
at = float(M["anneal_temp"].median()) if "anneal_temp" in M else 100.0
ati = float(M["anneal_time"].median()) if "anneal_time" in M else 30.0
recipes, enc = enumerate_candidates(features, bg, at, ati)
old_preds = eval_model.predict(enc)
new_preds = refit.predict(enc)

# arsivlenmis CSV ile tutarlilik (ayni uclu/kompozisyon anahtari uzerinden)
C = pd.read_csv(ROOT / "outputs/candidates_full/candidate_predictions.csv")
key = lambda df: df["A"] + "|" + df["B"] + "|" + df["X"] + "|" + df["arch"] + "|" + df["ETL"] + "|" + df["HTL"]
arch_map = dict(zip(key(C), C["Predicted_PCE"]))
recomputed = pd.Series(old_preds, index=key(recipes))
max_arsiv_fark = float(max(abs(recomputed[k] - v) for k, v in arch_map.items()))

rho = float(spearmanr(old_preds, new_preds).statistic)
top30_old = set(np.argsort(-old_preds)[:30])
top30_new = set(np.argsort(-new_preds)[:30])

def diverse30(preds):
    r = recipes.copy(); r["p"] = preds
    r = r.sort_values("p", ascending=False)
    return set(r.drop_duplicates(subset=["A", "B", "X", "HTL"]).head(30).index)

d30_old, d30_new = diverse30(old_preds), diverse30(new_preds)

out = {
    "model_ailesi": best, "n_egitim_final_refit": int(len(M)), "n_aday": int(len(recipes)),
    "arsiv_csv_ile_max_fark": round(max_arsiv_fark, 6),
    "spearman_rho": round(rho, 4),
    "ilk30_ortusme": len(top30_old & top30_new),
    "cesitlendirilmis_ilk30_ortusme": len(d30_old & d30_new),
    "tavan_eski": round(float(old_preds.max()), 2),
    "tavan_final_refit": round(float(new_preds.max()), 2),
    "ortalama_mutlak_puan_farki": round(float(np.mean(np.abs(new_preds - old_preds))), 3),
}
outdir = ROOT / "outputs/robustness"; outdir.mkdir(parents=True, exist_ok=True)
json.dump(out, open(outdir / "final_refit_aday_siralamasi.json", "w"), indent=2)
joblib.dump({"model": refit, "features": features, "best_name": best,
             "egitim": "tum_temiz_veri_41485"}, outdir / "final_refit_model.joblib")
print(json.dumps(out, indent=2))
