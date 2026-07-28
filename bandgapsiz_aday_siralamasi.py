# -*- coding: utf-8 -*-
"""Band-gap'siz yeniden siralama denetimi — adaylar uretilirken band gap tum
kompozisyonlarda ayni medyana (1,60 eV) sabitlenir; oysa farkli bilesimlerin
gercek band gap'leri farklidir. Bu betik, band gap (ve eksiklik bayragi)
icermeyen bir modeli arsivlenmis modelle AYNI egitim bolmesinde egitir,
756 adayi yeniden puanlar ve siralama kararliligini olcer (tez Bolum 5.4).

Girdi : data/processed/model_ready_dataset.csv, outputs/v4/best_model.joblib
Cikti : outputs/robustness/bandgapsiz_aday_siralamasi.json
Calistirma (repo kokunden): python bandgapsiz_aday_siralamasi.py
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
from perovskite_ml.validation.splits import holdout_split

cfg = load_config(str(ROOT / "config.yaml"))
M = pd.read_csv(ROOT / "data/processed/model_ready_dataset.csv", low_memory=False)
y = M[cfg["target"]].values
groups = M[cfg["group_col"]].fillna("unknown_doi").astype(str).values
X = M.drop(columns=[cfg["target"], cfg["group_col"]])

bundle = joblib.load(ROOT / "outputs/v4/best_model.joblib")
eval_model, features, best = bundle["model"], bundle["features"], bundle["best_name"]

BG_COLS = [c for c in features if c in ("band_gap", "band_gap_missing")]
feats_nobg = [f for f in features if f not in BG_COLS]
print(f"Cikartilan kolonlar: {BG_COLS} | kalan ozellik: {len(feats_nobg)}")

# arsiv modeliyle AYNI egitim bolmesi (DOI-grup 80/20, seed config'ten)
tr, te = holdout_split(X, y, groups, test_size=0.2, seed=cfg["seed"])
nobg = clone(build_models(cfg["seed"])[best])
nobg.fit(X.iloc[tr][feats_nobg], y[tr])

bg = float(M["band_gap"].median()) if "band_gap" in M else 1.6
at = float(M["anneal_temp"].median()) if "anneal_temp" in M else 100.0
ati = float(M["anneal_time"].median()) if "anneal_time" in M else 30.0
recipes, enc_full = enumerate_candidates(features, bg, at, ati)
_, enc_nobg = enumerate_candidates(feats_nobg, bg, at, ati)

old_preds = eval_model.predict(enc_full)
nobg_preds = nobg.predict(enc_nobg)

rho = float(spearmanr(old_preds, nobg_preds).statistic)
top30_old = set(np.argsort(-old_preds)[:30])
top30_new = set(np.argsort(-nobg_preds)[:30])

def diverse30(preds):
    r = recipes.copy(); r["p"] = preds
    r = r.sort_values("p", ascending=False)
    return set(r.drop_duplicates(subset=["A", "B", "X", "HTL"]).head(30).index)

d30_old, d30_new = diverse30(old_preds), diverse30(nobg_preds)

out = {
    "model_ailesi": best, "cikartilan_kolonlar": BG_COLS, "n_aday": int(len(recipes)),
    "spearman_rho": round(rho, 4),
    "ilk30_ortusme": len(top30_old & top30_new),
    "cesitlendirilmis_ilk30_ortusme": len(d30_old & d30_new),
    "tavan_bandgapli": round(float(old_preds.max()), 2),
    "tavan_bandgapsiz": round(float(nobg_preds.max()), 2),
    "ortalama_mutlak_puan_farki": round(float(np.mean(np.abs(nobg_preds - old_preds))), 3),
}
outdir = ROOT / "outputs/robustness"; outdir.mkdir(parents=True, exist_ok=True)
json.dump(out, open(outdir / "bandgapsiz_aday_siralamasi.json", "w"), indent=2)
print(json.dumps(out, indent=2))
