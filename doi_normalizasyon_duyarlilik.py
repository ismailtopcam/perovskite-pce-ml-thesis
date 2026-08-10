"""Duyarlilik analizi: DOI normalizasyonunun basari metriklerine etkisi.

v0.4.0 (1-C) itibariyla YON TERSINE DONDU: ana hat kanonik (normalize) DOI
gruplariyla calisir; bu betik HAM (v0.3.x) gruplamayi ham CSV'den yeniden
kurarak duyarlilik katmani olarak raporlar.
1) Normalize gruplar (ana hat, model_ready'deki anahtar)
2) Ham gruplar (ham CSV'den, normalize_doi=False ile yeniden kurulur)
Model: CatBoost, registry ile birebir ayni hiperparametreler.
Cikti: outputs/robustness/doi_normalizasyon_duyarlilik.json (tez Bolum 5.7)
"""
import json, re, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent

M = pd.read_csv(ROOT / "data/processed/model_ready_dataset.csv", low_memory=False)
y = M["JV_default_PCE"].values
X = M.drop(columns=["JV_default_PCE", "Ref_DOI_number"])

def normalize_doi(s: str) -> str:
    s = s.strip().lstrip("\ufeff").strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.rstrip("/.")

# Ana hat (v0.4.0): model_ready'deki anahtar zaten kanonik
g_norm = M["Ref_DOI_number"].fillna("unknown_doi").astype(str).values

# HAM gruplar: ham CSV'den, normalizasyon KAPALI cfg ile yeniden kur
import sys as _sys, copy as _copy
_sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config as _load_config
from perovskite_ml.data.load_data import load_raw as _load_raw
from perovskite_ml.data.clean_data import clean_rows as _clean_rows
_cfg = _load_config(str(ROOT / "config.yaml"))
_cfg_raw = _copy.deepcopy(_cfg); _cfg_raw.setdefault("grouping", {})["normalize_doi"] = False
_cleaned_raw, _ = _clean_rows(_load_raw(_cfg_raw), _cfg_raw)
g_raw = _cleaned_raw["Ref_DOI_number"].fillna("unknown_doi").astype(str).values
assert len(g_raw) == len(M), "Ham yeniden-kurulum satir sayisi model_ready ile uyusmuyor"
_chk = np.array([g if g == "unknown_doi" else normalize_doi(g) for g in g_raw])
assert (_chk == g_norm).all(), "Hizalama kontrolu: normalize(ham) != ana hat anahtari"

def make_model():
    return CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                             random_seed=42, verbose=0, allow_writing_files=False)

def run(groups, label):
    t0 = time.time()
    # holdout
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    m = make_model(); m.fit(X.iloc[tr], y[tr])
    p = m.predict(X.iloc[te])
    hold = dict(MAE=mean_absolute_error(y[te], p),
                RMSE=float(np.sqrt(mean_squared_error(y[te], p))),
                R2=r2_score(y[te], p),
                n_test=len(te))
    # GroupKFold(5)
    r2s, maes, rmses = [], [], []
    for tri, vai in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_model(); m.fit(X.iloc[tri], y[tri])
        p = m.predict(X.iloc[vai])
        r2s.append(r2_score(y[vai], p))
        maes.append(mean_absolute_error(y[vai], p))
        rmses.append(float(np.sqrt(mean_squared_error(y[vai], p))))
    print(f"\n[{label}]  ({time.time()-t0:.0f} sn)")
    print(f"  Holdout : R2={hold['R2']:.4f}  MAE={hold['MAE']:.4f}  RMSE={hold['RMSE']:.4f}  (n_test={hold['n_test']})")
    print(f"  GKF(5)  : R2={np.mean(r2s):.4f}±{np.std(r2s):.4f}  MAE={np.mean(maes):.4f}  RMSE={np.mean(rmses):.4f}")
    print(f"  R2 folds: {[round(x,3) for x in r2s]}")
    print(f"  Grup sayisi: {len(pd.unique(groups))}")
    return {
        "label": label,
        "holdout": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in hold.items()},
        "gkf5": {"R2_mean": round(float(np.mean(r2s)), 4), "R2_std": round(float(np.std(r2s)), 4),
                 "MAE_mean": round(float(np.mean(maes)), 4), "RMSE_mean": round(float(np.mean(rmses)), 4),
                 "R2_folds": [round(float(x), 4) for x in r2s]},
        "n_groups": int(len(pd.unique(groups))),
    }

print("Referans (v0.3.5 arsiv): HAM Holdout R2=0.3924 | HAM GKF R2=0.4129±0.0065")
res_norm = run(g_norm, "NORMALIZE gruplar — ana hat (v0.4.0)")
res_raw = run(g_raw, "HAM gruplar — duyarlilik (v0.3.x davranisi)")

outdir = ROOT / "outputs" / "robustness"
outdir.mkdir(parents=True, exist_ok=True)
payload = {"primary": "normalized_groups", "raw_groups": res_raw, "normalized_groups": res_norm,
           "delta_gkf_r2": round(res_raw["gkf5"]["R2_mean"] - res_norm["gkf5"]["R2_mean"], 4),
           "delta_holdout_r2": round(res_raw["holdout"]["R2"] - res_norm["holdout"]["R2"], 4)}
with open(outdir / "doi_normalizasyon_duyarlilik.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
print(f"\nKaydedildi: {outdir / 'doi_normalizasyon_duyarlilik.json'}")
