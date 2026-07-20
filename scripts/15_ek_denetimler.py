"""Stage 15 — Ek denetimler (nihai denetim geri bildirimi uzerine):
(1) CatBoost-HistGBM kat-bazli eslestirilmis karsilastirma (eslestirilmis t + Wilcoxon),
(2) Nihai modelin egitim R2'si (asiri-ogrenme farki),
(3) DummyRegressor sifir-bilgi tabani (ayni grup-CV protokolu).
Girdi : data/processed/model_ready_dataset.csv, outputs/v4/best_model.joblib
Cikti : outputs/ek_denetimler/paired_model_comparison.json
        outputs/ek_denetimler/train_test_gap.json
        outputs/ek_denetimler/dummy_baseline.json
Calistirma: python scripts/15_ek_denetimler.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.evaluate import metrics
from perovskite_ml.models.registry import build_models
from perovskite_ml.utils.manifest import write_manifest
from perovskite_ml.validation.splits import group_kfold, holdout_split


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    seed = cfg["seed"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "ek_denetimler"
    outdir.mkdir(parents=True, exist_ok=True)

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values
    groups = M[cfg["group_col"]].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[cfg["target"], cfg["group_col"]])
    folds = group_kfold(X, y, groups, n_splits=5)

    # --- (1) Eslestirilmis kat-bazli karsilastirma: CatBoost vs HistGBM ---
    modeller = build_models(seed)
    kat_r2 = {}
    for ad in ["CatBoost", "HistGBM"]:
        r2s = []
        for tri, vai in folds:
            m = clone(modeller[ad])
            m.fit(X.iloc[tri], y[tri])
            r2s.append(metrics(y[vai], m.predict(X.iloc[vai]))["R2"])
        kat_r2[ad] = r2s
        print(f"[stage15] {ad} kat R2'leri: {[round(v, 4) for v in r2s]}")
    farklar = np.array(kat_r2["CatBoost"]) - np.array(kat_r2["HistGBM"])
    t_ist, t_p = stats.ttest_rel(kat_r2["CatBoost"], kat_r2["HistGBM"])
    # Not: n=5'te Wilcoxon'un alabilecegi en kucuk iki-yonlu p 0,0625'tir (guc dusuk).
    w_ist, w_p = stats.wilcoxon(kat_r2["CatBoost"], kat_r2["HistGBM"])
    paired = {
        "folds_catboost_R2": kat_r2["CatBoost"],
        "folds_histgbm_R2": kat_r2["HistGBM"],
        "mean_diff_R2": float(np.mean(farklar)),
        "std_diff_R2": float(np.std(farklar, ddof=1)),
        "paired_t_stat": float(t_ist), "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_ist), "wilcoxon_p": float(w_p),
        "not": ("Kat skorlari ayni GroupKFold bolmelerinden geldigi icin eslestirilmistir; "
                "5 katla gucu dusuktur ve kat skorlari tam bagimsiz degildir — "
                "destekleyici kanit olarak okunmalidir."),
    }
    json.dump(paired, open(outdir / "paired_model_comparison.json", "w",
                           encoding="utf-8"), indent=2, ensure_ascii=False)

    # --- (2) Egitim-test farki (asiri-ogrenme) ---
    tr, te = holdout_split(X, y, groups, test_size=0.2, seed=seed)
    bundle = joblib.load(ROOT / cfg["paths"]["outputs_dir"] / "v4" / "best_model.joblib")
    model = bundle["model"]
    egitim = metrics(y[tr], model.predict(X.iloc[tr]))
    holdout = metrics(y[te], model.predict(X.iloc[te]))
    gap = {"model": bundle.get("best_name", "CatBoost"),
           "train": egitim, "holdout": holdout,
           "R2_gap_train_minus_holdout": float(egitim["R2"] - holdout["R2"])}
    json.dump(gap, open(outdir / "train_test_gap.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"[stage15] egitim R2={egitim['R2']:.3f}  holdout R2={holdout['R2']:.3f}")

    # --- (3) Sifir-bilgi tabani: DummyRegressor (egitim ortalamasi) ---
    d_mae, d_rmse, d_r2 = [], [], []
    for tri, vai in folds:
        d = DummyRegressor(strategy="mean")
        d.fit(X.iloc[tri], y[tri])
        mt = metrics(y[vai], d.predict(X.iloc[vai]))
        d_mae.append(mt["MAE"]); d_rmse.append(mt["RMSE"]); d_r2.append(mt["R2"])
    dummy = {"strategy": "mean",
             "MAE_mean": float(np.mean(d_mae)), "MAE_std": float(np.std(d_mae)),
             "RMSE_mean": float(np.mean(d_rmse)), "RMSE_std": float(np.std(d_rmse)),
             "R2_mean": float(np.mean(d_r2)), "R2_std": float(np.std(d_r2))}
    json.dump(dummy, open(outdir / "dummy_baseline.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"[stage15] Dummy: R2={dummy['R2_mean']:.4f}  MAE={dummy['MAE_mean']:.2f}")

    write_manifest(outdir, "15_ek_denetimler", cfg,
                   metrics={"paired_t_p": paired["paired_t_p"],
                            "mean_diff_R2": paired["mean_diff_R2"],
                            "train_R2": egitim["R2"], "holdout_R2": holdout["R2"],
                            "dummy_R2_mean": dummy["R2_mean"],
                            "dummy_MAE_mean": dummy["MAE_mean"]},
                   outputs=[str(outdir / "paired_model_comparison.json"),
                            str(outdir / "train_test_gap.json"),
                            str(outdir / "dummy_baseline.json")])
    print("[stage15] TAMAM ->", outdir)


if __name__ == "__main__":
    main()
