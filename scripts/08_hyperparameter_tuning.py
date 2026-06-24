"""Stage 8 — Grup-guvenli hiperparametre optimizasyonu (default vs tuned).
Girdi : data/processed/model_ready_dataset.csv
Cikti : outputs/tuning/search_results.csv, best_params.json, manifest
Calistirma: python scripts/08_hyperparameter_tuning.py
"""
import sys, json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.tuning import grid_search_grouped, grouped_cv_r2
from perovskite_ml.utils.manifest import write_manifest

def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    proc = ROOT / cfg["paths"]["processed_dir"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "tuning"
    outdir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values
    g = M[cfg["group_col"]].fillna("u").astype(str).values
    X = M.drop(columns=[cfg["target"], cfg["group_col"]])

    # birincil model: HistGBM (her ortamda kurulu, CatBoost'a cok yakin)
    def hgb(**p): return HistGradientBoostingRegressor(random_state=seed, **p)
    default_mean, default_std = grouped_cv_r2(hgb(), X, y, g, 5)

    grid = {
        "learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [31, 63],
        "l2_regularization": [0.0, 1.0],
        "max_iter": [200, 400],
    }
    best, rows = grid_search_grouped(hgb, grid, X, y, g, n_splits=3)  # arama 3-kat (hiz)
    # secilen ayarin durust 5-kat CV skoru
    best_params = {k: best[k] for k in grid}
    tuned_mean, tuned_std = grouped_cv_r2(hgb(**best_params), X, y, g, 5)

    pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).to_csv(outdir / "search_results.csv", index=False)
    json.dump({"model": "HistGradientBoosting", "best_params": best_params,
               "default_cv_r2": [round(default_mean,4), round(default_std,4)],
               "tuned_cv_r2": [round(tuned_mean,4), round(tuned_std,4)]},
              open(outdir / "best_params.json", "w"), indent=2)
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage8_tuning", cfg,
                   metrics={"default_cv_r2": round(default_mean,4), "tuned_cv_r2": round(tuned_mean,4)},
                   outputs=["outputs/tuning/best_params.json"])

    print("\n================= STAGE 8 RESULTS =================")
    print(f"Model: HistGradientBoosting | DOI-grup 5-kat CV")
    print(f"  Default ayar : R2 = {default_mean:.4f} +/- {default_std:.4f}")
    print(f"  Tuned   ayar : R2 = {tuned_mean:.4f} +/- {tuned_std:.4f}   (kazanim {tuned_mean-default_mean:+.4f})")
    print(f"  En iyi parametreler: {best_params}")
    print(f"  ({len(rows)} kombinasyon denendi; tam tablo: outputs/tuning/search_results.csv)")
    print("===================================================")

if __name__ == "__main__":
    main()
