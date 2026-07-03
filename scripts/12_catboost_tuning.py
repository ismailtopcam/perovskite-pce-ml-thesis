"""Stage 12 — CatBoost icin grup-guvenli hiperparametre aramasi (default vs tuned).

Amac: Bolum 4.8'deki "final model (CatBoost) de hiperparametre ayarina duyarsizdir"
iddiasini DOGRUDAN CatBoost uzerinde dogrulamak (Stage 8 ayni deneyi HistGBM'de yapar).

Girdi : data/processed/model_ready_dataset.csv
Cikti : outputs/tuning_catboost/search_results.csv, best_params.json, manifest
Calistirma: python scripts/12_catboost_tuning.py
"""
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.tuning import grid_search_grouped, grouped_cv_r2
from perovskite_ml.utils.manifest import write_manifest


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    seed = cfg["seed"]; tgt = cfg["target"]; gcol = cfg["group_col"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "tuning_catboost"
    outdir.mkdir(parents=True, exist_ok=True)

    M = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "model_ready_dataset.csv", low_memory=False)
    y = M[tgt].values
    g = M[gcol].fillna("u").astype(str).values
    X = M.drop(columns=[tgt, gcol])

    try:
        from catboost import CatBoostRegressor
    except Exception as e:
        print(f"[stage12] CatBoost kurulu degil ({e.__class__.__name__}); atlaniyor.")
        return

    def cb(**p):
        return CatBoostRegressor(verbose=0, random_seed=seed, allow_writing_files=False, **p)

    # Tezdeki/registry'deki varsayilan ayar: iterations=500, depth=6, learning_rate=0.05
    default_mean, default_std = grouped_cv_r2(cb(iterations=500, depth=6, learning_rate=0.05), X, y, g, 5)

    grid = {"depth": [6, 8], "learning_rate": [0.03, 0.05, 0.1], "iterations": [500]}
    best, rows = grid_search_grouped(cb, grid, X, y, g, n_splits=3)   # arama 3-kat (hiz)
    best_params = {k: best[k] for k in grid}
    tuned_mean, tuned_std = grouped_cv_r2(cb(**best_params), X, y, g, 5)  # secilenin durust 5-kat skoru

    pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).to_csv(outdir / "search_results.csv", index=False)
    json.dump({"model": "CatBoost", "best_params": best_params,
               "default_cv_r2": [round(default_mean, 4), round(default_std, 4)],
               "tuned_cv_r2": [round(tuned_mean, 4), round(tuned_std, 4)]},
              open(outdir / "best_params.json", "w"), indent=2)

    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage12_catboost_tuning", cfg,
                   metrics={"default_cv_r2": round(default_mean, 4), "tuned_cv_r2": round(tuned_mean, 4)},
                   outputs=["outputs/tuning_catboost/best_params.json"])

    print("\n================= STAGE 12: CatBoost grup-guvenli arama =================")
    print(f"  Default (500, 6, 0.05) : R2 = {default_mean:.4f} +/- {default_std:.4f}")
    print(f"  Tuned   {best_params}: R2 = {tuned_mean:.4f} +/- {tuned_std:.4f}   (kazanim {tuned_mean - default_mean:+.4f})")
    print(f"  ({len(rows)} kombinasyon; tam tablo: outputs/tuning_catboost/search_results.csv)")
    print("=========================================================================")


if __name__ == "__main__":
    main()
