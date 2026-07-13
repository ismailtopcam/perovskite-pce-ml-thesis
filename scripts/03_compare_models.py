"""Stage 3 — Model karsilastirmasi (DOI-grup-guvenli).
Girdi : data/processed/model_ready_dataset.csv  (Stage 2 ciktisi)
Cikti : outputs/v4/model_comparison_holdout.csv
        outputs/v4/model_comparison_groupkfold.csv
        outputs/v4/best_model.joblib
        outputs/v4/best_model_holdout_predictions.csv
        outputs/v4/metadata.json
Calistirma: python scripts/03_compare_models.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.registry import build_models
from perovskite_ml.models.evaluate import metrics
from perovskite_ml.validation.splits import holdout_split, group_kfold
from perovskite_ml.utils.manifest import write_manifest

def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    seed = cfg["seed"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "v4"
    outdir.mkdir(parents=True, exist_ok=True)

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values
    groups = M[cfg["group_col"]].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[cfg["target"], cfg["group_col"]])

    models = build_models(seed)
    print(f"[stage3] Aktif modeller: {list(models.keys())}")

    # --- holdout (DOI-grup 80/20) ---
    tr, te = holdout_split(X, y, groups, test_size=0.2, seed=seed)
    assert set(groups[tr]).isdisjoint(set(groups[te])), "DOI sizinti kontrolu basarisiz!"
    hold_rows = []
    fitted = {}
    for name, est in models.items():
        m = clone(est); m.fit(X.iloc[tr], y[tr])
        mt = metrics(y[te], m.predict(X.iloc[te]))
        hold_rows.append({"model": name, **mt}); fitted[name] = m
    hold_df = pd.DataFrame(hold_rows).sort_values("R2", ascending=False)

    # --- GroupKFold CV (5) ---
    folds = group_kfold(X, y, groups, n_splits=5)
    cv_rows = []
    for name, est in models.items():
        maes, rmses, r2s = [], [], []
        for tri, vai in folds:
            m = clone(est); m.fit(X.iloc[tri], y[tri])
            mt = metrics(y[vai], m.predict(X.iloc[vai]))
            maes.append(mt["MAE"]); rmses.append(mt["RMSE"]); r2s.append(mt["R2"])
        cv_rows.append({"model": name,
            "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
            "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
            "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
            "R2_folds": [round(x, 3) for x in r2s]})
    cv_df = pd.DataFrame(cv_rows).sort_values("R2_mean", ascending=False)

    # --- en iyi model (CV R2'ye gore) ---
    best = cv_df.iloc[0]["model"]
    best_model = clone(models[best]); best_model.fit(X.iloc[tr], y[tr])
    joblib.dump({"model": best_model, "features": list(X.columns), "best_name": best},
                outdir / "best_model.joblib")
    pred_df = pd.DataFrame({"DOI": groups[te], "y_true": y[te],
                            "y_pred": best_model.predict(X.iloc[te])})
    pred_df.to_csv(outdir / "best_model_holdout_predictions.csv", index=False)
    # --- gercek-vs-tahmin raporu (okunabilir kompozisyon + ornekler + scatter) ---
    from perovskite_ml.models.reporting import build_predictions_table, stratified_examples, save_scatter
    M_test = M.iloc[te]
    rep = build_predictions_table(M_test, y[te], best_model.predict(X.iloc[te]), groups[te])
    rep.to_csv(outdir / "actual_vs_predicted.csv", index=False)
    stratified_examples(rep).to_csv(outdir / "prediction_examples.csv", index=False)
    save_scatter(y[te], best_model.predict(X.iloc[te]), outdir / "actual_vs_predicted.png")
    err = rep["Mutlak_hata"]
    print(f"\n>> Gercek-vs-tahmin: MAE={err.mean():.2f} | hata<=1: {(err<=1).mean()*100:.0f}%  <=2: {(err<=2).mean()*100:.0f}%  <=3: {(err<=3).mean()*100:.0f}%")

    hold_df.to_csv(outdir / "model_comparison_holdout.csv", index=False)
    cv_df.drop(columns=["R2_folds"]).to_csv(outdir / "model_comparison_groupkfold.csv", index=False)
    json.dump({"best_model": best, "n_features": X.shape[1], "n_rows": len(M),
               "active_models": list(models.keys())},
              open(outdir / "metadata.json", "w"), indent=2)
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage3_models", cfg,
                   metrics={"best_model": best,
                            "best_cv_r2": round(float(cv_df.iloc[0]["R2_mean"]), 4),
                            "best_holdout_r2": round(float(hold_df.iloc[0]["R2"]), 4)},
                   outputs=["outputs/v4/model_comparison_holdout.csv",
                            "outputs/v4/model_comparison_groupkfold.csv",
                            "outputs/v4/best_model_holdout_predictions.csv"])

    print("\n================= STAGE 3 RESULTS =================")
    print(">> HOLDOUT (DOI-grup 80/20):")
    for _, r in hold_df.iterrows():
        print(f"   {r['model']:13s} R2={r['R2']:.3f}  MAE={r['MAE']:.3f}  RMSE={r['RMSE']:.3f}")
    print(">> GroupKFold CV (5-fold):")
    for _, r in cv_df.iterrows():
        print(f"   {r['model']:13s} R2={r['R2_mean']:.3f}+/-{r['R2_std']:.3f}  "
              f"MAE={r['MAE_mean']:.3f}  RMSE={r['RMSE_mean']:.3f}  folds={r['R2_folds']}")
    print(f">> EN IYI MODEL (CV R2): {best}")
    print("===================================================")

if __name__ == "__main__":
    main()
