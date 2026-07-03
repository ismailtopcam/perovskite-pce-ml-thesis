"""Stage 5 — SHAP aciklanabilirlik analizi (nihai/en iyi model uzerinde).
Girdi : data/processed/model_ready_dataset.csv, outputs/v4/best_model.joblib
Cikti : outputs/shap_full/shap_top_features.csv
        outputs/shap_full/shap_summary_plot.png
        outputs/shap_full/shap_top20_bar.png
        outputs/shap_full/metadata.json
Calistirma: python scripts/05_shap_analysis.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, shap
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.registry import build_models
from perovskite_ml.validation.splits import holdout_split

SHAP_SAMPLE = 3000   # SHAP degerleri buyuk veride pahali; temsili ornek

def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    seed = cfg["seed"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    v4 = ROOT / cfg["paths"]["outputs_dir"] / "v4"
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "shap_full"
    outdir.mkdir(parents=True, exist_ok=True)

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values
    groups = M[cfg["group_col"]].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[cfg["target"], cfg["group_col"]])

    # en iyi modeli yukle (yoksa yeniden egit)
    jp = v4 / "best_model.joblib"
    if jp.exists():
        bundle = joblib.load(jp); model = bundle["model"]; best = bundle["best_name"]
        print(f"[stage5] best_model.joblib yuklendi: {best}")
    else:
        meta = json.load(open(v4 / "metadata.json")) if (v4/"metadata.json").exists() else {"best_model":"CatBoost"}
        best = meta["best_model"]
        tr, te = holdout_split(X, y, groups, 0.2, seed)
        model = clone(build_models(seed)[best]); model.fit(X.iloc[tr], y[tr])
        print(f"[stage5] {best} yeniden egitildi")

    # SHAP (agac modeli icin TreeExplainer)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(SHAP_SAMPLE, len(X)), replace=False)
    Xs = X.iloc[idx]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)

    mean_abs = np.abs(sv).mean(axis=0)
    imp = (pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
             .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    imp.to_csv(outdir / "shap_top_features.csv", index=False)

    # grafikler
    plt.figure()
    shap.summary_plot(sv, Xs, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(outdir / "shap_summary_plot.png", dpi=300); plt.close()

    top20 = imp.head(20)[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(top20["feature"], top20["mean_abs_shap"])
    plt.xlabel("Ortalama |SHAP| (PCE puani)"); plt.tight_layout()
    plt.savefig(outdir / "shap_top20_bar.png", dpi=300); plt.close()

    json.dump({"model": best, "shap_sample": int(len(idx))},
              open(outdir / "metadata.json", "w"), indent=2)

    print("\n================= STAGE 5 RESULTS =================")
    print(f"Model: {best} | SHAP ornek: {len(idx)} kayit")
    print(">> En etkili 15 ozellik (ortalama |SHAP|):")
    for _, r in imp.head(15).iterrows():
        print(f"   {r['feature']:30s} {r['mean_abs_shap']:.4f}")
    print("===================================================")
    print("Grafikler: outputs/shap_full/")

if __name__ == "__main__":
    main()
