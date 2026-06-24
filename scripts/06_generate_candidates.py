"""Stage 6 — Aday kompozisyon uretimi + ekstrapolasyon analizi.
Girdi : data/processed/model_ready_dataset.csv, outputs/v4/best_model.joblib
Cikti : outputs/candidates_full/candidate_predictions.csv
        outputs/candidates_full/top30_diverse.csv
        outputs/candidates_full/metadata.json
Calistirma: python scripts/06_generate_candidates.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.candidates.candidate_space import enumerate_candidates, axis_sizes

def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    proc = ROOT / cfg["paths"]["processed_dir"]
    v4 = ROOT / cfg["paths"]["outputs_dir"] / "v4"
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "candidates_full"
    outdir.mkdir(parents=True, exist_ok=True)

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y_train = M[cfg["target"]].values
    bundle = joblib.load(v4 / "best_model.joblib")
    model, features, best = bundle["model"], bundle["features"], bundle["best_name"]

    # sabit varsayimlar = egitim medyanlari (belgeli)
    bg = float(M["band_gap"].median()) if "band_gap" in M else 1.6
    at = float(M["anneal_temp"].median()) if "anneal_temp" in M else 100.0
    ati = float(M["anneal_time"].median()) if "anneal_time" in M else 30.0

    recipes, enc = enumerate_candidates(features, bg, at, ati)
    preds = model.predict(enc)
    recipes["Predicted_PCE"] = preds
    recipes = recipes.sort_values("Predicted_PCE", ascending=False).reset_index(drop=True)
    recipes.to_csv(outdir / "candidate_predictions.csv", index=False)

    # cesitlendirilmis 30: A/B/X/HTL kombinasyonu benzersiz olacak sekilde
    diverse = recipes.drop_duplicates(subset=["A", "B", "X", "HTL"]).head(30)
    diverse.to_csv(outdir / "top30_diverse.csv", index=False)

    sizes = axis_sizes()
    n_expected = 1
    for v in sizes.values(): n_expected *= v
    meta = {"model": best, "axis_sizes": sizes, "n_candidates": len(recipes),
            "n_expected_product": n_expected,
            "assumptions": {"band_gap": bg, "anneal_temp": at, "anneal_time": ati},
            "pred_max": float(preds.max()), "pred_min": float(preds.min()),
            "pred_mean": float(preds.mean()),
            "train_pce_max": float(y_train.max()), "train_pce_p99": float(np.percentile(y_train,99))}
    json.dump(meta, open(outdir / "metadata.json", "w"), indent=2)

    print("\n================= STAGE 6 RESULTS =================")
    print(f"Model: {best}")
    print(f"Aday eksenleri: {sizes}")
    print(f"Beklenen kombinasyon (carpim): {n_expected}  -> uretilen: {len(recipes)}")
    print(f"Tahmini PCE: min={preds.min():.2f}  ort={preds.mean():.2f}  MAX={preds.max():.2f}")
    print(f"Egitim PCE:  MAX={y_train.max():.2f}  99.persentil={np.percentile(y_train,99):.2f}")
    print(f">> EKSTRAPOLASYON: aday tahmin tavani ({preds.max():.2f}) << egitim maksimumu ({y_train.max():.2f})")
    print(">> En yuksek 5 aday:")
    for _, r in recipes.head(5).iterrows():
        print(f"   PCE~{r['Predicted_PCE']:.2f} | A={r['A']} B={r['B']} X={r['X']} "
              f"{r['arch']}/{r['ETL']}/{r['HTL']}")
    print("===================================================")

if __name__ == "__main__":
    main()
