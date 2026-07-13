"""Stage 7 — Model tabanli onceliklendirme faydasi (zenginlestirme + lift egrisi).
Girdi : outputs/v4/best_model_holdout_predictions.csv (Stage 3'ten; y_true,y_pred)
Cikti : outputs/screening/enrichment.csv, gains_curve.png, metadata.json
Calistirma: python scripts/07_screening_utility.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.simulation.screening import enrichment_table, gains_curve
from perovskite_ml.utils.manifest import write_manifest

THRESHOLDS = [18, 20, 22]
K_FRACS = [0.05, 0.10, 0.20]
GAINS_THR = 18

def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    v4 = ROOT / cfg["paths"]["outputs_dir"] / "v4"
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "screening"
    outdir.mkdir(parents=True, exist_ok=True)

    pred_path = v4 / "best_model_holdout_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError("Once Stage 3'u calistir (best_model_holdout_predictions.csv gerekli).")
    df = pd.read_csv(pred_path)
    y_true, y_pred = df["y_true"].values, df["y_pred"].values
    rho = spearmanr(y_pred, y_true).correlation

    rows = enrichment_table(y_true, y_pred, THRESHOLDS, K_FRACS)
    et = pd.DataFrame(rows); et.to_csv(outdir / "enrichment.csv", index=False)

    fr, cap = gains_curve(y_true, y_pred, GAINS_THR)
    plt.figure(figsize=(6.5, 5))
    plt.plot(fr * 100, cap * 100, linewidth=2, label="Model-yonlendirmeli secim")
    plt.plot([0, 100], [0, 100], "--", color="gray", label="Rastgele secim")
    plt.xlabel("Incelenen hucre orani (%)")
    plt.ylabel(f"Yakalanan yuksek-verim (PCE>={GAINS_THR}) orani (%)")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir / "gains_curve.png", dpi=300); plt.close()

    json.dump({"n_test": int(len(df)), "spearman": float(rho), "gains_threshold": GAINS_THR},
              open(outdir / "metadata.json", "w"), indent=2)
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage7_screening", cfg,
                   metrics={"n_test": int(len(df)), "spearman": round(float(rho), 4),
                            "gains_threshold": GAINS_THR},
                   outputs=["outputs/screening/enrichment.csv"])

    print("\n================= STAGE 7 RESULTS =================")
    print(f"Holdout test: {len(df)} hucre | Spearman(tahmin,gercek) = {rho:.3f}")
    for thr in THRESHOLDS:
        sub = et[et.esik == thr]
        base = sub.iloc[0]["taban_oran"]
        print(f"\nYuksek-verim esigi PCE>={thr} (taban oran = {base*100:.1f}%):")
        for _, r in sub.iterrows():
            print(f"   top-%{int(r['k_frac']*100):2d} (k={int(r['k']):4d}): "
                  f"isabet={r['model_isabet']*100:4.1f}%  ->  zenginlestirme {r['zenginlestirme']:.2f}x")
    print("===================================================")
    print("Lift egrisi: outputs/screening/gains_curve.png")

if __name__ == "__main__":
    main()
