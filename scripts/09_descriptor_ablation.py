"""Stage 9 — Fiziksel tanimlayici ablasyonu (descriptor VAR vs YOK).
Goldschmidt t, oktahedral mu, Bartel tau eklemenin DOI-grup CV R2'ye etkisini olcer.
Girdi : data/processed/model_ready_dataset.csv (77 ozellik)
Cikti : outputs/ablation/descriptor_ablation.json
Calistirma: python scripts/09_descriptor_ablation.py
"""
import sys, json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.features.descriptors import add_descriptors
from perovskite_ml.models.tuning import grouped_cv_r2
from perovskite_ml.utils.manifest import write_manifest

def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    proc = ROOT / cfg["paths"]["processed_dir"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "ablation"; outdir.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values; g = M[cfg["group_col"]].fillna("u").astype(str).values
    Xwo = M.drop(columns=[cfg["target"], cfg["group_col"]])
    Xwith = add_descriptors(Xwo)
    m = lambda: HistGradientBoostingRegressor(random_state=cfg["seed"])
    r_wo, s_wo = grouped_cv_r2(m(), Xwo, y, g, 5)
    r_w, s_w = grouped_cv_r2(m(), Xwith, y, g, 5)
    res = {"descriptor_yok": {"r2": round(r_wo,4), "std": round(s_wo,4), "n_feat": Xwo.shape[1]},
           "descriptor_var": {"r2": round(r_w,4), "std": round(s_w,4), "n_feat": Xwith.shape[1]},
           "delta_r2": round(r_w - r_wo, 4)}
    json.dump(res, open(outdir / "descriptor_ablation.json","w"), indent=2)
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage9_ablation", cfg, metrics=res)
    print("\n================= STAGE 9 RESULTS =================")
    print(f"  Descriptor YOK : R2 = {r_wo:.4f} +/- {s_wo:.4f}  ({Xwo.shape[1]} ozellik)")
    print(f"  Descriptor VAR : R2 = {r_w:.4f} +/- {s_w:.4f}  ({Xwith.shape[1]} ozellik)")
    print(f"  Delta R2 = {r_w - r_wo:+.4f}  (fiziksel tanimlayicilar oranlarin fonksiyonu -> fazlalik)")
    print("===================================================")

if __name__ == "__main__":
    main()
