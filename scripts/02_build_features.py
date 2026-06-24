"""Stage 2 — Ozellik matrisi runner'i.
Girdi : data/processed/composition_vectorized.csv  (Stage 1 ciktisi)
Cikti : data/processed/model_ready_dataset.csv
Calistirma: python scripts/02_build_features.py
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perovskite_ml.config import load_config
from perovskite_ml.features.feature_builder import build_features
from perovskite_ml.validation.schema import validate_model_ready
from perovskite_ml.utils.manifest import write_manifest
import pandas as pd

def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    proc = ROOT / cfg["paths"]["processed_dir"]
    src = proc / "composition_vectorized.csv"
    if not src.exists():
        raise FileNotFoundError("Once Stage 1'i calistir: python scripts/01_prepare_data.py")
    df = pd.read_csv(src, low_memory=False)
    M = build_features(df, cfg)
    # veri-dogrulama / sema (sizinti kontrolu dahil) — gecmezse hat durur
    rep = validate_model_ready(M, cfg)
    M.to_csv(proc / "model_ready_dataset.csv", index=False)
    for w in rep["warnings"]:
        print(f"   [UYARI] {w}")
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage2_features", cfg,
                   metrics={"n_rows": rep["n_rows"], "n_features": int(M.shape[1]-2)},
                   outputs=["data/processed/model_ready_dataset.csv"])

    n_feat = M.shape[1] - 2   # hedef + grup haric
    print("\n================= STAGE 2 RESULTS =================")
    print(f"Model-ready satir   : {M.shape[0]}")
    print(f"Toplam ozellik      : {n_feat}  (+ hedef + DOI)")
    print(f"Toplam kolon        : {M.shape[1]}")
    # ozellik grubu kirilim
    groups = {}
    for c in M.columns:
        if c in (cfg["target"], cfg["group_col"]): continue
        key = c.split("_")[0]
        key = {"A":"komp_A","B":"komp_B","X":"komp_X","arch":"mimari",
               "ETL":"ETL","HTL":"HTL","solv":"cozucu"}.get(key, "diger")
        groups[key] = groups.get(key, 0) + 1
    print(f"Ozellik gruplari    : {groups}")
    print("===================================================")

if __name__ == "__main__":
    main()
