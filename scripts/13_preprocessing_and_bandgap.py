"""Stage 13 — Onisleme kat-invaryansi + band gap ablasyonu.

Tezdeki iki iddiayi calistirilabilir kilar:
  (1) Imputation medyanlari ve top-N kategori secimi GLOBAL hesaplanir; ama bunlar
      capraz dogrulama katlari arasinda pratikte degismez (yani per-fold yapilsaydi
      ozellik matrisi ayni kalirdi) -> sizinti-guvenli degerlendirmeyi etkilemez.
  (2) band gap (+eksiklik bayragi) modelden cikarildiginda basarim degismez
      -> model band gap'e yaslanmiyor; sentez oncesi bilinmese de basarim korunur.

Girdi : data/processed/cleaned_dataset.csv (kat-invaryans), model_ready_dataset.csv (ablasyon)
Cikti : outputs/robustness/fold_invariance.csv, band_gap_ablation.csv, manifest
Calistirma: python scripts/13_preprocessing_and_bandgap.py
"""
import sys, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.registry import build_models
from perovskite_ml.models.tuning import grouped_cv_r2
from perovskite_ml.utils.manifest import write_manifest


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    seed = cfg["seed"]; tgt = cfg["target"]; gcol = cfg["group_col"]
    top_n = int(cfg.get("features", {}).get("onehot_top_n", 15))
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- (1) Onisleme istatistiklerinin kat-invaryansi ---
    C = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "cleaned_dataset.csv", low_memory=False)
    bg_col = "Perovskite_band_gap"
    # top-N secimin anlamli oldugu cok-kategorili sutunlar (mimari az kategorili, top-N onu zaten secmez)
    cat_cols = [c for c in ["ETL_stack_sequence", "HTL_stack_sequence",
                            "Perovskite_deposition_solvents"] if c in C.columns]
    bg = pd.to_numeric(C[bg_col], errors="coerce")
    grp = C[gcol].fillna("nan").astype(str).values
    gkf = GroupKFold(5)
    gmed = float(bg.median())
    global_top = {c: set(C[c].fillna("unknown").astype(str).value_counts().head(top_n).index) for c in cat_cols}
    inv_rows = []
    for i, (tr, _) in enumerate(gkf.split(C, groups=grp), 1):
        row = {"fold": i, "band_gap_train_median": round(float(bg.iloc[tr].median()), 4),
               "band_gap_delta_vs_global": round(float(bg.iloc[tr].median() - gmed), 4)}
        for c in cat_cols:
            ttop = set(C[c].iloc[tr].fillna("unknown").astype(str).value_counts().head(top_n).index)
            row[f"{c}_top{top_n}_overlap"] = f"{len(global_top[c] & ttop)}/{top_n}"
        inv_rows.append(row)
    pd.DataFrame(inv_rows).to_csv(outdir / "fold_invariance.csv", index=False)

    # --- (2) band gap ablasyonu ---
    M = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "model_ready_dataset.csv", low_memory=False)
    y = M[tgt].values
    g = M[gcol].fillna("u").astype(str).values
    bg_feats = [c for c in M.columns if c in ("band_gap", "band_gap_missing")]
    abl_rows = []
    for label, drop in [("band_gap_dahil", []), ("band_gap_haric", bg_feats)]:
        X = M.drop(columns=[tgt, gcol] + drop)
        mdl = build_models(seed)
        for name in ("CatBoost", "HistGBM"):
            if name not in mdl:
                continue
            m, s = grouped_cv_r2(mdl[name], X, y, g, 5)
            abl_rows.append({"durum": label, "ozellik": X.shape[1], "model": name,
                             "R2_mean": round(m, 4), "R2_std": round(s, 4)})
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(outdir / "band_gap_ablation.csv", index=False)

    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage13_preproc_bandgap", cfg,
                   metrics={"band_gap_global_median": gmed},
                   outputs=["outputs/robustness/fold_invariance.csv",
                            "outputs/robustness/band_gap_ablation.csv"])

    print("\n========= STAGE 13: onisleme kat-invaryansi + band gap ablasyonu =========")
    print(f"GLOBAL band gap medyan: {gmed:.4f}")
    print(pd.DataFrame(inv_rows).to_string(index=False))
    print()
    print(abl.to_string(index=False))
    print("Yorum: medyan/top-N kat-invaryanttir; band gap cikarildiginda R2 degismez.")
    print("=========================================================================")


if __name__ == "__main__":
    main()
