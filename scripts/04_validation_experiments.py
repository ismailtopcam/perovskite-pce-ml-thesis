"""Stage 4 — Dogrulama metodolojisi deneyleri (tezin SE omurgasi).
Deney A: Rastgele KFold vs DOI-grup KFold  -> sizinti kaynakli yapay sisme.
Deney C: Ozellik kapsami (kompozisyon -> +cihaz -> +surec) -> PCE tek basina
         kompozisyonla aciklanamaz; surec bilgisi sart.
Girdi : data/processed/model_ready_dataset.csv
Cikti : outputs/v4/validation_experiments.csv
Calistirma: python scripts/04_validation_experiments.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.registry import build_models
from perovskite_ml.data.load_data import load_raw
from perovskite_ml.data.clean_data import clean_rows
from perovskite_ml.features.composition import vectorize_dataframe
from perovskite_ml.features.feature_builder import build_features
from perovskite_ml.utils.manifest import write_manifest

LEAK_COLS = ["JV_default_Voc", "JV_default_Jsc", "JV_default_FF"]

def cv_r2(model, X, y, splits):
    r2s = []
    for tri, vai in splits:
        m = clone(model); m.fit(X.iloc[tri], y[tri])
        r2s.append(r2_score(y[vai], m.predict(X.iloc[vai])))
    return float(np.mean(r2s)), float(np.std(r2s))

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

    # en iyi modeli sec (Stage 3 metadata'sindan; yoksa HistGBM)
    meta_path = outdir / "metadata.json"
    best_name = json.load(open(meta_path))["best_model"] if meta_path.exists() else "HistGBM"
    models = build_models(seed)
    model = models.get(best_name, models["HistGBM"])
    print(f"[stage4] Kullanilan model: {best_name if best_name in models else 'HistGBM'}")

    rows = []

    # --- Deney A: bolme stratejisi ---
    rand = list(KFold(5, shuffle=True, random_state=seed).split(X))
    grp = list(GroupKFold(5).split(X, y, groups))
    r_mean, r_std = cv_r2(model, X, y, rand)
    g_mean, g_std = cv_r2(model, X, y, grp)
    rows.append({"deney": "A_bolme", "ayar": "rastgele_KFold", "R2_mean": r_mean, "R2_std": r_std})
    rows.append({"deney": "A_bolme", "ayar": "DOI_grup_KFold", "R2_mean": g_mean, "R2_std": g_std})
    inflation = r_mean - g_mean

    # --- Deney B: olcum-sonrasi (leaky) degiskenler ON vs OFF ---
    b_done = False
    try:
        raw_full = pd.read_csv(ROOT / cfg["paths"]["raw_csv"], encoding="utf-8-sig", low_memory=False)
        have = [c for c in LEAK_COLS if c in raw_full.columns]
        if have:
            cleaned_b, _ = clean_rows(raw_full, cfg)
            vec_b = vectorize_dataframe(cleaned_b, cfg)
            Mb = build_features(vec_b, cfg)
            yb = Mb[cfg["target"]].values
            gb = Mb[cfg["group_col"]].fillna("unknown_doi").astype(str).values
            Xb_safe = Mb.drop(columns=[cfg["target"], cfg["group_col"]])
            leak = cleaned_b.loc[vec_b.index, have].apply(pd.to_numeric, errors="coerce").fillna(0).reset_index(drop=True)
            Xb_leak = pd.concat([Xb_safe.reset_index(drop=True), leak], axis=1)
            grp_b = list(GroupKFold(5).split(Xb_safe, yb, gb))
            safe_mean, safe_std = cv_r2(model, Xb_safe, yb, grp_b)
            leak_mean, leak_std = cv_r2(model, Xb_leak, yb, grp_b)
            rows.append({"deney": "B_olcum_sonrasi", "ayar": "leakage_safe", "R2_mean": safe_mean, "R2_std": safe_std})
            rows.append({"deney": "B_olcum_sonrasi", "ayar": f"leakage_ON (+{','.join(c.split('_')[-1] for c in have)})",
                         "R2_mean": leak_mean, "R2_std": leak_std})
            b_done = True
    except FileNotFoundError as e:
        # Ham CSV depoda dagitilmaz; yoksa yalnizca Deney B atlanir (A ve C etkilenmez).
        print(f"[stage4] Deney B atlandi — ham CSV bulunamadi: {e}")

    # --- Deney C: ozellik kapsami (DOI-grup ile) ---
    comp = [c for c in X.columns if c.split("_")[0] in ("A", "B", "X")]
    process = [c for c in X.columns if c.startswith("solv_") or c.startswith("anneal")]
    device = [c for c in X.columns if c not in comp and c not in process]
    scopes = [("kompozisyon", comp),
              ("+cihaz", comp + device),
              ("+surec_tam", comp + device + process)]
    for name, cols in scopes:
        m_mean, m_std = cv_r2(model, X[cols], y, grp)
        rows.append({"deney": "C_kapsam", "ayar": f"{name} ({len(cols)} ozellik)",
                     "R2_mean": m_mean, "R2_std": m_std})

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "validation_experiments.csv", index=False)
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage4_validation", cfg,
                   metrics={"random_kfold_r2": round(r_mean, 4),
                            "group_kfold_r2": round(g_mean, 4),
                            "inflation_r2": round(inflation, 4),
                            "experiment_b_done": b_done},
                   outputs=["outputs/v4/validation_experiments.csv"])

    print("\n================= STAGE 4 RESULTS =================")
    print(">> Deney A — Bolme stratejisi (ayni model, ayni ozellikler):")
    print(f"   Rastgele KFold : R2 = {r_mean:.3f} +/- {r_std:.3f}")
    print(f"   DOI-grup KFold : R2 = {g_mean:.3f} +/- {g_std:.3f}")
    print(f"   >> Sizinti kaynakli yapay sisme = {inflation:.3f} R2 puani")
    if b_done:
        b = df[df.deney == "B_olcum_sonrasi"]
        print(">> Deney B — Olcum-sonrasi degiskenler (DOI-grup KFold):")
        for _, r in b.iterrows():
            print(f"   {r['ayar']:32s} R2 = {r['R2_mean']:.3f} +/- {r['R2_std']:.3f}")
    print(">> Deney C — Ozellik kapsami (hepsi DOI-grup KFold):")
    for _, r in df[df.deney == "C_kapsam"].iterrows():
        print(f"   {r['ayar']:28s} R2 = {r['R2_mean']:.3f} +/- {r['R2_std']:.3f}")
    print("===================================================")

if __name__ == "__main__":
    main()
