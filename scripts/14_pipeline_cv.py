"""Stage 14 — Ön-işleme sızıntısı doğrulaması: fold-local vs global.

Amaç: imputation (medyan) ve top-k one-hot adımlarını sklearn Pipeline içinde
KAT-İÇİNDE fit ederek, global ön-işlemeyle elde edilen başarımdan farklı olup
olmadığını ölçmek. Aynı GroupKFold (DOI) bölmeleri üzerinde adil kıyas yapılır.

Beklenen sonuç: fark ~0 (global ön-işleme kaynaklı sızıntı ihmal edilebilir).
Çıktı: outputs/pipeline_cv/results.json + manifest.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.features.feature_builder import _first_number
from perovskite_ml.models.registry import build_models
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.base import clone

def build_raw_features(df, f):
    """Ön-işleme ÖNCESİ özellik matrisi: ham sayısal (NaN'lı) + ham kategorik (str).
    Eksiklik bayrakları ve ikili cihaz bilgileri satır-bazlı/deterministiktir (sızıntısız)."""
    comp = [c for c in df.columns if c.split("_")[0] in ("A", "B", "X")]
    X = df[comp].copy()
    bg = pd.to_numeric(df[f["band_gap_col"]], errors="coerce")
    X["band_gap_missing"] = bg.isna().astype(int); X["band_gap_raw"] = bg
    X["flexible"] = (df[f["flexible_col"]].astype(str).str.lower() == "true").astype(int)
    X["semitransparent"] = (df[f["semitransparent_col"]].astype(str).str.lower() == "true").astype(int)
    for col, name in [(f["anneal_temp_col"], "anneal_temp"), (f["anneal_time_col"], "anneal_time")]:
        v = df[col].map(_first_number)
        X[f"{name}_missing"] = v.isna().astype(int); X[f"{name}_raw"] = v
    for col, name in [(f["arch_col"], "arch"), (f["etl_col"], "ETL"),
                      (f["htl_col"], "HTL"), (f["solvent_col"], "solv")]:
        X[name + "_cat"] = df[col].fillna("unknown").astype(str)
    return X, comp

def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    f = cfg["features"]; seed = cfg["seed"]; tgt = cfg["target"]; gc = cfg["group_col"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "pipeline_cv"; outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(proc / "composition_vectorized.csv", low_memory=False)
    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    assert len(df) == len(M) and np.allclose(df[tgt].values, M[tgt].values), "satır hizası bozuk"
    y = df[tgt].values
    groups = df[gc].fillna("unknown_doi").astype(str).values   # headline ile aynı kural

    X, comp = build_raw_features(df, f)
    num_imp = ["band_gap_raw", "anneal_temp_raw", "anneal_time_raw"]
    cat = ["arch_cat", "ETL_cat", "HTL_cat", "solv_cat"]
    passth = comp + ["band_gap_missing", "anneal_temp_missing", "anneal_time_missing",
                     "flexible", "semitransparent"]
    def make_ct():
        return ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_imp),
            ("cat", OneHotEncoder(max_categories=f["onehot_top_n"] + 1,
                                  handle_unknown="infrequent_if_exist", sparse_output=False), cat),
            ("pass", "passthrough", passth)])

    base = build_models(seed)["CatBoost"]
    Xg = M.drop(columns=[tgt, gc])
    gkf = GroupKFold(n_splits=5); rA, rB = [], []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        mg = clone(base); mg.fit(Xg.iloc[tr], y[tr]); rA.append(r2_score(y[te], mg.predict(Xg.iloc[te])))
        pl = Pipeline([("prep", make_ct()), ("model", clone(base))])
        pl.fit(X.iloc[tr], y[tr]); rB.append(r2_score(y[te], pl.predict(X.iloc[te])))
        print(f"Fold {k+1}: GLOBAL={rA[-1]:.4f}  FOLD-LOCAL={rB[-1]:.4f}")

    res = {"global_mean": float(np.mean(rA)), "global_std": float(np.std(rA)),
           "foldlocal_mean": float(np.mean(rB)), "foldlocal_std": float(np.std(rB)),
           "delta": float(np.mean(rB) - np.mean(rA)),
           "global_folds": [float(x) for x in rA], "foldlocal_folds": [float(x) for x in rB]}
    json.dump(res, open(outdir / "results.json", "w"), indent=2)
    print(f"\nGLOBAL     R² = {res['global_mean']:.4f} ± {res['global_std']:.4f}")
    print(f"FOLD-LOCAL R² = {res['foldlocal_mean']:.4f} ± {res['foldlocal_std']:.4f}")
    print(f"Fark = {res['delta']:+.4f}")
    try:
        from perovskite_ml.utils.manifest import write_manifest
        write_manifest(outdir, "14_pipeline_cv", cfg, res, ["results.json"])
    except Exception:
        pass

if __name__ == "__main__":
    main()
