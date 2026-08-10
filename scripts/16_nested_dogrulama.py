"""Stage 16 (v0.4.0 / 1-C) — Ic-ice (nested) dogrulama ve muhurlu holdout.

Tezin v0.3.x kurulumunda model ailesi, holdout kayitlarini da iceren TUM veri
uzerindeki grup-CV skoruna gore secilmisti; holdout bu nedenle "muhurlu dis
test" niteligi tasimiyordu (tez 4.2'de acikca belirtilir). Bu betik iki
tamamlayici protokol kosar:

(a) MUHURLU HOLDOUT: model ailesi secimi YALNIZCA egitim bolmesindeki 5-katli
    GroupKFold ile yapilir; secilen tek model egitim bolmesinde yeniden egitilir
    ve holdout'ta BIR KEZ degerlendirilir. Holdout boylece secimden bagimsiz,
    muhurlu bir dis test isleviyle okunabilir.
(b) NESTED CV: dis 5-kat GroupKFold; her dis katta ic 3-kat GroupKFold ile model
    ailesi secilir, secilen model dis katin testinde degerlendirilir. Dis
    skorlarin ortalamasi, model-secim etkisinden arindirilmis genelleme
    kestirimidir.

Girdi : data/processed/model_ready_dataset.csv  (v0.4.0: kanonik DOI gruplari)
Cikti : outputs/nested/nested_results.json
        outputs/nested/sealed_holdout_model_selection.csv
        outputs/nested/nested_outer_folds.csv
Calistirma: python scripts/16_nested_dogrulama.py
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.registry import build_models
from perovskite_ml.validation.splits import holdout_split
from perovskite_ml.utils.manifest import write_manifest

INNER_SPLITS = 3
OUTER_SPLITS = 5
TRAIN_CV_SPLITS = 5


def scores(y_true, y_pred):
    return dict(R2=float(r2_score(y_true, y_pred)),
                MAE=float(mean_absolute_error(y_true, y_pred)),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def cv_r2(model, X, y, groups, n_splits):
    vals = []
    for tri, vai in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = clone(model)
        m.fit(X.iloc[tri], y[tri])
        vals.append(r2_score(y[vai], m.predict(X.iloc[vai])))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    seed = cfg["seed"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "nested"
    outdir.mkdir(parents=True, exist_ok=True)

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values
    groups = M[cfg["group_col"]].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[cfg["target"], cfg["group_col"]])

    t0 = time.time()

    # ---------- (a) Muhurlu holdout ----------
    tr, ho = holdout_split(X, y, groups, test_size=0.2, seed=seed)
    assert set(groups[tr]).isdisjoint(set(groups[ho])), "DOI sizinti kontrolu basarisiz!"
    Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]

    sel_rows = []
    for name, model in build_models(seed).items():
        m, s = cv_r2(model, Xtr, ytr, gtr, TRAIN_CV_SPLITS)
        sel_rows.append(dict(model=name, train_cv_r2_mean=round(m, 4), train_cv_r2_std=round(s, 4)))
        print(f"[a] egitim-ici {TRAIN_CV_SPLITS}-kat CV  {name:<12s} R2 = {m:.4f} +/- {s:.4f}")
    sel_df = pd.DataFrame(sel_rows).sort_values("train_cv_r2_mean", ascending=False)
    winner = sel_df.iloc[0]["model"]

    wm = clone(build_models(seed)[winner])
    wm.fit(Xtr, ytr)
    hold = scores(y[ho], wm.predict(X.iloc[ho]))
    print(f"[a] SECILEN (yalnizca egitim-ici CV ile): {winner}")
    print(f"[a] Muhurlu holdout: R2={hold['R2']:.4f}  MAE={hold['MAE']:.4f}  RMSE={hold['RMSE']:.4f}")
    sel_df.to_csv(outdir / "sealed_holdout_model_selection.csv", index=False)

    # ---------- (b) Nested CV ----------
    outer_rows = []
    for k, (otr, ote) in enumerate(GroupKFold(n_splits=OUTER_SPLITS).split(X, y, groups), 1):
        Xo, yo, go = X.iloc[otr], y[otr], groups[otr]
        best_name, best_r2 = None, -np.inf
        for name, model in build_models(seed).items():
            m, _ = cv_r2(model, Xo, yo, go, INNER_SPLITS)
            if m > best_r2:
                best_name, best_r2 = name, m
        fm = clone(build_models(seed)[best_name])
        fm.fit(Xo, yo)
        sc = scores(y[ote], fm.predict(X.iloc[ote]))
        outer_rows.append(dict(fold=k, selected=best_name, inner_r2=round(best_r2, 4),
                               R2=round(sc["R2"], 4), MAE=round(sc["MAE"], 4), RMSE=round(sc["RMSE"], 4)))
        print(f"[b] dis kat {k}: ic-secim={best_name} (ic R2={best_r2:.4f})  "
              f"dis R2={sc['R2']:.4f} MAE={sc['MAE']:.4f} RMSE={sc['RMSE']:.4f}")
    od = pd.DataFrame(outer_rows)
    od.to_csv(outdir / "nested_outer_folds.csv", index=False)

    nested = dict(R2_mean=round(float(od["R2"].mean()), 4), R2_std=round(float(od["R2"].std(ddof=0)), 4),
                  MAE_mean=round(float(od["MAE"].mean()), 4), RMSE_mean=round(float(od["RMSE"].mean()), 4),
                  selected_per_fold=od["selected"].tolist())
    print(f"[b] NESTED sonuc: R2 = {nested['R2_mean']} +/- {nested['R2_std']}  "
          f"MAE = {nested['MAE_mean']}  RMSE = {nested['RMSE_mean']}")

    results = {
        "protocol_a_sealed_holdout": {
            "selection": sel_rows,
            "selected_model": winner,
            "holdout": {k: round(v, 4) for k, v in hold.items()},
            "n_train": int(len(tr)), "n_holdout": int(len(ho)),
        },
        "protocol_b_nested_cv": {
            "outer_splits": OUTER_SPLITS, "inner_splits": INNER_SPLITS,
            "outer_folds": outer_rows, **nested,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(outdir / "nested_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage16_nested", cfg,
                   metrics={"sealed_holdout_R2": round(hold["R2"], 4),
                            "nested_R2_mean": nested["R2_mean"]},
                   outputs=["outputs/nested/nested_results.json"])
    print(f"\nTamam ({results['elapsed_sec']} sn). Cikti: outputs/nested/")


if __name__ == "__main__":
    main()
