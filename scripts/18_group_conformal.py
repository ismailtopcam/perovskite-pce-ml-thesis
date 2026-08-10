"""Stage 18 (v0.4.0 / 1-C) — Group-conformal: DOI'nin degisebilirlik birimi
olarak alindigi conformal araliklar + yayin-agirlikli kapsama raporu.

v0.3.x'te (Stage 10) uygunsuzluk skorlari KAYIT duzeyinde toplanir; ayni DOI'nin
bagimli kayitlari ve buyuk yayinlar quantile hesabinda birden cok kez sayilir.
Yeni bir yayina yonelik kapsama garantisi icin degisebilirlik birimi yayin (DOI)
olmalidir. Bu betik, Stage 10 ile BIREBIR ayni kalibrasyon/test ayrimi uzerinde
uc kurulumu karsilastirir:

  M0  Marginal (v0.3.x referansi): kayit-duzeyi skorlar.
  V1  Tek-temsilci alt-orneklem: her kalibrasyon DOI'sinden rastgele 1 kayit
      secilir (R tekrar), conformal quantile tekrarlarin medyanidir. Kalibrasyon
      dagilimi yayin-duzeyinde degisebilir hale gelir.
  V2  DOI-esit-agirlikli quantile: her kayit 1/n_DOI agirligiyla sayilir;
      sonlu-orneklem duzeltmesi etkin yayin sayisi G uzerinden yapilir
      (yaklasiklik; metinde belirtilir).

Kapsama, conformal-test uzerinde hem kayit-agirlikli hem yayin-agirlikli
(once DOI ici ortalama, sonra DOI'ler arasi ortalama) raporlanir.

Girdi : data/processed/model_ready_dataset.csv
        outputs/v4/best_model_holdout_predictions.csv (Stage 3, out-of-sample)
Cikti : outputs/conformal/group_conformal_results.json
        outputs/conformal/group_conformal_ozet.csv
Calistirma: python scripts/18_group_conformal.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.validation.splits import holdout_split
from perovskite_ml.utils.manifest import write_manifest

CAL_FRAC = 0.5   # Stage 10 ile ayni
CAL_SEED = 7     # Stage 10 ile ayni
SUB_REPS = 500   # V1 tekrar sayisi
SUB_SEED = 42
COVERAGES = (0.90, 0.80)


def conformal_quantile(scores, coverage):
    n = len(scores)
    k = min(int(np.ceil((n + 1) * coverage)), n)
    return float(np.sort(scores)[k - 1])


def weighted_quantile_effG(scores, weights, coverage, G):
    """DOI-esit-agirlikli ampirik quantile; sonlu-orneklem duzeltmesi etkin
    yayin sayisi G uzerinden: hedef seviye ceil((G+1)*cov)/G (<=1'e kirpilir)."""
    lvl = min(1.0, np.ceil((G + 1) * coverage) / G)
    order = np.argsort(scores)
    s, w = np.asarray(scores)[order], np.asarray(weights)[order]
    cw = np.cumsum(w) / np.sum(w)
    idx = int(np.searchsorted(cw, lvl, side="left"))
    return float(s[min(idx, len(s) - 1)])


def coverage_stats(err, half, g):
    inside = (err <= half).astype(float)
    rec = float(inside.mean())
    pub = float(pd.Series(inside).groupby(pd.Series(g)).mean().mean())
    return round(rec, 4), round(pub, 4)


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    seed = cfg["seed"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    outroot = ROOT / cfg["paths"]["outputs_dir"]
    outdir = outroot / "conformal"; outdir.mkdir(parents=True, exist_ok=True)

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values
    groups = M[cfg["group_col"]].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[cfg["target"], cfg["group_col"]])
    tr, ho = holdout_split(X, y, groups, test_size=0.2, seed=seed)
    assert set(groups[tr]).isdisjoint(set(groups[ho])), "DOI sizinti kontrolu basarisiz!"

    csv = pd.read_csv(outroot / "v4" / "best_model_holdout_predictions.csv")
    assert len(csv) == len(ho), "Holdout tahmin dosyasi bolmeyle hizali degil."
    yhat_ho, ytrue_ho, g_ho = csv["y_pred"].values, y[ho], groups[ho]

    cal_rel, cte_rel = next(GroupShuffleSplit(n_splits=1, test_size=CAL_FRAC,
                                              random_state=CAL_SEED).split(X.iloc[ho], y[ho], g_ho))
    assert set(g_ho[cal_rel]).isdisjoint(set(g_ho[cte_rel])), "Kalibrasyon/test DOI cakismasi!"

    res_cal = np.abs(ytrue_ho[cal_rel] - yhat_ho[cal_rel])
    g_cal = g_ho[cal_rel]
    err_cte = np.abs(ytrue_ho[cte_rel] - yhat_ho[cte_rel])
    g_cte = g_ho[cte_rel]

    cal_dois = pd.unique(g_cal)
    G = len(cal_dois)
    n_per_doi = pd.Series(res_cal).groupby(pd.Series(g_cal)).size()
    w_cal = np.array([1.0 / n_per_doi[d] for d in g_cal])

    rng = np.random.default_rng(SUB_SEED)
    doi_indices = {d: np.flatnonzero(g_cal == d) for d in cal_dois}

    results = {"n_calibration": int(len(res_cal)), "n_conformal_test": int(len(err_cte)),
               "n_calibration_dois": G, "n_test_dois": int(len(pd.unique(g_cte))),
               "sub_reps": SUB_REPS, "methods": {}}
    rows = []
    for cov in COVERAGES:
        # M0 — marginal (v0.3.x)
        q0 = conformal_quantile(res_cal, cov)
        r0, p0 = coverage_stats(err_cte, q0, g_cte)

        # V1 — tek-temsilci alt-orneklem
        qs = []
        for _ in range(SUB_REPS):
            pick = [idx[rng.integers(len(idx))] for idx in doi_indices.values()]
            qs.append(conformal_quantile(res_cal[np.array(pick)], cov))
        q1 = float(np.median(qs))
        q1_iqr = (round(float(np.percentile(qs, 25)), 3), round(float(np.percentile(qs, 75)), 3))
        r1, p1 = coverage_stats(err_cte, q1, g_cte)

        # V2 — DOI-esit-agirlikli quantile
        q2 = weighted_quantile_effG(res_cal, w_cal, cov, G)
        r2c, p2 = coverage_stats(err_cte, q2, g_cte)

        for name, q, rc, pc, extra in [
            ("M0_marginal", q0, r0, p0, None),
            ("V1_tek_temsilci", q1, r1, p1, {"quantile_iqr": q1_iqr}),
            ("V2_doi_agirlikli", q2, r2c, p2, None),
        ]:
            entry = dict(coverage_target=cov, half_width=round(q, 3),
                         coverage_record_weighted=rc, coverage_publication_weighted=pc)
            if extra:
                entry.update(extra)
            results["methods"].setdefault(name, []).append(entry)
            rows.append(dict(yontem=name, hedef=cov, yari_genislik=round(q, 3),
                             kapsama_kayit=rc, kapsama_yayin=pc))
            print(f"[{name}] hedef {cov:.0%}: yari-genislik ±{q:.2f} | "
                  f"kapsama kayit={rc:.3f} yayin={pc:.3f}")

    with open(outdir / "group_conformal_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(outdir / "group_conformal_ozet.csv", index=False)

    write_manifest(outroot / "manifests", "stage18_group_conformal", cfg,
                   metrics={}, outputs=["outputs/conformal/group_conformal_results.json"])
    print("\nTamam. Cikti: outputs/conformal/group_conformal_results.json / _ozet.csv")


if __name__ == "__main__":
    main()
