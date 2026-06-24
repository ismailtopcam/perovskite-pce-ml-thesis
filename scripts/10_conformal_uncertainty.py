"""Stage 10 — Tahmin belirsizliginin nicelenmesi (conformal + uygulanabilirlik-alani).

DOI-grup-guvenli tumevarimli (split) conformal tahmin araliklari ve kNN tabanli
uygulanabilirlik-alani analizi. Modeli yeniden egitmez; Stage 3'un urettigi
out-of-sample holdout tahminlerini kullanir (model bu kayitlari gormedi -> sizintisiz).

Girdi : data/processed/model_ready_dataset.csv
        outputs/v4/best_model_holdout_predictions.csv   (Stage 3, out-of-sample)
        outputs/candidates_full/candidate_predictions.csv (Stage 6, istege bagli)
Cikti : outputs/conformal/conformal_results.json
        outputs/conformal/conformal_summary.csv
        outputs/conformal/candidate_intervals.csv
Calistirma: python scripts/10_conformal_uncertainty.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.validation.splits import holdout_split
from perovskite_ml.utils.manifest import write_manifest

# --- conformal/AD ayarlari ---
CAL_FRAC = 0.5     # holdout'un yarisi kalibrasyon, yarisi conformal-test (DOI bazli)
CAL_SEED = 7       # kalibrasyon/test ayrimi icin tohum
N_BINS   = 4       # Mondrian: tahmin-degeri ceyrekleri
KNN_K    = 5       # uygulanabilirlik-alani komsu sayisi
AD_PCT   = 95      # AD esigi: egitim-ici uzakligin yuzdeligi
COVERAGES = (0.90, 0.80)


def conformal_quantile(scores, coverage):
    """Sonlu-orneklem conformal yuzdeligi: ceil((n+1)*coverage)/n. inci en kucuk artik."""
    n = len(scores)
    k = min(int(np.ceil((n + 1) * coverage)), n)
    return float(np.sort(scores)[k - 1])


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    seed = cfg["seed"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    outroot = ROOT / cfg["paths"]["outputs_dir"]
    outdir = outroot / "conformal"; outdir.mkdir(parents=True, exist_ok=True)

    # --- veri + Stage 3 ile birebir ayni holdout bolmesi (sizinti-guvenli) ---
    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values
    groups = M[cfg["group_col"]].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[cfg["target"], cfg["group_col"]])
    tr, ho = holdout_split(X, y, groups, test_size=0.2, seed=seed)
    assert set(groups[tr]).isdisjoint(set(groups[ho])), "DOI sizinti kontrolu basarisiz!"

    # --- out-of-sample holdout tahminleri (Stage 3 cikti'si) ---
    pred_path = outroot / "v4" / "best_model_holdout_predictions.csv"
    csv = pd.read_csv(pred_path)
    assert len(csv) == len(ho), "Holdout tahmin dosyasi bolmeyle hizali degil (boyut)."
    assert np.allclose(y[ho][:200], csv["y_true"].values[:200], atol=1e-6), \
        "Holdout tahminleri yeniden kurulan bolmeyle hizasiz (sira)."
    yhat_ho = csv["y_pred"].values
    ytrue_ho = y[ho]

    # --- holdout'u DOI bazli kalibrasyon + conformal-test'e ayir ---
    g_ho = groups[ho]
    cal_rel, cte_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=CAL_FRAC, random_state=CAL_SEED)
        .split(X.iloc[ho], y[ho], g_ho)
    )
    assert set(g_ho[cal_rel]).isdisjoint(set(g_ho[cte_rel])), "Kalibrasyon/test DOI cakismasi!"
    yhat_cal, yhat_cte = yhat_ho[cal_rel], yhat_ho[cte_rel]
    ytrue_cal, ytrue_cte = ytrue_ho[cal_rel], ytrue_ho[cte_rel]
    res = np.abs(ytrue_cal - yhat_cal)   # nonconformity = mutlak artik
    err = np.abs(ytrue_cte - yhat_cte)

    results = {"n_calibration": int(len(res)), "n_conformal_test": int(len(err))}

    # --- 1) Marginal (kosulsuz) conformal ---
    results["marginal"] = {}
    for cov in COVERAGES:
        q = conformal_quantile(res, cov)
        results["marginal"][str(int(cov * 100))] = {
            "half_width": round(q, 4),
            "width": round(2 * q, 4),
            "empirical_coverage": round(float((err <= q).mean()), 4),
        }

    # --- 2) Mondrian (tahmin-degeri ceyrekleri) conformal, %90 ---
    edges = np.quantile(yhat_cal, np.linspace(0, 1, N_BINS + 1))
    bin_cal = np.clip(np.digitize(yhat_cal, edges[1:-1]), 0, N_BINS - 1)
    bin_cte = np.clip(np.digitize(yhat_cte, edges[1:-1]), 0, N_BINS - 1)
    q_bin = {b: conformal_quantile(res[bin_cal == b], 0.90) for b in range(N_BINS)}
    q_per_test = np.array([q_bin[b] for b in bin_cte])
    results["mondrian_90"] = {
        "empirical_coverage": round(float((err <= q_per_test).mean()), 4),
        "mean_half_width": round(float(q_per_test.mean()), 4),
        "mean_width": round(float(2 * q_per_test.mean()), 4),
        "per_quartile_half_width": {f"Q{b + 1}": round(q_bin[b], 4) for b in range(N_BINS)},
    }

    # --- 3) Uygulanabilirlik-alani (kNN, olcekli ozellik uzayi) ---
    sc = StandardScaler().fit(X.iloc[tr])
    Xtr = sc.transform(X.iloc[tr])
    Xcte = sc.transform(X.iloc[ho[cte_rel]])
    nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(Xtr)
    ad_tr = nn.kneighbors(Xtr)[0][:, 1:KNN_K + 1].mean(axis=1)   # oz-komsu (mesafe 0) haric
    ad_cte = nn.kneighbors(Xcte)[0][:, :KNN_K].mean(axis=1)
    thr = float(np.quantile(ad_tr, AD_PCT / 100.0))
    inside = ad_cte <= thr
    q90 = conformal_quantile(res, 0.90)
    results["applicability_domain"] = {
        "knn_k": KNN_K, "threshold_pct": AD_PCT, "threshold": round(thr, 4),
        "frac_inside": round(float(inside.mean()), 4),
        "mae_inside": round(float(err[inside].mean()), 4),
        "mae_outside": round(float(err[~inside].mean()), 4),
        "coverage90_inside": round(float((err[inside] <= q90).mean()), 4),
        "coverage90_outside": round(float((err[~inside] <= q90).mean()), 4),
    }

    # --- 4) Adaylara uygula (marginal %90 + Mondrian ust-ceyrek) ---
    cand_path = outroot / "candidates_full" / "candidate_predictions.csv"
    if cand_path.exists():
        cand = pd.read_csv(cand_path)
        q_top = q_bin[N_BINS - 1]   # en yuksek tahmin ceyregi
        top = cand.nlargest(5, "Predicted_PCE").copy()
        top["marginal90_low"] = (top["Predicted_PCE"] - q90).round(2)
        top["marginal90_high"] = (top["Predicted_PCE"] + q90).round(2)
        top["mondrian_top_low"] = (top["Predicted_PCE"] - q_top).round(2)
        top["mondrian_top_high"] = (top["Predicted_PCE"] + q_top).round(2)
        top.round(4).to_csv(outdir / "candidate_intervals.csv", index=False)
        results["top_candidate"] = {
            "Predicted_PCE": round(float(top["Predicted_PCE"].iloc[0]), 2),
            "marginal_90": [float(top["marginal90_low"].iloc[0]), float(top["marginal90_high"].iloc[0])],
            "mondrian_top": [float(top["mondrian_top_low"].iloc[0]), float(top["mondrian_top_high"].iloc[0])],
        }

    # --- kaydet ---
    (outdir / "conformal_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"yontem": "Marginal", "hedef_kapsama": "%90",
         "yari_genislik_puan": results["marginal"]["90"]["half_width"],
         "ampirik_kapsama": results["marginal"]["90"]["empirical_coverage"]},
        {"yontem": "Marginal", "hedef_kapsama": "%80",
         "yari_genislik_puan": results["marginal"]["80"]["half_width"],
         "ampirik_kapsama": results["marginal"]["80"]["empirical_coverage"]},
        {"yontem": "Mondrian (ceyrek)", "hedef_kapsama": "%90",
         "yari_genislik_puan": results["mondrian_90"]["mean_half_width"],
         "ampirik_kapsama": results["mondrian_90"]["empirical_coverage"]},
    ]).to_csv(outdir / "conformal_summary.csv", index=False)
    write_manifest(outroot / "manifests", "stage10_conformal", cfg, metrics={
        "marginal90_half_width": results["marginal"]["90"]["half_width"],
        "marginal90_coverage": results["marginal"]["90"]["empirical_coverage"],
        "ad_frac_inside": results["applicability_domain"]["frac_inside"],
    })

    m = results["marginal"]; mo = results["mondrian_90"]; ad = results["applicability_domain"]
    print("\n================= STAGE 10 RESULTS =================")
    print(f"  Kalibrasyon n = {results['n_calibration']}  |  conformal-test n = {results['n_conformal_test']}")
    print(f"  Marginal %90 : +/- {m['90']['half_width']:.2f} puan   (ampirik kapsama {m['90']['empirical_coverage']*100:.1f}%)")
    print(f"  Marginal %80 : +/- {m['80']['half_width']:.2f} puan   (ampirik kapsama {m['80']['empirical_coverage']*100:.1f}%)")
    print(f"  Mondrian %90 : ort. +/- {mo['mean_half_width']:.2f} puan (kapsama {mo['empirical_coverage']*100:.1f}%)  ceyrek: {mo['per_quartile_half_width']}")
    print(f"  AD (kNN k={ad['knn_k']}): test ici {ad['frac_inside']*100:.1f}%  |  MAE ic {ad['mae_inside']:.2f} vs dis {ad['mae_outside']:.2f}")
    print(f"  AD kapsama %90: ic {ad['coverage90_inside']*100:.1f}% vs dis {ad['coverage90_outside']*100:.1f}%  -> mesafe guvenilirligi ayirt etmiyor")
    if "top_candidate" in results:
        tc = results["top_candidate"]
        print(f"  En yuksek aday PCE {tc['Predicted_PCE']:.2f} -> %90 marginal {tc['marginal_90']}  | Mondrian-ust {tc['mondrian_top']}")
    print("===================================================")


if __name__ == "__main__":
    main()
