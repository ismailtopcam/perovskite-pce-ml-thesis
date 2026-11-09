# -*- coding: utf-8 -*-
"""Tarama faydasinin belirsizligi — yayina (DOI) gore kumeli bootstrap.

Holdout kayitlari yayina gore kumelidir (ayni yayinin kardes cihazlari); kayit duzeyi
guven araliklari bu bagimliligi yok sayar. Bu denetim, Bolum 4.6'daki Spearman
korelasyonu ve ilk-k zenginlestirme carpanlari icin DOI'leri yeniden ornekleyen bir
bootstrap ile %95 araliklari verir. Zenginlestirme tanimi scripts/07 ile aynidir
(perovskite_ml.simulation.screening.enrichment_table).

Girdi : outputs/v4/best_model_holdout_predictions.csv (DOI, y_true, y_pred; 8.355 kayit)
Cikti : outputs/robustness/tarama_bootstrap.json (+ konsol ozeti)
Calistirma (repo kokunden): python tarama_bootstrap.py   (tez Bolum 4.6)
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.simulation.screening import enrichment_table

THRESHOLDS = [18, 20, 22]
K_FRACS = [0.05, 0.10]
B, SEED = 1000, 42


def stats(y_true, y_pred):
    out = {"spearman": float(spearmanr(y_pred, y_true).correlation)}
    for r in enrichment_table(y_true, y_pred, THRESHOLDS, K_FRACS):
        out[f"E{r['esik']}_top{int(round(r['k_frac'] * 100))}"] = float(r["zenginlestirme"])
    return out


def main():
    df = pd.read_csv(ROOT / "outputs" / "v4" / "best_model_holdout_predictions.csv")
    yt, yp, doi = df["y_true"].values, df["y_pred"].values, df["DOI"].fillna("u").astype(str).values
    point = stats(yt, yp)

    # kumelenme ozeti: ilk %5'lik dilim ve pozitif kayitlar kac yayindan geliyor
    order = np.argsort(yp)[::-1]
    k5 = max(1, int(0.05 * len(df)))
    top = order[:k5]
    kume = {"n_kayit": int(len(df)), "n_yayin": int(pd.Series(doi).nunique()),
            "ilk5_kayit": int(k5), "ilk5_yayin": int(pd.Series(doi[top]).nunique()),
            "ilk5_isabet_yayin_PCE18": int(pd.Series(doi[top][yt[top] >= 18]).nunique())}
    for t in THRESHOLDS:
        m = yt >= t
        kume[f"pozitif_PCE{t}"] = {"kayit": int(m.sum()), "yayin": int(pd.Series(doi[m]).nunique())}

    # DOI-kumeli bootstrap
    groups = pd.Series(np.arange(len(df))).groupby(doi).apply(lambda s: s.values)
    keys = list(groups.index)
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in point}
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups.iloc[j] for j in pick])
        s = stats(yt[idx], yp[idx])
        for k, v in s.items():
            boot[k].append(v)
    ci = {k: {"nokta": round(point[k], 4),
              "ci95": [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]}
          for k, v in boot.items()}

    res = {"yontem": f"DOI-kumeli bootstrap (B={B}, tohum {SEED}); zenginlestirme tanimi scripts/07 ile ayni",
           "kumelenme": kume, "istatistikler": ci}
    out = ROOT / "outputs" / "robustness" / "tarama_bootstrap.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("================ Tarama faydasi: DOI-kumeli bootstrap ================")
    print("Kumelenme:", kume)
    for k, v in ci.items():
        print(f"  {k:12s} nokta {v['nokta']:.4f}  %95 [{v['ci95'][0]:.3f}; {v['ci95'][1]:.3f}]")
    print("Cikti:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
