# -*- coding: utf-8 -*-
"""Birincil basarimin belirsizligi -- yayina (DOI) gore kumeli bootstrap.

Muhurlu holdout kayitlari yayina gore kumelidir (ayni yayinin kardes cihazlari); kayit
duzeyi bootstrap bu bagimliligi yok sayar ve araligi daraltir. Bu denetim, tezde
raporlanan holdout R2, MAE ve RMSE degerleri icin DOI'leri yeniden ornekleyen bir
bootstrap ile %95 araliklari verir (yeniden ornekleme tarama_bootstrap.py ile ayni).
Karsilastirma icin kayit duzeyi (iid) bootstrap araligi da yazilir.

Aralik yalnizca test orneklemesinden gelen belirsizligi yansitir (model sabit); CV'deki
kat sapmasi egitim kumesi degiskenligini de icerdigi icin bu aralik onun yerine gecmez.

Girdi : outputs/v4/best_model_holdout_predictions.csv (DOI, y_true, y_pred; 8.355 kayit)
Cikti : outputs/robustness/birincil_r2_bootstrap.json (+ konsol ozeti)
Calistirma (repo kokunden): python birincil_r2_bootstrap.py   (tez Bolum 4.2)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
B, SEED = 1000, 42


def stats(y_true, y_pred):
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"R2": 1.0 - ss_res / ss_tot,
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def ci95(values):
    return [round(float(np.nanpercentile(values, 2.5)), 6),
            round(float(np.nanpercentile(values, 97.5)), 6)]


def main():
    df = pd.read_csv(ROOT / "outputs" / "v4" / "best_model_holdout_predictions.csv")
    yt, yp = df["y_true"].to_numpy(float), df["y_pred"].to_numpy(float)
    doi = df["DOI"].fillna("u").astype(str).to_numpy()
    n = len(df)
    point = stats(yt, yp)

    # DOI-kumeli bootstrap (tarama_bootstrap.py ile ayni yeniden ornekleme)
    groups = pd.Series(np.arange(n)).groupby(doi).apply(lambda s: s.values)
    keys = list(groups.index)
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in point}
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups.iloc[j] for j in pick])
        for k, v in stats(yt[idx], yp[idx]).items():
            boot[k].append(v)

    # karsilastirma: kayit duzeyi (iid) bootstrap -- yayin icindeki bagimliligi yok sayar
    rng_iid = np.random.default_rng(SEED)
    iid = {k: [] for k in point}
    for _ in range(B):
        idx = rng_iid.integers(0, n, n)
        for k, v in stats(yt[idx], yp[idx]).items():
            iid[k].append(v)

    res = {"yontem": f"DOI-kumeli bootstrap (B={B}, tohum {SEED}); yuzdelik %95 araligi; "
                     "yeniden ornekleme tarama_bootstrap.py ile ayni",
           "n_kayit": int(n), "n_yayin": int(len(keys)),
           "istatistikler": {k: {"nokta": round(point[k], 6),
                                 "ci95_doi_kumeli": ci95(boot[k]),
                                 "ci95_kayit_duzeyi": ci95(iid[k]),
                                 "bootstrap_std_doi_kumeli": round(float(np.std(boot[k], ddof=1)), 6)}
                             for k in point},
           "not_": "Aralik yalnizca test orneklemesinden gelen belirsizligi yansitir (model sabit); "
                   "CV kat sapmasi egitim kumesi degiskenligini de icerir, bu aralik onun yerine gecmez."}
    out = ROOT / "outputs" / "robustness" / "birincil_r2_bootstrap.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("============ Birincil basarim: DOI-kumeli bootstrap ============")
    print(f"Holdout: {n:,} kayit, {len(keys):,} yayin")
    for k, v in res["istatistikler"].items():
        a, b = v["ci95_doi_kumeli"]
        c, d = v["ci95_kayit_duzeyi"]
        print(f"  {k:5s} nokta {v['nokta']:.4f}  DOI-kumeli %95 [{a:.3f}; {b:.3f}]  "
              f"kayit duzeyi %95 [{c:.3f}; {d:.3f}]")
    print("Cikti:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
