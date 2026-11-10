# -*- coding: utf-8 -*-
"""Tarama yonu denetimi — hedefin olcum protokolu heterojenligi model icin ne anlama geliyor?

Tez Bolum 3.3: hedef JV_default_PCE, kayit bazinda farkli olcum protokollerinden gelir
(ham veride %88,1 ters tarama, %8,0 stabilize, %1,7 ileri tarama). Bolum 5.7 bunun
sistematik oldugunu da olcer (ters tarama medyanda 0,66 puan yuksek). Ama tarama yonunun
BASARIMA etkisi olculmemistir. Bu denetim iki soruyu yanitlar:

  (1) Hedefi tek protokole sabitlersek (yalnizca ters tarama) basarim degisir mi?
  (2) Tarama yonu bir KOVARYAT olarak eklenirse basarim artar mi?
      (Alan olcum ANINA ait protokol bilgisidir, olcum SONUCU degildir; bu nedenle
       hedef sizintisi tanimina girmez — bkz. tez Bolum 2.3 ve isik_siddeti_denetimi.py.)

Protokol: 5 kat DOI-grup CV, CatBoost — scripts/11 ve isik_siddeti_denetimi.py ile ayni.
Temiz kume cekirdek hattin clean_rows islevi ile ham CSV'den satir indeksiyle kurulur.

Girdi : data/raw/...csv, data/processed/model_ready_dataset.csv
Cikti : outputs/robustness/tarama_yonu_denetimi.json
Calistirma (repo kokunden): python tarama_yonu_denetimi.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.data.load_data import load_raw, _resolve_raw_path
from perovskite_ml.data.clean_data import clean_rows
from perovskite_ml.models.registry import build_models
from perovskite_ml.models.tuning import grouped_cv_r2

SCAN = "JV_default_PCE_scan_direction"


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    tgt, gcol, seed = cfg["target"], cfg["group_col"], cfg["seed"]

    raw_df = load_raw(cfg)
    clean, _ = clean_rows(raw_df, cfg)
    S = pd.read_csv(_resolve_raw_path(cfg), usecols=[SCAN, tgt, gcol], low_memory=False)
    assert len(S) == len(raw_df), "ham CSV satir sayisi tutarsiz"

    M = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "model_ready_dataset.csv", low_memory=False)
    assert len(M) == len(clean), f"model-hazir {len(M)} != temiz {len(clean)}"
    assert np.allclose(M[tgt].values, clean[tgt].values), "satir eslesmesi bozuk"

    sd = S.loc[clean.index, SCAN].fillna("(bos)").astype(str).str.strip()
    dagilim = {str(k): int(v) for k, v in sd.value_counts().items()}
    pay = {k: round(v / len(sd), 4) for k, v in dagilim.items()}
    print("[tarama yonu] temiz kumede dagilim:", dagilim)

    # PCE ortalamasi protokole gore
    pce_by = clean.groupby(sd.values)[tgt].agg(["count", "mean", "median"]).round(3)
    print(pce_by.to_string())

    y_all = M[tgt].values
    g_all = M[gcol].fillna("u").astype(str).values
    X_all = M.drop(columns=[tgt, gcol])
    model = build_models(seed)["CatBoost"]

    rows = []

    def run(label, mask, X=None):
        Xs = (X if X is not None else X_all)[mask] if mask is not None else (X if X is not None else X_all)
        ys = y_all[mask] if mask is not None else y_all
        gs = g_all[mask] if mask is not None else g_all
        m, s = grouped_cv_r2(model, Xs, ys, gs, 5)
        rows.append(dict(kurulum=label, n=int(len(ys)), R2_mean=round(m, 6), R2_std=round(s, 6)))
        print(f"[CV] {label:38s} n={len(ys):6,} R2={m:.4f} ± {s:.4f}", flush=True)

    rev = (sd.str.lower() == "reversed").values
    stab = (sd.str.lower().str.startswith("stab")).values

    run("tam kume (taban cizgisi)", None)
    run("yalnizca ters tarama", rev)
    run("ters + stabilize (ileri haric)", rev | stab)

    # tarama yonu KOVARYAT olarak
    dummies = pd.get_dummies(sd.values, prefix="scan").astype(int)
    dummies.index = X_all.index
    X_cov = pd.concat([X_all, dummies], axis=1)
    run(f"tam kume + tarama yonu kovaryati (+{dummies.shape[1]})", None, X=X_cov)

    res = dict(dagilim=dagilim, pay=pay,
               pce_protokole_gore=pce_by.to_dict(),
               cv=rows,
               not_="Protokol: 5 kat DOI-grup CV, CatBoost; isik_siddeti_denetimi.py ile ayni.")
    out = ROOT / cfg["paths"]["outputs_dir"] / "robustness" / "tarama_yonu_denetimi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print("Cikti:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
