# -*- coding: utf-8 -*-
"""Olcum aydinlatma kosulu denetimi — PCE ust ucu ic mekan / dusuk isik olcumu mu?

Veri tabani her JV olcumunun isik kosulunu kaydeder (JV_light_intensity [mW/cm2],
JV_light_source_type, JV_light_spectra). Hat bu alanlari ne filtre ne ozellik olarak
kullanir (olcum anina ait protokol bilgisi). Bu denetim:
  (1) temiz veri kumesinde (41.485) isik siddeti dagilimini,
  (2) PCE > 30 kayitlarinin ve temizlikte %35 ustu diye dislanan kayitlarin isik kosulunu,
  (3) 100 mW/cm2 disi kayitlar dislandiginda birincil basarimin (5 kat DOI-grup CV,
      CatBoost; scripts/11 ile ayni protokol) ne kadar degistigini olcer.
Temiz kume, cekirdek hattin kendi temizleme islevi (clean_rows) ham CSV'ye yeniden
uygulanarak satir indeksiyle kurulur; model-hazir kumeyle satir satir eslesme denetlenir.

Girdi : data/raw/...csv, data/processed/model_ready_dataset.csv
Cikti : outputs/robustness/isik_siddeti_denetimi.json (+ konsol ozeti)
Calistirma (repo kokunden): python isik_siddeti_denetimi.py   (tez Bolum 3.4, 4.2, 4.7, 5.7)
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

LIGHT = ["JV_light_intensity", "JV_light_source_type", "JV_light_spectra"]


def vc(s, n=8):
    return {str(k): int(v) for k, v in s.fillna("(boş)").astype(str).value_counts().head(n).items()}


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    tgt, gcol, seed = cfg["target"], cfg["group_col"], cfg["seed"]

    # --- temiz kume: cekirdek hattin temizleme islevi, ham satir indeksi korunur
    raw_df = load_raw(cfg)
    clean, removed = clean_rows(raw_df, cfg)
    L = pd.read_csv(_resolve_raw_path(cfg), usecols=LIGHT + [tgt, gcol], low_memory=False)
    assert len(L) == len(raw_df), "ham CSV satir sayisi tutarsiz"
    I = pd.to_numeric(L["JV_light_intensity"], errors="coerce")

    C = clean[[tgt, gcol]].copy()
    C["I"] = I.loc[C.index].values
    C["kaynak"] = L.loc[C.index, "JV_light_source_type"].values
    C["spektrum"] = L.loc[C.index, "JV_light_spectra"].values

    # --- model-hazir kumeyle satir satir eslesme
    M = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "model_ready_dataset.csv", low_memory=False)
    assert len(M) == len(C), f"model-hazir {len(M)} != temiz {len(C)}"
    pce_eq = bool(np.allclose(M[tgt].values, C[tgt].values))
    doi_eq = float((M[gcol].fillna("u").astype(str).values == C[gcol].fillna("u").astype(str).values).mean())
    assert pce_eq and doi_eq > 0.999, "satir eslesmesi bozuk"

    i = C["I"]
    dag = {"dolu": int(i.notna().sum()), "esit_100": int((i == 100).sum()), "lt_50": int((i < 50).sum()),
           "50_ile_100_arasi": int(((i >= 50) & (i < 100)).sum()), "gt_100": int((i > 100).sum()),
           "bos": int(i.isna().sum())}

    def kuyruk(th):
        s = C[C[tgt] > th]
        return {"n": int(len(s)), "isik_lt_50": int((s["I"] < 50).sum()), "yayin": int(s[gcol].nunique())}

    hi = C[C[tgt] > 30].sort_values(tgt, ascending=False)
    gt30 = {"n": int(len(hi)), "yayin": int(hi[gcol].nunique()),
            "isik_siddeti_min": float(hi["I"].min()), "isik_siddeti_max": float(hi["I"].max()),
            "kaynak": vc(hi["kaynak"]), "spektrum": vc(hi["spektrum"]),
            "kayitlar": [{"doi": r[gcol], "PCE": round(float(r[tgt]), 2), "isik_mWcm2": float(r["I"]),
                          "kaynak": str(r["kaynak"]), "spektrum": str(r["spektrum"])} for _, r in hi.iterrows()]}

    # --- temizlikte %35 ustu diye dislanan kayitlar
    out_idx = removed.loc[removed["reason"] == "pce_out_of_range", "row_index"]
    P = pd.to_numeric(L[tgt], errors="coerce")
    ex = L.loc[out_idx].assign(PCE=P.loc[out_idx], I=I.loc[out_idx])
    ex35 = ex[ex["PCE"] > 35]
    dislanan = {"n_aralik_disi": int(len(ex)), "n_gt_35": int(len(ex35)), "yayin": int(ex35[gcol].nunique()),
                "isik_siddeti_min": float(ex35["I"].min()) if len(ex35) else None,
                "isik_siddeti_max": float(ex35["I"].max()) if len(ex35) else None,
                "kaynak": vc(ex35["JV_light_source_type"]), "spektrum": vc(ex35["JV_light_spectra"])}

    # --- duyarlilik: scripts/11 ile ayni protokol (5 kat DOI-grup CV, CatBoost)
    y_all = M[tgt].values
    subsets = [("tam", np.ones(len(M), bool)),
               ("isik_lt_50_dislandi", ~(C["I"] < 50).values),
               ("isik_100_disi_dislandi", ~((C["I"].notna()) & (C["I"] != 100)).values)]
    rows = []
    for label, mask in subsets:
        sub = M[mask]
        y = sub[tgt].values
        g = sub[gcol].fillna("u").astype(str).values
        X = sub.drop(columns=[tgt, gcol])
        m, s = grouped_cv_r2(build_models(seed)["CatBoost"], X, y, g, 5)
        rows.append({"alt_kume": label, "n": int(mask.sum()), "dislanan": int((~mask).sum()),
                     "model": "CatBoost", "R2_mean": round(m, 4), "R2_std": round(s, 4)})
        print(f"[duyarlilik] {label}: n={mask.sum()} R2={m:.4f} ± {s:.4f}", flush=True)

    res = {
        "kapsam": "temiz veri kumesi; ham CSV ile clean_rows satir indeksinden eslendi",
        "eslesme": {"n": int(len(C)), "model_hazir_PCE_esit": pce_eq, "model_hazir_DOI_esit_orani": round(doi_eq, 6)},
        "isik_siddeti_mWcm2": dag,
        "isik_kaynagi": vc(C["kaynak"]), "spektrum": vc(C["spektrum"]),
        "ust_kuyruk": {"PCE_gt_25": kuyruk(25), "PCE_gt_28": kuyruk(28), "PCE_gt_30": kuyruk(30)},
        "PCE_gt_30_kayitlari": gt30,
        "temizlikte_dislanan": dislanan,
        "duyarlilik_groupkfold": rows,
    }
    out = ROOT / cfg["paths"]["outputs_dir"] / "robustness" / "isik_siddeti_denetimi.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n================ Isik siddeti denetimi ================")
    print("Isik siddeti (mW/cm2):", dag)
    print("Ust kuyruk:", res["ust_kuyruk"])
    print(f"PCE>30: {gt30['n']} kayit / {gt30['yayin']} yayin; isik {gt30['isik_siddeti_min']}–{gt30['isik_siddeti_max']} mW/cm2; "
          f"kaynak {gt30['kaynak']}; spektrum {gt30['spektrum']}")
    print(f"Temizlikte dislanan >35: {dislanan['n_gt_35']} kayit / {dislanan['yayin']} yayin; "
          f"isik {dislanan['isik_siddeti_min']}–{dislanan['isik_siddeti_max']}")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Cikti:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
