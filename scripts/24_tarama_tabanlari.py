"""Stage 24 — Tarama faydasi icin ek tabanlar: daha yalin siralamalar ayni holdout'ta ne verir?

Tez Bolum 4.6 zenginlesmeyi yalnizca rastgele secime gore verir (PCE >= 18, ilk %5: 4,21x;
Spearman 0,610). Bu betik ayni holdout kayitlarini dort siralamayla puanlar:

  birincil : commit'li holdout tahminleri (outputs/v4/best_model_holdout_predictions.csv;
             model yeniden egitilmez)
  taban 1  : Ridge, 77 ozellik (registry.build_models yapilandirmasi), egitim bolmesinde egitilir
  taban 2  : CatBoost, yalnizca kompozisyon (Deney C'nin ilk basamagindaki 14 ozellik),
             ayni hiperparametreler
  taban 3  : model yok. Egitim bolmesinde (mimari, ETL, HTL) uclusunun ortalama PCE'si;
             egitimde gorulmeyen ucluye egitim ortalamasi. Uclu, aday_uyumluluk_filtresi.py
             ile ayni bicimde model_ready'deki arch_* / ETL_* / HTL_* one-hot sutunlarindan
             okunur (her blokta tam bir sutun 1'dir).

Bolme hattin holdout bolmesidir (holdout_split, tohum 42); holdout kayitlari commit'li
tahmin dosyasiyla ayni degilse betik durur.

Olcutler : Spearman; enrichment_table ile PCE >= 18, 20, 22 esiklerinde ilk %5, %10, %20.
Belirsizlik: tarama_bootstrap.py ile ayni DOI-kumeli bootstrap (B = 1.000, tohum 42; ayni
           yeniden ornekleme dizisi). Her siralama icin %95 aralik; birincil model ile her
           taban arasindaki eslestirilmis fark (Spearman; PCE >= 18 ilk %5) icin %95 aralik.
Baglar    : enrichment_table tahminleri np.argsort ile siralar; ayni tahmin degerini tasiyan
           kayitlarin sirasi tanimsizdir. Her siralama icin benzersiz deger sayisi, ilk-k
           sinirinda esit degerli kayit sayisi ve baglar rastgele bozuldugunda beklenen
           zenginlesme ayrica yazilir.
Denetim  : birincil model satiri 4,21x ve 0,610 vermelidir; vermezse betik cikti yazmadan durur.

Girdi : data/processed/model_ready_dataset.csv, outputs/v4/best_model_holdout_predictions.csv
Cikti : outputs/robustness/tarama_tabanlari.json
        outputs/manifests/manifest_stage24_tarama_tabanlari.json
Calistirma: python scripts/24_tarama_tabanlari.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.registry import build_models
from perovskite_ml.simulation.screening import enrichment_table
from perovskite_ml.utils.manifest import write_manifest
from perovskite_ml.validation.splits import holdout_split

ESIKLER = [18, 20, 22]
K_ORANLARI = [0.05, 0.10, 0.20]
B, BOOT_TOHUM = 1000, 42
GIRDILER = ["data/processed/model_ready_dataset.csv", "outputs/v4/best_model_holdout_predictions.csv"]
FARK_OLCUTLERI = ["spearman", "E18_top5"]


def _sha256(yol):
    h = hashlib.sha256()
    with open(ROOT / yol, "rb") as f:
        while parca := f.read(1 << 20):
            h.update(parca)
    return h.hexdigest()


def olcutler(y_true, y_pred):
    out = {"spearman": float(spearmanr(y_pred, y_true).correlation)}
    for r in enrichment_table(y_true, y_pred, ESIKLER, K_ORANLARI):
        out[f"E{r['esik']}_top{int(round(r['k_frac'] * 100))}"] = float(r["zenginlestirme"])
    return out


def bag_ozeti(y_true, y_pred):
    """Baglarin ilk-k secimine etkisi: sinirdaki esit degerli kayit sayisi ve baglar rastgele
    bozuldugunda beklenen zenginlesme (sinirin ustundekiler kesin, kalan yer bag grubundan orantili)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    out = {"benzersiz_tahmin_sayisi": int(len(np.unique(y_pred))), "k": {}}
    sirali = np.sort(y_pred)[::-1]
    for kf in K_ORANLARI:
        k = max(1, int(kf * n))
        sinir = sirali[k - 1]
        ustte, esit = y_pred > sinir, y_pred == sinir
        bos = k - int(ustte.sum())
        d = {"k": k, "sinirda_esit_kayit": int(esit.sum()), "bag_grubundan_alinan": bos}
        for esik in ESIKLER:
            poz = y_true >= esik
            beklenen = (poz & ustte).sum() + bos * (poz & esit).sum() / esit.sum()
            d[f"E{esik}_bag_beklenen"] = float((beklenen / k) / poz.mean())
        out["k"][f"top{int(round(kf * 100))}"] = d
    return out


def ci95(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    t0 = time.time()
    cfg = load_config(str(ROOT / "config.yaml"))
    seed = cfg["seed"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    outroot = ROOT / cfg["paths"]["outputs_dir"]

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values
    groups = M[cfg["group_col"]].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[cfg["target"], cfg["group_col"]])
    tr, te = holdout_split(X, y, groups, test_size=0.2, seed=seed)
    assert set(groups[tr]).isdisjoint(set(groups[te])), "DOI sizinti kontrolu basarisiz!"

    P = pd.read_csv(outroot / "v4" / "best_model_holdout_predictions.csv")
    ayni = (len(P) == len(te) == 8355 and np.array_equal(P["DOI"].astype(str).values, groups[te])
            and np.array_equal(P["y_true"].values, y[te]))
    if not ayni:
        sys.exit("[stage24] DUR: holdout kayitlari commit'li tahmin dosyasiyla ayni degil "
                 f"(dosya {len(P)} kayit, bolme {len(te)} kayit).")
    yt = y[te]
    print(f"[stage24] egitim {len(tr)} kayit, holdout {len(te)} kayit (commit'li tahmin dosyasiyla ayni)", flush=True)

    modeller = build_models(seed)
    siralama = {"birincil_catboost_77": P["y_pred"].values}

    ridge = clone(modeller["Ridge"]).fit(X.iloc[tr], y[tr])
    siralama["taban1_ridge_77"] = ridge.predict(X.iloc[te])

    komp = [c for c in X.columns if c.split("_")[0] in ("A", "B", "X")]
    assert len(komp) == 14, f"kompozisyon sutunu {len(komp)} (14 beklenir)"
    cb = clone(modeller["CatBoost"]).fit(X[komp].iloc[tr], y[tr])
    siralama["taban2_catboost_kompozisyon_14"] = cb.predict(X[komp].iloc[te])

    bloklar = {on: [c for c in X.columns if c.startswith(on)] for on in ("arch_", "ETL_", "HTL_")}
    for on, cols in bloklar.items():
        assert (X[cols].sum(axis=1) == 1).all(), f"{on} blogu tek-sicak degil"
    uclu = pd.Series(list(zip(*[X[cols].idxmax(axis=1).values for cols in bloklar.values()])), index=X.index)
    egitim_ort = float(y[tr].mean())
    uclu_ort = pd.Series(y[tr]).groupby(uclu.iloc[tr].values).mean()
    t3 = uclu.iloc[te].map(uclu_ort)
    gorulmeyen = int(t3.isna().sum())
    siralama["taban3_uclu_ortalamasi"] = t3.fillna(egitim_ort).values

    nokta = {ad: olcutler(yt, yp) for ad, yp in siralama.items()}
    b = nokta["birincil_catboost_77"]
    print(f"[stage24] birincil: Spearman {b['spearman']:.4f}, E18 ilk %5 {b['E18_top5']:.4f}", flush=True)
    if round(b["E18_top5"], 2) != 4.21 or round(b["spearman"], 3) != 0.610:
        sys.exit("[stage24] DUR: birincil model satiri 4,21x / 0,610 vermedi.")

    # DOI-kumeli bootstrap: tarama_bootstrap.py ile ayni kumeler, ayni tohum, ayni cekilis dizisi
    doi = P["DOI"].fillna("u").astype(str).values
    kumeler = pd.Series(np.arange(len(P))).groupby(doi).apply(lambda s: s.values)
    rng = np.random.default_rng(BOOT_TOHUM)
    boot = {ad: {k: [] for k in nokta[ad]} for ad in siralama}
    for i in range(B):
        sec = rng.integers(0, len(kumeler), len(kumeler))
        idx = np.concatenate([kumeler.iloc[j] for j in sec])
        for ad, yp in siralama.items():
            for k, v in olcutler(yt[idx], yp[idx]).items():
                boot[ad][k].append(v)
        if (i + 1) % 250 == 0:
            print(f"[stage24] bootstrap {i + 1}/{B} ({time.time() - t0:.0f} s)", flush=True)

    sonuc = {}
    for ad, yp in siralama.items():
        sonuc[ad] = {"olcutler": {k: {"nokta": nokta[ad][k], "ci95": ci95(boot[ad][k])} for k in nokta[ad]},
                     "baglar": bag_ozeti(yt, yp)}
    farklar = {}
    for ad in siralama:
        if ad == "birincil_catboost_77":
            continue
        farklar[ad] = {}
        for k in FARK_OLCUTLERI:
            d = np.asarray(boot["birincil_catboost_77"][k]) - np.asarray(boot[ad][k])
            farklar[ad][k] = {"nokta": nokta["birincil_catboost_77"][k] - nokta[ad][k], "ci95": ci95(d),
                              "pozitif_orani": float(np.mean(d > 0))}

    taban_oran = {f"PCE{e}": float((yt >= e).mean()) for e in ESIKLER}
    res = {
        "yontem": {
            "bolme": f"holdout_split (GroupShuffleSplit, test_size 0.2, tohum {seed}); commit'li tahmin dosyasiyla ayni",
            "n_egitim": int(len(tr)), "n_holdout": int(len(te)), "n_holdout_yayin": int(len(kumeler)),
            "esikler": ESIKLER, "k_oranlari": K_ORANLARI,
            "zenginlestirme": "perovskite_ml.simulation.screening.enrichment_table (scripts/07 ile ayni)",
            "bootstrap": f"DOI-kumeli, B={B}, tohum {BOOT_TOHUM}; tarama_bootstrap.py ile ayni cekilis dizisi; "
                         "%95 aralik 2,5 ve 97,5 yuzdelikler",
            "fark": "birincil - taban; ayni bootstrap orneklerinde eslestirilmis",
            "taban3": {"uclu_tanimi": "arch_* / ETL_* / HTL_* one-hot sutunlarinda 1 olan sutun",
                       "egitimde_farkli_uclu": int(len(uclu_ort)), "egitim_ortalamasi": egitim_ort,
                       "holdoutta_egitimde_gorulmeyen_uclu_kayit": gorulmeyen},
            "taban2_ozellikler": komp,
        },
        "taban_oran": taban_oran,
        "siralamalar": sonuc,
        "eslestirilmis_fark_birincil_eksi_taban": farklar,
    }
    out = outroot / "robustness" / "tarama_tabanlari.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    write_manifest(outroot / "manifests", "stage24_tarama_tabanlari", cfg,
                   metrics={"bootstrap_B": B, "bootstrap_tohumu": BOOT_TOHUM,
                            **{f"{ad}_spearman": round(nokta[ad]["spearman"], 6) for ad in siralama},
                            **{f"{ad}_E18_top5": round(nokta[ad]["E18_top5"], 6) for ad in siralama},
                            "girdi_sha256": {g: _sha256(g) for g in GIRDILER}},
                   outputs=["outputs/robustness/tarama_tabanlari.json"])

    print("\n================= STAGE 24 =================")
    for ad in siralama:
        o = sonuc[ad]["olcutler"]
        print(f"  {ad:32s} Spearman {o['spearman']['nokta']:.4f} [{o['spearman']['ci95'][0]:.3f}; "
              f"{o['spearman']['ci95'][1]:.3f}]  E18 ilk %5 {o['E18_top5']['nokta']:.3f} "
              f"[{o['E18_top5']['ci95'][0]:.3f}; {o['E18_top5']['ci95'][1]:.3f}]  "
              f"benzersiz tahmin {sonuc[ad]['baglar']['benzersiz_tahmin_sayisi']}")
    for ad, f in farklar.items():
        print(f"  fark (birincil - {ad}): " + "; ".join(
            f"{k} {v['nokta']:+.4f} [{v['ci95'][0]:+.3f}; {v['ci95'][1]:+.3f}]" for k, v in f.items()))
    print(f"Tamam ({time.time() - t0:.0f} s).")


if __name__ == "__main__":
    main()
