# -*- coding: utf-8 -*-
"""Kat atamasi denetimi — GroupKFold atamasinin ortama duyarliligi ve raporlanan sayilara etkisi.

Neden: sklearn GroupKFold, gruplari katlara dagitirken boyuta gore siralar. sklearn < 1.9 bu
siralamayi KARARSIZ yapar; esit boyutlu gruplarin sirasi kutuphane surumune ve islemcinin
komut setine (NumPy'nin AVX2 siralama yolu) gore degisir. Bu veride DOI gruplarinin neredeyse
tamami boyutunu baska bir grupla paylastigindan, ayni kod ve ayni tohumla farkli ortamlarda
FARKLI katlar olusur. Tezdeki 0,410 (arsiv) ile 0,412 (ikinci ortam) arasindaki farkin kaynagi budur.

Bu denetim sunlari olcer:
  (1) Yapi: grup sayisi, boyutunu baska grupla paylasan grup sayisi.
  (2) Atamalar: depoda kayitli arsiv atamasi, kararli atama, bu ortamin sklearn atamasi;
      aralarinda kat degistiren kayit payi. Istege bagli islemci denetimi (AVX2 yolu kapali).
  (3) CatBoost kat R2: kayitli arsiv atamasiyla (arsivdeki degerlerle karsilastirilir) ve kararli atamayla.
  (4) CatBoost–HistGBM eslestirilmis karsilastirmasi: arsiv katlarinda, BU ortamin HistGBM'i ile.
      (HistGBM sklearn surumune duyarlidir; arsivdeki p degeri surumler arasinda tasinmaz.)
  (5) Zenginlestirmenin bag duyarliligi: holdout tahminlerinde esit degerler varken ilk-k kesiminin
      siralamaya gore alabilecegi en dusuk / en yuksek deger.

Girdi : data/processed/model_ready_dataset.csv, kat_atamalari/,
        outputs/ek_denetimler/paired_model_comparison.json, outputs/v4/best_model_holdout_predictions.csv,
        outputs/screening/enrichment.csv
Cikti : outputs/robustness/kat_atamasi_denetimi.json
Calistirma (repo kokunden): python kat_atamasi_denetimi.py
"""
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.registry import build_models
from perovskite_ml.models.tuning import grouped_cv_r2_katlar
from perovskite_ml.validation.splits import (_dagit, _grup_indeksi, _sklearn_atama, kararli_kat_atamasi,
                                             kat_parmak_izi, kayitli_kat_atamasi)

ESIKLER = [18, 20, 22]
K_ORANLARI = [0.05, 0.10, 0.20]


def ozet(a):
    """Atamanin kisa kimligi (kat numarasi dizisinin sha256'si)."""
    return hashlib.sha256("".join(str(int(v)) for v in a).encode("ascii")).hexdigest()[:12]


def varsayilan_siralama_atamasi(groups, n_splits=5):
    """sklearn < 1.9 GroupKFold ile ayni kural: np.argsort'un varsayilan (kararsiz) siralamasi.
    Sonuc NumPy'nin bu islemcide sectigi siralama yoluna baglidir."""
    _, gi = _grup_indeksi(groups)
    return _dagit(gi, np.argsort(np.bincount(gi))[::-1], n_splits)


def degisen_pay(a, b, n_splits=5):
    """Iki atama arasinda kat degistiren kayit payi: (dogrudan, kat numaralari en iyi eslestirildiginde)."""
    dogrudan = float((a != b).mean())
    en_iyi = max(float((np.asarray(p)[a] == b).mean()) for p in itertools.permutations(range(n_splits)))
    return round(dogrudan, 6), round(1.0 - en_iyi, 6)


def veri(cfg):
    tgt, gcol = cfg["target"], cfg["group_col"]
    M = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "model_ready_dataset.csv", low_memory=False)
    return M.drop(columns=[tgt, gcol]), M[tgt].values, M[gcol].fillna("u").astype(str).values


def eslestirilmis(a, b):
    from scipy import stats
    a, b = np.asarray(a), np.asarray(b)
    f = a - b
    return dict(kat_farklari=[round(float(v), 6) for v in f], fark_ortalama=round(float(f.mean()), 6),
                fark_std=round(float(f.std(ddof=1)), 6), ilk_onde=int((f > 0).sum()), kat_sayisi=int(len(f)),
                eslestirilmis_t_p=round(float(stats.ttest_rel(a, b).pvalue), 6),
                wilcoxon_p=round(float(stats.wilcoxon(a, b).pvalue), 6))


def bag_duyarliligi(y_true, y_pred):
    """Ilk-k kesiminde esit tahminler varken isabet sayisinin alabilecegi en dusuk / en yuksek deger."""
    n = len(y_true)
    satirlar = []
    for esik in ESIKLER:
        poz = y_true >= esik
        taban = float(poz.mean())
        for kf in K_ORANLARI:
            k = max(1, int(kf * n))                      # screening.enrichment_table ile ayni
            sinir = np.sort(y_pred)[::-1][k - 1]
            ustte, bagli = y_pred > sinir, y_pred == sinir
            alinacak = k - int(ustte.sum())
            kesin = int(poz[ustte].sum())
            bag_poz = int(poz[bagli].sum())
            en_az = kesin + max(0, alinacak - (int(bagli.sum()) - bag_poz))
            en_cok = kesin + min(alinacak, bag_poz)
            satirlar.append(dict(esik=esik, k_oran=kf, k=k, sinirdaki_esit_tahmin=int(bagli.sum()),
                                 bunlardan_alinan=alinacak, isabet_en_az=en_az, isabet_en_cok=en_cok,
                                 zenginlestirme_en_az=round(en_az / k / taban, 6),
                                 zenginlestirme_en_cok=round(en_cok / k / taban, 6)))
    return satirlar


def islemci_denetimi():
    """Ayni NumPy ile AVX2 siralama yolu kapatildiginda varsayilan siralamanin verdigi atama.
    Yalnizca bu islemci AVX2 destekliyorsa anlamlidir; desteklenmiyorsa atlanir."""
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES="X86_V3", PYTHONUTF8="1")
    try:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--yalniz-atama"], cwd=str(ROOT),
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=600)
        if r.returncode != 0:
            return dict(calisti=False, neden=(r.stderr.strip().splitlines() or ["bilinmiyor"])[-1][:200])
        return dict(calisti=True, **json.loads(r.stdout.strip().splitlines()[-1]))
    except Exception as e:                               # denetim istege baglidir; ana sonucu durdurmaz
        return dict(calisti=False, neden=f"{type(e).__name__}: {e}"[:200])


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    X, y, g = veri(cfg)

    if "--yalniz-atama" in sys.argv:                     # islemci_denetimi alt sureci
        print(json.dumps(dict(varsayilan_siralama=ozet(varsayilan_siralama_atamasi(g)),
                              kararli=ozet(kararli_kat_atamasi(g)))))
        return

    import catboost, scipy, sklearn
    seed = cfg["seed"]

    # (1) Yapi
    ug, gi = _grup_indeksi(g)
    boyut = np.bincount(gi)
    ayni_boyutlu = int((np.bincount(boyut)[boyut] > 1).sum())
    yapi = dict(n_kayit=int(len(g)), n_grup=int(len(ug)), boyutunu_paylasan_grup=ayni_boyutlu,
                farkli_boyut_sayisi=int(len(np.unique(boyut))), kimlik=kat_parmak_izi(g, 5))
    print("[yapi]", yapi)

    # (2) Atamalar
    arsiv = kayitli_kat_atamasi(g, 5)
    if arsiv is None:
        raise SystemExit("Birincil yapilandirma icin kayitli kat atamasi yok (kat_atamalari/). "
                         "Arsiv ortaminda PEROVSKITE_KAT_KAYIT=1 ile bir arsiv betigi kosulmali.")
    kararli = kararli_kat_atamasi(g, 5)
    bu_ortam = _sklearn_atama(g, 5)
    varsayilan = varsayilan_siralama_atamasi(g, 5)
    d, e = degisen_pay(arsiv, kararli)
    atamalar = dict(
        arsiv_kayitli=dict(ozet=ozet(arsiv), kat_boyutlari=[int(v) for v in np.bincount(arsiv, minlength=5)]),
        kararli=dict(ozet=ozet(kararli), kat_boyutlari=[int(v) for v in np.bincount(kararli, minlength=5)]),
        arsiv_kararli_kat_degistiren_pay=d, arsiv_kararli_kat_degistiren_pay_en_iyi_eslesme=e,
        bu_ortam_sklearn=dict(ozet=ozet(bu_ortam), arsivle_ayni=bool(np.array_equal(bu_ortam, arsiv)),
                              kararliyla_ayni=bool(np.array_equal(bu_ortam, kararli))),
        bu_islemci_varsayilan_siralama=dict(ozet=ozet(varsayilan),
                                            arsivle_ayni=bool(np.array_equal(varsayilan, arsiv))),
        avx2_yolu_kapali=islemci_denetimi())
    k = atamalar["avx2_yolu_kapali"]
    if k.get("calisti"):
        k["arsivle_ayni"] = k["varsayilan_siralama"] == ozet(arsiv)
        k["kararli_degismedi"] = k["kararli"] == ozet(kararli)
    print("[atamalar]", json.dumps(atamalar, ensure_ascii=False))

    # (3) CatBoost kat R2
    modeller = build_models(seed)
    cb = modeller["CatBoost"]
    hg = modeller[[m for m in modeller if m.lower().startswith("hist")][0]]
    cb_arsiv = grouped_cv_r2_katlar(cb, X, y, g, 5, atama="kayit")
    cb_kararli = grouped_cv_r2_katlar(cb, X, y, g, 5, atama="kararli")
    ref = json.loads((ROOT / cfg["paths"]["outputs_dir"] / "ek_denetimler" / "paired_model_comparison.json")
                     .read_text(encoding="utf-8"))
    catboost_sonuc = dict(
        arsiv_katlari=dict(kat_R2=[round(v, 6) for v in cb_arsiv], ortalama=round(float(np.mean(cb_arsiv)), 6),
                           std=round(float(np.std(cb_arsiv)), 6),
                           arsiv_degerlerinden_en_buyuk_sapma=float(np.max(np.abs(
                               np.array(cb_arsiv) - np.array(ref["folds_catboost_R2"]))))),
        kararli_katlar=dict(kat_R2=[round(v, 6) for v in cb_kararli], ortalama=round(float(np.mean(cb_kararli)), 6),
                            std=round(float(np.std(cb_kararli)), 6)))
    print("[CatBoost]", json.dumps(catboost_sonuc, ensure_ascii=False))

    # (4) CatBoost – HistGBM, arsiv katlarinda, bu ortamin HistGBM'i ile
    hg_arsiv = grouped_cv_r2_katlar(hg, X, y, g, 5, atama="kayit")
    karsilastirma = dict(
        histgbm_kat_R2=[round(v, 6) for v in hg_arsiv],
        histgbm_arsiv_degerlerinden_en_buyuk_sapma=round(float(np.max(np.abs(
            np.array(hg_arsiv) - np.array(ref["folds_histgbm_R2"])))), 6),
        bu_ortam=eslestirilmis(cb_arsiv, hg_arsiv),
        arsiv=dict(fark_ortalama=round(ref["mean_diff_R2"], 6), fark_std=round(ref["std_diff_R2"], 6),
                   ilk_onde=int(sum(a > b for a, b in zip(ref["folds_catboost_R2"], ref["folds_histgbm_R2"]))),
                   kat_sayisi=5, eslestirilmis_t_p=round(ref["paired_t_p"], 6), wilcoxon_p=round(ref["wilcoxon_p"], 6)),
        not_="fark = CatBoost − HistGBM (ayni katlar). Arsiv satiri outputs/ek_denetimler/paired_model_comparison.json.")
    print("[CatBoost−HistGBM]", json.dumps(karsilastirma, ensure_ascii=False))

    # (5) Zenginlestirmenin bag duyarliligi
    P = pd.read_csv(ROOT / cfg["paths"]["outputs_dir"] / "v4" / "best_model_holdout_predictions.csv")
    E = pd.read_csv(ROOT / cfg["paths"]["outputs_dir"] / "screening" / "enrichment.csv")
    bag = bag_duyarliligi(P["y_true"].values, P["y_pred"].values)
    for s in bag:
        r = E[(E.esik == s["esik"]) & np.isclose(E.k_frac, s["k_oran"])].iloc[0]
        s["raporlanan"] = round(float(r["zenginlestirme"]), 6)
    zengin = dict(n_test=int(len(P)), benzersiz_tahmin=int(P["y_pred"].nunique()), satirlar=bag)
    print(pd.DataFrame(bag).to_string(index=False))

    res = dict(yapi=yapi, atamalar=atamalar, catboost=catboost_sonuc, catboost_histgbm=karsilastirma,
               zenginlestirme_bag_duyarliligi=zengin,
               ortam=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                          scikit_learn=sklearn.__version__, scipy=scipy.__version__, catboost=catboost.__version__),
               not_="Atama blogundaki 'bu_ortam', 'bu_islemci' ve 'avx2_yolu_kapali' alanlari kosulan makineye "
                    "baglidir; digerleri kayitli atamaya ve kararli siralamaya dayanir.")
    out = ROOT / cfg["paths"]["outputs_dir"] / "robustness" / "kat_atamasi_denetimi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Cikti:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
