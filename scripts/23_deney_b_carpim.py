"""Stage 23 — Deney B'nin 0,937'si neden 1,00 degil? (carpim ozelligi denetimi)

PCE = (Jsc x Voc x FF) / Pin iliskisi TAM oldugundan, Voc/Jsc/FF girdi verildiginde
R2 ~ 1,00 beklenir; tez Tablo 4.4'te gozlenen 0,937'dir. Okuyucu bu farki "demek ki JV
kolonlari tam sizinti degil" diye okuyabilir ve Deney B'nin gucu zayiflar. Gercek neden
modelleme tarafinda olmalidir: agac topluluklari CARPIMSAL iliskileri eksenlere paralel
bolmelerle ancak yaklasik temsil eder.

Bu betik hipotezi dogrudan sinar: ayni uc degiskenin CARPIMI tek bir turetilmis ozellik
olarak verilirse agac modeli iliskiyi tek bolmede yakalayabilir ve R2 1,00'e yaklasmalidir.
SONUC (9 Kasim 2026): hipotez dogrulanmadi — carpim tek basina verildiginde R2 duser. Acik,
cebirsel ozdesligin veride her kayit icin tutmamasindan gelir; ozdesligin %1 icinde tuttugu
kayitlarda R2 ~ 1'dir. Kaynak dagilimi 'ozdeslik_*' ciktilarinda (tez Bolum 4.3).

Kurulumlar (hepsi 5 kat DOI-grup CV, CatBoost):
  1. sizinti-guvenli (taban)                      -> tez 0,410
  2. + Voc, Jsc, FF ayri ayri                     -> tez 0,937
  3. + yalnizca carpim (Voc*Jsc*FF/100)           -> hipotez: ~1,00
  4. + carpim ve uc degisken birlikte

Girdi : data/raw/...csv
Cikti : outputs/robustness/deney_b_carpim.json
Calistirma: python scripts/23_deney_b_carpim.py
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.data.load_data import load_raw, _resolve_raw_path
from perovskite_ml.data.clean_data import clean_rows
from perovskite_ml.features.composition import vectorize_dataframe
from perovskite_ml.features.feature_builder import build_features
from perovskite_ml.models.registry import build_models
from perovskite_ml.models.tuning import grouped_cv_r2
from perovskite_ml.utils.manifest import write_manifest

LEAK = ["JV_default_Voc", "JV_default_Jsc", "JV_default_FF"]


def main():
    t0 = time.time()
    cfg = load_config(str(ROOT / "config.yaml"))
    seed, target, gcol = cfg["seed"], cfg["target"], cfg["group_col"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)

    # NOT: load_raw yalnizca config'te adi gecen kolonlari dondurur; JV_* kolonlari
    # icin ham CSV'nin tamami okunur (scripts/04 ile ayni yaklasim).
    raw = pd.read_csv(_resolve_raw_path(cfg), encoding="utf-8-sig", low_memory=False)
    cleaned, _ = clean_rows(raw, cfg)
    vec = vectorize_dataframe(cleaned, cfg)
    M = build_features(vec, cfg)
    y = M[target].values.astype(float)
    g = M[gcol].fillna("unknown_doi").astype(str).values
    X_safe = M.drop(columns=[target, gcol]).reset_index(drop=True)

    have = [c for c in LEAK if c in cleaned.columns]
    assert len(have) == 3, f"JV kolonlari eksik: {have}"
    leak = (cleaned.loc[vec.index, have].apply(pd.to_numeric, errors="coerce")
            .fillna(0).reset_index(drop=True))
    leak.columns = have

    # FF veri tabaninda yuzde mi kesir mi? medyanla karar ver (yuzde ise 100'e bol)
    ff_med = float(leak["JV_default_FF"].replace(0, np.nan).median())
    ff = leak["JV_default_FF"] / 100.0 if ff_med > 1.5 else leak["JV_default_FF"]
    carpim = (leak["JV_default_Voc"] * leak["JV_default_Jsc"] * ff).rename("JV_carpim")
    print(f"[stage23] FF medyani {ff_med:.3f} -> {'yuzde, 100e bolundu' if ff_med > 1.5 else 'kesir'}")
    # cebirsel kontrol: carpim PCE'yi ne kadar iyi veriyor?
    ok = carpim.replace(0, np.nan).notna() & (y > 0)
    r_alg = float(np.corrcoef(carpim[ok], y[ok])[0, 1])
    print(f"[stage23] carpim ile PCE korelasyonu: r = {r_alg:.4f} (n={int(ok.sum()):,})")

    model = build_models(seed)["CatBoost"]
    setups = [
        ("1. sizinti-guvenli (taban)", X_safe),
        ("2. + Voc, Jsc, FF ayri ayri", pd.concat([X_safe, leak], axis=1)),
        ("3. + yalnizca carpim", pd.concat([X_safe, carpim], axis=1)),
        ("4. + carpim ve uc degisken", pd.concat([X_safe, leak, carpim], axis=1)),
    ]
    rows = []
    for label, Xi in setups:
        m, s = grouped_cv_r2(model, Xi, y, g, 5, atama="kararli")
        rows.append(dict(kurulum=label, ozellik=int(Xi.shape[1]),
                         R2_mean=round(m, 6), R2_std=round(s, 6)))
        print(f"[CV] {label:32s} ({Xi.shape[1]:3d} oz.) R2={m:.4f} ± {s:.4f}", flush=True)

    # --- Ozdeslik kirilmasi: acik hangi kayitlardan geliyor? (tez Bolum 4.3) ---
    # Ham (doldurulmamis) Voc/Jsc/FF ile carpim; bagil sapma |carpim - PCE| / PCE.
    sd = (cleaned.loc[vec.index, "JV_default_PCE_scan_direction"].fillna("(bos)")
          .astype(str).str.strip().values)
    L = cleaned.loc[vec.index, have].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    ff_r = L["JV_default_FF"] / 100.0 if ff_med > 1.5 else L["JV_default_FF"]
    prod = L["JV_default_Voc"] * L["JV_default_Jsc"] * ff_r
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = ((prod - y) / y).abs()
    ok_all = (prod.notna() & (prod > 0)).values
    yon = {}
    for grp in ("Reversed", "Stabilised", "Forward"):
        m = ok_all & (sd == grp)
        if m.sum() > 20:
            yon[grp] = dict(n=int(m.sum()),
                            r=round(float(np.corrcoef(prod[m], y[m])[0, 1]), 6),
                            bagil_fark_medyan=round(float(rel[m].median()), 6),
                            yuzde1_icinde_pay=round(float((rel[m] < 0.01).mean()), 6))
    print("[stage23] tarama yonune gore ozdeslik:", yon)
    X_leak = pd.concat([X_safe, leak], axis=1)
    subsets = [("tum kayitlar", np.ones(len(y), bool)),
               ("stabilize haric", sd != "Stabilised"),
               ("stabilize + eksik JV haric", (sd != "Stabilised") & L.notna().all(axis=1).values),
               ("ozdeslik %1 icinde", (rel < 0.01).fillna(False).values)]
    ozd_rows = []
    for label, msk in subsets:
        m_, s_ = grouped_cv_r2(model, X_leak[msk], y[msk], g[msk], 5, atama="kararli")
        ozd_rows.append(dict(alt_kume=label, n=int(msk.sum()), R2_mean=round(m_, 6), R2_std=round(s_, 6)))
        print(f"[ozdeslik] {label:28s} n={msk.sum():6,} R2={m_:.4f} ± {s_:.4f}", flush=True)

    res = dict(ff_median_raw=ff_med, carpim_pce_korelasyon=round(r_alg, 6), cv=rows,
               ozdeslik_tarama_yonune_gore=yon, ozdeslik_alt_kume_cv=ozd_rows)
    with open(outdir / "deney_b_carpim.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage23_deney_b_carpim", cfg,
                   metrics={r["kurulum"]: r["R2_mean"] for r in rows},
                   outputs=["outputs/robustness/deney_b_carpim.json"])
    print(f"\nTamam ({time.time() - t0:.0f} s).")


if __name__ == "__main__":
    main()
