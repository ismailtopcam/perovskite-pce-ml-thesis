"""Stage 26 — Modele alinmayan alti sentez oncesi alanin basarima katkisi (duyarlilik analizi).

Tez ozellik kumesi 77 sutundur. Perovskit kalinligi, katki maddeleri, antisolvent, biriktirme
yontemi, arka kontak ve alttas alanlari modele girmez (Bolum 3.2; doluluklari icin bkz.
alan_doluluk_denetimi.py). Bu betik, bu alanlar ozellik kumesine eklenseydi DOI-grup CV
basariminin ne olacagini olcer. Birincil model, 77 ozellik ve tezin baslik sayilari DEGISMEZ.

Girdi ve hizalama
  Ham dosya ve temiz kume alan_doluluk_denetimi.py ile ayni kurulur (clean_rows +
  vectorize_dataframe; ham satirlar bu indeksle secilir). Ozellik matrisi model_ready'dir.
  Denetim 1: iki tablo ayni sirada olmali (ham JV_default_PCE = model-hazir hedef, butun
  satirlarda). Tutmuyorsa betik durur.

Bolme ve model
  Katlar: kat_atamasi(groups, 5, atama="kayit") — depoda kayitli arsiv atamasi. Kayit yoksa
  betik durur. Model: registry.build_models(config tohumu)["CatBoost"]. Butun varyantlarda
  ayni katlar ve ayni model.
  Denetim 2: taban varyanti (V0) outputs/v4/model_comparison_groupkfold.csv icindeki CatBoost
  satirini vermeli. Tutmuyorsa betik durur; oteki varyantlar kosulmaz.

Yeni ozelliklerin kodlanmasi (on isleme istatistikleri hatta oldugu gibi temiz kumenin
tamamindan hesaplanir; hedef kullanilmaz)
  Perovskite_thickness : metindeki butun sayilar okunur, en buyugu alinir (hattin tavlama
      ayristiricisi: feature_builder._first_number). Eksikler medyanla doldurulur; ayrica
      kalinlik_eksik bayragi.
  Bes kategorik alan   : deger kirpilir. Bos, "unknown" ve "none" (kucuk harfe cevrilmis tam
      eslesme) tek bir "bilinmiyor" kategorisine gider. Bilinen degerlerin en sik N'i
      (config: features.onehot_top_n) kendi sutununu alir, kalani "diger" sutununda toplanir.
      Esit sayili degerler deger metnine gore siralanir (ortamdan bagimsiz).
      Not: hattin _onehot_topn yardimcisi kullanilmaz; o, bosu "unknown" adli siradan bir
      kategori sayar ve "Unknown" / "none" degerlerini ayri tutar.

Varyantlar (her biri bes kat)
  V0  taban: 77 ozellik
  V1  V0 + kalinlik (deger ve eksiklik bayragi)
  V2  V0 + dusuk doluluklu uc alan: kalinlik, katki maddeleri, antisolvent
  V3  V0 + yuksek doluluklu uc alan: biriktirme yontemi, arka kontak, alttas
  V4  V0 + alti alanin tamami
  V5  V0 + yalnizca alti "bildirilmemis" bayragi (kalinlik_eksik ve bes alanin "bilinmiyor"
      sutunu). Amac: V4'teki farkin ne kadarinin alanin degerinden, ne kadarinin degerin
      bildirilmemis olmasindan geldigini ayirmak.

Olcutler : kat basina R2, MAE, RMSE (models.evaluate.metrics); ortalama ve ss (ddof = 0).
           V0'a gore kat bazinda eslestirilmis R2 ve MAE farki (varyant - V0); farkin
           ortalamasi; kac katta pozitif. p degeri hesaplanmaz.

Girdi : data/raw/Perovskite_database_content_all_data.csv, data/processed/model_ready_dataset.csv,
        outputs/v4/model_comparison_groupkfold.csv
Cikti : outputs/robustness/alinmayan_alan_katkisi.json, .csv
        outputs/manifests/manifest_stage26_alinmayan_alan_katkisi.json
Calistirma: python scripts/26_alinmayan_alan_katkisi.py
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.data.clean_data import clean_rows
from perovskite_ml.features.composition import vectorize_dataframe
from perovskite_ml.features.feature_builder import _first_number
from perovskite_ml.models.evaluate import metrics
from perovskite_ml.models.registry import build_models
from perovskite_ml.utils.manifest import write_manifest
from perovskite_ml.validation.splits import kat_atamasi, kat_parmak_izi, kayitli_kat_atamasi

N_KAT = 5
KALINLIK = "Perovskite_thickness"
KATEGORIK = [  # (ham sutun, sutun on eki, grup)
    ("Perovskite_additives_compounds", "katki", "dusuk"),
    ("Perovskite_deposition_quenching_media", "antisolvent", "dusuk"),
    ("Perovskite_deposition_procedure", "yontem", "yuksek"),
    ("Backcontact_stack_sequence", "arkakontak", "yuksek"),
    ("Substrate_stack_sequence", "alttas", "yuksek"),
]
BILINMEYEN = ("unknown", "none")
ARSIV_CV = "outputs/v4/model_comparison_groupkfold.csv"
MODEL_READY = "data/processed/model_ready_dataset.csv"
OLCUTLER = ["R2", "MAE", "RMSE"]
TOLERANS = 1e-9


def _sha256(yol):
    h = hashlib.sha256()
    with open(ROOT / yol, "rb") as f:
        while parca := f.read(1 << 20):
            h.update(parca)
    return h.hexdigest()


def kategorik_kodla(s, onek, top_n):
    """Bir ham alani one-hot sutunlara cevirir. (sutun sozlugu, kodlama ozeti) dondurur."""
    metin = s.fillna("").astype(str).str.strip()
    bilinmiyor = ((metin == "") | metin.str.lower().isin(BILINMEYEN)).to_numpy()
    sayim = metin[~bilinmiyor].value_counts()
    sirali = sorted(((str(v), int(n)) for v, n in sayim.items()), key=lambda vn: (-vn[1], vn[0]))
    ust = sirali[:top_n]
    deger = metin.to_numpy()
    sutunlar, bilgi = {}, []
    ayrilmis = {f"{onek}_diger", f"{onek}_bilinmiyor"}
    for i, (v, n) in enumerate(ust, 1):
        ad = f"{onek}_" + re.sub(r"[^0-9A-Za-z_]", "_", v)
        if ad in sutunlar or ad in ayrilmis:
            ad = f"{ad}_{i}"
        sutunlar[ad] = (deger == v) & ~bilinmiyor
        bilgi.append({"sutun": ad, "deger": v, "kayit": n})
    ust_degerler = {v for v, _ in ust}
    diger = ~bilinmiyor & ~np.isin(deger, list(ust_degerler))
    if diger.any():
        sutunlar[f"{onek}_diger"] = diger
    if bilinmiyor.any():
        sutunlar[f"{onek}_bilinmiyor"] = bilinmiyor
    ozet = {"on_ek": onek, "n_bilinmiyor": int(bilinmiyor.sum()), "n_bilinen": int((~bilinmiyor).sum()),
            "bilinen_farkli_deger": int(len(sirali)), "ust_sutunlar": bilgi,
            "diger_kayit": int(diger.sum()), "diger_farkli_deger": int(max(len(sirali) - top_n, 0)),
            "sinirda_esit_sayi": bool(len(sirali) > top_n and sirali[top_n - 1][1] == sirali[top_n][1]),
            "sutun_sayisi": int(len(sutunlar))}
    return sutunlar, ozet


def cv(model, X, y, bolmeler):
    kat = {o: [] for o in OLCUTLER}
    for egitim, dogrulama in bolmeler:
        m = clone(model)
        m.fit(X.iloc[egitim], y[egitim])
        mt = metrics(y[dogrulama], m.predict(X.iloc[dogrulama]))
        for o in OLCUTLER:
            kat[o].append(float(mt[o]))
    return kat


def main():
    t0 = time.time()
    cfg = load_config(str(ROOT / "config.yaml"))
    outroot = ROOT / cfg["paths"]["outputs_dir"]
    outdir = outroot / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)
    top_n = int(cfg["features"]["onehot_top_n"])
    hedef, grup_sutunu = cfg["target"], cfg["group_col"]

    # --- girdi ve hizalama ---
    ham = pd.read_csv(ROOT / cfg["paths"]["raw_csv"], encoding="utf-8-sig", low_memory=False)
    temiz_satirlar, _ = clean_rows(ham, cfg)
    vek = vectorize_dataframe(temiz_satirlar, cfg)
    temiz = ham.loc[vek.index]
    M = pd.read_csv(ROOT / MODEL_READY, low_memory=False)
    y = M[hedef].to_numpy(dtype=float)
    groups = M[grup_sutunu].fillna("unknown_doi").astype(str).to_numpy()
    X0 = M.drop(columns=[hedef, grup_sutunu])
    print(f"[stage26] ham {len(ham)}, temiz {len(temiz)}, model-hazir {M.shape[0]} x {X0.shape[1]}", flush=True)
    if len(ham) != 43398 or len(temiz) != 41485 or len(M) != 41485 or X0.shape[1] != 77:
        sys.exit("DUR: kayit / ozellik sayilari beklenenle (43.398 / 41.485 / 77) ayni degil.")
    ham_pce = pd.to_numeric(temiz[hedef], errors="coerce").to_numpy(dtype=float)
    en_buyuk_fark = float(np.nanmax(np.abs(ham_pce - y))) if not np.isnan(ham_pce).any() else float("nan")
    if not (en_buyuk_fark <= TOLERANS):
        sys.exit(f"DUR (denetim 1): ham ve model-hazir tablolar ayni sirada degil (en buyuk PCE farki {en_buyuk_fark}).")
    print(f"[stage26] denetim 1 tuttu: ham PCE = model-hazir hedef, {len(y)} satirda (en buyuk fark {en_buyuk_fark:g})",
          flush=True)

    # --- bolme ve model ---
    if kayitli_kat_atamasi(groups, N_KAT) is None:
        sys.exit("DUR: bu gruplar icin kayitli kat atamasi yok (kat_atamalari/).")
    kat = kat_atamasi(groups, N_KAT, atama="kayit")
    bolmeler = [(np.flatnonzero(kat != k), np.flatnonzero(kat == k)) for k in range(N_KAT)]
    for egitim, dogrulama in bolmeler:
        assert set(groups[egitim]).isdisjoint(groups[dogrulama]), "grup bolmesi ayrik degil"
    model = build_models(cfg["seed"])["CatBoost"]

    # --- taban (V0) ve denetim 2 ---
    t1 = time.time()
    v0 = cv(model, X0, y, bolmeler)
    arsiv = pd.read_csv(ROOT / ARSIV_CV).set_index("model").loc["CatBoost"]
    taban = {f"{o}_{ad}": float(fn(v0[o])) for o in OLCUTLER for ad, fn in (("mean", np.mean), ("std", np.std))}
    sapma = {k: abs(taban[k] - float(arsiv[k])) for k in taban}
    print(f"[stage26] V0: R2 {taban['R2_mean']:.6f} +/- {taban['R2_std']:.6f}  (arsiv {float(arsiv['R2_mean']):.6f} "
          f"+/- {float(arsiv['R2_std']):.6f}); en buyuk sapma {max(sapma.values()):.2e}  ({time.time() - t1:.0f} s)",
          flush=True)
    if max(sapma.values()) > TOLERANS:
        sys.exit(f"DUR (denetim 2): taban varyanti arsivdeki CatBoost satirini vermiyor: {sapma}")
    print("[stage26] denetim 2 tuttu: V0 = arsiv CatBoost satiri", flush=True)

    # --- yeni ozellikler ---
    kal = temiz[KALINLIK].map(_first_number)
    kal_eksik = kal.isna().to_numpy()
    medyan = float(kal.median())
    bloklar = {"kalinlik": pd.DataFrame({"kalinlik": kal.fillna(medyan).to_numpy(dtype=float),
                                         "kalinlik_eksik": kal_eksik.astype(int)})}
    kodlama = {KALINLIK: {"kural": "metindeki sayilarin en buyugu (feature_builder._first_number); eksik -> medyan",
                          "n_sayisal": int((~kal_eksik).sum()), "n_eksik": int(kal_eksik.sum()),
                          "medyan": medyan, "sutunlar": ["kalinlik", "kalinlik_eksik"]}}
    for alan, onek, grup in KATEGORIK:
        sutunlar, ozet = kategorik_kodla(temiz[alan], onek, top_n)
        bloklar[onek] = pd.DataFrame(sutunlar)
        kodlama[alan] = {"grup": f"{grup} doluluklu", **ozet}
    cakisan = [c for b in bloklar.values() for c in b.columns if c in X0.columns]
    tum = [c for b in bloklar.values() for c in b.columns]
    if cakisan or len(tum) != len(set(tum)):
        sys.exit(f"DUR: yeni sutun adlari cakisiyor: {cakisan}")

    bayraklar = pd.concat([bloklar["kalinlik"][["kalinlik_eksik"]]]
                          + [bloklar[onek][[f"{onek}_bilinmiyor"]] for _, onek, _ in KATEGORIK
                             if f"{onek}_bilinmiyor" in bloklar[onek].columns], axis=1)
    dusuk = ["kalinlik"] + [onek for _, onek, g in KATEGORIK if g == "dusuk"]
    yuksek = [onek for _, onek, g in KATEGORIK if g == "yuksek"]
    varyantlar = [
        ("V1", "V0 + kalinlik (deger ve eksiklik bayragi)", [bloklar["kalinlik"]]),
        ("V2", "V0 + dusuk doluluklu uc alan: kalinlik, katki maddeleri, antisolvent", [bloklar[b] for b in dusuk]),
        ("V3", "V0 + yuksek doluluklu uc alan: biriktirme yontemi, arka kontak, alttas", [bloklar[b] for b in yuksek]),
        ("V4", "V0 + alti alanin tamami", [bloklar[b] for b in dusuk + yuksek]),
        ("V5", "V0 + yalnizca alti 'bildirilmemis' bayragi", [bayraklar]),
    ]

    def kayit(kod, aciklama, X, katlar, eklenen):
        r = {"aciklama": aciklama, "n_ozellik": int(X.shape[1]), "eklenen_sutun_sayisi": int(len(eklenen)),
             "eklenen_sutunlar": list(eklenen), "katlar": katlar,
             "ozet": {f"{o}_{ad}": float(fn(katlar[o])) for o in OLCUTLER
                      for ad, fn in (("ortalama", np.mean), ("ss_ddof0", np.std))}}
        if kod != "V0":
            f = {}
            for o in ("R2", "MAE"):
                d = np.asarray(katlar[o]) - np.asarray(v0[o])
                f[f"{o}_kat_farklari"] = [float(v) for v in d]
                f[f"{o}_fark_ortalama"] = float(d.mean())
                f[f"{o}_pozitif_kat"] = int((d > 0).sum())
                f[f"{o}_negatif_kat"] = int((d < 0).sum())
            r["v0_farki"] = f
        return r

    sonuc = {"V0": kayit("V0", "taban: 77 ozellik", X0, v0, [])}
    for kod, aciklama, parcalar in varyantlar:
        t1 = time.time()
        ek = pd.concat(parcalar, axis=1)
        X = pd.concat([X0, ek], axis=1)
        sonuc[kod] = kayit(kod, aciklama, X, cv(model, X, y, bolmeler), ek.columns)
        o, f = sonuc[kod]["ozet"], sonuc[kod]["v0_farki"]
        print(f"[stage26] {kod}: {X.shape[1]} ozellik  R2 {o['R2_ortalama']:.4f} +/- {o['R2_ss_ddof0']:.4f}  "
              f"MAE {o['MAE_ortalama']:.4f}  RMSE {o['RMSE_ortalama']:.4f} | V0'a gore R2 {f['R2_fark_ortalama']:+.4f} "
              f"({f['R2_pozitif_kat']}/{N_KAT} kat pozitif)  MAE {f['MAE_fark_ortalama']:+.4f}  "
              f"({time.time() - t1:.0f} s)", flush=True)

    girdiler = [cfg["paths"]["raw_csv"], MODEL_READY, ARSIV_CV]
    res = {
        "yontem": {
            "amac": "duyarlilik analizi; birincil model ve 77 ozellik degismez",
            "model": "CatBoost (registry.build_models; model tohumu config tohumu)",
            "n_kayit": int(len(M)), "n_grup": int(len(np.unique(groups))), "n_kat": N_KAT,
            "kat_atamasi": "kat_atamasi(groups, 5, atama='kayit'); depoda kayitli arsiv atamasi",
            "kat_atamasi_kimligi": kat_parmak_izi(groups, N_KAT),
            "kat_boyutlari": [int(v) for v in np.bincount(kat, minlength=N_KAT)],
            "on_isleme": "medyan ve en sik kategoriler temiz kumenin tamamindan hesaplanir (hattaki gibi); hedef kullanilmaz",
            "kategorik_kural": f"kirp; bos / 'unknown' / 'none' -> bilinmiyor; bilinen degerlerin en sik {top_n}'i ayri "
                               "sutun, kalani diger; esit sayida deger metnine gore sira",
            "onehot_top_n": top_n,
            "fark_tanimi": "varyant - V0, kat bazinda eslestirilmis; ss ddof = 0; p degeri hesaplanmaz",
        },
        "denetimler": {
            "hizalama": {"tuttu": True, "n_satir": int(len(y)), "en_buyuk_PCE_farki": en_buyuk_fark},
            "taban": {"tuttu": True, "arsiv_dosyasi": ARSIV_CV,
                      "arsiv": {k: float(arsiv[k]) for k in taban}, "bu_kosu": taban,
                      "en_buyuk_mutlak_sapma": float(max(sapma.values()))},
        },
        "kodlama": kodlama,
        "varyantlar": sonuc,
    }
    (outdir / "alinmayan_alan_katkisi.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    satirlar = [{"varyant": kod, "kat": k + 1, "n_ozellik": r["n_ozellik"],
                 **{o: r["katlar"][o][k] for o in OLCUTLER}}
                for kod, r in sonuc.items() for k in range(N_KAT)]
    pd.DataFrame(satirlar).to_csv(outdir / "alinmayan_alan_katkisi.csv", index=False, lineterminator="\n")

    write_manifest(outroot / "manifests", "stage26_alinmayan_alan_katkisi", cfg,
                   metrics={**{f"{kod}_R2_ortalama": round(r["ozet"]["R2_ortalama"], 6) for kod, r in sonuc.items()},
                            **{f"{kod}_n_ozellik": r["n_ozellik"] for kod, r in sonuc.items()},
                            "kat_atamasi_kimligi": kat_parmak_izi(groups, N_KAT),
                            "girdi_sha256": {g: _sha256(g) for g in girdiler}},
                   outputs=["outputs/robustness/alinmayan_alan_katkisi.json",
                            "outputs/robustness/alinmayan_alan_katkisi.csv"])

    print("\n================= STAGE 26 =================")
    for kod, r in sonuc.items():
        o = r["ozet"]
        ek = "" if kod == "V0" else (f"  dR2 {r['v0_farki']['R2_fark_ortalama']:+.4f} "
                                     f"({r['v0_farki']['R2_pozitif_kat']}/{N_KAT})")
        print(f"  {kod}  {r['n_ozellik']:4d} ozellik  R2 {o['R2_ortalama']:.4f} +/- {o['R2_ss_ddof0']:.4f}  "
              f"MAE {o['MAE_ortalama']:.4f}  RMSE {o['RMSE_ortalama']:.4f}{ek}")
    print(f"Tamam ({time.time() - t0:.0f} s).")


if __name__ == "__main__":
    main()
