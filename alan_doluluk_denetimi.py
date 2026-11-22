# -*- coding: utf-8 -*-
"""Modele alinmayan sekiz alanin dolulugu — ham kumede (43.398 kayit) ve temiz kumede (41.485 kayit).

Tez ozellik kumesi 77 sutundur; perovskit kalinligi, biriktirme yontemi, antisolvent, katkilar,
arka kontak, alttas ve tasiyici katman kalinliklari modele girmez. Bu denetim, bu alanlarin
veri tabaninda ne olcude dolu oldugunu sayar (hakem okumasi H01).

Her alan icin, ham ve temiz kumede ayri ayri:
  dolu           : bos olmayan kayit orani (NaN ve yalnizca bosluk bos sayilir)
  bilinen        : dolu olup degeri (bosluklar kirpilip kucuk harfe cevrildikten sonra) tam olarak
                   "unknown" ya da "none" OLMAYAN kayit orani
  bilinen_parca  : ayni, ama deger "|", ";" ve ">>" ile parcalara ayrilir; butun parcalari
                   "unknown" / "none" / "nan" / bos olan kayit da bilinmeyen sayilir
  farkli deger   : dolu degerlerde ve bilinen degerlerde farkli deger sayisi

Temiz kume hattin kendi koduyla kurulur: clean_rows + vectorize_dataframe (scripts/04 ile ayni);
temiz kayitlarin ham satirlari bu indeksle secilir.

Karsilastirma: gorev notundaki gecici hesap (ham kume; "Unknown" ve "none" haric dolu orani ve
farkli deger sayisi) GECICI sozlugunde durur; cikti her alan icin farki da yazar.

Girdi : data/raw/Perovskite_database_content_all_data.csv
Cikti : outputs/robustness/alan_doluluk_denetimi.json
        outputs/manifests/manifest_alan_doluluk_denetimi.json
Calistirma (repo kokunden): python alan_doluluk_denetimi.py
Not   : run_thesis_audits.py listesinde degildir.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.data.clean_data import clean_rows
from perovskite_ml.features.composition import vectorize_dataframe
from perovskite_ml.utils.manifest import write_manifest

ALANLAR = ["Perovskite_thickness", "Perovskite_deposition_procedure", "Perovskite_deposition_quenching_media",
           "Perovskite_additives_compounds", "Backcontact_stack_sequence", "Substrate_stack_sequence",
           "ETL_thickness", "HTL_thickness_list"]
BILINMEYEN = ("unknown", "none")
# Gorev notundaki gecici hesap (ham kume): ("Unknown" ve "none" haric dolu orani %, farkli deger sayisi)
GECICI = {"Perovskite_thickness": (31.6, 369), "Perovskite_deposition_procedure": (98.7, 199),
          "Perovskite_deposition_quenching_media": (47.4, 94), "Perovskite_additives_compounds": (32.0, 1016),
          "Backcontact_stack_sequence": (99.1, 290), "Substrate_stack_sequence": (99.1, 194),
          "ETL_thickness": (43.5, 1420), "HTL_thickness_list": (23.4, 439)}


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while parca := f.read(1 << 20):
            h.update(parca)
    return h.hexdigest()


def _butun_parcalar_bilinmeyen(v):
    return all(x.strip().lower() in BILINMEYEN + ("nan", "") for x in re.split(r"\||;|>>", v))


def alan_ozeti(s):
    n = len(s)
    metin = s.fillna("").astype(str).str.strip()
    dolu = s.notna() & (metin != "")
    bilinen = dolu & ~metin.str.lower().isin(BILINMEYEN)
    bilinen_parca = dolu & ~metin.map(_butun_parcalar_bilinmeyen)
    return {"n": int(n),
            "dolu_kayit": int(dolu.sum()), "dolu_oran": float(dolu.mean()),
            "bilinen_kayit": int(bilinen.sum()), "bilinen_oran": float(bilinen.mean()),
            "bilinen_parca_kayit": int(bilinen_parca.sum()), "bilinen_parca_oran": float(bilinen_parca.mean()),
            "farkli_deger_dolu": int(metin[dolu].nunique()),
            "farkli_deger_bilinen": int(metin[bilinen].nunique()),
            "unknown_kayit": int((metin.str.lower() == "unknown").sum()),
            "none_kayit": int((metin.str.lower() == "none").sum())}


def main():
    t0 = time.time()
    cfg = load_config(str(ROOT / "config.yaml"))
    ham_yol = ROOT / cfg["paths"]["raw_csv"]
    ham = pd.read_csv(ham_yol, encoding="utf-8-sig", low_memory=False)
    temiz_satirlar, _ = clean_rows(ham, cfg)
    vek = vectorize_dataframe(temiz_satirlar, cfg)
    temiz = ham.loc[vek.index]
    print(f"ham {len(ham)} kayit, temiz {len(temiz)} kayit")
    if len(ham) != 43398 or len(temiz) != 41485:
        sys.exit("DUR: kayit sayilari beklenenle (43.398 / 41.485) ayni degil.")

    alanlar = {}
    for a in ALANLAR:
        h, t = alan_ozeti(ham[a]), alan_ozeti(temiz[a])
        g_oran, g_farkli = GECICI[a]
        alanlar[a] = {
            "ham": h, "temiz": t,
            "gecici_hesap_ham": {"bilinen_oran_yuzde": g_oran, "farkli_deger": g_farkli},
            "gecici_ile_fark": {
                "bilinen_oran_yuzde_puan": round(100 * h["bilinen_oran"], 1) - g_oran,
                "farkli_deger_bilinen": h["farkli_deger_bilinen"] - g_farkli,
                "farkli_deger_dolu": h["farkli_deger_dolu"] - g_farkli}}
        print(f"{a:40s} ham: dolu %{100 * h['dolu_oran']:.1f}  bilinen %{100 * h['bilinen_oran']:.1f} "
              f"({h['farkli_deger_bilinen']} / dolu {h['farkli_deger_dolu']})  | temiz: dolu %{100 * t['dolu_oran']:.1f}  "
              f"bilinen %{100 * t['bilinen_oran']:.1f} ({t['farkli_deger_bilinen']})  | gecici %{g_oran} ({g_farkli})")

    res = {"yontem": {"n_ham": int(len(ham)), "n_temiz": int(len(temiz)),
                      "temiz_kume": "clean_rows + vectorize_dataframe (hat kodu); ham satirlar bu indeksle secilir",
                      "dolu": "NaN degil ve kirpildiktan sonra bos degil",
                      "bilinen": "dolu ve degerin tamami (kirpilmis, kucuk harf) 'unknown' ya da 'none' degil",
                      "bilinen_parca": "dolu ve '|', ';', '>>' ile ayrilan parcalarin hepsi "
                                       "'unknown' / 'none' / 'nan' / bos degil",
                      "farkli_deger": "kirpilmis metin uzerinde; buyuk-kucuk harf ayrimi korunur"},
           "alanlar": alanlar}
    out = ROOT / "outputs" / "robustness" / "alan_doluluk_denetimi.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "alan_doluluk_denetimi", cfg,
                   metrics={"n_ham": int(len(ham)), "n_temiz": int(len(temiz)),
                            "girdi_sha256": {cfg["paths"]["raw_csv"]: _sha256(ham_yol)}},
                   outputs=["outputs/robustness/alan_doluluk_denetimi.json"])
    print(f"Cikti: {out.relative_to(ROOT)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
