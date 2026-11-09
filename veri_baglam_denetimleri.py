# -*- coding: utf-8 -*-
"""Veri baglami denetimleri — tezde anilan betimsel sayilarin tek kaynagi.

Temiz kume, cekirdek hattin clean_rows islevi ham CSV'ye yeniden uygulanarak satir indeksiyle
kurulur (isik_siddeti_denetimi.py ile ayni yontem). Bolumler:
  1) yil_kapsami        : Ref_publication_date'e gore temiz kumenin yil dagilimi (Bolum 3.1, 5.4, 5.6)
  2) tarama_yonu        : JV_default_PCE_scan_direction dagilimi (ham, n ile; Bolum 3.3) ve ayni cihazda
                          ters-ileri / ters-stabilize PCE farklari (temiz kume; Bolum 5.7)
  3) boyutluluk_yapi    : boyutluluk ve ABX3 yapi bayraklari (temiz kume; Bolum 3.5, 1.6)
  4) kursunsuz          : B-site'ta Pb icermeyen kayitlarin Sn / diger B-site dagilimi (Bolum 4.4)
  5) bandgap_medyanlari : model-hazir kumede (band gap eksik olmayan) B/X bilesimine gore band gap
                          medyanlari (Bolum 3.12, 5.4)
  6) tavlama_baglami    : model-hazir kumede (tavlama eksik olmayan) tavlama sicakligi/suresi medyanlari;
                          genel, MA ve FA agirlikli + SnO2-np ailesi (Bolum 3.12)
Girdi : data/raw/...csv, data/processed/model_ready_dataset.csv
Cikti : outputs/robustness/veri_baglam_denetimleri.json (+ konsol ozeti)
Calistirma (repo kokunden): python veri_baglam_denetimleri.py
"""
import sys, json, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.data.load_data import load_raw, _resolve_raw_path
from perovskite_ml.data.clean_data import clean_rows

RAWCOLS = ["Ref_publication_date", "JV_default_PCE", "JV_default_PCE_scan_direction", "JV_reverse_scan_PCE",
           "JV_forward_scan_PCE", "Stabilised_performance_PCE", "Perovskite_dimension_0D", "Perovskite_dimension_2D",
           "Perovskite_dimension_2D3D_mixture", "Perovskite_dimension_3D", "Perovskite_dimension_3D_with_2D_capping_layer",
           "Perovskite_composition_perovskite_ABC3_structure", "Perovskite_composition_perovskite_inspired_structure",
           "Perovskite_composition_b_ions"]


def truthy(s):
    return s.astype(str).str.strip().str.upper().eq("TRUE")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def q(s):
    s = s.dropna()
    return {"n": int(len(s)), "medyan": round(float(s.median()), 3) if len(s) else None,
            "p10": round(float(s.quantile(.10)), 3) if len(s) else None, "p90": round(float(s.quantile(.90)), 3) if len(s) else None}


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    clean, _ = clean_rows(load_raw(cfg), cfg)
    R = pd.read_csv(_resolve_raw_path(cfg), usecols=RAWCOLS, low_memory=False)
    C = R.loc[clean.index]
    res = {"temiz_n": int(len(C)), "ham_n": int(len(R))}

    # 1) yil kapsami
    yr = C["Ref_publication_date"].astype(str).str.extract(r"((?:19|20)\d{2})")[0].astype(float)
    res["yil_kapsami"] = {"yil_bilinen": int(yr.notna().sum()), "en_eski": int(yr.min()), "en_yeni": int(yr.max()),
                          "le_2019": int((yr <= 2019).sum()), "y2020": int((yr == 2020).sum()), "ge_2021": int((yr >= 2021).sum()),
                          "le_2020_yuzde": round(100 * float((yr <= 2020).sum()) / float(yr.notna().sum()), 1),
                          "le_2019_yuzde": round(100 * float((yr <= 2019).sum()) / float(yr.notna().sum()), 1)}

    # 2) tarama yonu
    d = R["JV_default_PCE_scan_direction"].fillna("(bos)").astype(str).str.strip()
    res["tarama_yonu"] = {"ham_dagilim": {k: int(v) for k, v in d.value_counts().items()}}
    rev, fwd, stab = num(C["JV_reverse_scan_PCE"]), num(C["JV_forward_scan_PCE"]), num(C["Stabilised_performance_PCE"])
    for lab, other in (("ters_eksi_ileri", fwd), ("ters_eksi_stabilize", stab)):
        m = rev.notna() & other.notna()
        diff = (rev - other)[m]
        res["tarama_yonu"][lab] = {"n": int(m.sum()), "medyan": round(float(diff.median()), 2), "ortalama": round(float(diff.mean()), 2),
                                   "ters_yuksek_yuzde": round(100 * float((diff > 0).mean()), 1)}

    # 3) boyutluluk / yapi
    low = truthy(C["Perovskite_dimension_0D"]) | truthy(C["Perovskite_dimension_2D"]) | truthy(C["Perovskite_dimension_2D3D_mixture"])
    abc3_false = ~truthy(C["Perovskite_composition_perovskite_ABC3_structure"])
    inspired = truthy(C["Perovskite_composition_perovskite_inspired_structure"])
    res["boyutluluk_yapi"] = {"dusuk_boyutlu_0D_2D_2D3D": int(low.sum()), "dusuk_boyutlu_yuzde": round(100 * float(low.mean()), 1),
                              "ABC3_disi": int(abc3_false.sum()), "ABC3_disi_yuzde": round(100 * float(abc3_false.mean()), 1),
                              "ABC3_disi_perovskit_esinli": int((abc3_false & inspired).sum()),
                              "birlikte": int((low | abc3_false).sum()), "birlikte_yuzde": round(100 * float((low | abc3_false).mean()), 1),
                              "3D_capping": int(truthy(C["Perovskite_dimension_3D_with_2D_capping_layer"]).sum())}

    # 4..6) model-hazir kume
    M = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "model_ready_dataset.csv", low_memory=False)
    assert len(M) == len(C)
    pce = M[cfg["target"]]
    nopb = M["B_Pb"] <= 1e-9
    res["kursunsuz"] = {"n": int(nopb.sum()), "Sn_iceren": int((nopb & (M["B_Sn"] > 0)).sum()),
                        "Sn_icermeyen_diger_B": int((nopb & (M["B_Sn"] <= 0)).sum()),
                        "medyan_PCE_Sn_iceren": round(float(pce[nopb & (M["B_Sn"] > 0)].median()), 2),
                        "medyan_PCE_Sn_icermeyen": round(float(pce[nopb & (M["B_Sn"] <= 0)].median()), 2),
                        "medyan_PCE_tum": round(float(pce.median()), 2)}
    # 4b) kursunsuz kayitlarin ham B-site iyonlari ve ABX3 bayragi (temiz kume, ham alanlardan)
    bion = C["Perovskite_composition_b_ions"].fillna("").astype(str)
    toks = bion.apply(lambda s: [t.strip() for t in re.split(r"[;|]", s) if t.strip()])
    has_pb, has_sn = toks.apply(lambda L: "Pb" in L), toks.apply(lambda L: "Sn" in L)
    nopb_raw = (~has_pb) & bion.str.strip().ne("")
    nosn = nopb_raw & ~has_sn
    abc3_ok = truthy(C["Perovskite_composition_perovskite_ABC3_structure"])
    bisb = toks.apply(lambda L: ("Bi" in L) or ("Sb" in L))
    res["kursunsuz"].update({
        "ham_kursunsuz_n": int(nopb_raw.sum()), "ham_Sn_iceren": int((nopb_raw & has_sn).sum()),
        "ham_Sn_icermeyen": int(nosn.sum()),
        "ham_Sn_icermeyen_ilk_iyonlar": {k: int(v) for k, v in toks[nosn].explode().value_counts().head(8).items()},
        "ham_Sn_icermeyen_Bi_veya_Sb_yuzde": round(100 * float(bisb[nosn].mean()), 1),
        "ABC3_disi_yuzde_Sn_icermeyen": round(100 * float((~abc3_ok[nosn]).mean()), 1),
        "ABC3_disi_yuzde_Sn_iceren": round(100 * float((~abc3_ok[nopb_raw & has_sn]).mean()), 1)})
    bgok = M["band_gap_missing"] == 0
    bg = M["band_gap"].where(bgok)
    pb = (M["B_Pb"] - 1).abs() <= 1e-6
    res["bandgap_medyanlari"] = {
        "tum": q(bg),
        "Pb": q(bg[pb]),
        "Pb_I": q(bg[pb & ((M["X_I"] - 1).abs() <= 1e-6)]),
        "Pb_I0.83_Br0.17": q(bg[pb & ((M["X_Br"] - 0.17).abs() <= 0.03)]),
        "Pb_I0.6_Br0.4": q(bg[pb & ((M["X_Br"] - 0.40).abs() <= 0.03)]),
        "Pb0.5_Sn0.5": q(bg[((M["B_Sn"] - 0.5).abs() <= 0.06)]),
    }
    tok = M["anneal_temp_missing"] == 0
    snc = [c for c in M.columns if c.startswith("ETL_") and "SnO2_np" in c and "___" not in c]
    sn = (M[snc[0]] == 1) if snc else pd.Series(False, index=M.index)
    fa = M["A_FA"] >= 0.75
    ma = (M["A_MA"] - 1).abs() <= 1e-6
    res["tavlama_baglami"] = {
        "etl_kolonu": snc[0] if snc else None,
        "tum": {"sicaklik": q(M["anneal_temp"].where(tok)), "sure": q(M["anneal_time"].where(M.get("anneal_time_missing", 0) == 0))},
        "MA": {"sicaklik": q(M["anneal_temp"].where(tok & ma)), "sure": q(M["anneal_time"].where((M.get("anneal_time_missing", 0) == 0) & ma))},
        "FA_ge_0.75_SnO2np": {"sicaklik": q(M["anneal_temp"].where(tok & fa & sn)),
                              "sure": q(M["anneal_time"].where((M.get("anneal_time_missing", 0) == 0) & fa & sn))},
    }

    out = ROOT / "outputs" / "robustness" / "veri_baglam_denetimleri.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("================ Veri baglami denetimleri ================")
    for k, v in res.items():
        print(f"{k}: {v}")
    print("Cikti:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
