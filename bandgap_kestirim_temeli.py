# -*- coding: utf-8 -*-
"""Band gap degerlerinin kestirim temeli — olcum mu, kompozisyondan atama mi?

Veri tabani her band gap degerinin nasil elde edildigini Perovskite_band_gap_estimation_basis
alaninda kaydeder. Bu denetim temiz veri kumesinde (cekirdek hattin clean_rows islevi ham CSV'ye
yeniden uygulanarak) band gap degeri bulunan kayitlarda bu alanin dagilimini verir
(Bolum 3.2'deki "bilgi statusu" tartismasi; Bolum 4.9'daki band gap ablasyonu).

Girdi : data/raw/...csv
Cikti : outputs/robustness/bandgap_kestirim_temeli.json (+ konsol ozeti)
Calistirma (repo kokunden): python bandgap_kestirim_temeli.py   (tez Bolum 3.2, 4.9)
"""
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.data.load_data import load_raw, _resolve_raw_path
from perovskite_ml.data.clean_data import clean_rows

COL_BG, COL_BASIS = "Perovskite_band_gap", "Perovskite_band_gap_estimation_basis"


def grup(s):
    s = str(s).strip().lower()
    if s in ("", "nan", "unknown"):
        return "belirtilmemis"
    if s == "composition":
        return "kompozisyondan_atama"
    if "literature" in s:
        return "literatur"
    if any(k in s for k in ("absorption", "tauc", "eqe", "ups", "pl", "ellipsometry", "reflect")):
        return "olcum"
    return "diger"


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    clean, _ = clean_rows(load_raw(cfg), cfg)
    R = pd.read_csv(_resolve_raw_path(cfg), usecols=[COL_BG, COL_BASIS], low_memory=False)
    C = R.loc[clean.index]
    bg = pd.to_numeric(C[COL_BG], errors="coerce")
    S = C.loc[bg.notna(), COL_BASIS]
    g = S.map(grup)
    n = int(len(S))
    res = {"temiz_n": int(len(C)), "band_gap_dolu": n, "band_gap_bos": int(bg.isna().sum()),
           "gruplar": {k: {"n": int(v), "yuzde": round(100 * v / n, 1)} for k, v in g.value_counts().items()},
           "ham_kategoriler": {str(k): int(v) for k, v in S.fillna("(boş)").astype(str).value_counts().head(12).items()}}
    out = ROOT / "outputs" / "robustness" / "bandgap_kestirim_temeli.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("================ Band gap kestirim temeli (temiz kume) ================")
    print(f"Temiz {res['temiz_n']} kayit; band gap dolu {n}, bos {res['band_gap_bos']}")
    for k, v in res["gruplar"].items():
        print(f"  {k:22s} {v['n']:6d}  (%{v['yuzde']})")
    print("Ham kategoriler:", res["ham_kategoriler"])
    print("Cikti:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
