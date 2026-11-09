# -*- coding: utf-8 -*-
"""Aday A-site izgara noktalarinin veri destegi.

Aday uzayinin A-site ekseni (Bolum 3.12) literatur recetelerinin yuvarlanmis izgara
noktalaridir. Bu denetim, her izgara noktasinin model-hazir kumede ve DOI-grup egitim
bolmesinde (scripts/03 ile ayni holdout_split, tohum 42) birebir ya da yakin bantta kac
kayitla temsil edildigini ve Saliba vd. (2016) ozgun recetesinin (Cs0.05(MA0.17FA0.83)0.95
~ FA0.79 MA0.16 Cs0.05) karsiligini sayar; ayrica birincil listenin ilk 10 adayinin A-site
dagilimini verir. (Birlikte gorulme denetimi — aday_uyumluluk_filtresi.py — kompozisyonu kapsamaz.)

Girdi : data/processed/model_ready_dataset.csv, outputs/candidates_full/top30_compat_N5.csv
Cikti : outputs/robustness/aday_izgara_destegi.json (+ konsol ozeti)
Calistirma (repo kokunden): python aday_izgara_destegi.py   (tez Bolum 3.12, 4.5, 5.6)
"""
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.validation.splits import holdout_split

GRID = {  # aday eksenindeki etiket -> (FA, MA, Cs)
    "FA": (1.0, 0.0, 0.0), "MA": (0.0, 1.0, 0.0), "FA0.83_Cs0.17": (0.83, 0.0, 0.17),
    "FA0.85_MA0.15": (0.85, 0.15, 0.0), "FA0.8_MA0.1_Cs0.1": (0.80, 0.10, 0.10),
    "MA0.9_Cs0.1": (0.0, 0.90, 0.10), "Cs0.2_FA0.8": (0.80, 0.0, 0.20),
}
SALIBA = ("Saliba_2016_FA0.79_MA0.16_Cs0.05", (0.79, 0.16, 0.05))
TOLS = [0.005, 0.01, 0.03]


def counts(M, fa, ma, cs):
    out = {}
    for t in TOLS:
        m = ((M["A_FA"] - fa).abs() <= t) & ((M["A_MA"] - ma).abs() <= t) & ((M["A_Cs"] - cs).abs() <= t)
        out[f"tol_{t}"] = int(m.sum())
    return out


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    M = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values
    groups = M[cfg["group_col"]].fillna("unknown_doi").astype(str).values
    tr, te = holdout_split(M, y, groups, test_size=0.2, seed=cfg["seed"])
    T = M.iloc[tr]

    res = {"model_hazir_n": int(len(M)), "egitim_bolmesi_n": int(len(T)), "izgara": {}}
    for lab, (fa, ma, cs) in list(GRID.items()) + [SALIBA]:
        res["izgara"][lab] = {"model_hazir": counts(M, fa, ma, cs), "egitim": counts(T, fa, ma, cs)}

    top = pd.read_csv(ROOT / "outputs" / "candidates_full" / "top30_compat_N5.csv")
    res["birincil_ilk10_A"] = {str(k): int(v) for k, v in top.head(10)["A"].value_counts().items()}
    res["birincil_ilk30_A"] = {str(k): int(v) for k, v in top.head(30)["A"].value_counts().items()}

    out = ROOT / "outputs" / "robustness" / "aday_izgara_destegi.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("================ Aday A-site izgara destegi ================")
    for lab, v in res["izgara"].items():
        print(f"  {lab:34s} model-hazir {v['model_hazir']}  egitim {v['egitim']}")
    print("Birincil ilk 10 A-site:", res["birincil_ilk10_A"])
    print("Cikti:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
