"""Stage 25 — Deney A farkinin (rastgele bolme - DOI-grup bolmesi) bolme tohumuna duyarliligi.

Tezin ana karsilastirmasi (Tablo 4.4, Deney A): rastgele KFold R2 0,481, DOI-grup KFold
R2 0,410, fark 0,071. Bu fark tek bir kat atamasindan gelir (rastgele bolme tohum 42;
grup bolmesi arsivdeki kayitli atama). Bu betik ayni CatBoost yapilandirmasi ve ayni
77 ozellikle iki bolmeyi 10 farkli tohumla yeniden kurar ve farkin dagilimini verir.

Her tohum (0-9) icin:
  rastgele : KFold(5, shuffle=True, random_state=tohum)
  DOI-grup : tohumlu 5 kat. Kanonik DOI'ler siralanir, np.random.default_rng(tohum) ile
             karistirilir, her grup o an en az kayitli kata atanir (kayit sayisi dengeli;
             sklearn GroupKFold kullanilmadigi icin her ortamda ayni sonucu verir).
Model tohumu sabittir (config: 42); degisen yalnizca bolmedir.

Bu kosu arsiv degerlerini YENIDEN URETMEZ: arsivdeki 0,481 / 0,410 baska bir kat
atamasina aittir ve ciktida ayri bir satir olarak anilir (outputs/v4/validation_experiments.csv).

Girdi : data/processed/model_ready_dataset.csv, outputs/v4/validation_experiments.csv
Cikti : outputs/robustness/deney_a_tohum_duyarliligi.json, .csv
        outputs/manifests/manifest_stage25_deney_a_tohum.json
Calistirma: python scripts/25_deney_a_tohum_duyarliligi.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.evaluate import metrics
from perovskite_ml.models.registry import build_models
from perovskite_ml.utils.manifest import write_manifest

TOHUMLAR = list(range(10))
N_KAT = 5
GIRDILER = ["data/processed/model_ready_dataset.csv", "outputs/v4/validation_experiments.csv"]


def _sha256(yol):
    h = hashlib.sha256()
    with open(ROOT / yol, "rb") as f:
        while parca := f.read(1 << 20):
            h.update(parca)
    return h.hexdigest()


def tohumlu_grup_katlari(groups, n_kat, tohum):
    """Kayit basina kat numarasi. Sirali benzersiz gruplar tohumla karistirilir; her grup
    o an en az kayitli kata eklenir (esitlikte en kucuk numarali kat)."""
    ug, gi = np.unique(np.asarray(groups), return_inverse=True)
    gi = np.asarray(gi).ravel()
    boyut = np.bincount(gi, minlength=len(ug))
    sira = np.random.default_rng(tohum).permutation(len(ug))
    yuk = np.zeros(n_kat, dtype=np.int64)
    g2k = np.empty(len(ug), dtype=int)
    for g in sira:
        k = int(np.argmin(yuk))
        yuk[k] += boyut[g]
        g2k[g] = k
    return g2k[gi]


def cv(model, X, y, bolmeler):
    r2, mae = [], []
    for egitim, dogrulama in bolmeler:
        m = clone(model)
        m.fit(X.iloc[egitim], y[egitim])
        mt = metrics(y[dogrulama], m.predict(X.iloc[dogrulama]))
        r2.append(mt["R2"])
        mae.append(mt["MAE"])
    return r2, mae


def ozet(v):
    v = np.asarray(v, dtype=float)
    return {"ortalama": float(v.mean()), "ss_ddof1": float(v.std(ddof=1)), "ss_ddof0": float(v.std(ddof=0)),
            "en_kucuk": float(v.min()), "en_buyuk": float(v.max())}


def main():
    t0 = time.time()
    cfg = load_config(str(ROOT / "config.yaml"))
    proc = ROOT / cfg["paths"]["processed_dir"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[cfg["target"]].values
    groups = M[cfg["group_col"]].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[cfg["target"], cfg["group_col"]])
    model = build_models(cfg["seed"])["CatBoost"]
    n_grup = int(len(np.unique(groups)))
    print(f"[stage25] {len(M)} kayit, {X.shape[1]} ozellik, {n_grup} DOI grubu; tohumlar {TOHUMLAR}", flush=True)

    satirlar = []
    for tohum in TOHUMLAR:
        t1 = time.time()
        rastgele = list(KFold(N_KAT, shuffle=True, random_state=tohum).split(X))
        kat = tohumlu_grup_katlari(groups, N_KAT, tohum)
        grup = [(np.flatnonzero(kat != k), np.flatnonzero(kat == k)) for k in range(N_KAT)]
        for egitim, dogrulama in grup:
            assert set(groups[egitim]).isdisjoint(groups[dogrulama]), "grup bolmesi ayrik degil"
        r2_r, mae_r = cv(model, X, y, rastgele)
        r2_g, mae_g = cv(model, X, y, grup)
        s = {"tohum": tohum,
             "R2_rastgele": float(np.mean(r2_r)), "R2_grup": float(np.mean(r2_g)),
             "fark_R2": float(np.mean(r2_r) - np.mean(r2_g)),
             "MAE_rastgele": float(np.mean(mae_r)), "MAE_grup": float(np.mean(mae_g)),
             "fark_MAE": float(np.mean(mae_r) - np.mean(mae_g)),
             "R2_rastgele_kat_ss": float(np.std(r2_r)), "R2_grup_kat_ss": float(np.std(r2_g)),
             "grup_kat_boyutlari": [int(v) for v in np.bincount(kat, minlength=N_KAT)],
             "R2_rastgele_katlar": [float(v) for v in r2_r], "R2_grup_katlar": [float(v) for v in r2_g],
             "MAE_rastgele_katlar": [float(v) for v in mae_r], "MAE_grup_katlar": [float(v) for v in mae_g]}
        satirlar.append(s)
        print(f"[stage25] tohum {tohum}: R2 rastgele {s['R2_rastgele']:.4f}  grup {s['R2_grup']:.4f}  "
              f"fark {s['fark_R2']:+.4f} | MAE rastgele {s['MAE_rastgele']:.4f}  grup {s['MAE_grup']:.4f}  "
              f"fark {s['fark_MAE']:+.4f}  ({time.time() - t1:.0f} s)", flush=True)

    arsiv = pd.read_csv(ROOT / "outputs" / "v4" / "validation_experiments.csv").set_index(["deney", "ayar"])
    a_r = float(arsiv.loc[("A_bolme", "rastgele_KFold"), "R2_mean"])
    a_g = float(arsiv.loc[("A_bolme", "DOI_grup_KFold"), "R2_mean"])

    sutunlar = ["R2_rastgele", "R2_grup", "fark_R2", "MAE_rastgele", "MAE_grup", "fark_MAE"]
    res = {
        "yontem": {
            "model": "CatBoost (registry.build_models; model tohumu config tohumu)",
            "n_kayit": int(len(M)), "n_ozellik": int(X.shape[1]), "n_grup": n_grup, "n_kat": N_KAT,
            "tohumlar": TOHUMLAR,
            "rastgele": "KFold(5, shuffle=True, random_state=tohum)",
            "grup": "sirali kanonik DOI'ler default_rng(tohum) ile karistirilir; her grup en az kayitli kata",
            "fark_tanimi": "fark_R2 = R2_rastgele - R2_grup; fark_MAE = MAE_rastgele - MAE_grup "
                           "(her deger bes katin ortalamasi)",
        },
        "tohumlar": satirlar,
        "ozet": {c: ozet([s[c] for s in satirlar]) for c in sutunlar},
        "arsiv": {"aciklama": "outputs/v4/validation_experiments.csv; bu kosuda yeniden uretilmez "
                              "(rastgele bolme tohum 42, grup bolmesi kayitli arsiv atamasi)",
                  "R2_rastgele": a_r, "R2_grup": a_g, "fark_R2": a_r - a_g},
    }
    (outdir / "deney_a_tohum_duyarliligi.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    pd.DataFrame(satirlar)[["tohum"] + sutunlar + ["R2_rastgele_kat_ss", "R2_grup_kat_ss"]].to_csv(
        outdir / "deney_a_tohum_duyarliligi.csv", index=False, lineterminator="\n")

    o = res["ozet"]
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage25_deney_a_tohum", cfg,
                   metrics={"bolme_tohumlari": TOHUMLAR,
                            "fark_R2_ortalama": round(o["fark_R2"]["ortalama"], 6),
                            "fark_R2_ss_ddof1": round(o["fark_R2"]["ss_ddof1"], 6),
                            "fark_R2_en_kucuk": round(o["fark_R2"]["en_kucuk"], 6),
                            "fark_R2_en_buyuk": round(o["fark_R2"]["en_buyuk"], 6),
                            "girdi_sha256": {g: _sha256(g) for g in GIRDILER}},
                   outputs=["outputs/robustness/deney_a_tohum_duyarliligi.json",
                            "outputs/robustness/deney_a_tohum_duyarliligi.csv"])

    print("\n================= STAGE 25 =================")
    for c in sutunlar:
        print(f"  {c:13s} ort {o[c]['ortalama']:.4f}  ss {o[c]['ss_ddof1']:.4f}  "
              f"[{o[c]['en_kucuk']:.4f}; {o[c]['en_buyuk']:.4f}]")
    print(f"  arsiv: rastgele {a_r:.4f}  grup {a_g:.4f}  fark {a_r - a_g:.4f}")
    print(f"Tamam ({time.time() - t0:.0f} s).")


if __name__ == "__main__":
    main()
