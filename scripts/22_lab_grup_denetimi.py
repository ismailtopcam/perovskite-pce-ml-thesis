"""Stage 22 — Laboratuvar duzeyi kalinti sizinti: bas yazar vekiliyle gruplama.

Tez Bolum 5.7 ve 6.3 su kalinti riski kabul ediyor: "kanonik DOI-grup ayrimi ayni yayin
kaynakli sizintiyi giderir; ancak ayni laboratuvarin FARKLI yayinlari arasindaki
benzerlikten dogan kalinti risk surmektedir" ve cozumu gelecek calismaya (DataSAIL)
birakiyor. Veri tabaninda Ref_lead_author alani bulundugundan bu risk ol culebilir:
bas yazar, laboratuvar icin kusurlu ama tutarli bir vekildir.

Kurulum: DOI yerine kanoniklestirilmis bas yazar anahtariyla 5 kat GroupKFold.
Bas yazar grubu DOI grubundan DAHA KABA oldugundan (her DOI'nin tek bir bas yazari var),
bu bolme DOI-ayrikligini da kapsar. ANCAK aradaki fark bir ust sinir DEGILDIR: bas yazar
vekili iki yonde de kusurludur — ayni adli farkli arastirmacilar birleserek etkiyi fazla,
bir laboratuvarin farkli bas yazarlara bolunen yayinlari ise ayni-lab kayitlarini kat
icinde birakarak etkiyi eksik gosterir. Fark bu nedenle yonu belirlenemeyen bir NOKTA
KESTIRIMI olarak okunmalidir.

Girdi : data/raw/...csv, data/processed/model_ready_dataset.csv
Cikti : outputs/robustness/lab_grup_denetimi.json
Calistirma: python scripts/22_lab_grup_denetimi.py
"""
import sys, json, re, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.data.load_data import load_raw, _resolve_raw_path
from perovskite_ml.data.clean_data import clean_rows
from perovskite_ml.models.registry import build_models
from perovskite_ml.models.tuning import grouped_cv_r2
from perovskite_ml.utils.manifest import write_manifest

AUTHOR = "Ref_lead_author"


def canon_author(s):
    """Kucuk harf, noktalama ve bosluk sadelestirmesi; bos/eksik -> kendi tekil grubu."""
    s = ("" if pd.isna(s) else str(s)).strip().lower()
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    t0 = time.time()
    cfg = load_config(str(ROOT / "config.yaml"))
    seed, target, doi_col = cfg["seed"], cfg["target"], cfg["group_col"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw(cfg)
    clean, _ = clean_rows(raw_df, cfg)
    A = pd.read_csv(_resolve_raw_path(cfg), usecols=[AUTHOR, target], low_memory=False)
    assert len(A) == len(raw_df), "ham CSV satir sayisi tutarsiz"

    M = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "model_ready_dataset.csv", low_memory=False)
    assert len(M) == len(clean), f"model-hazir {len(M)} != temiz {len(clean)}"
    assert np.allclose(M[target].values, clean[target].values), "satir eslesmesi bozuk"

    author = A.loc[clean.index, AUTHOR].map(canon_author).values
    # bos bas yazar -> her kayit kendi tekil grubunda (muhafazakar: birlestirme yapma)
    author = np.array([a if a else f"__bos_{i}" for i, a in enumerate(author)], dtype=object)

    doi = M[doi_col].fillna("unknown_doi").astype(str).values
    y = M[target].values.astype(float)
    X = M.drop(columns=[target, doi_col])

    n_auth = pd.Series(author).nunique()
    n_doi = pd.Series(doi).nunique()
    sizes = pd.Series(author).value_counts()
    dois_per_author = pd.DataFrame({"a": author, "d": doi}).groupby("a")["d"].nunique()
    bos = int(sum(1 for a in author if str(a).startswith("__bos_")))

    print(f"[stage22] bas yazar grubu: {n_auth:,} | DOI grubu: {n_doi:,} | bas yazari bos kayit: {bos:,}")
    print(f"[stage22] en buyuk yazar grubu: {int(sizes.iloc[0]):,} kayit ({sizes.iloc[0]/len(M):.1%})")
    print(f"[stage22] yazar basina DOI: medyan {dois_per_author.median():.0f} | "
          f"maks {int(dois_per_author.max())} | >1 yayinli yazar: {int((dois_per_author > 1).sum()):,}")

    model = build_models(seed)["CatBoost"]
    rows = []
    for label, g in [("DOI-grup (taban cizgisi)", doi), ("bas yazar grubu (lab vekili)", author)]:
        m, s = grouped_cv_r2(model, X, y, g, 5, atama="kararli")
        rows.append(dict(kurulum=label, grup_sayisi=int(pd.Series(g).nunique()),
                         R2_mean=round(m, 6), R2_std=round(s, 6)))
        print(f"[CV] {label:32s} R2={m:.4f} ± {s:.4f}", flush=True)

    fark = rows[0]["R2_mean"] - rows[1]["R2_mean"]
    print(f"\n>> Laboratuvar duzeyi fark (nokta kestirim, ust sinir DEGIL) = {fark:.4f} R2 puani")
    print(f"   kiyas: bas yazar kurulumunun kat std'si {rows[1]['R2_std']:.4f} — fark bunun altinda kalirsa")
    print("   etkinin buyuklugu bu veriyle ayirt edilemez.")

    res = dict(n_author_groups=int(n_auth), n_doi_groups=int(n_doi),
               n_missing_author=bos,
               largest_author_group=int(sizes.iloc[0]),
               largest_author_share=round(float(sizes.iloc[0] / len(M)), 4),
               dois_per_author_median=float(dois_per_author.median()),
               dois_per_author_max=int(dois_per_author.max()),
               multi_pub_authors=int((dois_per_author > 1).sum()),
               cv=rows, lab_doi_r2_farki=round(float(fark), 4),
               not_="Fark bir ust sinir degildir; bas yazar vekili iki yonde de kusurludur.")
    with open(outdir / "lab_grup_denetimi.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage22_lab_grup", cfg,
                   metrics={"lab_doi_r2_farki": res["lab_doi_r2_farki"]},
                   outputs=["outputs/robustness/lab_grup_denetimi.json"])
    print(f"Tamam ({time.time() - t0:.0f} s).")


if __name__ == "__main__":
    main()
