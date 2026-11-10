"""Stage 19 — Deney D (recete-grup bolmesi) + gurultu tabani kiyasinin ayni evrende yapilmasi.

Tez Bolum 5.2 soyle diyor: DOI-grup ayrimi YAYIN sizintisini hedefler, RECETE tekrarini
degil; kayitlarin ~%42'si birden cok yayinda tekrar eden bir receteye ait. Dolayisiyla
R2 = 0,410 "gorulmemis yayina" genellemeyi olcer, "gorulmemis receteye" degil. O test
gelecek calismaya birakilmisti (Bolum 6.4). Bu betik onu yapar.

Deneyler
--------
D0  DOI-grup KFold (taban cizgisi; tezdeki 0,410 +/- 0,016 ile karsilastirilir)
D1  RECETE-grup KFold  -> gorulmemis receteye genelleme
D2  RECETE + DOI birlesik bilesen grubu -> ne recete ne yayin siniri asar
    (recete-DOI iki parcali grafiginin baglantili bilesenleri; union-find).
    Grafik tek bir dev bilesende birlestiginden bu bolme kararsizdir; bilesen
    yapisi (en buyuk bilesen, kat R2 araligi) ayrica raporlanir.
D0' D0'in KENDI test katlarinda, recetesi o katin egitim kisminda hic gecmeyen
    kayitlar ile gorulmus-recete kayitlarinin hata karsilastirmasi (tez Bolum 5.2'de
    raporlanan birincil recete-duzeyi olcum; R2 yaninda alt kume varyansi da verilir).

M3 (ayni evren kiyasi)
----------------------
Tez Bolum 5.1'de model RMSE'si (4,02; TUM 41.485 kayit) ile taban RMSE (3,64; yalnizca
>=2 yayinda tekrar eden recetelere ait 17.409 kayit) karsilastiriliyor ve metin bunun
"yaklasik" oldugunu kabul ediyor. Burada D0'in kat-disi (OOF) tahminleri saklanip model
RMSE'si AYNI 17.409 kayitlik alt kumede yeniden hesaplanir -> kiyas tam olur.

Recete tanimi script 17 (K1) ile ozdes: bayraksiz ozellikler, 2 ondalik yuvarlama.

Girdi : data/processed/model_ready_dataset.csv
Cikti : outputs/robustness/recete_grup_ve_taban.json
        outputs/robustness/recete_grup_ve_taban_ozet.csv
        outputs/robustness/d0_oof_tahminler.csv
Calistirma: python scripts/19_recete_grup_ve_taban.py
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.registry import build_models
from perovskite_ml.utils.manifest import write_manifest

ROUND_DEC = 2
N_SPLITS = 5


def build_recipe_key(X, round_dec=ROUND_DEC):
    """Script 17'deki K1 tanimi: eksiklik bayraklari haric, 2 ondalik yuvarlama."""
    cols = [c for c in X.columns if not c.lower().endswith(("_missing", "_flag"))]
    return X[cols].round(round_dec).astype(str).agg("|".join, axis=1), cols


def union_find_components(keys_a, keys_b):
    """Iki anahtar dizisinin (recete, DOI) birlesik bilesenleri -> grup etiketi."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in zip(keys_a, keys_b):
        union(("R", a), ("D", b))
    # kok dugumleri tamsayi etiketine indir (pandas/sklearn gruplama icin 1-B dizi gerekir)
    roots = [find(("R", a)) for a in keys_a]
    ids = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots], dtype=np.int64)


def cv_eval(model, X, y, groups, n_splits=N_SPLITS, collect_oof=False):
    """GroupKFold; kat basina R2 ve istenirse kat-disi (OOF) tahminler."""
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    r2s, maes, rmses = [], [], []
    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    for k, (tri, vai) in enumerate(splits):
        m = clone(model)
        m.fit(X.iloc[tri], y[tri])
        pred = m.predict(X.iloc[vai])
        r2s.append(r2_score(y[vai], pred))
        maes.append(mean_absolute_error(y[vai], pred))
        rmses.append(float(np.sqrt(np.mean((y[vai] - pred) ** 2))))
        if collect_oof:
            oof[vai] = pred
            fold_id[vai] = k
    res = dict(r2_mean=float(np.mean(r2s)), r2_std=float(np.std(r2s)),
               mae_mean=float(np.mean(maes)), rmse_mean=float(np.mean(rmses)),
               fold_r2=[round(v, 4) for v in r2s],
               n_groups=int(pd.Series(groups).nunique()))
    return (res, oof, fold_id) if collect_oof else (res, None, None)


def main():
    t0 = time.time()
    cfg = load_config(str(ROOT / "config.yaml"))
    seed, target, doi_col = cfg["seed"], cfg["target"], cfg["group_col"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = ROOT / cfg["paths"]["processed_dir"]
    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[target].values.astype(float)
    doi = M[doi_col].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[target, doi_col])

    model = build_models(seed)["CatBoost"]       # tezin nihai modeli
    recipe, key_cols = build_recipe_key(X)
    recipe = recipe.values

    # kapsanan kume: >=2 FARKLI yayinda gorulen receteler (tez Tablo 5.1 ile ayni olcut)
    ndoi = pd.DataFrame({"r": recipe, "d": doi}).groupby("r")["d"].nunique()
    covered_recipes = set(ndoi[ndoi >= 2].index)
    covered = np.array([r in covered_recipes for r in recipe])

    comp = union_find_components(recipe, doi)

    print(f"[stage19] recete anahtari: {len(key_cols)} ozellik, {ROUND_DEC} ondalik")
    print(f"[stage19] benzersiz recete: {pd.Series(recipe).nunique():,} | "
          f"cok-yayinli recete: {len(covered_recipes):,} | kapsanan kayit: {covered.sum():,} "
          f"({covered.mean():.1%})")
    print(f"[stage19] DOI grubu: {pd.Series(doi).nunique():,} | birlesik bilesen: {pd.Series(comp).nunique():,}")

    results = {}

    print("\n[D0] DOI-grup KFold ...", flush=True)
    d0, oof, fold_id = cv_eval(model, X, y, doi, collect_oof=True)
    results["D0_DOI_grup"] = d0
    print(f"     R2 = {d0['r2_mean']:.4f} +/- {d0['r2_std']:.4f} | MAE {d0['mae_mean']:.3f} | RMSE {d0['rmse_mean']:.3f}")

    print("[D1] RECETE-grup KFold ...", flush=True)
    d1, _, _ = cv_eval(model, X, y, recipe)
    results["D1_recete_grup"] = d1
    print(f"     R2 = {d1['r2_mean']:.4f} +/- {d1['r2_std']:.4f}")

    print("[D2] RECETE+DOI birlesik bilesen KFold ...", flush=True)
    d2, _, _ = cv_eval(model, X, y, comp)
    results["D2_recete_DOI_bilesen"] = d2
    print(f"     R2 = {d2['r2_mean']:.4f} +/- {d2['r2_std']:.4f}")

    # ---------- M3: ayni evren kiyasi ----------
    resid = y - oof
    rmse_all = float(np.sqrt(np.mean(resid ** 2)))
    rmse_cov = float(np.sqrt(np.mean(resid[covered] ** 2)))
    mae_cov = float(np.mean(np.abs(resid[covered])))

    # taban: kapsanan kumede havuzlanmis recete-ici varyans (script 17 K1 kayit-agirlikli)
    dfc = pd.DataFrame({"r": recipe[covered], "y": y[covered]})
    gm = dfc.groupby("r")["y"]
    ss_w = float(((dfc["y"] - gm.transform("mean")) ** 2).sum())
    n_cov, n_rec = len(dfc), dfc["r"].nunique()
    floor_rmse = float(np.sqrt(ss_w / max(1, n_cov - n_rec)))
    var_cov = float(dfc["y"].var(ddof=1))
    results["M3_ayni_evren"] = dict(
        n_covered=int(n_cov), n_recipes=int(n_rec),
        model_rmse_all_records=round(rmse_all, 4),
        model_rmse_covered=round(rmse_cov, 4),
        model_mae_covered=round(mae_cov, 4),
        floor_rmse_covered=round(floor_rmse, 4),
        gap_same_universe=round(rmse_cov - floor_rmse, 4),
        gap_thesis_style=round(rmse_all - floor_rmse, 4),
        var_covered=round(var_cov, 4),
        ceiling_r2_within_covered=round(1 - floor_rmse ** 2 / var_cov, 4),
        model_r2_covered=round(1 - rmse_cov ** 2 / var_cov, 4),
    )
    m3 = results["M3_ayni_evren"]
    print("\n[M3] Ayni evren kiyasi (yalnizca cok-yayinli recete kayitlari):")
    print(f"     kapsanan kayit {m3['n_covered']:,} / recete {m3['n_recipes']:,}")
    print(f"     model RMSE (tum kayit)   = {m3['model_rmse_all_records']}")
    print(f"     model RMSE (ayni evren)  = {m3['model_rmse_covered']}")
    print(f"     taban RMSE (ayni evren)  = {m3['floor_rmse_covered']}")
    print(f"     fark: ayni evrende {m3['gap_same_universe']} | tez uslubunda {m3['gap_thesis_style']}")

    # ---------- D0 katlarinda GORULMEMIS recete alt kumesi (tez Bolum 5.2) ----------
    # Her katta, recetesi o katin egitim kisminda hic gecmeyen test kayitlari ayrilir;
    # hata gorulmus-recete kayitlariyla karsilastirilir. Kat basina metrikler hesaplanip
    # katlar uzerinden ortalanir (R2 alt kume varyansina duyarli oldugundan varyans da raporlanir).
    rows_u = []
    for k in range(N_SPLITS):
        te = fold_id == k
        seen = set(recipe[~te])
        unseen = te & np.array([r not in seen for r in recipe])
        for lab, msk in (("gorulmemis_recete", unseen), ("gorulmus_recete", te & ~unseen)):
            e = y[msk] - oof[msk]
            rows_u.append(dict(kat=k, alt_kume=lab, n=int(msk.sum()), test_n=int(te.sum()),
                               MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                               varyans=float(np.var(y[msk], ddof=1)), R2=float(r2_score(y[msk], oof[msk]))))
    U = pd.DataFrame(rows_u)
    share = U[U.alt_kume == "gorulmemis_recete"].assign(pay=lambda d: d.n / d.test_n)["pay"]
    agg = U.groupby("alt_kume").agg(n=("n", "sum"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean"),
                                     varyans=("varyans", "mean"), R2=("R2", "mean"))
    results["D0_gorulmemis_recete"] = dict(
        test_payi_min=round(float(share.min()), 4), test_payi_max=round(float(share.max()), 4),
        ozet={k: {m: round(float(v), 4) for m, v in r.items()} for k, r in agg.to_dict("index").items()},
        kat_bazinda=U.round(4).to_dict("records"))
    print("\n[D0] Gorulmemis recete alt kumesi (kat ortalamalari):")
    print(agg.round(3).to_string())
    print(f"     test katindaki pay: {share.min():.3f} - {share.max():.3f}")

    # ---------- D2 bilesen yapisi (tez Bolum 5.2) ----------
    cs = pd.Series(comp).value_counts()
    results["D2_bilesen_yapisi"] = dict(
        bilesen_sayisi=int(len(cs)), en_buyuk_bilesen=int(cs.iloc[0]),
        en_buyuk_pay=round(float(cs.iloc[0] / len(y)), 4), tek_kayitlik_bilesen=int((cs == 1).sum()),
        kat_r2_min=round(float(min(d2["fold_r2"])), 4), kat_r2_max=round(float(max(d2["fold_r2"])), 4))
    print(f"[D2] en buyuk bilesen: {int(cs.iloc[0]):,} kayit ({cs.iloc[0] / len(y):.1%}); "
          f"kat R2 {min(d2['fold_r2']):.3f} - {max(d2['fold_r2']):.3f}")

    pd.DataFrame({"index": np.arange(len(y)), "y_true": y, "y_oof": oof,
                  "fold": fold_id, "recipe_covered": covered}).to_csv(
        outdir / "d0_oof_tahminler.csv", index=False)
    with open(outdir / "recete_grup_ve_taban.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    pd.DataFrame([
        dict(deney="D0 DOI-grup", grup_sayisi=d0["n_groups"], R2=round(d0["r2_mean"], 4), R2_std=round(d0["r2_std"], 4)),
        dict(deney="D1 recete-grup", grup_sayisi=d1["n_groups"], R2=round(d1["r2_mean"], 4), R2_std=round(d1["r2_std"], 4)),
        dict(deney="D2 recete+DOI bilesen", grup_sayisi=d2["n_groups"], R2=round(d2["r2_mean"], 4), R2_std=round(d2["r2_std"], 4)),
    ]).to_csv(outdir / "recete_grup_ve_taban_ozet.csv", index=False)

    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage19_recete_grup_ve_taban", cfg,
                   metrics={**{k: round(v["r2_mean"], 4) for k, v in results.items() if "r2_mean" in v},
                            "M3_gap_same_universe": results["M3_ayni_evren"]["gap_same_universe"],
                            "D2_en_buyuk_bilesen_payi": results["D2_bilesen_yapisi"]["en_buyuk_pay"]},
                   outputs=["outputs/robustness/recete_grup_ve_taban.json"])
    print(f"\nTamam ({time.time() - t0:.0f} s). Cikti: outputs/robustness/recete_grup_ve_taban.json")


if __name__ == "__main__":
    main()
