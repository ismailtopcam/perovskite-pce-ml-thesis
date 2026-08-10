"""Stage 17 (v0.4.0 / 1-C) — Gurultu tabani / tavan R2'nin ayni evrende,
yayin-agirlikli ve recete-tanimina duyarli yeniden hesabi.

v0.3.x'te (thesis_data_analysis.py, D1; tez Tablo 5.1) taban varyansi yalnizca
cok-yayinli tekrar eden recete alt kumesinden hesaplanir ve tum veri kumesinin
toplam varyansina oranlanirdi; bu, gurultunun veri genelinde sabit oldugu
(homoskedastik) varsayimina dayanir ve kayit-agirliklidir (buyuk yayinlar
quantile/pooling'te daha cok sayilir). Bu betik ayni buyuklugu uc eksende
saglamlastirir:

1) AYNI EVREN SINIRLARI: tabani tum-kayit evrenine (i) yalnizca kapsanan
   kayitlarin gurultusuyle (alt sinir; kapsanmayanlara 0 gurultu) ve (ii)
   homoskedastik genellemeyle (nokta kestirim) yayarak TABAN ARALIGI verir.
2) YAYIN-AGIRLIKLI: her recete grubunda once yayin (DOI) ortalamalari alinir;
   grup-ici varyans yayinlar-arasi duzeyde, her yayin BIR kez sayilarak
   hesaplanir. Toplam varyans da ayni birimde (yayin-recete hucre ortalamalari)
   hesaplanir -> buyuk yayinlarin agirligi kalkar.
3) RECETE TANIMI DUYARLILIGI: (K1) bayraksiz 2-ondalik (v0.3.x tanimi),
   (K2) bayraklar dahil 2-ondalik, (K3) bayraksiz yuvarlamasiz tam esitlik.

Girdi : data/processed/model_ready_dataset.csv
Cikti : outputs/robustness/gurultu_tabani_v2.json
        outputs/robustness/gurultu_tabani_v2_ozet.csv
Calistirma: python scripts/17_gurultu_tabani_v2.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.utils.manifest import write_manifest

ROUND_DEC = 2  # v0.3.x ile ayni yuvarlama hassasiyeti


def build_key(df, cols, round_dec):
    sub = df[cols]
    if round_dec is not None:
        sub = sub.round(round_dec)
    return sub.astype(str).agg("|".join, axis=1)


def analyze(M, key_cols, label, round_dec, target, doi):
    y_all = M[target].astype(float)
    n_all = len(M)
    var_total_rw = float(y_all.var(ddof=1))

    d = M.copy()
    d["_k"] = build_key(d, key_cols, round_dec)

    # cok-yayinli receteler (>=2 farkli DOI)
    ndoi = d.groupby("_k")[doi].nunique()
    keep = ndoi[ndoi >= 2].index
    cov = d[d["_k"].isin(keep)]
    n_cov, n_rec = len(cov), cov["_k"].nunique()

    out = dict(label=label, key_cols=len(key_cols), round_dec=round_dec,
               n_records_all=n_all, n_records_covered=n_cov, n_recipes_multi_doi=int(n_rec),
               coverage_share=round(n_cov / n_all, 4),
               var_total_record_weighted=round(var_total_rw, 4))
    if n_cov == 0:
        return out

    # --- kayit-agirlikli (v0.3.x ile ayni cekirdek) ---
    gm = cov.groupby("_k")[target]
    resid = cov[target].astype(float) - gm.transform("mean")
    ss_w = float((resid ** 2).sum())
    var_w_rw = ss_w / max(1, n_cov - n_rec)
    out["record_weighted"] = dict(
        pooled_noise_var=round(var_w_rw, 4),
        within_recipe_std_median=round(float(gm.std(ddof=1).dropna().median()), 4),
        # ayni-evren taban araligi: alt sinir (kapsanmayanlara 0 gurultu) .. homoskedastik
        floor_rmse_lower=round(float(np.sqrt(ss_w / n_all)), 4),
        floor_rmse_homosked=round(float(np.sqrt(var_w_rw)), 4),
        ceiling_r2_homosked=round(1 - var_w_rw / var_total_rw, 4),
        ceiling_r2_upper=round(1 - (ss_w / n_all) / var_total_rw, 4),
    )

    # --- yayin-agirlikli: once yayin(DOI) ortalamalari, her yayin bir kez ---
    cell = cov.groupby(["_k", doi])[target].mean().rename("cell_mean").reset_index()
    g = cell.groupby("_k")["cell_mean"]
    resid_pw = cell["cell_mean"] - g.transform("mean")
    n_cells, n_rec_pw = len(cell), cell["_k"].nunique()
    ss_pw = float((resid_pw ** 2).sum())
    var_w_pw = ss_pw / max(1, n_cells - n_rec_pw)
    # toplam varyans ayni birimde: yayin-recete hucre ortalamalari (tum veri)
    cell_all = d.groupby(["_k", doi])[target].mean()
    var_total_pw = float(cell_all.var(ddof=1))
    out["publication_weighted"] = dict(
        n_publication_recipe_cells=int(n_cells),
        pooled_noise_var=round(var_w_pw, 4),
        var_total_cell_level=round(var_total_pw, 4),
        floor_rmse_homosked=round(float(np.sqrt(var_w_pw)), 4),
        ceiling_r2_homosked=round(1 - var_w_pw / var_total_pw, 4),
    )
    return out


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    proc = ROOT / cfg["paths"]["processed_dir"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)
    target, doi = cfg["target"], cfg["group_col"]

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    M[doi] = M[doi].fillna("unknown_doi").astype(str)

    feat = [c for c in M.columns if c not in (target, doi)]
    no_flag = [c for c in feat if not c.lower().endswith(("_missing", "_flag"))]

    variants = [
        (no_flag, "K1: bayraksiz, 2-ondalik (v0.3.x tanimi)", ROUND_DEC),
        (feat,    "K2: bayraklar dahil, 2-ondalik", ROUND_DEC),
        (no_flag, "K3: bayraksiz, yuvarlamasiz (tam esitlik)", None),
    ]
    results = []
    for cols, label, rd in variants:
        r = analyze(M, cols, label, rd, target, doi)
        results.append(r)
        print(f"\n[{label}]")
        print(f"  kapsanan kayit: {r.get('n_records_covered', 0):,}/{r['n_records_all']:,} "
              f"(pay {r.get('coverage_share', 0):.1%}), cok-yayinli recete: {r.get('n_recipes_multi_doi', 0):,}")
        if "record_weighted" in r:
            rw = r["record_weighted"]
            print(f"  kayit-agirlikli : taban RMSE {rw['floor_rmse_lower']} .. {rw['floor_rmse_homosked']} | "
                  f"tavan R2 {rw['ceiling_r2_homosked']} .. {rw['ceiling_r2_upper']}")
            pw = r["publication_weighted"]
            print(f"  yayin-agirlikli : taban RMSE {pw['floor_rmse_homosked']} | tavan R2 {pw['ceiling_r2_homosked']} "
                  f"(hucre-duzeyi toplam varyans {pw['var_total_cell_level']})")

    with open(outdir / "gurultu_tabani_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    rows = []
    for r in results:
        row = dict(tanim=r["label"], kapsanan=r.get("n_records_covered", 0),
                   pay=r.get("coverage_share", 0))
        if "record_weighted" in r:
            rw, pw = r["record_weighted"], r["publication_weighted"]
            row.update(taban_rmse_alt=rw["floor_rmse_lower"], taban_rmse_homosked=rw["floor_rmse_homosked"],
                       tavan_r2_homosked=rw["ceiling_r2_homosked"], tavan_r2_ust=rw["ceiling_r2_upper"],
                       yayin_agirlikli_taban=pw["floor_rmse_homosked"], yayin_agirlikli_tavan=pw["ceiling_r2_homosked"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "gurultu_tabani_v2_ozet.csv", index=False)

    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage17_gurultu_tabani_v2", cfg,
                   metrics={}, outputs=["outputs/robustness/gurultu_tabani_v2.json"])
    print("\nTamam. Cikti: outputs/robustness/gurultu_tabani_v2.json / _ozet.csv")


if __name__ == "__main__":
    main()
