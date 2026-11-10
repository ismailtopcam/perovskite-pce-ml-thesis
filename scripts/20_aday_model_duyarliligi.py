"""Stage 20 — Aday siralamasinin MODEL AILESINE duyarliligi.

Tez Bolum 5.7 bunu acik bir sinirlilik olarak yaziyor: aday siralamasi yalnizca
CatBoost ile uretilmistir; HistGBM ya da XGBoost ile yeniden puanlamanin ilk 30
listesini ne olcude korudugu sinanmamistir. Ustelik ilk dort gradyan artirma
modelinin toplu basarimi katlar-arasi standart sapmanin altinda ayrisiyor
(Tablo 4.2), yani "en iyi model" secimi istatistiksel olarak marjinal. Bu betik
o bosluku kapatir.

Kurulum: dort GBM de AYNI DOI-grup egitim bolmesinde (holdout protokolu,
splits.holdout_split, tohum config'ten) egitilir ve AYNI 756 adayi puanlar.
Raporlanan: modeller arasi Spearman rho matrisi, ilk-10/ilk-30 ortusmeleri,
model-ortalamasi siralamasi ve her modelin holdout basarimi.

Girdi : data/processed/model_ready_dataset.csv
Cikti : outputs/robustness/aday_model_duyarliligi.json
        outputs/robustness/aday_tahminleri_4model.csv
        outputs/robustness/aday_model_ortalamasi_ilk30.csv
Calistirma: python scripts/20_aday_model_duyarliligi.py
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.registry import build_models
from perovskite_ml.validation.splits import holdout_split
from perovskite_ml.candidates.candidate_space import enumerate_candidates
from perovskite_ml.utils.manifest import write_manifest

GBMS = ["CatBoost", "HistGBM", "XGBoost", "LightGBM"]


def main():
    t0 = time.time()
    cfg = load_config(str(ROOT / "config.yaml"))
    seed, target, doi_col = cfg["seed"], cfg["target"], cfg["group_col"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)

    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    y = M[target].values.astype(float)
    groups = M[doi_col].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[target, doi_col])

    tr, te = holdout_split(X, y, groups, test_size=0.2, seed=seed)
    print(f"[stage20] holdout: egitim {len(tr):,} / test {len(te):,}")

    # aday uzayi — Stage 6 ile ayni sabit varsayimlar (egitim medyanlari)
    bg = float(M["band_gap"].median()) if "band_gap" in M else 1.6
    at = float(M["anneal_temp"].median()) if "anneal_temp" in M else 100.0
    ati = float(M["anneal_time"].median()) if "anneal_time" in M else 30.0
    recipes, enc = enumerate_candidates(list(X.columns), bg, at, ati)
    print(f"[stage20] aday: {len(recipes)} | sabitler bg={bg} at={at} ati={ati}")

    models = build_models(seed)
    preds, perf = {}, {}
    for name in GBMS:
        if name not in models:
            print(f"[stage20] {name} yok, atlandi"); continue
        m = models[name]
        m.fit(X.iloc[tr], y[tr])
        yh = m.predict(X.iloc[te])
        perf[name] = dict(holdout_r2=round(float(r2_score(y[te], yh)), 4),
                          holdout_mae=round(float(mean_absolute_error(y[te], yh)), 4))
        preds[name] = np.asarray(m.predict(enc), dtype=float)
        print(f"     {name:10s} holdout R2={perf[name]['holdout_r2']:.4f} | "
              f"aday tavani {preds[name].max():.2f}")

    P = pd.DataFrame(preds)
    cand = pd.concat([recipes.reset_index(drop=True), P], axis=1)
    cand["ortalama"] = P.mean(axis=1)
    cand["std"] = P.std(axis=1)

    names = list(P.columns)
    rho = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            rho.loc[a, b] = round(float(spearmanr(P[a], P[b]).statistic), 4)

    def topk(col, k):
        return set(cand.sort_values(col, ascending=False).head(k).index)

    ov = {}
    for k in (10, 30):
        for b in names[1:]:
            ov[f"CatBoost_vs_{b}_top{k}"] = len(topk("CatBoost", k) & topk(b, k))
        ov[f"CatBoost_vs_ortalama_top{k}"] = len(topk("CatBoost", k) & topk("ortalama", k))

    print("\n[Spearman rho — adaylarin model-ler arasi sira korelasyonu]")
    print(rho.to_string())
    print("\n[Ilk-k ortusmeleri]")
    for k, v in ov.items():
        print(f"  {k:34s} {v}")

    cand.sort_values("ortalama", ascending=False).to_csv(
        outdir / "aday_tahminleri_4model.csv", index=False)
    top30 = cand.sort_values("ortalama", ascending=False).head(30)
    top30.to_csv(outdir / "aday_model_ortalamasi_ilk30.csv", index=False)

    out = dict(holdout_performance=perf,
               spearman_rho=rho.to_dict(),
               topk_overlap=ov,
               candidate_ceiling={n: round(float(P[n].max()), 4) for n in names},
               ensemble_ceiling=round(float(cand["ortalama"].max()), 4),
               max_disagreement_points=round(float(cand["std"].max()), 4),
               mean_disagreement_points=round(float(cand["std"].mean()), 4))
    with open(outdir / "aday_model_duyarliligi.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage20_aday_model_duyarliligi",
                   cfg, metrics={"ensemble_ceiling": out["ensemble_ceiling"]},
                   outputs=["outputs/robustness/aday_model_duyarliligi.json"])

    print(f"\n[Model-ortalamasi ilk 5]")
    for _, r in top30.head(5).iterrows():
        print(f"   {r['ortalama']:.2f} (+/-{r['std']:.2f}) | A={r['A']} B={r['B']} X={r['X']} "
              f"{r['arch']}/{r['ETL']}/{r['HTL']}")
    print(f"\nTamam ({time.time() - t0:.0f} s).")


if __name__ == "__main__":
    main()
