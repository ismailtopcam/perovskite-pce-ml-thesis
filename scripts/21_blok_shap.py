"""Stage 21 — Blok toplamli (grouped) SHAP: one-hot kodlamanin onem siralamasina etkisi.

Tez Tablo 4.6 / Sekil 4.8 tek bir surekli degiskeni (A-site FA orani) tek tek one-hot
sutunlariyla (HTL: PTAA, ETL: TiO2-c|TiO2-mp ...) ayni eksende siralar. 16 kategoriye
bolunmus bir degiskenin toplam katkisi 16 sutuna dagildigi icin her biri tek basina
kucuk gorunur; bu, "A-site FA orani en etkili degisken" sonucunu kismen kodlama
biciminin bir artefakti haline getirebilir. Bu betik ayni SHAP degerlerini SEMANTIK
degisken duzeyinde toplayarak ikinci bir siralama verir.

Tez Bolum 2.1 notunda da ayni ayrim var: "29 oznitelik ... bu tezin one-hot kodlama
sonrasi 77 kolonuyla ayni sayim duzeyinde degildir."

Protokol tez Bolum 3.11 ile ayni: aciklanan model DOI-grup EGITIM bolmesinde egitilir,
SHAP sabit tohumla tum model-hazir veriden cekilen 3.000 kayitlik orneklemde hesaplanir.

Girdi : data/processed/model_ready_dataset.csv
Cikti : outputs/robustness/blok_shap.json
        outputs/robustness/blok_shap_sutun.csv   (sutun duzeyi; Tablo 4.6 karsiligi)
        outputs/robustness/blok_shap_blok.csv    (semantik degisken duzeyi)
Calistirma: python scripts/21_blok_shap.py
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.models.registry import build_models
from perovskite_ml.validation.splits import holdout_split
from perovskite_ml.utils.manifest import write_manifest

N_SAMPLE = 3000


def block_of(col):
    """Sutunu semantik degiskene esle (tez Bolum 3.7'deki ozellik dagilimi)."""
    if col.startswith("A_"):   return "A-site kompozisyon"
    if col.startswith("B_"):   return "B-site kompozisyon"
    if col.startswith("X_"):   return "X-site kompozisyon"
    if col.startswith("arch_"): return "Hucre mimarisi"
    if col.startswith("ETL_"): return "ETL"
    if col.startswith("HTL_"): return "HTL"
    if col.startswith("solv_"): return "Cozucu sistemi"
    if col.startswith("band_gap"): return "Band gap (+eksiklik)"
    if col.startswith("anneal_temp"): return "Tavlama sicakligi (+eksiklik)"
    if col.startswith("anneal_time"): return "Tavlama suresi (+eksiklik)"
    if col == "flexible": return "Esneklik"
    if col == "semitransparent": return "Yari saydamlik"
    return "diger"


def main():
    t0 = time.time()
    import shap
    cfg = load_config(str(ROOT / "config.yaml"))
    seed, target, doi_col = cfg["seed"], cfg["target"], cfg["group_col"]
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)

    M = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "model_ready_dataset.csv", low_memory=False)
    y = M[target].values.astype(float)
    groups = M[doi_col].fillna("unknown_doi").astype(str).values
    X = M.drop(columns=[target, doi_col])

    tr, _ = holdout_split(X, y, groups, test_size=0.2, seed=seed)
    model = build_models(seed)["CatBoost"]
    model.fit(X.iloc[tr], y[tr])

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(N_SAMPLE, len(X)), replace=False)
    Xs = X.iloc[idx]
    sv = shap.TreeExplainer(model).shap_values(Xs)
    mean_abs = np.abs(sv).mean(axis=0)

    col = pd.DataFrame({"ozellik": X.columns, "mean_abs_shap": mean_abs})
    col["blok"] = col["ozellik"].map(block_of)
    col = col.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # BIRINCIL olcut: once ornek bazinda blok-ici SHAP toplami, sonra mutlak deger ortalamasi.
    # SHAP toplanabilir oldugundan bu, blogun ornek bazinda tahmine NET katkisidir. Ayni one-hot
    # degiskenin sutunlari ayni ornekte zit isaretli SHAP tasiyabildiginden, sutun-ortalamalarinin
    # toplami (ikincil olcut) cok-sutunlu bloklari sistematik olarak sisirir.
    blocks = pd.Series([block_of(c) for c in X.columns], index=X.columns)
    sv_df = pd.DataFrame(sv, columns=X.columns)
    grp_abs = {b: float(np.abs(sv_df[blocks.index[blocks == b]].sum(axis=1)).mean())
               for b in blocks.unique()}
    blok = (col.groupby("blok")
              .agg(sutun_ort_toplami=("mean_abs_shap", "sum"),
                   sutun_sayisi=("ozellik", "count"),
                   en_buyuk_sutun=("mean_abs_shap", "max"))
              .reset_index())
    blok["grup_mean_abs_shap"] = blok["blok"].map(grp_abs)
    blok = blok.sort_values("grup_mean_abs_shap", ascending=False).reset_index(drop=True)
    blok["pay_grup"] = (blok["grup_mean_abs_shap"] / blok["grup_mean_abs_shap"].sum()).round(4)
    blok["pay_sutun_toplami"] = (blok["sutun_ort_toplami"] / blok["sutun_ort_toplami"].sum()).round(4)
    blok["sisme_orani"] = (blok["sutun_ort_toplami"] / blok["grup_mean_abs_shap"]).round(3)

    # Deney C gruplari (scripts/04 ile ayni kural: kompozisyon 14 / cihaz 43 / surec 20 sutun),
    # ayni birincil olcutle (ornek bazinda grup toplami, sonra |.| ortalamasi) — tez Bolum 4.4.
    comp = [c for c in X.columns if c.split("_")[0] in ("A", "B", "X")]
    proc = [c for c in X.columns if c.startswith("solv_") or c.startswith("anneal")]
    dev = [c for c in X.columns if c not in comp and c not in proc]
    deney_c = {name: float(np.abs(sv_df[cols].sum(axis=1)).mean())
               for name, cols in (("kompozisyon", comp), ("cihaz", dev), ("surec", proc))}
    tot_c = sum(deney_c.values())
    deney_c_pay = {k: round(v / tot_c, 4) for k, v in deney_c.items()}
    print(f"[stage21] Deney C gruplari ({len(comp)}/{len(dev)}/{len(proc)} sutun) pay: {deney_c_pay}")

    print(f"[stage21] SHAP ornegi: {len(Xs)} kayit, {X.shape[1]} sutun\n")
    print("[Sutun duzeyi — ilk 12 (tez Tablo 4.6 karsiligi)]")
    print(col.head(12).to_string(index=False))
    print("\n[Semantik degisken duzeyi — birincil: ornek bazinda grup toplami |SHAP|]")
    print(blok.to_string(index=False))

    col.to_csv(outdir / "blok_shap_sutun.csv", index=False)
    blok.to_csv(outdir / "blok_shap_blok.csv", index=False)
    out = dict(n_sample=int(len(Xs)), seed=seed,
               sutun_ilk10=col.head(10).to_dict("records"),
               blok=blok.to_dict("records"),
               deney_c_grup_mean_abs_shap={k: round(v, 6) for k, v in deney_c.items()},
               deney_c_pay=deney_c_pay,
               olcut_notu="Birincil olcut: ornek bazinda blok-ici SHAP toplaminin mutlak deger ortalamasi; "
                          "sutun_ort_toplami cok sutunlu bloklari sisirdigi icin yalnizca karsilastirma icindir.")
    with open(outdir / "blok_shap.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage21_blok_shap", cfg,
                   metrics={"top_block_grouped": blok.iloc[0]["blok"]},
                   outputs=["outputs/robustness/blok_shap.json"])
    print(f"\nTamam ({time.time() - t0:.0f} s).")


if __name__ == "__main__":
    main()
