#!/usr/bin/env python3
"""
Tez veri-boslugu analizi — D7 + D1 + C1(opsiyon)

Calistirma:
    python3 thesis_data_analysis.py /yol/model_ready.csv

Uretir:
    D7  DOI grup-boyutu dagilimi (4.1'e tek cumle)
    D1  Indirgenemez gurultu tabani -> tavan R2 / taban RMSE (5.1'i ampirik yapar)
    C1  Band-gap dagilim quantile'lari (5.4 opsiyonel somutlastirma)

ML YOK — saf betimsel istatistik. Hicbir sayi uydurulmaz; yoksa kolon hatasi verir.
"""
import sys
import numpy as np
import pandas as pd

# --- Kolon yapilandirmasi: gerekirse kendi adlarinla degistir ---
CONFIG = {
    "pce":      ["JV_default_PCE", "PCE", "pce", "PCE_default"],
    "doi":      ["DOI", "doi", "Ref_DOI_number", "ref_doi", "publication_doi", "group"],
    "band_gap": ["band_gap", "bandgap", "Eg", "band_gap_eV"],
    # kompozisyon oran kolonlari bu on-eklerle aranir:
    "comp_prefixes": ("A_", "B_", "X_"),
}
ROUND_DEC = 2  # recete anahtari icin oranlari/band-gap'i yuvarlama hassasiyeti


def _find(df, candidates, required=True, what=""):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(
            f"[HATA] '{what}' kolonu bulunamadi. Aranan adlar: {candidates}\n"
            f"       Mevcut kolonlardan birini CONFIG'e ekleyin.\n"
            f"       Ilk 40 kolon: {list(df.columns)[:40]}"
        )
    return None


def _comp_cols(df, prefixes):
    cols = [c for c in df.columns if c.startswith(prefixes)]
    # 'missing'/flag kolonlarini disla (band_gap_missing gibi)
    cols = [c for c in cols if not c.lower().endswith(("_missing", "_flag"))]
    return cols


def section(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


def d7_group_distribution(df, doi_col):
    section("D7 — DOI GRUP-BOYUTU DAGILIMI")
    sizes = df.groupby(doi_col).size()
    n_groups = sizes.shape[0]
    q = sizes.quantile([.5, .75, .90, .95, .99]).round(2).to_dict()
    sings = int((sizes == 1).sum())
    print(f"Kayit sayisi              : {len(df):,}")
    print(f"Benzersiz DOI (grup)      : {n_groups:,}")
    print(f"Grup basina ort. kayit    : {len(df)/n_groups:.2f}")
    print(f"Medyan grup boyutu        : {sizes.median():.0f}")
    print(f"Ceyreklikler (50/75/90/95/99): {q}")
    print(f"En buyuk grup             : {int(sizes.max()):,} kayit")
    print(f"Tek-kayitlik grup (singleton): {sings:,}  (%{100*sings/n_groups:.1f})")
    top1 = 100 * sizes.sort_values(ascending=False).head(max(1, n_groups//100)).sum() / len(df)
    print(f"En buyuk %1 grubun kapsadigi kayit payi: %{top1:.1f}")
    print("\n-> Teze (4.1) cumle: 41.485 kayit 7.396 DOI'ye dagilmistir; medyan grup "
          "boyutu ~{:.0f}, ortalama {:.1f}, en buyuk grup {} kayit. (yukaridaki gercek "
          "sayilarla degistir)".format(sizes.median(), len(df)/n_groups, int(sizes.max())))


def d1_noise_floor(df, pce_col, doi_col, bg_col, comp_cols):
    section("D1 — INDIRGENEMEZ GURULTU TABANI -> TAVAN R2 / TABAN RMSE")
    y = df[pce_col].astype(float)
    var_total = y.var(ddof=1)
    print(f"PCE toplam varyansi (sigma^2_tot): {var_total:.3f}  (std {np.sqrt(var_total):.3f})")

    def floor_for(keys, label, require_multi_doi):
        sub = df.dropna(subset=keys + [pce_col]).copy()
        key = sub[keys].round(ROUND_DEC).astype(str).agg("|".join, axis=1)
        sub["_k"] = key
        g = sub.groupby("_k")
        # cok-kayitli gruplar
        sizes = g.size()
        multi = sizes[sizes >= 2].index
        sub_m = sub[sub["_k"].isin(multi)]
        if require_multi_doi and doi_col is not None:
            ndoi = sub_m.groupby("_k")[doi_col].nunique()
            keep = ndoi[ndoi >= 2].index
            sub_m = sub_m[sub_m["_k"].isin(keep)]
        if sub_m.empty:
            print(f"\n[{label}] tekrarli recete bulunamadi (esle yetersiz).")
            return
        gm = sub_m.groupby("_k")[pce_col]
        means = gm.transform("mean")
        resid = sub_m[pce_col].astype(float) - means
        n_rec = sub_m["_k"].nunique()
        n_obs = len(sub_m)
        ss_w = float((resid**2).sum())
        dof = max(1, n_obs - n_rec)
        var_w = ss_w / dof
        rmse_floor = np.sqrt(ss_w / n_obs)
        within_std = gm.std(ddof=1).dropna()
        r2_ceiling = 1 - var_w / var_total
        print(f"\n[{label}]  (multi-DOI={require_multi_doi})")
        print(f"  Tekrar eden recete sayisi   : {n_rec:,}  ({n_obs:,} kayit)")
        print(f"  Recete-ici PCE std (medyan)  : {within_std.median():.3f}")
        print(f"  Recete-ici PCE std (ortalama): {within_std.mean():.3f}")
        print(f"  Pooled gurultu varyansi      : {var_w:.3f}")
        print(f"  ==> Taban RMSE (indirgenemez): {rmse_floor:.3f} PCE puani")
        print(f"  ==> Tavan R2 (homoskedastik varsayim): {r2_ceiling:.3f}")
        print(f"      (Tezdeki sonuc: RMSE 4.02, R2 0.413 — karsilastir.)")

    # Recete tanimi gevsekten sikiya: gevsek = gurultuyu fazla sayar (tavani DUSUK
    # gosterir, alt sinir); siki (tum ozellikler) = tezin 77-ozellikli modeline uyan
    # gercek tavan kestirimi (ama esle seyrek olabilir).
    # 1) yalnizca kompozisyon (alt sinir)
    if comp_cols:
        floor_for(comp_cols, "Kompozisyon-kosullu (tavan ALT siniri)", require_multi_doi=True)
    # 2) kompozisyon + band gap
    if comp_cols and bg_col:
        floor_for(comp_cols + [bg_col], "Kompozisyon+band gap", require_multi_doi=True)
    # 3) TUM ozellikler (en siki kosullama = gercek tavan kestirimi)
    drop = {pce_col, doi_col, "_k"}
    feat_all = [c for c in df.columns
                if c not in drop and not c.lower().endswith(("_missing", "_flag"))
                and pd.api.types.is_numeric_dtype(df[c])]
    if feat_all:
        floor_for(feat_all, f"TUM ozellikler ({len(feat_all)}) = gercek tavan kestirimi",
                  require_multi_doi=True)
    print("\nNOT: 'Tavan R2', gurultunun veri genelinde sabit oldugu varsayimi altinda bir "
          "kestirimdir; recete tanimina ve multi-DOI sartina duyarlidir. Birden cok "
          "tanimi raporlamak en durust yaklasimdir.")


def c1_bandgap(df, bg_col):
    section("C1 (opsiyon) — BAND-GAP DAGILIM QUANTILE'LARI")
    full = df[bg_col].astype(float)
    # model_ready girdisinde band_gap medyanla doldurulmustur; gercek dagilim icin
    # eksiklik bayragiyla gozlemli alt kumeye inilmelidir (bayrak yoksa dropna yeterli).
    flag = f"{bg_col}_missing"
    if flag in df.columns:
        bg = full[df[flag] == 0].dropna()
        n_imp = int((df[flag] == 1).sum())
        print(f"Doldurulmus (imputed) kayit: {n_imp:,} (%{100*n_imp/len(df):.1f}) — asagidaki "
              f"istatistikler yalnizca gozlemli degerler uzerindendir.")
    else:
        bg = full.dropna()
        if len(bg) == len(df):
            print("[UYARI] Eksiklik bayragi yok ve kolonda NaN yok; kolon medyanla "
                  "doldurulmus olabilir — istatistikler medyana yanli cikabilir.")
    q = bg.quantile([.05, .25, .5, .75, .95]).round(3).to_dict()
    print(f"n (band_gap gozlemli)     : {len(bg):,}")
    print(f"Quantile 5/25/50/75/95    : {q}")
    print(f"Ortalama / std            : {bg.mean():.3f} / {bg.std():.3f}")
    frac_tandem = 100 * ((bg >= 1.70) & (bg <= 1.80)).mean()
    frac_wide = 100 * (bg >= 1.70).mean()
    print(f"1.70-1.80 eV (tandem penceresi) payi : %{frac_tandem:.1f}")
    print(f">=1.70 eV (genis-bant) payi          : %{frac_wide:.1f}")
    if flag in df.columns:
        frac_wide_all = 100 * (full >= 1.70).mean()
        print(f">=1.70 eV payi (imputed dahil, referans): %{frac_wide_all:.1f}")
    print("\n-> 5.4 icin: 75. persentil {:.2f} / 95. persentil {:.2f} ise, egitim "
          "kutlesinin ~1.6 eV'de yogunlastigi ve tandem bolgesinin (>=1.70) yalnizca "
          "%{:.1f}'unu kapsadigi yazilabilir.".format(q[0.75], q[0.95], frac_wide))


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Kullanim: python3 thesis_data_analysis.py model_ready.csv")
    path = sys.argv[1]
    df = pd.read_csv(path)
    print(f"Yuklendi: {path}  ->  {df.shape[0]:,} satir x {df.shape[1]} kolon")

    pce = _find(df, CONFIG["pce"], True, "PCE")
    doi = _find(df, CONFIG["doi"], False, "DOI")
    bg  = _find(df, CONFIG["band_gap"], False, "band_gap")
    comp = _comp_cols(df, CONFIG["comp_prefixes"])
    print(f"Kullanilan kolonlar: PCE='{pce}', DOI='{doi}', band_gap='{bg}', "
          f"kompozisyon={len(comp)} kolon ({comp[:6]}{'...' if len(comp)>6 else ''})")

    if doi:
        d7_group_distribution(df, doi)
    else:
        section("D7"); print("DOI kolonu yok -> grup dagilimi atlandi. CONFIG['doi']'ye ekleyin.")

    d1_noise_floor(df, pce, doi, bg, comp)

    if bg:
        c1_bandgap(df, bg)
    else:
        section("C1"); print("band_gap kolonu yok -> atlandi.")


if __name__ == "__main__":
    main()
