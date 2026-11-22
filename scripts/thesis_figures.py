"""Tez sekillerinin repo icinden yeniden uretimi (ML yok, yalnizca gorsellestirme).

Sekil numaralari tez v126'ya goredir; v125 ve oncesinde 4.9-4.14 birer kucuktu.
Tezdeki sekillerden ucu hat betiklerince uretilir (4.9/4.10 SHAP -> scripts/05,
4.12 sacilim -> scripts/03). Bu betik geri kalan VERI sekillerini uretir; Sekil 3.1
(mimari semasi) veri sekli degildir. Sekil 3.1 ile 4.9/4.10'un tezdeki Turkce surumleri
scripts/tez_sekilleri_ek.py'dedir.
Sekil 4.11 (kazanc egrisi) tez v126'dan itibaren bu betikte cizilir. scripts/07'nin
urettigi outputs/screening/gains_curve.png arsiv ciktisi olarak kalir; tezde kullanilan
surum buradaki sekil_4_11_kazanc_egrisi.png dosyasidir (egri (0, 0) noktasindan baslar).
Uc dogrulama deneyinin (A, B, C) ozet sekli (Sekil 4.8) de buradadir.
Sayilar Turkce gosterimle yazilir (tez v120): ondalik virgul, binlik nokta (_save ->
_tr_eksenler; metin icindeki sayilar _ondalik / _tr ile).

  Sekil 4.1  PCE dagilimi                      <- data/processed/model_ready_dataset.csv
  Sekil 4.2  A-site katyon oranlari            <- model_ready
  Sekil 4.3  Band gap dagilimi + PCE iliskisi  <- model_ready (yalnizca gozlemli degerler)
  Sekil 4.4  Cihaz-yigini en sik kategoriler   <- data/processed/cleaned_dataset.csv
  Sekil 4.5  Yayin (DOI) basina kayit sayisi   <- model_ready
  Sekil 4.6  Medyanla doldurulan oranlar       <- model_ready (*_missing bayraklari)
  Sekil 4.7  Model GroupKFold karsilastirmasi  <- outputs/v4/model_comparison_groupkfold.csv
  Sekil 4.8  Deney A/B/C dogrulama ozeti       <- outputs/v4/validation_experiments.csv
  Sekil 4.11 Kazanc egrisi (PCE >= 18)         <- outputs/v4/best_model_holdout_predictions.csv
  Sekil 4.13 Isaretli hata dagilimi            <- outputs/v4/actual_vs_predicted.csv
  Sekil 4.14 Conformal kapsama/genislik        <- outputs/conformal/* + holdout tahminleri

data/processed dosyalari yoksa (once scripts/01-02 kosulmali) ilgili sekiller atlanir;
commit'li outputs/ dosyalarindan uretilenler her ortamda calisir.
Cikti : outputs/figures/*.png
Calistirma: python scripts/thesis_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch

# Sekil ici yazilar tez kilavuzu geregi Times New Roman (mathtext: STIX serif eslenigi)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config
from perovskite_ml.simulation.screening import gains_curve

DPI = 300  # tez yerlesimindeki ozgun kanvaslarla birebir


def _tr(n):
    """Binlik ayirici olarak nokta (Turkce gosterim): 41485 -> '41.485'."""
    return f"{n:,}".replace(",", ".")


def _ondalik(x, basamak):
    """Ondalik ayirici olarak virgul (Turkce gosterim): 12.031 -> '12,03'."""
    return f"{x:.{basamak}f}".replace(".", ",")


def _tr_sayi_metni(s):
    """Eksen etiketini Turkce gosterime cevirir: '0.2' -> '0,2', '25000' -> '25.000'."""
    isaret = ""
    if s[:1] in ("\u2212", "-"):
        isaret, s = s[0], s[1:]
    tam, _, kesir = s.partition(".")
    if tam.isdigit() and len(tam) >= 4:
        tam = _tr(int(tam))
    return isaret + tam + ("," + kesir if kesir else "")


class _TrFormatter(ticker.ScalarFormatter):
    """ScalarFormatter'in sectigi basamak sayisini korur, yalnizca ayiricilari Turkcelestirir."""

    def __call__(self, x, pos=None):
        return _tr_sayi_metni(super().__call__(x, pos))


def _tr_eksenler(fig):
    """Dogrusal sayisal eksenlere Turkce bicim uygular; log ve elle etiketlenmis eksenlere dokunmaz."""
    for ax in fig.axes:
        for eksen in (ax.xaxis, ax.yaxis):
            if type(eksen.get_major_formatter()) is ticker.ScalarFormatter:
                eksen.set_major_formatter(_TrFormatter())


# Sekil 4.4: veri tabani kategori adlarinin tezdeki yazimi (Tablo 4.6, Sekil 4.9 ile ayni)
_KATEGORI_TR = {"nip": "n-i-p", "pin": "p-i-n", "Back contacted": "Arka kontakl\u0131",
                "Front contacted": "\u00d6n kontakl\u0131", "Unknown": "Bilinmiyor", "none": "Yok"}


def _kategori_adi(t):
    if t in _KATEGORI_TR:
        return _KATEGORI_TR[t]
    return t.replace("TiO2", "TiO\u2082").replace(" | ", "|")


def _save(fig, outdir, name, made):
    _tr_eksenler(fig)
    fig.tight_layout()
    path = outdir / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    made.append(name)
    print(f"  [ok] {name}")


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    tgt, gcol = cfg["target"], cfg["group_col"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    outputs = ROOT / cfg["paths"]["outputs_dir"]
    outdir = outputs / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    made, skipped = [], []
    MAVI, TURUNCU, YESIL = "#2E7A9C", "#D9622B", "#17A589"
    GRI = "#9AA0A6"

    # ---- islenmis veri gerektiren sekiller (4.1-4.6) ----
    mr_path = proc / "model_ready_dataset.csv"
    if mr_path.exists():
        M = pd.read_csv(mr_path, low_memory=False)
        pce = M[tgt].astype(float)

        # Sekil 4.1 — PCE dagilimi (ortalama/medyan cizgili, tez ozgun tasarimi)
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.hist(pce, bins=np.arange(0, 36, 0.5), color=MAVI,
                edgecolor="white", linewidth=0.4)
        ax.axvline(pce.mean(), color=TURUNCU, linestyle="--", linewidth=2.2,
                   label=f"Ortalama = {_ondalik(pce.mean(), 2)}")
        ax.axvline(pce.median(), color="black", linestyle=":", linewidth=2.2,
                   label=f"Medyan = {_ondalik(pce.median(), 2)}")
        ax.set_xlabel("PCE (%)"); ax.set_ylabel("Kay\u0131t say\u0131s\u0131")
        ax.legend()
        _save(fig, outdir, "sekil_4_01_pce_dagilimi.png", made)

        # Sekil 4.2 — A-site katyon oranlari (FA/MA/Cs)
        adlar = [("A_FA", "FA (Formamidinyum)"), ("A_MA", "MA (Metilamonyum)"),
                 ("A_Cs", "Cs (Sezyum)")]
        fig, axes = plt.subplots(1, 3, figsize=(10.31, 2.77))
        for ax, (col, baslik) in zip(axes, adlar):
            ax.hist(M[col], bins=25, color=MAVI, edgecolor="white", linewidth=0.4)
            ax.set_title(baslik); ax.set_xlabel("A-site oran\u0131")
            ax.set_yscale("log")
            ax.grid(alpha=0.3); ax.set_axisbelow(True)
        axes[0].set_ylabel("Kay\u0131t say\u0131s\u0131 (log)")
        _save(fig, outdir, "sekil_4_02_asite_oranlari.png", made)

        # Sekil 4.3 — band gap dagilimi (doldurmasiz) + band gap'e gore ortalama PCE
        obs = M[M["band_gap_missing"] == 0]
        bg = obs["band_gap"].astype(float)
        fig, axes = plt.subplots(1, 2, figsize=(10.366667, 3.07))
        axes[0].hist(bg, bins=np.arange(1.0, 2.5 + 1e-9, 0.05), color=MAVI,
                     edgecolor="white", linewidth=0.4)
        axes[0].set_xlabel("Band gap (eV)")
        axes[0].set_ylabel("Kay\u0131t say\u0131s\u0131")
        axes[0].set_title(f"Band gap da\u011f\u0131l\u0131m\u0131 (doldurmas\u0131z, n = {_tr(len(obs))})")
        axes[0].set_xlim(1.0, 2.5)
        bins = np.arange(1.1, 2.5 + 1e-9, 0.1)
        grp = obs.groupby(pd.cut(bg, bins), observed=False)[tgt].agg(["mean", "count"])
        centers = np.array([iv.mid for iv in grp.index])
        mask = (grp["count"] >= 30).values
        axes[1].plot(centers[mask], grp["mean"][mask], "o-", color=TURUNCU,
                     linewidth=3, markersize=9)
        axes[1].set_xlabel("Band gap (eV)")
        axes[1].set_ylabel("Ortalama PCE (%)")
        axes[1].set_title("Band gap\u2019e g\u00f6re ortalama PCE")
        axes[1].set_xlim(1.0, 2.5)
        _save(fig, outdir, "sekil_4_03_bandgap.png", made)

        # Sekil 4.5 — yayin (DOI) basina kayit sayisi
        known = M[M[gcol].notna()]
        sizes = known.groupby(known[gcol].astype(str)).size()
        p95, p99 = np.percentile(sizes, [95, 99])
        fig, ax = plt.subplots(figsize=(6.9, 3.3))
        ax.hist(sizes, bins=np.arange(0.5, sizes.max() + 1.5, 1),
                color=MAVI, edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xlim(0, 61)
        ax.axvline(sizes.median(), color=TURUNCU, linestyle="--", linewidth=2.5,
                   label=f"medyan = {sizes.median():.0f}")
        ax.legend(loc="upper right")
        kutu = f"{_tr(sizes.shape[0])} yay\u0131n\np95={p95:.0f} \u00b7 p99={p99:.0f} \u00b7 maks={sizes.max()}"
        ax.text(0.98, 0.72, kutu, transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#888888"))
        ax.set_xlabel("Bir yay\u0131ndaki kay\u0131t say\u0131s\u0131")
        ax.set_ylabel("Yay\u0131n say\u0131s\u0131 (log)")
        ax.grid(alpha=0.3); ax.set_axisbelow(True)
        _save(fig, outdir, "sekil_4_05_doi_dagilimi.png", made)

        # Sekil 4.6 — medyanla doldurulan degiskenlerin oranlari
        satirlar = [("band_gap_missing", "Band gap"),
                    ("anneal_temp_missing", "Tavlama s\u0131cakl\u0131\u011f\u0131"),
                    ("anneal_time_missing", "Tavlama s\u00fcresi")]
        vals = [100 * M[c].mean() for c, _ in satirlar]
        fig, ax = plt.subplots(figsize=(6.37, 2.47))
        bars = ax.barh([n for _, n in satirlar], vals, color=MAVI)
        for b, v in zip(bars, vals):
            ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"%{_ondalik(v, 1)}", va="center")
        ax.set_xlabel("Doldurma (imputation) uygulanan kay\u0131t oran\u0131 (%)")
        ax.set_xlim(0, max(vals) * 1.2)
        _save(fig, outdir, "sekil_4_06_eksik_oranlar.png", made)
    else:
        skipped += ["4.1", "4.2", "4.3", "4.5", "4.6"]
        print(f"  [atlandi] model_ready_dataset.csv yok ({mr_path}) -> once scripts/01-02")

    # Sekil 4.4 — cihaz-yigini en sik kategoriler (ilk 5 + '<panel> other');
    # her cubugun ucunda kayit sayisi (kucuk kategorilerin cubugu gorunmedigi icin)
    cl_path = proc / "cleaned_dataset.csv"
    if cl_path.exists():
        C = pd.read_csv(cl_path, low_memory=False)
        panels = [("Cell_architecture", "H\u00fccre mimarisi", "Mimari"),
                  ("ETL_stack_sequence", "ETL", "ETL"),
                  ("HTL_stack_sequence", "HTL", "HTL")]
        X_PAY_4_4 = 1.30  # sayi etiketi cerceveye degmesin
        fig, axes = plt.subplots(1, 3, figsize=(10.88, 3.07))
        for ax, (col, baslik, kisa) in zip(axes, panels):
            vc = C[col].fillna("Unknown").astype(str).value_counts()
            top = vc.head(5)
            other = int(vc.iloc[5:].sum())
            etiket = [_kategori_adi(t) for t in top.index]
            deger = list(top.values)
            if len(vc) > 6 and other > 0:
                etiket.append("Di\u011fer"); deger.append(other)
            elif len(vc) == 6:
                etiket.append(_kategori_adi(vc.index[5]))
                deger.append(int(vc.iloc[5]))
            sirali = sorted(zip(deger, etiket))
            bars = ax.barh([e for _, e in sirali], [v for v, _ in sirali], color=MAVI)
            en_buyuk = max(deger)
            for b, (v, _) in zip(bars, sirali):
                ax.text(v + 0.015 * en_buyuk, b.get_y() + b.get_height() / 2,
                        _tr(int(v)), va="center", ha="left")
            ax.set_xlim(0, en_buyuk * X_PAY_4_4)
            ax.set_title(baslik)
            ax.set_xlabel("Kay\u0131t say\u0131s\u0131")
        _save(fig, outdir, "sekil_4_04_cihaz_kategorileri.png", made)
    else:
        skipped.append("4.4")
        print(f"  [atlandi] cleaned_dataset.csv yok ({cl_path})")

    # ---- yalnizca commit'li outputs/ gerektiren sekiller ----

    # Sekil 4.7 — GroupKFold model karsilastirmasi (deger etiketli)
    cmp_path = outputs / "v4" / "model_comparison_groupkfold.csv"
    if cmp_path.exists():
        cmp = pd.read_csv(cmp_path).sort_values("R2_mean")
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        model_adi = cmp["model"].replace({"RandomForest": "Random Forest"})
        ax.barh(model_adi, cmp["R2_mean"], xerr=cmp["R2_std"],
                color=MAVI, capsize=4)
        for y, (r2, sd) in enumerate(zip(cmp["R2_mean"], cmp["R2_std"])):
            ax.text(r2 + sd + 0.012, y, _ondalik(r2, 3), va="center")
        ax.set_xlabel("GroupKFold CV R\u00b2 (5 katl\u0131; ortalama \u00b1 standart sapma)")
        ax.set_xlim(0, 0.5)
        _save(fig, outdir, "sekil_4_07_model_karsilastirma.png", made)
    else:
        skipped.append("4.7")

    # Sekil 4.11 — kazanc egrisi: model sirasiyla incelenen oran ve yakalanan PCE >= 18 orani
    hold_path = outputs / "v4" / "best_model_holdout_predictions.csv"
    if hold_path.exists():
        G = pd.read_csv(hold_path)
        oran, yakalanan = gains_curve(G["y_true"].values, G["y_pred"].values, 18)
        oran = np.concatenate([[0.0], oran])            # egri (0, 0) noktasindan baslar
        yakalanan = np.concatenate([[0.0], yakalanan])
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(oran * 100, yakalanan * 100, color=MAVI, linewidth=2,
                label="Model y\u00f6nlendirmeli se\u00e7im")
        ax.plot([0, 100], [0, 100], "--", color="gray", label="Rastgele se\u00e7im")
        ax.set_xlabel("\u0130ncelenen kay\u0131t oran\u0131 (%)")
        ax.set_ylabel("Yakalanan y\u00fcksek verimli kay\u0131t (PCE \u2265 18) oran\u0131 (%)")
        ax.legend()
        _save(fig, outdir, "sekil_4_11_kazanc_egrisi.png", made)
    else:
        skipped.append("4.11")

    # Sekil 4.13 — isaretli hata dagilimi (tahmin - gercek)
    avp_path = outputs / "v4" / "actual_vs_predicted.csv"
    if avp_path.exists():
        avp = pd.read_csv(avp_path)
        err = avp["Model_tahmini"] - avp["Gercek_PCE"]
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.hist(err, bins=90, color=YESIL, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=1.6)
        ax.set_xlabel("Tahmin hatas\u0131 (tahmin \u2212 ger\u00e7ek, PCE puan\u0131)")
        ax.set_ylabel("Kay\u0131t say\u0131s\u0131")
        _save(fig, outdir, "sekil_4_13_hata_dagilimi.png", made)
    else:
        skipped.append("4.13")

    # Sekil 4.14 — hedeflenen ve gozlenen kapsama (sol) + 60 orneklik aralik (sag)
    conf_path = outputs / "conformal" / "conformal_summary.csv"
    if conf_path.exists() and hold_path.exists():
        conf = pd.read_csv(conf_path)
        marg = conf[conf["yontem"].str.startswith("Marginal")].copy()
        hedef = marg["hedef_kapsama"].str.replace("%", "").astype(float)
        hw = {int(h): float(w) for h, w in zip(hedef, marg["yari_genislik_puan"])}
        fig, axes = plt.subplots(1, 2, figsize=(10.366667, 3.27))
        xs = np.arange(len(marg))
        etik = [("%" + str(int(h)) + " (\u00b1" + f"{hw[int(h)]:.2f}" + ")").replace(".", ",")
                for h in hedef]
        axes[0].bar(xs - 0.2, hedef, width=0.4, label="Hedeflenen", color=MAVI)
        gozlenen = 100 * marg["ampirik_kapsama"].to_numpy()
        axes[0].bar(xs + 0.2, gozlenen, width=0.4,
                    label="G\u00f6zlenen", color=TURUNCU)
        for x, g in zip(xs, gozlenen):
            axes[0].text(x + 0.2, g + 1.2, "%" + _ondalik(g, 1), ha="center", va="bottom")
        axes[0].set_xticks(xs); axes[0].set_xticklabels(etik)
        axes[0].set_ylabel("Kapsama (%)"); axes[0].set_ylim(0, 105)
        axes[0].set_title("Hedeflenen ve g\u00f6zlenen kapsama")
        axes[0].legend(loc="upper right", ncol=2, frameon=False)
        hw90 = hw.get(90, list(hw.values())[0])
        H = pd.read_csv(hold_path)
        rng = np.random.default_rng(42)
        sec = H.iloc[np.sort(rng.choice(len(H), size=60, replace=False))]
        sec = sec.sort_values("y_pred").reset_index(drop=True)
        icinde = (sec["y_true"] - sec["y_pred"]).abs() <= hw90
        xs2 = np.arange(len(sec))
        band_etiket = ("\u00b1" + f"{hw90:.2f}" + " (%90)").replace(".", ",")
        axes[1].vlines(xs2, sec["y_pred"] - hw90, sec["y_pred"] + hw90,
                       color=MAVI, alpha=0.35, linewidth=4, label=band_etiket)
        axes[1].scatter(xs2[icinde], sec["y_true"][icinde], s=45, color=MAVI,
                        zorder=3, label="aral\u0131kta")
        axes[1].scatter(xs2[~icinde], sec["y_true"][~icinde], s=60, marker="x",
                        color=TURUNCU, linewidths=2.5, zorder=3, label="d\u0131\u015f\u0131nda")
        axes[1].set_xlabel("Rastgele se\u00e7ilen 60 holdout kayd\u0131 (tahmine g\u00f6re s\u0131ral\u0131)")
        axes[1].set_ylabel("PCE (%)")
        axes[1].set_title("Conformal aral\u0131k ve ger\u00e7ek de\u011ferler")
        # gosterge hicbir isaretin ustune gelmesin: ust sinir yukari, tek satir, cercevesiz
        axes[1].set_ylim(top=32)
        axes[1].legend(loc="upper left", ncol=3, frameon=False)
        print(f"  [4.14] 60 kayit: {int(icinde.sum())} aralikta, {int((~icinde).sum())} disinda")
        _save(fig, outdir, "sekil_4_14_conformal.png", made)
    else:
        skipped.append("4.14")

    # Sekil 4.8 — uc dogrulama deneyi (A: bolme stratejisi, B: olcum sonrasi girdi, C: ozellik kapsami)
    den_path = outputs / "v4" / "validation_experiments.csv"
    if den_path.exists():
        D = pd.read_csv(den_path).set_index(["deney", "ayar"])
        # (deney, ayar, eksen etiketi, tezin birincil kurulumu mu)
        paneller = [
            ("Deney A: b\u00f6lme stratejisi", [
                ("A_bolme", "rastgele_KFold", "Rastgele\nKFold", False),
                ("A_bolme", "DOI_grup_KFold", "DOI-grup\nKFold", True)]),
            ("Deney B: \u00f6l\u00e7\u00fcm sonras\u0131 girdi", [
                ("B_olcum_sonrasi", "leakage_safe",
                 "S\u0131z\u0131nt\u0131-g\u00fcvenli\n(77 \u00f6zellik)", True),
                ("B_olcum_sonrasi", "leakage_ON (+Voc,Jsc,FF)", "+ Voc, Jsc, FF", False)]),
            ("Deney C: \u00f6zellik kapsam\u0131", [
                ("C_kapsam", "kompozisyon (14 ozellik)", "Kompozisyon\n(14)", False),
                ("C_kapsam", "+cihaz (57 ozellik)", "+ cihaz\n(57)", False),
                ("C_kapsam", "+surec_tam (77 ozellik)", "+ s\u00fcre\u00e7\n(77)", True)]),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(10.366667, 4.2), sharey=True,
                                 gridspec_kw={"width_ratios": [2, 2, 3]})
        for ax, (baslik, cubuklar) in zip(axes, paneller):
            xs3 = np.arange(len(cubuklar))
            ort = np.array([D.loc[(d, a), "R2_mean"] for d, a, _, _ in cubuklar])
            ss = np.array([D.loc[(d, a), "R2_std"] for d, a, _, _ in cubuklar])
            ax.bar(xs3, ort, yerr=ss, width=0.62, capsize=4,
                   color=[MAVI if birincil else GRI for *_, birincil in cubuklar])
            for x, m, sd in zip(xs3, ort, ss):
                ax.text(x, m + sd + 0.015, _ondalik(m, 3), ha="center", va="bottom")
            ax.set_xticks(xs3)
            ax.set_xticklabels([e for _, _, e, _ in cubuklar])
            ax.set_xlim(-0.6, len(cubuklar) - 0.4)
            ax.set_title(baslik)
        axes[0].set_ylim(0, 1.08)
        axes[0].set_ylabel("\u00c7apraz do\u011frulama R\u00b2\n(5 katl\u0131; ortalama \u00b1 ss)")
        axes[2].legend(handles=[Patch(color=MAVI, label="Tezin birincil kurulumu"),
                                Patch(color=GRI, label="Kar\u015f\u0131la\u015ft\u0131rma kurulumu")],
                       loc="upper left", frameon=False)
        _save(fig, outdir, "sekil_4_08_deney_abc_dogrulama.png", made)
    else:
        skipped.append("4.8")

    print(f"\n[thesis_figures] uretilen: {len(made)} sekil -> {outdir}")
    if skipped:
        print(f"[thesis_figures] atlanan: {skipped}")


if __name__ == "__main__":
    main()
