"""Tez sekillerinin repo icinden yeniden uretimi (ML yok, yalnizca gorsellestirme).

Tezdeki 13 sekilden dordu zaten hat betiklerince uretilir (4.8/4.9 SHAP -> scripts/05,
4.10 kazanc egrisi -> scripts/07, 4.11 sacilim -> scripts/03). Bu betik geri kalan
VERI sekillerinin esdegerlerini uretir; Sekil 3.1 (mimari semasi) veri sekli degildir.

  Sekil 4.1  PCE dagilimi                      <- data/processed/model_ready_dataset.csv
  Sekil 4.2  A-site katyon oranlari            <- model_ready
  Sekil 4.3  Band gap dagilimi + PCE iliskisi  <- model_ready (yalnizca gozlemli degerler)
  Sekil 4.4  Cihaz-yigini en sik kategoriler   <- data/processed/cleaned_dataset.csv
  Sekil 4.5  Yayin (DOI) basina kayit sayisi   <- model_ready
  Sekil 4.6  Medyanla doldurulan oranlar       <- model_ready (*_missing bayraklari)
  Sekil 4.7  Model GroupKFold karsilastirmasi  <- outputs/v4/model_comparison_groupkfold.csv
  Sekil 4.12 Isaretli hata dagilimi            <- outputs/v4/actual_vs_predicted.csv
  Sekil 4.13 Conformal kapsama/genislik        <- outputs/conformal/* + holdout tahminleri

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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config

DPI = 150


def _tr(n):
    """Binlik ayirici olarak nokta (Turkce gosterim): 41485 -> '41.485'."""
    return f"{n:,}".replace(",", ".")


def _save(fig, outdir, name, made):
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

    # ---- islenmis veri gerektiren sekiller (4.1-4.6) ----
    mr_path = proc / "model_ready_dataset.csv"
    if mr_path.exists():
        M = pd.read_csv(mr_path, low_memory=False)
        pce = M[tgt].astype(float)

        # Sekil 4.1 — PCE dagilimi
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pce, bins=70, color="#4878a8", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("PCE (%)"); ax.set_ylabel("Kayit sayisi")
        ax.set_title(f"PCE dagilimi (n = {_tr(len(M))})")
        _save(fig, outdir, "sekil_4_01_pce_dagilimi.png", made)

        # Sekil 4.2 — A-site katyon oranlari
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
        for ax, col, name in zip(axes, ["A_FA", "A_MA", "A_Cs"], ["FA", "MA", "Cs"]):
            ax.hist(M[col], bins=40, color="#4878a8", edgecolor="white", linewidth=0.3)
            ax.set_title(f"A-site {name} orani"); ax.set_xlabel("oran")
            ax.set_yscale("log")
        axes[0].set_ylabel("Kayit sayisi (log)")
        _save(fig, outdir, "sekil_4_02_asite_oranlari.png", made)

        # Sekil 4.3 — band gap (yalnizca gozlemli) + band gap'e gore ortalama PCE
        obs = M[M["band_gap_missing"] == 0]
        bg = obs["band_gap"].astype(float)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].hist(bg, bins=60, range=(1.2, 2.4), color="#4878a8",
                     edgecolor="white", linewidth=0.3)
        axes[0].set_xlabel("Band gap (eV)"); axes[0].set_ylabel("Kayit sayisi")
        axes[0].set_title(f"Band gap dagilimi (gozlemli, n = {_tr(len(obs))})")
        bins = np.arange(1.2, 2.4 + 1e-9, 0.05)
        grp = obs.groupby(pd.cut(bg, bins), observed=False)[tgt].agg(["mean", "count"])
        centers = [iv.mid for iv in grp.index]
        mask = grp["count"] >= 30      # seyrek kutulari gizle (gurultulu ortalama)
        axes[1].plot(np.array(centers)[mask], grp["mean"][mask], "o-", color="#a85248")
        axes[1].set_xlabel("Band gap (eV)"); axes[1].set_ylabel("Ortalama PCE (%)")
        axes[1].set_title("Band gap'e gore ortalama PCE (kutu n >= 30)")
        _save(fig, outdir, "sekil_4_03_bandgap.png", made)

        # Sekil 4.5 — DOI basina kayit sayisi (tez basligiyla uyumlu: yalnizca
        # DOI'si bilinen yayinlar; x-ekseni 60 kayitta kesilir, kuyruk maks 127).
        # DOI'siz kayitlarin tek yapay grupta toplandigi kurulum (7.397 grup,
        # en buyuk grup 201) yalnizca bolme mantigini ilgilendirir; tez metninde
        # ayrica raporlanir.
        known = M[M[gcol].notna()]
        sizes = known.groupby(known[gcol].astype(str)).size()
        p95, p99 = np.percentile(sizes, [95, 99])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sizes, bins=np.arange(0.5, sizes.max() + 1.5, 1),
                color="#4878a8", edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_xlim(0, 60)
        ax.axvline(sizes.median(), color="#a85248", linestyle="--", linewidth=1.2,
                   label=f"medyan = {sizes.median():.0f}")
        ax.legend()
        ax.text(0.98, 0.80, f"p95={p95:.0f} · p99={p99:.0f} · maks={sizes.max()}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#b0b8c0"))
        ax.set_xlabel("Yayin basina kayit sayisi"); ax.set_ylabel("Yayin sayisi (log)")
        ax.set_title(f"Yayin (DOI) basina kayit dagilimi "
                     f"({_tr(sizes.shape[0])} yayin; eksen 60'ta kesildi)")
        _save(fig, outdir, "sekil_4_05_doi_dagilimi.png", made)

        # Sekil 4.6 — medyanla doldurulan degiskenlerin oranlari
        flags = [("band_gap_missing", "Band gap"),
                 ("anneal_temp_missing", "Tavlama sicakligi"),
                 ("anneal_time_missing", "Tavlama suresi")]
        vals = [100 * M[c].mean() for c, _ in flags]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        bars = ax.barh([n for _, n in flags], vals, color="#4878a8")
        for b, v in zip(bars, vals):
            ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"%{v:.1f}", va="center")
        ax.set_xlabel("Eksik (medyanla doldurulan) oran %")
        ax.set_title("Doldurma uygulanan degiskenlerde eksiklik orani")
        ax.set_xlim(0, max(vals) * 1.25)
        _save(fig, outdir, "sekil_4_06_eksik_oranlar.png", made)
    else:
        skipped += ["4.1", "4.2", "4.3", "4.5", "4.6"]
        print(f"  [atlandi] model_ready_dataset.csv yok ({mr_path}) -> once scripts/01-02")

    # Sekil 4.4 — cihaz-yigini en sik kategoriler (ham etiketler icin cleaned_dataset)
    cl_path = proc / "cleaned_dataset.csv"
    if cl_path.exists():
        C = pd.read_csv(cl_path, low_memory=False)
        panels = [("Cell_architecture", "Mimari"),
                  ("ETL_stack_sequence", "ETL"),
                  ("HTL_stack_sequence", "HTL")]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, (col, name) in zip(axes, panels):
            vc = C[col].fillna("(bos)").astype(str).value_counts().head(8)[::-1]
            labels = [v if len(v) <= 28 else v[:25] + "..." for v in vc.index]
            ax.barh(labels, vc.values, color="#4878a8")
            ax.set_title(f"{name} — en sik 8 kategori")
            ax.tick_params(axis="y", labelsize=8)
        _save(fig, outdir, "sekil_4_04_cihaz_kategorileri.png", made)
    else:
        skipped.append("4.4")
        print(f"  [atlandi] cleaned_dataset.csv yok ({cl_path})")

    # ---- yalnizca commit'li outputs/ gerektiren sekiller ----

    # Sekil 4.7 — GroupKFold model karsilastirmasi
    cmp_path = outputs / "v4" / "model_comparison_groupkfold.csv"
    if cmp_path.exists():
        cmp = pd.read_csv(cmp_path).sort_values("R2_mean")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(cmp["model"], cmp["R2_mean"], xerr=cmp["R2_std"],
                color="#4878a8", capsize=3)
        ax.set_xlabel("GroupKFold CV R² (ort. ± std)")
        ax.set_title("Modellerin DOI-grup 5-kat CV karsilastirmasi")
        ax.set_xlim(0, 0.5)
        _save(fig, outdir, "sekil_4_07_model_karsilastirma.png", made)
    else:
        skipped.append("4.7")

    # Sekil 4.12 — isaretli hata dagilimi (tahmin - gercek)
    avp_path = outputs / "v4" / "actual_vs_predicted.csv"
    if avp_path.exists():
        avp = pd.read_csv(avp_path)
        err = avp["Model_tahmini"] - avp["Gercek_PCE"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(err, bins=80, color="#4878a8", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="#a85248", linewidth=1)
        ax.set_xlabel("Isaretli hata (tahmin − gercek, PCE puani)")
        ax.set_ylabel("Kayit sayisi")
        over = 100 * (err > 5).mean(); under = 100 * (err < -5).mean()
        ax.set_title(f"Holdout isaretli hata dagilimi "
                     f"(>+5 puan: %{over:.1f}, <−5 puan: %{under:.1f})")
        _save(fig, outdir, "sekil_4_12_hata_dagilimi.png", made)
    else:
        skipped.append("4.12")

    # Sekil 4.13 — conformal kapsama (sol) + ±yari-genislik bandi (sag)
    conf_path = outputs / "conformal" / "conformal_summary.csv"
    hold_path = outputs / "v4" / "best_model_holdout_predictions.csv"
    if conf_path.exists() and hold_path.exists():
        conf = pd.read_csv(conf_path)
        marg = conf[conf["yontem"].str.startswith("Marginal")]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        xs = np.arange(len(marg))
        hedef = marg["hedef_kapsama"].str.replace("%", "").astype(float)
        axes[0].bar(xs - 0.18, hedef, width=0.36, label="Hedef", color="#b0b8c0")
        axes[0].bar(xs + 0.18, 100 * marg["ampirik_kapsama"], width=0.36,
                    label="Gozlenen", color="#4878a8")
        axes[0].set_xticks(xs, [f"%{int(h)}" for h in hedef])
        axes[0].set_ylabel("Kapsama %"); axes[0].set_ylim(0, 100)
        axes[0].set_title("Marginal conformal: hedef vs gozlenen kapsama")
        axes[0].legend()
        if (hedef == 90.0).any():
            hw90 = float(marg.loc[hedef[hedef == 90.0].index[0], "yari_genislik_puan"])
        else:
            hw90 = float(marg["yari_genislik_puan"].iloc[0])
        H = pd.read_csv(hold_path)
        axes[1].scatter(H["y_pred"], H["y_true"], s=4, alpha=0.25, color="#4878a8",
                        edgecolors="none")
        lim = np.array([0, 35])
        axes[1].plot(lim, lim, color="#333333", linewidth=0.8)
        axes[1].plot(lim, lim + hw90, "--", color="#a85248", linewidth=0.9)
        axes[1].plot(lim, lim - hw90, "--", color="#a85248", linewidth=0.9,
                     label=f"±{hw90:.2f} puan (%90 bandi)")
        axes[1].set_xlabel("Tahmin (PCE)"); axes[1].set_ylabel("Gercek (PCE)")
        axes[1].set_xlim(lim); axes[1].set_ylim(lim)
        axes[1].set_title("Holdout: tahmin bandi (marginal %90)")
        axes[1].legend(loc="upper left", fontsize=8)
        _save(fig, outdir, "sekil_4_13_conformal.png", made)
    else:
        skipped.append("4.13")

    print(f"\n[thesis_figures] uretilen: {len(made)} sekil -> {outdir}")
    if skipped:
        print(f"[thesis_figures] atlanan: {skipped}")


if __name__ == "__main__":
    main()
