"""Tez sekilleri (tez v119): Sekil 3.1 hat semasi ile Sekil 4.8 / 4.9 SHAP grafiklerinin Turkce surumleri.

Sekil 3.1 : yazilim hattinin veri-akis semasi (veri gerektirmez). Ozgun gorselin kaynak dosyasi
            depoda yoktu; bu betik onun yerine gecer (sayilar Turkce gosterimle, alti model).
Sekil 4.8 : en etkili 20 ozelligin ortalama |SHAP| cubuk grafigi
Sekil 4.9 : SHAP beeswarm ozet grafigi
4.8/4.9 icin SHAP degerleri scripts/05_shap_analysis.py ile ayni model (outputs/v4/best_model.joblib)
ve ayni orneklemle (sample_for_shap, config tohumu) yeniden hesaplanir; ortalama |SHAP| degerleri
commit'li outputs/shap_full/shap_top_features.csv ile karsilastirilir, fark 1e-9'u asarsa betik durur.
Etiketler Turkce, ozellik adlari Tablo 4.6 ile ayni, ondaliklar virgullu, yazi tipi Times New Roman
(tez kilavuzu). Kanvas olculeri tez belgesindeki yerlesim olculeriyle aynidir.
Girdi : data/processed/model_ready_dataset.csv, outputs/v4/best_model.joblib (yalnizca 4.8/4.9)
Cikti : outputs/figures/sekil_3_1_hat_semasi.png, sekil_4_8_shap_ilk20.png, sekil_4_9_shap_beeswarm.png
Calistirma (repo kokunden): python scripts/tez_sekilleri_ek.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perovskite_ml.config import load_config

DPI = 300
LACIVERT = "#1F3A68"
MAVI_K, MAVI_D = "#2F6FB3", "#EEF4FB"
TUR_K, TUR_D = "#D9772B", "#FDF1E5"
MOR_K, MOR_D = "#7B5EA7", "#F3EFF9"
YES_K, YES_D = "#4E8F3A", "#F0F6EC"
H = 16.0  # standart kutu yuksekligi (sema birimi)


def _kutu(ax, x, y, w, h, kenar, dolgu, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                linewidth=lw, edgecolor=kenar, facecolor=dolgu, zorder=2))


def _ok(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.5, color=LACIVERT, zorder=3, shrinkA=0, shrinkB=0))


def _blok(ax, x, y, w, baslik, satirlar, etiket, kenar=MAVI_K, dolgu=MAVI_D, baslik_renk=LACIVERT,
          madde=False, lw=1.4):
    """Baslik + iki satir metin + depo klasoru etiketi; tum kutularda ayni dikey duzen."""
    _kutu(ax, x, y, w, H, kenar, dolgu, lw)
    ax.text(x + w / 2, y + 12.5, baslik, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=baslik_renk, zorder=4)
    for i, s in enumerate(satirlar):
        yy = y + 8.4 - 3.0 * i
        if madde:
            ax.text(x + 3.5, yy, "•  " + s, ha="left", va="center", fontsize=9.2, zorder=4)
        else:
            ax.text(x + w / 2, yy, s, ha="center", va="center", fontsize=9.2, zorder=4)
    ax.text(x + w / 2, y + 2.3, etiket, ha="center", va="center", fontsize=8.2,
            style="italic", color=kenar, zorder=4)


def sekil_3_1(path):
    """Sekil 3.1 — sizinti-guvenli ve tekrar uretilebilir ML hattinin veri-akis semasi."""
    fig = plt.figure(figsize=(5.709, 7.134))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 125); ax.axis("off")

    # 1) Ham veri tabani
    _kutu(ax, 10, 114.5, 80, 9, LACIVERT, "#F3F6FB", lw=1.8)
    ax.text(50, 119.0, "Ham veri tabanı: Perovskite Database (43.398 kayıt × 410 sütun)",
            ha="center", va="center", fontsize=11, fontweight="bold", color=LACIVERT, zorder=4)
    _ok(ax, 50, 114.5, 50, 111.3)
    # 2) Veri temizleme
    _blok(ax, 14, 95.3, 72, "Veri temizleme",
          ["PCE / aralık filtresi · modül kaydı çıkarma · iyon katsayısı doğrulaması",
           "→ 41.485 temiz kayıt"], "data/")
    _ok(ax, 50, 95.3, 50, 92.1)
    # 3) Ozellik uretimi
    _blok(ax, 14, 76.1, 72, "Özellik üretimi",
          ["A/B/X kompozisyon vektörleri + band gap, hücre mimarisi,",
           "ETL/HTL, çözücü, tavlama → 77 özellik"], "features/")
    _ok(ax, 38.5, 76.1, 38.5, 72.9)
    # 4) Sizinti-guvenli dogrulama + yan kutu: dogrulama deneyleri
    _blok(ax, 4, 56.9, 69, "Sızıntı-güvenli doğrulama",
          ["Ölçüm sonrası kolonların merkezî dışlanması (hedef sızıntısı)",
           "DOI bazlı grup-güvenli eğitim / test ayrımı"], "validation/",
          kenar=TUR_K, dolgu=TUR_D, baslik_renk="#8A3D0C", madde=True, lw=1.6)
    _kutu(ax, 77.5, 56.9, 20, H, MOR_K, MOR_D)
    for yy, s in [(70.4, "Doğrulama"), (67.9, "deneyleri")]:
        ax.text(87.5, yy, s, ha="center", va="center", fontsize=10, fontweight="bold",
                color="#4B2E83", zorder=4)
    for i, s in enumerate(["A: bölme stratejisi", "B: ölçüm sonrası", "C: özellik kapsamı"]):
        ax.text(79.2, 64.6 - 2.6 * i, s, ha="left", va="center", fontsize=8.0, color="#3B2470", zorder=4)
    ax.plot([73, 77.5], [64.9, 64.9], linestyle=(0, (2, 2)), linewidth=1.2, color=MOR_K, zorder=3)
    ax.plot([73, 77.5], [64.9, 64.9], linestyle="none", marker="o", markersize=2.6, color=MOR_K, zorder=4)
    _ok(ax, 38.5, 56.9, 38.5, 53.7)
    # 5) Model egitimi ve degerlendirme
    _blok(ax, 8, 37.7, 61, "Model eğitimi ve değerlendirme",
          ["Ridge · RF · HistGBM · XGBoost · LightGBM · CatBoost",
           "5 katlı GroupKFold çapraz doğrulama + DOI-grup holdout"], "models/")
    # 6) dallanma: aciklanabilirlik ve aday uretimi
    ax.plot([38.5], [37.7], marker="o", markersize=3.2, color=LACIVERT, zorder=4)
    _ok(ax, 38.5, 37.7, 25, 33.3)
    _ok(ax, 38.5, 37.7, 74, 33.3)
    _blok(ax, 4, 17.3, 42, "Açıklanabilirlik",
          ["SHAP ile özellik katkılarının", "çözümlenmesi"], "explain/")
    _blok(ax, 52, 17.3, 44, "Aday üretimi ve tarama",
          ["756 aday yapılandırma ·", "ekstrapolasyon analizi"], "candidates/ · simulation/")
    # 7) Tekrar uretilebilirlik altyapisi
    _kutu(ax, 4, 3.5, 92, 11, YES_K, YES_D, lw=1.6)
    ax.text(50, 11.0, "Tekrar üretilebilirlik altyapısı", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color="#2F5E22", zorder=4)
    ax.text(50, 6.8, "config.yaml (veri/özellik parametreleri) · birim testler · koşu-manifesti / loglama",
            ha="center", va="center", fontsize=8.8, zorder=4)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def sekil_4_8_4_9(cfg, bar_path, swarm_path):
    import joblib
    from perovskite_ml.explain.shap_analysis import (
        sample_for_shap, compute_shap_values, shap_importance, save_shap_plots_tez)
    proc = ROOT / cfg["paths"]["processed_dir"]
    outputs = ROOT / cfg["paths"]["outputs_dir"]
    M = pd.read_csv(proc / "model_ready_dataset.csv", low_memory=False)
    X = M.drop(columns=[cfg["target"], cfg["group_col"]])
    bundle = joblib.load(outputs / "v4" / "best_model.joblib")
    idx, Xs = sample_for_shap(X, cfg["seed"])
    sv = compute_shap_values(bundle["model"], Xs)
    imp = shap_importance(X.columns, sv)
    ref = pd.read_csv(outputs / "shap_full" / "shap_top_features.csv")
    fark = float(np.max(np.abs(ref.set_index("feature")["mean_abs_shap"]
                               - imp.set_index("feature")["mean_abs_shap"].reindex(ref["feature"]).values)))
    sira_ayni = list(ref["feature"]) == list(imp["feature"])
    print(f"  model: {bundle['best_name']} | SHAP ornegi: {len(idx)} | maks. fark: {fark:.2e} | sira ayni: {sira_ayni}")
    if fark > 1e-9 or not sira_ayni:
        raise SystemExit("SHAP degerleri commit'li shap_top_features.csv ile tutmuyor; sekiller uretilmedi.")
    save_shap_plots_tez(sv, Xs, imp, bar_path, swarm_path)


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    outdir = ROOT / cfg["paths"]["outputs_dir"] / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    sekil_3_1(outdir / "sekil_3_1_hat_semasi.png")
    print("  [ok] sekil_3_1_hat_semasi.png")
    if "--yalniz-3-1" in sys.argv:
        return
    sekil_4_8_4_9(cfg, outdir / "sekil_4_8_shap_ilk20.png", outdir / "sekil_4_9_shap_beeswarm.png")
    print("  [ok] sekil_4_8_shap_ilk20.png, sekil_4_9_shap_beeswarm.png")


if __name__ == "__main__":
    main()
