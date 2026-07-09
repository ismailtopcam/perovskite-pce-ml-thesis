"""SHAP aciklanabilirlik cekirdegi (Stage 5'in modul karsiligi).

Ornekleme, TreeExplainer hesaplama, ortalama |SHAP| siralamasi ve grafik uretimi
burada tanimlanir; scripts/05_shap_analysis.py yalnizca ince bir calistiricidir.
Agir bagimliliklar (shap, matplotlib) fonksiyon icinde yuklenir ki modulun
import edilmesi test/CI ortamlarinda bu paketleri gerektirmesin.

Determinizm sozlesmesi: ayni seed + ayni veri + ayni model -> ayni ornek indisleri,
ayni SHAP degerleri, ayni CSV. (Refaktor sonrasi cikti esitligi commit'li
outputs/shap_full/shap_top_features.csv ile bit-bit dogrulanmistir.)
"""
import numpy as np
import pandas as pd

SHAP_SAMPLE = 3000   # SHAP degerleri buyuk veride pahali; temsili ornek


def sample_for_shap(X: pd.DataFrame, seed: int, n: int = SHAP_SAMPLE):
    """Seed'e bagli deterministik ornek. Donen: (indisler, X_alt_kume)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return idx, X.iloc[idx]


def compute_shap_values(model, Xs: pd.DataFrame):
    """Agac modeli icin kesin TreeExplainer SHAP degerleri."""
    import shap
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(Xs)


def shap_importance(feature_names, sv) -> pd.DataFrame:
    """Ortalama |SHAP| ile azalan sirali ozellik onem tablosu."""
    mean_abs = np.abs(sv).mean(axis=0)
    return (pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
              .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def save_shap_plots(sv, Xs: pd.DataFrame, imp: pd.DataFrame, outdir, dpi: int = 300):
    """Beeswarm ozet grafigi + top-20 |SHAP| bar grafigi (PNG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.summary_plot(sv, Xs, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(outdir / "shap_summary_plot.png", dpi=dpi); plt.close()

    top20 = imp.head(20)[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(top20["feature"], top20["mean_abs_shap"])
    plt.xlabel("Ortalama |SHAP| (PCE puani)"); plt.tight_layout()
    plt.savefig(outdir / "shap_top20_bar.png", dpi=dpi); plt.close()
