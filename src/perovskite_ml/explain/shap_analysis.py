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
    plt.rcParams["font.family"] = "Times New Roman"  # tez kilavuzu: sekil yazilari TNR
    plt.rcParams["mathtext.fontset"] = "stix"
    import shap

    plt.figure()
    shap.summary_plot(sv, Xs, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(outdir / "shap_summary_plot.png", dpi=dpi); plt.close()

    top20 = imp.head(20)[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(top20["feature"], top20["mean_abs_shap"])
    plt.xlabel("Ortalama |SHAP| (PCE puani)"); plt.tight_layout()
    plt.savefig(outdir / "shap_top20_bar.png", dpi=dpi); plt.close()


# --- Tez Sekil 4.8 / 4.9 (tez v119): Turkce etiketler, Tablo 4.6 adlari, ondalik virgul ---
# Ilk 15 ad Tablo 4.6 ile birebir; 16-20. siradakiler ayni kalipla adlandirildi.
TEZ_OZELLIK_ADLARI = {
    "A_FA": "A-site FA oranı", "A_MA": "A-site MA oranı", "A_other": "A-site diğer katyon oranı",
    "B_Pb": "B-site Pb oranı", "X_I": "X-site I oranı",
    "solv_DMF": "DMF çözücü", "solv_DMF__DMSO": "DMF;DMSO çözücü", "solv_GBL": "GBL çözücü",
    "solv_DMF____IPA": "DMF >> IPA çözücü (iki adımlı)",
    "solv_DMF__DMSO____IPA": "DMF;DMSO >> IPA çözücü (iki adımlı)",
    "HTL_none": "HTL: yok", "HTL_PEDOT_PSS": "HTL: PEDOT:PSS", "HTL_Spiro_MeOTAD": "HTL: Spiro-MeOTAD",
    "HTL_PTAA": "HTL: PTAA", "ETL_TiO2_c___TiO2_mp": "ETL: TiO$_2$-c|TiO$_2$-mp", "ETL_PCBM_60": "ETL: PCBM-60",
    "anneal_time": "Tavlama süresi", "anneal_temp": "Tavlama sıcaklığı", "band_gap": "Band gap",
    "semitransparent": "Yarı saydamlık",
}


def tez_ozellik_adi(kolon: str) -> str:
    """Kolon adini tezdeki okunur adina cevirir; listede yoksa kolon adini dondurur."""
    return TEZ_OZELLIK_ADLARI.get(kolon, kolon)


def _virgul(v, _pos=None):
    """Eksen etiketi: ondalik virgul ve Unicode eksi (Turkce gosterim)."""
    s = f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
    return s.replace(".", ",").replace("-", "\u2212")


def save_shap_plots_tez(sv, Xs: pd.DataFrame, imp: pd.DataFrame, bar_path, swarm_path,
                        bar_size=(8.0, 6.316), swarm_size=(8.0, 8.0), dpi: int = 300):
    """Tez Sekil 4.8 (ilk 20 ortalama |SHAP|) ve Sekil 4.9 (beeswarm).

    Etiketler Turkce, ozellik adlari Tablo 4.6 ile ayni, ondaliklar virgullu, yazi tipi
    Times New Roman. Kanvas oranlari tez belgesindeki yerlesim olculeriyle ayni secilir
    (4.8: 1,267; 4.9: 1,0), boylece gorsel belgede bozulmadan gorunur.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    import shap

    top20 = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=bar_size)
    ax.barh([tez_ozellik_adi(f) for f in top20["feature"]], top20["mean_abs_shap"])
    ax.set_xlabel("Ortalama |SHAP| (PCE puanı)")
    ax.xaxis.set_major_formatter(FuncFormatter(_virgul))
    fig.tight_layout(); fig.savefig(bar_path, dpi=dpi); plt.close(fig)

    plt.figure()
    shap.summary_plot(sv, Xs, feature_names=[tez_ozellik_adi(c) for c in Xs.columns], show=False,
                      max_display=20, plot_size=swarm_size, color_bar_label="Özellik değeri")
    fig = plt.gcf()
    ax, cax = fig.axes[0], fig.axes[-1]
    ax.set_xlabel("SHAP değeri (model çıktısına etkisi, PCE puanı)", fontsize=13)
    ax.xaxis.set_major_formatter(FuncFormatter(_virgul))
    cax.set_yticks([0, 1]); cax.set_yticklabels(["Düşük", "Yüksek"])
    fig.tight_layout(); fig.savefig(swarm_path, dpi=dpi); plt.close(fig)
