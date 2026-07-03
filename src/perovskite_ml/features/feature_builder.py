"""Stage 2 — model-ready ozellik matrisi.
Kompozisyon vektorlerine cihaz + surec degiskenlerini ekler:
  - band gap: sayisal + medyan doldurma + eksik bayragi
  - esneklik / yari saydamlik: ikili
  - mimari, ETL, HTL, cozucu: one-hot (top-N + other)
  - tavlama sicaklik/sure: saglam sayisal ayristirma + medyan + eksik bayragi
Cikti yalnizca sayisal/ikili sutunlar + hedef (PCE) + grup (DOI) icerir."""
import re
import numpy as np
import pandas as pd

def _first_number(x):
    """Cok-degerli/araliksal girdiden ('100 | 150', '>100', '100-150 C') temsili sayiyi MAX olarak cikarir.

    Annealing icin max secimi kasitlidir: cok-asamali tavlamada kristallesmeyi
    belirleyen en yuksek/nihai sicaklik adimi en bilgilendiricidir; '100-150 C'
    gibi araliklarda da ust sinir nihai islenme kosulunu temsil eder. 'RT' gibi
    sayisiz degerler NaN dondurur (sonradan medyanla doldurulur).
    """
    if pd.isna(x):
        return np.nan
    nums = re.findall(r"\d+\.?\d*", str(x))
    if not nums:
        return np.nan
    vals = [float(n) for n in nums]
    return max(vals)

def _onehot_topn(df, col, top_n, prefix):
    s = df[col].fillna("unknown").astype(str)
    top = s.value_counts().head(top_n).index
    s = s.where(s.isin(top), prefix + "_other")
    return pd.get_dummies(s, prefix=prefix)

from perovskite_ml.features.descriptors import add_descriptors

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    comp_cols = [c for c in df.columns if c.split("_")[0] in ("A", "B", "X")]
    out = df[comp_cols].copy()

    # band gap
    bg = pd.to_numeric(df[f["band_gap_col"]], errors="coerce")
    out["band_gap_missing"] = bg.isna().astype(int)
    out["band_gap"] = bg.fillna(bg.median())

    # ikili cihaz bilgileri
    for col, name in [(f["flexible_col"], "flexible"), (f["semitransparent_col"], "semitransparent")]:
        out[name] = (df[col].astype(str).str.lower() == "true").astype(int)

    # tavlama sicaklik / sure
    for col, name in [(f["anneal_temp_col"], "anneal_temp"), (f["anneal_time_col"], "anneal_time")]:
        v = df[col].map(_first_number)
        out[f"{name}_missing"] = v.isna().astype(int)
        out[name] = v.fillna(v.median())

    # one-hot kategorikler
    top_n = f["onehot_top_n"]
    for col, prefix in [(f["arch_col"], "arch"), (f["etl_col"], "ETL"),
                        (f["htl_col"], "HTL"), (f["solvent_col"], "solv")]:
        out = pd.concat([out, _onehot_topn(df, col, top_n, prefix)], axis=1)

    # fiziksel tanimlayicilar (Goldschmidt t, oktahedral mu, Bartel tau)
    if cfg.get("features", {}).get("add_descriptors", True):
        out = add_descriptors(out)

    # hedef + grup
    out[cfg["target"]] = df[cfg["target"]].values
    out[cfg["group_col"]] = df[cfg["group_col"]].astype(str).values

    # LightGBM icin guvenli sutun adlari
    out.columns = [re.sub(r"[^0-9A-Za-z_]", "_", str(c)) for c in out.columns]
    return out
