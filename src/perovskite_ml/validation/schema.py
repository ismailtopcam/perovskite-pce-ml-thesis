"""Veri-dogrulama / sema katmani (Breck vd., 2019 operasyonel hali).
Hatti calistirmadan once veri sozlesmesini (data contract) denetler:
- zorunlu kolonlar var mi
- hedef sayisal ve aralikta mi
- kompozisyon oranlari site basina ~1'e toplaniyor mu
- model-ready icinde SIZINTI (olcum sonrasi) kolonu KALMIS mi  <-- kritik
Sert ihlalde SchemaError firlatir; yumusak uyarilari rapor doner."""
from __future__ import annotations

# Olcum-sonrasi / hedef-sizdiran kolon on-ekleri: model-ready'de ASLA bulunmamali
LEAKAGE_PREFIXES = ("JV_", "EQE_", "Stabilised_", "Stabilized_", "Stability_",
                    "Outdoor_", "MPP_", "Voc", "Jsc", "Encapsulation_", "Module_")

class SchemaError(Exception):
    pass

def _is_leaky(col: str) -> bool:
    return any(col.startswith(p) for p in LEAKAGE_PREFIXES)

def validate_clean(df, cfg) -> dict:
    rep = {"n_rows": len(df), "warnings": []}
    if len(df) == 0:
        raise SchemaError("Temiz veri seti bos.")
    tgt = cfg["target"]
    if tgt not in df.columns:
        raise SchemaError(f"Hedef kolon yok: {tgt}")
    import pandas as pd
    y = pd.to_numeric(df[tgt], errors="coerce")
    if y.isna().any():
        raise SchemaError(f"Hedefte sayisal olmayan {int(y.isna().sum())} deger var (temizleme eksik).")
    lo, hi = cfg["cleaning"]["pce_min"], cfg["cleaning"]["pce_max"]
    if (y < lo).any() or (y > hi).any():
        raise SchemaError(f"Hedef [{lo},{hi}] araligini asiyor (aykiri temizligi eksik).")
    if cfg["group_col"] not in df.columns:
        rep["warnings"].append(f"Grup kolonu yok: {cfg['group_col']} (DOI-grup ayrim yapilamaz).")
    return rep

def validate_model_ready(df, cfg) -> dict:
    """model-ready veri sozlesmesi. EN KRITIK kontrol: sizinti kolonu kalmamis olmali."""
    rep = {"n_rows": len(df), "n_features": df.shape[1], "warnings": []}
    tgt, grp = cfg["target"], cfg["group_col"]
    feature_cols = [c for c in df.columns if c not in (tgt, grp)]
    leaked = [c for c in feature_cols if _is_leaky(c)]
    if leaked:
        raise SchemaError(f"SIZINTI: model-ready icinde olcum-sonrasi kolon(lar) var: {leaked[:8]}")
    if tgt not in df.columns:
        raise SchemaError(f"Hedef kolon yok: {tgt}")
    import pandas as pd, numpy as np
    # kompozisyon oranlari site basina ~1'e toplanmali (tolerans 0.02), tum-sifir satir haric
    for site, ions in cfg["known_ions"].items():
        cols = [f"{site}_{i}" for i in ions if f"{site}_{i}" in df.columns]
        cols += [c for c in df.columns if c == f"{site}_other"]
        if cols:
            s = df[cols].sum(axis=1)
            bad = ((s > 0) & ((s < 0.98) | (s > 1.02))).sum()
            if bad:
                rep["warnings"].append(f"{site}-site: {int(bad)} satirda oran toplami 1 degil.")
    # tamamen-NaN ozellik kolonu olmamali
    allnan = [c for c in feature_cols if df[c].isna().all()]
    if allnan:
        rep["warnings"].append(f"Tamamen bos ozellik kolonlari: {allnan[:8]}")
    return rep
