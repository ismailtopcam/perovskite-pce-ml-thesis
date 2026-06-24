"""Satir temizligi: gecersiz PCE, moduller, eksik/uyumsuz kompozisyon kayitlari.
Cikarilan her kayit nedeniyle birlikte loglanir (tekrar uretilebilirlik)."""
import pandas as pd

def clean_rows(df: pd.DataFrame, cfg: dict):
    target = cfg["target"]
    comp = cfg["columns"]["composition"]
    module_flag = cfg["columns"]["module_flag"]
    pmin, pmax = cfg["cleaning"]["pce_min"], cfg["cleaning"]["pce_max"]

    removed = []            # (index, reason)
    df = df.copy()

    # 1) PCE sayisal mi
    df[target] = pd.to_numeric(df[target], errors="coerce")
    mask = df[target].isna()
    for i in df.index[mask]: removed.append((i, "pce_not_numeric"))
    df = df[~mask]

    # 2) PCE araligi
    mask = ~df[target].between(pmin, pmax)
    for i in df.index[mask]: removed.append((i, "pce_out_of_range"))
    df = df[~mask]

    # 3) modul kayitlari
    if module_flag in df.columns:
        mask = df[module_flag].astype(str).str.lower() == "true"
        for i in df.index[mask]: removed.append((i, "module_record"))
        df = df[~mask]

    # 4) eksik / uyumsuz kompozisyon (iyon-katsayi sayisi)
    def comp_ok(row):
        for ions_col, coef_col in [(comp["a_ions"], comp["a_coef"]),
                                   (comp["b_ions"], comp["b_coef"]),
                                   (comp["x_ions"], comp["x_coef"])]:
            ions, coefs = row.get(ions_col), row.get(coef_col)
            if pd.isna(ions) or pd.isna(coefs):
                return False, "missing_ion_or_coef"
            il = [s.strip() for s in str(ions).split(";") if s.strip()]
            cl = [s.strip() for s in str(coefs).split(";") if s.strip()]
            if len(il) == 0 or len(il) != len(cl):
                return False, "ion_coef_count_mismatch"
            try:
                [float(x) for x in cl]
            except ValueError:
                return False, "coef_not_numeric"
        return True, None

    keep = []
    for i, row in df.iterrows():
        ok, reason = comp_ok(row)
        if ok: keep.append(i)
        else: removed.append((i, reason))
    df = df.loc[keep]

    removed_log = pd.DataFrame(removed, columns=["row_index", "reason"])
    return df, removed_log
