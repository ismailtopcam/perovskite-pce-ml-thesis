"""Kontrollu aday kompozisyon uzayi. Aday eksenleri ACIKCA tanimlanir ve
kartezyen carpimla enumere edilir -> aday sayisi tekrar uretilebilir.
Her aday, modelin egitimde gordugu 77-ozellik formatina kodlanir."""
import itertools
import pandas as pd

# --- Aday eksenleri (her biri belgeli) ---
# A-site karisimlari (FA, MA, Cs) -> toplam 1
A_LEVELS = {
    "FA":            {"A_FA": 1.0},
    "MA":            {"A_MA": 1.0},
    "FA0.83_Cs0.17": {"A_FA": 0.83, "A_Cs": 0.17},
    "FA0.85_MA0.15": {"A_FA": 0.85, "A_MA": 0.15},
    "FA0.8_MA0.1_Cs0.1": {"A_FA": 0.8, "A_MA": 0.1, "A_Cs": 0.1},
    "MA0.9_Cs0.1":   {"A_MA": 0.9, "A_Cs": 0.1},
    "Cs0.2_FA0.8":   {"A_Cs": 0.2, "A_FA": 0.8},
}
# B-site
B_LEVELS = {
    "Pb":          {"B_Pb": 1.0},
    "Pb0.5_Sn0.5": {"B_Pb": 0.5, "B_Sn": 0.5},
}
# X-site (I, Br) -> toplam 1
X_LEVELS = {
    "I":            {"X_I": 1.0},
    "I0.83_Br0.17": {"X_I": 0.83, "X_Br": 0.17},
    "I0.6_Br0.4":   {"X_I": 0.6, "X_Br": 0.4},
}
ARCH_LEVELS = ["arch_nip", "arch_pin"]
ETL_LEVELS  = ["ETL_TiO2_c___TiO2_mp", "ETL_SnO2_np", "ETL_PCBM_60"]
HTL_LEVELS  = ["HTL_Spiro_MeOTAD", "HTL_PTAA", "HTL_PEDOT_PSS"]
SOLV_LEVEL  = "solv_DMF__DMSO"   # baskin cozucu sistemi sabitlenir

def enumerate_candidates(feature_names, band_gap, anneal_temp, anneal_time):
    """Tum eksenlerin kartezyen carpimi. Donen: (recipes_df, encoded_df)."""
    recipes, rows = [], []
    feat_set = set(feature_names)
    for a, b, x, arch, etl, htl in itertools.product(
            A_LEVELS, B_LEVELS, X_LEVELS, ARCH_LEVELS, ETL_LEVELS, HTL_LEVELS):
        row = {f: 0.0 for f in feature_names}
        # kompozisyon
        for k, v in A_LEVELS[a].items(): row[k] = v
        for k, v in B_LEVELS[b].items(): row[k] = v
        for k, v in X_LEVELS[x].items(): row[k] = v
        # sayisal (sabit varsayimlar — belgeli)
        if "band_gap" in feat_set: row["band_gap"] = band_gap
        if "anneal_temp" in feat_set: row["anneal_temp"] = anneal_temp
        if "anneal_time" in feat_set: row["anneal_time"] = anneal_time
        # kategorik one-hot (yalnizca modelde var olan sutunlar)
        for col in [arch, etl, htl, SOLV_LEVEL]:
            if col in feat_set: row[col] = 1.0
        rows.append(row)
        recipes.append({"A": a, "B": b, "X": x,
                        "arch": arch.replace("arch_", ""),
                        "ETL": etl.replace("ETL_", ""),
                        "HTL": htl.replace("HTL_", "")})
    enc = pd.DataFrame(rows)[feature_names]   # sutun sirasi modelinkiyle ayni
    return pd.DataFrame(recipes), enc

def axis_sizes():
    return {"A": len(A_LEVELS), "B": len(B_LEVELS), "X": len(X_LEVELS),
            "arch": len(ARCH_LEVELS), "ETL": len(ETL_LEVELS), "HTL": len(HTL_LEVELS),
            "solvent": 1}
