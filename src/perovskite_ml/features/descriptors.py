"""Fiziksel/kristalokimyasal tanimlayicilar: Goldschmidt tolerans faktoru (t),
oktahedral faktor (mu) ve Bartel tau faktoru. A/B/X site oranlarindan, iyon
yariplari uzerinden hesaplanir.

Iyon yariplari (efektif, Angstrom):
- Inorganik iyonlar: Shannon (1976) efektif iyonik yariplari.
- Molekuler katyonlar (MA, FA): Kieslich, Sun & Cheetham (2014) efektif yariplari.
- Halid perovskitlere tolerans faktoru uygulamasi: Travis vd. (2016).
- Tau faktoru: Bartel vd. (2019), n_A = 1 (halid perovskit A-site +1).
"""
import numpy as np

# Angstrom
R_A = {"FA": 2.53, "MA": 2.17, "Cs": 1.88, "Rb": 1.72, "K": 1.64}
R_B = {"Pb": 1.19, "Sn": 1.10, "Ge": 0.73}
R_X = {"I": 2.20, "Br": 1.96, "Cl": 1.81}

def _eff_radius(row, prefix, radii):
    """Bilinen iyonlar uzerinden oran-agirlikli ortalama yaricap.
    Bilinen iyon orani yoksa NaN (ornegin site tamamen *_other ise)."""
    w = 0.0; rw = 0.0
    for ion, r in radii.items():
        col = f"{prefix}_{ion}"
        if col in row:
            f = row[col]
            if f and f > 0:
                w += f; rw += f * r
    return (rw / w) if w > 0 else np.nan

def tolerance_factor(rA, rB, rX):
    return (rA + rX) / (np.sqrt(2.0) * (rB + rX))

def octahedral_factor(rB, rX):
    return rB / rX

def tau_factor(rA, rB, rX, n_A=1.0):
    # Bartel vd. (2019): tau = (rX/rB) - n_A*(n_A - (rA/rB)/ln(rA/rB))
    ratio = rA / rB
    return (rX / rB) - n_A * (n_A - ratio / np.log(ratio))

def add_descriptors(df):
    """A_/B_/X_ oran sutunlarindan t, mu, tau ekler. Hesaplanamayan satirlar
    medyanla doldurulur, bir eksiklik bayragi tutulur."""
    out = df.copy()
    rA = out.apply(lambda r: _eff_radius(r, "A", R_A), axis=1)
    rB = out.apply(lambda r: _eff_radius(r, "B", R_B), axis=1)
    rX = out.apply(lambda r: _eff_radius(r, "X", R_X), axis=1)
    t = tolerance_factor(rA, rB, rX)
    mu = octahedral_factor(rB, rX)
    tau = tau_factor(rA, rB, rX)
    miss = (t.isna() | mu.isna() | tau.isna()).astype(int)
    out["tolerance_factor"] = t.fillna(t.median())
    out["octahedral_factor"] = mu.fillna(mu.median())
    out["tau_factor"] = tau.fillna(tau.median())
    out["descriptor_missing"] = miss
    return out
