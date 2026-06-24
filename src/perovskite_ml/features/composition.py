"""A/B/X kompozisyon bilgisini sabit sayisal sutunlara cevirir (Algoritma 3.2).
Katsayilar her site icin toplami 1 olacak sekilde normalize edilir."""
import pandas as pd

def parse_site(ions: str, coefs: str):
    """'Cs; FA; MA' + '0.05; 0.79; 0.16' -> [('Cs',0.05),('FA',0.79),('MA',0.16)] (normalize)."""
    il = [s.strip() for s in str(ions).split(";") if s.strip()]
    cl = [s.strip() for s in str(coefs).split(";") if s.strip()]
    if len(il) == 0 or len(il) != len(cl):
        raise ValueError("Iyon ve katsayi sayisi eslesmiyor.")
    cv = [float(x) for x in cl]
    total = sum(cv)
    if total <= 0:
        raise ValueError("Katsayi toplami sifir veya negatif.")
    cv = [x / total for x in cv]
    return list(zip(il, cv))

def vectorize_site(parsed, known_ions, prefix: str) -> dict:
    """Bilinen iyonlar ayri sutuna, digerleri {prefix}_other'a toplanir."""
    res = {f"{prefix}_{ion}": 0.0 for ion in known_ions}
    res[f"{prefix}_other"] = 0.0
    for ion, coef in parsed:
        key = f"{prefix}_{ion}"
        if ion in known_ions:
            res[key] += coef
        else:
            res[f"{prefix}_other"] += coef
    return res

def vectorize_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    comp = cfg["columns"]["composition"]
    known = cfg["known_ions"]
    sites = [("A", comp["a_ions"], comp["a_coef"]),
             ("B", comp["b_ions"], comp["b_coef"]),
             ("X", comp["x_ions"], comp["x_coef"])]
    rows = []
    for _, r in df.iterrows():
        feat = {}
        for site, ions_col, coef_col in sites:
            parsed = parse_site(r[ions_col], r[coef_col])
            feat.update(vectorize_site(parsed, known[site], site))
        rows.append(feat)
    vec = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, vec], axis=1)
