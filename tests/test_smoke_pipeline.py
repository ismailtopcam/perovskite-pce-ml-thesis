"""Uctan uca DUMAN testi: temizleme -> vektorlestirme -> ozellik uretimi ->
grup-guvenli bolme -> egitim/tahmin -> sema dogrulama -> kosu-manifesti.

Sentetik ~30 kayitla, paket modullerinin BIRLIKTE calistigini kanitlar; hafif
CI kumesiyle (yalnizca scikit-learn) kosar. 41k'lik gercek kosumun yerini
tutmaz — README'deki 'CI uyumluluk / tam uretim ortami' ayrimina tabidir."""
import json
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from perovskite_ml.data.clean_data import clean_rows
from perovskite_ml.features.composition import vectorize_dataframe
from perovskite_ml.features.feature_builder import build_features
from perovskite_ml.utils.manifest import write_manifest
from perovskite_ml.validation.schema import validate_clean, validate_model_ready

CFG = {
    "seed": 42,
    "target": "JV_default_PCE",
    "group_col": "Ref_DOI_number",
    "cleaning": {"pce_min": 0.0, "pce_max": 35.0},
    "known_ions": {"A": ["FA", "MA", "Cs", "Rb", "K"],
                   "B": ["Pb", "Sn", "Ge"],
                   "X": ["I", "Br", "Cl"]},
    "columns": {
        "module_flag": "Module",
        "composition": {
            "a_ions": "Perovskite_composition_a_ions",
            "a_coef": "Perovskite_composition_a_ions_coefficients",
            "b_ions": "Perovskite_composition_b_ions",
            "b_coef": "Perovskite_composition_b_ions_coefficients",
            "x_ions": "Perovskite_composition_c_ions",
            "x_coef": "Perovskite_composition_c_ions_coefficients",
        },
    },
    "features": {
        "onehot_top_n": 3,
        "add_descriptors": False,
        "band_gap_col": "Perovskite_band_gap",
        "solvent_col": "Perovskite_deposition_solvents",
        "anneal_temp_col": "Perovskite_deposition_thermal_annealing_temperature",
        "anneal_time_col": "Perovskite_deposition_thermal_annealing_time",
        "arch_col": "Cell_architecture",
        "etl_col": "ETL_stack_sequence",
        "htl_col": "HTL_stack_sequence",
        "flexible_col": "Cell_flexible",
        "semitransparent_col": "Cell_semitransparent",
    },
}


def make_raw(n_per_doi=8, n_doi=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_doi):
        for _ in range(n_per_doi):
            fa = rng.choice([1.0, 0.8, 0.85])
            rows.append({
                "Ref_DOI_number": f"10.1000/smoke.{d}",
                "Perovskite_composition_a_ions": "FA; MA" if fa < 1 else "FA",
                "Perovskite_composition_a_ions_coefficients":
                    f"{fa}; {round(1 - fa, 2)}" if fa < 1 else "1",
                "Perovskite_composition_b_ions": "Pb",
                "Perovskite_composition_b_ions_coefficients": "1",
                "Perovskite_composition_c_ions": "I; Br",
                "Perovskite_composition_c_ions_coefficients": "0.83; 0.17",
                "Perovskite_band_gap": rng.choice([1.55, 1.6, np.nan]),
                "Cell_architecture": rng.choice(["nip", "pin"]),
                "Cell_flexible": "false",
                "Cell_semitransparent": "false",
                "ETL_stack_sequence": rng.choice(["TiO2-c", "SnO2-np"]),
                "HTL_stack_sequence": rng.choice(["Spiro-MeOTAD", "PTAA", "none"]),
                "Perovskite_deposition_solvents": rng.choice(["DMF; DMSO", "DMF"]),
                "Perovskite_deposition_thermal_annealing_temperature": "100",
                "Perovskite_deposition_thermal_annealing_time": "20",
                "Module": "false",
                "JV_default_Voc": 1.1,      # sizinti adayi -> ozellige GECMEMELI
                "JV_default_PCE": float(np.clip(rng.normal(14 + fa, 2), 1, 30)),
            })
    # temizlemenin calistigini gosterecek iki bozuk kayit
    rows.append({**rows[0], "JV_default_PCE": "not-a-number"})
    rows.append({**rows[1], "JV_default_PCE": 99.0})
    return pd.DataFrame(rows)


def test_smoke_end_to_end():
    raw = make_raw()

    # 1) temizleme + sema
    clean, removed = clean_rows(raw, CFG)
    assert len(removed) == 2 and len(clean) == 32
    validate_clean(clean, CFG)

    # 2) kompozisyon vektorlestirme + ozellik uretimi + sizinti semasi
    vec = vectorize_dataframe(clean, CFG)
    M = build_features(vec, CFG)
    rep = validate_model_ready(M, CFG)
    assert rep["n_rows"] == 32
    assert not any(c.startswith("JV_") for c in M.columns if c != CFG["target"])

    # 3) DOI-grup-guvenli bolme + egitim + tahmin
    y = M[CFG["target"]].values
    groups = M[CFG["group_col"]].astype(str).values
    X = M.drop(columns=[CFG["target"], CFG["group_col"]])
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                    random_state=CFG["seed"]).split(X, y, groups))
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    model = HistGradientBoostingRegressor(max_iter=25, random_state=CFG["seed"])
    model.fit(X.iloc[tr], y[tr])
    pred = model.predict(X.iloc[te])
    assert np.isfinite(pred).all() and len(pred) == len(te)

    # 4) kosu-manifesti: seed + paket surumleri + metrik kaydi
    with tempfile.TemporaryDirectory() as tmp:
        path = write_manifest(tmp, "smoke", CFG,
                              metrics={"n_test": int(len(te))},
                              outputs=["(gecici duman kosusu)"])
        man = json.load(open(path, encoding="utf-8"))
    assert man["seed"] == 42 and man["metrics"]["n_test"] == len(te)
    assert man["versions"]["python"] and "scikit-learn" in man["versions"]
