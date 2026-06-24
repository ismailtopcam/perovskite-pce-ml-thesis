import pandas as pd
from perovskite_ml.features.descriptors import add_descriptors, tolerance_factor

def test_mapbi3_tolerance_factor_matches_literature():
    df = pd.DataFrame([{"A_MA":1.0,"B_Pb":1.0,"X_I":1.0}])
    r = add_descriptors(df)
    assert 0.90 <= r["tolerance_factor"].iloc[0] <= 0.92   # lit ~0.91

def test_fapbi3_higher_t_than_cspbi3():
    fa = add_descriptors(pd.DataFrame([{"A_FA":1.0,"B_Pb":1.0,"X_I":1.0}]))["tolerance_factor"].iloc[0]
    cs = add_descriptors(pd.DataFrame([{"A_Cs":1.0,"B_Pb":1.0,"X_I":1.0}]))["tolerance_factor"].iloc[0]
    assert fa > cs   # FA daha buyuk A-site -> daha yuksek t

def test_descriptor_columns_added():
    out = add_descriptors(pd.DataFrame([{"A_MA":1.0,"B_Pb":1.0,"X_I":1.0}]))
    for c in ["tolerance_factor","octahedral_factor","tau_factor","descriptor_missing"]:
        assert c in out.columns
